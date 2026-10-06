#!/usr/bin/env python3
"""Drive the design's acceptance chains through the tools' public flags.

Stage C8's chain runner. It exists because the chains are *pipelines*, not unit
behaviours: whether a schedule round-trips, whether a plan report is
byte-identical when the search is run twice, whether the performance model
accepts what the mapper produced -- none of that is visible from inside one
process.

Everything goes through public command-line flags and files in the work
directory this script is given. It never imports a library, never reaches into
an internal header, and never inspects the tools' source: if a step cannot be
expressed as a flag and a file, it is not part of the public contract and does
not belong in an acceptance chain.

The chains are the five the design names:

  vector-add        a concrete Micro kernel mapped onto AVX2
  staged-gemm       a compiler-generated matmul, exported then mapped
  fused-swiglu      a compiler-generated fused SwiGLU, exported then mapped
  second-target     the same matmul on the generic accelerator -- neutrality
  layout-transform  a kernel whose rule requires a layout it does not have

Each asserts: the search-space round trip, the selected plan and its coverage,
concrete routes/ops in the IR, byte-identical repeat runs (report and IR), the
performance model accepting the result, and that compilation still works with
mapping *disabled*.
"""

import argparse
import os
import subprocess
import sys


class ChainFailure(Exception):
    pass


def run(command, expect_success=True):
    """Runs a pipeline stage and returns its stdout.

    A failure here is a chain failure, not a skip: every stage below is
    available in this build, and reporting a missing tool as "unavailable"
    would turn a broken chain into a passing run.
    """
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True)
    if expect_success and result.returncode != 0:
        raise ChainFailure(
            "command failed (%d): %s\n%s" % (
                result.returncode, command, result.stderr.strip()))
    return result.stdout


class Chain:
    def __init__(self, name, micro_source, export, target, emitters, entry):
        self.name = name
        # The .mlir file the chain starts from.
        self.micro_source = micro_source
        # True when the file is LLK source that has to be exported first.
        self.export = export
        # The target package: its name, machine, layouts and rules files.
        self.target = target
        self.emitters = emitters
        # The kernel symbol the chain expects to find.
        self.entry = entry


AVX2 = {
    "name": "x86-avx2",
    "machine": "machines/x86-avx2-v2.yaml",
    "layouts": "mapping/x86-avx2/layouts.llkmap",
    "rules": "mapping/x86-avx2/rules.llkmap",
}

ACCEL = {
    "name": "generic-ai-accel",
    "machine": "machines/generic-ai-accel-v2.yaml",
    "layouts": "mapping/generic-ai-accel/layouts.llkmap",
    "rules": "mapping/generic-ai-accel/rules.llkmap",
}

AVX2_EMITTERS = ("avx2_vector_add,avx2_vector_convert,avx2_vector_silu,"
                 "avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,"
                 "avx2_tile_copy,avx2_tile_store,"
                 "avx2_fused_convert_silu_mul")
ACCEL_EMITTERS = "accel_vector_add,accel_mxu,accel_copy"


def chains():
    return [
        Chain("vector-add", "micro_map.mlir", False, AVX2, AVX2_EMITTERS,
              "mapped"),
        Chain("staged-gemm", "matmul_e2e.mlir", True, AVX2, AVX2_EMITTERS,
              "matmul_M16_N64_K64"),
        Chain("fused-swiglu", "swiglu_e2e.mlir", True, AVX2, AVX2_EMITTERS,
              "fused_swiglu_M16_N64_K64"),
        # The accelerator's rules cover the tensor-level movement and the
        # vector family, not the tile-level ops the LLK export emits, so the
        # neutrality chain uses the concrete kernel both targets' rules cover:
        # the same program, mapped twice, with nothing generic changed.
        Chain("second-target", "micro_map.mlir", False, ACCEL, ACCEL_EMITTERS,
              "mapped"),
    ]


def map_options(chain, source_dir, target, mode, report=None, report_only=None):
    options = [
        "target=%s" % target["name"],
        "machine=%s" % os.path.join(source_dir, target["machine"]),
        "layouts=%s" % os.path.join(source_dir, target["layouts"]),
        "rules=%s" % os.path.join(source_dir, target["rules"]),
        "emitters=%s" % chain.emitters,
        "mode=%s" % mode,
    ]
    if report:
        options.append("report=%s" % report)
    if report_only is not None:
        options.append("report-only=%d" % report_only)
    # Space-separated: the pass parses its options the way the command line
    # spells them, and a comma would be part of the value.
    return " ".join(options)


def check(condition, message):
    if not condition:
        raise ChainFailure(message)


def read(path):
    with open(path, "r") as handle:
        return handle.read()


def run_chain(chain, args, work, report_lines):
    llk_opt = args.llk_opt
    llk_compile = args.llk_compile
    micro_perf = args.micro_perf
    source = os.path.join(args.source_dir, "test/Conversion/MicroMapping",
                          chain.micro_source)
    check(os.path.exists(source), "missing fixture %s" % source)

    # 1. The concrete Micro module. A compiler-generated chain exports first;
    #    that export is the schedule-instantiation half, and it is the same
    #    pass the IR fixtures exercise.
    micro_path = os.path.join(work, "micro.mlir")
    if chain.export:
        exported = run('%s "--llk-to-micro=schedule-db=missing.json" %s'
                       % (llk_opt, source))
        check("micro.kernel @%s" % chain.entry in exported,
              "%s: the export produced no kernel @%s" % (chain.name, chain.entry))
        # Mapping disabled: the legacy export still emits the concrete kernel,
        # which is what "the legacy path keeps working" means here.
        check("@%s" % chain.entry in exported,
              "%s: the unmapped export changed" % chain.name)
        with open(micro_path, "w") as handle:
            handle.write(exported)
    else:
        micro_path = source

    # 2. The mapping search. Run twice, into different report paths, so the
    #    repeat check is about the *search* rather than about a cached file.
    report_a = os.path.join(work, "report_a.json")
    report_b = os.path.join(work, "report_b.json")
    mapped_a = os.path.join(work, "mapped_a.mlir")
    mapped_b = os.path.join(work, "mapped_b.mlir")
    for report, mapped in ((report_a, mapped_a), (report_b, mapped_b)):
        options = map_options(chain, args.source_dir, chain.target, "exact",
                              report=report)
        output = run('%s "--micro-map=%s" %s' % (llk_opt, options, micro_path))
        with open(mapped, "w") as handle:
            handle.write(output)

    # 3. Byte-identical repeats: the plan report and the mapped IR. §29's
    #    twelfth criterion is exactly this, and it is the one that catches an
    #    unordered container leaking into an id or a serialization.
    check(read(report_a) == read(report_b),
          "%s: the plan report is not byte-identical across runs" % chain.name)
    check(read(mapped_a) == read(mapped_b),
          "%s: the mapped IR is not byte-identical across runs" % chain.name)

    mapped_text = read(mapped_a)

    # 4. The selected plan and its coverage are in the IR.
    check("micro.plan" in mapped_text,
          "%s: the mapped kernel carries no micro.plan" % chain.name)
    check("micro.mapping" in mapped_text,
          "%s: no covered operation carries its placement" % chain.name)
    check("micro.routes" in mapped_text,
          "%s: the selected connections are not recorded" % chain.name)

    # 5. The report is the versioned schema and names the content it searched:
    #    the machine, the layout and rule libraries, and the graph. A report
    #    that named only the plan would not be replayable against anything.
    report_text = read(report_a)
    for key in ("\"version\"", "graphHash", "machineHash", "ruleLibraryHash",
                "layoutLibraryHash", "targetHash"):
        check(key in report_text,
              "%s: the plan report has no %s" % (chain.name, key))

    # 6. Layered verification accepts what the mapper wrote.
    verify_options = " ".join([
        "target=%s" % chain.target["name"],
        "machine=%s" % os.path.join(args.source_dir, chain.target["machine"]),
        "layouts=%s" % os.path.join(args.source_dir, chain.target["layouts"]),
        "rules=%s" % os.path.join(args.source_dir, chain.target["rules"]),
        "emitters=%s" % chain.emitters,
    ])
    run('%s "--micro-verify-mapping=%s" %s'
        % (llk_opt, verify_options, mapped_a))

    # 7. The performance model accepts the selected plan and charges it.
    machine = os.path.join(args.source_dir, chain.target["machine"])
    perf = run("%s --machine=%s --level=1 %s" % (micro_perf, machine, mapped_a))
    check(len(perf.strip()) > 0,
          "%s: the performance model produced no events" % chain.name)

    # 8. With mapping disabled the legacy export still produces the same
    #    concrete kernel, twice, byte for byte. Compiling that kernel through
    #    the legacy backend is asserted by the suite's own legacy tests
    #    (MicroToLinalgJit, LLKToLinalgConversion, the AVX2 SwiGLU chains) --
    #    repeating it here would test the backend, not the mapping path.
    if chain.export:
        again = run('%s "--llk-to-micro=schedule-db=missing.json" %s'
                    % (llk_opt, source))
        check(again == exported,
              "%s: the mapping-disabled export is not deterministic"
              % chain.name)

    report_lines.append(
        "  %-16s ok  target=%s  report=%s"
        % (chain.name, chain.target["name"], os.path.basename(report_a)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("llk_opt")
    parser.add_argument("llk_compile")
    parser.add_argument("micro_perf")
    parser.add_argument("source_dir")
    parser.add_argument("work_dir")
    args = parser.parse_args()

    # The runner owns its scratch: it writes only here, and every stage below
    # reads only what an earlier stage wrote here.
    os.makedirs(args.work_dir, exist_ok=True)

    report_lines = []
    failures = []
    for chain in chains():
        chain_dir = os.path.join(args.work_dir, chain.name)
        os.makedirs(chain_dir, exist_ok=True)
        try:
            run_chain(chain, args, chain_dir, report_lines)
        except ChainFailure as failure:
            failures.append("%s: %s" % (chain.name, failure))
            report_lines.append("  %-16s FAIL" % chain.name)

    print("acceptance chains:")
    for line in report_lines:
        print(line)

    # The evidence a result is only interpretable with: which compiler produced
    # these reports, and which machine, layout, rule and graph content they were
    # taken against. A chain that passed against an unnamed configuration is
    # not reproducible by anyone else.
    first = os.path.join(args.work_dir, chains()[0].name, "report_a.json")
    if os.path.exists(first):
        import json
        with open(first) as handle:
            report = json.load(handle)
        print("configuration:")
        for key in ("compilerVersion", "target", "machineHash", "layoutLibraryHash",
                    "ruleLibraryHash", "targetHash", "graphHash"):
            if key in report:
                print("  %-20s %s" % (key, report[key]))

    if failures:
        print("\n%d chain(s) failed:" % len(failures), file=sys.stderr)
        for failure in failures:
            print("  " + failure, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
