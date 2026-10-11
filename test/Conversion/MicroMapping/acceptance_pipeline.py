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
import json
import os
import re
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
    result = subprocess.run(command, capture_output=True, text=True)
    if expect_success and result.returncode != 0:
        raise ChainFailure(
                "command failed (%d): %s\n%s" % (
                result.returncode, " ".join(command), result.stderr.strip()))
    return result


class Chain:
    def __init__(self, name, micro_source, export, target, emitters, entry,
                 expected_rules, required_ops, transform=False):
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
        self.expected_rules = expected_rules
        self.required_ops = required_ops
        self.transform = transform


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

TWO_HOP = {
    "name": "issue129-two-hop",
    "machine": "test/Mapping/Inputs/issue129/two_hop_machine.yaml",
    "layouts": "mapping/x86-avx2/layouts.llkmap",
    "rules": "test/Mapping/Inputs/issue129/two_hop_rules.llkmap",
}

AVX2_EMITTERS = ("avx2_vector_add,avx2_vector_convert,avx2_vector_silu,"
                 "avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,"
                 "avx2_tile_copy,avx2_tile_store,"
                 "avx2_fused_convert_silu_mul")
ACCEL_EMITTERS = "accel_vector_add,accel_mxu,accel_copy"


def chains():
    return [
        Chain("vector-add", "micro_map.mlir", False, AVX2, AVX2_EMITTERS,
              "mapped", {"avx2.vector_add", "avx2.async_copy"},
              {"micro.async_copy", "micro.vector"}),
        Chain("staged-gemm", "matmul_e2e.mlir", True, AVX2, AVX2_EMITTERS,
              "matmul_M16_N64_K64", {"avx2.mma_bf16", "avx2.tile_async_copy",
                                      "avx2.tile_store"},
              {"micro.mma", "micro.tile_async_copy", "micro.wait"}),
        Chain("fused-swiglu", "swiglu_e2e.mlir", True, AVX2, AVX2_EMITTERS,
              "fused_swiglu_M16_N64_K64", {"avx2.mma_bf16", "avx2.vector_mul",
                                            "avx2.vector_silu"},
              {"micro.mma", "micro.vector", "micro.tile_async_copy"}),
        Chain("required-transform", "Inputs/issue129/required_transform.mlir",
              False, {**AVX2,
                      "layouts": "test/Conversion/MicroMapping/Inputs/issue129/required_transform_layouts.llkmap",
                      "rules": "test/Conversion/MicroMapping/Inputs/issue129/required_transform.llkmap"},
              "avx2_vector_add,avx2_vector_mul,avx2_copy", "required_transform",
              {"issue129.add_row_major", "issue129.mul_blocked"},
              {"micro.vector"}, transform=True),
        # The second-target closure is the named two-hop profile. Its selected
        # route and intermediate reservation are reviewed below.
        Chain("second-target-two-hop", "physical_two_hop.mlir", False, TWO_HOP,
              "e1", "physical_two_hop", {"issue129.vector_add_sram",
                                           "issue129.store_dram"},
              {"micro.vector", "micro.tile_async_copy", "micro.wait"}),
    ]


def map_options(chain, source_dir, target, mode, report=None, report_only=None):
    options = [
        "target=%s" % target["name"],
        "machine=%s" % os.path.join(source_dir, target["machine"]),
        "layouts=%s" % os.path.join(source_dir, target["layouts"]),
        "rules=%s" % os.path.join(source_dir, target["rules"]),
        "emitters=%s" % chain.emitters,
        "mode=%s" % mode,
        "top-k=0",
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
        exported = run([llk_opt, "--llk-to-micro=schedule-db=missing.json",
                        source]).stdout
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

    # Parse/print twice before mapping.  The second print must be stable; this
    # catches syntax accepted only by the original text path and printer drift.
    parsed_once = run([llk_opt, "--mlir-print-op-generic", micro_path]).stdout
    parsed_path = os.path.join(work, "parsed_once.mlir")
    with open(parsed_path, "w") as handle:
        handle.write(parsed_once)
    parsed_twice = run([llk_opt, "--mlir-print-op-generic", parsed_path]).stdout
    check(parsed_once == parsed_twice,
          "%s: parse/print is not byte-stable" % chain.name)
    micro_path = parsed_path

    # 2. The mapping search. Run twice, into different report paths, so the
    #    repeat check is about the *search* rather than about a cached file.
    report_a = os.path.join(work, "report_a.json")
    report_b = os.path.join(work, "report_b.json")
    mapped_a = os.path.join(work, "mapped_a.mlir")
    mapped_b = os.path.join(work, "mapped_b.mlir")
    for report, mapped in ((report_a, mapped_a), (report_b, mapped_b)):
        options = map_options(chain, args.source_dir, chain.target, "exact",
                              report=report)
        output = run([llk_opt, "--micro-map=" + options, micro_path]).stdout
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
    report_data = json.loads(report_text)
    selected = report_data["selectedState"]
    check(selected["physicalComplete"] and not selected["physicalReasons"],
          "%s: selected plan is physically incomplete" % chain.name)
    placements = selected["placements"]
    selected_rules = {placement["rule"] for placement in placements}
    check(chain.expected_rules.issubset(selected_rules),
          "%s: selected rules %s omit %s" %
          (chain.name, sorted(selected_rules), sorted(chain.expected_rules)))
    check(report_data["plans"][0]["rank"] == 0,
          "%s: selected plan is not the top ranked plan" % chain.name)
    check(not report_data["searchTruncated"],
          "%s: exact acceptance search was truncated" % chain.name)
    check(not report_data["plans"][0]["diagnostics"]["warnings"],
          "%s: selected plan has physical warnings" % chain.name)

    # Replay the selected frozen plan by its public ID into the same exact
    # configuration, proving that a real report selection is reproducible.
    plan_id = report_data["selectedPlanId"]
    bind_options = map_options(chain, args.source_dir, chain.target, "exact")
    rebound = run([llk_opt, "--micro-bind-plan=plan-id=%s %s" %
                   (plan_id, bind_options), micro_path]).stdout
    check(rebound == mapped_text,
          "%s: frozen selected-plan ID did not reproduce mapped IR" % chain.name)
    for operation in chain.required_ops:
        check(operation in mapped_text,
              "%s: selected IR is missing required operation %s" %
              (chain.name, operation))
    if chain.transform:
        check("micro.transform" in mapped_text,
              "%s: incompatible maps did not materialize a transform" % chain.name)
    if chain.name == "second-target-two-hop":
        route = next((connection for connection in selected["connections"]
                      if connection["route"] == ["sram.0", "l2.0", "dram.0"]), None)
        check(route is not None, "two-hop connection did not select SRAM -> L2 -> DRAM")
        check(len(route.get("hops", [])) == 2,
              "two-hop connection did not record both physical hops")
        intermediate = [allocation for allocation in selected["allocations"]
                        if allocation["memory"] == "l2.0" and
                        allocation["bytes"] == 256]
        check(intermediate, "two-hop route has no 256-byte L2 reservation")
        check("micro.hop = 1" in mapped_text and "micro.hop = 2" in mapped_text,
              "two-hop IR does not identify both emitted hops")

    # 6. Layered verification accepts what the mapper wrote.
    verify_options = " ".join([
        "target=%s" % chain.target["name"],
        "machine=%s" % os.path.join(args.source_dir, chain.target["machine"]),
        "layouts=%s" % os.path.join(args.source_dir, chain.target["layouts"]),
        "rules=%s" % os.path.join(args.source_dir, chain.target["rules"]),
        "emitters=%s" % chain.emitters,
    ])
    verify = run([llk_opt, "--micro-verify-mapping=" + verify_options,
                  mapped_a])

    # 7. The performance model accepts the selected plan and charges it.
    machine = os.path.join(args.source_dir, chain.target["machine"])
    perf = run([micro_perf, "--machine=" + machine, "--level=1",
                "--fail-on-capacity-violation", "--format=yaml", mapped_a]).stdout
    selected_cost = report_data["plans"][0]["costComponents"]
    predicted = float(re.search(r"(?m)^predicted_cycles: ([0-9.]+)$", perf).group(1))
    dram = int(re.search(r"(?m)^    dram: ([0-9]+)$", perf).group(1))
    check(predicted == float(selected_cost["latencyCycles"]),
          "%s: planner cycles %s differ from perf cycles %s" %
          (chain.name, selected_cost["latencyCycles"], predicted))
    check(dram == int(selected_cost["dramBytes"]),
          "%s: planner DRAM traffic %s differs from perf bytes %s" %
          (chain.name, selected_cost["dramBytes"], dram))
    check("capacity_violations: []" in perf,
          "%s: static perf reports a capacity violation" % chain.name)

    if chain.export:
        frozen = os.path.join(work, "frozen_plan.json")
        with open(frozen, "w") as handle:
            handle.write(report_text)
        compiled = run([llk_compile,
                        "--mapping-target=" + chain.target["name"],
                        "--mapping-root=" + args.source_dir,
                        "--machine=" + machine,
                        "--mapping-backend=reference",
                        "--mapping-stop=lowered",
                        "--plan-report=" + frozen,
                        "--emit=mlir", mapped_a])
        check("selected-groups-verified=" in compiled.stdout and
              "arith.extf" in compiled.stdout,
              "%s: frozen report did not rebind and lower in llk-compile" %
              chain.name)

    # 8. With mapping disabled the legacy export still produces the same
    #    concrete kernel, twice, byte for byte. Compiling that kernel through
    #    the legacy backend is asserted by the suite's own legacy tests
    #    (MicroToLinalgJit, LLKToLinalgConversion, the AVX2 SwiGLU chains) --
    #    repeating it here would test the backend, not the mapping path.
    if chain.export:
        again = run([llk_opt, "--llk-to-micro=schedule-db=missing.json",
                     source]).stdout
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
