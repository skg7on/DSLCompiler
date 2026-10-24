#!/usr/bin/env python3
"""Framework E7 (#182): both ARM target names through the public CLI.

The ARM targets are static mapping targets. What this runner establishes is
that a caller who knows only the command line gets a complete plan, a verified
mapped kernel, a frozen replay of the reported plan id, and an analysis that
agrees with the plan -- for both target names and both contraction dtypes, on
any host. It never claims instruction execution: there is no ARM hardware in
this path, and every number below is modeled.

Everything goes through public flags and files in the work directory this
script is given; it imports no project library. The four cases are the whole
product: `arm-sve` and `arm-sme` x f32 and bf16 contraction. Each case runs its
own subdirectory so the repeat and replay controls compare files that only that
case wrote.

The assertions are concrete decisions, not "a rule matched":

  * the selected compute identity and the accumulator memory are the target's
    own (`vector_engine/svp` + `acc.0` for SVE, `matrix_engine/za` + `za.0` for
    SME), with the exact 1024-byte accumulator footprint the fixtures declare;
  * the plan is physically complete with no reasons, and its content hashes
    are well-formed and identical across targets for the same source;
  * a fresh process replays the reported `selectedPlanId` to byte-identical
    normalized IR;
  * `micro-perf` re-derives the planner's cycles, per-level traffic and live
    storage peaks from the bound IR under the shared conventions;
  * a bounded exact search either agrees with the uncapped best explored cost
    or reports its own truncation. Nothing here asserts global optimality or
    treats a modeled cost as a measurement.
"""

import json
import os
import re
import subprocess
import sys


class CaseFailure(Exception):
    pass


def run(command, expect_success=True):
    result = subprocess.run(command, capture_output=True, text=True)
    if expect_success and result.returncode != 0:
        raise CaseFailure(
            "command failed (%d): %s\n%s\n%s" % (
                result.returncode, " ".join(command),
                result.stderr.strip(), result.stdout.strip()))
    return result


def check(condition, message):
    if not condition:
        raise CaseFailure(message)


def read(path):
    with open(path, "r") as handle:
        return handle.read()


HEX_ID = re.compile(r"[0-9a-f]{16}")
PEAK = re.compile(r"storage plan: memory '([^']+)' peak (\d+) bytes of (\d+)")


# The two target packages, exactly as their plugins name them: the ten declared
# emitter keys and the concrete `micro-perf` profile. `compute` and
# `accumulator` are the identities this runner expects the *plan* to carry, so a
# policy or machine change that swapped them would fail here rather than pass as
# "some engine was selected".
TARGETS = {
    "arm-sve": {
        "machine": "machines/arm-sve-v2.yaml",
        "layouts": "mapping/arm-sve/layouts.llkmap",
        "rules": "mapping/arm-sve/rules.llkmap",
        "emitters": ",".join([
            "sve_vector_add", "sve_vector_convert", "sve_vector_silu",
            "sve_vector_mul", "sve_reduce", "sve_copy", "sve_tile_copy",
            "sve_tile_store", "sve_fused_convert_silu_mul", "sve_mma"]),
        "compute": {"vector_engine": "svp"},
        "accumulator": "acc.0",
        "accumulator_capacity": 2048,
        "emitter": "sve_mma",
        "mma_rules": {"f32": "sve.mma_f32", "bf16": "sve.mma_bf16"},
    },
    "arm-sme": {
        "machine": "machines/arm-sme-v2.yaml",
        "layouts": "mapping/arm-sme/layouts.llkmap",
        "rules": "mapping/arm-sme/rules.llkmap",
        "emitters": ",".join([
            "sme_vector_add", "sme_vector_convert", "sme_vector_silu",
            "sme_vector_mul", "sme_reduce", "sme_copy", "sme_tile_copy",
            "sme_tile_store", "sme_fused_convert_silu_mul", "sme_mma_za"]),
        "compute": {"matrix_engine": "za"},
        "accumulator": "za.0",
        "accumulator_capacity": 4096,
        "emitter": "sme_mma_za",
        "mma_rules": {"f32": "sme.mma_f32", "bf16": "sme.mma_bf16"},
    },
}

# Both dtypes share the same fixture shape and kernel symbol; only the declared
# element type and the rule table entry differ.
FIXTURES = (("f32", "contraction_f32.mlir"), ("bf16", "contraction_bf16.mlir"))


def target_paths(target, source_dir):
    """The absolute target files every stage must be given, unchanged."""
    return {
        "machine": os.path.join(source_dir, target["machine"]),
        "layouts": os.path.join(source_dir, target["layouts"]),
        "rules": os.path.join(source_dir, target["rules"]),
    }


def map_options(target, target_name, paths, mode, top_k, report=None):
    options = [
        "target=%s" % target_name,
        "machine=%s" % paths["machine"],
        "layouts=%s" % paths["layouts"],
        "rules=%s" % paths["rules"],
        "emitters=%s" % target["emitters"],
        "mode=%s" % mode,
        "top-k=%d" % top_k,
    ]
    if report:
        options.append("report=%s" % report)
    # The pass parses its options the way the command line spells them, so a
    # space separates them and every path stays absolute.
    return " ".join(options)


def verify_options(target, target_name, paths):
    # Verification resolves ids recorded on the kernel. Give it the target's
    # whole declared key list rather than only the keys this fixture happens to
    # use: resolving the registry is part of what the pass establishes.
    return " ".join([
        "target=%s" % target_name,
        "machine=%s" % paths["machine"],
        "layouts=%s" % paths["layouts"],
        "rules=%s" % paths["rules"],
        "emitters=%s" % target["emitters"],
    ])


def parse_scalar(value):
    if value == "[]":
        return []
    if value.startswith("[") and value.endswith("]"):
        body = value[1:-1].strip()
        return [] if not body else [parse_scalar(part.strip())
                                    for part in body.split(",")]
    if value in ("true", "false"):
        return value == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")


def parse_yaml(text):
    """Parses the block-mapping subset `micro-perf --format=yaml` emits.

    Standard library only, like the suite's other pipeline runners. The
    indentation is significant because `totals.bytes.acc` and
    `tiles.live_bytes.acc` are different measurements with the same key.
    """
    root = {}
    stack = [(-1, root)]
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, separator, value = line.strip().partition(":")
        if not separator:
            raise CaseFailure("unparseable perf line: %r" % line)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = value.strip()
        if not value:
            child = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = parse_scalar(value)
    return root


def storage_peaks(notes):
    """The storage plan's peak per memory, as the report states it."""
    peaks = {}
    for note in notes:
        match = PEAK.search(note)
        if match:
            peaks[match.group(1)] = (int(match.group(2)), int(match.group(3)))
    return peaks


def check_hashes(report, name):
    for key in ("inputModuleHash", "machineHash", "layoutLibraryHash",
                "ruleLibraryHash", "targetHash", "graphHash"):
        value = report.get(key)
        check(isinstance(value, str) and HEX_ID.fullmatch(value),
              "%s: report has no well-formed %s (got %r)" % (name, key, value))
    check(report["inputModuleHash"] != "0" * 16,
          "%s: the source module hash is the unfinalized sentinel" % name)
    check(report["sourceBindingHash"] == "0" * 16,
          "%s: binding-free search recorded a source binding" % name)


def check_selected_identity(report, target, target_name, dtype, name):
    state = report["selectedState"]
    plan = report["plans"][0]

    check(report["target"] == target_name,
          "%s: report target %r is not %r" %
          (name, report["target"], target_name))
    check(state["materialized"] is True,
          "%s: selected plan is not materialized" % name)
    check(state["physicalComplete"] is True,
          "%s: selected plan is physically incomplete" % name)
    check(state["physicalReasons"] == [],
          "%s: a complete plan carries physical reasons: %s" %
          (name, state["physicalReasons"]))
    check(report["searchTruncated"] is False,
          "%s: the unbounded deterministic search claims truncation" % name)
    check(report["plans"] and len(report["plans"]) == report["counts"]["plans"],
          "%s: reported plans do not match the retained count" % name)
    check(plan["rank"] == 0,
          "%s: the selected plan is not the top-ranked plan" % name)
    check(report["selectedPlanId"] == plan["id"] == state["id"],
          "%s: selected id %r, plan id %r, state id %r disagree" %
          (name, report["selectedPlanId"], plan["id"], state["id"]))
    check(HEX_ID.fullmatch(report["selectedPlanId"]),
          "%s: selected plan id is not 16-digit hex" % name)

    # The source identity is the unbound module: no candidate binding was
    # supplied, and both search entry points must agree about that.
    check(report["sourceBindingHash"] == state["sourceBindingHash"],
          "%s: report and selected state disagree about the source binding" %
          name)
    check(report["sourceBindingCandidate"] == "",
          "%s: the search unexpectedly consulted a candidate" % name)
    check(report["sourceBinding"] == "",
          "%s: the search unexpectedly carried a binding" % name)

    compute = [placement for placement in state["placements"]
               if placement["compute"]]
    check(len(compute) == 1,
          "%s: expected exactly one compute placement, got %d" %
          (name, len(compute)))
    placement = compute[0]
    check(placement["compute"] == target["compute"],
          "%s: selected compute identity is %s, not %s" %
          (name, placement["compute"], target["compute"]))
    check(placement["rule"] == target["mma_rules"][dtype],
          "%s: selected rule %r is not %r" %
          (name, placement["rule"], target["mma_rules"][dtype]))
    check(placement["emitter"] == target["emitter"],
          "%s: selected emitter %r is not %r" %
          (name, placement["emitter"], target["emitter"]))
    check(placement["executor"] == "worker.0",
          "%s: selected executor %r is not the machine's worker" %
          (name, placement["executor"]))

    # The accumulator is a memory identity, not a label: the MMA's third input
    # and its output must be recorded in the target's own accumulator memory.
    ports = {(port["port"]["direction"], port["port"]["index"]): port["memory"]
             for port in placement["portMemories"]}
    check(ports == {("input", 2): target["accumulator"],
                    ("output", 0): target["accumulator"]},
          "%s: MMA port memories %s do not name %s for both accumulator ports" %
          (name, ports, target["accumulator"]))

    accumulator = [allocation for allocation in state["allocations"]
                   if allocation["memory"] == target["accumulator"]]
    check(accumulator,
          "%s: no allocation lives in %s" % (name, target["accumulator"]))
    check(all(allocation["bytes"] == 1024 for allocation in accumulator),
          "%s: the accumulator footprint is not the fixture's 1024 bytes" % name)
    check(any(not allocation.get("aliasOf") for allocation in accumulator),
          "%s: every accumulator allocation is an alias" % name)
    check(len([allocation for allocation in state["allocations"]
               if allocation["memory"] == "sram.0"]) >= 2,
          "%s: the staged operands have no SRAM reservation" % name)
    check(any(allocation.get("borrowed") for allocation in state["allocations"]
              if allocation["memory"] == "dram.0"),
          "%s: the borrowed DRAM arguments are not reserved" % name)

    # The accumulator is carried on a connection, so the bound IR's routes and
    # the budget's accounting speak about the same storage.
    check(any(connection["route"] == [target["accumulator"]]
              for connection in state["connections"]),
          "%s: no selected route crosses %s" %
          (name, target["accumulator"]))
    return plan


def check_perf_parity(plan, target, target_name, mapped, perf, machine, name):
    cost = plan["costComponents"]
    peaks = storage_peaks(plan["diagnostics"]["storageNotes"])
    analysis = parse_yaml(run([
        perf, "--machine=" + machine, "--level=1",
        "--fail-on-capacity-violation", "--format=yaml", mapped,
    ]).stdout)

    check(analysis["complete"] is True,
          "%s: the bound kernel's analysis is incomplete: %s" %
          (name, analysis.get("incomplete_reasons")))
    check(analysis["incomplete_reasons"] == [],
          "%s: the bound kernel reports incomplete reasons" % name)
    check(analysis["capacity_violations"] == [],
          "%s: the analysis reports a capacity violation" % name)
    check(analysis["machine"] == target_name,
          "%s: analysis ran against %r, not %r" %
          (name, analysis["machine"], target_name))
    check(analysis["level"] == 1,
          "%s: analysis level is %r" % (name, analysis["level"]))
    check(analysis["kernel"] == "arm_contraction",
          "%s: analysis kernel %r is the fixture's" %
          (name, analysis["kernel"]))

    # Cycles: the planner's predicted latency and the analysis of the bound IR
    # are the same number under the shared cost convention. Neither is a
    # measurement.
    check(analysis["predicted_cycles"] == int(float(cost["latencyCycles"])),
          "%s: planner cycles %s differ from analysis cycles %s" %
          (name, cost["latencyCycles"], analysis["predicted_cycles"]))
    check(analysis["totals"]["bytes"]["dram"] == cost["dramBytes"],
          "%s: planner DRAM traffic %s differs from analysis bytes %s" %
          (name, cost["dramBytes"], analysis["totals"]["bytes"]["dram"]))
    check(analysis["totals"]["bytes"]["sram"] == cost["localBytes"],
          "%s: planner local traffic %s differs from analysis bytes %s" %
          (name, cost["localBytes"], analysis["totals"]["bytes"]["sram"]))

    # Live storage: the plan's own peaks are what the analysis must reproduce,
    # keyed by memory kind. `acc` is the kind label for whichever accumulator
    # memory the target names (acc.0 or za.0).
    check(target["accumulator"] in peaks,
          "%s: the storage plan states no peak for %s" %
          (name, target["accumulator"]))
    check("sram.0" in peaks,
          "%s: the storage plan states no SRAM peak" % name)
    accumulator_peak, accumulator_capacity = peaks[target["accumulator"]]
    check(accumulator_peak == 1024,
          "%s: the accumulator peak is %d, not the fixture's 1024 bytes" %
          (name, accumulator_peak))
    check(accumulator_capacity == target["accumulator_capacity"],
          "%s: %s capacity is %d, not %d" %
          (name, target["accumulator"], accumulator_capacity,
           target["accumulator_capacity"]))
    check(analysis["tiles"]["live_bytes"]["acc"] == accumulator_peak,
          "%s: analysis accumulator live peak %s differs from the plan's %s" %
          (name, analysis["tiles"]["live_bytes"]["acc"], accumulator_peak))
    check(analysis["tiles"]["live_bytes"]["sram"] == peaks["sram.0"][0],
          "%s: analysis SRAM live peak %s differs from the plan's %s" %
          (name, analysis["tiles"]["live_bytes"]["sram"], peaks["sram.0"][0]))
    for memory, (peak, capacity) in peaks.items():
        check(peak <= capacity,
              "%s: %s peak %d exceeds capacity %d" %
              (name, memory, peak, capacity))


def check_bounded_exact(llk_opt, target, target_name, paths, source, cost,
                        work, name):
    report_path = os.path.join(work, "bounded_exact.json")
    options = map_options(target, target_name, paths, "exact", 1,
                          report=report_path)
    run([llk_opt, "--micro-map=" + options, source])
    bounded = json.loads(read(report_path))

    check(bounded["searchOptions"]["mode"] == "exact",
          "%s: bounded search did not run in exact mode" % name)
    check(bounded["searchOptions"]["topK"] == 1,
          "%s: bounded search did not record topK=1" % name)
    check(len(bounded["plans"]) <= 1,
          "%s: the bounded exact search retained more than its cap" % name)
    if bounded["searchTruncated"]:
        # A cap that bit is reported as one; nothing is claimed about plans the
        # cap discarded.
        check(len(bounded["plans"]) == 1,
              "%s: a truncated search retained no plan to report" % name)
    else:
        check(bounded["plans"][0]["costComponents"]["latencyCycles"] ==
              cost["latencyCycles"],
              "%s: the bounded exact search disagrees with the uncapped best "
              "explored cost" % name)


def run_case(target_name, target, dtype, fixture, args, work, seen_sources,
             report_lines):
    name = "%s/%s" % (target_name, dtype)
    source = os.path.join(args.source_dir, "test/Conversion/MicroMapping",
                          "Inputs/arm", fixture)
    check(os.path.exists(source), "%s: missing fixture %s" % (name, source))
    paths = target_paths(target, args.source_dir)
    for path in paths.values():
        check(os.path.exists(path), "%s: missing target file %s" % (name, path))

    # 1. The unbounded deterministic search, twice, in fresh processes and into
    #    different report/IR paths. The repeat control compares what the search
    #    wrote, not a cached file.
    report_a = os.path.join(work, "report_a.json")
    report_b = os.path.join(work, "report_b.json")
    mapped_a = os.path.join(work, "mapped_a.mlir")
    mapped_b = os.path.join(work, "mapped_b.mlir")
    for report, mapped in ((report_a, mapped_a), (report_b, mapped_b)):
        options = map_options(target, target_name, paths, "deterministic", 0,
                              report=report)
        output = run([args.llk_opt, "--micro-map=" + options, source]).stdout
        with open(mapped, "w") as handle:
            handle.write(output)

    check(read(report_a) == read(report_b),
          "%s: the plan report is not byte-identical across runs" % name)
    check(read(mapped_a) == read(mapped_b),
          "%s: the mapped IR is not byte-identical across runs" % name)

    report = json.loads(read(report_a))
    mapped_text = read(mapped_a)
    check_hashes(report, name)
    check(report["searchOptions"]["mode"] == "deterministic",
          "%s: the report does not record deterministic mode" % name)
    check(report["searchOptions"]["topK"] == 0,
          "%s: the report does not record the uncapped retention" % name)
    plan = check_selected_identity(report, target, target_name, dtype, name)

    # The same source must hash the same under both targets: the graph is the
    # source's identity, not the target's.
    previous = seen_sources.setdefault(fixture, report["inputModuleHash"])
    check(previous == report["inputModuleHash"],
          "%s: the source hash differs between target names" % name)

    # 2. The mapped IR carries the plan and its coverage, and a fresh
    #    verification pass resolves every recorded id against the target.
    for marker in ("micro.plan", "micro.mapping", "micro.routes"):
        check(marker in mapped_text,
              "%s: the mapped kernel has no %s" % (name, marker))
    plan_id_value = int(report["selectedPlanId"], 16)
    signed_id = (plan_id_value - (1 << 64) if plan_id_value >= (1 << 63)
                 else plan_id_value)
    check(("id = %d : i64" % plan_id_value) in mapped_text or
          ("id = %d : i64" % signed_id) in mapped_text,
          "%s: the mapped plan does not record the reported id" % name)
    check("truncated = false" in mapped_text,
          "%s: the bound plan does not record an untruncated search" % name)
    run([args.llk_opt, "--micro-verify-mapping=" +
         verify_options(target, target_name, paths), mapped_a])

    # 3. Frozen replay: the reported id, in a new process, with identical
    #    target and search inputs, reproduces the selected state and IR.
    search_options = map_options(target, target_name, paths, "deterministic", 0)
    rebound_text = run([
        args.llk_opt,
        "--micro-bind-plan=plan-id=%s %s" % (report["selectedPlanId"],
                                             search_options),
        source,
    ]).stdout
    check(rebound_text == mapped_text,
          "%s: the frozen selected-plan id did not reproduce the mapped IR" %
          name)
    rebound_path = os.path.join(work, "rebound.mlir")
    with open(rebound_path, "w") as handle:
        handle.write(rebound_text)
    normalized_mapped = run([args.llk_opt, "--mlir-print-op-generic",
                             mapped_a]).stdout
    normalized_rebound = run([args.llk_opt, "--mlir-print-op-generic",
                              rebound_path]).stdout
    check(normalized_mapped == normalized_rebound,
          "%s: normalized replay IR differs from the mapped IR" % name)
    # The bound plan carries the content it was searched against, so a replay
    # that lost a hash would be a plan nobody can re-derive.
    for key in ("machineHash", "layoutLibraryHash", "ruleLibraryHash",
                "targetHash", "graphHash"):
        check(report[key] in normalized_rebound,
              "%s: the replayed plan lost its %s provenance" % (name, key))
    check("binding_hash = 0" in normalized_rebound,
          "%s: the replayed plan invented a source binding" % name)

    # 4. The shared analysis accepts the bound kernel and re-derives the plan.
    check_perf_parity(plan, target, target_name, mapped_a, args.micro_perf,
                      paths["machine"], name)

    # 5. A bounded exact search either agrees with the best explored cost or
    #    says it was truncated.
    check_bounded_exact(args.llk_opt, target, target_name, paths, source,
                        plan["costComponents"], work, name)

    compute = ", ".join("%s=%s" % item for item in target["compute"].items())
    report_lines.append(
        "  %-14s ok  plan=%s  cycles=%s  compute=%s  acc=%s peak=1024"
        % (name, report["selectedPlanId"],
           plan["costComponents"]["latencyCycles"], compute,
           target["accumulator"]))


def main():
    if len(sys.argv) != 5:
        print(__doc__, file=sys.stderr)
        print("usage: arm_target_pipeline.py <llk-opt> <micro-perf> "
              "<source-dir> <work-dir>", file=sys.stderr)
        return 2

    class Args:
        pass

    args = Args()
    args.llk_opt, args.micro_perf, args.source_dir, args.work_dir = sys.argv[1:]

    # The runner owns its scratch: it writes only here, and each case reads only
    # what it wrote.
    os.makedirs(args.work_dir, exist_ok=True)

    report_lines = []
    failures = []
    seen_sources = {}
    for target_name, target in TARGETS.items():
        for dtype, fixture in FIXTURES:
            case_dir = os.path.join(args.work_dir, "%s-%s" %
                                    (target_name, dtype))
            os.makedirs(case_dir, exist_ok=True)
            try:
                run_case(target_name, target, dtype, fixture, args, case_dir,
                         seen_sources, report_lines)
            except CaseFailure as failure:
                failures.append(str(failure))
                report_lines.append("  %-14s FAIL" % ("%s/%s" %
                                                      (target_name, dtype)))

    print("arm target pipeline:")
    for line in report_lines:
        print(line)

    if failures:
        print("\n%d case(s) failed:" % len(failures), file=sys.stderr)
        for failure in failures:
            print("  " + failure, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
