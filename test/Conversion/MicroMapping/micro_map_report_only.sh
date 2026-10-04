#!/bin/sh
#===- micro_map_report_only.sh - --micro-map report-only=1 (design §21) --===//
#
# Design §21 requires the mapping tool to be able to emit a plan report
# *without* modifying the input IR. `--micro-map` binds the selected plan by
# default, so `report-only=1` is what makes inspection possible: same target,
# same search, same report -- no `bindPlanOntoModule`.
#
# The checks, in order:
#
#   1. `report-only=1 report=<path>` writes the report (a script, not FileCheck,
#      because the report is a second output stream in a second file);
#   2. the printed IR is byte-identical to the input file -- no `micro.plan` on
#      the kernel and no `micro.mapping` on any node;
#   3. the report is the *same* report the binding run writes, so report-only
#      changes only whether the plan is bound, not what is reported;
#   4. the guards: a search that produces no complete plan still fails (a
#      report-only run is not a way to swallow a real failure), and asking for
#      a report-only run with no report path is a usage error rather than a
#      silent no-op.
#
# On byte-identity: the fixture report_only_kernel.mlir is written in exactly
# MLIR's canonical printed form (no comments, canonical ssa names and attribute
# order), so the pass's stdout can be diffed against the file itself rather
# than against a second llk-opt run. That is the stronger statement -- the
# bytes the user's file had are the bytes they get back.
#
# Usage: micro_map_report_only.sh <llk-opt> <source-dir> <work-dir>
#===- -------------------------------------------------------------------===//

set -eu

LLK_OPT=$1
SRC=$2
WORK=$3

mkdir -p "$WORK"

# The five required target keys plus a deterministic mode, so the selected plan
# -- and therefore the report -- is reproducible. `report-only=1` is appended to
# the same option string, exactly as `report=` is elsewhere.
OPTIONS="target=x86-avx2 machine=$SRC/machines/x86-avx2-v2.yaml layouts=$SRC/mapping/x86-avx2/layouts.llkmap rules=$SRC/mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store mode=deterministic"
KERNEL="$SRC/test/Conversion/MicroMapping/report_only_kernel.mlir"
NO_PLAN="$SRC/test/Conversion/MicroMapping/report_only_no_plan.mlir"

# 1. A report-only run writes the report.
"$LLK_OPT" "--micro-map=$OPTIONS report-only=1 report=$WORK/report_only.json" \
    "$KERNEL" > "$WORK/report_only.mlir"

grep -q '"version"' "$WORK/report_only.json"
grep -q '"compilerVersion"' "$WORK/report_only.json"
grep -q '"inputModuleHash"' "$WORK/report_only.json"
grep -q '"selectedPlanId"' "$WORK/report_only.json"

# 2. ...and leaves the IR exactly as it was read.
diff "$KERNEL" "$WORK/report_only.mlir"

# The stamping grep, spelled out so a future fixture that happens to contain a
# `micro.plan` substring cannot make this pass vacuously. ERE (`-E`) rather than
# a BRE `\|` alternation, which is a grep extension and not POSIX.
if grep -Eq 'micro\.plan|micro\.mapping' "$WORK/report_only.mlir"; then
  echo "report-only=1 stamped the IR; it must not bind the plan" >&2
  exit 1
fi

# 2b. The fixture is not inert: without `report-only=1` the same options do bind
#     the plan, so the diff above is evidence about `report-only`, not about a
#     kernel the mapper declines to map.
"$LLK_OPT" "--micro-map=$OPTIONS report=$WORK/bound.json" \
    "$KERNEL" > "$WORK/bound.mlir"
grep -q 'micro\.plan' "$WORK/bound.mlir"
grep -q 'micro\.mapping' "$WORK/bound.mlir"

# 3. The report-only report is byte-identical to the binding run's: both are
#    written before binding, off the same search and the same module hash.
diff "$WORK/bound.json" "$WORK/report_only.json"

# 4a. A search that cannot produce a plan still fails the pass, report-only or
#     not, with the search's own diagnostic.
if "$LLK_OPT" "--micro-map=$OPTIONS report-only=1 report=$WORK/no_plan.json" \
      "$NO_PLAN" > "$WORK/no_plan.out" 2>&1; then
  echo "report-only=1 succeeded although the search found no plan" >&2
  exit 1
fi
grep -q 'the search produced no complete plan' "$WORK/no_plan.out"
# ...and no report is written for a search that failed, so a stale or empty
# report cannot be mistaken for a successful one.
if [ -e "$WORK/no_plan.json" ]; then
  echo "a failed report-only run left a report file behind" >&2
  exit 1
fi

# 4b. `report-only=1` with nothing to report into is a usage error: the run
#     would otherwise search and then discard the result silently.
if "$LLK_OPT" "--micro-map=$OPTIONS report-only=1" \
      "$KERNEL" > "$WORK/no_path.out" 2>&1; then
  echo "report-only=1 without report= succeeded" >&2
  exit 1
fi
grep -q 'report-only requires report=' "$WORK/no_path.out"
