#!/bin/sh
#===- micro_map_report.sh - --micro-map report wiring (design §22.2) -----===//
#
# Checks the `report=<path>` option's contract, which FileCheck cannot express
# because it speaks about a single stdout stream:
#
#   1. the report file is produced and is well-formed enough to carry a version
#      and a selected plan id;
#   2. the report is metadata -- the mapped IR is byte-identical with and
#      without `report=`;
#   3. two separate llk-opt invocations produce byte-identical reports.
#
# Usage: micro_map_report.sh <llk-opt> <source-dir> <work-dir>
#===- -------------------------------------------------------------------===//

set -eu

LLK_OPT=$1
SRC=$2
WORK=$3

mkdir -p "$WORK"

# The five required target keys plus a deterministic mode, so the selected plan
# is reproducible. `report=` is appended to the same option string.
OPTIONS="target=x86-avx2 machine=$SRC/machines/x86-avx2-v2.yaml layouts=$SRC/mapping/x86-avx2/layouts.llkmap rules=$SRC/mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_mma,avx2_reduce,avx2_copy mode=deterministic"
KERNEL="$SRC/test/Conversion/MicroMapping/micro_map.mlir"

"$LLK_OPT" "--micro-map=$OPTIONS report=$WORK/report_a.json" "$KERNEL" > "$WORK/with_report.mlir"
"$LLK_OPT" "--micro-map=$OPTIONS report=$WORK/report_b.json" "$KERNEL" > "$WORK/with_report_b.mlir"
"$LLK_OPT" "--micro-map=$OPTIONS" "$KERNEL" > "$WORK/without_report.mlir"

# 1. The report exists and names a version and a selected plan.
grep -q '"version"' "$WORK/report_a.json"
grep -q '"selectedPlanId"' "$WORK/report_a.json"
grep -q '"compilerVersion"' "$WORK/report_a.json"

# 2. The report is metadata: the mapped IR is byte-identical either way.
diff "$WORK/without_report.mlir" "$WORK/with_report.mlir"

# 3. Two separate invocations produce byte-identical reports and IR.
diff "$WORK/report_a.json" "$WORK/report_b.json"
diff "$WORK/with_report.mlir" "$WORK/with_report_b.mlir"
