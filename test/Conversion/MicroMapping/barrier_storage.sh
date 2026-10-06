#!/bin/sh
#===- barrier_storage.sh - the storage plan runs on the selected plan -------===//
#
# The storage plan (`finalizeStoragePlan`, task B3) must run on the plan the CLI
# actually produces -- not only in the unit tests. This checks the three
# observable consequences of wiring it in (task C1):
#
#   1. a multi-engine movement is barrier-synchronized, so the mapped IR carries
#      a `micro.barrier` stamped with the connection it represents;
#   2. the report's `selectedState` carries populated allocations, synchronization
#      decisions and the plan-step DAG, rather than empty arrays;
#   3. the B6 barrier verification fires: the mapped IR verifies, and deleting the
#      barrier makes verification fail with the "kernel does not express"
#      diagnostic. (The barrier was never emitted before the fix, so the mapping
#      half of this test could not have passed.)
#
# A script because FileCheck speaks about one stdout stream, while this compares
# the IR, the report file, and a verification outcome across runs.
#
# Usage: barrier_storage.sh <llk-opt> <source-dir> <work-dir>
#===- -------------------------------------------------------------------===//

set -eu

LLK_OPT=$1
SRC=$2
WORK=$3

mkdir -p "$WORK"

FIXTURES="$SRC/test/Conversion/MicroMapping"
MAP_OPTIONS="target=barrier machine=$FIXTURES/barrier_machine.yaml layouts=$SRC/mapping/x86-avx2/layouts.llkmap rules=$FIXTURES/unmaterialized_rules.llkmap emitters=e1 mode=deterministic require-executable=1"
VERIFY_OPTIONS="target=barrier machine=$FIXTURES/barrier_machine.yaml layouts=$SRC/mapping/x86-avx2/layouts.llkmap rules=$FIXTURES/unmaterialized_rules.llkmap emitters=e1"

# 1. Map, keeping the report and the mapped IR.
"$LLK_OPT" "--micro-map=$MAP_OPTIONS report=$WORK/report.json" \
    "$FIXTURES/barrier_sync.mlir" > "$WORK/mapped.mlir"

# The mapped IR carries the barrier over the movement's tokens.
grep -q 'micro\.barrier' "$WORK/mapped.mlir"

# 2. The report's selected state is populated: concrete allocations in SRAM and
#    DRAM, a synchronization decision that requires a barrier, and plan steps.
grep -q '"allocations"' "$WORK/report.json"
grep -q '"memory": "sram.0"' "$WORK/report.json"
grep -q '"memory": "dram.0"' "$WORK/report.json"
grep -q '"requiresBarrier": true' "$WORK/report.json"
grep -q '"kind": "movement"' "$WORK/report.json"
grep -q '"kind": "synchronization"' "$WORK/report.json"
grep -q '"stepEdges"' "$WORK/report.json"

# 3. The mapped IR verifies -- which requires the required barrier to be
#    represented (design §18.2, task B6).
"$LLK_OPT" "--micro-verify-mapping=$VERIFY_OPTIONS" "$WORK/mapped.mlir" \
    > /dev/null

# 3b. Delete the barrier and verification must fail: the required
#     synchronization the plan recorded is no longer expressed.
grep -v 'micro\.barrier' "$WORK/mapped.mlir" > "$WORK/no_barrier.mlir"
if "$LLK_OPT" "--micro-verify-mapping=$VERIFY_OPTIONS" "$WORK/no_barrier.mlir" \
        > /dev/null 2> "$WORK/no_barrier.err"; then
  echo "barrier_storage: verification passed without the required barrier" >&2
  exit 1
fi
grep -q 'does not express' "$WORK/no_barrier.err"
