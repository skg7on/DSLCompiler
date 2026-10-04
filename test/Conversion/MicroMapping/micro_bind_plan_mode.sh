#!/bin/sh
#===- micro_bind_plan_mode.sh - mode-aware plan-id reproduction (design §21) ==//
#
# A plan id is a content hash, so reproducing one means replaying the search
# that produced it -- with the same mode, beam-width, and top-k, not merely the
# same target files. This script pins that contract:
#
#   1. an id from `--micro-map mode=beam` binds again with `--micro-bind-plan
#      mode=beam` (the documented report -> bind workflow, in beam mode), and
#      the bound plan's own decimal id reproduces the same plan;
#   2. the mode actually drives the search: an id from `mode=deterministic` is
#      NOT found by `mode=beam` with the same cap, and the diagnostic names the
#      mode that was searched.
#
# The fixture rules (bind_plan_mode_rules.llkmap) deliberately make the two
# modes pick different plans, so a binder that ignored `mode` (or forced a
# fixed one) cannot pass.
#
# Usage: micro_bind_plan_mode.sh <llk-opt> <source-dir> <work-dir>
#===- --------------------------------------------------------------------===//

set -eu

LLK_OPT=$1
SRC=$2
WORK=$3

mkdir -p "$WORK"

TARGET="target=probe machine=$SRC/test/Conversion/MicroMapping/probe_machine.yaml layouts=$SRC/mapping/x86-avx2/layouts.llkmap rules=$SRC/test/Conversion/MicroMapping/bind_plan_mode_rules.llkmap emitters=e1"
KERNEL="$SRC/test/Conversion/MicroMapping/micro_map.mlir"

# 1. Beam mode round-trips: map in beam, bind the reported id in beam.
"$LLK_OPT" "--micro-map=$TARGET mode=beam top-k=8 report=$WORK/beam.json" \
    "$KERNEL" > "$WORK/beam_map.mlir"
BEAM_HEX=$(sed -n 's/.*"selectedPlanId": "\([0-9a-f][0-9a-f]*\)".*/\1/p' \
             "$WORK/beam.json")
[ -n "$BEAM_HEX" ] || { echo "beam report had no selectedPlanId" >&2; exit 1; }

"$LLK_OPT" "--micro-bind-plan=plan-id=$BEAM_HEX $TARGET mode=beam top-k=8" \
    "$KERNEL" > "$WORK/beam_bind.mlir"
grep -q 'micro\.plan' "$WORK/beam_bind.mlir" || {
  echo "the beam id did not bind in beam mode" >&2; exit 1; }

# The bound plan's own decimal id names the same plan: binding it again by that
# spelling succeeds, proving the report's hex and the emitted decimal agree.
BEAM_DEC=$(grep -o 'micro\.plan = {[^}]*}' "$WORK/beam_bind.mlir" |
    sed -n 's/.*id = \([0-9-][0-9-]*\) : i64.*/\1/p')
[ -n "$BEAM_DEC" ] || { echo "the bound IR carried no plan id" >&2; exit 1; }
"$LLK_OPT" "--micro-bind-plan=plan-id=$BEAM_DEC $TARGET mode=beam top-k=8" \
    "$KERNEL" > "$WORK/beam_decimal.mlir"
grep -q 'micro\.plan' "$WORK/beam_decimal.mlir" || {
  echo "the bound plan's decimal id did not reproduce" >&2; exit 1; }

# 2. The requested mode drives the search. With top-k=1 the deterministic search
#    keeps the costly canonical plan and the beam search the cheap one, so the
#    deterministic id must be absent from a beam search.
"$LLK_OPT" "--micro-map=$TARGET mode=deterministic top-k=1 report=$WORK/det.json" \
    "$KERNEL" > "$WORK/det_map.mlir"
DET_HEX=$(sed -n 's/.*"selectedPlanId": "\([0-9a-f][0-9a-f]*\)".*/\1/p' \
            "$WORK/det.json")
[ -n "$DET_HEX" ] || { echo "deterministic report had no selectedPlanId" >&2; exit 1; }
[ "$DET_HEX" != "$BEAM_HEX" ] || {
  echo "fixture is degenerate: both modes reported the same plan" >&2; exit 1; }

# 2a. Binding it in its own mode succeeds.
"$LLK_OPT" "--micro-bind-plan=plan-id=$DET_HEX $TARGET mode=deterministic top-k=1" \
    "$KERNEL" > "$WORK/det_bind.mlir"
grep -q 'micro\.plan' "$WORK/det_bind.mlir" || {
  echo "the deterministic id did not bind in deterministic mode" >&2; exit 1; }

# 2b. Binding it in beam mode fails, and the diagnostic names the searched mode.
if "$LLK_OPT" "--micro-bind-plan=plan-id=$DET_HEX $TARGET mode=beam top-k=1" \
      "$KERNEL" > "$WORK/mismatch.mlir" 2>&1; then
  echo "expected the deterministic id to be absent from a beam search" >&2
  exit 1
fi
grep -q 'no plan in the beam search has id' "$WORK/mismatch.mlir" || {
  echo "the mismatch diagnostic did not name the beam search" >&2; exit 1; }

echo "micro_bind_plan_mode: ok"
