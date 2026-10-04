#!/bin/sh
#===- micro_bind_plan_roundtrip.sh - report -> bind reproduces the IR (design §21)
#
# `--micro-map report=<path>` and `--micro-bind-plan plan-id=<id>` are two public
# entry points into the same search-and-bind engine. The report names the plan
# `--micro-map` bound; handing that id back to `--micro-bind-plan` with the same
# search options must bind *the same* plan. That is a stronger statement than
# "a plan was bound" -- and stronger than the id-parsing and mode tests, which
# only prove the reported id is accepted -- so this script asserts the bound IR
# is byte-identical to the mapped IR.
#
# The three cases:
#   1. deterministic mode, shipped x86-avx2 target, over the hand-authored
#      copy/add kernel the other mapping tests use (micro_map.mlir);
#   2. beam mode over the probe target whose fixture rules make the beam search
#      select a different plan from the deterministic one
#      (bind_plan_mode_rules.llkmap), so the round-trip holds for the mode the
#      report -> bind workflow is documented with, not only for the canonical
#      deterministic order;
#   2b. the same beam search with a non-default `beam-width`. The width is part
#      of the replay contract -- the fixture's frontier exceeds one entry, so a
#      width of 1 truncates the search and folds `searchTruncated` into the
#      plan's content hash, giving a different id from the default width 64. A
#      binder that dropped `beam-width` would replay 64, miss the narrow id, and
#      fail this round-trip;
#   3. exact mode over a kernel produced by `--llk-to-micro` (matmul_e2e.mlir),
#      saved to disk and mapped in a second invocation. This closes the gap the
#      other mapping tests leave open: the round-trip is proven on
#      compiler-generated IR, not only on a hand-authored kernel.
#
# Why this needs Task 3/4: the id is read as the report prints it (bare 16-digit
# hex) and the search must be replayed with the same mode and top-k. Before
# those fixes `--micro-bind-plan` could not read the report's spelling, and did
# not carry `mode`, so it either failed to parse the id or matched a different
# search -- in both cases this script fails at the bind, or at the diff when a
# different plan is bound.
#
# Usage: micro_bind_plan_roundtrip.sh <llk-opt> <source-dir> <work-dir>
#===- -------------------------------------------------------------------===//

set -eu

LLK_OPT=$1
SRC=$2
WORK=$3

mkdir -p "$WORK"

# roundtrip <name> <kernel> <search-options>
#   Maps <kernel> with `report=` using <search-options>, reads the selected plan
#   id from the report, binds that id with the same options, and asserts the two
#   IR streams are byte-identical.
roundtrip() {
  name=$1
  kernel=$2
  options=$3
  "$LLK_OPT" "--micro-map=$options report=$WORK/$name.report.json" "$kernel" \
      > "$WORK/$name.mapped.mlir"
  id=$(sed -n 's/.*"selectedPlanId": "\([0-9a-f][0-9a-f]*\)".*/\1/p' \
        "$WORK/$name.report.json")
  [ -n "$id" ] || { echo "roundtrip($name): the report had no selectedPlanId" >&2
    exit 1; }
  "$LLK_OPT" "--micro-bind-plan=plan-id=$id $options" "$kernel" \
      > "$WORK/$name.bound.mlir"
  # The bound IR must be the mapped IR, byte for byte. On failure diff prints
  # the differing lines and exits non-zero, which fails the script under `set -e`.
  diff "$WORK/$name.mapped.mlir" "$WORK/$name.bound.mlir"
  # Guard against a vacuous pass: the kernel really did receive a plan.
  grep -q 'micro\.plan' "$WORK/$name.bound.mlir"
}

# The shipped AVX2 target: the same five keys plus every emitter the mapper may
# bind, so a complete plan is found and the report always names one.
X86="target=x86-avx2 machine=$SRC/machines/x86-avx2-v2.yaml layouts=$SRC/mapping/x86-avx2/layouts.llkmap rules=$SRC/mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store"

# 1. Deterministic mode. `top-k=8` is passed to both invocations explicitly so
#    the two searches are pinned to the same cap by this script rather than by
#    the passes' shared default.
roundtrip deterministic "$SRC/test/Conversion/MicroMapping/micro_map.mlir" \
    "$X86 mode=deterministic top-k=8"

# 2. Beam mode. The probe target's rules force beam to prefer a different plan
#    than algorithm order would, so a binder that ignored the mode would bind a
#    different plan and the diff would catch it.
PROBE="target=probe machine=$SRC/test/Conversion/MicroMapping/probe_machine.yaml layouts=$SRC/mapping/x86-avx2/layouts.llkmap rules=$SRC/test/Conversion/MicroMapping/bind_plan_mode_rules.llkmap emitters=e1"
roundtrip beam "$SRC/test/Conversion/MicroMapping/micro_map.mlir" \
    "$PROBE mode=beam top-k=8"

# 2b. Beam mode with a non-default beam-width. The fixture's first level holds
#     two instances, so width 1 truncates the beam and records the truncation
#     in the plan's content hash; the id then differs from the default width 64.
#     Binding it only succeeds if `--micro-bind-plan` replays the same width.
roundtrip beam-narrow "$SRC/test/Conversion/MicroMapping/micro_map.mlir" \
    "$PROBE mode=beam beam-width=1 top-k=8"

# 3. The full chain: lower with the compiler, then map and bind the lowered IR
#    in separate invocations. The lowered module is written to a file first, so
#    both the mapping and the binding run over the exact same bytes -- the
#    assertion is about the passes, never about re-running `--llk-to-micro`.
"$LLK_OPT" --llk-to-micro="schedule-db=missing.json" \
    "$SRC/test/Conversion/MicroMapping/matmul_e2e.mlir" \
    > "$WORK/lowered.mlir" 2>/dev/null
roundtrip lowered "$WORK/lowered.mlir" "$X86 mode=exact top-k=8"

echo "micro_bind_plan_roundtrip: ok"
