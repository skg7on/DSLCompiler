#!/bin/sh
#===- micro_map_candidate.sh - `--micro-map candidate=` (phase-4 T4) ----===//
#
# `--micro-map` searches a kernel and binds the best plan. A `micro.candidate`
# is the persistent form of a search-space point, so `candidate=<sym>` binds the
# search to that point: its values pin the rule parameters of the same name, and
# its *layout-kind* parameter (resolved by `kind`, never by name) becomes the
# bound layout every rule must offer. This script pins the wiring end to end.
#
# The cases:
#
#   1. the bound plan records the candidate's hash -- the IR's `binding_hash`
#      equals the report's `sourceBindingHash` (the binding's `stableHash`) and
#      the report names the candidate (`sourceBindingCandidate`) with its values
#      (`sourceBinding`), so the plan is traceable to the `micro.candidate` it
#      came from and replayable from the report alone;
#   2. the layout parameter is named `block_shape`, not `layout`/`tile_layout`,
#      and the feature still works -- the pass reads the parameter's `kind`;
#   3. `candidate=` absent is byte-for-byte today's behaviour: a plan binds with
#      `binding_hash = 0` and no `sourceBinding`;
#   4. a candidate no rule can satisfy is a *search* failure carrying the
#      frontier's diagnostics, never a silently wrong plan: a pinned parameter
#      that contradicts a `require` (`VW=4` vs `VW == lanes(f32) == 8`) and a
#      layout value that is a legal micro kind but not the id the rule declares
#      (`row_major` vs `blocked`) both end in `no_matching_rule`;
#   5. an unknown candidate symbol is a load error naming it;
#   6. the kind-to-layout bridge: the *shipped* AVX2 rules declare
#      `avx2.blocked_2d`, and that layout declares `implements blocked`, so the
#      bound Micro kind `blocked` resolves to that target id and the same
#      fixture maps under the shipped target. Before the bridge the two
#      namespaces never compared equal, so a binding could only veto rules --
#      never select a layout.
#   7. the report -> bind round-trip survives a binding (ruling S8): the id the
#      `--micro-map candidate=` run reports is reproduced by `--micro-bind-plan`
#      with the *same* `candidate=`, byte-identically -- and is absent without
#      it, because the id folds in the binding's hash.
#
# Usage: micro_map_candidate.sh <llk-opt> <source-dir> <work-dir>
#===- -------------------------------------------------------------------===//

set -eu

LLK_OPT=$1
SRC=$2
WORK=$3

mkdir -p "$WORK"

FIXTURE="$SRC/test/Conversion/MicroMapping/micro_map_candidate.mlir"
# The fixture target: its layout id `blocked` is the spelling the search space
# binds (see binding_layouts.llkmap).
PROBE="target=probe machine=$SRC/machines/x86-avx2-v2.yaml layouts=$SRC/test/Conversion/MicroMapping/binding_layouts.llkmap rules=$SRC/test/Conversion/MicroMapping/binding_rules.llkmap emitters=binding_vector_add,binding_copy"
# The shipped AVX2 target, whose layout ids are namespaced (`avx2.blocked_2d`).
AVX2="target=x86-avx2 machine=$SRC/machines/x86-avx2-v2.yaml layouts=$SRC/mapping/x86-avx2/layouts.llkmap rules=$SRC/mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store,avx2_fused_convert_silu_mul"

fail() {
  echo "micro_map_candidate.sh: $1" >&2
  exit 1
}

# Runs a mapping that must fail, and requires the diagnostic to contain `needle`.
expect_failure() {
  options=$1
  needle=$2
  out=$3
  if "$LLK_OPT" "--micro-map=$options" "$FIXTURE" > "$out" 2>&1; then
    fail "expected the mapping to fail, but it succeeded: $options"
  fi
  grep -q "$needle" "$out" ||
    fail "the failure did not carry '$needle': see $out"
}

# --- 1 & 2. A satisfiable candidate binds, and the plan records its hash. -----
# `report=` is used only to read the binding's `stableHash` back; the IR is the
# thing under test.
"$LLK_OPT" "--micro-map=$PROBE mode=deterministic candidate=candidate_17 report=$WORK/bound.json" \
    "$FIXTURE" > "$WORK/bound.mlir"

grep -q 'micro\.plan' "$WORK/bound.mlir" || fail "candidate_17 bound no plan"
grep -q 'micro\.mapping' "$WORK/bound.mlir" || fail "candidate_17 placed no node"

# The binding is recorded in the IR as the candidate's stable hash. The report
# prints the hash as 16 hex digits; the IR prints the same 64-bit value as a
# signed i64. Shell arithmetic is only portable below 2^63, so a hash with its
# top bit set is converted by two's complement on the *text* rather than by
# `$((0x...))`, which saturates to INT64_MAX under some shells (dash) instead of
# wrapping to the signed value. The complement's top nibble is below 8, so the
# arithmetic below is exact everywhere.
HEX=$(sed -n 's/.*"sourceBindingHash": "\([0-9a-f][0-9a-f]*\)".*/\1/p' \
        "$WORK/bound.json" | head -n 1)
[ -n "$HEX" ] || fail "the report carried no sourceBindingHash"
[ ${#HEX} -eq 16 ] || fail "sourceBindingHash is not 16 hex digits: '$HEX'"
top=$(printf '%s' "$HEX" | cut -c1)
case $top in
  8|9|a|b|c|d|e|f|A|B|C|D|E|F)
    # signed(V) == -(2^64 - V) == -1 - not(V); not(V) fits in a signed 64-bit.
    NOT=$(printf '%s' "$HEX" |
            tr '0123456789abcdefABCDEF' 'fedcba9876543210FEDCBA9876543210')
    HASH=$(( -1 - 0x$NOT ))
    ;;
  *)
    HASH=$((0x$HEX))
    ;;
esac
[ "$HASH" -ne 0 ] || fail "candidate_17's stableHash is zero"
grep -q "binding_hash = $HASH : i64" "$WORK/bound.mlir" ||
  fail "the bound IR does not record the candidate's stableHash ($HASH)"

# ...and the values are the candidate's -- the report echoes the binding it
# searched at. This does *not* by itself show the layout axis was consumed
# (`sourceBinding` is just the binding's values, whatever the rules do with
# them); that the axis is live is shown by the outcome: the same candidate with
# `block_shape = "row_major"` (case 4) is a non-match, so the value decides.
grep -q '"sourceBinding": "VW=i:8;block_shape=s:blocked"' "$WORK/bound.json" ||
  fail "the report's sourceBinding is not candidate_17's values"

# ...and it names the candidate itself, so a consumer holding only the report
# can replay the search at the same `candidate=` (ruling S8).
grep -q '"sourceBindingCandidate": "candidate_17"' "$WORK/bound.json" ||
  fail "the report did not name the source candidate"

# --- 3. No candidate is today's behaviour. -----------------------------------
"$LLK_OPT" "--micro-map=$PROBE mode=deterministic report=$WORK/unbound.json" \
    "$FIXTURE" > "$WORK/unbound.mlir"
grep -q 'micro\.plan' "$WORK/unbound.mlir" ||
  fail "a binding-free mapping bound no plan"
grep -q 'binding_hash = 0 : i64' "$WORK/unbound.mlir" ||
  fail "a binding-free mapping recorded a non-zero binding_hash"
grep -q '"sourceBinding": ""' "$WORK/unbound.json" ||
  fail "a binding-free search recorded a sourceBinding"
grep -q '"sourceBindingCandidate": ""' "$WORK/unbound.json" ||
  fail "a binding-free search recorded a source candidate"

# --- 4. A candidate no rule can satisfy is a search failure. -----------------
# A pinned parameter that contradicts a `require`: VW=4, but the machine's f32
# vector width is 8, so `VW == lanes(element_type)` is unsatisfiable.
expect_failure "$PROBE mode=deterministic candidate=candidate_pinned_low" \
    'the search produced no complete plan' "$WORK/pinned.out"
grep -q 'no_matching_rule' "$WORK/pinned.out" ||
  fail "the pinned rejection did not report no_matching_rule"

# A layout value that is a legal micro layout kind but not the id the rule
# declares: the rule is a non-match, never a silently different layout.
expect_failure "$PROBE mode=deterministic candidate=candidate_foreign_layout" \
    'the search produced no complete plan' "$WORK/foreign.out"
grep -q 'no_matching_rule' "$WORK/foreign.out" ||
  fail "the layout rejection did not report no_matching_rule"

# --- 5. An unknown candidate symbol is named in the error. -------------------
expect_failure "$PROBE mode=deterministic candidate=does_not_exist" \
    "no micro.candidate named 'does_not_exist'" "$WORK/unknown.out"

# --- 6. The kind-to-layout bridge: a bound kind selects a shipped layout. -----
# The same satisfiable candidate, mapped against the shipped AVX2 target whose
# `avx2.blocked_2d` declares `implements blocked`. The bound Micro kind
# `blocked` resolves to that target id, so the rule's `require layout operand0
# satisfies avx2.blocked_2d` is satisfied and the node maps.
if ! "$LLK_OPT" "--micro-map=$AVX2 mode=deterministic candidate=candidate_17" \
      "$FIXTURE" > "$WORK/shipped.mlir" 2> "$WORK/shipped.err"; then
  fail "the shipped AVX2 target did not map under the bound kind: $(cat "$WORK/shipped.err")"
fi
grep -q 'avx2.blocked_2d' "$WORK/shipped.mlir" ||
  fail "the shipped mapping did not record the selected avx2.blocked_2d layout"

# --- 7. The report -> bind round-trip holds under a binding (ruling S8). -----
# A plan id folds in the binding's hash, so the id `--micro-map
# candidate=candidate_17` reported is reproducible only by a `--micro-bind-plan`
# given the *same* `candidate=`. Without it the id is genuinely absent from the
# search -- which is what makes the option load-bearing rather than decorative.
ID=$(sed -n 's/.*"selectedPlanId": "\([0-9a-f][0-9a-f]*\)".*/\1/p' \
        "$WORK/bound.json" | head -n 1)
[ -n "$ID" ] || fail "the report carried no selectedPlanId"

if ! "$LLK_OPT" "--micro-bind-plan=plan-id=$ID $PROBE mode=deterministic candidate=candidate_17" \
      "$FIXTURE" > "$WORK/roundtrip.mlir" 2> "$WORK/roundtrip.err"; then
  fail "the bound id was not reproducible with the same candidate=: $(cat "$WORK/roundtrip.err")"
fi
# Reproducing the id reproduces the *plan*: the replayed IR is byte-identical
# to the `--micro-map candidate=` output.
diff "$WORK/bound.mlir" "$WORK/roundtrip.mlir" ||
  fail "the replayed plan is not the plan the report named"

if "$LLK_OPT" "--micro-bind-plan=plan-id=$ID $PROBE mode=deterministic" \
      "$FIXTURE" > "$WORK/roundtrip_nocand.out" 2>&1; then
  fail "a binding-derived id was found by a binding-free bind-plan search"
fi
grep -q 'no plan in the deterministic search has id' "$WORK/roundtrip_nocand.out" ||
  fail "the binding-free replay did not report the missing id"
