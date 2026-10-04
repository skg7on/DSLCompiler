#!/bin/sh
#===- micro_bind_plan_id.sh - plan-id spellings (design §21) ------------===//
#
# Checks that `--micro-bind-plan` accepts the id that `--micro-map report=`
# actually prints, closing the documented report -> bind workflow. A plan id is
# an unsigned 64-bit content hash; the report spells it as bare lowercase hex,
# so that is the canonical form the binder must accept verbatim.
#
# Accepted spellings, all denoting the same 64-bit hash:
#   1. bare lowercase hex, 16 digits -- exactly what the report prints;
#   2. `0x`/`0X`-prefixed hex;
#   3. unsigned decimal;
#   4. signed decimal (the same hash read as a signed i64).
# Anything else is a usage error carrying the "must be a 64-bit" diagnostic.
#
# Spelling rule, in order: `0x`/`0X` forces hex; else exactly 16 hex digits is
# hex (the report's `%016llx` width, so its token round-trips even when all 16
# digits are decimal); else all-hex-digits containing a hex letter (a-f) is hex;
# else decimal. So `12345` is decimal and `12ab` is hex, but the 16-digit
# `1234567890123456` is hex. This script pins the rule and the precedence by
# asserting the exact id the binder echoes back.
#
# Usage: micro_bind_plan_id.sh <llk-opt> <source-dir> <work-dir>
#===- -------------------------------------------------------------------===//

set -eu

LLK_OPT=$1
SRC=$2
WORK=$3

mkdir -p "$WORK"

# The five required target keys. `--micro-map` runs in deterministic mode so the
# printed id is reproducible across runs (beam/exact reproducibility is a
# separate task). The binder forces deterministic mode internally, so the id
# round-trips; it takes no `mode=` option of its own.
TARGET="target=x86-avx2 machine=$SRC/machines/x86-avx2-v2.yaml layouts=$SRC/mapping/x86-avx2/layouts.llkmap rules=$SRC/mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_mma,avx2_reduce,avx2_copy"
MAP_OPTIONS="$TARGET mode=deterministic"
BIND="$TARGET top-k=8"
KERNEL="$SRC/test/Conversion/MicroMapping/micro_map.mlir"

# A successful bind stamps `micro.plan` onto the kernel; a parse failure does
# not get that far, so grepping for the marker proves the id was accepted.
probe_accepts() {
  spelling=$1
  out=$2
  "$LLK_OPT" "--micro-bind-plan=plan-id=$spelling $BIND" "$KERNEL" > "$out"
  grep -q 'micro\.plan' "$out"
}

# A rejected id must fail and carry the diagnostic naming the accepted forms.
probe_rejects() {
  spelling=$1
  out=$2
  if "$LLK_OPT" "--micro-bind-plan=plan-id=$spelling $BIND" "$KERNEL" \
        > "$out" 2>&1; then
    echo "expected 'plan-id=$spelling' to be rejected" >&2
    exit 1
  fi
  grep -q 'must be a 64-bit' "$out"
}

# A well-formed id that names no plan still parses: the diagnostic echoes the
# id back as unsigned decimal, which pins how the spelling was interpreted.
probe_interprets_as() {
  spelling=$1
  expected=$2
  out=$3
  if "$LLK_OPT" "--micro-bind-plan=plan-id=$spelling $BIND" "$KERNEL" \
        > "$out" 2>&1; then
    echo "expected 'plan-id=$spelling' to name no plan" >&2
    exit 1
  fi
  grep -q "no plan in the deterministic search has id $expected" "$out"
}

# 1. Obtain the id the report actually prints.
"$LLK_OPT" "--micro-map=$MAP_OPTIONS report=$WORK/report.json" "$KERNEL" \
    > "$WORK/mapped.mlir"
HEX=$(sed -n 's/.*"selectedPlanId": "\([0-9a-f][0-9a-f]*\)".*/\1/p' \
        "$WORK/report.json")
if [ -z "$HEX" ]; then
  echo "report did not carry a hex selectedPlanId" >&2
  exit 1
fi

# 2. The report's own spelling binds -- the core of the fix.
probe_accepts "$HEX" "$WORK/hex.mlir"

# 3. `0x`-prefixed hex binds too.
probe_accepts "0x$HEX" "$WORK/hex0x.mlir"

# 4. Decimal binds. `$((0x...))` is signed 64-bit arithmetic, so it is only
#    well-defined here while the id's top hex digit is below 8; the report's
#    deterministic id is `0081...`, so this runs. Guarded so a future high-bit id
#    skips this probe rather than leaning on implementation-defined overflow; the
#    high-bit cases below still cover decimal parsing, signed and unsigned.
top=$(printf '%s' "$HEX" | cut -c1)
case $top in
  8|9|a|b|c|d|e|f|A|B|C|D|E|F) : ;;
  *)
    DEC=$((0x$HEX))
    probe_accepts "$DEC" "$WORK/decimal.mlir"
    ;;
esac

# 5. Signed and unsigned decimals of a high-bit hash denote the same id: both
#    are accepted by the parser (they reach the "no plan" diagnostic, not the
#    parse error) and both resolve to the same unsigned 64-bit value.
probe_interprets_as -1 18446744073709551615 "$WORK/signed.mlir"
probe_interprets_as 18446744073709551615 18446744073709551615 \
    "$WORK/unsigned.mlir"

# 6. Ambiguity rule: a digit-only string is decimal (`12345`, not `0x12345`),
#    while a hex letter forces hex (`12ab` == 4779).
probe_interprets_as 12345 12345 "$WORK/decimal_only.mlir"
probe_interprets_as 12ab 4779 "$WORK/hex_letter.mlir"

# 6b. ...but the report always prints exactly 16 hex digits, so at that exact
#     width hex wins even when every digit is decimal: 0x1234567890123456 must be
#     looked up as 1311768467284833366, never as decimal 1234567890123456.
probe_interprets_as 1234567890123456 1311768467284833366 \
    "$WORK/hex_digits_only.mlir"

# 7. Anything else is rejected with the diagnostic.
probe_rejects "0xbeefg" "$WORK/bad_hex.mlir"
probe_rejects "not-a-number" "$WORK/bad_text.mlir"
