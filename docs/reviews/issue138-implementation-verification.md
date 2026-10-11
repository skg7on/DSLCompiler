# Issue #138 implementation verification

Baseline: `fb99595`; isolated branch `refactor/kernel-lowering-framework`.
Approved plan: `docs/superpowers/plans/2026-10-11-issue138-kernel-lowering-framework.md`.

## Execution ledger

- T1: missing header RED; registry lookup/order/default capabilities/pointer stability/duplicate policy GREEN. Commit `e5ff3ed`.
- T2: missing contract helper link symbols RED; bucket boundaries, dimensions, accumulator attributes, dynamic description and strict export validation GREEN. Commit `15c911e`.
- T3: unregistered SwiGLU RED; registry capability and existing Linalg tests GREEN. Commit `bfa1291`.
- T4: unregistered RoPE/Attention RED; sink flags and existing Linalg tests GREEN. Commit `e74d308`.
- T5/T6: absent Micro capabilities RED; registered SwiGLU, Matmul and Linalg compatibility adapter GREEN. Driver/search-space/three root collectors migrated together in `af66317`.
- T7: missing Matmul shape remark RED; existing transform tests and new Matmul/Attention/dynamic SwiGLU remarks GREEN. Commit `917d00a`; RoPE coverage added in final audit.
- T8: missing linalg.matmul FileCheck RED; source A/B/init operands and result preserved GREEN. Commit `810a13f`; explicit nonzero f32 init coverage added in final audit.
- T9: explicit Linalg accumulator override test RED; adapter now preserves an explicit TypeAttr or falls back to result element type, matching baseline. Seven framework GTests GREEN.

## Rulings

- Ruling: T5 and T6 share one migration commit because concrete, search-space and schedule-replay root collectors must share the same supported set. Cost if wrong: less granular history; both export modes have independent regression coverage.
- Ruling: full suite is verified at integration boundaries rather than after every commit. The first check-llk exposed missing standalone test dependencies; T9 repairs its build closure. Cost if wrong: a regression is localized to a wider commit range.
- Ruling: retain the forced AVX2 hardware acceptance test unchanged on ARM64 and report its expected architecture failure; use pinned Linux CI for hardware/static-link acceptance. Cost if wrong: CI remains the required execution gate.

## Local evidence

Host: aarch64 macOS; local LLVM/MLIR `24.0.0git`, Debug, static library artifacts.
Compilation uses `-fno-rtti -fno-exceptions -Werror=deprecated-declarations`.

- Seven framework GTests pass.
- Fourteen focused conversion/transform tests pass.
- Thirteen Micro/mapped export and invocation tests pass or report their existing host skip.
- Eight complete pre-refactor Micro/search-space outputs compare byte-identical after migration.
- Scoped operator-type dispatch and deprecated-API audits return no matches.
- Full check-llk: 1,243 registered, one disabled, six skipped, one failed (`MappedAVX2Acceptance`: selected target requires x86_64, host is aarch64), 1,235 passed.
- Initial nine full-suite failures included missing standalone executables; explicit build and T9 check-llk dependencies resolve those. The remaining failure is solely the forced hardware gate.

Independent review and pinned Linux CI results will be recorded below.

## Independent review

A separate reviewer inspected the complete implementation and found one Important
issue: the new Matmul Linalg hook accumulated in the output dtype rather than the
declared accumulator dtype. No other Critical/Important issues or deferred minors
were reported.

Final: fixed accumulator precision —
`KernelContractTest.MatmulF32AccumulationPreservesNonzeroBF16Init` RED→GREEN.
The test executes the emitted scalar regions through actual MLIR constant folding:
bf16 init 256 plus two unit products now yields 258 with f32 accumulation, versus
256 before the fix. FileCheck also covers dynamic init dimensions and equal-width
f16/bf16 conversions through f32. This is portable scalar-IR numerical validation,
not native JIT execution.

Final: Ruling: honor accumulator precision rather than the plan's direct original-init
Linalg construction — source numerical semantics require it. Cost if wrong: extra
conversion operations; the nonzero-init regression pins their numerical purpose.

Reviewer declined to judge Linux static-link CI and forced AVX2 execution on ARM64;
those remain assigned to pinned Linux CI, with local failure reported above.

## Post-review local verification

All build targets compile after the precision fix. Full check-llk now registers
1,245 tests: 1,237 passed, six skipped, one disabled, one failed solely in forced
AVX2 execution on ARM64. The eight complete Micro baseline outputs still match
byte-for-byte. Scoped deprecated API and operator-type dispatch audits are clean.
Eight framework GTests and the expanded Matmul FileCheck pass.
