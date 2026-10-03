# AVX2 Mapping Target and Second-Target Conformance (D7 / epic #67)

**Goal:** Package the AVX2 mapping configuration behind the `MappingTarget` interface, prove a second, unrelated target needs no change to generic code, and show both map a bounded fixture end to end through D6.

**Spec:** design §14.3 (target bundles), §19 (plugin boundary); plan §4 D7.

## Structure

Two target packages, one library:

| File | Responsibility |
|---|---|
| `include/LLK/Target/X86/Mapping/AVX2MappingTarget.h` + `lib/.../AVX2MappingTarget.cpp` | AVX2 configuration paths and emitter keys |
| `include/LLK/Target/GenericAccelerator/Mapping/GenericAcceleratorMappingTarget.h` + `lib/.../*.cpp` | the second target, same interface |
| `mapping/generic-ai-accel/layouts.llkmap`, `rules.llkmap` | the accelerator's declarative configuration |
| `test/Mapping/avx2_target.cpp`, `test/Mapping/generic_accelerator_target.cpp` | tests |

The packages live in a new `LLKTargetMapping` library rather than inside `LLKTargetX86`: that library is CPU feature detection, and folding MLIR-dependent mapping code into it would drag MLIR into `CpuFeaturesTest`.

Emitter keys are declared **in code**, not in `rules.llkmap`: they name C++ emitters the plugin provides, so the plugin is their only owner. `verifyMappingTarget` already rejects a rule naming a key the target does not declare.

## Design decisions

1. **The only target-specific names live in the target package.** `machines/x86-avx2-v2.yaml` and `mapping/x86-avx2/*` were already target-owned; this adds the C++ that names them. Nothing was added to `LLK/Mapping`, `LLK/Machine`, or Micro ODS.
2. **The second target is a *different shape*, not a renamed copy.** The accelerator asks for a `pe` executor where AVX2 asks for `worker`; both are satisfied because each machine declares its own owner refinement. It has different compute kinds, a three-dimensional layout, and its own emitters.
3. **The conformance test asserts a negative.** It reads the three generic Micro ODS files and fails if any target word appears — so a leak is caught by CI rather than by review.

## Verification Results (2026-10-03)

- **Build:** `ninja -C build` — clean.
- **New tests:** `MappingAvx2TargetTest` **4/4**, `MappingGenericAcceleratorTargetTest` **4/4**.
- **Full suite:** `ctest --test-dir build --output-on-failure` — 102 registered, **100 passed, 2 skipped, 0 failed**.
- **Deprecated-API audit:** clean.
- **Legacy path:** every pre-existing test still passes unchanged, including the `LLKPerf` and JIT suites that exercise the original AVX2 compilation path. Mapping code is additive; nothing routes through it yet.

### Note on the conformance test

The first version matched substrings and reported `MicroOps.td` as mentioning `npu` — a false positive, since the match was the `npu` inside `$inputs`. The test now matches whole words (`npu_engine` would still be caught, `inputs` is not). Worth recording because the assertion is only as good as its matching rule.

### What D7 does *not* do

It does not emit code. A selected `CoveringPlan` is still not bound to concrete Micro-IR: that is the `#50` revision, which consumes `CoveringPlan` plus the bundle/emitter pair this package declares. `#53` then documents the whole workflow.
