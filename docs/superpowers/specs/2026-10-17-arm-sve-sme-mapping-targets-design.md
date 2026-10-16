# ARM SVE and SME Mapping Targets — Design

**Date:** 2026-10-17

**Status:** Design for review. Delivers two static-conformance mapping targets; claims no executable backend.

**Scope:** Machine profiles, LLKMap policy, target packages, registration and acceptance for `arm-sve` and `arm-sme`.

**Related:** [framework goal and design spec](2026-10-12-generic-microir-dnn-compiler-framework-design.md) §4.3, [LLKMap layout grammar](../../design/llkmap-layout-grammar.md), [LLKMap rule grammar](../../design/llkmap-rule-grammar.md), [ARCHITECTURE.md](../../../ARCHITECTURE.md) §5.

## 1. Summary

Add two mapping targets that model ARM SVE and ARM SME, in the shape the framework already prescribes for a new target: a machine profile, LLKMap layout and rule policy, and a thin target package that loads them. They participate in extraction, matching, placement, connectivity, covering, binding and reporting. Their emitters verify well-formed bundles and refuse to lower, which is the static-conformance boundary the generic accelerator already occupies.

The work changes no shared interface. `lib/Mapping` and `lib/Machine` gain no target-name branch, the `micro` dialect is not extended, and `LLKMachine`/`LLKMapping` acquire no new dependency. Everything target-specific is data plus a per-target loader.

`arm-sve` is the first target on which a contraction is placed somewhere other than a matrix engine, and `arm-sme` places the same contraction on a ZA matrix engine. The pair exists to make that placement difference a real, rankable outcome rather than a relabelled one.

## 2. Current State

This section records what was true before this design, because the request that prompted it assumed otherwise.

There is no ARM mapping target. `mapping/` contains only `generic-ai-accel` and `x86-avx2`; `machines/` contains only `generic-ai-accel-v2.yaml` and `x86-avx2-v2.yaml`; `lib/Target/MappingTargets.cpp` registers only those two names. Nothing in the mapping, search or binding path consults an ARM profile.

What does exist for AArch64 is unrelated to mapping:

- `include/LLK/Target/AArch64/TargetSVE.h` and `lib/Target/AArch64/TargetSVE.cpp` implement **CPU feature detection** — a `CpuFeatures` struct probing NEON/SVE/SVE2/BF16/I8MM. Nothing includes the header; the library it compiles into is linked by no target.
- `include/LLK/Runtime/KernelKey.h` carries `CpuIsa::NEON` and `CpuIsa::SVE`, consumed only by the legacy kernel-cache path (`runtime/JitCache.cpp`), never by `MappingTarget`.
- `include/LLK/Transforms/Passes.td` advertises a `target-isa` option with values `"avx2, avx512, sve"`. The option is never read; `lib/Transforms/TileAndVectorize.cpp` hardcodes AVX2 tile dimensions.
- `include/LLK/Transforms/Common/VectorizationUtils.h` defines `SVE_F32_MIN`, which nothing references.
- `schedules/aarch64_sve/` exists and is empty.

SME appears nowhere in the repository — not in `docs/`, not in `include/`, not in `lib/`.

The documents mention ARM exactly once, in a file explicitly marked historical: `docs/design/legacy-cpu-architecture.md` records the target profile as "extensible to … AArch64+NEON, AArch64+SVE". Nothing in the current framework contract commits to SVE or SME, so this design adopts rather than replaces a position.

Two properties of the current framework decide what is feasible:

- **A modeled target is cheap.** `MappingTarget` resolves widths *from the machine* — a layout may require `VW == machine.compute("vector_engine").lanes(element_type)` — and target engines plug in as data through `refines` over the abstract Micro `Owner` vocabulary (`group`, `worker`, `vector`, `matrix`, `transfer`). A new target therefore needs no generic code.
- **An executable ARM target is not reachable here.** Three independent blockers: `micro::LayoutAttr`'s width is a concrete `int64_t` with no scalable marker, so SVE's runtime vector length cannot be represented; `micro.mma` is defined only as a tile-shaped `acc += lhs x rhs`, with no ZA accumulator array, streaming mode or outer-product concept; and no host available to this project (Apple Silicon has NEON only; CI is x86) could produce the execution evidence the framework requires. This design accordingly stops at modeling.

## 3. Goals and Non-Goals

**Goals.**

1. Add `arm-sve` and `arm-sme` as registered mapping targets with their own machine profiles and LLKMap policy.
2. Exercise the full mapping path on ARM: a workload extracts, matches, places, connects, covers and binds against each profile, and reports a plan.
3. Make contraction placement a genuine target difference: vector engine on `arm-sve`, ZA matrix engine on `arm-sme`.
4. Extend framework coverage without extending framework interfaces.
5. Pin the static-conformance boundary with a test that fails if either target ever claims to lower.

**Non-Goals.**

1. No executable ARM backend, no lowering emitters, no `codegenRequirements`, no JIT.
2. No `micro` dialect change — no scalable `LayoutAttr`, no ZA or outer-product operation, no `scalableVecDims` work.
3. No machine-schema change. The profiles describe what the current schema can express.
4. No cleanup of the pre-existing SVE hooks. §2 lists them; repairing them is separate work with its own justification. In particular `lib/Target/AArch64/TargetSVE.cpp:59-76` nests a function definition inside an `#if defined(__linux__)` branch and cannot compile on aarch64 Linux. That is a live latent breakage and should have its own issue, not a rider on this one.
5. No new entry in the `llk-tune` default-machine selector. Callers pass `--machine` explicitly.

## 4. Normative Decisions

| Decision | Choice | Consequence |
|---|---|---|
| Deliverable | Static-conformance profiles | Emitters verify and refuse to lower; no backend is claimed |
| Packaging | Two targets, `arm-sve` and `arm-sme` | Two packages, two profiles, two policy pairs, two registrations |
| Vector length | One modeled length per profile, **512 bits** | `arm-sve` lanes f32/bf16/i8 = 16/32/64 |
| Coverage | Vector rules for SVE, matrix rules for SME | SVE covers the vector half of the AVX2 rule set; SME adds ZA matrix rules |
| Contraction | Vector engine on SVE, matrix engine on SME | `micro.mma` binds on both; placement differs |

SVL = 512 bits is chosen because it is a legal SME length (SME permits 128, 256 or 512), it yields the canonical documented ZA geometry, and its lane counts are clearly distinct from the AVX2 profile's 8/16/32, so a ranking difference between the two cannot be an artefact of identical widths.

## 5. Machine Profiles

Both profiles model an ARMv9-A core with SVE2. The scheme is the schema's: a target invents its own labels and each declares the abstract class it refines.

`machines/arm-sve-v2.yaml` — executors `cluster.0` (`cluster`, refines `group`) → `core.0` (`core`, refines `worker`) → `svp.0` (`svp`, refines `vector`). Memory chain `dram.0` → `l2.0` → `l1d.0` (`sram`) → `rf.0`, each with capacity, alignment, bandwidth, latency and banks; a `dma.0` transfer engine refining `transfer`; `links` carrying the cost of each hop. One compute node:

```yaml
compute:
  - id: svp
    kind: vector_engine
    refines: [vector]
    attached_to: core.0
    element_types: [f32, bf16, i8]
    supported_layouts: [row_major, vectorized]
    shapes: [[16], [32], [64]]
    lanes: {f32: 16, bf16: 32, i8: 64}
```

`machines/arm-sme-v2.yaml` — the same profile plus a `za_unit.0` executor (`za_unit`, refines `matrix`, parent `core.0`), so an abstract owner of matrix kind resolves as it does in the AVX2 profile, and:

```yaml
memories:
  - id: za.0
    kind: acc
    visible_from: core.0
    capacity_bytes: 4096      # SVL/8 x SVL/8 bytes = 64 x 64 at SVL=512
    alignment_bytes: 64
    banks: 4                  # ZA0..ZA3
compute:
  - id: za_engine
    kind: matrix_engine
    refines: [matrix]
    attached_to: core.0
    shapes: [[16, 16, 16]]
    element_types: [f32, bf16]
    accumulator_dtypes: [f32]
```

`acc` is already a member of the `MemorySpace` enum, so ZA storage needs no new vocabulary. The 4096 bytes is arithmetic, not estimation: per the Arm Architecture Reference Manual (B1.4.8), ZA is an `(SVL/8) x (SVL/8)` byte array — 64 x 64 = 4096 bytes at SVL=512. For 32-bit elements the non-widening `FMOPA` produces an `SVL_S x SVL_S` sum of outer products where `SVL_S = SVL/32`, so ZA.S is four `(SVL/32) x (SVL/32)` = 16x16 f32 quadrants, ZA0.S–ZA3.S, which together form the 32x32 f32 accumulator the SME micro-kernel literature configures. The profile models one quadrant per matrix-engine issue and reports `banks: 4` for the four quadrants.

Both profiles carry a header comment stating that the numbers are modeled seeds for a class of ARMv9 cores, not measurements of any part, matching the existing profiles' convention.

## 6. Target Layouts

`mapping/arm-sve/layouts.llkmap` declares `sve.row_major` (implementing `row_major`) and `sve.blocked_2d`, whose width follows the machine rather than a constant:

```
layout sve.blocked_2d(int M, int N, int VW) {
  param M in [4..16];
  param N in [4..16];
  param VW in [4..16];
  require rank == 2;
  require VW == machine.compute("vector_engine").lanes(element_type);
  require N % VW == 0;
  implements blocked;
  map (m, n) -> (m, floordiv(n, VW), mod(n, VW));
}
```

The domain widens to `[4..16]` because the modeled machine has 16 f32 lanes; AVX2's `[4..8]` reflects its 8. This is the mechanism that lets a scalable-width ISA be modeled without a scalable-width type: the profile pins the length, the layout reads it back.

`mapping/arm-sme/layouts.llkmap` repeats those two and adds a ZA-aware layout implementing `blocked` for a tile resident in `acc`.

## 7. Target Rules

`mapping/arm-sve/rules.llkmap` mirrors the vector half of `mapping/x86-avx2/rules.llkmap`: `micro.vector` variants `add`, `mul`, `convert`, `silu`; `micro.reduce` `sum`; `micro.async_copy` (naming its DRAM source and SRAM destination, as the existing copy rules must, because its boundary values are plain tensors that state no memory kind); `micro.tile_async_copy`; `micro.tile_store`; and the fused `convert -> silu -> mul` graph rule.

Contraction binds on the vector engine:

```
rule sve.mma_f32 v1 {
  match micro.mma(input[0].element_type = f32);
  param VW in [4..16];
  require VW == machine.compute("vector_engine").lanes("f32");
  require executor kind worker;
  require compute kind vector_engine;
  require layout lhs satisfies sve.row_major;
  input "lhs";
  input "rhs";
  input "acc";
  output "result";
  bundle "sve.mma.f32" { VW = VW };
  emit "sve_mma";
  cost 16;
}
```

This models SVE performing the contraction with vector `FMLA`/`BFDOT` on the vector unit. It is a real implementation strategy for SVE, not a placeholder, and it is the reason `arm-sve` can be exercised end to end on the project's primary workload.

`mapping/arm-sme/rules.llkmap` carries the same vector rules and replaces the contraction rule with one binding to the matrix engine and the ZA accumulator:

```
rule sme.mma_f32 v1 {
  match micro.mma(input[0].element_type = f32);
  require executor kind worker;
  require compute kind matrix_engine;
  require memory output "result" kind acc;
  input "lhs";
  input "rhs";
  input "acc";
  output "result";
  bundle "sme.mma.f32";
  emit "sme_mma_za";
  cost 8;
}
```

The two targets now map the same matmul at different costs on different engines. Emitter keys follow the established convention: `sve_vector_add`, `sve_vector_convert`, `sve_vector_silu`, `sve_vector_mul`, `sve_mma`, `sve_reduce`, `sve_copy`, `sve_tile_copy`, `sve_tile_store`, `sve_fused_convert_silu_mul`, and the corresponding `sme_*` set including `sme_mma_za`.

## 8. Packaging and Registration

Each target follows `GenericAcceleratorMappingTarget` exactly: a header declaring `emitterKeys()` and `createMappingTarget()`, and a source whose `createMappingTarget()` names its profile and policy paths and returns the `FileMappingTarget` from `loadMappingTarget` unchanged. Neither subclasses `MappingTarget`, so `codegenRequirements()` returns `nullopt` and `buildSelectedBackendPipeline()` takes the base rejection — the same construction that makes the generic accelerator static-only.

```
machines/arm-sve-v2.yaml
machines/arm-sme-v2.yaml
mapping/arm-sve/{layouts,rules}.llkmap
mapping/arm-sme/{layouts,rules}.llkmap
include/LLK/Target/AArch64/Mapping/{SVEMappingTarget.h,SMEMappingTarget.h}
lib/Target/AArch64/Mapping/{SVEMappingTarget.cpp,SMEMappingTarget.cpp}
lib/Target/CMakeLists.txt        # two sources added to LLKTargetMapping
lib/Target/MappingTargets.cpp    # two registerMappingTarget calls
test/Mapping/{arm_sve_target.cpp,arm_sme_target.cpp}
test/Mapping/CMakeLists.txt      # two sources added
test/Machine/machine_model_loader.cpp   # loader cases for both profiles
```

Both dispatch paths are covered. The registry path (`llk-compile`/`llk-tune` `--mapping-target arm-sve`) uses the two registrations. The pass-spec path (`--micro-map target=arm-sve machine=... layouts=... rules=... emitters=...`) consults no registry and needs only the policy files, which is how the FileCheck and shell tests drive it.

## 9. Acceptance

Testing is TDD: each target's tests are written before its package compiles.

Per target, mirroring `test/Mapping/generic_accelerator_target.cpp`:

- **Every rule's emit key is declared.** A rule that names an undeclared emitter is a load-time error, so this pins policy and plugin together.
- **Every declared key has an emitter.** `isKnownEmitter` agrees with what the factory returns.
- **Emitters verify well-formed bundles** — `verify()` succeeds on a bundle whose parameters are typed correctly.
- **A workload maps end to end**: a vector node matches, places, connects and covers against the profile.
- **`DoesNotClaimAnExecutableBackend`** — for every declared key, `lower()` fails with `target_lowering_unsupported`. This is the test that makes the static boundary explicit rather than incidental.

Additionally:

- **Loader cases** for both profiles in `test/Machine/machine_model_loader.cpp`, including that `arm-sme`'s `za` node is an `acc`-kind memory and that `arm-sve` declares no matrix engine.
- **A pass-spec test** driving `--micro-map` with `target=arm-sve`, proving the path that needs no registration.
- **A placement-difference assertion**: `micro.mma` on `arm-sve` binds a vector-engine bundle and on `arm-sme` binds a matrix-engine bundle. Without this, the two targets could silently collapse into the same behavior.

The full suite must pass under `ninja check-llk` on the local build, and CI must pass under LLVM 22.1.8. No change here touches a version-sensitive API, so no new compatibility guard is expected.

## 10. Claims and Limits

What this work establishes: the mapping framework accepts, loads, verifies, searches and binds against two new ARM target descriptions, with contraction placement differing between them by construction.

What it does not establish, stated so no reader infers it:

- **The vector length is modeled, not scalable.** SVE's defining property is a runtime-determined length; these profiles pin one. A second or third length is another profile file, not a schema change, and comparisons across lengths are the intended way to explore that dimension — but nothing here represents a length-agnostic program.
- **ZA is modeled as an accumulator memory and a matrix engine.** No ZA semantics exist in the IR, and no SME instruction is emitted or claimed.
- **No performance number is validated.** The profiles' costs are modeled estimates. The framework's rule that predicted metrics stay separate from verified measurement applies unchanged, and missing measurements must not rank as zero.
- **Latencies, bandwidths and capacities are calibration seeds** for a class of ARMv9-A cores. They are not measurements, and a plan ranked by them is a modeled plan, exactly as for the existing profiles.

## 11. Dependencies and Sequencing

The work is self-contained and can land as one increment. Order: profiles and policy first (they are pure data and can be verified by the loader tests alone), then the packages and registrations, then the per-target GTests. Each step is independently testable, which is what lets the TDD sequence hold.

The one external fact this design depends on is the Arm Architecture Reference Manual's ZA definition and the `FMOPA` element geometry. Both are cited in §5 and should be re-checked if a future revision changes the modeled SVL.
