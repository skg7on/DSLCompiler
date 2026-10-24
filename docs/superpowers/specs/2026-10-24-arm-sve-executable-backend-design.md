# Arm SVE fixed-VL executable backend: bounded design

**Status:** design gate for issue #184 (framework E9). Not implemented support.
**Parent:** #175. **Predecessor:** the static Arm mapping increment E1–E8
(#176–#183), whose acceptance ledger is
`docs/reviews/arm-mapping-targets-acceptance.md`.
**Date:** 2026-10-24.

## 1. Intent and boundaries

Add the first *executable* Arm backend: selected `arm-sve` emitters for a bounded
fixed-VL=512 vector subset, an AOT cross-lowered artifact, and emulated
selected-target numerical execution under QEMU. The increment proves that a
plan selected by the static target can be lowered to AArch64 SVE code and that
code computes the declared result on a runner that reproduces in CI.

The first executable subset is:

- target `arm-sve`, fixed VL = 512 bits;
- signatures `micro.vector "add"` and `micro.vector "mul"` on f32 rank-two
  tiles;
- fixtures `vector_add_f32.mlir`, `vector_mul_f32.mlir` (8×8, partial-vector
  tails) and one new full-width 16×16 fixture;
- runner: QEMU linux-user `qemu-aarch64` 8.2 with `sve-max-vq=4` (512-bit VL),
  in CI on the existing x86_64 Ubuntu 24.04 runner.

Non-goals for this increment:

- executable SME and ZA state, streaming mode, or the contraction fixture;
- bf16/i8 execution, conversions, SiLU, reduction, and the fused epilogue;
- native SVE execution (no supported host exists in this project's fleet; see
  §7), and any native or emulated timing claim;
- retargeting CPU adapter schedules, scalable/length-agnostic Micro semantics,
  calibrated performance, or persistence;
- physical SRAM/L2 realization. The static plan's memory identities remain
  analysis facts; the executable path consumes the ABI boundary buffers.

Static completion is not executable completion, and this design does not close
#175.

## 2. Audited current behavior

Revision `39766d4` (merge of PR #250):

- `arm-sve` and `arm-sme` load machine profiles and typed LLKMap policy, map,
  verify, replay and analyze complete static plans. Every Arm emitter reports
  `hasLowering() == false` and `lower()` returns
  `target_lowering_unsupported`; neither target returns
  `codegenRequirements()`; selected-target compilation is rejected (E4–E8
  tests and the static ledger).
- `llk-compile` exposes `--emit=llvm|mlir|micro|micro-search`. `llvm` is the
  only executable mode and it JITs for the host through `LLK::Runtime`
  (`runtime/MappedExecutable.cpp`). There is no object/AOT emission and no
  target-triple option.
- The runtime entry contract is MLIR's C-interface wrapper
  `_mlir_ciface_<kernel>` over rank-two memref descriptors, checked by
  `KernelAbi` v1 (`runtime/MappedExecutable.cpp`, `include/LLK/Runtime/KernelAbi.h`).
  `MappedJitEvidence` and the capability receipt already record plan, machine,
  ABI, selected/reference group counts and lowered IR.
- `TargetCodegenRequirements` carries `architecture`, `cpu` and
  `requiredFeatures`; `checkHostRequirements` compares them with host facts
  before JIT. There is no vector-length requirement field.

Toolchain and runner probes performed for this design (2026-10-24):

| Fact | Evidence |
|---|---|
| Local LLVM 24 and CI LLVM 22.1.6/8 contain `ArmSVE`/`ArmSME` dialects and AArch64 SME codegen | `mlir-opt --show-dialects`; `git ls-tree llvmorg-22.1.6`; `llvm/docs/AArch64SME.rst` at the tag |
| Clang targeting AArch64 emits SVE for a fixed 512-bit VL | `clang --target=aarch64-linux-gnu -march=armv8-a+sve -msve-vector-bits=512` produced `fadd z0.s, z0.s, z1.s` |
| LLVM emits SME streaming code for `aarch64_pstate_sm_body` | `llc -mattr=+sme` produced `smstart sm` / `zero {}` / `smstop sm` |
| Apple M5 exposes SME but not non-streaming SVE | `hw.optional.arm.FEAT_SME{,2}=1`; no `FEAT_SVE` |
| ZA access needs explicit enablement, not just streaming | native `smstart; zero {za}; smstop` returns 0; `smstart sm; zero {za}` traps with SIGILL |
| CI's LLVM build has no AArch64 target | `ci.yml` passes `-DLLVM_TARGETS_TO_BUILD=Native`; the runner is x86_64 |
| Ubuntu 24.04 ships `qemu-user` 8.2.2, whose linux-user AArch64 supports SVE and SME | Launchpad noble `qemu-user`; QEMU linux-user SME series merged for 7.1 |
| QEMU linux-user VL is set with `-cpu max,sve-max-vq=4` (4×128 = 512 bits) | QEMU AArch64 CPU properties; default linux-user VL is 512 bits |

## 3. Chosen scope and alternatives

**Chosen: fixed-VL=512 SVE vector first.**

Alternatives considered:

1. **SME stateful contraction first.** It would exercise the distinguishing
   matrix path and can run natively on the Apple M5 host. It requires streaming
   entry/exit, ZA ownership and preservation, commit/save/restore around calls,
   extraction, and an SME ABI boundary, and the ZA-enablement trap above shows
   how easy an incorrect entry sequence is. It is a larger first increment than
   the vector subset, and its native evidence would live on developer hardware
   rather than the required CI runner.
2. **Length-agnostic vector semantics.** It requires canonical scalable Micro
   types or VL ranges in mapping/analysis, which is a framework change, not a
   bounded backend. It stays a later comparison.
3. **SVE fixed-VL vector.** It reuses the shipped typed vector rules and
   fixtures, needs no canonical change, has exact f32 numerical semantics, and
   its runner is reproducible in CI. Its cost is new AOT/emission and harness
   infrastructure, shared with every later Arm backend.

SVE is therefore the first executable increment; SME is the immediate
follow-on (§11).

## 4. Architectural facts and assumptions

- The modeled VL remains 512 bits. The machine profile's per-dtype `lanes`
  (f32 16, bf16 32, i8 64) stay the source of the blocked layout width `VW`.
  This increment executes only f32; the other widths remain static policy.
- The canonical schedule, search, plan and analysis keep no instruction or
  register facts. They record VL=512 as a modeling fact and the selected `VW`
  as a layout parameter, exactly as today.
- The selected backend derives scalable vector shapes from the selected VW and
  emits standard `vector` dialect operations on `vector<vscale x N x f32>`
  types with masked tails. It does not emit fixed-width `vector<16xf32>` and
  rely on a target-machine minimum-width option: that option is a backend
  command-line setting, not an IR contract, and is not settable through the JIT
  target-machine builder. Scalable IR plus an exact runtime VL check makes the
  width assumption explicit and testable.
- "Fixed-VL" therefore means: the runtime requires an implemented VL of exactly
  512 bits and refuses otherwise. Code is not length-agnostic, and no scalable
  semantic guarantee is claimed. The actual VL is recorded in evidence.
- Host requirements for the executable subset are `architecture=aarch64`,
  `requiredFeatures=["sve"]`, and VL=512. They are checked before invocation:
  the emulated harness checks `AT_HWCAP` and `PR_SVE_GET_VL`; the native JIT
  path extends `checkHostRequirements` with a VL probe. A failed check is a
  stable capability refusal, never a silent reference fallback.
- The static targets' memory identities (`sram`, `acc`) remain modeled. The
  executable ABI exchanges caller-owned rank-two buffers; this increment does
  not claim physical SRAM placement or DMA realization.

## 5. Selected lowering contract

- `sve_vector_add` and `sve_vector_mul` become real emitters with the same
  contract as the AVX2 emitters: validate the whole bundle, ports, layouts and
  selected parameters before any rewrite; keep the covered Micro operation and
  its `micro.mapping` provenance intact until the target backend pass consumes
  it; never half-rewrite on error. Every other `arm-sve` key keeps the
  rejecting default emitter.
- A new `SVEBackendLoweringPass` runs after the generic Micro→Linalg
  conversion and vectorization. It reads only opaque mapping metadata, tiles
  the covered operation into scalable SVE vectors derived from the selected
  `VW`, and emits masked tail handling. It rejects an unsupported dtype, a
  missing or conflicting `VW`, an f32 width other than 16, or an operation it
  cannot cover.
- The AArch64 target is configured with `+sve` (and no `-msve-vector-bits`
  backend option). The acceptance asserts the lowered artifact contains SVE
  vector/predicate instructions, not merely AArch64 code.
- `arm-sve` gains `codegenRequirements()` for the executable subset:
  `architecture=aarch64`, `requiredFeatures=["sve"]`. Contraction and the
  unimplemented emitters still refuse lowering, so the shipped static fixtures
  remain static-only. The existing static-only tests that assert *both* Arm
  targets lack codegen requirements are narrowed to the unimplemented subset
  rather than deleted. Documentation and the E8 ledger must be corrected where
  they say selected-target compilation is rejected for `arm-sve` *without
  qualification*; the qualified statement is "rejected for signatures without
  an implemented emitter".
- Reference lowering is never substituted: the evidence records
  `backendGroupsRealized` and `referenceGroupsLowered`, and the executable
  acceptance requires zero reference groups for the covered vector operation.

## 6. Artifact and runner

### 6.1 AOT object emission

`llk-compile` gains target-neutral AOT flags:

```text
--emit=object
--target-triple=aarch64-unknown-linux-gnu
--target-features=+sve
--emit-identity=<path.json>
```

The pass pipeline is unchanged through `--mapping-stop=lowered`; the AOT path
runs the same `prepareMappedKernelForInvocation` preparation the runtime uses,
translates with the unified ConvertToLLVM pipeline for the requested triple,
and emits an object plus a JSON identity sidecar. The exported entry is the
same `_mlir_ciface_<kernel>` wrapper with the same `KernelAbi` v1 contract, so
the harness and the JIT path speak one ABI. `--emit=llvm` keeps its host-JIT
meaning; the new flags are rejected with `--emit=llvm|mlir|micro|micro-search`.

### 6.2 Freestanding harness

A small AArch64 harness, built without libc:

- checks `AT_HWCAP` for SVE and `PR_SVE_GET_VL` for exactly 512 bits before
  calling the kernel, exiting with a distinct code when the requirement fails;
- allocates aligned input and output buffers, fills inputs from literal tables,
  pre-fills outputs with sentinel bytes, constructs rank-two memref descriptors
  (offset 0, unit row stride) and calls `_mlir_ciface_<kernel>` once;
- compares each output element against the literal expected f32 bits, verifies
  the sentinel region outside the logical extent is untouched, and exits 0 only
  on a complete match;
- carries a source hash in its identity record so the harness itself is pinned.

The harness is built by the CI job from a cross toolchain (Ubuntu's
`gcc-aarch64-linux-gnu`, recorded separately from the kernel's LLVM identity)
and linked statically with the LLK-produced object.

### 6.3 Runner and CI

The ARM-evidence CI job:

1. builds LLVM with `-DLLVM_TARGETS_TO_BUILD=X86;AArch64` so the pinned
   LLVM 22.1.8 can cross-emit AArch64 (the default `Native` build cannot). This
   changes the LLVM build inputs and therefore requires a new build-cache key;
2. installs `qemu-user` and `gcc-aarch64-linux-gnu`;
3. builds the project and the AOT object, then the harness;
4. runs `qemu-aarch64 -cpu max,sve-max-vq=4 ./harness`;
5. publishes the object, identity JSON, harness source hash, qemu version, CPU
   model and captured VL as CI artifacts.

The job never runs on ARM hardware, and a host without qemu or without the
AArch64 LLVM target fails the job rather than skipping the numerical check.

## 7. Evidence tiers and identities

| Tier | What is established | Runner |
|---|---|---|
| T0 static | Complete plan, mapped verification, shared analysis, replay | Any host (E1–E8) |
| T1 cross-lowered artifact | The selected plan lowers to an AArch64 object containing SVE instructions | CI x86_64, pinned LLVM 22.1.8 |
| T2 emulated selected-target numerical execution | That object computes the declared f32 result, tails included, with no reference fallback | CI x86_64, `qemu-aarch64` 8.2, VL=512 |
| T3 native selected-target numerical execution | — | **Omitted:** no host in this project exposes non-streaming SVE. Apple M5 exposes SME/streaming SVE only; CI is x86_64. |

Every artifact records: source graph hash, selected plan id, machine/policy/
target hashes, compiler version, LLVM version, target triple, target features
(`+sve`), ABI hash and version, required and actual VL, capability-receipt
statuses, selected/reference group counts, runner (qemu version, CPU model) and
harness source hash. A missing field is a failure, not zero.

T2 is functional evidence only. It does not calibrate cycles, bytes or storage,
and modeled `micro-perf` numbers stay separate. The existing AVX2
selected-versus-reference acceptance remains the executable baseline and is not
reclassified.

## 8. Numerical contract and negative controls

- f32 add and multiply use IEEE-754 binary32 operations with no fast-math or
  contraction flags. Expected outputs are literal bit patterns, including
  signed zeros, infinities and a NaN case compared by classification.
- Tails: the 8×8 fixtures exercise partial scalable vectors. A new 16×16
  fixture exercises full-width vectors. Both compare every element and the
  untouched canary region.
- Initialization: outputs are sentinel-filled before invocation; every logical
  element must be overwritten and no byte outside the logical extent may
  change. No accumulator or fused initialization is claimed in this increment.
- Negative controls:
  - unsupported dtype or operation: the emitter/backend refuses lowering with a
    stable diagnostic and no partial rewrite;
  - `arm-sme` and every unimplemented `sve_*` key still reject lowering;
  - missing `sve` feature or VL ≠ 512: capability refusal before invocation,
    distinct from a numerical mismatch;
  - wrong host architecture: existing `checkHostRequirements` refusal;
  - tampered or foreign plan/emitter: rejection by the bundle contract;
  - any reference-group fallback: failure, never a passing run.

## 9. Acceptance fixtures and expected evidence

Fixtures under `test/Conversion/MicroMapping/Inputs/arm/`:

- existing `vector_add_f32.mlir`, `vector_mul_f32.mlir` (8×8, tails);
- new `vector_add_f32_full.mlir` and `vector_mul_f32_full.mlir` (16×16,
  full-width), same typed shape and direct-MicroIR style.

Expected acceptance:

1. Each fixture maps on `arm-sve` (deterministic mode), verifies and reports a
   complete plan exactly as the static ledger records.
2. The selected plan lowers through the SVE subset with
   `referenceGroupsLowered == 0`, producing AArch64 code that contains SVE
   vector/predicate instructions.
3. The cross-lowered object's identity sidecar is complete and agrees with the
   plan report.
4. Under `qemu-aarch64 -cpu max,sve-max-vq=4`, every output element matches the
   literal expectation, canaries hold, and the reported VL is 512.
5. Removing SVE from the qemu CPU, or running on x86_64 without qemu, fails the
   evidence job; it never skips into a pass.
6. `arm-sme`, unsupported dtypes/operations and tampered plans keep rejecting.

## 10. Implementation sequence

The implementation plan is written separately from this design and owns the
task breakdown, exact files and test-first order. Its required shape:

1. AOT object emission with identity sidecar and ABI wrapper reuse.
2. Freestanding harness and a qemu-backed evidence runner, wired into CI with
   the AArch64 LLVM target enabled.
3. `sve_vector_add`/`sve_vector_mul` bundle handoff and the SVE backend pass.
4. `codegenRequirements` for the executable subset and the VL/feature checks.
5. Numerical evidence for full-width and tail cases, including canaries and
   initialization.
6. Negative controls, capability receipts and documentation corrections.
7. A revision-pinned acceptance ledger for the executable increment.

Each task must be independently testable and fail before its implementation.

## 11. Follow-on issues and limits

This design authorizes no implementation by itself. After approval, create:

- **E10 — executable fixed-VL SVE subset**: implement tasks 1–7 above against
  `arm-sve`, with CI-emulated numerical evidence; blocked by #184's approval.
- **E11 — SME stateful backend**: design and implement a native/QEMU SME
  contraction path, including streaming entry/exit, explicit ZA enablement,
  ownership/preservation and extraction; blocked by E10's shared AOT and
  evidence infrastructure.
- #175 remains open until its first executable-backend acceptance is actually
  met. #54 (direct Micro interpretation) and #51/#52 (persistence/calibration)
  stay related capabilities, not automatic dependencies.

Limits this design accepts and states:

- No native SVE runner exists here; T2 is emulated functional evidence.
- No executable SME, no physical ZA, no calibrated timing, no CPU-schedule
  retargeting.
- The modeled SRAM/acc storage identities are not physically realized by the
  vector subset; only the ABI boundary buffers are.
- Small direct fixtures do not establish arbitrary DNN support or
  length-agnostic SVE.

## References

- Static Arm acceptance: `docs/reviews/arm-mapping-targets-acceptance.md`.
- Static design and plan:
  `docs/superpowers/specs/2026-10-17-arm-sve-sme-mapping-targets-design.md`,
  `docs/superpowers/plans/2026-10-17-arm-sve-sme-mapping-targets.md` §11.
- LLVM SME ABI and code generation: <https://llvm.org/docs/AArch64SME.html>.
- MLIR ArmSME dialect: <https://mlir.llvm.org/docs/Dialects/ArmSME/>.
- Clang vectors and extended vectors:
  <https://clang.llvm.org/docs/LanguageExtensions.html#vectors-and-extended-vectors>.
- Apple SME documentation:
  <https://github.com/apple-oss-distributions/xnu/blob/main/doc/arm/sme.md>.
- QEMU linux-user SME series (merged for 7.1):
  <https://patchew.org/QEMU/20220708151540.18136-1-richard.henderson@linaro.org/>.
- Ubuntu 24.04 `qemu-user` 8.2.2:
  <https://launchpad.net/ubuntu/noble/arm64/qemu-user/1:8.2.2+ds-0ubuntu1>.
