# Arm SME stateful executable backend: bounded design

**Status:** design gate for issue #253 (framework E11). Not implemented support.
**Parent:** #175. **Predecessor:** E10 (#252), whose evidence ledger is
`docs/reviews/2026-10-24-arm-sve-executable-acceptance.md`.
**Date:** 2026-10-24.

## 1. Intent and boundaries

Add the first executable **stateful** Arm backend: selected `arm-sme` lowering
of the accepted initialized 16x16x16 f32 contraction, with explicit streaming
entry/exit, ZA ownership, extraction and an ABI boundary. The increment proves
that a plan selected by the static `arm-sme` target lowers to Arm SME code and
computes the declared result on a runner that reproduces.

The first acceptance runner is a **native Apple M5** (macOS arm64, FEAT_SME),
recorded as evidence tier T3. QEMU linux-user 8.2 (SME1) is the secondary,
CI-automated runner (T2). Native T3 gates the increment; T2 keeps a
reproducible check in CI, which cannot run macOS Arm hardware.

The first executable subset is:

- target `arm-sme`;
- signature: the direct-MicroIR initialized contraction
  `contraction_f32.mlir` (`micro.mma` 16x16x16, f32 inputs and accumulator,
  `%init` preloaded into the accumulator);
- dtype: **f32 only**; bf16 and i8 contraction remain statically mapped but are
  not executable in this increment;
- native runner: Apple M5 with streaming vector length (SVL) 512 bits;
- emulated runner: `qemu-aarch64` 8.2.2 (SME1) on the existing x86_64 Ubuntu
  24.04 CI runner.

Non-goals for this increment:

- bf16 execution, widening `BFMOPA`, SiLU, reduction, conversion, fused
  epilogue, or any other `sme_*` signature;
- SME2 operations (2-way/4-way outer products, ZA array load/store), multi-tile
  kernels, or more than one ZA tile;
- ZA preservation across calls, ZA spill/restore across unrelated calls, or
  nested streaming calls through non-streaming code;
- retargeting CPU adapter schedules, length-agnostic or dynamic shape support,
  calibrated timing, persistence, or measurement;
- physical SRAM/L2/DMA realization. The modeled `sram`/`za.0` identities remain
  analysis facts; the executable path consumes the ABI boundary buffers and the
  architectural ZA array.

Static completion is not executable completion, and this design does not by
itself close #175.

## 2. Audited current behavior

Revision: `5ec9feb` (merge of E10, PR #255).

- `arm-sme` loads a machine profile and typed LLKMap policy, maps, verifies,
  replays and analyzes complete static plans, including the initialized f32 and
  bf16 contractions with `za.0` accumulator storage. Its factory returns a bare
  `FileMappingTarget`: no `codegenRequirements()`, no selected backend
  pipeline, and verify-only `DeclaredTargetEmitter`s. Selected-target
  compilation is rejected with "has no ISA requirements".
- `micro.mma` lowers to `linalg.matmul` with `micro.mapping` provenance in
  `MicroToLinalg`. The AVX2 `avx2_mma` path is the in-tree executable
  contraction template; the SVE path is the executable elementwise template.
- E10 supplies the shared AOT object path (`llk-compile --emit=object`,
  `--target-triple`, `--target-features`, `--emit-identity`), the
  `MappedObjectIdentity` sidecar, the freestanding-harness + QEMU evidence
  driver, the capability receipt, and the `backendGroupsRealized` /
  `referenceGroupsLowered` accounting contract.
- No ArmSME dialect op, `smstart`/`smstop`, streaming attribute, ZA operation or
  SME/SVL host probe exists anywhere in `lib/`, `include/`, `runtime/`,
  `tools/` or `test/`.
- `TargetCodegenRequirements` carries `architecture`, `cpu`,
  `requiredFeatures` and an optional non-streaming `vectorLengthBits`. Host
  detection is SVE-only (`PR_SVE_GET_VL`); there is no SME or SVL probe.
- `MappedCompilation::lowerToBackendForm` runs `micro-to-linalg`, then the
  selected target's pre-bufferization backend pipeline, then One-Shot
  Bufferize, then `convert-linalg-to-loops`. There is no post-bufferization
  target hook.

Probes performed for this design (2026-10-24):

| Fact | Evidence |
|---|---|
| The development host is an Apple M5 with SME and SME2 | `sysctl -n hw.optional.arm.FEAT_SME hw.optional.arm.FEAT_SME2` returns `1 1`; no `FEAT_SVE` OID exists |
| The M5 streaming vector length is exactly 512 bits | an SME kernel entered with `smstart sm` reports `CNTB` = 64 bytes on two consecutive calls |
| `smstart` (implicit ZA and SM) enables ZA; `zero {za}` then executes | native `smstart; zero {za}; smstop` exits 0 |
| `smstart sm` alone does not enable ZA | native `smstart sm; zero {za}; smstop sm` traps with SIGILL (exit 132) |
| `RDSVL` is undefined when streaming mode is off on this host | native `rdsvl x, #1` outside streaming traps with SIGILL |
| MLIR and LLVM 22.1.8 / 24 expose the production ArmSME pipeline | `arm-sme-vector-legalization`, `convert-arith-to-arm-sme`, `convert-vector-to-arm-sme`, `convert-arm-sme-to-scf`, `convert-arm-sme-to-llvm`, `enable-arm-streaming`, `arm-sme-outer-product-fusion`; tile allocation is inside `convert-arm-sme-to-llvm` |
| `enable-arm-streaming` option spellings match in LLVM 22.1.8 and 24 | `streaming-mode={disabled,streaming,streaming-locally,streaming-compatible}`, `za-mode={disabled,new-za,in-za,out-za,inout-za,preserves-za}`, `if-required-by-ops`, `if-scalable-and-supported` |
| MLIR's reference 16x16 f32 matmul uses scalable `[4]x[4]` tiles and `vector<[4]x[4]xf32>` | `mlir/test/Integration/Dialect/Linalg/CPU/ArmSME/matmul.mlir` (vscale 4 at SVL 512 gives 16x16) |
| Both Apple clang 21 and project LLVM 24 emit `rdsvl`/`cntd` before `smstart` in a locally-streaming function or a caller of a streaming callee | disassembly of `__arm_locally_streaming __arm_new("za")` functions and of a plain caller calling a `__arm_streaming` callee |
| Caller-managed streaming works natively | assembly shim `smstart sm` -> streaming wrapper -> `sm_enabled`+`new_za` kernel -> `smstop sm`, called twice, exits 0 with `CNTB` = 64 both times |
| A naked assembly shim must preserve `x30` | a `bl` without saving the link register loops on the final `ret` |
| QEMU 8.2 supports FEAT_SME in linux-user mode | QEMU `docs/system/arm/emulation.rst` and `target/arm/tcg/cpu64.c` at v8.2.0; not locally executable because this macOS host has no `qemu-aarch64` or AArch64 Linux cross toolchain |
| Ubuntu 24.04 CI ships `qemu-user` 8.2.2 and `gcc-aarch64-linux-gnu` | `.github/workflows/ci.yml`; E10 evidence run |

## 3. Chosen scope and alternatives

**Chosen: native M5 first, caller-managed streaming, f32-only contraction,
reusing MLIR ArmSME lowering.**

Alternatives considered:

1. **QEMU-first emulated acceptance.** It is CI-reproducible and was the E10
   tier model, but the project now has a documented SME-capable native host
   (Apple M5). A native run is stronger evidence and avoids basing the first
   stateful backend on an emulator's SME model. It stays the secondary tier
   because CI cannot execute it.
2. **Native and QEMU both gating.** It doubles runner, harness and ABI work and
   makes CI unable to verify half the gate. QEMU remains automated evidence,
   not a second gate.
3. **`streaming-locally` (`aarch64_pstate_sm_body`) entry.** It is MLIR's
   reference default, but it is unusable on this host: both compilers emit
   `rdsvl`/`cntd` before `smstart`, and those instructions are undefined while
   streaming mode is off on a machine without non-streaming SVE. See §5.
4. **Externally managed SM and ZA with no function attributes.** It also runs,
   but it discards the ABI's streaming contract and cannot express ZA
   enablement/preservation through LLVM attributes. Caller-managed streaming
   plus `new-za` keeps the standard ZA sequence while making only PSTATE.SM the
   caller's responsibility.
5. **Hand-emitting ArmSME instructions.** It would duplicate tile allocation,
   slicing and instruction selection that MLIR already provides. The design
   reuses the production passes and rejects the test-only `-test-lower-to-arm-sme`
   wrapper.
6. **Running ArmSME conversion in the shared runtime LLVM pipeline.** It would
   apply SME conversion to every target on the JIT/AOT path. The conversion
   belongs in a target-owned pass.
7. **Bufferizing inside the target pass.** It would make a target pass own the
   generic pipeline's bufferization. A target-neutral post-bufferization hook
   keeps the boundary.

## 4. Architectural facts and assumptions

- The canonical MicroIR, schedule, search, plan and shared analysis gain **no**
  streaming, register or instruction facts. They continue to record the
  16x16x16 shape, f32 dtypes, the selected `za` matrix engine and the
  `za.0` accumulator budget.
- SVL and VL are independent. The machine profile's 512-bit vector length and
  4096-byte `za.0` capacity are modeling facts. This increment executes at an
  actual SVL of exactly 512 bits; the runner verifies `CNTB` = 64 bytes before
  the first call. A different SVL is a capability refusal, not a retry.
- z/ZA widths, streaming mode, ZA enablement, tile allocation and slice
  addressing are backend/runner facts. They live in the `arm-sme` target
  package, LLVM attributes and the harness, never in shared code.
- MLIR's ArmSME tile type for f32 is `vector<[4]x[4]xf32>`; at SVL 512 the
  scalable dimension is 4 (`vscale*4 = 16`), so one tile is the fixture's
  16x16 f32 accumulator.
- The ZA array is 4096 bytes at SVL 512 and holds four logical ZA.S tiles.
  This increment uses exactly one tile. No physical ZA access, spill or
  multi-tile arrangement is claimed.
- `+sme` is the target feature; the AArch64 backend supplies the streaming-SVE
  instructions the Tile/Vector lowering needs. The acceptance asserts genuine
  SME instructions (`fmopa`, `zero {za}`, `smstart za`), not merely AArch64
  code.
- The ABI wraps the same `_mlir_ciface_<kernel>` entry and `KernelAbi` v1
  rank-two descriptors as E10. On Mach-O the object symbol gains the platform
  underscore; the identity record keeps the MLIR-level `wrapperSymbol`.
- Reference lowering is never substituted: a selected run must report
  `backendGroupsRealized > 0` and `referenceGroupsLowered == 0`.

## 5. Streaming and ZA ABI

### 5.1 Entry/exit decision

The kernel and the exported invocation wrapper are compiled for **streaming
mode with caller-managed PSTATE.SM**:

| Entity | LLVM function attributes | Emitted state changes |
|---|---|---|
| Contraction kernel (contains `arm_sme` ops) | `aarch64_pstate_sm_enabled`, `aarch64_new_za` | `smstart za`, `zero {za}` at entry; `smstop za` at exit |
| Exported `_mlir_ciface_*` wrapper | `aarch64_pstate_sm_enabled` | none; it assumes SM is enabled |
| Assembly entry shim (runner-owned) | none (hand-written) | `smstart sm` before the call, `smstop sm` after it |

`enable-arm-streaming` with `streaming-mode=streaming`,
`za-mode=new-za` and `if-required-by-ops=true` attaches the two kernel
attributes. The invocation attributes are target policy
(`invocationAttributes` in the codegen contract, §6.4), applied to the wrapper
by generic AOT/JIT code.

The assembly shim preserves `x29`/`x30`, executes `smstart sm`, calls the
wrapper, executes `smstop sm`, and returns. It is the only function that
changes PSTATE.SM. Every function in the generated call chain is marked
streaming, so no compiler-generated transition runs while SM is enabled.

### 5.2 Why not `streaming-locally`

`streaming-locally` maps to `aarch64_pstate_sm_body`, which asks the callee to
manage its own entry/exit. On this host that produces an invalid prologue: both
Apple clang 21 and project LLVM 24 emit `rdsvl`/`cntd` (to size scalable spills)
**before** `smstart`, and those instructions trap while SM is off on an Apple
M5, which has no non-streaming SVE. The same hazard applies to a plain caller of
a streaming callee. The design therefore does not use `sm_body` and does not
depend on compiler-generated transitions.

### 5.3 ZA enablement, preservation and commit

- **Enablement**: `aarch64_new_za` produces `smstart za` (explicitly enabling
  ZA, not only streaming mode) followed by `zero {za}`. This is the sequence the
  #184 probe identified: `smstart sm` alone leaves ZA disabled and a ZA access
  traps.
- **Initialization**: the fixture's `%init` is loaded into the ZA tile before
  the accumulation (the lowered `vector.transfer_read` of the initializer
  becomes the tile's initial value). The kernel additionally zeroes ZA first,
  so the tile never depends on prior state.
- **Ownership**: ZA is owned by the kernel invocation. `new_za` means ZA is not
  live on entry or exit and is not preserved across calls. The kernel performs
  no ZA spill or restore.
- **Commit/extraction**: the accumulator tile is read back slice-by-slice into
  memory after the final accumulation. No ZA state is returned to the caller.
- **Calls**: the shipped path has no call inside the accumulation. If a future
  increment introduces one, all participating functions must be streaming and
  the ABI's ZA save/restore rules apply; that is out of scope here.
- **ABI runtime**: `new_za` code consults `TPIDR2_EL0` and calls
  `__arm_tpidr2_save` only when it is nonzero. At the exported boundary it is
  zero on the native runner (measured by the probe). QEMU/Linux does not leave
  it zero, so the entry shim writes zero to `TPIDR2_EL0` after `smstart sm`
  and before the kernel call, making the no-incoming-ZA precondition explicit
  and identical on both runners. The native runner links the platform
  compiler-rt implementation. The freestanding QEMU runner supplies explicit
  trap stubs for `__arm_tpidr2_save`/`__arm_tpidr2_restore`; the shim's clear
  means the single-threaded, top-level call flow never enters them, and a stub
  firing is a test failure, not a silent continuation.

### 5.4 Across-call correctness

The harness calls the shim, and therefore the kernel, twice with different
inputs and initializers. Each call must return its own exact result. A leaked
accumulator would make the second result depend on the first and fail the
comparison. This is the positive evidence for fresh per-call ZA; the
"streaming-only entry" negative control (§9) covers incorrect enablement.

### 5.5 What stays visible to canonical layers

Visible to canonical IR/search/analysis: the 16x16x16 contraction shape, the
f32 input and accumulator dtypes, the initialized-accumulator dataflow, the
selected `za` matrix engine, and the `za.0` capacity/live bytes. Not visible:
SVL, streaming mode, ZA enablement, tile allocation, slice layout, register
widths or the entry shim. No new canonical operation or type is introduced.

## 6. Selected lowering contract

### 6.1 Emitter

`SMEEmitter` replaces the default declared emitter for the `arm-sme` package.
`hasLowering()` is true only for `sme_mma_za`. `hasLoweringFor(group)` accepts
exactly one covered `micro.mma` operation with:

- f32 inputs and f32 accumulator;
- static shape 16x16x16;
- rank-two tile operands and a tile result in `#micro.memory<acc>`;
- the initializer-as-accumulator dataflow the static fixture establishes.

Every other `sme_*` key, including `sme_mma_za` covering bf16, keeps the
rejecting default behavior. The emitter validates the whole bundle and group
before any mutation and never half-rewrites. It does not rely on a
target-specific handoff attribute surviving `MicroToLinalg`; the target passes
key off `micro.mapping` as the SVE backend does.

### 6.2 Three-stage lowering

All three stages are owned by the `arm-sme` target package and run only for a
selected-target compilation.

1. **Pre-bufferization (`SMEVectorLoweringPass`)** runs after `micro-to-linalg`.
   It finds the selected `linalg.matmul` (`emitter == "sme_mma_za"`,
   `micro.op == "mma"`, and selected bundle `sme.mma.f32`), validates the
   signature, tiles with scalable `[4]x[4]` sizes, and vectorizes to
   `vector.contract` while preserving the initialized accumulator
   `vector.transfer_read` and the result `vector.transfer_write`. The bundle
   name, not the operand type, distinguishes the f32 signature from a bf16
   contraction: `MicroToLinalg` casts narrower inputs to the f32 accumulator
   type, so both reach the pass with f32 operands and an f32 result. It rejects
   a shape other than 16x16x16, a missing selection, or a conflicting value,
   without partial rewrites.
2. **Post-bufferization (`SMEArmLoweringPass`)** runs after One-Shot Bufferize
   and before `convert-linalg-to-loops`. The reference ArmSME pipeline applies
   its Vector patterns after bufferization because the unit-dim transfer
   patterns operate on memrefs, so this pass first applies the permutation-map,
   unit-dim, contract-to-outerproduct and mask patterns, then the production
   sequence up to streaming enablement:

   ```text
   arm-sme-vector-legalization
   convert-arith-to-arm-sme
   convert-vector-to-arm-sme
   convert-arm-sme-to-scf
   convert-vector-to-scf
   enable-arm-streaming{streaming-mode=streaming za-mode=new-za if-required-by-ops}
   ```

   It does **not** run `arm-sme-outer-product-fusion`: fusion creates
   2-way/4-way widening operations that are not part of the SME1 acceptance
   surface. It stops before `convert-scf-to-cf`. The module still carries
   `arm_sme.outerproduct`/tile operations, so the required evidence is checked
   here, before the late stage erases it: on success it sets
   `llk.backend_groups_realized`, and if the selected contraction did not become
   an SME tile computation it emits a diagnostic and leaves the group
   unrealized. The result is still structured SCF.
3. **Late (`SMELateLoweringPass`)** runs *after* the shared ABI/ownership
   preparation (`prepareMappedKernelForInvocation`) and before code emission:

   ```text
   convert-scf-to-cf
   convert-arm-sme-to-llvm
   canonicalize, cse
   ```

   Tile allocation, performed inside `convert-arm-sme-to-llvm`, needs the
   flattened control flow this stage produces. Ownership-based buffer
   deallocation, which preparation runs, accepts only structured loops and
   refuses any function with a backedge. Running the flattening in stage 2 made
   every ArmSME artifact uncompilable ("Only structured control-flow loops are
   supported"); splitting it keeps the shared ownership contract and the
   target-owned control-flow change apart. Stage 3 leaves the ArmSME dialect's
   LLVM intrinsic operations (`arm_sme.intr.*`), which the runtime's LLVM
   translation converts to `llvm.aarch64.sme.*` calls.

The runtime LLVM stage (`addKernelToLLVMPasses`) is unchanged and target-neutral.
It sees ArmSME intrinsic operations after `convert-arm-sme-to-llvm`; the runtime
registers the ArmSME translation interface (Task 6).

### 6.3 Framework hooks

- `MappedCompilation::lowerToBackendForm` gains a second, target-neutral hook:

  ```cpp
  virtual void MappingTarget::buildPostBufferizationBackendPipeline(
      mlir::OpPassManager &pm) const;  // default: no-op
  ```

  It is invoked after One-Shot Bufferize and before `convert-linalg-to-loops`.
  The existing pre-bufferization hook and all other targets are unchanged.
- A third target-neutral hook covers passes that must follow the shared
  ownership preparation:

  ```cpp
  virtual llvm::Error MappingTarget::buildLateBackendPipeline(
      mlir::OpPassManager &pm) const;  // default: no-op
  ```

  `llk::runLateBackendPipeline(target, module)` builds and runs it. The shared
  `finishCompilation` calls it for a selected-target executable after
  `prepareMappedKernelForInvocation` and before `createMappedExecutable`, and
  `llk-compile --emit=object` calls it before `emitMappedObject`. The split is
  the ownership/control-flow contract: preparation owns structured loops, the
  target owns any later flattening. It is inert for every other target.
- No generic mapping, machine or perf code branches on `sme` or `arm-sme`.

### 6.4 Codegen contract and invocation attributes

`TargetCodegenRequirements` gains two fields with default initializers:

```cpp
std::optional<int64_t> streamingVectorLengthBits = std::nullopt;
std::vector<std::string> invocationAttributes{};
```

`arm-sme` publishes `{architecture="aarch64", cpu="", requiredFeatures={"sme"},
vectorLengthBits=nullopt, streamingVectorLengthBits=512,
invocationAttributes={"aarch64_pstate_sm_enabled"}}`.

The generic execution paths apply `invocationAttributes` as LLVM function
attributes on the exported `_mlir_ciface_*` wrapper:

- AOT: `MappedObjectOptions` carries the attributes; `emitMappedObject` applies
  them to the wrapper before translation.
- JIT: `MappedJitOptions.selectedTarget` already carries the contract; the
  runtime applies the same attributes.

They are recorded in the identity sidecar. A nonempty list also removes
inherited Arm ZA-ownership attributes from the wrapper (`aarch64_new_za`,
`aarch64_in_za`, `aarch64_out_za`, `aarch64_inout_za`,
`aarch64_preserves_za`). MLIR's C-interface wrapper copies the kernel's
discardable attributes; a wrapper with `new_za` would establish a `TPIDR2_EL0`
save block around the kernel, which the kernel's own `new_za` prologue would
then have to save. The wrapper therefore carries only the target's declared
invocation attributes. `checkHostRequirements` verifies
`streamingVectorLengthBits` when the host reports an SVL; a host that cannot
report it is a refusal, not an assumed match. The runner additionally verifies
SVL with `CNTB` under the entry shim. The JIT path is not the native evidence
path; the AOT object plus harness is. Consequently the JIT path refuses
selected `arm-sme` execution on hosts without an SVL probe (including macOS),
while AOT emission and the native harness remain supported. On macOS,
`llvm::sys::getHostCPUFeatures()` also does not report `sme`, so the JIT route
refuses with "selected target requires host feature 'sme'" even on an
SME-capable M5. That refusal is recorded as a capability limit rather than
worked around with an assumed SVL or an assumed feature set.

## 7. Artifact and runner

### 7.1 Native Mach-O object (T3, gating)

```text
llk-compile --mapping-target=arm-sme --mapping-backend=selected-target \
  --mapping-stop=lowered --emit=object \
  --target-triple=arm64-apple-macosx14.0.0 --target-features=+sme \
  --output=kernel.o --emit-identity=identity.json \
  test/Conversion/MicroMapping/Inputs/arm/contraction_f32.mlir
```

The object is a 64-bit Mach-O `CPU_TYPE_ARM64` file exporting the wrapper. The
identity record carries the triple, `+sme`, SVL, invocation attributes, plan,
machine, target, ABI hashes and group counts.

### 7.2 Native harness and shim

- A macOS C harness compiled by Apple clang without SME codegen; it links the
  LLK object and an assembly shim assembled with `-march=armv9-a+sme`.
- The harness checks `sysctl hw.optional.arm.FEAT_SME` (exit 70 when absent).
  The shim returns the `CNTB` value measured with SM enabled; the harness
  requires exactly 64 bytes (exit 71 otherwise). It then fills A/B/init from
  literal tables, sentinel-fills the output, calls twice, compares every result
  bit, and checks a 32-word canary after the logical extent.
- The shim's source hash and the harness source hash are recorded.
- Disassembly (via the LLVM build's `llvm-objdump`) must contain an SME
  accumulation (`fmopa`), an explicit `zero {za}`, and `smstart za`; a
  non-SME object fails the evidence test.

### 7.3 QEMU linux-user object (T2, CI)

The same pipeline targets `aarch64-unknown-linux-gnu` with `+sme`. A
freestanding harness (E10 style, no libc) plus the same assembly shim:

- checks `AT_HWCAP2` for `HWCAP2_SME` (exit 70) and the `CNTB` value returned
  by the shim (exit 71 when it is not 64);
- supplies the documented `__arm_tpidr2_*` trap stubs;
- runs under `qemu-aarch64 -cpu max,sme=on`;
- negative control `-cpu max,sme=off` must exit 70;
- the wrong-SVL control uses QEMU's SME vector-length property if 8.2 exposes
  one; otherwise it is explicitly omitted and the `CNTB` assertion is the
  guard.

The CI job reuses the E10 `LLK_BUILD_ARM_EVIDENCE` shape and uploads the
object, identity JSON, harness/shim sources, disassembly and runner record.
The native T3 run is not CI-reproducible; its command, host identity and
artifact hashes are recorded in the acceptance ledger.

## 8. Evidence tiers and identities

| Tier | What is established | Runner |
|---|---|---|
| T0 static | Complete plan, verification, analysis, replay | Any host (E1-E8) |
| T1 cross-lowered artifact | The selected plan lowers to an object containing SME instructions | CI x86_64 and macOS arm64 |
| T2 emulated selected-target numerical execution | The object computes the declared result under QEMU SME1 | CI x86_64, qemu 8.2, SVL 512 |
| T3 native selected-target numerical execution | The object computes the declared result on Apple M5 hardware | macOS arm64, SVL 512 |

Every artifact records: source graph hash, selected plan id, machine/policy/
target hashes, compiler version, LLVM version, target triple, target features,
ABI version/hash, required and actual SVL, streaming/ZA mode and invocation
attributes, capability-receipt statuses, selected/reference group counts,
runner identity (OS build, CPU, SME/SME2 features, Apple clang version for the
harness, shim/harness source hashes) and object/harness SHA-256 digests. A
missing field is a failure, not zero.

T2 and T3 are functional evidence only. Neither calibrates cycles, bytes,
storage or timing; `micro-perf` numbers remain modeled predictions. The
selected-AVX2 acceptance remains the executable baseline and is not
reclassified.

## 9. Numerical contract and negative controls

- The contraction is f32 IEEE-754 binary32 with no fast-math or reassociation
  flags. A, B and `%init` are integer-valued and small enough that every
  product, every partial sum and the initialized total are exactly
  representable. The expected result is therefore independent of the
  hardware's internal accumulation order, and the harness compares literal
  bit patterns.
- Signed zero, infinities and NaNs are **not** claimed for the contraction;
  they remain covered by the E10 vector evidence. No fused epilogue or SiLU is
  claimed.
- The output buffer is sentinel-filled before the call; every logical element
  must be overwritten and no byte outside the logical extent may change.
- Two calls with different inputs must each match their own expected result.
- Negative controls:
  - bf16 contraction selected-target compilation refuses with a stable
    diagnostic; `referenceGroupsLowered` never substitutes a reference path;
  - i8 contraction retains no plan (existing E6 behavior);
  - missing SME: native `FEAT_SME` absent exits 70; QEMU `sme=off` exits 70;
  - incorrect ZA enablement: a test-only shim that executes `smstart sm` only
    makes the kernel's ZA access trap with SIGILL (the driver requires signal
    4), matching the #184 probe;
  - calling the wrapper directly without the streaming shim (SM off) must not
    return a matching result: the driver accepts a SIGILL or a non-zero exit
    and never treats that call as a pass;
  - wrong SVL: QEMU property when available; otherwise explicitly omitted and
    documented, with `CNTB` = 64 as the guard;
  - tampered or foreign bundle: rejected by the existing bundle contract;
  - any reference-group fallback: failure.

## 10. Acceptance fixtures and expected evidence

Fixture: `test/Conversion/MicroMapping/Inputs/arm/contraction_f32.mlir`
(kernel `arm_contraction`, three inputs `%a`, `%b`, `%init`, one output).

Expected acceptance:

1. The static `arm-sme` plan is unchanged: `sme.mma_f32`, `matrix_engine = za`,
   `za.0` 4096 B, 1024 B live peak.
2. Selected-target lowering realizes the contraction with
   `selectedGroupsVerified == 1`, `backendGroupsRealized == 1` and
   `referenceGroupsLowered == 0`.
3. The object contains `fmopa`, `zero {za}` and `smstart za`.
4. Native M5: both calls match their literal expected bits, the canary holds,
   SVL is reported as 512, and the required exit code is 0.
5. QEMU SME1: the same numerical result and canary hold, and `sme=off` exits 70.
6. bf16 and i8 contractions refuse as in §9; the capability receipt reports
   f32 contraction selected lowering as supported after a selected compile and
   the bf16 variant as unsupported with its reason.
7. The E10 vector evidence and the AVX2 baseline remain unchanged.

## 11. Implementation sequence

The implementation plan is written separately and owns exact files, tests and
order. Its required shape, test-first per task:

1. Framework: `streamingVectorLengthBits`, `invocationAttributes`, their generic
   application and identity recording, the post-bufferization target hook, and
   the late backend hook that runs after ownership preparation.
2. `arm-sme` package: target wrapper, codegen contract, `SMEEmitter` for
   `sme_mma_za`, and the static-only tests narrowed to the unimplemented subset.
3. SME lowering: `SMEVectorLoweringPass`, `SMEArmLoweringPass` and
   `SMELateLoweringPass`, including initialized-accumulator handling,
   rejections, group accounting and the ownership/control-flow split.
4. Native evidence: Mach-O AOT, wrapper attributes, assembly shim, macOS
   harness and driver, disassembly gate, negative controls, ledger identity.
5. QEMU evidence: linux object, freestanding harness with ABI stubs, CI job,
   `sme=off` and SVL controls (or a documented omission).
6. Capability receipt, documentation corrections and the acceptance ledger,
   including the #175 closure statement.

## 12. Limits and follow-ons

- bf16 and every other `sme_*` signature remain static-only.
- No SME2 (2-way/4-way outer products, ZA array load/store), multi-tile kernels,
  or streaming length other than 512 bits.
- No ZA preservation across unrelated calls, no nested streaming calls, no
  physical ZA spill/restore, no physical SRAM/L2 realization.
- No calibrated timing, measured-metric ranking or persistence; emulator and
  native functional runs do not feed `micro-perf`.
- The native T3 run is manual and pinned by identity; CI proves T2 only.
- Length-agnostic/dynamic-shape SME, CPU-schedule retargeting and additional
  operators remain separate increments. #54 (Micro interpreter) and #51/#52
  (measurement/prediction validation) remain related, not automatic
  dependencies.
- The pre-existing merge corruption in `lib/Target/AArch64/TargetSVE.cpp` is
  unrelated dead code and is not part of this increment.

## References

- E10 design: `docs/superpowers/specs/2026-10-24-arm-sve-executable-backend-design.md`.
- E10 acceptance: `docs/reviews/2026-10-24-arm-sve-executable-acceptance.md`.
- Static Arm acceptance: `docs/reviews/arm-mapping-targets-acceptance.md`.
- LLVM SME ABI and code generation: <https://llvm.org/docs/AArch64SME.html>.
- MLIR ArmSME dialect: <https://mlir.llvm.org/docs/Dialects/ArmSME/>.
- MLIR ArmSME matmul integration test:
  `mlir/test/Integration/Dialect/Linalg/CPU/ArmSME/matmul.mlir`.
- QEMU 8.2 AArch64 emulation: `docs/system/arm/emulation.rst`.
- Arm SME supplement:
  <https://documentation-service.arm.com/static/62015c6c965f7d118e3f5f4c>.
- Apple SME support: <https://github.com/apple-oss-distributions/xnu/blob/main/doc/arm/sme.md>.
