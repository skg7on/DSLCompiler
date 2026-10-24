# Arm SME stateful executable acceptance (E11)

Implements `docs/superpowers/specs/2026-10-24-arm-sme-stateful-backend-design.md`
and `docs/superpowers/plans/2026-10-24-arm-sme-stateful-backend.md`.

**Scope:** executable `arm-sme` f32 16x16x16 initialized contraction through a
native Apple M5 run (T3, gating) and an intended QEMU linux-user SME1 run
(T2, CI, pending).

## Identity

| Item | Value |
|---|---|
| Implementation head | `a7663c5` (the tree that produced the evidence; the native run is `dc2202b`, the cross-emission and QEMU driver is `a7663c5`) |
| Design head | `1f7959e` (last design-doc revision; initial design `5118131`) |
| Native host | `Mac17,2`, macOS `27.0.1`, `FEAT_SME=1`, `FEAT_SME2=1`, SVL=512 |
| Native harness compiler | `Apple clang version 21.0.0 (clang-2100.3.34.2)` |
| QEMU | Not observed: no QEMU record exists yet. This macOS host has no `qemu-aarch64` and no AArch64 Linux cross toolchain, and the stacked branch receives no CI until the design PR merges and the PR is retargeted to `main`. |
| QEMU CPU | `max,sme=on,sme-default-vector-length=64` (the property string the registered driver passes; unexecuted) |
| CI LLVM/MLIR | 22.1.8, `LLVM_TARGETS_TO_BUILD="X86;AArch64"` |

The native values are copied from the Task 6 evidence record; the design head is
the last commit that revised the design document. No QEMU version, exit code or
digest is recorded here because the T2 run has not executed.

## Accepted executable domain

| Item | Value |
|---|---|
| Target | `arm-sme` |
| Operation | `micro.mma` 16x16x16, f32 inputs and accumulator, initialized from `%init` |
| Streaming | `aarch64_pstate_sm_enabled` kernel and wrapper; the wrapper's inherited ZA ownership is removed so it carries only the declared invocation attributes; assembly `smstart sm`/`smstop sm` shim; the shim clears `TPIDR2_EL0` before each kernel call |
| ZA | `aarch64_new_za`: `smstart za`, `zero {za}`, slice extraction, `smstop za`; not preserved across calls. `TPIDR2_EL0` is zero on the native runner and is explicitly cleared on QEMU/Linux, where it is not |
| ABI | `_mlir_ciface_<kernel>`, inputs then borrowed outputs, rank-two descriptors |
| SVL | exactly 512 bits; `CNTB` = 64 bytes checked before the first call |

## Evidence records

Native (T3), copied verbatim from the Task 6 evidence record
(`ctest -V -R '^ArmSMENativeEvidence$'`, commit `dc2202b`):

```text
arm sme native evidence: host=Mac17,2 macos=27.0.1
  clang='Apple clang version 21.0.0 (clang-2100.3.34.2)'
  wrapper=_mlir_ciface_arm_contraction plan=0xa5cae39089a17f3e
  graph=0x2781069dae8bbfd9 abi=0x4007b5fdb8b3e09b svl=512
  selected-groups=6 reference-groups=0
  object-sha256=946b59617d0f1f20acc9d7b28639bbd8009f6bcc2374b9c662578e561882e020
  harness-sha256=cacfb6ee1b9e08e034fbc6db448437b510dda6537720036bf5121ad58473a749
  shim-sha256=7b10b068de0910008d48805453f04a3a87d6b35ee18aeaf3e33b636582316881
  za-probe=SIGILL
arm sme native evidence: ok
```

QEMU (T2): **pending CI; no record produced.** The registered test is
`ArmSMEExecutionEvidence`, which requires `-DLLK_BUILD_ARM_EVIDENCE=ON` and the
cross toolchain; its exact command is

```sh
ctest --test-dir build -R '^ArmSMEExecutionEvidence$' --output-on-failure
```

The driver is `test/Execution/arm_sme_qemu_evidence.py`. On success it prints one
`arm sme qemu evidence:` record with the same identity fields and the
object/harness/shim digests. Until that run happens this row is open, and no
document here claims a QEMU result.

`selectedGroupsVerified` covers the contraction group plus the fixture's
movement groups (the staged DRAM-to-SRAM and SRAM-to-accumulator copies and the
result copy), which the emitter claims and the generic Micro-to-Linalg bridge
carries. `backendGroupsRealized` counts the contraction group the ArmSME pass
realized; `referenceGroupsLowered` is zero. The two numbers are not expected to
be equal.

## Disassembly

The object contains `fmopa`, `zero {za}` and `smstart za`. On the native Mach-O
object `llvm-objdump -d --mattr=+sme` shows `zero {za}` (0x40), `smstart za`
(0x44) and `fmopa za0.s, p1/m, p2/m, z1.s, z2.s` (0x180). A non-SME object
fails the evidence test; it is not skipped.

## Negative controls

| Control | Required | Native | QEMU |
|---|---|---|---|
| bf16 contraction selected compile | refuses; no reference fallback | pass — `ArmSMEBackend.SelectedBf16ContractionIsRefused` and the receipt's `unsupported` before any compile | pending CI |
| i8 contraction mapping | retains no plan | pass — `ArmDtypeCoverage.UnsupportedSignaturesRetainNoPlan` (no plan, not a search cap) | pending CI |
| missing SME | exit 70 | the generated native harness implements the check and exit 70 (`host_has_sme()` reads `sysctl hw.optional.arm.FEAT_SME`); it is not exercised on this `FEAT_SME=1` host, and CMake registers the test only when `sysctl hw.optional.arm.FEAT_SME` is 1 | pending CI (`sme=off`, exit 70 expected) |
| `smstart sm` then `zero {za}` | SIGILL | pass — `za-probe=SIGILL` (manual `za_probe` exit 132) | pending CI (`SIGILL` or `not-modeled-by-qemu` recorded) |
| wrapper without the streaming shim | never a matching pass | pass — exit 132 (SIGILL) | pending CI |
| wrong SVL | exit 71 | fixed at 512 and checked with `CNTB`; no wrong-SVL variant exists on this host | pending CI (`sme-default-vector-length=32`, exit 71 expected) |
| reference-group fallback | zero groups | pass — `reference-groups=0` | pending CI |

## Claims and limits

- T3 is native functional evidence on the recorded M5; T2 is the intended
  emulated functional evidence, recorded once the T2 run executes. Neither
  calibrates cycles, bytes, storage or timing; `micro-perf` numbers remain
  modeled predictions.
- The native T3 run is not reproducible in CI. CI is set up to prove T2; the
  first CI run of `ArmSMEExecutionEvidence` is the T2 evidence and is not yet
  recorded. The T3 record is pinned by the host, toolchain and artifact digests
  above.
- No bf16/i8 SME execution, no SME2, no multi-tile kernels, no ZA preservation
  across calls, no physical SRAM/L2/DMA realization, no calibrated timing.
- The JIT path refuses selected `arm-sme` on hosts without an SVL probe
  (`checkHostRequirements` never assumes SVL=512). On macOS,
  `llvm::sys::getHostCPUFeatures()` also does not report `sme`, so the JIT
  route refuses with "selected target requires host feature 'sme'" even on an
  SME-capable M5; the AOT path does not consult the host check. The Python
  frontend's selected-target check uses the no-vector-length overload, so it
  also refuses `arm-sve` and `arm-sme` even on capable Linux hardware; that is
  fail-closed and is left to the later frontend integration increment. The AOT
  object plus the native/QEMU harness is the executable route.
- The streaming ABI shims (`__arm_sc_memcpy`, `__arm_sc_memmove`) are
  runner-provided byte copies because Apple's SDK exports neither; they are not
  claimed as the platform compiler-rt implementation.
- The E10 SVE evidence and the AVX2 baseline are unchanged.
- #175 closes with this ledger: a non-AVX2 selected-target backend has
  reproducible numerical execution evidence, including initialized
  accumulation, a documented runner identity and explicit omissions.
