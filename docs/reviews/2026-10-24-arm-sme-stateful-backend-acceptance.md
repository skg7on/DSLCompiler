# Arm SME stateful executable acceptance (E11)

Implements `docs/superpowers/specs/2026-10-24-arm-sme-stateful-backend-design.md`
and `docs/superpowers/plans/2026-10-24-arm-sme-stateful-backend.md`.

**Scope:** executable `arm-sme` f32 16x16x16 initialized contraction through a
native Apple M5 run (T3, gating) and a QEMU linux-user SME1 run (T2, CI).

## Identity

| Item | Value |
|---|---|
| Implementation head | `9c37020` (the PR head whose CI produced the T2 evidence; the native run is `dc2202b`, the TPIDR2/wrapper fixes are `a2fff20` and `8e525c3`) |
| Design head | `1f7959e` (last design-doc revision; initial design `5118131`; merged as `ae22f6b`) |
| Native host | `Mac17,2`, macOS `27.0.1`, `FEAT_SME=1`, `FEAT_SME2=1`, SVL=512 |
| Native harness compiler | `Apple clang version 21.0.0 (clang-2100.3.34.2)` |
| QEMU | `qemu-aarch64 version 8.2.2 (Debian 1:8.2.2+ds-0ubuntu1.18)`, Ubuntu 24.04 x86_64, CI run `40106477994` |
| QEMU CPU | `max,sme=on,sme-default-vector-length=64`; reported SVL 512 |
| CI LLVM/MLIR | 22.1.8, `LLVM_TARGETS_TO_BUILD="X86;AArch64"` |

The native values are copied from the Task 6 evidence record; the T2 values are
from the CI identity sidecar and artifact digests for the green
`ArmSMEExecutionEvidence` run. The driver's stdout record is not persisted on a
passing run (a deferred follow-up); the identity sidecar and the artifact
digests below are the T2 record.

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

QEMU (T2), executed in CI run `40106477994` (head `9c37020`, LLVM 22.1.8):

```text
test: ArmSMEExecutionEvidence (ctest) - passed
runner: qemu-aarch64 version 8.2.2 (Debian 1:8.2.2+ds-0ubuntu1.18)
cpu: max,sme=on,sme-default-vector-length=64
identity: compilerVersion=0.1.0+fcc1334 llvmVersion=22.1.8
  triple=aarch64-unknown-linux-gnu features=[+sme] svl=512
  plan=0xa5cae39089a17f3e graph=0x2781069dae8bbfd9 abi=0x4007b5fdb8b3e09b
  machine=0xa61b18f698708108 target=0x1eccb8f7ec38a447
  selected-groups=6 reference-groups=0 backend-groups-realized=1
  invocationAttributes=["aarch64_pstate_sm_enabled"]
artifacts:
  kernel.o  sha256 26d939caec6f00964fd3890f0b7084d63b08929ab4904863b6298731d96ae345
  harness   sha256 530b59b921108e0e87a5b3c6a6b9c51c7d068403ed83a1e58e22118c9d729287
  shim.o    sha256 1d5a477860f8cefb06bb055061c57b578bba2ff17faaee594a87952edf40e24d
  harness.c sha256 ce250609f1caf9faa74f9b8d1e9decbd5f35acec3a65e60ac871e314d54f5210
  shim.S    sha256 6e8ea4432a82885030783f7be844743434b0709cf981e64a5a5cbaf34e909e18
```

`compilerVersion` is the CI build's version string, not the PR head; the head is
recorded in the identity table. A passing test means the driver's positive run
exited 0 (two exact integer-valued f32 calls with different inputs and
initializers, canary intact), every control produced its required outcome, and
the disassembly gate found `fmopa`, `zero {za}` and `smstart za`. The driver
prints one `arm sme qemu evidence:` line, including the ZA-probe note, but does
not persist it on success (a deferred follow-up); the identity sidecar and the
digests above are the T2 record. The ZA probe outcome is either SIGILL or the
explicitly recorded `not-modeled-by-qemu` omission; the driver fails on any
other result.

`selectedGroupsVerified` covers the contraction group plus the fixture's
movement groups (the staged DRAM-to-SRAM and SRAM-to-accumulator copies and the
result copy), which the emitter claims and the generic Micro-to-Linalg bridge
carries. `backendGroupsRealized` counts the contraction group the ArmSME pass
realized; `referenceGroupsLowered` is zero. The two numbers are not expected to
be equal.

## Disassembly

The object contains `fmopa`, `zero {za}` and `smstart za`. On the native Mach-O
object `llvm-objdump -d --mattr=+sme` shows `zero {za}` (0x40), `smstart za`
(0x44) and `fmopa za0.s, p1/m, p2/m, z1.s, z2.s` (0x180). The CI ELF object
passes the same gate with `aarch64-linux-gnu-objdump`. A non-SME object fails
the evidence test; it is not skipped.

## Negative controls

| Control | Required | Native | QEMU |
|---|---|---|---|
| bf16 contraction selected compile | refuses; no reference fallback | pass — `ArmSMEBackend.SelectedBf16ContractionIsRefused` and the receipt's `unsupported` before any compile | pass — the same GTest ran in the CI full suite |
| i8 contraction mapping | retains no plan | pass — `ArmDtypeCoverage.UnsupportedSignaturesRetainNoPlan` (no plan, not a search cap) | pass — the same GTest ran in the CI full suite |
| missing SME | exit 70 | the generated native harness implements the check and exit 70 (`host_has_sme()` reads `sysctl hw.optional.arm.FEAT_SME`); it is not exercised on this `FEAT_SME=1` host, and CMake registers the test only when `sysctl hw.optional.arm.FEAT_SME` is 1 | pass — the driver requires exit 70 under `sme=off`; a different code fails the test |
| `smstart sm` then `zero {za}` | SIGILL | pass — `za-probe=SIGILL` (manual `za_probe` exit 132) | pass — the driver requires SIGILL or a recorded `not-modeled-by-qemu`; exit 0 fails |
| wrapper without the streaming shim | never a matching pass | pass — exit 132 (SIGILL) | pass — the driver fails if the direct-call run exits 0 |
| wrong SVL | exit 71 | fixed at 512 and checked with `CNTB`; no wrong-SVL variant exists on this host | pass — the driver requires exit 71 under `sme-default-vector-length=32` |
| reference-group fallback | zero groups | pass — `reference-groups=0` | pass — the identity records `referenceGroupsLowered=0` and the driver rejects a fallback |

## Claims and limits

- T3 is native functional evidence on the recorded M5; T2 is the emulated
  functional evidence from the green `ArmSMEExecutionEvidence` run. Neither
  calibrates cycles, bytes, storage or timing; `micro-perf` numbers remain
  modeled predictions.
- The native T3 run is not reproducible in CI; CI proves T2. The T3 record is
  pinned by the host, toolchain and artifact digests above, and the T2 record by
  the identity sidecar and digests above.
- The QEMU driver prints but does not persist its evidence line on success; the
  identity sidecar and artifact digests are the record. Persisting the line and
  disassembly is a deferred minor follow-up, not a gap in the accepted evidence.
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
