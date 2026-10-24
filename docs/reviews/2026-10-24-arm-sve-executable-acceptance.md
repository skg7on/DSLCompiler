# Arm SVE executable subset acceptance (E10)

This ledger records the first executable Arm increment, E10 (#252), from actual
CI and local artifacts. It implements the approved E9 design and plan
(`docs/superpowers/specs/2026-10-24-arm-sve-executable-backend-design.md`,
`docs/superpowers/plans/2026-10-24-arm-sve-executable-backend.md`) and extends
the static acceptance in `docs/reviews/arm-mapping-targets-acceptance.md`.

**Scope:** executable `arm-sve` f32 `micro.vector "add"` and `"mul"` on
directly authored rank-two MicroIR fixtures, emitted as a cross-lowered
AArch64 object and executed under QEMU linux-user at a fixed 512-bit vector
length. It establishes **emulated** selected-target numerical execution only.

## Revision and identity

| Item | Value |
|---|---|
| Implementation head | `4e529ae` (PR #255, `feat/arm-sve-executable-backend`) |
| CI run | `40079860609` (`build-and-test` job `120830966861`, head `4e529ae`, conclusion success) |
| CI OS / host | Ubuntu 24.04, x86_64 |
| CI compiler | GNU 13.3.0 (`/usr/bin/cc`, `/usr/bin/c++`) |
| CI LLVM/MLIR | 22.1.8, built from source with `LLVM_TARGETS_TO_BUILD="X86;AArch64"` |
| Compiler identity in artifacts | `0.1.0+71dec3d` (the `LLK_COMPILER_VERSION` reported by the run) |
| Runner | `qemu-aarch64 version 8.2.2 (Debian 1:8.2.2+ds-0ubuntu1.18)` |
| Emulated CPU / VL | `-cpu max,sve-max-vq=4`, reported VL = 512 bits |
| Harness toolchain | `aarch64-linux-gnu-gcc`, static, `-nostdlib -ffreestanding -fno-builtin -fno-stack-protector` |

The first CI run on this branch failed the evidence step because the C `_start`
prologue ran before the inline stack-pointer read, so the harness misparsed
auxv and exited 70. Commit `4e529ae` replaces the entry with an assembly
`_start` that moves the initial `sp` into `x0` and branches to a C
`harness_main`; this ledger records the successful run after that fix.

## Full registered suite

CI, head `4e529ae`, LLVM 22.1.8:

- 1,486 registered tests; 1,485 counted by CTest; **0 failed**.
- 1 disabled, not run: `TritonStagedExecution.SwiGLUCorrectness` (entry 179).
- 3 skipped, each a pre-existing legacy JIT case: `SwiGLUScalar.JitCompilationSmoke`
  (176), `SwiGLUScalar.E2EWithAbiWrapper` (178), `SwiGLUVectorAVX2.Correctness`
  (181).
- `ArmSVEExecutionEvidence` passed both in its dedicated workflow step and in
  the full CTest run. The required selected-AVX2 acceptance step passed.
- The Sphinx HTML workflow, the LLVM-22 analysis workflow and the coverage
  workflow completed successfully on the same head.

Local macOS arm64 (developer host, LLVM 24.0.0git), final head:

- 1,478 counted tests; 3 failed — `MappedAVX2Acceptance`,
  `GenericElementwiseAVX2Test`, `FrameworkPrimitivesAVX2Test` — all labeled
  `selected-avx2` and all the documented unsupported-host refusal on this
  aarch64 host.
- 6 skipped (the three legacy JIT cases and three selected-AVX2 host cases) and
  1 disabled. The qemu evidence test is not registered locally because
  `LLK_BUILD_ARM_EVIDENCE` is off and this host has no qemu or AArch64 Linux
  cross toolchain.
- The Sphinx manual builds with `-n -W --keep-going`.

## Accepted executable domain

| Item | Accepted value |
|---|---|
| Target | `arm-sve` only; `arm-sme` remains static-only |
| Operations | `micro.vector "add"` and `"mul"` |
| Element type | f32 |
| Shapes | rank-two, static; 8×8 (partial row, masked tail) and 16×16 (full VL row) |
| Lowered form | scalable `vector<8x[4]xf32>` / `vector<16x[4]xf32>` with masks; `vscale*4` tiling; per-row rank-one transfers |
| Vector length | exactly 512 bits required; refused otherwise |
| ABI | `_mlir_ciface_<kernel>`, inputs then borrowed outputs, rank-two row-major descriptors |

Everything else keeps refusing: bf16 and i8 add/multiply, f32 SiLU, reduction,
conversion, contraction, the fused epilogue and `arm-sme`. Reference-mode
bf16/i8 compiles remain portable; selected-target compiles of them fail through
the shared `realized no selected compute groups` contract rather than falling
back.

## Per-case evidence

CI artifacts (`selected-avx2-execution-evidence`, including
`test/ArmSVEExecutionEvidence/<case>/{kernel.o,harness,harness.c,identity.json}`
and `arm-sve-evidence.log`):

| Case | Wrapper | Plan id | Source graph hash | ABI hash | Selected / reference groups |
|---|---|---|---|---|---|
| `vector_add_f32` | `_mlir_ciface_arm_vector_add_f32` | `0x76fa9b3ca0bfd88a` | `0xe9cfcb3369d5a696` | `0xd67412592339f9d4` | 1 / 0 |
| `vector_mul_f32` | `_mlir_ciface_arm_vector_mul_f32` | `0xd1aae3428af2057` | `0x437a006612397a4c` | `0xd67412592339f9d4` | 1 / 0 |
| `vector_add_f32_full` | `_mlir_ciface_arm_vector_add_f32_full` | `0x76fa9b3ca0bfd88a` | `0x8c986565ab7f9ad6` | `0x54461be6c8375cd4` | 1 / 0 |
| `vector_mul_f32_full` | `_mlir_ciface_arm_vector_mul_f32_full` | `0xd1aae3428af2057` | `0x866d8fdf366fa88c` | `0x54461be6c8375cd4` | 1 / 0 |

Each identity record also names the compiler version, LLVM 22.1.8, triple
`aarch64-unknown-linux-gnu`, `targetFeatures = ["+sve"]`, `machineHash`
`0xa46ec655e0002e62`, `targetHash` `0x99d58065f6794169`, `abiVersion = 1` and
`backendGroupsRealized = 1`. The add/multiply plan ids coincide across widths
by design: plan ids hash decisions, not tile shapes, while the source graph
hash distinguishes the two shapes.

Artifact digests from the run:

| Case | Object sha256 | Harness sha256 |
|---|---|---|
| `vector_add_f32` | `6db31900029099deaa8f13777e65c957295b81f842e62ca386642d0c7f228013` | `720d9dbecd3ab15178418aa1d5d1682f4e7fc3d474e9cbce9ad6b600feb64518` |
| `vector_mul_f32` | `bff1e255018c49fedc5a941e8f1ab611f49571ba1d85b38cfda69f845e15682e` | `8aab1f6b0c09dac60296ffb00c333207d29e1b2f971744b572b2929f57c5496b` |
| `vector_add_f32_full` | `2abac224d57144d0423a484e696309d48e01ca2abd7d65a0ee42418c8c372cbe` | `6dff72f524ff47435d304cd386474f5ec0e1603af6fa656aa37d1d3b12013f57` |
| `vector_mul_f32_full` | `0883f42bd7c108d773196cc5a772a0164cfc77bb42c48b3f975c679730371c00` | `8c93af57342e123358d3c4e439449e305d9a46f345e8762d4283c0a91a7e061b` |

## SVE code generation

The driver disassembles every object with `aarch64-linux-gnu-objdump` before it
runs and requires both a scalable `zN.<b|h|s|d>` operand and an SVE-only
mnemonic. The uploaded `vector_add_f32/kernel.o` disassembles to, among 32
matching instructions:

```
ptrue   p0.s
ld1w    { z1.s }, p1/z, [x19, x9, lsl #2]
ld1w    { z2.s }, p1/z, [x22, x9, lsl #2]
fadd    z1.s, z1.s, z2.s
```

The object is an `EM_AARCH64` ELF and exports the `_mlir_ciface_*` wrapper. A
NEON-only object cannot pass the check.

## Emulated numerical execution

Under `qemu-aarch64 -cpu max,sve-max-vq=4` all four harnesses exited 0 and
printed `pass`:

- Inputs and expected results are literal f32 bit patterns; comparison is exact
  bits, with NaN compared by classification and signed zero by bits.
- The first four elements of both operands are `+0`, `-0`, `+inf` and a quiet
  NaN; the 8×8 cases additionally exercise the masked partial-row path.
- Outputs are caller-owned; a 32-word sentinel tail after the logical result
  must remain untouched.
- The harness verifies HWCAP SVE and `PR_SVE_GET_VL == 512` bits (bytes × 8)
  before calling the kernel.

Negative controls in the same run:

| Control | Required | Result |
|---|---|---|
| `-cpu max,sve=off` | exit 70 (no SVE); any other code fails the test | the test passed |
| `-cpu max,sve-default-vector-length=128` | exit 71 (VL ≠ 512); any other code fails the test | the test passed |

The harness prints its refusal reason and the driver records any non-required
exit code as a case failure, so a passing `ArmSVEExecutionEvidence` is the
observed evidence that both controls produced their required codes. A missing
qemu, cross toolchain or disassembler fails the test; nothing skips.

## Static, AOT and capability evidence

- `llk-compile --emit=object --target-triple=aarch64-unknown-linux-gnu
  --target-features=+sve --output=… --emit-identity=…` emits the object and
  the identity record above; AOT flags are rejected on other `--emit` modes,
  an invalid `--mapping-stop` is a diagnostic, identical output paths are
  rejected, and write failures are reported.
- The capability receipt draws the executable boundary from artifacts: a
  statically mapped f32 add reads `not_checked` before compilation, `supported`
  after a selected compile with reference lowering still `not_checked`, and
  SiLU reads `unsupported` with its plan-derived reason.
- The shared analysis and `micro-perf` parity from E1–E8 are unchanged and not
  reclassified by this increment.

## Reproduction

```sh
# LLVM must be built with the AArch64 target for cross emission.
cmake -S . -B build -DLLVM_PROJECT_BUILD_DIR=<llvm build> \
      -DLLK_BUILD_TOOLS=ON -DLLK_BUILD_ARM_EVIDENCE=ON
ninja -C build llk-compile LLKMappingTests
ctest --test-dir build -R '^ArmSVEExecutionEvidence$' --output-on-failure
ninja -C build check-llk
```

The evidence test requires `qemu-aarch64` and `aarch64-linux-gnu-gcc`; it fails
rather than skips when either is missing.

## Claims and limits

- This increment delivers **emulated** selected-target numerical execution for
  the f32 add/multiply subset. It does not deliver native SVE execution: Apple
  Silicon exposes SME/streaming SVE but not non-streaming SVE, and the CI runner
  is x86_64. Cross-lowered artifacts plus QEMU are the strongest evidence
  available in this fleet.
- E11 (#253) supersedes the SME half of this boundary: `arm-sme` now executes
  the f32 16x16x16 initialized contraction on a native Apple M5 (T3) and under
  QEMU SME1 (T2). bf16/i8 SME execution, ZA preservation across calls, SME2 and
  calibrated timing remain out of scope. No ZA realization beyond the one
  contraction tile, no non-SME signature execution and no CPU-schedule
  retargeting is claimed. This ledger's SVE evidence is unchanged.
- Emulated functional results never calibrate native timing; `micro-perf`
  numbers remain modeled predictions.
- The modeled SRAM/accumulator identities remain analysis facts; the executable
  path covers the ABI boundary buffers, not a physical SRAM realization.
- The AVX2 selected-versus-reference acceptance remains the executable baseline
  and is unchanged.
- #175 closes with E11's ledger
  (`docs/reviews/2026-10-24-arm-sme-stateful-backend-acceptance.md`), which
  records the native T3 run and the QEMU T2 run.
