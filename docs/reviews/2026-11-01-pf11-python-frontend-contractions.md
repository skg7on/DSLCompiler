# PF11 verification record: counted contractions and f32 SwiGLU

Date: 2026-11-01
Issue: [#222](https://github.com/skg7on/DSLCompiler/issues/222)
Parent: [#203](https://github.com/skg7on/DSLCompiler/issues/203)
Plan: `docs/superpowers/plans/2026-10-18-python-frontend-refinement.md` §PF11
Implementation commit: `cd4889b41c821d95c40f2b2751293d4b7821cbe8`

## Delivered

A captured counted-K loop with one or two `dot` accumulators is recovered as
`llk.matmul` or `llk.fused_swiglu` only when the tile program is exactly the
whole-tensor contraction. The recovered root is imported into the existing
contraction source family (`llk-contraction-v1`), anchored on a tile derived from
the target machine, and executed through the shared mapped compiler.

`llk.fused_swiglu` now accepts two uniform storage contracts: the original bf16
operands/result with an f32 accumulator, and uniform f32. Mixed storage, a
non-f32 accumulator, and a result whose element type differs from its operands
are verifier errors. The Linalg lowering keeps the f32 form f32 end to end (no
extension or truncation) and keeps the bf16 form's widen/round-once behavior.

## Accepted family

- One `counted_loop` starting at zero and traversing exactly `K` in whole `BK`
  blocks; exactly one output store; no other data writes.
- `llk.matmul` for one dot; `llk.fused_swiglu` for two dots that read the
  identical left-hand tile, two distinct `[K, N]` weights, and exactly
  `silu(gate) * up`.
- Loads are the canonical compact row-major coordinates
  (`row = program_id * block + lane`, `col = k + lane`), proven from the
  `div`/`rem` decomposition of the flat address and its residue bounds.
- Uniform f32 and positive static extents. The launch block equals the output
  tile and `grid * block == shape`.
- Zero accumulators, or (matmul only) a caller-provided `[M, N]` accumulator
  read at the output coordinates. The caller accumulator becomes the public
  native `init` port; a zero accumulator has a compiler-owned, zero-filled
  `init` port.
- `silu` exact maps to `#llk.math_mode<strict>`, native maps to
  `#llk.math_mode<bounded_fast>`. The policy is part of the source contract, the
  emitted `micro.numerical_contract`, the workload identity and the cache key.

## Rejected by name

| Pattern | Code |
|---|---|
| Second writer / additional store | `unsupported_frontend_program` |
| Two different left-hand tiles | `unsupported_frontend_program` |
| Reduction extent other than `K` | `unsupported_frontend_program` |
| Extra reader of a dot input or result | `unsupported_frontend_program` |
| A `dot` without its own loop-carried accumulator | `unsupported_frontend_control_flow` (capture) |
| Any explicit intermediate cast (the bf16 rounding pattern) | `unsupported_numerical_contract` |
| Non-strict-f32 numerical policy | `unsupported_numerical_contract` |
| Fused SwiGLU with a caller accumulator | `unsupported_numerical_contract` |

## Anchor and capacity

`lib/Frontend/Importer.cpp` derives the schedule anchor from the target's
`MachineModel`: the largest rank-three matrix fragment is the instruction shape,
and the tile is the largest divisor of `M x N x K` whose whole-operand
materialization (`(M*K + K*N + M*N + M*N)` per spatial occurrence) fits the
machine's SRAM capacity. The static `pad` boundary policy realizes an extent the
tile does not divide exactly; an extent with no feasible tile fails explicitly
instead of exceeding the machine.

The exported search space is pinned to that anchor. Placement (layout, memory
path, owner hierarchy) is *not* a source choice: those parameters and their
constraints are removed, so the mapping stage solves them exactly as it does for a
directly authored kernel.

## Test evidence

Native (compiler-free fixtures captured by the real PF04 producer):

```sh
cmake --build build --target LLKFrontendTests
build/bin/LLKFrontendTests --gtest_filter='FrontendContractions.*'
```

10 tests pass: four import/contract tests and six execution/negative tests.

| Test | Evidence |
|---|---|
| `RecoversCanonicalMatmulAsOneContractionRoot` | one `llk.matmul`, `llk-contraction-v1`, M/N/K = 32/64/16, init port 2, tile search domain |
| `RecoversFusedSwiGLUOverOneSharedInput` | one `llk.fused_swiglu`, `lhs/rhs0/rhs1/output`, no compiler-owned port |
| `PreservesACallerAccumulatorAsAPublicPort` | public `init` maps to native port 2, read effect |
| `RejectsPatternsOutsideTheDeclaredFamily` | four negative fixtures |
| `ReferenceExecutesCanonicalMatmul` | 32x64x16 and 7x9x5 executed, guarded storage, input bytes unchanged |
| `ReferenceExecutesFusedSwiGLU` | 16x64x16 shared-X fusion executed; `micro.mma` and `"silu"` in MicroIR |
| `ReferencePreservesACallerAccumulator` | `init + A@B` executed, `init` read-only |
| `RelaxedActivationIsADifferentNumericalIdentity` | workload and native-request identities differ; relaxed result executed |
| `NaNPropagatesThroughTheFusedEpilogue` | NaN input yields NaN output; guards intact |
| `SelectedTargetExecutesOrReportsUnavailableHost` | selected AVX2 on a qualified host, else `native_backend_unavailable` |

Dialect:

```sh
ctest -R 'DialectSwiGLUF32|DialectOpsInvalid'
```

`test/Dialect/swiglu_f32.mlir` checks the uniform-f32 lowering (two f32
`linalg.matmul`, `math.exp`, no `arith.extf`/`arith.truncf`) and the unchanged
bf16 widen/round behavior. `test/Dialect/ops_invalid.mlir` rejects mixed
storage, a mismatched result element type and a bf16 accumulator.

Python (FFI lane, no Torch imported):

```sh
LLK_FRONTEND_LIBRARY=build/lib/libLLKFrontendCAPI.so \
    python -m pytest test/python/test_frontend_contractions.py -q
```

12 tests pass, covering the same shapes and negatives through `@llk.jit`, plus
the launch contract (the grid tiles the output domain; input extents may differ).

Full frontend binary: 70/70 pass, including the pre-existing PF05–PF10 tests.

Full local suite:

```sh
ctest --output-on-failure
```

1693 registered, 1689 passed, 4 failed. All four failures are the
`selected-avx2` lanes (`MappedAVX2Acceptance`, `GenericElementwiseAVX2Test`,
`FrameworkPrimitivesAVX2Test`, `DynamicMappedAVX2Acceptance`), whose required
execution asserts an x86_64 AVX2 host; the local host is aarch64. They are CI
lanes, not local evidence. Four legacy JIT tests are skipped and one disabled,
as in the recorded baseline.

The pre-existing `LLKToMicroLoweringInvalid` case that expected the *exporter* to
reject a non-f32 accumulator now expects the dialect's diagnostics: PF11 moved
that rejection into `llk.fused_swiglu`'s verifier, where it applies to every
consumer rather than only the Micro export.

## Limitations (explicit, not silent)

- Uniform f32 only. bf16 captured sources decline because their explicit casts
  are numerical sites the fused f32 epilogue does not realize. The native bf16
  `llk.fused_swiglu` contract is unchanged and tested.
- Static extents only, and only shapes whose anchored `pad` schedule fits the
  target's staging capacity. A shape with no feasible tile is an explicit
  error, not a padded copy that exceeds the machine.
- The automatic SwiGLU rewrite for Torch/Inductor graphs remains PF13 work; no
  Torch fixture or eager comparison is claimed here.
- Dynamic contractions (#191–#199) and measured-performance identity remain
  separate.
