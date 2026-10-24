# Arm SVE/SME static mapping acceptance

This ledger records the bounded static-mapping increment for issue #175 as
delivered by slices E1–E7 (issues #176–#182). It is written from actual
artifacts: the merged source head, the test binaries' own output, the versioned
plan reports and the registered CTest run. It is not a design prediction.

**Scope:** `arm-sve` and `arm-sme` as static mapping/analysis targets at modeled
VL=SVL=512, mapped from directly authored MicroIR fixtures. This increment
delivers no executable SVE/SME backend, no physical ZA realization, no
calibrated timing and no automatic retargeting of the CPU adapter's schedules.
Static completion is not executable completion.

## Revision and identity

| Item | Value |
|---|---|
| Source head | `95d35d7` (merge of PR #249) |
| E1 selected vector contraction analysis | `b0d01f1` (PR #236, issue #176) |
| E2 SVE/SME machine profiles | `beb27e7` (PR #237, issue #177) |
| E3 typed ARM LLKMap policy | `2fd65c3` (PR #239, issue #178) |
| E4 package registration | `7bacc57` (PR #240, issue #179) |
| E5 contraction placement and storage | `d957b4a` (PR #245, issue #180) |
| E6 dtype coverage and rejection | `9cdb3b1` (PR #248, issue #181) |
| E7 CLI replay and analysis parity | `95d35d7` (PR #249, issue #182) |
| CI run | `40056509174`, `build-and-test` job `120760562038`, head `cdc344a` |
| CI LLVM/MLIR | 22.1.8, built from source on Ubuntu 24.04 |
| Local LLVM/MLIR | 24.0.0git (`/Users/skg7on/Workspace/Projects/llvm-project/build`) |
| Local compiler | Homebrew clang 20.1.8, target `arm64-apple-darwin27.0.0` |
| Local CMake / host | 4.2.3 / macOS 27.0.1, arm64 |

The E1–E4 commits landed before this ledger's slices; their tests are included
because the contract matrix below depends on them.

## Full registered suite

### CI, LLVM 22.1.8 (required x86_64 runner)

Run `40056509174` (`build-and-test`, head `cdc344a`, the E7 branch that merged
as `95d35d7`):

- 1,459 registered tests; 1,458 passed; 0 failed.
- 1 disabled, not run: `TritonStagedExecution.SwiGLUCorrectness` (CTest entry
  177, a Triton staging test unrelated to issue #177).
- 3 skipped, each a pre-existing legacy JIT case (CTest entries 174, 176 and
  179, which are not the E1/E2 issue numbers):
  `SwiGLUScalar.JitCompilationSmoke`,
  `SwiGLUScalar.E2EWithAbiWrapper`,
  `SwiGLUVectorAVX2.Correctness`.
- `ArmTargetPipeline` ran and passed. No ARM test skipped for host capability:
  the ARM path is static, and its fixtures are read and mapped on any host.

### Local, macOS arm64

Implementation head `95d35d7`; this documentation-only change adds no code.
Configured with `-DBUILD_TESTING=ON -DLLK_BUILD_E2E_TESTS=ON`, built with
`ninja -C build` (every target), then ran `ctest --output-on-failure`:

- 1,459 registered tests; 1,458 counted by CTest; 3 failed; 1 disabled; 6
  skipped.
- All 3 failures carry the `selected-avx2` label and are the documented
  unsupported-host limitation of this aarch64 host, not ARM evidence and not a
  regression from this increment:
  - `MappedAVX2Acceptance` (#206): three cases fail with `selected target
    requires architecture 'x86_64' but host architecture is 'aarch64'`;
  - `GenericElementwiseAVX2Test` (#212): its execution case fails with
    `required selected AVX2 execution needs an x86_64 AVX2 host`; its lowering
    case passes;
  - `FrameworkPrimitivesAVX2Test` (#214): the same required-host failure.
- Skipped: `SwiGLUScalar.JitCompilationSmoke`, `SwiGLUScalar.E2EWithAbiWrapper`
  and `SwiGLUVectorAVX2.Correctness` (pre-existing legacy JIT cases);
  `MappedAcceptance.SelectedAvx2ExecutesThePaddedTailFixtures`,
  `MappedAcceptance.SelectedAvx2BackendExecutesNumerically` and the
  `MappedAcceptance` wrapper (selected AVX2 host capability).
- Disabled: `TritonStagedExecution.SwiGLUCorrectness`.
- Every Arm test passed on this host: `ArmPolicy.*`, `ArmTargets.*`,
  `ArmPlacement.*`, `ArmDtypeCoverage.*`, `ArmFeasibility.*`,
  `ArmComputeIdentity.*` and `ArmTargetPipeline`. None skipped for host
  capability.

`ninja -C build check-llk` alone does not build
`GenericElementwise{,AVX2}Test` or `FrameworkPrimitives{,AVX2}Test` in this
tree, so it reports those four as `Not Run`; the counts above come from the
complete `ninja -C build` followed by CTest, which is what CI does. That build
gap is pre-existing and independent of the Arm packages.

The required green run is the CI one above: the same registered suite on the
supported x86_64 runner passes with the selected-AVX2 controls and the static
ARM tests active.

## Contract matrix: E1–E7

Each row names the shipped behavior and the tests that establish it. Test names
are the GTest cases registered by `test/Mapping/CMakeLists.txt`, the Perf and
Machine categories, or the CTest `ArmTargetPipeline` entry.

| Slice | Contract | Owning evidence |
|---|---|---|
| E1 (#176) | Shared contraction analysis reads the *selected* compute resource. A vector-only machine charges the selected vector engine and its accumulator lanes; a matrix selection keeps matrix identity; missing, conflicting, unknown or foreign-executor engines are rejected; masked tails satisfy vector lane divisibility; unsupported dtype pairs and overflowing products are errors. | `test/Perf/l1_resource_dag.cpp`: `ContractionSupport.AcceptsSupportedVectorAndMatrixPairs`, `ContractionSupport.RejectsUnsupportedKindsAndDtypePairs`, `SelectedContraction.*` (identity, throughput, fallback, selection conflicts, accumulator lanes, overflow, shared engine pool, selected/materialized agreement); `test/Perf/legality.cpp`: `Legality.VectorOnlyMachine*` masked/unmasked tails |
| E2 (#177) | Two bounded v2 profiles: SVE has a 512-bit vector engine, no matrix engine, `acc.0` (2,048 bytes) and no separate register file; SME adds `za_eng.0`, a modeled 16×16×16 `za` matrix engine and `za.0` (4,096 bytes). VL and SVL are both 512 as an explicit experimental control; every cost is modeled. | `test/Machine/machine_model_loader.cpp`: `MachineModelLoader.LoadsArmSve`, `MachineModelLoader.LoadsArmSme` |
| E3 (#178) | Typed per-dtype rules and one finite layout domain per kind. Blocked layouts solve VW ∈ {16, 32, 64}; typed predicates admit f32/bf16/i8 add and multiply, bf16↔f32 conversion, f32 SiLU and reduce sum, f32/bf16 contraction, and the fused bf16→f32→SiLU→f32-multiply epilogue; unsupported signatures match no rule. | `test/Mapping/arm_policy.cpp`: `ArmPolicy.LoadsBothTargetsWithExactlyTheDeclaredEmitters`, `DeclaresTheSpecifiedRuleIdsAndCosts`, `EachTargetDeclaresOneLayoutPerKind`, `BlockedLayoutSolvesOneWidthPerDtype`, `RowMajorLayoutSolvesWithoutParameters`, `TypedPredicatesAcceptOnlyTheSupportedSignatures`, `TypedPredicatesRejectUnsupportedSignatures`, `FusedGraphKeepsPerPortWidths`, `ContractionRequirementsNameStorageAndCompute` |
| E4 (#179) | Both packages register through `registerAllMappingTargets`, load the conventional machine/layout/rule paths, honor a machine override, and reject missing configuration and malformed bundles. Every rule bundle has a verifying emitter; the ten declared keys per target match policy. | `test/Mapping/arm_targets.cpp`: `ArmTargets.RegistryLoadsBoth`, `RegistryUsesMachineOverride`, `MissingMachineOverrideReturnsError`, `MissingConfigurationReturnsError`, `EveryRuleBundleHasAVerifyingEmitter`, `DeclaredKeysMatchPolicyAndHaveEmitters`, `RejectsUnknownCrossKeyAndMalformedBundles`, `NoExecutableBackend` |
| E5 (#180) | One identical f32 source maps to different concrete engines on the two targets; the bound plan carries initialized accumulator storage and realized movement; selected-target compilation is refused; a failed fixture propagates its error rather than yielding a plan. | `test/Mapping/arm_placement.cpp`: `ArmPlacement.SameTypedContractionSelectsDifferentEngines`, `SelectedTargetCompilationRejectsMissingIsa`, `FailedFixturePropagatesError`, `SelectedBundlesVerifyButCannotLower`; fixtures `test/Conversion/MicroMapping/Inputs/arm/contraction_{f32,bf16}.mlir` |
| E6 (#181) | The advertised dtype table maps completely on both targets; each conversion direction, f32 SiLU/reduce and the fused epilogue produce complete plans; blocked widths stay per operand (f32 16, bf16 32, i8 64); unsupported signatures and infeasible or tampered plans retain no plan. | `test/Mapping/arm_placement.cpp`: `ArmDtypeCoverage.TypedSignaturesMapCompletely`, `FusedEpilogueSolvesEachPortAtItsOwnWidth`, `UnsupportedSignaturesRetainNoPlan`, `MissingLaneWidthRetainsNoPlan`, `ArmFeasibility.VariantsRejectForTheirChangedFact`, `ArmComputeIdentity.TamperedSelectionIsNeverReplaced` |
| E7 (#182) | Both target names and both contraction dtypes run through the public `--micro-map`/`--micro-verify-mapping`/`--micro-bind-plan` flags; reports carry concrete identities and honest limits; a fresh process replays the reported plan id; `micro-perf` agrees with the planner. | CTest `ArmTargetPipeline` (`test/Conversion/MicroMapping/arm_target_pipeline.py`) |

## Accepted domain

Directly authored MicroIR only. The CPU adapter's matmul/SwiGLU exports are not
accepted ARM workloads: their tile sizes, owners, storage and overlap choices
come from AVX2 schedules, and this increment does not retarget them.

| Family | Accepted static signatures |
|---|---|
| Contraction | M=N=K=16, f32 initialization and output; f32×f32→f32 and bf16×bf16→f32 |
| Elementwise | add and multiply for f32, bf16 and i8; both operands and result share one dtype |
| Conversion | bf16→f32 and f32→bf16 |
| Activation / reduction | f32 SiLU; f32 reduce sum |
| Fused graph | bf16→f32 conversion, f32 SiLU, f32 multiply by an f32 gate |
| Movement | typed async copy (`dram`→`sram`, `sram`→`acc`, `acc`→`sram`), tile copy, tile store |
| Layout widths | blocked VW ∈ {16, 32, 64}, solved per port dtype: f32 16, bf16 32, i8 64 |

The contraction fixture is identical for both targets: it stages A/B/init
DRAM→SRAM, copies init SRAM→acc (initialized accumulation), computes one MMA,
copies the result acc→SRAM and yields it. It names no concrete engine id, uses
rank-two row-major tiles, generic `worker` ownership, no spatial replication and
no overlapping pipeline. It is a bounded conformance program, not a tuned GEMM.

Selected identities on that fixture:

| Target | Rule (f32 / bf16) | Emitter | Compute identity | Accumulator memory | Capacity | Live peak |
|---|---|---|---|---|---|---|
| `arm-sve` | `sve.mma_f32` / `sve.mma_bf16` | `sve_mma` | `vector_engine = svp` | `acc.0` | 2,048 B | 1,024 B |
| `arm-sme` | `sme.mma_f32` / `sme.mma_bf16` | `sme_mma_za` | `matrix_engine = za` | `za.0` | 4,096 B | 1,024 B |

Both targets declare ten emitter keys:
`<prefix>_vector_add`, `_vector_convert`, `_vector_silu`, `_vector_mul`,
`_reduce`, `_copy`, `_tile_copy`, `_tile_store`, `_fused_convert_silu_mul`, and
`sve_mma` or `sme_mma_za`. Bundles name no executable instruction.

## Negative controls

Every rejection below is a complete-plan decision made before top-K retention;
the E6 negatives search with `topK=1` and assert `searchTruncated` is false, so
nothing survived a cap and nothing was hidden by one.

| Control | Observed result |
|---|---|
| i8 contraction, bf16 activation, f64 vector signature | Retained no complete plan; f64 is not a representable tile element type, so that control runs at the workload-graph layer where the advertised table lives |
| Temporary profile variant: accumulator capacity 512 | Both initialized 1,024-byte contraction dtypes rejected for capacity |
| Temporary profile variant: accumulator memory and its links removed | Refused by target verification (structurally valid YAML) |
| Temporary profile variant: `sram_to_acc.0` removed | Initialized movement has no legal hop; no complete plan |
| Fixture requesting an unmodeled memory kind (`rf`) | Rejected; neither profile models that kind |
| Profile without an f32 lane fact | Layout solve fails instead of choosing a width |
| Tampered selected compute: unknown id, vector+matrix conflict, engine on another executor | Complete-plan rejection on a copy; the untouched plan is still accepted, so no replacement engine is inferred |
| Unknown cross key, malformed bundle, missing machine override, missing configuration | Rejected by the package/loader tests (E4) |
| Unsupported typed signature | Matches no rule (E3) |

## Static-only boundary

As delivered by E1–E7:

- Every ARM emitter reports `hasLowering() == false`; a valid bundle's `lower()`
  returns `target_lowering_unsupported`.
- Neither package supplies codegen requirements, and `llk-compile` rejects
  selected-target compilation for both.
- Static bind/report tests assert physical completeness separately and never
  require executable lowering. Reference execution is not counted as selected
  ARM evidence.

Framework E10 (issue #184) changes part of that boundary for `arm-sve` only.
`llk-compile` now rejects selected-target compilation for signatures without an
implemented emitter; the f32 add/multiply subset is executable as of E10. The
package publishes an AArch64 `+sve` code-generation contract, and that subset
has an emulated AOT acceptance path: a cross-compiled static object run under
`qemu-aarch64`. The emulated run is functional evidence at the modeled vector
length, not a measurement. Every other `arm-sve` emitter key still reports
`hasLowering() == false`, every other dtype still counts as reference-lowered,
and a selected run refuses such a kernel rather than reporting reference code
as target execution. `arm-sme` is unchanged: it supplies no codegen contract
and selected-target compilation is still rejected.

## CLI parity and frozen replay (E7)

`ArmTargetPipeline` drove `--micro-map` with explicit
`target`/`machine`/`layouts`/`rules`/`emitters` (all ten keys),
`mode=deterministic`, `top-k=0`; verified the mapped IR with
`--micro-verify-mapping`; replayed the reported id with `--micro-bind-plan` in a
fresh process; and ran
`micro-perf --machine=<profile> --level=1 --fail-on-capacity-violation --format=yaml`
on the bound kernel.

Local result (macOS arm64, 2026-10-24, source `95d35d7`):

| Case | Selected plan id | Planner cycles | DRAM bytes | Local bytes | Accumulator live peak | Parity |
|---|---|---|---|---|---|---|
| `arm-sve` f32 | `3b8ac351ee12f5ee` | 992 | 3,072 | 5,120 | 1,024 | cycles, traffic and peaks agree |
| `arm-sve` bf16 | `d30519f2487e408a` | 976 | 2,048 | 4,096 | 1,024 | agree |
| `arm-sme` f32 | `a5cae39089a17f3e` | 752 | 3,072 | 5,120 | 1,024 | agree |
| `arm-sme` bf16 | `bd6ec457936ddbc0` | 736 | 2,048 | 4,096 | 1,024 | agree |

For every case:

- the report is materialized and physically complete with empty
  `physicalReasons`, and `selectedPlanId == selectedState.id`;
- the source module hash is identical across the two targets for the same
  fixture (`40f808722ebf48d6` at this revision) and the search is binding-free;
- two fresh-process runs produce byte-identical reports and mapped IR;
- the fresh-process `--micro-bind-plan` replay reproduces byte-identical
  normalized IR, retaining the plan, its routes and the machine, layout, rule,
  target and graph hashes;
- `micro-perf` reports `complete: true`, no incomplete reasons and no capacity
  violations, and its `predicted_cycles`, per-level traffic and
  `tiles.live_bytes.acc`/`sram` equal the planner's latency, DRAM/local bytes
  and storage-plan peaks;
- a bounded exact search at `top-k=1` either agrees with the uncapped best
  explored cost or reports `searchTruncated` itself. No claim of global
  optimality is made, and modeled cycles are never presented as measurements.

These plan ids are content hashes of the selected plan at this revision. They
are reproducible evidence for the runs above, not a cross-version constant.

## Reproduction

```sh
ninja -C build llk-opt micro-perf LLKMappingTests
./build/bin/LLKMappingTests --gtest_filter='ArmTargets.*:ArmPolicy.*:ArmPlacement.*:ArmDtypeCoverage.*:ArmFeasibility.*:ArmComputeIdentity.*'
ctest --test-dir build -R '^ArmTargetPipeline$' --output-on-failure
ninja -C build check-llk
```

## Claims and limits

- `arm-sve` and `arm-sme` are static mapping and analysis targets at modeled
  VL=SVL=512. The deliverable is a complete, verified, replayable and analyzed
  plan over the bounded direct-MicroIR domain above.
- The `arm-sve` f32 add/multiply subset is executable as of E10 through an
  emulated AArch64 acceptance path (a cross-compiled static object run under
  `qemu-aarch64`). That path is functional evidence, not a measurement, and it
  does not widen this static ledger's delivery: no native SVE execution, no SME
  execution, no physical ZA allocation or addressing, no length-agnostic
  semantics, no calibrated timing and no automatic retargeting of CPU schedules
  is delivered or implied. The SVE profile's vector-contraction selection does
  not make its vector engine a matrix engine, and the SME ZA model is storage
  accounting, not an instruction semantic.
- Every `arm-sve` signature outside the executable f32 add/multiply subset, and
  every `arm-sme` signature, remains static mapping and analysis. Their selected
  lowering is rejected rather than emulated.
- Small direct fixtures do not establish arbitrary DNN support. Both targets
  still need their own packages, and neither replaces the AVX2 executable path.
- The executable-target baseline is unchanged. The selected-versus-reference
  AVX2 matrix in :source:`issue67-final-acceptance.md <docs/reviews/issue67-final-acceptance.md>`
  still governs AVX2 execution, and this ledger neither reclassifies reference
  runs nor revises that historical record.
- Closing this static increment does not close #175. Its executable-backend
  workstream (E10, #184) must compare fixed-VL execution, length-agnostic
  semantics and SME stateful lowering, and must publish its own evidence.
