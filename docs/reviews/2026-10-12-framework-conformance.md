# Framework conformance acceptance review

Revision-pinned, capability-based acceptance for the generic MicroIR mapping
and search compiler framework (tracker #145). This review covers the delivered
analysis, workload/source-search and primitive/target conformance increments.
It does not claim arbitrary DNN import, arbitrary-rank invocation or hardware
execution for the analysis-only accelerator.

- **Original tested tree:** `feat/framework-acceptance` at `c5aba8f`
  (C1+C2+C3 integrated; see the stacked-PR note below).
- **Local toolchain:** LLVM/MLIR 24.0.0git, `darwin arm64`.
- **Required selected-execution gate:** Linux `x86_64` CI with LLVM/MLIR 22.1.8.
- **Mapped ABI:** v1, rank-two compact descriptors, at most 12 ports, f32 /
  bf16 / f16 / i32 / i8.

## Acceptance matrix

| Row | Capability | Verdict | Evidence | Retained limit |
|---|---|---|---|---|
| A | Analysis coverage | **Passed** | A1–A4 (#165–#168); `CompletePlanEvaluationTest`; incomplete plans rejected before top-K | Extraction must model an operation; unmodelled leaves reject ranking |
| B | Workload contracts and source search | **Passed** | B1–B7 (#170–#202); `FrameworkSourceTest`, `GenericTuneCLI`, `GenericElementwiseTest` | Bounded rank-two f32 elementwise family; contraction options remain adapter-specific |
| C1 | Stage-specific capability receipts | **Passed** | `FrameworkCapabilitiesTest`; `llk-tune` per-candidate receipts; `llk-compile` stderr receipt | Evidence must match the source/target/plan/contract it credits |
| C2 | Movement, storage and resource conformance | **Passed** | `FrameworkConformance` chains over two-compute / named-ports / two-hop / padded-layout / capacity-topk | Fixture scope; no new planner or second evaluator |
| C3 | Composed vectors and reduction | **Passed (static + selected lowering)** | `FrameworkPrimitivesTest`, `FrameworkConformance`, `FrameworkSelectedEvidence` manifest | Selected AVX2 **execution** requires `x86_64`; rank-one invocation is unsupported |
| C4 | Target extension boundaries | **Passed** | `FrameworkTargetPortability` 3/3; `FrameworkBoundaryScan`; `FrameworkEvidenceManifest` | Test-only wrapper; no new physical target |
| — | Generic accelerator execution | **Unsupported** | No codegen contract; `selectedLowering = unsupported` with reason | Analysis-only target; no hardware claim |
| — | Rank-one / rank-three invocation | **Unsupported** | `runtime_capability: mapped ABI v1 requires rank-two ports` | Analysis and selected lowering remain rank-neutral |
| — | Measurement persistence / calibration | **Out of scope** | Tracked by #51/#52 and increment D (#161–#164) | This review records static and selected-lowering evidence only |

## What each row establishes

### A — Analysis coverage

Unknown computational leaves are diagnosed, and ranked plans require complete
audited semantic/physical analysis before top-K retention. Explicit
`PartialReport` artifacts remain inspectable but `materialized=false` and cannot
enter ranked results. Evidence: #149 / PR #168, Linux LLVM/MLIR 22.1.8 CI run
`38662314977` (1,269 registered / 1,265 passed / zero failed / three legacy JIT
skips / one disabled), required selected AVX2 13/13.

### B — Workload contracts and source search

The rank-neutral versioned contract, source-adapter registry, direct MicroIR
candidates, bounded Linalg elementwise import, realized tile schedules and the
generic CLI/replay/numerical path are merged. Evidence: PRs #170, #171, #173,
#185, #187, #200 and #202; B5–B7 CI green. `GenericElementwiseTest` checks
full/tiled add and multiply numerically, and `GenericElementwiseAVX2Test`
requires selected execution without reference fallback.

### C1 — Stage-specific capability receipts

`CapabilityReceipt` v2 reports `supported`, `unsupported` or `not_checked` for
structure, interpretation, mapping, physical plan, reference lowering, selected
lowering, ABI eligibility, preparation, host compatibility and invocation.
Identities cover the source graph, canonical contract, target content and plan
id; stale analysis or compilation cannot establish support. A reference-only
compilation leaves selected lowering `not_checked`; selected-only compilation
does the same for reference lowering. Executable preparation does not establish
invocation, and a returned call does not establish numerical correctness.
Evidence: `FrameworkCapabilitiesTest` and the integrated checks recorded below.

### C2 — Movement, storage and resource conformance

One strict `search -> finalized plan -> bind -> shared analysis` chain checks:

- selected same-kind compute engines (`vpu.a`/`vpu.b`);
- per-occurrence memory bindings (`sram.a`/`sram.b`) and 256-byte borrowed
  descriptors;
- the `sram.0 -> l2.0 -> dram.0` route, `dma.a`/`dma.b`, shared intermediate
  storage and literal traffic `sram=256 / l2=768 / dram=512`;
- the padded physical image (480 bytes) and the capacity-before-top-K result;
- a direct baseline and an added logical view preserving producer alias and
  dependency, 256-byte SRAM/L2 traffic, a 512-byte SRAM peak and no additional
  allocation; and a physical transform with
  64 work items, 256 bytes, 8 cycles on `vpu.0`, plus one extra 256-byte SRAM
  allocation.

Search and materialized costs agree without a second planner call. Evidence:
`FrameworkConformance` plus `Routing.*`, `StorageLiveness.*`,
`ExactResourceOracleTest.*` and `CoveringSearch.*` 155/155.

### C3 — Composed vectors and reduction

- The 8x8 f32 sum produces `[8]`, charges its 64 input elements in the scheduled
  event, maps on both profiles, lowers to `vector.multi_reduction` /
  `vector.reduction` with `vector<8xf32>` and zero reference fallback, and
  refuses executable preparation with the rank-two ABI reason.
- Invalid axes, `[8,1]` results and mismatched dtypes are refused by the
  existing verifier; `max` and `product` variants have no rule on either
  profile.
- Add → multiply executes on the reference backend everywhere and on selected
  AVX2 on a supported host, checking every f32 element against
  `(left[i] + right[i]) * left[i]` with positive backend realization and zero
  reference groups.
- `TargetEmitter::verifyOperations` and `verifyTargetOperationCoverage` validate
  covered variants/types before analysis or scoring. The generic accelerator
  advertises analysis-only f32 mul/sum; bf16 is rejected as a candidate with
  unchanged IR, and f64 is outside the Micro tile type system.

Evidence: `FrameworkPrimitivesTest` (5 passed, 1 selected-AVX2 skip on this
host), `FrameworkConformance.FrameworkPrimitivesMapStaticallyOnBothProfiles`,
`FrameworkSelectedEvidence`, and the validated manifest
`framework_selected_evidence.manifest.json` (3 rows; source, replay identity,
plan id, target, compiler, mode, strict `micro-perf` traffic/storage and
counters). These rows record selected lowering only: reference lowering and
numerical validation remain `not_checked`. The two rank-one reduction results
contain eight elements and report the checked ABI preparation refusal. Numeric
add → multiply execution is established by the separate runtime tests above.

### C4 — Target extension boundaries

- A test-only wrapper forwards every `MappingTarget` method while changing only
  `name()`. It is never registered globally. Renaming preserves the plan id,
  rule/executor/memory/compute bindings and every capability status except
  `targetIdentity`; a renamed AVX2 target still lowers with realized groups and
  no reference fallback.
- Changing machine data changes legal behavior without any name-based dispatch:
  removing `vpu.b` reduces the two-compute covering set to `{vpu.a}`, enlarging
  SRAM makes the previously infeasible `r.small` rule feasible, and shrinking
  DRAM to 512 bytes produces no plan with a `MemoryCapacityExceeded` reason.
- `FrameworkBoundaryScan` finds no target-name or semantic-operator policy in
  generic `LLKMapping`/`LLKMachine` code and no `LLKPerf`/`LLKRuntime`/JIT link.
  `LLKMapping` links `LLKMachine LLVM LLK::MLIR`; `LLKMachine` links `LLVM`.
- `FrameworkEvidenceManifest` requires the three selected primitive cases
  exactly once, matching source/symbol/target/mode and analyzed/lowered
  plan/replay/compiler identities. All selected groups must be verified and
  realized with zero reference fallback. The dependency-free reader checks
  actual `micro-perf` completeness, reason lists and unsigned traffic/storage
  maps; comments and duplicate keys cannot establish support. Unsupported
  stages require reasons. Resource and portability rows are checked by their
  separate CTests rather than this selected-lowering manifest.

## Host, skip and CI limits

The local full serial CTest run on `darwin arm64` executed 1,305 tests:
1,302 passed and three required selected-AVX2 binaries failed with
`selected target requires architecture 'x86_64' but host architecture is
'aarch64'`:

- `MappedAVX2Acceptance`
- `GenericElementwiseAVX2Test`
- `FrameworkPrimitivesAVX2Test`

Six legacy/selected JIT cases were skipped and one test is disabled; the skip
reasons are unchanged from PR #133/#202. These are host limitations, not
portable failures, and they are **not** counted as selected execution evidence.
The required selected-AVX2 numerical gate is the Linux `x86_64` CI job with
zero selected-path skips.

**Stacked-PR note.** C1 (#205) is green. C2 (#206) and C3 (#207) were opened as
a stack because CI runs only for pull requests targeting `main`; this C4 review
records the integrated local evidence first. After #205–#207 merge, this
revision must be retargeted and the CI run id plus exact registered/skip counts
filled in before the C4 pull request is merged. No CI run is claimed here that
has not completed.

## Reproduction

```bash
# Static/portability/boundary evidence (portable).
./build/bin/LLKConversionTests --gtest_filter='Framework*'
./build/bin/FrameworkPrimitivesTest
ctest --test-dir build -R 'FrameworkBoundaryScan|FrameworkEvidenceManifest'

# Selected lowering and the manifest.
ctest --test-dir build -R 'FrameworkSelectedEvidence|FrameworkEvidenceManifest'

# Full suite (serial is authoritative).
ctest --test-dir build --output-on-failure
```

The selected execution gate is
`ctest --test-dir build -L selected-avx2 --no-tests=error --output-on-failure`
on a Linux `x86_64` host with AVX2.

The test counts and tested-tree revision above are the original C4 snapshot.
The review fixes add producer/validator/boundary regression tests and enforce
fresh producer evidence through a CTest fixture. Their integrated counts and
revision must be recorded after the updated stack is verified; the original
snapshot is not evidence for the changed tree.
