# Issue #149 / analysis increment A validation

Base: `936f4fc` (A3 merged). Implementation: the commit containing this record.

## Change and plan reconciliation

Complete-plan evaluation defaults to `AnalysisRequirement::Complete`, independent
of executable backend availability. Complete scoring refuses missing physical
facts, audited semantic incompleteness and analysis errors; it cannot use
synthesized-event fallback. Mapped search/tuning explicitly request Complete.
`PartialReport` with partial binding remains inspectable with warnings and
`materialized=false`; executable binding always requires complete analysis.

A target-neutral post-evaluator check in CoveringSearch rejects existing partial
materialization/physical flags before retention. This extends the plan's file
list so a custom callback cannot accidentally rank a report artifact; it adds no
performance/runtime dependency and no parallel completeness schema.

Positive fixture repairs make formerly implicit physical facts explicit:
copy-port DRAM/SRAM memory requirements; the two-hop fixture's DRAM-to-L2
boundary store link (dma.b, 32 bytes/cycle). Its 256-byte store costs eight cycles,
so the measured-route expected total is now 1072 rather than 1064 cycles.
Negative analysis controls retain an explicitly unmodeled store endpoint.
Three binder tests inspect ambiguous-memory artifacts through direct PartialReport
rather than ranked search. Three synthetic-hop tests discard the now-valid
selected snapshot before testing plan-step synthesis; selected/materialized
snapshot parity remains separately covered.

## Red/green evidence

The new complete-plan tests first failed on retained unknown scalar computation,
extraction-error fallback, missing physical facts, incomplete alternatives
masking complete ones, and partial artifacts entering ranking. After the fix,
22/22 CompletePlanEvaluationTest cases passed. They include fully modeled ranking
without a backend and executable strictness despite a PartialReport argument.
Existing capacity/topK, exact-resource, measured-cost and static-parity tests
are covered by the full registered suite.

## Fresh local build

Host: Darwin 27.0.0 arm64; LLVM/MLIR `24.0.0git`.

```
cmake -S . -B build -G Ninja -DLLVM_PROJECT_BUILD_DIR=/Users/skg7on/Workspace/Projects/llvm-project/build
cmake --build build --target LLKConversionTests -j 6
build/bin/LLKConversionTests --gtest_filter=CompletePlanEvaluationTest.*
cmake --build build --target check-llk -j 6
```

After fixture reconciliation the full run registered 1268 tests: 1260 passed,
zero functional/fixture failures, six skipped, one disabled, and one required
selected-execution host failure (`MappedAVX2Acceptance`: target x86_64 versus
host aarch64). Its six failing inner cases are:

- `InvokesTheCompilerGeneratedMatmulAndChecksEveryElement`
- `InvokesTheCompilerGeneratedSwiGLUAndChecksEveryElement`
- `MapsAndInvokesTwoIndependentVectorOutputs`
- `LeavesTheCallersBuffersOwnedByTheCaller`
- `SelectedAvx2ExecutesThePaddedTailFixtures`
- `SelectedAvx2BackendExecutesNumerically`

Skipped CTest entries: `SwiGLUScalar.JitCompilationSmoke`,
`SwiGLUScalar.E2EWithAbiWrapper`, `SwiGLUVectorAVX2.Correctness`,
`MappedAcceptance.SelectedAvx2ExecutesThePaddedTailFixtures`,
`MappedAcceptance.SelectedAvx2BackendExecutesNumerically`, `MappedAcceptance`.
Disabled: `TritonStagedExecution.SwiGLUCorrectness`.

The required-selected test is deliberately unchanged. Local results do not
establish AVX2 execution. Linux CI must supply the full green selected gate.
Sphinx is unavailable in the local Python environment; CI validates the manual.
