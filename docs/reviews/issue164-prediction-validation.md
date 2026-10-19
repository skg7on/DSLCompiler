# Phase D: verified records and held-out prediction validation

Issue #164 (D4) completes the record/report slice of Phase D. The authoritative
plan is [measurement validation](../superpowers/plans/2026-10-12-microir-measurement-validation.md).
D1 defined validated records, D2 added atomic exact-key storage, and D3 integrated
opt-in raw samples, verified persistence and explicit reuse. D4 consumes those
records without changing machine YAML, static scoring or event durations.

## Delivered contracts

- `validatePredictions` validates a single comparable regime, preserves input row
  order, rejects duplicate executable identities and returns absolute/signed
  relative/absolute relative errors. Median and nearest-rank p90 use independent
  deterministic assertions: 100 ns versus 80/200 ns gives 0.25/0.5, median 0.375,
  p90 0.5. Nonfinite inputs or derived errors fail.
- Complete target, backend, compiler, environment, protocol, verification,
  adapter, workload contract, candidate bindings and bottleneck distinguish
  regimes. Dtype/shape remain in the contract. Comparing full fixture/verifier
  content is conservative: changed data/tolerances create additional groups.
- Missing/zero prediction time yields `no_prediction_time_basis`. Excluded-only
  groups have null summaries, never a zero-error claim. Counts distinguish
  records, comparable observations and raw repeated timing samples.
- `llk-validate` reads explicit files or nonrecursive record directories and
  emits JSON/text. Corrupt records fail with paths. Full-identity manifests
  reject duplicates, train/calibration overlap and missing holdout records.
  Without a manifest, output is explicitly not held-out. No model is fitted.
- Report writes cannot replace evidence records/manifests, including symlinks
  and hard links. The reporter does not change input records, machine profiles
  or search behavior.

## Baselines and provenance

The checked-in [arithmetic baseline](issue164-prediction-baseline.json) is produced
from [three synthetic records](../../test/Tuning/Inputs/prediction_validation) and
an explicit [holdout manifest](../../test/Tuning/Inputs/prediction_validation_holdout.json).
It contains two comparable errors and one excluded time basis, nine raw samples,
complete fixture source/target/backend/compiler/protocol identities and no fitted
parameters. The opaque synthetic identities describe test fixtures, not a claim
that any compiler measured 80 ns or 200 ns. Paths in that published JSON are
normalized relative to the repository; other fields are tool output.

`PredictionValidationMappedCLI` separately compiles a real bounded elementwise
source, numerically verifies outputs through D3, persists five raw timed samples,
selects its full identity through a holdout manifest and runs `llk-validate`.
On compatible x86 it requires selected-target execution; on arm64 it uses the
explicit portable reference backend. It asserts provenance and pipeline behavior,
not timing/error thresholds. It retains every actual record, fixture, tuning
report, manifest and baseline under `build/prediction-validation-evidence/`.
CI publishes this directory in the `selected-avx2-execution-evidence` artifact;
exact tested revision, counts and CI run are recorded in the implementation PR.
A single example does not establish general model accuracy or a sample minimum.

The [manual](../manual/source/tools/validation.rst) documents an optional device
corpus command with explicit contraction, fusion, elementwise and movement stores.
Movement records can come from the existing contract-based mapped measurement API;
this increment does not add a new collector or fabricate contraction provenance
for the legacy concrete-Micro CLI. Selected/reference times and distinct machines
remain separate. Independently measured event attribution, fitting/confidence,
sample minimums and held-out calibration thresholds require a later design.

## Reproduction

```sh
cmake -S . -B build -G Ninja -DLLVM_PROJECT_BUILD_DIR=<llvm-project>/build
cmake --build build --target llk-validate LLKPerfTests FrameworkSourceTest -j 8
build/bin/LLKPerfTests --gtest_filter='PredictionValidation.*:MeasurementRecord.*'
build/bin/FrameworkSourceTest --gtest_filter='MeasurementStoreTest.*'
ctest --test-dir build -R '^PredictionValidationCLI$' --output-on-failure
build/bin/llk-validate test/Tuning/Inputs/prediction_validation/ \
  --holdout-manifest=test/Tuning/Inputs/prediction_validation_holdout.json

# Real verified records; requires llk-tune and E2E support.
cmake --build build --target llk-tune -j 8
ctest --test-dir build -R '^PredictionValidationMappedCLI$' --output-on-failure

# Phase D increment exit: all registered tests, including required AVX2 suites.
cmake --build build --target check-llk -j 8
```

The required AVX2 suites explicitly fail on an incompatible arm64 host. Preserve
those failures and skip reasons rather than interpreting reference results as
selected-target execution. The pinned compatible x86 CI run supplies the complete
increment-exit evidence.
