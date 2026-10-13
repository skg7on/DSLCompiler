# Verified Measurement Persistence and Prediction Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist verified whole-kernel measurements and report prediction error by workload/target regime without changing default static estimates or inventing event attribution.

**Architecture:** Reuse `MappedMeasurementIdentity`, typed input/provider/verifier callbacks and the existing CLI timing loop. Add versioned record/protocol/environment data, an atomic local record store and an explicit validation report. Exact identity matches may supply optional measured candidate metrics; operation/connection calibration and automatic fitting remain separate future designs.

**Tech Stack:** C++20, LLVM JSON/filesystem/Error, existing mapped runtime, GTest, Python CLI regressions and steady-clock timing.

**Spec:** [Framework design §§4.5, 5–8](../specs/2026-10-12-generic-microir-dnn-compiler-framework-design.md); [index](2026-10-12-generic-microir-framework-implementation.md); issue [#51](https://github.com/skg7on/DSLCompiler/issues/51) / [#52](https://github.com/skg7on/DSLCompiler/issues/52) handoff.

## Global Constraints

- “Measurement persistence and calibration are separate capability work (#51/#52).”
- “Measured and predicted metrics remain distinct.”
- “Rejected candidates carry reasons; missing measurement values never rank as zero.”
- “Frozen replay preserves full source/candidate content, bindings, plan, machine/layout/rule content, compiler version, ABI, numerical contract and execution mode.”
- “`LLKMapping` remains independent of performance, runtime and JIT libraries.”
- No automatic fitting, MachineModel-default mutation or distributed benchmarking in this increment. Whole-kernel timing must not become a per-event LatencyProvider override.
- B's contract-aware identity and C's concrete capability/acceptance receipts are prerequisites for generic examples. Record APIs also support the existing contraction examples.

## Review Focus

1. A record with changed source/plan/target/ABI/backend/compiler content must miss despite matching a short hash: D1/D2.
2. NaN, infinity, nonpositive samples, truncated JSON or unknown schema must fail loading rather than create rankable zero metrics: D1/D2.
3. CPU/thread/affinity/protocol/input-verifier changes must invalidate reuse and remain visible: D2/D3.
4. Numerical verification must succeed before storing or returning measured metrics; unavailable measurements remain optional: D3.
5. Unknown clock, zero sample groups and incomparable measured modes must be reported without invented error/accuracy claims: D4.

---

## New file map and interfaces

Create `include/LLK/Perf/MeasurementRecord.h`, `lib/Perf/MeasurementRecord.cpp` in `LLKPerf` for plain record validation/serialization (no runtime/JIT types). Create `include/LLK/Conversion/MicroMapping/MeasurementStore.h`, `lib/Conversion/MicroMapping/MeasurementStore.cpp` in `LLKMappedTuning` for persistence/integration. Create `lib/Perf/PredictionValidation.cpp` and its header for deterministic validation statistics; add root CMake source registrations.

```cpp
// namespace mlir::llk::perf
struct MeasurementProtocol {
  uint32_t version = 1;
  uint64_t warmup = 0;
  uint64_t repeat = 0;
  std::string timer = "steady_clock";
  std::string statistic = "median";
  std::string scope = "per_invocation";
};
struct MeasurementEnvironment {
  std::string architecture;
  std::string cpuIdentity;
  std::string isa;
  uint64_t threads = 0;
  std::string affinity;
  std::string clockPolicy;
};
struct VerificationIdentity {
  std::string fixtureContent;
  std::string verifierVersion;
  std::string numericalContract;
};
struct MeasurementRecord {
  uint32_t version = 1;
  std::string executableIdentity; // complete canonical identity, not hash only
  std::string executableHash;
  std::string targetIdentity;
  std::string backendIdentity;
  MeasurementEnvironment environment;
  MeasurementProtocol protocol;
  VerificationIdentity verification;
  bool verified = false;
  std::vector<double> samplesNs;
  double medianNs = 0.0;
  CandidateMetrics prediction; // predicted fields remain predicted
  std::string sourceAdapter;
  std::string workloadContract;
  std::string candidateBindings;
  std::string bottleneck;
  std::string timestampUtc;
  std::string compilerVersion;
};
llvm::Error validateMeasurementRecord(const MeasurementRecord &);
llvm::json::Object measurementRecordToJson(const MeasurementRecord &);
llvm::Expected<MeasurementRecord> measurementRecordFromJson(
    const llvm::json::Value &);
std::string canonicalMeasurementKey(const MeasurementRecord &);
std::string measurementContentHash(llvm::StringRef);
```

The existing private `stableContentHash(StringRef)` is factored into `perf::measurementContentHash` in `lib/Perf/MeasurementRecord.cpp`, used by both identity construction and record validation. Preserve its audited decimal-rendered FNV-1a algorithm; this move does not change identity hashes. Full canonical identity/environment/protocol/verification are compared on lookup; digest equality alone is insufficient. `fixtureContent` carries complete canonical fixture content, including expected outputs/tolerances, rather than only an unresolvable short digest. Records with missing identity fields are not reusable.

```cpp
// namespace mlir::llk::tuning
struct MeasurementLookup {
  std::string executableIdentity;
  perf::MeasurementEnvironment environment;
  perf::MeasurementProtocol protocol;
  perf::VerificationIdentity verification;
};
class MeasurementStore {
public:
  explicit MeasurementStore(std::string directory);
  llvm::Error write(const perf::MeasurementRecord &);
  llvm::Expected<std::optional<perf::MeasurementRecord>>
  lookup(const MeasurementLookup &) const;
private:
  std::string directory;
};
```

Store one validated JSON file per canonical key, digest-named for indexing. Use an adjacent unique temporary file and atomic rename; readers never observe half a record. This is a local single-writer workflow; concurrent writers/distributed stores are outside scope. Refuse an existing path whose full key differs, even if its digest collides. Timestamp/sample values are not key fields; environment/protocol/verification are.

## D1: Define validated reproducible measurement records

**Files:** Create `include/LLK/Perf/MeasurementRecord.h`, `lib/Perf/MeasurementRecord.cpp`, `test/Perf/measurement_record.cpp`, `test/Perf/measurement_test_support.h`; Modify root `CMakeLists.txt`, `test/Perf/CMakeLists.txt`, `lib/Conversion/MicroMapping/MappedTuningSession.cpp`.

**Interfaces:** Produces record validation/JSON/key functions above. Reuses current canonical executable identity and cost fields; adds no event-duration provider.

- [ ] **Write a complete positive test record and failing tests**:

```cpp
inline perf::MeasurementRecord validRecord() {
  perf::MeasurementRecord record;
  record.executableIdentity = "complete-test-executable-identity";
  record.executableHash = perf::measurementContentHash(
      record.executableIdentity);
  record.targetIdentity = "complete-test-target-content";
  record.backendIdentity = "selected-target:test-compiler:test-abi";
  record.environment = {"x86_64", "test-cpu", "avx2", 1,
                        "single-thread-unpinned", "observed-default"};
  record.protocol = {1, 2, 3, "steady_clock", "median", "per_invocation"};
  record.verification = {"complete-test-fixture", "scalar-f32-v1",
                         "strict-f32"};
  record.verified = true;
  record.samplesNs = {80.0, 100.0, 90.0};
  record.medianNs = 90.0;
  record.prediction.predictedNs = 100.0;
  record.prediction.predictedCycles = 300;
  record.sourceAdapter = "linalg-elementwise-v1";
  record.workloadContract = "complete-test-contract";
  record.candidateBindings = "T0=4,T1=8";
  record.timestampUtc = "2026-10-12T00:00:00Z";
  record.compilerVersion = "test-compiler";
  return record;
}
TEST(MeasurementRecord, RoundTripsVerifiedSamplesAndPredictionSeparately) {
  auto record = validRecord();
  auto decoded = perf::measurementRecordFromJson(
      llvm::json::Value(perf::measurementRecordToJson(record)));
  ASSERT_TRUE(bool(decoded)) << llvm::toString(decoded.takeError());
  EXPECT_EQ(decoded->samplesNs, record.samplesNs);
  EXPECT_EQ(decoded->medianNs, 90.0);
  EXPECT_EQ(decoded->prediction.predictedNs, 100.0);
  EXPECT_EQ(perf::canonicalMeasurementKey(*decoded),
            perf::canonicalMeasurementKey(record));
}
```

Put the inline `validRecord` helper in `test/Perf/measurement_test_support.h`, with its required includes, and include it from D1/D2/D4 tests. `perf::measurementContentHash(StringRef)` is defined in this task's record library; LLKMappedTuning already consumes LLKPerf, so no dependency reversal is needed. Add negative tests for version=2, unverified records, empty executable/target/backend/env/verifier identity, samples/repeat mismatch, zero/negative/NaN/infinite samples, false median, hash mismatch, malformed field types and missing keys.
- [ ] **Run red:** build/run `LLKPerfTests` filter `MeasurementRecord.*`; expected missing schema/helpers, then failing validation.
- [ ] **Implement strict parsing and median validation**. Sort a copy of finite positive samples, use the middle value for odd counts and mean of the two middle values for even counts. Require `repeat == samplesNs.size()`, `repeat > 0`, valid protocol version/statistic/timer/scope, `threads > 0`, verified true and full identity fields. Use stable `measurement_record:` errors naming fields. Round-trip sample order; serialization does not relabel static predictions as measured. Unknown schema major fails; record keys use tagged length-delimited canonical fields.

```cpp
if (!record.verified)
  return llvm::createStringError("measurement_record: numerical verification required");
for (double sample : record.samplesNs)
  if (!std::isfinite(sample) || sample <= 0.0)
    return llvm::createStringError("measurement_record: samples_ns must be finite and positive");
```

- [ ] **Run green:** all record tests plus current identity/key tests. Expected: lossless round-trip and deliberate refusal of unusable measurements.
- [ ] **Commit:** stage named files; `git commit -m 'feat: define verified measurement record schema'`.

## D2: Persist records atomically and require exact reuse keys

**Files:** Create `include/LLK/Conversion/MicroMapping/MeasurementStore.h`, `lib/Conversion/MicroMapping/MeasurementStore.cpp`, `test/Tuning/measurement_store.cpp`; Modify root `CMakeLists.txt`, `test/Tuning/CMakeLists.txt`. Include `../Perf/measurement_test_support.h`; link the new static store tests in `FrameworkSourceTest` to `LLKMappedTuning` without executing a JIT.

**Interfaces:** Produces `MeasurementStore`/`MeasurementLookup`. The store only returns validated verified records; an identity miss returns `std::nullopt`, while malformed content returns Error with its path.

- [ ] **Write the temporary-directory fixture and failing restart/miss test**:

```cpp
class MeasurementStoreTest : public ::testing::Test {
protected:
  llvm::SmallString<128> directory;
  void SetUp() override {
    ASSERT_FALSE(llvm::sys::fs::createUniqueDirectory("llk-measurement", directory));
  }
  void TearDown() override {
    EXPECT_FALSE(llvm::sys::fs::remove_directories(directory));
  }
};
```

```cpp
TEST_F(MeasurementStoreTest, ReopenHitsOnlyTheCompleteIdentity) {
  const auto record = validRecord();
  tuning::MeasurementStore writer(directory.str().str());
  ASSERT_FALSE(writer.write(record));
  tuning::MeasurementStore reader(directory.str().str());
  tuning::MeasurementLookup key{record.executableIdentity, record.environment,
                                record.protocol, record.verification};
  auto hit = reader.lookup(key);
  ASSERT_TRUE(bool(hit)) << llvm::toString(hit.takeError());
  ASSERT_TRUE(hit->has_value());
  EXPECT_EQ((*hit)->medianNs, 90.0);
  key.environment.threads = 2;
  auto miss = reader.lookup(key);
  ASSERT_TRUE(bool(miss)) << llvm::toString(miss.takeError());
  EXPECT_FALSE(miss->has_value());
}
```

Define `directory` as `llvm::SmallString<128>` and include LLVM filesystem headers. Repeat the miss assertion for source/plan/backend/mode/compiler/ABI changes inside the canonical executable identity, changed affinity/CPU/protocol and fixture/verifier/contract content. Create a truncated JSON file at a known key path and assert corrupt-record Error. Test a simulated digest collision using an injected test-only digest function or a supplied filename fixture; full-key mismatch must refuse reuse.
- [ ] **Run red:** build/run store tests; expected missing store and no durable restart lookup.
- [ ] **Implement atomic local writes**: validate, make directory, build digest path, compare any existing full key, create unique adjacent temporary, write/flush/close, rename. On any failure return a path-bearing Error and remove only the owned temporary file. Read only the exact key's file; parse and validate before comparing complete canonical key. Do not scan by workload/shape and select an approximate match.

```cpp
// Lookup outcome policy:
// absent path -> optional<Record>{};
// invalid JSON/schema/record -> Error;
// full key mismatch -> optional<Record>{};
// exact validated full key -> Record.
```

Include protocol/environment/verifier in path indexing and full comparison. Use canonical identity rather than process-local hash values. Test that an interrupted temporary file cannot mask an intact record.
- [ ] **Run green:** `FrameworkSourceTest` measurement-store suite plus record/key tests. Expected: process-restart reuse, exact misses and no partial-file success.
- [ ] **Commit:** stage named files; `git commit -m 'feat: persist exact verified kernel measurements'`.

## D3: Integrate optional persistence with existing verified measurement

**Files:** Modify `include/LLK/Conversion/MicroMapping/MappedTuningSession.h`, `lib/Conversion/MicroMapping/MappedTuningSession.cpp`, `tools/llk-tune/llk-tune.cpp`, `test/Tuning/mapped_tuning_session.cpp`; Create `test/Tuning/measurement_store_cli.py`; Modify `test/Tuning/CMakeLists.txt`, `docs/manual/source/tools/tuning.rst`.

**Interfaces:** Preserve existing `inputs`, `measure`, `verify` callbacks. Add the following proposed detailed provider and option members; configured metadata is required for persistence/reuse, and a detailed provider returns its raw samples:

```cpp
// namespace mlir::llk::tuning
struct MappedMeasurementSamples {
  perf::CandidateMetrics metrics;
  std::vector<double> samplesNs;
};
using MappedDetailedMeasurementProvider = std::function<
    llvm::Expected<std::optional<MappedMeasurementSamples>>(
        const MappedMeasurementRequest &)>;
// Add to MappedMeasurementOptions:
MappedDetailedMeasurementProvider measureDetailed;
MeasurementStore *store = nullptr; // borrowed for the whole synchronous session
bool reuseExact = false;
perf::MeasurementProtocol protocol;
perf::MeasurementEnvironment environment;
perf::VerificationIdentity verification;
std::function<llvm::Error(const perf::MeasurementRecord &)> recordSink;
// Add to MappedTuningReport:
uint64_t measurementRecordHits = 0;
uint64_t measurementRecordMisses = 0;
uint64_t measurementRecordsWritten = 0;
```

Aggregate and detailed providers are mutually exclusive. New optional CLI `--measurement-store=<directory>` and `--measurement-reuse=off|exact` default to no store/reuse off. Reuse requires a measured objective/provider context and validated fixture; it cannot bypass executable/host capability checks. The store is the persistent sink; `recordSink` is an optional observer invoked after successful validated persistence.

- [ ] **Write failing sequencing test** with test callbacks recording calls in a `std::vector<std::string>`. A fresh candidate must produce `inputs, measure, verify, persist`; a numerical failure must never append persist or enter measured ranking. An exact reusable verified record can produce a measured hit only after source/plan/backend/ABI/host/protocol/fixture checks. Missing measured secondary metrics stay unrankable.

```cpp
EXPECT_EQ(calls, (std::vector<std::string>{
    "inputs", "measure", "verify", "persist"}));
EXPECT_EQ(report.measurementRecordsWritten, 1u);
EXPECT_TRUE(report.ranked.front().ranking.metrics.measuredNs.has_value());
```

The callback harness wraps the existing real mapped measurement request and executable; do not mock compilation as successful. Add CLI tests: default predicted-only run creates no store; explicit measurement writes one record; second identical process reports an exact hit; changed threads/source/backend/fixture produces miss; bad output fixture fails numerical verification with no record. Existing `mapped_tune_cli.py` already exercises measurement identity and warmup/repeat sidecars, so reuse its command setup and actual flags.
- [ ] **Run red:** mapped-tuning and store CLI tests; expected missing persistence options/counts or sequencing.
- [ ] **Implement successful-record persistence after verification**. Extract the existing timing protocol into reusable record data while retaining its current warmup/steady-clock/per-invocation behavior. Capture raw timed samples, environment and verifier/fixture identity; verify outputs through the existing callback before constructing a reusable record. Existing provider callbacks returning only aggregate metrics remain valid but cannot persist reusable sample records without protocol evidence. Cache hits are tagged `measured_record`, fresh runs `measured_runtime`, static estimates `predicted_static`; absence remains optional.

```cpp
if (llvm::Error verification = options.measurement.verify(
        selected, buffers.inputs, buffers.outputs))
  return std::move(verification);
// Only now validate/write the fully populated MeasurementRecord.
```

Use `MappedMeasurementIdentity::canonical` and B's generic source contract as the executable identity source. Host capability rejection remains explicit on non-AVX2 machines. A whole-kernel hit updates candidate measured latency only; it does not set `PlanPlacement::measuredCycles` or `ConnectionPlan::measuredCycles`. Keep `measureTop`/attempt limits and measured-cohort filtering unchanged; identity hits and timed attempts have separate counts.
- [ ] **Run green:** new persistence CLI/sequencing tests, existing measured-objective/missing-metric tests and typed numerical fixtures. Supported x86 CI supplies selected measurement evidence; ordinary tests use deterministic record samples for exact assertions, not fragile timing thresholds.
- [ ] **Commit:** stage named files; `git commit -m 'feat: reuse verified measurement records explicitly'`.

## D4: Report held-out prediction error without implicit calibration

**Files:** Create `include/LLK/Perf/PredictionValidation.h`, `lib/Perf/PredictionValidation.cpp`, `tools/llk-validate/llk-validate.cpp`, `test/Perf/prediction_validation.cpp`, `test/Tuning/prediction_validation_cli.py`; Modify root/Perf/Tuning CMake, manual tool/features pages and framework acceptance review.

**Interfaces:** New dependency-neutral statistics API:

```cpp
// namespace mlir::llk::perf
struct PredictionErrorRow {
  std::string identity;
  double predictedNs = 0.0;
  double measuredNs = 0.0;
  double absoluteErrorNs = 0.0;
  double signedRelativeError = 0.0;
  double absoluteRelativeError = 0.0;
};
struct PredictionValidation {
  std::vector<PredictionErrorRow> rows;
  std::vector<std::string> excluded;
  double medianAbsoluteRelativeError = 0.0;
  double p90AbsoluteRelativeError = 0.0;
};
llvm::Expected<PredictionValidation> validatePredictions(
    llvm::ArrayRef<MeasurementRecord> records);
llvm::json::Object predictionValidationToJson(const PredictionValidation &);
```

CLI reads explicit record files/directories and outputs JSON/text grouping by full target/environment/backend/protocol/numerical regime plus adapter/dtype/shape/bottleneck/decision metadata. Never merge selected and reference times or two machines into one accuracy statistic. Training/calibration record sets and held-out validation identities are separate; this increment fits no model.

- [ ] **Write failing deterministic tests**:

```cpp
TEST(PredictionValidation, ComputesRelativeErrorAgainstMeasuredLatency) {
  auto a = validRecord();
  a.samplesNs = {80.0, 80.0, 80.0}; a.medianNs = 80.0;
  a.prediction.predictedNs = 100.0;
  auto b = validRecord();
  b.executableIdentity += "-second";
  b.executableHash = perf::measurementContentHash(b.executableIdentity);
  b.samplesNs = {200.0, 200.0, 200.0}; b.medianNs = 200.0;
  b.prediction.predictedNs = 100.0;
  auto result = perf::validatePredictions({a, b});
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  ASSERT_EQ(result->rows.size(), 2u);
  EXPECT_DOUBLE_EQ(result->rows[0].absoluteRelativeError, 0.25);
  EXPECT_DOUBLE_EQ(result->rows[1].absoluteRelativeError, 0.5);
  EXPECT_DOUBLE_EQ(result->medianAbsoluteRelativeError, 0.375);
  EXPECT_DOUBLE_EQ(result->p90AbsoluteRelativeError, 0.5);
}
```

Keep input order for rows; sort a copy for summary metrics. `validatePredictions` accepts one comparable regime and rejects mixed environments/protocols/backend/numerical identities; the CLI partitions records by those canonical fields and invokes it once per group. Add empty-group, duplicate identity, unavailable `predictedNs`, mixed-regime and nonfinite tests. A missing clock/time conversion must produce an excluded `no_prediction_time_basis` row, not a zero prediction. Bad verified-record structure is an Error; the CLI emits incomparable regimes as separate groups.
- [ ] **Run red:** build/run validation tests and CLI; expected missing APIs/report tool.
- [ ] **Implement exact statistics and grouping**:

```cpp
row.absoluteErrorNs = std::abs(row.predictedNs - row.measuredNs);
row.signedRelativeError = (row.predictedNs - row.measuredNs) / row.measuredNs;
row.absoluteRelativeError = std::abs(row.signedRelativeError);
```

Median uses the record convention; p90 uses nearest-rank `ceil(0.9*n)-1`. With zero valid rows, JSON summary values are null with `status="no_comparable_samples"`; struct numeric defaults must not leak as a zero-error claim. Report actual prediction origin and counts. Accept an explicit holdout manifest of full identities and reject overlap with any supplied calibration/training identities. Add a small CLI round-trip using deterministic records and a separate optional hardware corpus command covering contraction/fusion/elementwise/movement regimes; no threshold is imposed until enough on-device samples establish a baseline.
- [ ] **Run green:** validation/record/store tests, CLI round-trip and full suite. Publish an uncalibrated baseline report with source/plan/target/compiler/protocol identities, observed errors, excluded cases and sample counts. Keep machine YAML and default static scoring byte-identical.
- [ ] **Commit:** stage named files; `git commit -m 'feat: report measured prediction validation by regime'`.

## Increment acceptance and subsequent calibration gate

- [ ] Records survive restart, require numerical verification and preserve full reproducible identity/environment/protocol.
- [ ] Default predicted-only tuning remains unchanged; exact reuse is explicit and missing metrics remain unrankable.
- [ ] Validation reports quantify held-out error and expose unavailable/incomparable data.
- [ ] No aggregate kernel timing is misrepresented as an operation/connection duration.
- [ ] A later calibration design must define independently measured event attribution, fit parameters/confidence, regime keys, sample minimums and held-out acceptance thresholds before implementing a LatencyProvider or model fitting. #52's record/report work can close independently of that future design.
