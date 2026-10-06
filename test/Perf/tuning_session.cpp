//===- tuning_session.cpp - Ranking, session loop, and schedule YAML
//-------===//
//
// Covers issue #50, ranking half:
//   - ranking is by the space's primary metric, then its secondary metrics,
//     then the candidate id, so the order is deterministic
//   - a session generates, rejects illegal candidates with stable reasons,
//     binds the legal ones, costs them with L0/L1, and keeps top-K
//   - schedule records carry the workload identity, machine identity, tile
//     decisions, and metrics the acceptance criteria name
//
// The ranking tests build metrics by hand so the comparison rules are pinned
// without depending on the simulator's numbers. The session test uses the
// shipped AVX2 model and a two-choice space, so one candidate is legal and one
// is rejected by a known constraint.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Machine/MachineModelLoader.h"
#include "LLK/Perf/ScheduleRecord.h"
#include "LLK/Perf/TuningSession.h"
#include "LLK/Runtime/MappedExecutable.h"

#include "LLK/Dialect/Micro/MicroDialect.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <chrono>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mlir::llk::perf {
namespace {

using llvm::StringRef;

TuningResult resultWith(std::string id, uint64_t cycles, double matrixUtil,
                        uint64_t dramBytes) {
  TuningResult result;
  result.legal = true;
  result.candidate.id = std::move(id);
  result.metrics.predictedCycles = cycles;
  result.metrics.matrixUtilization = matrixUtil;
  result.metrics.dramBytes = dramBytes;
  return result;
}

std::vector<std::string> ids(llvm::ArrayRef<TuningResult> results) {
  std::vector<std::string> out;
  for (const TuningResult &result : results)
    out.push_back(result.candidate.id);
  return out;
}

SearchObjective latencyObjective() {
  SearchObjective objective;
  objective.direction = ObjectiveDirection::Minimize;
  objective.primaryMetric = "latency_cycles";
  objective.secondaryMetrics = {"matrix_utilization", "dram_bytes"};
  return objective;
}

//===----------------------------------------------------------------------===//
// Ranking
//===----------------------------------------------------------------------===//

TEST(Ranking, OrdersByThePrimaryMetric) {
  std::vector<TuningResult> results = {
      resultWith("candidate_b", 200, 1.0, 0),
      resultWith("candidate_a", 100, 0.0, 999),
  };
  std::vector<TuningResult> ranked =
      rankTuningResults(results, latencyObjective());
  EXPECT_EQ(ids(ranked),
            (std::vector<std::string>{"candidate_a", "candidate_b"}));
}

TEST(Ranking, BreaksTiesOnTheSecondaryMetricsInOrder) {
  // Equal cycles: higher matrix utilization wins.
  std::vector<TuningResult> results = {
      resultWith("candidate_low", 100, 0.5, 10),
      resultWith("candidate_high", 100, 0.9, 20),
  };
  std::vector<TuningResult> ranked =
      rankTuningResults(results, latencyObjective());
  EXPECT_EQ(ranked.front().candidate.id, "candidate_high");

  // Equal cycles and utilization: lower dram bytes wins.
  results = {
      resultWith("candidate_more_dram", 100, 0.9, 20),
      resultWith("candidate_less_dram", 100, 0.9, 10),
  };
  ranked = rankTuningResults(results, latencyObjective());
  EXPECT_EQ(ranked.front().candidate.id, "candidate_less_dram");
}

TEST(Ranking, BreaksAnExactTieOnTheCandidateId) {
  std::vector<TuningResult> results = {
      resultWith("candidate_b", 100, 0.9, 10),
      resultWith("candidate_a", 100, 0.9, 10),
  };
  std::vector<TuningResult> ranked =
      rankTuningResults(results, latencyObjective());
  EXPECT_EQ(ids(ranked),
            (std::vector<std::string>{"candidate_a", "candidate_b"}));
}

TEST(Ranking, HonorsAMaximizeObjective) {
  SearchObjective objective;
  objective.direction = ObjectiveDirection::Maximize;
  objective.primaryMetric = "matrix_utilization";
  std::vector<TuningResult> results = {
      resultWith("candidate_low", 100, 0.5, 10),
      resultWith("candidate_high", 999, 0.9, 10),
  };
  std::vector<TuningResult> ranked = rankTuningResults(results, objective);
  EXPECT_EQ(ranked.front().candidate.id, "candidate_high");
}

TEST(Ranking, IsStableForTheSameInput) {
  std::vector<TuningResult> results = {
      resultWith("candidate_c", 100, 0.5, 10),
      resultWith("candidate_a", 100, 0.5, 10),
      resultWith("candidate_b", 100, 0.5, 10),
  };
  std::vector<TuningResult> ranked =
      rankTuningResults(results, latencyObjective());
  EXPECT_EQ(ids(ranked), (std::vector<std::string>{"candidate_a", "candidate_b",
                                                   "candidate_c"}));
  EXPECT_EQ(ids(rankTuningResults(results, latencyObjective())), ids(ranked));
}

//===----------------------------------------------------------------------===//
// Schedule YAML
//===----------------------------------------------------------------------===//

ScheduleRecord sampleRecord() {
  ScheduleRecord record;
  record.workload = "fused_swiglu";
  record.target = "x86-avx2-cpu";
  record.machine = "machines/x86-avx2-v2.yaml";
  record.shape.M = 8;
  record.shape.N = 64;
  record.shape.K = 64;
  record.candidate.id = "candidate_0123456789abcdef";
  record.candidate.values = {{"BM", 8},
                             {"BN", 64},
                             {"BK", 64},
                             {"pipeline_stages", 1},
                             {"vector_width", 8}};
  record.candidate.symbolicValues = {{"tile_layout", "row_major"},
                                     {"memory_path", "dram:sram:acc"},
                                     {"owner_mapping", "worker/vector_engine"},
                                     {"fragment_shape", "16x16x32"},
                                     {"tail_policy", "mask"}};
  record.tile.mBucket = 2;
  record.tile.workerTile = {8, 64, 64};
  record.tile.declaredFragment = {16, 16, 32};
  record.tile.fragmentShape = {8, 16, 32};
  record.tile.tileLayout = "row_major";
  record.tile.memoryPath = {"dram", "sram", "acc"};
  record.tile.outerOwner = "worker";
  record.tile.fragmentOwner = "vector_engine";
  record.tile.tailPolicy = "mask";
  record.tile.pipelineStages = 1;
  record.tile.vectorWidth = 8;
  record.metrics.predictedCycles = 123456;
  record.metrics.predictedNs = 123456.0;
  record.metrics.matrixUtilization = 0.82;
  record.metrics.dmaUtilization = 0.61;
  record.metrics.dramBytes = 67108864;
  record.metrics.sramBytes = 134217728;
  record.metrics.bottleneck = "matrix_engine";
  record.perfLevel = 1;
  return record;
}

std::string yamlOf(llvm::ArrayRef<ScheduleRecord> records) {
  std::string text;
  llvm::raw_string_ostream os(text);
  writeScheduleYaml(os, records);
  return text;
}

TEST(ScheduleRecord, WritesTheIdentityShapeAndTileDecisions) {
  std::string yaml = yamlOf(sampleRecord());

  EXPECT_NE(yaml.find("schema_version: 1\n"), std::string::npos);
  EXPECT_NE(yaml.find("workload: fused_swiglu\n"), std::string::npos);
  EXPECT_NE(yaml.find("target: x86-avx2-cpu\n"), std::string::npos);
  EXPECT_NE(yaml.find("machine: machines/x86-avx2-v2.yaml\n"),
            std::string::npos);
  EXPECT_NE(yaml.find("M_bucket: 2\n"), std::string::npos);
  EXPECT_NE(yaml.find("  M: 8\n"), std::string::npos);
  EXPECT_NE(yaml.find("  accumulator: f32\n"), std::string::npos);

  EXPECT_NE(yaml.find("id: candidate_0123456789abcdef\n"), std::string::npos);
  EXPECT_NE(yaml.find("    BM: 8\n"), std::string::npos);
  EXPECT_NE(yaml.find("      worker: [8, 64, 64]\n"), std::string::npos);
  EXPECT_NE(yaml.find("      fragment: [16, 16, 32]\n"), std::string::npos);
  EXPECT_NE(yaml.find("    memory_path: [dram, sram, acc]\n"),
            std::string::npos);
  EXPECT_NE(yaml.find("      outer: worker\n"), std::string::npos);
  EXPECT_NE(yaml.find("      fragment: vector_engine\n"), std::string::npos);
  EXPECT_NE(yaml.find("    tail_policy: mask\n"), std::string::npos);
  EXPECT_NE(yaml.find("    pipeline_stages: 1\n"), std::string::npos);
}

TEST(ScheduleRecord, WritesPredictedMetricsAndMeasurement) {
  std::string yaml = yamlOf(sampleRecord());
  EXPECT_NE(yaml.find("  perf_level: 1\n"), std::string::npos);
  EXPECT_NE(yaml.find("  predicted_cycles: 123456\n"), std::string::npos);
  EXPECT_NE(yaml.find("  matrix_utilization: 0.820\n"), std::string::npos);
  EXPECT_NE(yaml.find("  dram_bytes: 67108864\n"), std::string::npos);
  EXPECT_NE(yaml.find("  bottleneck: matrix_engine\n"), std::string::npos);
  EXPECT_NE(yaml.find("measurement:\n  measured: false\n"), std::string::npos);
}

TEST(ScheduleRecord, WritesMeasurementWhenPresent) {
  ScheduleRecord record = sampleRecord();
  record.measured = true;
  record.warmup = 5;
  record.repeat = 30;
  record.medianNs = 98765.0;
  record.measuredGflops = 540.0;

  std::string yaml = yamlOf(record);
  EXPECT_NE(yaml.find("  measured: true\n"), std::string::npos);
  EXPECT_NE(yaml.find("  warmup: 5\n"), std::string::npos);
  EXPECT_NE(yaml.find("  repeat: 30\n"), std::string::npos);
  EXPECT_NE(yaml.find("  median_ns: 98765.0\n"), std::string::npos);
  EXPECT_NE(yaml.find("  measured_gflops: 540.0\n"), std::string::npos);
}

TEST(ScheduleRecord, SeparatesMultipleRecordsIntoDocuments) {
  std::string yaml = yamlOf({sampleRecord(), sampleRecord()});
  EXPECT_NE(yaml.find("\n---\n"), std::string::npos);
  size_t first = yaml.find("schema_version: 1");
  ASSERT_NE(first, std::string::npos);
  EXPECT_NE(yaml.find("schema_version: 1", first + 1), std::string::npos);
}

//===----------------------------------------------------------------------===//
// Tuning session
//===----------------------------------------------------------------------===//

const machine::MachineModel &avx2() {
  static std::optional<machine::MachineModel> model = [] {
    auto loaded = machine::loadMachineModel(std::string(LLK_MACHINE_DIR) +
                                            "/x86-avx2-v2.yaml");
    if (!loaded) {
      ADD_FAILURE() << llvm::toString(loaded.takeError());
      return std::optional<machine::MachineModel>();
    }
    return std::optional<machine::MachineModel>(std::move(*loaded));
  }();
  return *model;
}

SearchParam integerParam(std::string name, std::vector<SearchChoice> choices) {
  return SearchParam{std::move(name), "integer", std::move(choices)};
}

SearchParam symbolicParam(std::string name, std::string kind,
                          std::vector<SearchChoice> choices) {
  return SearchParam{std::move(name), std::move(kind), std::move(choices)};
}

/// `bfloat16` is the top half of an `f32`, which is all the test needs to seed
/// operands and read results: 1.0 and 4096.0 are both exact in it.
uint16_t toBf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
}

float fromBf16(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

/// A space with two candidates that differ only in num_threads: 8 is legal on
/// the AVX2 model, 16 is not.
SearchSpace threadChoiceSpace() {
  SearchSpace space;
  space.name = "fused_swiglu_M8_N64_K64";
  space.workload = "fused_swiglu";
  space.params = {
      integerParam("BM", {SearchChoice(int64_t{8})}),
      integerParam("BN", {SearchChoice(int64_t{64})}),
      integerParam("BK", {SearchChoice(int64_t{64})}),
      integerParam("pipeline_stages", {SearchChoice(int64_t{1})}),
      integerParam("vector_width", {SearchChoice(int64_t{8})}),
      integerParam("num_threads",
                   {SearchChoice(int64_t{8}), SearchChoice(int64_t{16})}),
      symbolicParam("tile_layout", "layout", {SearchChoice("row_major")}),
      symbolicParam("memory_path", "memory_path",
                    {SearchChoice("dram:sram:acc")}),
      symbolicParam("owner_mapping", "owner_mapping",
                    {SearchChoice("worker/vector_engine")}),
      symbolicParam("fragment_shape", "fragment_shape",
                    {SearchChoice("16x16x32")}),
      symbolicParam("tail_policy", "tail_policy", {SearchChoice("mask")}),
  };
  space.constraints = {
      SearchConstraint{
          ConstraintKind::MappingExtent, {"num_threads", "BM", "BN"}, {}},
      SearchConstraint{ConstraintKind::SramCapacity, {"BM", "BN", "BK"}, {}},
      SearchConstraint{ConstraintKind::AccCapacity, {"BM", "BN"}, {}},
  };
  return space;
}

WorkloadShape swigluShape() {
  WorkloadShape shape;
  shape.M = 8;
  shape.N = 64;
  shape.K = 64;
  return shape;
}

std::unique_ptr<MLIRContext> perfContext() {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::micro::MicroDialect>();
  return std::make_unique<MLIRContext>(registry);
}

TEST(TuningSession, RanksLegalCandidatesAndRejectsTheRest) {
  auto context = perfContext();
  TuningSessionOptions options;
  options.topK = 10;
  options.machinePath = "machines/x86-avx2-v2.yaml";

  auto report = runTuningSession(*context, threadChoiceSpace(), swigluShape(),
                                 avx2(), options);
  ASSERT_TRUE(bool(report)) << llvm::toString(report.takeError());

  EXPECT_EQ(report->generated, 2u);
  ASSERT_EQ(report->ranked.size(), 1u);
  ASSERT_EQ(report->rejected.size(), 1u);
  EXPECT_EQ(report->ranked.front().result.candidate.values.at("num_threads"),
            8);
  EXPECT_EQ(report->rejected.front().candidate.values.at("num_threads"), 16);
  EXPECT_EQ(
      report->rejected.front().rejectionReason.rfind("mapping_extent:", 0), 0u);
  EXPECT_GT(report->ranked.front().result.metrics.predictedCycles, 0u);
  EXPECT_EQ(report->machineName, "x86-avx2");
}

TEST(TuningSession, IsDeterministicAcrossRuns) {
  auto context = perfContext();
  TuningSessionOptions options;
  options.machinePath = "machines/x86-avx2-v2.yaml";

  auto first = runTuningSession(*context, threadChoiceSpace(), swigluShape(),
                                avx2(), options);
  auto second = runTuningSession(*context, threadChoiceSpace(), swigluShape(),
                                 avx2(), options);
  ASSERT_TRUE(bool(first)) << llvm::toString(first.takeError());
  ASSERT_TRUE(bool(second)) << llvm::toString(second.takeError());

  ASSERT_EQ(first->ranked.size(), second->ranked.size());
  for (size_t i = 0; i < first->ranked.size(); ++i)
    EXPECT_EQ(first->ranked[i].result.candidate.id,
              second->ranked[i].result.candidate.id);
}

TEST(TuningSession, KeepsAtMostTopK) {
  SearchSpace space = threadChoiceSpace();
  // Three legal candidates.
  space.params[5] = integerParam(
      "num_threads", {SearchChoice(int64_t{1}), SearchChoice(int64_t{2}),
                      SearchChoice(int64_t{8}), SearchChoice(int64_t{16})});

  auto context = perfContext();
  TuningSessionOptions options;
  options.topK = 2;
  options.machinePath = "machines/x86-avx2-v2.yaml";
  options.perfLevel = 0;

  auto report =
      runTuningSession(*context, space, swigluShape(), avx2(), options);
  ASSERT_TRUE(bool(report)) << llvm::toString(report.takeError());
  EXPECT_EQ(report->ranked.size(), 2u);
  // Ordering is by predicted cycles, not by generation order.
  for (size_t i = 1; i < report->ranked.size(); ++i)
    EXPECT_LE(report->ranked[i - 1].result.metrics.predictedCycles,
              report->ranked[i].result.metrics.predictedCycles);
}

TEST(TuningSession, BuildsScheduleRecordsWithTileDecisions) {
  auto context = perfContext();
  TuningSessionOptions options;
  options.machinePath = "machines/x86-avx2-v2.yaml";

  SearchSpace space = threadChoiceSpace();
  WorkloadShape shape = swigluShape();
  auto report = runTuningSession(*context, space, shape, avx2(), options);
  ASSERT_TRUE(bool(report)) << llvm::toString(report.takeError());

  std::vector<ScheduleRecord> records =
      buildScheduleRecords(*report, space, shape);
  ASSERT_EQ(records.size(), 1u);

  const ScheduleRecord &record = records.front();
  EXPECT_EQ(record.schemaVersion, 1u);
  EXPECT_EQ(record.workload, "fused_swiglu");
  EXPECT_EQ(record.target, "x86-avx2");
  EXPECT_EQ(record.machine, "machines/x86-avx2-v2.yaml");
  EXPECT_EQ(record.shape.M, 8);
  EXPECT_EQ(record.candidate.values.at("BM"), 8);
  EXPECT_EQ(record.tile.workerTile, (std::vector<int64_t>{8, 64, 64}));
  EXPECT_EQ(record.tile.memoryPath,
            (std::vector<std::string>{"dram", "sram", "acc"}));
  EXPECT_EQ(record.tile.outerOwner, "worker");
  EXPECT_GT(record.metrics.predictedCycles, 0u);

  // The persisted record round-trips to YAML with the tile decisions.
  std::string yaml = yamlOf(records);
  EXPECT_NE(yaml.find("      worker: [8, 64, 64]\n"), std::string::npos);
  EXPECT_NE(yaml.find("workload: fused_swiglu\n"), std::string::npos);
}

//===----------------------------------------------------------------------===//
// Stage C6: the objective is checked, and candidates can be measured
//===----------------------------------------------------------------------===//

TEST(Objective, RejectsAMetricTheModelDoesNotProduce) {
  // `capacity_spill_bytes` is named by the design and not modelled. Ranking by
  // it would order every candidate equally and still look like a decision, so
  // it is a usage error rather than a silent zero.
  SearchObjective objective;
  objective.primaryMetric = "capacity_spill_bytes";
  llvm::Error error = validateObjective(objective);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("capacity_spill_bytes"),
            std::string::npos);

  // Refused wherever it appears, not only as the primary metric.
  SearchObjective secondary;
  secondary.primaryMetric = "latency_cycles";
  secondary.secondaryMetrics = {"not_a_metric"};
  EXPECT_TRUE(static_cast<bool>(validateObjective(secondary)));
}

TEST(Objective, AcceptsEveryMetricTheTunerProduces) {
  for (llvm::StringRef name :
       {"latency_cycles", "dram_bytes", "sram_bytes", "matrix_utilization",
        "dma_utilization", "measured_ns", "measured_gflops"}) {
    SearchObjective objective;
    objective.primaryMetric = name.str();
    EXPECT_FALSE(static_cast<bool>(validateObjective(objective))) << name.str();
  }
}

TEST(TuningSession, RefusesAnObjectiveItCannotRankBy) {
  auto context = perfContext();
  SearchSpace space = threadChoiceSpace();
  space.objective.primaryMetric = "capacity_spill_bytes";

  auto report = runTuningSession(*context, space, swigluShape(), avx2(), {});
  ASSERT_FALSE(static_cast<bool>(report));
  EXPECT_NE(llvm::toString(report.takeError()).find("capacity_spill_bytes"),
            std::string::npos);
}

TEST(Ranking, OrdersByANonLatencyPrimaryMetricThenASecondary) {
  // A has the smaller DRAM footprint, B the better utilization. Ranking by
  // dram_bytes must put A first even though B wins on utilization alone --
  // which is what proves the primary really is the primary.
  std::vector<TuningResult> results = {
      resultWith("a", /*cycles=*/100, /*matrixUtil=*/0.50, /*dramBytes=*/200),
      resultWith("b", /*cycles=*/100, /*matrixUtil=*/0.90, /*dramBytes=*/400),
  };
  SearchObjective objective;
  objective.primaryMetric = "dram_bytes";
  objective.direction = ObjectiveDirection::Minimize;
  objective.secondaryMetrics = {"matrix_utilization"};

  std::vector<TuningResult> ranked = rankTuningResults(results, objective);
  ASSERT_EQ(ranked.size(), 2u);
  EXPECT_EQ(ranked[0].candidate.id, "a");
  EXPECT_EQ(ranked[1].candidate.id, "b");

  // A tie on the primary falls through to the secondary, which is maximized.
  results[1].metrics.dramBytes = 200;
  ranked = rankTuningResults(results, objective);
  EXPECT_EQ(ranked[0].candidate.id, "b");
}

TEST(Measurement, AMissKeepsTheStaticScoreAndTheRanking) {
  auto context = perfContext();
  TuningSessionOptions options;
  options.measurement.measureTop = 1;
  options.measurement.provider = [](mlir::ModuleOp, const TuningResult &)
      -> llvm::Expected<std::optional<CandidateMetrics>> {
    // No measurement available: not a failure, just nothing to add.
    return std::optional<CandidateMetrics>();
  };

  auto report = runTuningSession(*context, threadChoiceSpace(), swigluShape(),
                                 avx2(), options);
  ASSERT_TRUE(bool(report)) << llvm::toString(report.takeError());

  // The candidate is still ranked, still legal, with its static cost standing.
  ASSERT_EQ(report->ranked.size(), 1u);
  EXPECT_FALSE(report->ranked.front().measured.has_value());
  EXPECT_GT(report->ranked.front().result.metrics.predictedCycles, 0u);
  EXPECT_TRUE(report->ranked.front().result.legal);
}

TEST(Measurement, AProviderThatObservedSomethingRecordsItWithItsIdentity) {
  auto context = perfContext();
  TuningSessionOptions options;
  options.measurement.measureTop = 1;
  options.measurement.targetIdentity = "x86-avx2";
  options.measurement.machineIdentity = "machines/x86-avx2-v2.yaml";
  options.measurement.abiIdentity = "memref2d-descriptors";
  options.measurement.provider = [](mlir::ModuleOp, const TuningResult &)
      -> llvm::Expected<std::optional<CandidateMetrics>> {
    CandidateMetrics observed;
    observed.measuredNs = 1234.5;
    observed.measuredGflops = 4.5;
    return std::optional<CandidateMetrics>(observed);
  };

  auto report = runTuningSession(*context, threadChoiceSpace(), swigluShape(),
                                 avx2(), options);
  ASSERT_TRUE(bool(report)) << llvm::toString(report.takeError());
  ASSERT_EQ(report->ranked.size(), 1u);

  const RankedCandidate &best = report->ranked.front();
  ASSERT_TRUE(best.measured.has_value());
  EXPECT_DOUBLE_EQ(best.measured->measuredNs.value(), 1234.5);
  // The identity travels with the number: a cycle count from another target or
  // another ABI is a different measurement, not the same one.
  EXPECT_EQ(best.measuredTarget, "x86-avx2");
  EXPECT_EQ(best.measuredMachine, "machines/x86-avx2-v2.yaml");
  EXPECT_EQ(best.measuredAbi, "memref2d-descriptors");
}

TEST(Measurement, ACandidateThatCannotBeCompiledIsRejectedWithItsReason) {
  auto context = perfContext();
  TuningSessionOptions options;
  options.measurement.measureTop = 1;
  options.measurement.provider = [](mlir::ModuleOp, const TuningResult &)
      -> llvm::Expected<std::optional<CandidateMetrics>> {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the target emitter refused the bundle");
  };

  auto report = runTuningSession(*context, threadChoiceSpace(), swigluShape(),
                                 avx2(), options);
  ASSERT_TRUE(bool(report)) << llvm::toString(report.takeError());

  // A plan that cannot become code is not a candidate, whatever it was
  // predicted to cost -- and the reason is kept, not summarized away.
  EXPECT_TRUE(report->ranked.empty());
  ASSERT_EQ(report->rejected.size(), 2u);
  const TuningResult *measuredRejection = nullptr;
  for (const TuningResult &rejected : report->rejected)
    if (rejected.rejectionReason.rfind("measurement:", 0) == 0)
      measuredRejection = &rejected;
  ASSERT_TRUE(measuredRejection);
  EXPECT_NE(measuredRejection->rejectionReason.find(
                "the target emitter refused the bundle"),
            std::string::npos);
}

TEST(Measurement, IsOffUnlessAProviderIsSupplied) {
  auto context = perfContext();
  // The default session is exactly the static ranking it always was: no
  // provider means no compilation, no invocation and no measurement.
  auto report = runTuningSession(*context, threadChoiceSpace(), swigluShape(),
                                 avx2(), {});
  ASSERT_TRUE(bool(report)) << llvm::toString(report.takeError());
  ASSERT_EQ(report->ranked.size(), 1u);
  EXPECT_FALSE(report->ranked.front().measured.has_value());
  EXPECT_EQ(report->schemaVersion, 1u);
}

TEST(Measurement, InvokesATopCandidateAndRecordsWhatItComputed) {
  auto context = perfContext();

  // The space's own dtypes: bf16 operands fit the AVX2 model's SRAM, and the
  // test hands over bf16 buffers to match.
  WorkloadShape shape = swigluShape();

  TuningSessionOptions options;
  options.measurement.measureTop = 1;
  options.measurement.targetIdentity = "x86-avx2";
  options.measurement.machineIdentity = "machines/x86-avx2-v2.yaml";
  options.measurement.abiIdentity = "memref2d-descriptors";

  unsigned invocations = 0;
  options.measurement.provider = [&](mlir::ModuleOp boundModule,
                                     const TuningResult &)
      -> llvm::Expected<std::optional<CandidateMetrics>> {
    // Compile the candidate the session bound. This is what makes a
    // measurement an answer about code that ran rather than a prediction about
    // code that might.
    std::string entry;
    boundModule->walk([&](mlir::Operation *op) {
      if (entry.empty() && op->getName().getStringRef() == "micro.kernel")
        entry = mlir::cast<mlir::StringAttr>(op->getAttr("sym_name")).str();
    });
    if (entry.empty())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "no kernel to compile");

    ::llk::MappedCompileOptions compileOptions;
    compileOptions.entrySymbol = entry;
    llvm::Expected<::llk::MappedCompilation> compiled =
        ::llk::compileConcreteMicroKernel(boundModule, compileOptions);
    if (!compiled)
      return compiled.takeError();
    if (!compiled->executable)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "no executable was produced");

    const ::llk::KernelAbi &abi = compiled->executable->abi();
    const uint16_t one = toBf16(1.0f);
    llvm::SmallVector<std::vector<uint16_t>, 4> inputStorage;
    inputStorage.reserve(abi.inputs.size());
    for (const ::llk::KernelAbi::Port &port : abi.inputs) {
      size_t elements = 1;
      for (int64_t dim : port.shape)
        elements *= static_cast<size_t>(dim);
      inputStorage.emplace_back(elements, one);
    }
    llvm::SmallVector<MemRef2D, 4> inputDescriptors;
    for (size_t i = 0; i < abi.inputs.size(); ++i) {
      const std::vector<int64_t> &portShape = abi.inputs[i].shape;
      std::vector<uint16_t> &storage = inputStorage[i];
      inputDescriptors.push_back(MemRef2D{storage.data(), storage.data(), 0,
                                          portShape[0], portShape[1],
                                          portShape[1], 1});
    }
    if (abi.outputs.size() != 1u)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "the kernel does not declare exactly one output");
    const std::vector<int64_t> &outShape = abi.outputs.front().shape;
    std::vector<uint16_t> output(static_cast<size_t>(outShape[0] * outShape[1]),
                                 toBf16(-1.0f));
    MemRef2D outputDescriptor{output.data(), output.data(), 0, outShape[0],
                              outShape[1],   outShape[1],   1};

    llvm::SmallVector<MemRef2D *, 4> inputPointers;
    for (MemRef2D &descriptor : inputDescriptors)
      inputPointers.push_back(&descriptor);
    llvm::SmallVector<MemRef2D *, 1> outputPointers{&outputDescriptor};

    const auto start = std::chrono::steady_clock::now();
    llvm::Error error =
        compiled->executable->invoke(inputPointers, outputPointers);
    const auto stop = std::chrono::steady_clock::now();
    if (error)
      return std::move(error);
    ++invocations;

    // Every output is silu(gate) * up over an all-ones operands: the gate and
    // the up projection both sum K=64 ones to 64, silu(64) is 64 to f32
    // precision, and the product is 4096. A number this specific is only right
    // if the copy, the contraction, the activation and the write-back all
    // happened -- which is what makes this a measurement rather than a timing.
    for (uint16_t value : output)
      EXPECT_NEAR(fromBf16(value), 4096.0f, 1.0f);

    CandidateMetrics measured;
    measured.measuredNs =
        std::chrono::duration<double, std::nano>(stop - start).count();
    measured.measuredGflops = 2.0 * static_cast<double>(shape.M) * shape.N *
                              shape.K / measured.measuredNs.value();
    return std::optional<CandidateMetrics>(measured);
  };

  auto report =
      runTuningSession(*context, threadChoiceSpace(), shape, avx2(), options);
  ASSERT_TRUE(bool(report)) << llvm::toString(report.takeError());
  ASSERT_EQ(report->ranked.size(), 1u);
  EXPECT_EQ(invocations, 1u);

  const RankedCandidate &best = report->ranked.front();
  ASSERT_TRUE(best.measured.has_value());
  EXPECT_GT(best.measured->measuredNs.value(), 0.0);
  EXPECT_EQ(best.measuredTarget, "x86-avx2");
  EXPECT_EQ(best.measuredAbi, "memref2d-descriptors");
}

} // namespace
} // namespace mlir::llk::perf
