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

#include "LLK/Machine/MachineModelLoader.h"
#include "LLK/Perf/ScheduleRecord.h"
#include "LLK/Perf/TuningSession.h"

#include "LLK/Dialect/Micro/MicroDialect.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

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

} // namespace
} // namespace mlir::llk::perf
