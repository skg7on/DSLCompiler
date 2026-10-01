//===- machine_model_loader.cpp - MachineModel YAML loader tests ----------===//
//
// Covers issue #45:
//   - the shipped x86-avx2-cpu and generic-ai-accel-v1 models load
//   - structurally or semantically invalid YAML is rejected with a
//     deterministic diagnostic
//   - the typed model exposes what the L0/L1 cost models need: engine counts
//     and throughput, fragment shapes and dtypes, layouts, owner counts and
//     mappings, memory capacity/bandwidth/latency, DMA issue data, sync costs
//
//===----------------------------------------------------------------------===//

#include "LLK/Perf/MachineModel.h"
#include "LLK/Perf/MachineModelLoader.h"

#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

#include <string>

namespace mlir::llk::perf {
namespace {

using llvm::StringRef;

const std::string kValidYaml = R"YAML(
schema_version: 1
name: test-machine
description: "compact model used by the loader unit tests"
clock_hz: 1000000000

compute:
  owners:
    worker:
      count: 2
      maps_to: worker_threads
    lane:
      count: 4
      parent: worker
    vector_engine:
      count: 4
      parent: lane
  worker_threads:
    count: 2
  matrix_engines:
    - name: test-mxu
      count: 2
      owner: vector_engine
      tile_shapes:
        - [1, 8, 8]
        - [4, 8, 8]
      input_dtypes: [f32, bf16]
      accumulator_dtypes: [f32]
      issue_cycles: 1
      latency_cycles: 5
      flops_per_cycle: 16
      supported_layouts: [row_major, vectorized]
  vector_engines:
    - name: test-vec
      count: 2
      owner: vector_engine
      lanes:
        f32: 8
        bf16: 16
      issue_cycles: 1
      latency_cycles: 4
      supported_layouts: [row_major, vectorized]

memory:
  dram:
    capacity_bytes: 1024
    bandwidth_bytes_per_cycle: 32
    latency_cycles: 100
    supported_layouts: [row_major]
  sram:
    alias: l1
    capacity_bytes: 4096
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 4
    banks: 4
    supported_layouts: [row_major, blocked]

dma:
  engines: 1
  max_outstanding: 2
  setup_cycles: 4
  owner: worker
  maps_to: load_store_units
  paths:
    - src: dram
      dst: sram

sync:
  barrier_cycles: 32
  wait_cycles: 2
)YAML";

/// Returns kValidYaml with the first occurrence of `from` replaced by `to`.
std::string mutate(StringRef from, StringRef to) {
  std::string text = kValidYaml;
  size_t pos = text.find(from.str());
  EXPECT_NE(pos, std::string::npos)
      << "fixture does not contain: " << from.str();
  if (pos == std::string::npos)
    return text;
  text.replace(pos, from.size(), to.str());
  return text;
}

/// Loads `yaml`, expecting failure, and returns the diagnostic text.
std::string loadError(StringRef yaml) {
  auto model = parseMachineModel(yaml, "test-machine.yaml");
  if (model) {
    ADD_FAILURE() << "expected a load error, but the model loaded";
    return std::string();
  }
  return llvm::toString(model.takeError());
}

void expectErrorContains(StringRef yaml, StringRef needle) {
  std::string message = loadError(yaml);
  EXPECT_NE(message.find(needle.str()), std::string::npos)
      << "expected diagnostic containing '" << needle.str()
      << "', got: " << message;
}

const std::string &machineDir() {
  static const std::string dir = LLK_MACHINE_DIR;
  return dir;
}

std::string shippedPath(StringRef fileName) {
  return machineDir() + "/" + fileName.str();
}

//===----------------------------------------------------------------------===//
// Shipped models
//===----------------------------------------------------------------------===//

TEST(MachineModelLoader, LoadsAvx2CpuModel) {
  auto model = loadMachineModel(shippedPath("x86-avx2-cpu.yaml"));
  ASSERT_TRUE(static_cast<bool>(model)) << llvm::toString(model.takeError());

  EXPECT_EQ(model->schemaVersion, 1u);
  EXPECT_EQ(model->name, "x86-avx2-cpu");
  EXPECT_GT(model->clockHz, 0u);
  EXPECT_GE(model->workerThreads, 1u);

  // Owner counts and owner mappings.
  ASSERT_NE(model->findOwner("worker"), nullptr);
  EXPECT_EQ(model->findOwner("worker")->count, model->workerThreads);
  EXPECT_EQ(model->findOwner("worker")->mapsTo,
            std::optional<std::string>("worker_threads"));
  ASSERT_NE(model->findOwner("lane"), nullptr);
  EXPECT_EQ(model->findOwner("lane")->parent,
            std::optional<std::string>("worker"));
  ASSERT_NE(model->findOwner("vector_engine"), nullptr);
  EXPECT_EQ(model->getOwnerCount("vector_engine"),
            model->findOwner("vector_engine")->count);

  // AVX2 is a vector machine, not a systolic array: no PE-level owners.
  EXPECT_EQ(model->findOwner("pe"), nullptr);
  EXPECT_EQ(model->findOwner("pe_group"), nullptr);

  // Vector fragment shapes per dtype.
  const VectorEngineModel *vec = model->findVectorEngine("avx2-vector");
  ASSERT_NE(vec, nullptr);
  EXPECT_EQ(vec->count, model->workerThreads);
  EXPECT_EQ(vec->lanes.at("f32"), 8);
  EXPECT_EQ(vec->lanes.at("bf16"), 16);
  EXPECT_GT(vec->issueCycles, 0u);
  EXPECT_GT(vec->latencyCycles, 0u);
  EXPECT_FALSE(vec->supportedLayouts.empty());
  ASSERT_TRUE(vec->owner.has_value());
  EXPECT_EQ(*vec->owner, "vector_engine");

  // Matrix (FMA) fragment shapes and dtypes.
  const MatrixEngineModel *fma = model->findMatrixEngine("avx2-fma");
  ASSERT_NE(fma, nullptr);
  ASSERT_EQ(fma->tileShapes.size(), 2u);
  EXPECT_EQ(fma->tileShapes[0][0], 1);
  EXPECT_EQ(fma->tileShapes[0][1], 8);
  EXPECT_EQ(fma->tileShapes[0][2], 8);
  EXPECT_EQ(fma->tileShapes[1][0], 4);
  EXPECT_GT(fma->issueCycles, 0u);
  EXPECT_GT(fma->latencyCycles, 0u);
  ASSERT_TRUE(fma->flopsPerCycle.has_value());
  EXPECT_GT(*fma->flopsPerCycle, 0.0);

  // Memory hierarchy: capacities, bandwidth, latency, banks, layouts.
  const MemoryLevelModel *dram = model->findMemory("dram");
  ASSERT_NE(dram, nullptr);
  EXPECT_GT(dram->capacityBytes, 0u);
  EXPECT_GT(dram->bandwidthBytesPerCycle, 0.0);
  EXPECT_GT(dram->latencyCycles, 0u);

  const MemoryLevelModel *sram = model->findMemory("sram");
  ASSERT_NE(sram, nullptr);
  EXPECT_EQ(sram->alias, std::optional<std::string>("l1"));
  EXPECT_GT(sram->capacityBytes, 0u);
  ASSERT_TRUE(sram->banks.has_value());
  EXPECT_GT(*sram->banks, 0u);
  EXPECT_FALSE(sram->supportedLayouts.empty());

  EXPECT_EQ(model->findMemory("scratch"), nullptr);

  // DMA engines, queue depth, issue cost, and legal copy paths.
  EXPECT_GT(model->dma.engines, 0u);
  EXPECT_GT(model->dma.maxOutstanding, 0u);
  EXPECT_FALSE(model->dma.paths.empty());
  EXPECT_NE(model->findCopyPath("dram", "sram"), nullptr);
  EXPECT_NE(model->findCopyPath("sram", "dram"), nullptr);
  EXPECT_EQ(model->findCopyPath("acc", "dram"), nullptr);

  // Sync costs.
  EXPECT_GT(model->sync.barrierCycles, 0u);
}

TEST(MachineModelLoader, LoadsGenericAcceleratorModel) {
  auto model = loadMachineModel(shippedPath("generic-ai-accel-v1.yaml"));
  ASSERT_TRUE(static_cast<bool>(model)) << llvm::toString(model.takeError());

  EXPECT_EQ(model->name, "generic-ai-accel-v1");

  // Accelerator-style owner hierarchy.
  const OwnerModel *cluster = model->findOwner("cluster");
  ASSERT_NE(cluster, nullptr);
  EXPECT_EQ(cluster->parent, std::nullopt);

  const OwnerModel *core = model->findOwner("core");
  ASSERT_NE(core, nullptr);
  EXPECT_EQ(core->parent, std::optional<std::string>("cluster"));

  const OwnerModel *peGroup = model->findOwner("pe_group");
  ASSERT_NE(peGroup, nullptr);
  EXPECT_EQ(peGroup->parent, std::optional<std::string>("core"));

  const OwnerModel *pe = model->findOwner("pe");
  ASSERT_NE(pe, nullptr);
  EXPECT_EQ(pe->parent, std::optional<std::string>("pe_group"));

  const OwnerModel *matrixOwner = model->findOwner("matrix_engine");
  ASSERT_NE(matrixOwner, nullptr);
  EXPECT_EQ(matrixOwner->parent, std::optional<std::string>("pe"));

  EXPECT_GT(model->getOwnerCount("core"), 1u);
  EXPECT_GT(model->getOwnerCount("pe"), 1u);

  // Matrix fragments: systolic-style shapes with accelerator dtypes.
  const MatrixEngineModel *mxu = model->findMatrixEngine("mxu");
  ASSERT_NE(mxu, nullptr);
  ASSERT_FALSE(mxu->tileShapes.empty());
  EXPECT_EQ(mxu->tileShapes[0], (std::array<int64_t, 3>{16, 16, 32}));

  // Vector fragments.
  const VectorEngineModel *vpu = model->findVectorEngine("vpu");
  ASSERT_NE(vpu, nullptr);
  EXPECT_FALSE(vpu->lanes.empty());

  // DMA queue depth is deeper than a CPU's load/store path.
  EXPECT_GT(model->dma.engines, 1u);
  EXPECT_GT(model->dma.maxOutstanding, 1u);
}

//===----------------------------------------------------------------------===//
// Required schema fields
//===----------------------------------------------------------------------===//

TEST(MachineModelLoader, RejectsMissingSchemaVersion) {
  expectErrorContains(mutate("schema_version: 1\n", ""), "schema_version");
}

TEST(MachineModelLoader, RejectsUnsupportedSchemaVersion) {
  expectErrorContains(mutate("schema_version: 1", "schema_version: 7"),
                      "unsupported schema_version 7");
}

TEST(MachineModelLoader, RejectsMissingName) {
  expectErrorContains(mutate("name: test-machine\n", ""), "name");
}

TEST(MachineModelLoader, RejectsEmptyName) {
  expectErrorContains(mutate("name: test-machine", "name: \"\""),
                      "name must not be empty");
}

TEST(MachineModelLoader, RejectsMissingClock) {
  expectErrorContains(mutate("clock_hz: 1000000000\n", ""), "clock_hz");
}

TEST(MachineModelLoader, RejectsZeroClock) {
  expectErrorContains(mutate("clock_hz: 1000000000", "clock_hz: 0"),
                      "clock_hz must be positive");
}

TEST(MachineModelLoader, RejectsMissingMemorySection) {
  expectErrorContains(R"YAML(
schema_version: 1
name: no-memory
clock_hz: 1
compute:
  owners:
    worker:
      count: 1
  matrix_engines:
    - name: mxu
      count: 1
      tile_shapes:
        - [1, 1, 1]
      input_dtypes: [f32]
      accumulator_dtypes: [f32]
      issue_cycles: 1
      latency_cycles: 1
      supported_layouts: [row_major]
  vector_engines:
    - name: vec
      count: 1
      lanes:
        f32: 8
      issue_cycles: 1
      latency_cycles: 1
      supported_layouts: [row_major]
dma:
  engines: 1
  max_outstanding: 1
  paths:
    - src: dram
      dst: sram
sync:
  barrier_cycles: 1
  wait_cycles: 0
)YAML",
                      "memory");
}

TEST(MachineModelLoader, RejectsMissingComputeSection) {
  expectErrorContains(R"YAML(
schema_version: 1
name: no-compute
clock_hz: 1
memory:
  dram:
    capacity_bytes: 1
    bandwidth_bytes_per_cycle: 1
    latency_cycles: 1
    supported_layouts: [row_major]
dma:
  engines: 1
  max_outstanding: 1
  paths:
    - src: dram
      dst: dram
sync:
  barrier_cycles: 1
  wait_cycles: 0
)YAML",
                      "compute");
}

TEST(MachineModelLoader, RejectsUnknownTopLevelKey) {
  expectErrorContains(
      mutate("clock_hz: 1000000000", "clock_hz: 1000000000\nclock_rate_hz: 3"),
      "unknown key 'clock_rate_hz'");
}

TEST(MachineModelLoader, RejectsUnknownNestedKey) {
  expectErrorContains(mutate("    capacity_bytes: 4096",
                             "    banks_note: 4\n    capacity_bytes: 4096"),
                      "unknown key 'banks_note'");
}

//===----------------------------------------------------------------------===//
// Compute validation
//===----------------------------------------------------------------------===//

TEST(MachineModelLoader, RejectsZeroEngineCount) {
  expectErrorContains(mutate("      count: 2\n      owner: vector_engine",
                             "      count: 0\n      owner: vector_engine"),
                      "count must be positive");
}

TEST(MachineModelLoader, RejectsZeroWorkerThreads) {
  expectErrorContains(mutate("  worker_threads:\n    count: 2",
                             "  worker_threads:\n    count: 0"),
                      "worker_threads");
}

TEST(MachineModelLoader, RejectsUnknownOwner) {
  expectErrorContains(mutate("    - name: test-mxu\n      count: 2\n"
                             "      owner: vector_engine",
                             "    - name: test-mxu\n      count: 2\n"
                             "      owner: hypervisor"),
                      "unknown owner 'hypervisor'");
}

TEST(MachineModelLoader, RejectsUndeclaredOwner) {
  expectErrorContains(mutate("    - name: test-mxu\n      count: 2\n"
                             "      owner: vector_engine",
                             "    - name: test-mxu\n      count: 2\n"
                             "      owner: dma"),
                      "not declared in compute.owners");
}

TEST(MachineModelLoader, RejectsUnknownOwnerParent) {
  expectErrorContains(mutate("      count: 4\n      parent: worker",
                             "      count: 4\n      parent: socket"),
                      "unknown parent 'socket'");
}

TEST(MachineModelLoader, RejectsOwnerParentCycle) {
  // worker -> vector_engine, vector_engine -> lane, lane -> worker.
  expectErrorContains(mutate("      maps_to: worker_threads",
                             "      maps_to: worker_threads\n"
                             "      parent: vector_engine"),
                      "cycle");
}

TEST(MachineModelLoader, RejectsUnknownDType) {
  expectErrorContains(
      mutate("input_dtypes: [f32, bf16]", "input_dtypes: [f32, fp8]"),
      "unknown dtype 'fp8'");
}

TEST(MachineModelLoader, RejectsUnknownAccumulatorDType) {
  expectErrorContains(
      mutate("accumulator_dtypes: [f32]", "accumulator_dtypes: [f64]"),
      "unknown dtype 'f64'");
}

TEST(MachineModelLoader, RejectsUnknownVectorLaneDType) {
  expectErrorContains(mutate("        bf16: 16", "        fp4: 16"),
                      "unknown dtype 'fp4'");
}

TEST(MachineModelLoader, RejectsNonThreeDimensionalTileShape) {
  expectErrorContains(mutate("        - [4, 8, 8]", "        - [4, 8]"),
                      "exactly 3 dimensions");
}

TEST(MachineModelLoader, RejectsNonPositiveTileShape) {
  expectErrorContains(mutate("        - [4, 8, 8]", "        - [4, 0, 8]"),
                      "must be positive");
}

TEST(MachineModelLoader, RejectsUnknownLayout) {
  expectErrorContains(mutate("[row_major, blocked]", "[row_major, zigzag]"),
                      "unknown layout 'zigzag'");
}

TEST(MachineModelLoader, RejectsZeroIssueCycles) {
  expectErrorContains(mutate("      issue_cycles: 1\n      latency_cycles: 5",
                             "      issue_cycles: 0\n      latency_cycles: 5"),
                      "issue_cycles must be positive");
}

TEST(MachineModelLoader, RejectsZeroLatencyCycles) {
  expectErrorContains(mutate("      issue_cycles: 1\n      latency_cycles: 5",
                             "      issue_cycles: 1\n      latency_cycles: 0"),
                      "latency_cycles must be positive");
}

TEST(MachineModelLoader, RejectsDuplicateEngineName) {
  expectErrorContains(mutate("    - name: test-vec", "    - name: test-mxu"),
                      "duplicate resource name 'test-mxu'");
}

TEST(MachineModelLoader, RejectsEmptyEngineSections) {
  expectErrorContains(R"YAML(
schema_version: 1
name: no-engines
clock_hz: 1
compute:
  owners:
    worker:
      count: 1
  matrix_engines: []
  vector_engines: []
memory:
  dram:
    capacity_bytes: 1
    bandwidth_bytes_per_cycle: 1
    latency_cycles: 1
    supported_layouts: [row_major]
  sram:
    capacity_bytes: 1
    bandwidth_bytes_per_cycle: 1
    latency_cycles: 1
    supported_layouts: [row_major]
dma:
  engines: 1
  max_outstanding: 1
  paths:
    - src: dram
      dst: sram
sync:
  barrier_cycles: 1
  wait_cycles: 0
)YAML",
                      "no compute engines");
}

//===----------------------------------------------------------------------===//
// Memory and DMA validation
//===----------------------------------------------------------------------===//

TEST(MachineModelLoader, RejectsZeroMemoryCapacity) {
  expectErrorContains(
      mutate("    capacity_bytes: 4096", "    capacity_bytes: 0"),
      "capacity_bytes must be positive");
}

TEST(MachineModelLoader, RejectsNonPositiveBandwidth) {
  expectErrorContains(mutate("    bandwidth_bytes_per_cycle: 32",
                             "    bandwidth_bytes_per_cycle: 0"),
                      "bandwidth_bytes_per_cycle must be positive");
}

TEST(MachineModelLoader, RejectsNonPositiveMemoryLatency) {
  expectErrorContains(
      mutate("    latency_cycles: 100", "    latency_cycles: 0"),
      "latency_cycles must be positive");
}

TEST(MachineModelLoader, RejectsUnknownMemorySpace) {
  expectErrorContains(mutate("  sram:\n", "  registers:\n"),
                      "unknown memory space 'registers'");
}

TEST(MachineModelLoader, RejectsAliasCollidingWithLevelName) {
  expectErrorContains(mutate("    alias: l1", "    alias: dram"),
                      "conflicts with memory level");
}

TEST(MachineModelLoader, RejectsDuplicateCopyPath) {
  expectErrorContains(mutate("    - src: dram\n      dst: sram",
                             "    - src: dram\n      dst: sram\n"
                             "    - src: dram\n      dst: sram"),
                      "duplicate copy path");
}

TEST(MachineModelLoader, RejectsIdenticalCopyPathEndpoints) {
  expectErrorContains(mutate("    - src: dram\n      dst: sram",
                             "    - src: dram\n      dst: dram"),
                      "must differ");
}

TEST(MachineModelLoader, RejectsUnknownCopyPathEndpoint) {
  expectErrorContains(mutate("      dst: sram", "      dst: registers"),
                      "unknown memory space 'registers'");
}

TEST(MachineModelLoader, RejectsZeroDmaEngines) {
  expectErrorContains(mutate("  engines: 1\n  max_outstanding: 2",
                             "  engines: 0\n  max_outstanding: 2"),
                      "engines must be positive");
}

TEST(MachineModelLoader, RejectsZeroDmaOutstanding) {
  expectErrorContains(mutate("  engines: 1\n  max_outstanding: 2",
                             "  engines: 1\n  max_outstanding: 0"),
                      "max_outstanding must be positive");
}

//===----------------------------------------------------------------------===//
// Optional fields and defaults
//===----------------------------------------------------------------------===//

TEST(MachineModelLoader, DefaultsWorkerThreadsToOne) {
  auto model = parseMachineModel(
      mutate("  worker_threads:\n    count: 2\n", ""), "test-machine.yaml");
  ASSERT_TRUE(static_cast<bool>(model)) << llvm::toString(model.takeError());
  EXPECT_EQ(model->workerThreads, 1u);
}

TEST(MachineModelLoader, WorkerThreadsIsIndependentOfWorkerOwnerCount) {
  auto model = parseMachineModel(mutate("  worker_threads:\n    count: 2",
                                        "  worker_threads:\n    count: 3"),
                                 "test-machine.yaml");
  ASSERT_TRUE(static_cast<bool>(model)) << llvm::toString(model.takeError());
  EXPECT_EQ(model->workerThreads, 3u);
  EXPECT_EQ(model->getOwnerCount("worker"), 2u);
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

TEST(MachineModelLoader, DiagnosticCarriesSourceLocation) {
  std::string message =
      loadError(mutate("clock_hz: 1000000000", "clock_hz: 0"));
  EXPECT_NE(message.find("test-machine.yaml"), std::string::npos) << message;
}

TEST(MachineModelLoader, ReportsUnreadableFile) {
  auto model = loadMachineModel("/nonexistent/machines/nope.yaml");
  ASSERT_FALSE(static_cast<bool>(model));
  std::string message = llvm::toString(model.takeError());
  EXPECT_NE(message.find("/nonexistent/machines/nope.yaml"), std::string::npos)
      << message;
}

TEST(MachineModelLoader, ReportsMalformedYaml) {
  expectErrorContains("schema_version: 1\nname: [unterminated\n", "error");
}

TEST(MachineModel, VerifyRejectsZeroClockOnHandBuiltModel) {
  MachineModel model;
  model.schemaVersion = 1;
  model.name = "hand-built";
  model.clockHz = 0;

  llvm::Error error = verifyMachineModel(model);
  ASSERT_TRUE(static_cast<bool>(error));
  std::string message = llvm::toString(std::move(error));
  EXPECT_NE(message.find("clock_hz"), std::string::npos) << message;
}

TEST(MachineModel, VerifyAcceptsHandBuiltModel) {
  MachineModel model;
  model.schemaVersion = 1;
  model.name = "hand-built";
  model.clockHz = 1;
  model.workerThreads = 1;
  model.owners.push_back({"worker", 1, std::nullopt, std::nullopt});
  model.matrixEngines.push_back({"mxu",
                                 1,
                                 {{1, 1, 1}},
                                 {"f32"},
                                 {"f32"},
                                 1,
                                 1,
                                 1.0,
                                 {"row_major"},
                                 std::optional<std::string>("worker")});
  model.memory["dram"] = {"dram",       std::nullopt, 1, 1.0, 1,
                          std::nullopt, {"row_major"}};
  model.memory["sram"] = {"sram",       std::nullopt, 1, 1.0, 1,
                          std::nullopt, {"row_major"}};
  model.dma.engines = 1;
  model.dma.maxOutstanding = 1;
  model.dma.paths.push_back({"dram", "sram", std::nullopt, std::nullopt});

  EXPECT_FALSE(static_cast<bool>(verifyMachineModel(model)));
}

} // namespace
} // namespace mlir::llk::perf
