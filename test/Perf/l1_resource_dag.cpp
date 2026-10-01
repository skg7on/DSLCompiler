//===- l1_resource_dag.cpp - L1 resource scheduler tests ------------------===//
//
// Covers issue #46, L1 half:
//   - copy -> wait -> mma is scheduled in that order
//   - independent copies overlap once the machine has more than one DMA engine
//   - pipelining a loop overlaps movement with compute
//   - owner counts cap how many execution tiles run at once
//   - the bottleneck classification is stable and names the busiest resource
//
// These tests build a small machine inline so the numbers under test come from
// the scheduler rather than from the shipped calibration seeds. The machine is
// deliberately extreme -- one flop per cycle per engine, long DMA transfers --
// so that a scheduling difference is visible as a doubling, not as a rounding
// error.
//
//===----------------------------------------------------------------------===//

#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Perf/MachineModelLoader.h"
#include "LLK/Perf/MicroCostModel.h"
#include "LLK/Perf/MicroDAG.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

namespace mlir::llk::perf {
namespace {

struct Parsed {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::Operation *kernel = nullptr;
};

std::unique_ptr<Parsed> parseKernel(llvm::StringRef source,
                                    llvm::StringRef symbol = "") {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::micro::MicroDialect>();

  auto parsed = std::make_unique<Parsed>();
  parsed->context = std::make_unique<mlir::MLIRContext>(registry);
  parsed->module = mlir::parseSourceString<mlir::ModuleOp>(
      source, mlir::ParserConfig(parsed->context.get()));
  if (!parsed->module) {
    ADD_FAILURE() << "the fixture does not parse";
    return nullptr;
  }
  if (mlir::failed(mlir::verify(*parsed->module))) {
    ADD_FAILURE() << "the fixture does not verify";
    return nullptr;
  }

  auto kernel = findMicroKernel(parsed->module.get(), symbol);
  if (!kernel) {
    ADD_FAILURE() << llvm::toString(kernel.takeError());
    return nullptr;
  }
  parsed->kernel = *kernel;
  return parsed;
}

/// A machine with one flop per cycle per matrix engine, a wide accumulator, and
/// DMA slots the caller chooses. Only the parts the scheduler reads are set.
std::string testMachine(unsigned dmaEngines, unsigned workers,
                        unsigned matrixEngines) {
  return llvm::formatv(R"YAML(
schema_version: 1
name: test-machine
clock_hz: 1000000000

compute:
  owners:
    worker:
      count: {0}
  matrix_engines:
    - name: mxu
      count: {1}
      tile_shapes:
        - [1, 1, 1]
      input_dtypes: [f32, bf16]
      accumulator_dtypes: [f32]
      issue_cycles: 1
      latency_cycles: 1
      flops_per_cycle: 1
      supported_layouts: [row_major]
  vector_engines:
    - name: vpu
      count: 1
      lanes:
        f32: 8
        bf16: 16
      issue_cycles: 1
      latency_cycles: 1
      supported_layouts: [row_major]

memory:
  dram:
    capacity_bytes: 1048576
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 1
    supported_layouts: [row_major]
  sram:
    capacity_bytes: 65536
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 1
    supported_layouts: [row_major]
  acc:
    capacity_bytes: 65536
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 1
    supported_layouts: [row_major]

dma:
  engines: {2}
  max_outstanding: 1
  paths:
    - src: dram
      dst: sram
    - src: sram
      dst: dram
    - src: sram
      dst: acc
    - src: acc
      dst: sram

sync:
  barrier_cycles: 1
  wait_cycles: 0
)YAML",
                       workers, matrixEngines, dmaEngines);
}

MachineModel parseMachine(llvm::StringRef yaml) {
  auto model = parseMachineModel(yaml, "test-machine.yaml");
  if (!model) {
    ADD_FAILURE() << llvm::toString(model.takeError());
    return MachineModel();
  }
  return *model;
}

/// Two DRAM reads staged through SRAM, a wait, then one 16x16x32 MMA.
constexpr const char *kChainKernel = R"MLIR(
module {
  micro.kernel @chain {
    %a_ext = tensor.empty() : tensor<16x32xbf16>
    %b_ext = tensor.empty() : tensor<32x16xbf16>
    %a_tile, %a_tok = micro.async_copy %a_ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<16x32xbf16> -> tensor<16x32xbf16>, !micro.async_token
    %b_tile, %b_tok = micro.async_copy %b_ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<32x16xbf16> -> tensor<32x16xbf16>, !micro.async_token
    micro.wait %a_tok, %b_tok
    %a_frag = micro.tile_view %a_tile {shape = array<i64: 16, 32>} : tensor<16x32xbf16> -> !micro.tile<16x32xbf16, memory = #micro.memory<sram>>
    %b_frag = micro.tile_view %b_tile {shape = array<i64: 32, 16>} : tensor<32x16xbf16> -> !micro.tile<32x16xbf16, memory = #micro.memory<sram>>
    %acc = micro.tile_alloc : !micro.tile<16x16xf32, memory = #micro.memory<acc>>
    %result = micro.mma %a_frag, %b_frag, %acc {shape = array<i64: 16, 16, 32>, input = #micro.dtype<bf16>, accumulator = #micro.dtype<f32>} : !micro.tile<16x32xbf16, memory = #micro.memory<sram>>, !micro.tile<32x16xbf16, memory = #micro.memory<sram>>, !micro.tile<16x16xf32, memory = #micro.memory<acc>> -> !micro.tile<16x16xf32, memory = #micro.memory<acc>>
    micro.yield
  }
}
)MLIR";

/// The same kernel with the MMA removed, so only the DMA stream is left.
constexpr const char *kCopiesKernel = R"MLIR(
module {
  micro.kernel @copies {
    %a_ext = tensor.empty() : tensor<16x32xbf16>
    %b_ext = tensor.empty() : tensor<32x16xbf16>
    %a_tile, %a_tok = micro.async_copy %a_ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<16x32xbf16> -> tensor<16x32xbf16>, !micro.async_token
    %b_tile, %b_tok = micro.async_copy %b_ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<32x16xbf16> -> tensor<32x16xbf16>, !micro.async_token
    micro.wait %a_tok, %b_tok
    micro.yield
  }
}
)MLIR";

/// A loop whose body is a pipelined copy/wait/vector chain. `stages` is the one
/// number the pipeline tests vary.
constexpr const char *kPipelinedKernel = R"MLIR(
module {
  micro.kernel @pipelined {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    micro.for %i = %c0 to %c4 step %c1 {
      micro.pipeline stages = 2 {
        %a_ext = tensor.empty() : tensor<16x32xbf16>
        %a_tile, %a_tok = micro.async_copy %a_ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<16x32xbf16> -> tensor<16x32xbf16>, !micro.async_token
        micro.wait %a_tok
        %a_frag = micro.tile_view %a_tile {shape = array<i64: 16, 32>} : tensor<16x32xbf16> -> !micro.tile<16x32xbf16, memory = #micro.memory<sram>>
        %activated = micro.vector "silu" %a_frag : !micro.tile<16x32xbf16, memory = #micro.memory<sram>> -> !micro.tile<16x32xbf16, memory = #micro.memory<sram>>
      }
    }
    micro.yield
  }
}
)MLIR";

/// Four independent MMA tiles spread over the machine's workers.
constexpr const char *kSpatialKernel = R"MLIR(
module {
  micro.kernel @spatial {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %acc = micro.tile_alloc : !micro.tile<16x16xf32, memory = #micro.memory<acc>>
    %a_frag = micro.tile_alloc : !micro.tile<16x32xbf16, memory = #micro.memory<sram>>
    %b_frag = micro.tile_alloc : !micro.tile<32x16xbf16, memory = #micro.memory<sram>>
    micro.spatial_for %w = %c0 to %c4 step %c1 map = #micro.map<worker> {
      %result = micro.mma %a_frag, %b_frag, %acc {shape = array<i64: 16, 16, 32>, input = #micro.dtype<bf16>, accumulator = #micro.dtype<f32>} : !micro.tile<16x32xbf16, memory = #micro.memory<sram>>, !micro.tile<32x16xbf16, memory = #micro.memory<sram>>, !micro.tile<16x16xf32, memory = #micro.memory<acc>> -> !micro.tile<16x16xf32, memory = #micro.memory<acc>>
    }
    micro.yield
  }
}
)MLIR";

std::string withStages(llvm::StringRef source, unsigned stages) {
  std::string text = source.str();
  const std::string from = "stages = 2";
  size_t position = text.find(from);
  EXPECT_NE(position, std::string::npos);
  if (position != std::string::npos)
    text.replace(position, from.size(), "stages = " + std::to_string(stages));
  return text;
}

//===----------------------------------------------------------------------===//
// Ordering
//===----------------------------------------------------------------------===//

TEST(L1ResourceDag, CopyWaitMmaIsScheduledInOrder) {
  auto parsed = parseKernel(kChainKernel);
  ASSERT_TRUE(parsed);
  MachineModel model = parseMachine(testMachine(/*dmaEngines=*/1, 2, 4));

  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  ASSERT_EQ(dag->events.size(), 4u);
  ASSERT_EQ(dag->events[0].kind, EventKind::AsyncCopy);
  ASSERT_EQ(dag->events[1].kind, EventKind::AsyncCopy);
  ASSERT_EQ(dag->events[2].kind, EventKind::Wait);
  ASSERT_EQ(dag->events[3].kind, EventKind::Mma);

  // The wait depends on both copies, not just the one before it.
  EXPECT_EQ(dag->events[2].deps.size(), 2u);

  L1Report report = scheduleL1(*dag, model);
  ASSERT_EQ(report.schedule.size(), 4u);
  EXPECT_LE(report.schedule[0].finish, report.schedule[2].start);
  EXPECT_LE(report.schedule[1].finish, report.schedule[2].start);
  EXPECT_LE(report.schedule[2].finish, report.schedule[3].start);
  EXPECT_EQ(report.predictedCycles, report.schedule[3].finish);
}

TEST(L1ResourceDag, IndependentCopiesOverlapWithMoreDmaEngines) {
  auto parsed = parseKernel(kCopiesKernel);
  ASSERT_TRUE(parsed);

  MachineModel serialMachine =
      parseMachine(testMachine(/*dmaEngines=*/1, 2, 4));
  MachineModel parallelMachine =
      parseMachine(testMachine(/*dmaEngines=*/2, 2, 4));

  auto dag = buildMicroDAG(parsed->kernel, serialMachine);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());

  L1Report serial = scheduleL1(*dag, serialMachine);
  L1Report parallel = scheduleL1(*dag, parallelMachine);

  // With one engine the second copy waits for the first; with two they start
  // together, because neither depends on the other.
  EXPECT_EQ(serial.schedule[1].start, serial.schedule[0].finish);
  EXPECT_EQ(parallel.schedule[0].start, 0u);
  EXPECT_EQ(parallel.schedule[1].start, 0u);
  EXPECT_LT(parallel.predictedCycles, serial.predictedCycles);
}

TEST(L1ResourceDag, PipeliningOverlapsMovementWithCompute) {
  auto parsed = parseKernel(withStages(kPipelinedKernel, 2));
  ASSERT_TRUE(parsed);
  MachineModel model = parseMachine(testMachine(/*dmaEngines=*/1, 2, 4));

  auto pipelined = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(pipelined))
      << llvm::toString(pipelined.takeError());
  L1Report pipelinedReport = scheduleL1(*pipelined, model);

  auto serialParsed = parseKernel(withStages(kPipelinedKernel, 1));
  ASSERT_TRUE(serialParsed);
  auto serial = buildMicroDAG(serialParsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(serial)) << llvm::toString(serial.takeError());
  L1Report serialReport = scheduleL1(*serial, model);

  EXPECT_LT(pipelinedReport.predictedCycles, serialReport.predictedCycles);
  EXPECT_GT(pipelinedReport.overlapEfficiency, 0.0);
  EXPECT_EQ(serialReport.overlapEfficiency, 0.0);
}

TEST(L1ResourceDag, OwnerCountCapsConcurrentExecutionTiles) {
  auto parsed = parseKernel(kSpatialKernel);
  ASSERT_TRUE(parsed);

  MachineModel wideMachine = parseMachine(testMachine(/*dmaEngines=*/1, 4, 4));
  MachineModel narrowMachine =
      parseMachine(testMachine(/*dmaEngines=*/1, 2, 4));

  auto dag = buildMicroDAG(parsed->kernel, wideMachine);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  ASSERT_EQ(dag->events.size(), 4u);

  L1Report wide = scheduleL1(*dag, wideMachine);
  L1Report narrow = scheduleL1(*dag, narrowMachine);

  // Four workers run all four tiles at once; two workers need two rounds.
  EXPECT_EQ(wide.predictedCycles, dag->events[0].minCycles);
  EXPECT_EQ(narrow.predictedCycles, 2 * dag->events[0].minCycles);

  // An owner cap is a real schedule constraint, not something L0 knows about,
  // so the static bound stays below it either way.
  EXPECT_LE(computeL0StaticBound(*dag, narrowMachine).predictedCycles,
            narrow.predictedCycles);
  EXPECT_LE(computeL0StaticBound(*dag, wideMachine).predictedCycles,
            wide.predictedCycles);
}

//===----------------------------------------------------------------------===//
// Reporting
//===----------------------------------------------------------------------===//

TEST(L1ResourceDag, BottleneckNamesTheBusiestResource) {
  auto copies = parseKernel(kCopiesKernel);
  ASSERT_TRUE(copies);
  MachineModel model = parseMachine(testMachine(/*dmaEngines=*/1, 2, 4));

  auto copyDag = buildMicroDAG(copies->kernel, model);
  ASSERT_TRUE(static_cast<bool>(copyDag))
      << llvm::toString(copyDag.takeError());
  L1Report copyReport = scheduleL1(*copyDag, model);

  // Nothing but movement, on a single engine, so DMA is saturated.
  EXPECT_GT(copyReport.dmaUtilization, 0.75);
  EXPECT_EQ(copyReport.bottleneck, "dma");

  auto chain = parseKernel(kChainKernel);
  ASSERT_TRUE(chain);
  auto chainDag = buildMicroDAG(chain->kernel, model);
  ASSERT_TRUE(static_cast<bool>(chainDag))
      << llvm::toString(chainDag.takeError());
  L1Report chainReport = scheduleL1(*chainDag, model);

  // The same schedule must classify the same way every time.
  L1Report repeated = scheduleL1(*chainDag, model);
  EXPECT_EQ(chainReport.bottleneck, repeated.bottleneck);
  EXPECT_EQ(chainReport.predictedCycles, repeated.predictedCycles);
  EXPECT_GT(chainReport.matrixUtilization, 0.0);
  EXPECT_GE(chainReport.dramBandwidthUtilization, 0.0);
}

TEST(L1ResourceDag, ScheduleBeatsTheStaticBoundOnlyWhenItOverlaps) {
  auto parsed = parseKernel(kChainKernel);
  ASSERT_TRUE(parsed);
  MachineModel model = parseMachine(testMachine(/*dmaEngines=*/1, 2, 4));

  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());

  L0Report bound = computeL0StaticBound(*dag, model);
  L1Report schedule = scheduleL1(*dag, model);

  // L0 assumes perfect overlap, so the real schedule can only be slower.
  EXPECT_GE(schedule.predictedCycles, bound.predictedCycles);
  EXPECT_DOUBLE_EQ(schedule.predictedNs,
                   static_cast<double>(schedule.predictedCycles) * 1e9 / 1e9);
}

} // namespace
} // namespace mlir::llk::perf
