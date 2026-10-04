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
#include "LLK/Machine/MachineModelLoader.h"
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
///
/// Built in the v2 shape: `workers` are executors, `matrixEngines` and
/// `dmaEngines` are counts of concrete capability and transfer nodes, which is
/// what the scheduler's slot arithmetic reads.
std::string testMachine(unsigned dmaEngines, unsigned workers,
                        unsigned matrixEngines) {
  std::string yaml = "schema: llk.machine.v2\n"
                     "target: test-machine\n"
                     "clock_hz: 1000000000\n"
                     "worker_threads: " +
                     std::to_string(workers) +
                     "\nsync:\n  barrier_cycles: 1\n  wait_cycles: 0\n"
                     "executors:\n"
                     "  - id: cluster.0\n    kind: cluster\n";
  for (unsigned i = 0; i < workers; ++i)
    yaml += "  - id: worker." + std::to_string(i) +
            "\n    kind: worker\n    parent: cluster.0\n";

  yaml += "memories:\n";
  for (const auto &level : {std::pair<const char *, uint64_t>{"dram", 1048576},
                            {"l2", 262144},
                            {"sram", 65536},
                            {"acc", 65536}}) {
    yaml += "  - id: " + std::string(level.first) +
            ".0\n    kind: " + level.first +
            "\n    visible_from: cluster.0\n    capacity_bytes: " +
            std::to_string(level.second) +
            "\n    alignment_bytes: 64\n"
            "    supported_layouts: [row_major]\n"
            "    bandwidth_bytes_per_cycle: 64\n    latency_cycles: 1\n";
  }

  // One capability node carrying the whole multiplicity: v1's
  // `matrix_engines: count: N` is one engine *class*, and a schedule that
  // names it must share all N slots rather than get one.
  yaml += "compute:\n";
  yaml += "  - id: mxu\n    kind: matrix_engine\n    attached_to: worker.0\n"
          "    element_types: [f32, bf16]\n    accumulator_dtypes: [f32]\n"
          "    shapes: [[1, 1, 1]]\n    issue_cycles: 1\n"
          "    latency_cycles: 1\n    throughput_per_cycle: 1\n"
          "    concurrency: " +
          std::to_string(matrixEngines) +
          "\n    supported_layouts: [row_major]\n";
  yaml += "  - id: vpu\n    kind: vector_engine\n    attached_to: worker.0\n"
          "    element_types: [f32, bf16]\n    shapes: [[8]]\n"
          "    lanes: {f32: 8, bf16: 16}\n    issue_cycles: 1\n"
          "    latency_cycles: 1\n    supported_layouts: [row_major]\n";

  yaml += "transfer_engines:\n";
  for (unsigned i = 0; i < dmaEngines; ++i)
    yaml += "  - id: dma." + std::to_string(i) +
            "\n    kind: dma\n    attached_to: cluster.0\n"
            "    count: 1\n    max_outstanding: 1\n";

  yaml += "links:\n";
  const char *paths[6][2] = {{"dram", "sram"}, {"sram", "dram"},
                             {"sram", "acc"},  {"acc", "sram"},
                             {"dram", "l2"},   {"l2", "sram"}};
  for (unsigned i = 0; i < 6; ++i)
    yaml += "  - id: " + std::string(paths[i][0]) + "_to_" + paths[i][1] +
            ".0\n    source: " + paths[i][0] +
            ".0\n    destination: " + paths[i][1] +
            ".0\n    bandwidth_bytes_per_cycle: 64\n    latency_cycles: 1\n"
            "    transaction_bytes: 64\n    transfer_engines: [dma.0]\n";
  return yaml;
}

machine::MachineModel parseMachine(llvm::StringRef yaml) {
  auto model = machine::parseMachineModel(yaml, "test-machine.yaml");
  if (!model) {
    ADD_FAILURE() << llvm::toString(model.takeError());
    return machine::MachineModel();
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
  machine::MachineModel model =
      parseMachine(testMachine(/*dmaEngines=*/1, 2, 4));

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

  machine::MachineModel serialMachine =
      parseMachine(testMachine(/*dmaEngines=*/1, 2, 4));
  machine::MachineModel parallelMachine =
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
  machine::MachineModel model =
      parseMachine(testMachine(/*dmaEngines=*/1, 2, 4));

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

  machine::MachineModel wideMachine =
      parseMachine(testMachine(/*dmaEngines=*/1, 4, 4));
  machine::MachineModel narrowMachine =
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

TEST(L1ResourceDag, OwnerOccupancyCanBeTheBottleneck) {
  auto parsed = parseKernel(kSpatialKernel);
  ASSERT_TRUE(parsed);
  // One worker and plenty of matrix engines: nothing but the owner limits how
  // much of the kernel runs at once, so that is the bottleneck.
  machine::MachineModel model =
      parseMachine(testMachine(/*dmaEngines=*/1, 1, 4));

  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  L1Report report = scheduleL1(*dag, model);

  EXPECT_DOUBLE_EQ(report.matrixUtilization, 0.25);
  EXPECT_EQ(report.bottleneck, "owner_occupancy");
}

TEST(L1ResourceDag, NamingAnEngineDoesNotDuplicateItsSlots) {
  // Eight independent MMAs, half naming the machine's only matrix engine and
  // half leaving it implicit. All eight must draw on the same four engines: if
  // the two spellings landed in different pools, each would be sized as if it
  // owned the machine and the schedule would run twice the hardware it has.
  std::string body;
  for (unsigned i = 0; i < 8; ++i) {
    std::string engine = (i % 2 == 0) ? ", engine = \"mxu\"" : std::string();
    body += "    %r" + std::to_string(i) +
            " = micro.mma %a, %b, %c {shape = array<i64: 1, 1, 1>, input = "
            "#micro.dtype<bf16>, accumulator = #micro.dtype<f32>" +
            engine +
            "} : !micro.tile<1x1xbf16, memory = #micro.memory<sram>>, "
            "!micro.tile<1x1xbf16, memory = #micro.memory<sram>>, "
            "!micro.tile<1x1xf32, memory = #micro.memory<acc>> -> "
            "!micro.tile<1x1xf32, memory = #micro.memory<acc>>\n";
  }
  std::string source =
      "module {\n  micro.kernel @mixed {\n"
      "    %a = micro.tile_alloc : !micro.tile<1x1xbf16, memory = "
      "#micro.memory<sram>>\n"
      "    %b = micro.tile_alloc : !micro.tile<1x1xbf16, memory = "
      "#micro.memory<sram>>\n"
      "    %c = micro.tile_alloc : !micro.tile<1x1xf32, memory = "
      "#micro.memory<acc>>\n" +
      body + "    micro.yield\n  }\n}\n";

  auto parsed = parseKernel(source);
  ASSERT_TRUE(parsed);
  machine::MachineModel model =
      parseMachine(testMachine(/*dmaEngines=*/1, 4, 4));

  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  ASSERT_EQ(dag->events.size(), 8u);

  L1Report report = scheduleL1(*dag, model);
  EXPECT_LE(report.matrixUtilization, 1.0);
  EXPECT_GE(report.predictedCycles,
            computeL0StaticBound(*dag, model).predictedCycles);
}

TEST(L1ResourceDag, BottleneckNamesTheBusiestResource) {
  auto copies = parseKernel(kCopiesKernel);
  ASSERT_TRUE(copies);
  machine::MachineModel model =
      parseMachine(testMachine(/*dmaEngines=*/1, 2, 4));

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
  machine::MachineModel model =
      parseMachine(testMachine(/*dmaEngines=*/1, 2, 4));

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

namespace mlir::llk::perf {

//===----------------------------------------------------------------------===//
// Route-hop accounting
//===----------------------------------------------------------------------===//

namespace {

/// The same copy twice: once as written, once carrying the route a selected
/// plan chose through the hierarchy.
constexpr llvm::StringLiteral kPlainCopy = R"mlir(
module {
  micro.kernel @copy {
    %ext = tensor.empty() : tensor<8x8xf32>
    %t, %tok = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    micro.yield
  }
}
)mlir";

constexpr llvm::StringLiteral kRoutedCopy = R"mlir(
module {
  micro.kernel @copy attributes {micro.routes = [{value = 0 : i64, kind = "transfer", route = ["dram.0", "l2.0", "sram.0"]}]} {
    %ext = tensor.empty() : tensor<8x8xf32>
    %t, %tok = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    micro.yield
  }
}
)mlir";

/// Two movements that share their endpoint *kinds* (`l2 -> sram`) but target
/// different nodes of the same kind. Each copy carries the identity the binder
/// stamps on a per-hop copy -- its connection value and destination node -- so
/// the two links are told apart by node id, not by kind.
constexpr llvm::StringLiteral kTwoSameKindRoutes = R"mlir(
module {
  micro.kernel @two_routes attributes {micro.routes = [{value = 0 : i64, kind = "transfer", route = ["l2.0", "sram.0"]}, {value = 1 : i64, kind = "transfer", route = ["l2.0", "sram.1"]}]} {
    %a = tensor.empty() : tensor<8x8xf32>
    %b = tensor.empty() : tensor<8x8xf32>
    %ta, %toka = micro.async_copy %a {src_memory = #micro.memory<l2>, dst_memory = #micro.memory<sram>, micro.value = 0 : i64, micro.dst_node = "sram.0"} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    %tb, %tokb = micro.async_copy %b {src_memory = #micro.memory<l2>, dst_memory = #micro.memory<sram>, micro.value = 1 : i64, micro.dst_node = "sram.1"} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    micro.wait %toka, %tokb
    micro.yield
  }
}
)mlir";

/// A copy that carries the connection value but not a stamped destination
/// node: the route is applied as a whole, so the value alone must pick the
/// right one when two routes share the endpoint kind pair.
constexpr llvm::StringLiteral kValueTaggedCopy = R"mlir(
module {
  micro.kernel @valued attributes {micro.routes = [{value = 0 : i64, kind = "transfer", route = ["dram.0", "l2.0", "sram.0"]}, {value = 1 : i64, kind = "transfer", route = ["dram.0", "l2.0", "sram.1"]}]} {
    %ext = tensor.empty() : tensor<8x8xf32>
    %t, %tok = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>, micro.value = 1 : i64} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    micro.yield
  }
}
)mlir";

/// A routed copy inside a two-trip loop. The same op is built for every
/// unrolled iteration, so a route claimed once must stay claimed by that op.
constexpr llvm::StringLiteral kRoutedLoopCopy = R"mlir(
module {
  micro.kernel @loop_copy attributes {micro.routes = [{value = 0 : i64, kind = "transfer", route = ["dram.0", "l2.0", "sram.0"]}]} {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    micro.for %i = %c0 to %c2 step %c1 {
      %ext = tensor.empty() : tensor<8x8xf32>
      %t, %tok = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
      micro.yield
    }
    micro.yield
  }
}
)mlir";

/// A machine with two SRAM nodes of the same kind, reached by links that cost
/// 5 and 50 cycles, so a wrong-link charge is visible as equal cost.
constexpr llvm::StringLiteral kTwoSramMachine = R"yaml(
schema: llk.machine.v2
target: two-sram
clock_hz: 1000000000
worker_threads: 1
executors:
  - id: cluster.0
    kind: cluster
memories:
  - id: dram.0
    kind: dram
    visible_from: cluster.0
    capacity_bytes: 1048576
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 1
  - id: l2.0
    kind: l2
    visible_from: cluster.0
    capacity_bytes: 262144
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 1
  - id: sram.0
    kind: sram
    visible_from: cluster.0
    capacity_bytes: 65536
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 1
  - id: sram.1
    kind: sram
    visible_from: cluster.0
    capacity_bytes: 65536
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 1
transfer_engines:
  - id: dma.0
    kind: dma
    attached_to: cluster.0
    count: 1
    max_outstanding: 1
links:
  - id: dram_to_l2.0
    source: dram.0
    destination: l2.0
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 20
    transaction_bytes: 64
    transfer_engines: [dma.0]
  - id: l2_to_sram.0
    source: l2.0
    destination: sram.0
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 5
    transaction_bytes: 64
    transfer_engines: [dma.0]
  - id: l2_to_sram.1
    source: l2.0
    destination: sram.1
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 50
    transaction_bytes: 64
    transfer_engines: [dma.0]
)yaml";

} // namespace

TEST(L1ResourceDag, ARoutedMovementIsChargedPerHop) {
  auto plain = parseKernel(kPlainCopy);
  auto routed = parseKernel(kRoutedCopy);
  ASSERT_TRUE(plain);
  ASSERT_TRUE(routed);

  machine::MachineModel model = parseMachine(testMachine(1, 1, 1));
  auto plainDag = buildMicroDAG(plain->kernel, model);
  auto routedDag = buildMicroDAG(routed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(plainDag))
      << llvm::toString(plainDag.takeError());
  ASSERT_TRUE(static_cast<bool>(routedDag))
      << llvm::toString(routedDag.takeError());

  // Unrouted: one transfer from endpoint to endpoint.
  EXPECT_EQ(plainDag->events.size(), 1u);
  // Routed: the plan chose dram -> l2 -> sram, so the simulator sees both
  // links.
  ASSERT_EQ(routedDag->events.size(), 2u);
  for (const MicroEvent &event : routedDag->events)
    EXPECT_EQ(event.costKind, mapping::CostEventKind::TransferHop);

  // The second hop waits for the first: data has to arrive before it moves on.
  ASSERT_EQ(routedDag->events[1].deps.size(), 1u);
  EXPECT_EQ(routedDag->events[1].deps[0], routedDag->events[0].id);
}

TEST(L1ResourceDag, SameKindEndpointsChargeTheirOwnRoute) {
  // Two movements from L2 to SRAM, routed to two different SRAM nodes of the
  // same kind. Matching by kind alone would charge both the first route; the
  // node-id identity is what gives each movement its own link.
  auto parsed = parseKernel(kTwoSameKindRoutes);
  ASSERT_TRUE(parsed);
  machine::MachineModel model = parseMachine(kTwoSramMachine);

  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  // Two copies plus the wait that joins them.
  ASSERT_EQ(dag->events.size(), 3u);
  ASSERT_EQ(dag->events[0].kind, EventKind::AsyncCopy);
  ASSERT_EQ(dag->events[1].kind, EventKind::AsyncCopy);

  // Same bytes, same bandwidth, different links: the copy stamped `sram.0`
  // charges the 5-cycle link (5 + 256/64 = 9) and the one stamped `sram.1` the
  // 50-cycle link (54). Equal costs would mean both were charged the same link;
  // these exact numbers pin each copy to its own node's link.
  EXPECT_EQ(dag->events[0].minCycles, 9u);
  EXPECT_EQ(dag->events[1].minCycles, 54u);
}

TEST(L1ResourceDag, AValueTaggedCopyTakesItsOwnRoute) {
  // Two routes share the `dram -> sram` kind pair but end at different SRAM
  // nodes. The copy names the connection value (1) but no destination node, so
  // the value alone must select the `sram.1` route -- the order fallback would
  // take the first (`sram.0`) route.
  auto parsed = parseKernel(kValueTaggedCopy);
  ASSERT_TRUE(parsed);
  machine::MachineModel model = parseMachine(kTwoSramMachine);

  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  ASSERT_EQ(dag->events.size(), 2u);
  // dram -> l2 (20 + 4 = 24) then l2 -> sram.1 (50 + 4 = 54).
  EXPECT_EQ(dag->events[0].minCycles, 24u);
  EXPECT_EQ(dag->events[1].minCycles, 54u);
}

TEST(L1ResourceDag, AnUnrolledRoutedCopyKeepsItsRoute) {
  // One routed copy in a two-trip loop. Every unrolled iteration builds the
  // same op, so the route it was matched to must stay with that op -- not be
  // consumed by the first iteration and leave the second charged end to end.
  auto parsed = parseKernel(kRoutedLoopCopy);
  ASSERT_TRUE(parsed);
  machine::MachineModel model = parseMachine(testMachine(1, 1, 1));

  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());

  // Two iterations times the two hops of dram -> l2 -> sram.
  ASSERT_EQ(dag->events.size(), 4u);
  for (const MicroEvent &event : dag->events)
    EXPECT_EQ(event.costKind, mapping::CostEventKind::TransferHop);
}

TEST(L1ResourceDag, EveryEventCarriesItsSharedCostCategory) {
  auto parsed = parseKernel(kPlainCopy);
  ASSERT_TRUE(parsed);
  machine::MachineModel model = parseMachine(testMachine(1, 1, 1));
  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dag));
  ASSERT_FALSE(dag->events.empty());
  EXPECT_EQ(dag->events[0].costKind, mapping::CostEventKind::TransferHop);
}

namespace {
/// Two workers, each with its own vector engine, so a mapped op's executor
/// decides which engine the cost model charges.
constexpr llvm::StringLiteral kTwoEngineMachine = R"yaml(
schema: llk.machine.v2
target: two-engine
clock_hz: 1000000000
worker_threads: 2
executors:
  - id: cluster.0
    kind: cluster
  - id: worker.0
    kind: worker
    parent: cluster.0
  - id: cluster.1
    kind: cluster
  - id: worker.1
    kind: worker
    parent: cluster.1
memories:
  - id: sram.0
    kind: sram
    visible_from: cluster.0
    capacity_bytes: 1048576
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 4
compute:
  - id: vpu.a
    kind: vector_engine
    attached_to: worker.0
    element_types: [f32]
    shapes: [[8]]
    lanes: {f32: 8}
    issue_cycles: 1
    latency_cycles: 1
    supported_layouts: [row_major]
  - id: vpu.b
    kind: vector_engine
    attached_to: worker.1
    element_types: [f32]
    shapes: [[8]]
    lanes: {f32: 8}
    issue_cycles: 1
    latency_cycles: 1
    supported_layouts: [row_major]
)yaml";

/// One vector add whose `micro.mapping` names the executor the plan selected.
std::string mappedVectorKernel(llvm::StringRef executor) {
  return std::string(
             "module {\n  micro.kernel @mapped {\n"
             "    %t = micro.tile_alloc : !micro.tile<8xf32, memory = "
             "#micro.memory<sram>>\n"
             "    %r = micro.vector \"add\" %t, %t {micro.mapping = {executor "
             "= \"") +
         executor.str() +
         "\"}} : !micro.tile<8xf32, memory = #micro.memory<sram>>, "
         "!micro.tile<8xf32, memory = #micro.memory<sram>> -> "
         "!micro.tile<8xf32, memory = #micro.memory<sram>>\n"
         "    micro.yield\n  }\n}\n";
}
} // namespace

// §17.2: the mapped executor is a modelled decision, so a vector op is charged
// on the engine attached to the executor the plan selected -- not on the
// machine's declaration-order first engine. Two kernels differing only in the
// selected executor therefore charge two different engines.
TEST(L1ResourceDag, AMappedOpChargesTheEngineItsExecutorOwns) {
  machine::MachineModel model = parseMachine(kTwoEngineMachine);

  auto first = parseKernel(mappedVectorKernel("worker.0"));
  ASSERT_TRUE(first);
  auto dagA = buildMicroDAG(first->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dagA)) << llvm::toString(dagA.takeError());
  ASSERT_EQ(dagA->events.size(), 1u);
  EXPECT_EQ(dagA->events[0].resourceName, "vpu.a");

  auto second = parseKernel(mappedVectorKernel("worker.1"));
  ASSERT_TRUE(second);
  auto dagB = buildMicroDAG(second->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dagB)) << llvm::toString(dagB.takeError());
  ASSERT_EQ(dagB->events.size(), 1u);
  EXPECT_EQ(dagB->events[0].resourceName, "vpu.b");
}

} // namespace mlir::llk::perf
