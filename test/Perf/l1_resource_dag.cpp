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
#include "LLK/Mapping/CostEvent.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/EventSchedule.h"
#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/StorageLiveness.h"
#include "LLK/Mapping/StoragePlan.h"
#include "LLK/Perf/MicroCostModel.h"
#include "LLK/Perf/MicroDAG.h"
#include "LLK/Perf/MicroPerfReport.h"
#include "LLK/Perf/SelectedKernelAnalysis.h"

#include "resource_regression_fixture.h"

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
/// Built in the v2 shape: `workers` are executors, and `matrixEngines` /
/// `dmaEngines` are the *multiplicity* of the one capability/transfer node that
/// carries them -- a node's own `concurrency` (compute) or `count` (transfer)
/// is its concurrency, which is what the scheduler's slot arithmetic reads.
/// Declaring the engines as one node with a count is what the shipped machine
/// profiles do (`machines/*-v2.yaml`), and it keeps a link that names `dma.0`
/// able to use every engine the machine offers.
std::string testMachine(unsigned dmaEngines, unsigned workers,
                        unsigned matrixEngines) {
  std::string yaml =
      "schema: llk.machine.v2\n"
      "target: test-machine\n"
      "clock_hz: 1000000000\n"
      "worker_threads: " +
      std::to_string(workers) +
      "\nsync:\n  barrier_cycles: 1\n  wait_cycles: 0\n"
      "executors:\n"
      "  - id: cluster.0\n    kind: cluster\n    refines: [group]\n";
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
  yaml += "  - id: mxu\n    kind: matrix_engine\n    refines: [matrix]\n"
          "    attached_to: worker.0\n"
          "    element_types: [f32, bf16]\n    accumulator_dtypes: [f32]\n"
          "    shapes: [[1, 1, 1]]\n    issue_cycles: 1\n"
          "    latency_cycles: 1\n    throughput_per_cycle: 1\n"
          "    concurrency: " +
          std::to_string(matrixEngines) +
          "\n    supported_layouts: [row_major]\n";
  yaml += "  - id: vpu\n    kind: vector_engine\n    refines: [vector]\n"
          "    attached_to: worker.0\n"
          "    element_types: [f32, bf16]\n    shapes: [[8]]\n"
          "    lanes: {f32: 8, bf16: 16}\n    issue_cycles: 1\n"
          "    latency_cycles: 1\n    supported_layouts: [row_major]\n";

  yaml += "transfer_engines:\n";
  yaml += "  - id: dma.0\n    kind: dma\n    refines: [transfer]\n"
          "    attached_to: cluster.0\n"
          "    count: " +
          std::to_string(dmaEngines) + "\n    max_outstanding: 1\n";

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

/// The bound form of `kCopiesKernel`: the same two independent `dram.0 ->
/// sram.0` copies, each carrying the route its plan selected. A bound copy
/// names the engine of the link it travels; an unrouted copy is unmapped
/// analysis and only names the abstract `dma` pool.
constexpr const char *kRoutedCopiesKernel = R"MLIR(
module {
  micro.kernel @routed_copies attributes {micro.routes = [{value = 0 : i64, kind = "transfer", route = ["dram.0", "sram.0"]}, {value = 1 : i64, kind = "transfer", route = ["dram.0", "sram.0"]}]} {
    %a_ext = tensor.empty() : tensor<8x8xf32>
    %b_ext = tensor.empty() : tensor<8x8xf32>
    %a_tile, %a_tok = micro.async_copy %a_ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>, micro.value = 0 : i64, micro.dst_node = "sram.0"} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    %b_tile, %b_tok = micro.async_copy %b_ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>, micro.value = 1 : i64, micro.dst_node = "sram.0"} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
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
  auto parsed = parseKernel(kRoutedCopiesKernel);
  ASSERT_TRUE(parsed);

  machine::MachineModel serialMachine =
      parseMachine(testMachine(/*dmaEngines=*/1, 2, 4));
  machine::MachineModel parallelMachine =
      parseMachine(testMachine(/*dmaEngines=*/2, 2, 4));

  auto dag = buildMicroDAG(parsed->kernel, serialMachine);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  ASSERT_EQ(dag->events.size(), 3u);
  // Both copies name the engine their link selects; task R2 sizes that engine's
  // pool by the node's own `count`, which is what `dmaEngines` sets here.
  ASSERT_EQ(dag->events[0].resourceName, "dma.0");
  ASSERT_EQ(dag->events[1].resourceName, "dma.0");

  L1Report serial = scheduleL1(*dag, serialMachine);
  L1Report parallel = scheduleL1(*dag, parallelMachine);

  // With one slot on the named engine the second copy waits for the first; with
  // two they start together, because neither depends on the other.
  EXPECT_EQ(serial.schedule[1].start, serial.schedule[0].finish);
  EXPECT_EQ(parallel.schedule[0].start, 0u);
  EXPECT_EQ(parallel.schedule[1].start, 0u);
  EXPECT_LT(parallel.predictedCycles, serial.predictedCycles);
}

// Unmapped analysis: a hand-written kernel that declares no route names the
// abstract `dma` pool, which is deliberately partial -- a single slot. It must
// not silently claim every engine the machine declares, so a second engine does
// not widen it. A strict caller rejects such a stream through
// `mapping::validateEventResources` instead of trusting the partial cost.
TEST(L1ResourceDag, UnmappedCopiesShareOnePartialTransferPool) {
  auto parsed = parseKernel(kCopiesKernel);
  ASSERT_TRUE(parsed);
  machine::MachineModel model =
      parseMachine(testMachine(/*dmaEngines=*/2, 2, 4));

  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  ASSERT_EQ(dag->events.size(), 3u);
  EXPECT_EQ(dag->events[0].resourceName, "dma");
  EXPECT_EQ(dag->events[1].resourceName, "dma");

  L1Report report = scheduleL1(*dag, model);
  EXPECT_EQ(report.schedule[1].start, report.schedule[0].finish);

  mapping::PlanEventDAG normalized;
  for (const MicroEvent &event : dag->events)
    normalized.events.push_back(normalizedPlanEvent(event));
  EXPECT_TRUE(
      static_cast<bool>(mapping::validateEventResources(normalized, model)));
}

namespace {
/// One worker, one `dram.0 -> sram.0` link whose transfers run on `dma.a`, and
/// a second engine `dma.b` that no link names -- the "unused sibling" the
/// pooling repair must ignore.
constexpr llvm::StringLiteral kOneNamedLinkMachine = R"yaml(
schema: llk.machine.v2
target: named-dma
clock_hz: 1000000000
worker_threads: 1
executors:
  - id: cluster.0
    kind: cluster
    refines: [group]
memories:
  - id: dram.0
    kind: dram
    visible_from: cluster.0
    capacity_bytes: 1048576
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 0
  - id: sram.0
    kind: sram
    visible_from: cluster.0
    capacity_bytes: 65536
    alignment_bytes: 64
    supported_layouts: [row_major]
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 0
transfer_engines:
  - id: dma.a
    kind: dma
    refines: [transfer]
    attached_to: cluster.0
    count: 1
    max_outstanding: 1
  - id: dma.b
    kind: dma
    refines: [transfer]
    attached_to: cluster.0
    count: 1
    max_outstanding: 1
links:
  - id: dram_to_sram.0
    source: dram.0
    destination: sram.0
    bandwidth_bytes_per_cycle: 64
    latency_cycles: 10
    transaction_bytes: 64
    transfer_engines: [dma.a]
)yaml";
} // namespace

// Issue #129, task R2: the materialized-kernel path resolves a transfer to the
// engine its *link* names and pools it by that node's own count, exactly as the
// plan path does. Two independent copies name `dma.a`; the machine also
// declares an unused `dma.b`, whose presence must not widen `dma.a`'s pool.
// Before the repair the pool took the whole machine's engine count, so the
// unused node made the two copies overlap when the engine they name has a
// single slot.
TEST(L1ResourceDag, NamedTransferEngineCountIsItsOwnConcurrency) {
  auto parsed = parseKernel(kRoutedCopiesKernel);
  ASSERT_TRUE(parsed);
  machine::MachineModel model = parseMachine(kOneNamedLinkMachine);
  ASSERT_EQ(model.transferEngineCount(), 2u);

  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  ASSERT_EQ(dag->events.size(), 3u);
  ASSERT_EQ(dag->events[0].resourceName, "dma.a");
  ASSERT_EQ(dag->events[1].resourceName, "dma.a");
  const uint64_t copyCycles = dag->events[0].minCycles;
  ASSERT_GT(copyCycles, 0u);

  L1Report alone = scheduleL1(*dag, model);
  EXPECT_EQ(alone.schedule[0].start, 0u);
  EXPECT_EQ(alone.schedule[1].start, copyCycles);
  EXPECT_EQ(alone.schedule[1].finish, 2 * copyCycles);

  // Removing the engine no link names cannot change the schedule: `dma.a` was
  // never entitled to `dma.b`'s slot.
  model.transferEngines.pop_back();
  L1Report withoutUnused = scheduleL1(*dag, model);
  EXPECT_EQ(withoutUnused.schedule[1].start, alone.schedule[1].start);
  EXPECT_EQ(withoutUnused.predictedCycles, alone.predictedCycles);

  // Doubling the *named* node's own count is what buys overlap.
  model.transferEngines.front().count = 2;
  L1Report doubled = scheduleL1(*dag, model);
  EXPECT_EQ(doubled.schedule[0].start, 0u);
  EXPECT_EQ(doubled.schedule[1].start, 0u);
  EXPECT_EQ(doubled.schedule[1].finish, copyCycles);
  EXPECT_LT(doubled.predictedCycles, alone.predictedCycles);
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

// A transform-only kernel: the conversion is real work, not the zero-cost
// "everything else" path an unhandled op used to fall through.
TEST(L1ResourceDag, TransformIsAChargedEvent) {
  auto parsed = parseKernel(R"mlir(
module {
  micro.kernel @transform_only {
    %a = tensor.empty() : tensor<8x8xf32>
    %t = micro.transform %a {src_map = affine_map<(d0,d1)->(d0,d1)>, dst_map = affine_map<(d0,d1)->(d1,d0)>} : tensor<8x8xf32> -> tensor<8x8xf32>
    micro.yield
  }
}
)mlir");
  ASSERT_TRUE(parsed);
  auto model = parseMachine(testMachine(1, 1, 1));
  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(bool(dag)) << llvm::toString(dag.takeError());
  ASSERT_EQ(dag->events.size(), 1u);
  EXPECT_EQ(dag->events.front().kind, EventKind::Transform);
  EXPECT_EQ(dag->events.front().costKind, mapping::CostEventKind::Transform);
  EXPECT_GT(dag->events.front().minCycles, 0u);
}

TEST(L1ResourceDag, TransformHonorsItsSelectedResource) {
  auto parsed = parseKernel(R"mlir(
module {
  micro.kernel @selected_transform {
    %a = tensor.empty() : tensor<8x8xf32>
    %t = micro.transform %a {src_map = affine_map<(d0,d1)->(d0,d1)>, dst_map = affine_map<(d0,d1)->(d1,d0)>, micro.compute_resource = "vpu.slow", micro.memory_node = "sram.0"} : tensor<8x8xf32> -> tensor<8x8xf32>
    micro.yield
  }
}
)mlir");
  ASSERT_TRUE(parsed);
  auto model = parseMachine(testMachine(1, 1, 1));
  auto slow = *model.findCompute("vpu");
  slow.id = "vpu.slow";
  slow.lanes["f32"] = 2;
  model.computes.push_back(slow);
  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(bool(dag)) << llvm::toString(dag.takeError());
  ASSERT_EQ(dag->events.size(), 1u);
  EXPECT_EQ(dag->events.front().resourceName, "vpu.slow");
  EXPECT_EQ(dag->events.front().minCycles, 32u);

  parsed->kernel->walk([&](mlir::Operation *transform) {
    if (transform->getName().getStringRef() != "micro.transform")
      return;
    transform->setAttr("micro.compute_resource",
                       mlir::StringAttr::get(parsed->context.get(), "missing"));
  });
  auto invalid = buildMicroDAG(parsed->kernel, model);
  EXPECT_FALSE(bool(invalid));
  if (!invalid)
    llvm::consumeError(invalid.takeError());
}

/// A copy into SRAM, a layout conversion of what it landed, and a consumer that
/// reads the converted value. The transform is charged, materializes its own
/// output buffer, and both edges around it are data dependencies.
constexpr const char *kTransformChainKernel = R"mlir(
module {
  micro.kernel @transform_chain {
    %ext = tensor.empty() : tensor<8x8xf32>
    %t, %tok = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    %tv = micro.tile_view %t {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %tr = micro.transform %tv {src_map = affine_map<(d0,d1)->(d0,d1)>, dst_map = affine_map<(d0,d1)->(d1,d0)>} : !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %tr, %tr : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)mlir";

TEST(L1ResourceDag, TransformChainKeepsItsDependency) {
  auto parsed = parseKernel(kTransformChainKernel);
  ASSERT_TRUE(parsed);
  machine::MachineModel model = parseMachine(testMachine(1, 1, 1));
  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  // The tile view is zero-cost metadata, so only copy, transform, vector
  // remain.
  ASSERT_EQ(dag->events.size(), 3u);

  const MicroEvent &copy = dag->events[0];
  const MicroEvent &transform = dag->events[1];
  const MicroEvent &vector = dag->events[2];

  EXPECT_EQ(llvm::count_if(dag->events,
                           [](const MicroEvent &event) {
                             return event.kind == EventKind::Transform;
                           }),
            1u);
  EXPECT_EQ(transform.kind, EventKind::Transform);
  EXPECT_EQ(transform.costKind, mapping::CostEventKind::Transform);
  // A nonidentity conversion costs cycles.
  EXPECT_GT(transform.minCycles, 0u);

  // The conversion depends on what produced its input...
  ASSERT_EQ(transform.deps.size(), 1u);
  EXPECT_EQ(transform.deps[0], copy.id);
  // ...and the consumer depends on the conversion, not on the copy.
  EXPECT_EQ(vector.kind, EventKind::Vector);
  ASSERT_EQ(vector.deps.size(), 1u);
  EXPECT_EQ(vector.deps[0], transform.id);

  // The conversion materializes a fresh output buffer in the memory its input
  // landed in: the copy's 256 bytes plus the transform's own 256.
  EXPECT_EQ(transform.tileMemory, "sram");
  auto live = dag->liveTileBytesByMemory.find("sram");
  ASSERT_NE(live, dag->liveTileBytesByMemory.end());
  EXPECT_EQ(live->second, 512u);
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
    refines: [group]
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
    refines: [transfer]
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
    refines: [group]
  - id: worker.0
    kind: worker
    parent: cluster.0
  - id: cluster.1
    kind: cluster
    refines: [group]
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
    refines: [vector]
    attached_to: worker.0
    element_types: [f32]
    shapes: [[8]]
    lanes: {f32: 8}
    issue_cycles: 1
    latency_cycles: 1
    supported_layouts: [row_major]
  - id: vpu.b
    kind: vector_engine
    refines: [vector]
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

//===----------------------------------------------------------------------===//
// Gather and barrier (task B6)
//===----------------------------------------------------------------------===//

namespace {

/// Two producer tiles combined by an explicit `sum` gather, followed by a
/// collective barrier. Neither op may fall through the zero-cost path: the
/// gather is compute and the barrier is synchronization.
constexpr const char *kGatherBarrierKernel = R"MLIR(
module {
  micro.kernel @gather_barrier {
    %a = micro.tile_alloc : !micro.tile<8xf32, memory = #micro.memory<sram>>
    %b = micro.tile_alloc : !micro.tile<8xf32, memory = #micro.memory<sram>>
    %g = micro.gather %a, %b kind = "sum" : !micro.tile<8xf32, memory = #micro.memory<sram>>, !micro.tile<8xf32, memory = #micro.memory<sram>> -> !micro.tile<8xf32, memory = #micro.memory<sram>>
    micro.barrier scope = "executor_group"
    micro.yield
  }
}
)MLIR";

} // namespace

TEST(L1ResourceDag, GatherIsComputeAndBarrierIsSynchronization) {
  auto parsed = parseKernel(kGatherBarrierKernel);
  ASSERT_TRUE(parsed);
  machine::MachineModel model = parseMachine(testMachine(1, 1, 1));

  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  ASSERT_EQ(dag->events.size(), 2u);

  // The gather is charged as compute on the vector engine, over its elements.
  const MicroEvent &gather = dag->events[0];
  EXPECT_EQ(gather.kind, EventKind::Reduce);
  EXPECT_EQ(gather.costKind, mapping::CostEventKind::Compute);
  EXPECT_EQ(gather.resource, ResourceKind::VectorEngine);
  EXPECT_EQ(gather.workItems, 8u);

  // The barrier is its own synchronization event, charged the machine's
  // barrier cost -- not free program order and not a DMA transfer.
  const MicroEvent &barrier = dag->events[1];
  EXPECT_EQ(barrier.kind, EventKind::Barrier);
  EXPECT_EQ(barrier.costKind, mapping::CostEventKind::Synchronization);
  EXPECT_EQ(barrier.resource, ResourceKind::Sync);
  EXPECT_EQ(barrier.minCycles, model.sync.barrierCycles);
}

TEST(L1ResourceDag, BarrierDependsOnTheTokensItCovers) {
  auto parsed = parseKernel(R"MLIR(
module {
  micro.kernel @barrier_on_copy {
    %ext = tensor.empty() : tensor<8x8xf32>
    %t, %tok = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    micro.barrier %tok scope = "executor_group"
    micro.yield
  }
}
)MLIR");
  ASSERT_TRUE(parsed);
  machine::MachineModel model = parseMachine(testMachine(1, 1, 1));

  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  ASSERT_EQ(dag->events.size(), 2u);
  EXPECT_EQ(dag->events[0].kind, EventKind::AsyncCopy);
  ASSERT_EQ(dag->events[1].kind, EventKind::Barrier);
  // The barrier waits on the token the copy produced.
  ASSERT_EQ(dag->events[1].deps.size(), 1u);
  EXPECT_EQ(dag->events[1].deps[0], dag->events[0].id);
}

//===----------------------------------------------------------------------===//
// Task B8: selected-plan and materialized event parity
//===----------------------------------------------------------------------===//

/// A two-hop movement (dram.0 -> l2.0 -> sram.0), one awaited copy per hop --
/// exactly the shape the binder materializes for a routed `Transfer`: the plan
/// routes and the per-hop stamps travel with the copies.
constexpr llvm::StringLiteral kTwoHopKernel = R"mlir(
module {
  micro.kernel @two_hop attributes {micro.routes = [{value = 7 : i64, kind = "transfer", route = ["dram.0", "l2.0", "sram.0"], id = 500 : i64}]} {
    %ext = tensor.empty() : tensor<8x8xf32>
    %t1, %tok1 = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<l2>, micro.value = 7 : i64, micro.dst_node = "l2.0", micro.connection = 500 : i64, micro.hop = 1 : i64} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    micro.wait %tok1
    %t2, %tok2 = micro.async_copy %t1 {src_memory = #micro.memory<l2>, dst_memory = #micro.memory<sram>, micro.value = 7 : i64, micro.dst_node = "sram.0", micro.connection = 500 : i64, micro.hop = 2 : i64} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    micro.wait %tok2
    micro.yield
  }
}
)mlir";

// A selected plan and its materialized kernel account for the same normalized
// events: kind, resource, work, bytes, latency and dependency edges agree. The
// plan path builds the stream from the plan's step DAG; the perf path builds it
// from the emitted IR and normalizes it through the same construction.
TEST(L1ResourceDag, SelectedPlanEventsMatchItsMaterializedKernel) {
  auto parsed = parseKernel(kTwoHopKernel);
  ASSERT_TRUE(parsed);
  machine::MachineModel model = parseMachine(testMachine(1, 1, 1));

  auto kernelDag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(kernelDag))
      << llvm::toString(kernelDag.takeError());

  mapping::CoveringPlan plan;
  mapping::PlanConnection connection;
  connection.id = 500;
  connection.value = 7;
  connection.kind = mapping::ConnectionKind::Transfer;
  connection.route = {"dram.0", "l2.0", "sram.0"};
  connection.engines = {"dma.0"};
  connection.cost.localBytes = 256;
  connection.workItems = 64;
  plan.connectionPlans.push_back(connection);
  plan.steps = {
      mapping::PlanStep{0, mapping::PlanStepKind::Movement, 0, 500},
      mapping::PlanStep{1, mapping::PlanStepKind::Synchronization, 0, 500}};

  llvm::Expected<mapping::PlanEventDAG> planDag =
      mapping::buildPlanEvents(plan, model);
  ASSERT_TRUE(static_cast<bool>(planDag))
      << llvm::toString(planDag.takeError());
  ASSERT_EQ(planDag->events.size(), kernelDag->events.size());

  for (size_t i = 0; i < planDag->events.size(); ++i) {
    mapping::PlanCostEvent normalized =
        normalizedPlanEvent(kernelDag->events[i]);
    EXPECT_EQ(planDag->events[i].event.kind, normalized.event.kind) << i;
    EXPECT_EQ(planDag->events[i].event.resource, normalized.event.resource)
        << i;
    EXPECT_EQ(planDag->events[i].workItems, normalized.workItems) << i;
    EXPECT_EQ(planDag->events[i].bytes, normalized.bytes) << i;
    EXPECT_DOUBLE_EQ(planDag->events[i].event.cost.latencyCycles,
                     normalized.event.cost.latencyCycles)
        << i;
    EXPECT_EQ(planDag->events[i].deps, normalized.deps) << i;
  }
}

// The selected plan and its bound kernel are one analysis (issue #129, task
// R6). The plan's final events are the derived snapshot the shared
// selected-kernel analysis produced from the materialized kernel, so *every*
// normalized field agrees -- kind, resource, work, bytes, dependencies and the
// owner-occupancy pool -- and the plan schedules to the same cost. Before the
// repair the plan charged its own rule-local estimate (five cycles) while the
// kernel charged the machine's elementwise formula (eight), and a plan carried
// no owner pool at all.
TEST(L1ResourceDag, SelectedPlanEventsMatchItsBoundKernel) {
  auto parsed = parseKernel(R"mlir(
module {
  micro.kernel @selected_add {
    %t = micro.tile_alloc : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {executor = "worker.0"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)mlir");
  ASSERT_TRUE(parsed);
  machine::MachineModel model = parseMachine(testMachine(1, 1, 1));

  llvm::Expected<SelectedKernelAnalysis> analysis =
      analyzeSelectedKernel(parsed->kernel, model, /*requireComplete=*/true);
  ASSERT_TRUE(static_cast<bool>(analysis))
      << llvm::toString(analysis.takeError());
  ASSERT_EQ(analysis->events.events.size(), 1u);
  // The mapped op names the executor the plan selected as its owner pool, and
  // the whole stream is complete.
  EXPECT_EQ(analysis->events.events[0].owner, "worker.0");
  EXPECT_TRUE(analysis->complete) << analysis->incompleteReasons.front();

  // The plan the analysis describes: one placement of the add on the executor
  // the kernel recorded, with the search's *rule-local* estimate (5) -- which
  // the derived snapshot must replace with the kernel's own work (8).
  mapping::CoveringPlan plan;
  mapping::PlanPlacement placement;
  placement.node = 0;
  placement.instance = 100;
  placement.executor = "worker.0";
  placement.computeRequirements = {"vector_engine"};
  placement.computeBindings["vector_engine"] = "vpu";
  placement.cost.latencyCycles = 5.0;
  placement.workItems = 64;
  plan.placements.push_back(placement);
  plan.steps = {mapping::PlanStep{0, mapping::PlanStepKind::Compute, 0, 0}};

  ASSERT_FALSE(
      bool(mapping::attachPlanAnalysisEvents(plan, analysis->events, model)));

  llvm::Expected<mapping::PlanEventDAG> events =
      mapping::buildPlanEvents(plan, model);
  ASSERT_TRUE(static_cast<bool>(events)) << llvm::toString(events.takeError());
  ASSERT_EQ(events->events.size(), analysis->events.events.size());
  EXPECT_EQ(events->source, mapping::PlanEventSource::Snapshot);

  // The plan's stream is the kernel's, field for field. These are asserted
  // against *literal* values rather than against `analysis->events`, which
  // `buildPlanEvents` returns verbatim once the snapshot is attached -- a
  // comparison to itself could never fail (issue #129, task R8 review).
  ASSERT_EQ(events->events.size(), 1u);
  const mapping::PlanCostEvent &planned = events->events.front();
  EXPECT_EQ(planned.event.kind, mapping::CostEventKind::Compute);
  EXPECT_EQ(planned.event.resource, "vpu");
  EXPECT_EQ(planned.owner, "worker.0");
  EXPECT_EQ(planned.workItems, 64u);
  // The machine's elementwise formula, not the placement's rule-local estimate
  // of 5: the snapshot replaced the search's number.
  EXPECT_DOUBLE_EQ(planned.event.cost.latencyCycles, 8.0);
  EXPECT_EQ(planned.bytes, 0u);

  // The scheduled cost is the same schedule of the same stream.
  llvm::Expected<mapping::Cost> scheduled =
      mapping::schedulePlanEvents(*events, model);
  ASSERT_TRUE(static_cast<bool>(scheduled))
      << llvm::toString(scheduled.takeError());
  EXPECT_EQ(scheduled->latencyCycles,
            double(analysis->schedule.predictedCycles));
  // The machine's elementwise formula: 64 elements over 8 f32 lanes.
  EXPECT_DOUBLE_EQ(events->events[0].event.cost.latencyCycles, 8.0);
}

// Normalized plan events are scheduled by the *same* resource scheduler the
// performance evaluator uses: resource multiplicity overlaps independent work.
TEST(L1ResourceDag, PlanEventsShareThePerformanceScheduler) {
  mapping::PlanEventDAG dag;
  dag.events.push_back(mapping::makePlanCostEvent(
      mapping::CostEventKind::TransferHop, "dma.0", 10.0, 64, 64));
  dag.events.push_back(mapping::makePlanCostEvent(
      mapping::CostEventKind::TransferHop, "dma.0", 10.0, 64, 64));

  machine::MachineModel serial = parseMachine(testMachine(1, 1, 1));
  machine::MachineModel parallel = parseMachine(testMachine(2, 1, 1));

  llvm::Expected<mapping::Cost> serialCost =
      mapping::schedulePlanEvents(dag, serial);
  llvm::Expected<mapping::Cost> parallelCost =
      mapping::schedulePlanEvents(dag, parallel);
  ASSERT_TRUE(static_cast<bool>(serialCost))
      << llvm::toString(serialCost.takeError());
  ASSERT_TRUE(static_cast<bool>(parallelCost))
      << llvm::toString(parallelCost.takeError());

  // One engine serializes the two independent hops; two run them at once.
  EXPECT_DOUBLE_EQ(serialCost->latencyCycles, 20.0);
  EXPECT_DOUBLE_EQ(parallelCost->latencyCycles, 10.0);
  EXPECT_EQ(serialCost->localBytes, 128u);
}

// The binder stamps the plan-selected executor on an emitted `micro.transform`
// as `micro.engine`; the perf model charges that engine's vector unit rather
// than the machine's declaration-order default.
TEST(L1ResourceDag, TransformChargesItsSelectedEngine) {
  machine::MachineModel model = parseMachine(kTwoEngineMachine);

  auto selected = parseKernel(R"mlir(
module {
  micro.kernel @selected {
    %a = tensor.empty() : tensor<8x8xf32>
    %t = micro.transform %a {src_map = affine_map<(d0,d1)->(d0,d1)>, dst_map = affine_map<(d0,d1)->(d1,d0)>, micro.engine = "worker.1"} : tensor<8x8xf32> -> tensor<8x8xf32>
    micro.yield
  }
}
)mlir");
  ASSERT_TRUE(selected);
  auto selectedDag = buildMicroDAG(selected->kernel, model);
  ASSERT_TRUE(static_cast<bool>(selectedDag))
      << llvm::toString(selectedDag.takeError());
  ASSERT_EQ(selectedDag->events.size(), 1u);
  EXPECT_EQ(selectedDag->events[0].kind, EventKind::Transform);
  EXPECT_EQ(selectedDag->events[0].resourceName, "vpu.b");

  // Without the stamp the machine's first engine of the class is the default.
  auto unstamped = parseKernel(R"mlir(
module {
  micro.kernel @unstamped {
    %a = tensor.empty() : tensor<8x8xf32>
    %t = micro.transform %a {src_map = affine_map<(d0,d1)->(d0,d1)>, dst_map = affine_map<(d0,d1)->(d1,d0)>} : tensor<8x8xf32> -> tensor<8x8xf32>
    micro.yield
  }
}
)mlir");
  ASSERT_TRUE(unstamped);
  auto unstampedDag = buildMicroDAG(unstamped->kernel, model);
  ASSERT_TRUE(static_cast<bool>(unstampedDag))
      << llvm::toString(unstampedDag.takeError());
  ASSERT_EQ(unstampedDag->events.size(), 1u);
  EXPECT_EQ(unstampedDag->events[0].resourceName, "vpu.a");
}

//===----------------------------------------------------------------------===//
// Issue #129, task R6: the shared selected-kernel analysis
//===----------------------------------------------------------------------===//

namespace {
/// One worker with two matrix engines of its own and a `concurrency` the caller
/// picks, so an event's owner pool and its resource pool vary independently.
/// Two engines attached to one single-count executor is the case a scheduler
/// that dropped or aggregated owner constraints would let overlap.
std::string ownerPoolMachineYaml(unsigned workerConcurrency) {
  std::string yaml =
      "schema: llk.machine.v2\n"
      "target: owner-pool\n"
      "clock_hz: 1000000000\n"
      "worker_threads: 1\n"
      "sync:\n  barrier_cycles: 1\n  wait_cycles: 0\n"
      "executors:\n"
      "  - id: cluster.0\n    kind: cluster\n    refines: [group]\n"
      "  - id: worker.0\n    kind: worker\n    parent: cluster.0\n"
      "    concurrency: " +
      std::to_string(workerConcurrency) + "\nmemories:\n";
  for (const auto &level :
       {std::pair<const char *, const char *>{"sram", "sram"},
        {"acc", "acc"}}) {
    yaml += "  - id: " + std::string(level.first) +
            ".0\n    kind: " + level.second +
            "\n    visible_from: cluster.0\n    capacity_bytes: 1048576\n"
            "    alignment_bytes: 64\n    supported_layouts: [row_major]\n"
            "    bandwidth_bytes_per_cycle: 64\n    latency_cycles: 1\n";
  }
  yaml += "compute:\n";
  yaml += "  - id: mxu.a\n    kind: matrix_engine\n    refines: [matrix]\n"
          "    attached_to: worker.0\n"
          "    element_types: [bf16]\n    accumulator_dtypes: [f32]\n"
          "    shapes: [[1, 1, 1]]\n    issue_cycles: 1\n"
          "    latency_cycles: 1\n    throughput_per_cycle: 1\n"
          "    concurrency: 1\n    supported_layouts: [row_major]\n"
          "  - id: mxu.b\n    kind: matrix_engine\n    refines: [matrix]\n"
          "    attached_to: worker.0\n"
          "    element_types: [bf16]\n    accumulator_dtypes: [f32]\n"
          "    shapes: [[1, 1, 1]]\n    issue_cycles: 1\n"
          "    latency_cycles: 1\n    throughput_per_cycle: 1\n"
          "    concurrency: 1\n    supported_layouts: [row_major]\n";
  return yaml;
}

/// Two independent MMAs on one executor, each naming a different engine, so the
/// only constraint that can serialize them is the owner-occupancy pool.
constexpr const char *kTwoEngineMappedKernel = R"mlir(
module {
  micro.kernel @two_engine_mapped {
    %a = micro.tile_alloc : !micro.tile<1x1xbf16, memory = #micro.memory<sram>>
    %b = micro.tile_alloc : !micro.tile<1x1xbf16, memory = #micro.memory<sram>>
    %c0 = micro.tile_alloc : !micro.tile<1x1xf32, memory = #micro.memory<acc>>
    %c1 = micro.tile_alloc : !micro.tile<1x1xf32, memory = #micro.memory<acc>>
    %r0 = micro.mma %a, %b, %c0 {shape = array<i64: 1, 1, 1>, input = #micro.dtype<bf16>, accumulator = #micro.dtype<f32>, engine = "mxu.a", micro.mapping = {executor = "worker.0"}} : !micro.tile<1x1xbf16, memory = #micro.memory<sram>>, !micro.tile<1x1xbf16, memory = #micro.memory<sram>>, !micro.tile<1x1xf32, memory = #micro.memory<acc>> -> !micro.tile<1x1xf32, memory = #micro.memory<acc>>
    %r1 = micro.mma %a, %b, %c1 {shape = array<i64: 1, 1, 1>, input = #micro.dtype<bf16>, accumulator = #micro.dtype<f32>, engine = "mxu.b", micro.mapping = {executor = "worker.0"}} : !micro.tile<1x1xbf16, memory = #micro.memory<sram>>, !micro.tile<1x1xbf16, memory = #micro.memory<sram>>, !micro.tile<1x1xf32, memory = #micro.memory<acc>> -> !micro.tile<1x1xf32, memory = #micro.memory<acc>>
    micro.yield
  }
}
)mlir";
} // namespace

// Two engines of one executor are two resource pools, but one owner pool: a
// single-count executor runs them one at a time however many engines it owns.
// A scheduler that ignored the owner, or widened it to the machine's worker
// count, would run both at once -- which is not this machine.
TEST(L1ResourceDag, OwnerPoolIsTheSelectedExecutorNotItsEngineCount) {
  auto parsed = parseKernel(kTwoEngineMappedKernel);
  ASSERT_TRUE(parsed);

  machine::MachineModel serial =
      parseMachine(ownerPoolMachineYaml(/*workerConcurrency=*/1));
  llvm::Expected<SelectedKernelAnalysis> narrow =
      analyzeSelectedKernel(parsed->kernel, serial, /*requireComplete=*/true);
  ASSERT_TRUE(static_cast<bool>(narrow)) << llvm::toString(narrow.takeError());
  ASSERT_EQ(narrow->events.events.size(), 2u);
  // Both events name the selected executor as their owner-occupancy pool.
  EXPECT_EQ(narrow->events.events[0].owner, "worker.0");
  EXPECT_EQ(narrow->events.events[1].owner, "worker.0");
  EXPECT_EQ(narrow->events.events[0].event.resource, "mxu.a");
  EXPECT_EQ(narrow->events.events[1].event.resource, "mxu.b");

  const uint64_t perEngine = narrow->events.events[0].event.cost.latencyCycles;
  EXPECT_EQ(perEngine, 2u); // 2 flops over one flop per cycle
  EXPECT_EQ(narrow->schedule.predictedCycles, 2 * perEngine);

  machine::MachineModel wide =
      parseMachine(ownerPoolMachineYaml(/*workerConcurrency=*/2));
  llvm::Expected<SelectedKernelAnalysis> broad =
      analyzeSelectedKernel(parsed->kernel, wide, /*requireComplete=*/true);
  ASSERT_TRUE(static_cast<bool>(broad)) << llvm::toString(broad.takeError());
  EXPECT_EQ(broad->schedule.predictedCycles, perEngine);
}

// An incomplete stream is reported, never silently approximated: a loop whose
// trip count is unknown is charged a single iteration, which the lenient
// analysis labels and the strict analysis refuses.
TEST(L1ResourceDag, IncompleteStreamIsLabelledAndStrictAnalysisRefusesIt) {
  auto parsed = parseKernel(R"mlir(
module {
  micro.kernel @unknown_trip {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %n = arith.addi %c4, %c1 : index
    %a = micro.tile_alloc : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.for %i = %c0 to %n step %c1 {
      %r = micro.vector "add" %a, %a : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    }
    micro.yield
  }
}
)mlir");
  ASSERT_TRUE(parsed);
  machine::MachineModel model = parseMachine(testMachine(1, 1, 1));

  llvm::Expected<SelectedKernelAnalysis> lenient =
      analyzeSelectedKernel(parsed->kernel, model, /*requireComplete=*/false);
  ASSERT_TRUE(static_cast<bool>(lenient))
      << llvm::toString(lenient.takeError());
  EXPECT_FALSE(lenient->complete);
  ASSERT_FALSE(lenient->incompleteReasons.empty());
  EXPECT_NE(lenient->incompleteReasons.front().find("non-static bounds"),
            std::string::npos)
      << lenient->incompleteReasons.front();

  llvm::Expected<SelectedKernelAnalysis> strict =
      analyzeSelectedKernel(parsed->kernel, model, /*requireComplete=*/true);
  EXPECT_FALSE(static_cast<bool>(strict));
  if (!strict) {
    const std::string text = llvm::toString(strict.takeError());
    EXPECT_NE(text.find("incomplete"), std::string::npos) << text;
    EXPECT_NE(text.find("non-static bounds"), std::string::npos) << text;
  }
}

// The traffic and peak summaries are the analysis's own, and the L0 block reads
// the same traffic rule, so a plan consumer and the report cannot disagree.
TEST(L1ResourceDag, TrafficAndPeakSummariesMatchTheStaticBound) {
  auto parsed = parseKernel(kTwoHopKernel);
  ASSERT_TRUE(parsed);
  machine::MachineModel model = parseMachine(testMachine(1, 1, 1));

  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  llvm::Expected<SelectedKernelAnalysis> analysis =
      analyzeSelectedKernel(parsed->kernel, model, /*requireComplete=*/false);
  ASSERT_TRUE(static_cast<bool>(analysis))
      << llvm::toString(analysis.takeError());

  L0Report bound = computeL0StaticBound(*dag, model);
  EXPECT_EQ(analysis->trafficBytes, bound.bytesByMemory);
  EXPECT_EQ(analysis->peakBytes, dag->liveTileBytesByMemory);
  // The two-hop dram.0 -> l2.0 -> sram.0 movement of 256 bytes touches all
  // three levels; the intermediate sees the value once on each hop, so it
  // carries 512.
  EXPECT_EQ(analysis->trafficBytes.at("dram"), 256u);
  EXPECT_EQ(analysis->trafficBytes.at("l2"), 512u);
  EXPECT_EQ(analysis->trafficBytes.at("sram"), 256u);
}

// `micro-perf` and the planner are two consumers of one analysis (issue #129,
// task R6): the report's L0 byte block, live peak and L1 timeline are the
// shared analysis's own, not a second scheduling of the same stream.
TEST(L1ResourceDag, MicroPerfReportDerivesFromTheSharedAnalysis) {
  auto parsed = parseKernel(kChainKernel);
  ASSERT_TRUE(parsed);
  machine::MachineModel model = parseMachine(testMachine(1, 2, 4));

  llvm::Expected<MicroPerfReport> report =
      analyzeKernel(parsed->kernel, model, 1);
  ASSERT_TRUE(static_cast<bool>(report)) << llvm::toString(report.takeError());
  llvm::Expected<SelectedKernelAnalysis> analysis =
      analyzeSelectedKernel(parsed->kernel, model, /*requireComplete=*/false);
  ASSERT_TRUE(static_cast<bool>(analysis))
      << llvm::toString(analysis.takeError());

  ASSERT_TRUE(report->l1.has_value());
  EXPECT_EQ(report->l1->predictedCycles, analysis->schedule.predictedCycles);
  EXPECT_EQ(report->l0.bytesByMemory, analysis->trafficBytes);
  EXPECT_EQ(report->l0.liveTileBytesByMemory, analysis->peakBytes);
  ASSERT_EQ(report->l1->schedule.size(), analysis->schedule.entries.size());
  for (size_t i = 0; i < report->l1->schedule.size(); ++i) {
    EXPECT_EQ(report->l1->schedule[i].start,
              analysis->schedule.entries[i].start)
        << i;
    EXPECT_EQ(report->l1->schedule[i].finish,
              analysis->schedule.entries[i].finish)
        << i;
  }
}

// The analysis's `cost` is the *same* Cost `schedulePlanEvents` derives from
// the same event stream, field for field. Checking only latency would hide a
// cost whose byte totals and utilizations were never accumulated: the byte
// total, the DRAM dimension and both load factors are what this pins.
TEST(L1ResourceDag, AnalysisCostMatchesTheSharedPlanCost) {
  auto parsed = parseKernel(kChainKernel);
  ASSERT_TRUE(parsed);
  machine::MachineModel model = parseMachine(testMachine(1, 2, 4));

  llvm::Expected<SelectedKernelAnalysis> analysis =
      analyzeSelectedKernel(parsed->kernel, model, /*requireComplete=*/false);
  ASSERT_TRUE(static_cast<bool>(analysis))
      << llvm::toString(analysis.takeError());

  llvm::Expected<mapping::Cost> planned =
      mapping::schedulePlanEvents(analysis->events, model);
  ASSERT_TRUE(static_cast<bool>(planned))
      << llvm::toString(planned.takeError());

  EXPECT_DOUBLE_EQ(analysis->cost.latencyCycles, planned->latencyCycles);
  EXPECT_EQ(analysis->cost.localBytes, planned->localBytes);
  EXPECT_EQ(analysis->cost.dramBytes, planned->dramBytes);
  EXPECT_DOUBLE_EQ(analysis->cost.computeUtilization,
                   planned->computeUtilization);
  EXPECT_DOUBLE_EQ(analysis->cost.transferUtilization,
                   planned->transferUtilization);

  // The chain moves four values through the hierarchy, so a byte total of zero
  // would mean the loop never accumulated (the defect this test pins).
  EXPECT_GT(analysis->cost.localBytes, 0u);
  EXPECT_GT(analysis->cost.dramBytes, 0u);
  EXPECT_GT(analysis->cost.transferUtilization, 0.0);
}

// The shared analysis is static and deterministic: the same kernel and machine
// produce the same stream, schedule, cost and summaries every run. This is the
// parity baseline -- measured estimates are a separate mode and are never
// applied inside this analysis.
TEST(L1ResourceDag, SelectedKernelAnalysisIsStaticAndDeterministic) {
  auto parsed = parseKernel(kChainKernel);
  ASSERT_TRUE(parsed);
  machine::MachineModel model = parseMachine(testMachine(1, 2, 4));

  llvm::Expected<SelectedKernelAnalysis> first =
      analyzeSelectedKernel(parsed->kernel, model, /*requireComplete=*/false);
  llvm::Expected<SelectedKernelAnalysis> second =
      analyzeSelectedKernel(parsed->kernel, model, /*requireComplete=*/false);
  ASSERT_TRUE(static_cast<bool>(first)) << llvm::toString(first.takeError());
  ASSERT_TRUE(static_cast<bool>(second)) << llvm::toString(second.takeError());

  EXPECT_EQ(first->complete, second->complete);
  EXPECT_EQ(first->incompleteReasons, second->incompleteReasons);
  EXPECT_EQ(first->trafficBytes, second->trafficBytes);
  EXPECT_EQ(first->peakBytes, second->peakBytes);
  EXPECT_EQ(first->schedule.predictedCycles, second->schedule.predictedCycles);
  EXPECT_EQ(first->schedule.sequentialCycles,
            second->schedule.sequentialCycles);
  EXPECT_DOUBLE_EQ(first->cost.latencyCycles, second->cost.latencyCycles);
  ASSERT_EQ(first->events.events.size(), second->events.events.size());
  for (size_t i = 0; i < first->events.events.size(); ++i) {
    EXPECT_EQ(first->events.events[i].event.resource,
              second->events.events[i].event.resource)
        << i;
    EXPECT_EQ(first->events.events[i].deps, second->events.events[i].deps) << i;
    EXPECT_EQ(first->events.events[i].owner, second->events.events[i].owner)
        << i;
    EXPECT_DOUBLE_EQ(first->events.events[i].event.cost.latencyCycles,
                     second->events.events[i].event.cost.latencyCycles)
        << i;
  }
}

//===----------------------------------------------------------------------===//
// Issue #129, task R6: strict unknown / cap controls
//===----------------------------------------------------------------------===//

// An unknown compute node is a hard error at extraction, never a silently
// dropped or re-derived engine.
TEST(L1ResourceDag, UnknownComputeNodeIsRejected) {
  auto parsed = parseKernel(R"mlir(
module {
  micro.kernel @ghost_engine {
    %a = micro.tile_alloc : !micro.tile<1x1xbf16, memory = #micro.memory<sram>>
    %b = micro.tile_alloc : !micro.tile<1x1xbf16, memory = #micro.memory<sram>>
    %c = micro.tile_alloc : !micro.tile<1x1xf32, memory = #micro.memory<acc>>
    %r = micro.mma %a, %b, %c {shape = array<i64: 1, 1, 1>, input = #micro.dtype<bf16>, accumulator = #micro.dtype<f32>, engine = "mxu.ghost"} : !micro.tile<1x1xbf16, memory = #micro.memory<sram>>, !micro.tile<1x1xbf16, memory = #micro.memory<sram>>, !micro.tile<1x1xf32, memory = #micro.memory<acc>> -> !micro.tile<1x1xf32, memory = #micro.memory<acc>>
    micro.yield
  }
}
)mlir");
  ASSERT_TRUE(parsed);
  machine::MachineModel model = parseMachine(testMachine(1, 1, 1));
  llvm::Expected<SelectedKernelAnalysis> analysis =
      analyzeSelectedKernel(parsed->kernel, model, /*requireComplete=*/false);
  EXPECT_FALSE(static_cast<bool>(analysis));
  if (!analysis) {
    const std::string text = llvm::toString(analysis.takeError());
    EXPECT_NE(text.find("mxu.ghost"), std::string::npos) << text;
  }
}

// A kernel that unrolls past the event cap is a cap error -- never a silent
// single-iteration score.
TEST(L1ResourceDag, UnrollingPastTheEventCapIsACapError) {
  std::string source =
      "module {\n  micro.kernel @too_big {\n"
      "    %c0 = arith.constant 0 : index\n"
      "    %cN = arith.constant 200000 : index\n"
      "    %c1 = arith.constant 1 : index\n"
      "    %a = micro.tile_alloc : !micro.tile<8x8xf32, memory = "
      "#micro.memory<sram>>\n"
      "    micro.for %i = %c0 to %cN step %c1 {\n"
      "      %r = micro.vector \"add\" %a, %a : !micro.tile<8x8xf32, memory = "
      "#micro.memory<sram>>, !micro.tile<8x8xf32, memory = "
      "#micro.memory<sram>> -> !micro.tile<8x8xf32, memory = "
      "#micro.memory<sram>>\n"
      "    }\n"
      "    micro.yield\n  }\n}\n";
  auto parsed = parseKernel(source);
  ASSERT_TRUE(parsed);
  machine::MachineModel model = parseMachine(testMachine(1, 1, 1));
  llvm::Expected<SelectedKernelAnalysis> analysis =
      analyzeSelectedKernel(parsed->kernel, model, /*requireComplete=*/false);
  EXPECT_FALSE(static_cast<bool>(analysis));
  if (!analysis) {
    const std::string text = llvm::toString(analysis.takeError());
    EXPECT_NE(text.find("too large"), std::string::npos) << text;
  }
}

// A repeated operand and a producer chain keep their dependency edges: the
// consumer of a value depends on the event that produced it, however many times
// it reads it, and a chain orders end to end.
TEST(L1ResourceDag, RepeatedOperandsAndChainsKeepTheirDependencies) {
  auto parsed = parseKernel(R"mlir(
module {
  micro.kernel @chain {
    %a = tensor.empty() : tensor<8x8xf32>
    %t, %tok = micro.async_copy %a {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    micro.wait %tok
    %tv = micro.tile_view %t {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r1 = micro.vector "add" %tv, %tv : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r2 = micro.vector "add" %r1, %r1 : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)mlir");
  ASSERT_TRUE(parsed);
  machine::MachineModel model = parseMachine(testMachine(1, 1, 1));
  llvm::Expected<SelectedKernelAnalysis> analysis =
      analyzeSelectedKernel(parsed->kernel, model, /*requireComplete=*/false);
  ASSERT_TRUE(static_cast<bool>(analysis))
      << llvm::toString(analysis.takeError());
  // copy, wait, vector, vector: the tile_view is zero-cost metadata.
  ASSERT_EQ(analysis->events.events.size(), 4u);

  // 0 copy, 1 wait, 2 first add, 3 second add.
  const mapping::PlanCostEvent &wait = analysis->events.events[1];
  const mapping::PlanCostEvent &first = analysis->events.events[2];
  const mapping::PlanCostEvent &second = analysis->events.events[3];
  EXPECT_EQ(wait.deps, (std::vector<uint32_t>{0}));
  // The first add depends both on its data producer (the copy, which the
  // zero-cost tile view forwarded) and on the wait that orders that copy.
  EXPECT_EQ(first.deps, (std::vector<uint32_t>{0, 1}));
  // The repeated operand is one dependency, not two.
  EXPECT_EQ(second.deps, (std::vector<uint32_t>{2}));
}

// A strict analysis of a bound kernel carries the *finalized plan's* R5 storage
// liveness, not the extraction's own allocation accounting (issue #129, task
// R6 step 4: R5 liveness runs for strict analysis). The fixture is the shared
// `parallel-overlap` case, whose R5 peak is a literal the R5 suite states: two
// spatial occurrences keep both 256-byte results resident, so L2 holds 512 --
// a relation the kernel-side allocation accounting does not see at all.
TEST(L1ResourceDag, StrictAnalysisCarriesTheFinalizedPlansStorageLiveness) {
  llvm::Expected<issue129::ResourceCase> built =
      issue129::resourceCase("parallel-overlap");
  ASSERT_TRUE(bool(built)) << llvm::toString(built.takeError());
  issue129::ResourceCase &resource = *built;

  mapping::MappingSearchOptions options;
  options.mode = mapping::SearchMode::Exact;
  llvm::Expected<mapping::MappingSearchResult> result =
      issue129::searchCase(resource, options);
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  mapping::CoveringPlan plan = result->plans.front();
  if (llvm::Error error = mapping::finalizeStoragePlan(
          resource.graph, plan, resource.target->machine()))
    FAIL() << llvm::toString(std::move(error));

  llvm::Expected<mlir::Operation *> kernel =
      findMicroKernel(resource.source.get(), "");
  ASSERT_TRUE(bool(kernel)) << llvm::toString(kernel.takeError());

  // The kernel-only form runs no plan liveness and says so.
  llvm::Expected<SelectedKernelAnalysis> kernelOnly = analyzeSelectedKernel(
      *kernel, resource.target->machine(), /*requireComplete=*/false);
  ASSERT_TRUE(bool(kernelOnly)) << llvm::toString(kernelOnly.takeError());
  EXPECT_FALSE(kernelOnly->storageLivenessFromPlan);

  // The strict form, given the plan and its graph, carries the plan's liveness.
  llvm::Expected<SelectedKernelAnalysis> strict =
      analyzeSelectedKernel(*kernel, resource.target->machine(),
                            /*requireComplete=*/true, plan, resource.graph);
  ASSERT_TRUE(bool(strict)) << llvm::toString(strict.takeError());
  EXPECT_TRUE(strict->storageLivenessFromPlan);
  ASSERT_TRUE(strict->peakBytes.count("l2.0"))
      << "the plan's liveness peak names no l2.0";
  EXPECT_EQ(strict->peakBytes.at("l2.0"), 512u);
  EXPECT_NE(strict->peakBytes, kernelOnly->peakBytes);
  // The ordering the peak relies on is the plan's own, verbatim.
  EXPECT_EQ(strict->requiredReuseEdges,
            mapping::requiredReuseEdgesFor(plan.allocations));
}

} // namespace mlir::llk::perf
