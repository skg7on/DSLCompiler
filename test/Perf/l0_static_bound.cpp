//===- l0_static_bound.cpp - L0 static bound tests ------------------------===//
//
// Covers issue #46, L0 half:
//   - a tiny tile GEMM reports its flops and its bytes per memory space
//   - live materialized tile bytes are attributed to the right memory
//   - the prediction is the worse of the compute and memory lower bounds
//   - logical tile views cost nothing and produce no event
//   - loop trip counts multiply work
//   - a loop with non-static bounds is run once and says so
//   - a kernel that does not fit is reported, and one that names a memory
//     space the machine does not model is rejected
//
// The kernel sources are the fixture: they are what a lowering would emit, so
// the numbers here are derived from tile shapes and dtypes rather than from
// anything the test hard-codes about the operation.
//
//===----------------------------------------------------------------------===//

#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Machine/MachineModelLoader.h"
#include "LLK/Perf/MicroCostModel.h"
#include "LLK/Perf/MicroDAG.h"
#include "LLK/Perf/MicroPerfReport.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

namespace mlir::llk::perf {
namespace {

/// A parsed kernel plus the context and module that own it.
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

machine::MachineModel avx2Model() {
  auto model = machine::loadMachineModel(std::string(LLK_MACHINE_DIR) +
                                         "/x86-avx2-v2.yaml");
  if (!model) {
    ADD_FAILURE() << llvm::toString(model.takeError());
    return machine::MachineModel();
  }
  return *model;
}

/// One 16x16x32 bf16 GEMM fragment: two DRAM reads staged through SRAM, one
/// MMA, one store of the f32 accumulator back to DRAM.
constexpr const char *kGemmKernel = R"MLIR(
module {
  micro.kernel @gemm_tile attributes {workload = "gemm", target = "x86-avx2-cpu"} {
    %a_ext = tensor.empty() : tensor<16x32xbf16>
    %b_ext = tensor.empty() : tensor<32x16xbf16>
    %a_tile, %a_tok = micro.async_copy %a_ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<16x32xbf16> -> tensor<16x32xbf16>, !micro.async_token
    %b_tile, %b_tok = micro.async_copy %b_ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<32x16xbf16> -> tensor<32x16xbf16>, !micro.async_token
    micro.wait %a_tok, %b_tok
    %a_frag = micro.tile_view %a_tile {shape = array<i64: 16, 32>} : tensor<16x32xbf16> -> !micro.tile<16x32xbf16, memory = #micro.memory<sram>>
    %b_frag = micro.tile_view %b_tile {shape = array<i64: 32, 16>} : tensor<32x16xbf16> -> !micro.tile<32x16xbf16, memory = #micro.memory<sram>>
    %acc = micro.tile_alloc : !micro.tile<16x16xf32, memory = #micro.memory<acc>>
    %result = micro.mma %a_frag, %b_frag, %acc {shape = array<i64: 16, 16, 32>, input = #micro.dtype<bf16>, accumulator = #micro.dtype<f32>} : !micro.tile<16x32xbf16, memory = #micro.memory<sram>>, !micro.tile<32x16xbf16, memory = #micro.memory<sram>>, !micro.tile<16x16xf32, memory = #micro.memory<acc>> -> !micro.tile<16x16xf32, memory = #micro.memory<acc>>
    micro.tile_store %result {dst_memory = #micro.memory<dram>} : !micro.tile<16x16xf32, memory = #micro.memory<acc>>
    micro.yield
  }
}
)MLIR";

TEST(L0StaticBound, CountsGemmFragmentWorkAndBytes) {
  auto parsed = parseKernel(kGemmKernel);
  ASSERT_TRUE(parsed);
  machine::MachineModel model = avx2Model();

  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  L0Report report = computeL0StaticBound(*dag, model);

  // 2 * 16 * 16 * 32 multiply-accumulates.
  EXPECT_EQ(report.totalFlops, 16384u);

  // A and B are read from DRAM (1024 bytes each) and C is written back
  // (1024 bytes); the staging copies are charged to SRAM as well.
  EXPECT_EQ(report.totalBytesDram, 3072u);
  EXPECT_EQ(report.totalBytesSram, 2048u);
  EXPECT_EQ(report.bytesByMemory.at("dram"), 3072u);
  EXPECT_EQ(report.bytesByMemory.at("sram"), 2048u);
  EXPECT_EQ(report.bytesByMemory.at("acc"), 1024u);

  EXPECT_EQ(report.liveTileBytesByMemory.at("sram"), 2048u);
  EXPECT_EQ(report.liveTileBytesByMemory.at("acc"), 1024u);
}

TEST(L0StaticBound, PredictedCyclesIsTheWorseBound) {
  auto parsed = parseKernel(kGemmKernel);
  ASSERT_TRUE(parsed);
  machine::MachineModel model = avx2Model();

  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  L0Report report = computeL0StaticBound(*dag, model);

  EXPECT_GT(report.computeCyclesLowerBound, 0u);
  EXPECT_GT(report.memoryCyclesLowerBound, 0u);
  EXPECT_EQ(report.predictedCycles, std::max(report.computeCyclesLowerBound,
                                             report.memoryCyclesLowerBound));
}

TEST(L0StaticBound, BottleneckNamesTheDominantPath) {
  auto parsed = parseKernel(kGemmKernel);
  ASSERT_TRUE(parsed);
  machine::MachineModel model = avx2Model();

  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  L0Report report = computeL0StaticBound(*dag, model);

  // This fragment moves far more than it multiplies, so DRAM sets the bound.
  EXPECT_GT(report.memoryCyclesLowerBound, report.computeCyclesLowerBound);
  EXPECT_EQ(report.bottleneck, "dram_bandwidth");
}

TEST(L0StaticBound, LogicalTileViewsProduceNoEvent) {
  auto parsed = parseKernel(kGemmKernel);
  ASSERT_TRUE(parsed);
  machine::MachineModel model = avx2Model();

  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());

  // Two copies, one wait, one mma, one store. The two tile_views and the
  // tile_alloc are metadata.
  EXPECT_EQ(dag->events.size(), 5u);
  for (const MicroEvent &event : dag->events) {
    EXPECT_NE(event.kind, EventKind::TileView);
    EXPECT_NE(event.kind, EventKind::TilePartition);
  }
}

TEST(L0StaticBound, LoopTripCountsMultiplyWork) {
  constexpr const char *kLooped = R"MLIR(
module {
  micro.kernel @looped {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %acc = micro.tile_alloc : !micro.tile<16x16xf32, memory = #micro.memory<acc>>
    %a_frag = micro.tile_alloc : !micro.tile<16x32xbf16, memory = #micro.memory<sram>>
    %b_frag = micro.tile_alloc : !micro.tile<32x16xbf16, memory = #micro.memory<sram>>
    micro.for %i = %c0 to %c4 step %c1 {
      %result = micro.mma %a_frag, %b_frag, %acc {shape = array<i64: 16, 16, 32>, input = #micro.dtype<bf16>, accumulator = #micro.dtype<f32>} : !micro.tile<16x32xbf16, memory = #micro.memory<sram>>, !micro.tile<32x16xbf16, memory = #micro.memory<sram>>, !micro.tile<16x16xf32, memory = #micro.memory<acc>> -> !micro.tile<16x16xf32, memory = #micro.memory<acc>>
    }
    micro.yield
  }
}
)MLIR";

  auto parsed = parseKernel(kLooped);
  ASSERT_TRUE(parsed);
  machine::MachineModel model = avx2Model();

  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());

  EXPECT_EQ(dag->events.size(), 4u);
  L0Report report = computeL0StaticBound(*dag, model);
  EXPECT_EQ(report.totalFlops, 4 * 16384u);

  // A loop body holds one copy of its tiles however many times it runs.
  EXPECT_EQ(report.liveTileBytesByMemory.at("acc"), 1024u);
  EXPECT_EQ(report.liveTileBytesByMemory.at("sram"), 2048u);
}

TEST(L0StaticBound, NonStaticLoopIsRunOnceAndReported) {
  constexpr const char *kDynamicLoop = R"MLIR(
module {
  micro.kernel @dynamic {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %bound = arith.addi %c0, %c4 : index
    %acc = micro.tile_alloc : !micro.tile<16x16xf32, memory = #micro.memory<acc>>
    %a_frag = micro.tile_alloc : !micro.tile<16x32xbf16, memory = #micro.memory<sram>>
    %b_frag = micro.tile_alloc : !micro.tile<32x16xbf16, memory = #micro.memory<sram>>
    micro.for %i = %c0 to %bound step %c1 {
      %result = micro.mma %a_frag, %b_frag, %acc {shape = array<i64: 16, 16, 32>, input = #micro.dtype<bf16>, accumulator = #micro.dtype<f32>} : !micro.tile<16x32xbf16, memory = #micro.memory<sram>>, !micro.tile<32x16xbf16, memory = #micro.memory<sram>>, !micro.tile<16x16xf32, memory = #micro.memory<acc>> -> !micro.tile<16x16xf32, memory = #micro.memory<acc>>
    }
    micro.yield
  }
}
)MLIR";

  auto parsed = parseKernel(kDynamicLoop);
  ASSERT_TRUE(parsed);
  machine::MachineModel model = avx2Model();

  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());

  // One iteration, and the caller is told the count is a guess.
  EXPECT_EQ(dag->events.size(), 1u);
  ASSERT_EQ(dag->warnings.size(), 1u);
  EXPECT_NE(dag->warnings.front().find("non-static bounds"), std::string::npos)
      << dag->warnings.front();
}

TEST(L0StaticBound, LayoutTransformIsNotFree) {
  constexpr const char *kSameLayout = R"MLIR(
module {
  micro.kernel @logical {
    %t = micro.tile_alloc : !micro.tile<32x32xbf16, memory = #micro.memory<sram>, layout = #micro.layout<row_major>>
    %frag = micro.tile_partition %t {shape = array<i64: 16, 16>} : !micro.tile<32x32xbf16, memory = #micro.memory<sram>, layout = #micro.layout<row_major>> -> !micro.tile<16x16xbf16, memory = #micro.memory<sram>, layout = #micro.layout<row_major>>
    micro.yield
  }
}
)MLIR";

  constexpr const char *kChangedLayout = R"MLIR(
module {
  micro.kernel @transformed {
    %t = micro.tile_alloc : !micro.tile<32x32xbf16, memory = #micro.memory<sram>, layout = #micro.layout<row_major>>
    %frag = micro.tile_partition %t {shape = array<i64: 16, 16>} : !micro.tile<32x32xbf16, memory = #micro.memory<sram>, layout = #micro.layout<row_major>> -> !micro.tile<16x16xbf16, memory = #micro.memory<sram>, layout = #micro.layout<vectorized, vector = 8>>
    micro.yield
  }
}
)MLIR";

  machine::MachineModel model = avx2Model();

  auto logical = parseKernel(kSameLayout);
  ASSERT_TRUE(logical);
  auto logicalDag = buildMicroDAG(logical->kernel, model);
  ASSERT_TRUE(static_cast<bool>(logicalDag))
      << llvm::toString(logicalDag.takeError());
  EXPECT_TRUE(logicalDag->events.empty());

  auto transformed = parseKernel(kChangedLayout);
  ASSERT_TRUE(transformed);
  auto transformedDag = buildMicroDAG(transformed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(transformedDag))
      << llvm::toString(transformedDag.takeError());

  // The same op with a different result layout is real work.
  ASSERT_EQ(transformedDag->events.size(), 1u);
  EXPECT_EQ(transformedDag->events.front().kind, EventKind::TilePartition);
  EXPECT_EQ(transformedDag->events.front().workItems, 256u);
  EXPECT_GT(transformedDag->events.front().minCycles, 0u);
}

TEST(L0StaticBound, OverflowingTileIsReportedAsACapacityViolation) {
  constexpr const char *kTooBig = R"MLIR(
module {
  micro.kernel @too_big {
    %big = micro.tile_alloc : !micro.tile<256x256xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)MLIR";

  auto parsed = parseKernel(kTooBig);
  ASSERT_TRUE(parsed);
  machine::MachineModel model = avx2Model();

  auto report = analyzeKernel(parsed->kernel, model, /*level=*/0);
  ASSERT_TRUE(static_cast<bool>(report)) << llvm::toString(report.takeError());

  const machine::MemoryNode *sram = model.findMemoryOfKind("sram");
  ASSERT_NE(sram, nullptr);
  ASSERT_EQ(report->capacityViolations.size(), 1u);
  EXPECT_EQ(report->capacityViolations.front(),
            "memory sram requires 262144 bytes but machine has " +
                std::to_string(sram->capacityBytes) + " bytes");
}

//===----------------------------------------------------------------------===//
// Issue #129, task R7: a mapped kernel's occupancy is its plan's, not a sum
//===----------------------------------------------------------------------===//

/// Three 64x64xf32 kernel allocations (16384 bytes each) is what the extraction
/// sees -- a monotonic sum of 49152 bytes, which overflows the shipped
/// `sram.0` (32768). The kernel's recorded plan, though, aliases two of them
/// onto the first, so the *live* peak is one 16384-byte buffer and it fits. The
/// report must read the plan's relation rather than the extraction's sum.
TEST(L0StaticBound, AMappedKernelsOccupancyIsThePlansLivePeak) {
  auto parsed = parseKernel(R"MLIR(
module {
  micro.kernel @planned_fit attributes {micro.plan = {allocations = [
      {begin_step = 0 : i64, bytes = 16384 : i64, end_step = 2 : i64, id = 1 : i64, memory = "sram.0", value = 0 : i64},
      {alias_of = 1 : i64, begin_step = 1 : i64, bytes = 16384 : i64, end_step = 2 : i64, id = 2 : i64, memory = "sram.0", value = 1 : i64},
      {alias_of = 1 : i64, begin_step = 2 : i64, bytes = 16384 : i64, end_step = 2 : i64, id = 3 : i64, memory = "sram.0", value = 2 : i64}
    ]}} {
    %a = micro.tile_alloc : !micro.tile<64x64xf32, memory = #micro.memory<sram>>
    %b = micro.tile_alloc : !micro.tile<64x64xf32, memory = #micro.memory<sram>>
    %c = micro.tile_alloc : !micro.tile<64x64xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)MLIR");
  ASSERT_TRUE(parsed);
  machine::MachineModel model = avx2Model();

  // Premise: the extraction's sum overflows, so this test discriminates the two
  // accountings rather than passing either way.
  const machine::MemoryNode *sram = model.findMemoryOfKind("sram");
  ASSERT_NE(sram, nullptr);
  ASSERT_GT(3u * 16384u, sram->capacityBytes);

  auto report = analyzeKernel(parsed->kernel, model, /*level=*/0);
  ASSERT_TRUE(static_cast<bool>(report)) << llvm::toString(report.takeError());
  ASSERT_EQ(report->l0.liveTileBytesByMemory.count("sram"), 1u);
  EXPECT_EQ(report->l0.liveTileBytesByMemory.at("sram"), 16384u);
  EXPECT_TRUE(report->capacityViolations.empty());
}

/// The complement: reading the plan's peak does not weaken the check. A plan
/// whose own recorded buffer exceeds its memory is still a violation, named
/// against the node that cannot hold it.
TEST(L0StaticBound, AMappedKernelsOverCapacityPlanIsStillRejected) {
  auto parsed = parseKernel(R"MLIR(
module {
  micro.kernel @planned_overflow attributes {micro.plan = {allocations = [
      {begin_step = 0 : i64, bytes = 40000 : i64, end_step = 2 : i64, id = 1 : i64, memory = "sram.0", value = 0 : i64}
    ]}} {
    %a = micro.tile_alloc : !micro.tile<64x64xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)MLIR");
  ASSERT_TRUE(parsed);
  machine::MachineModel model = avx2Model();

  auto report = analyzeKernel(parsed->kernel, model, /*level=*/0);
  ASSERT_TRUE(static_cast<bool>(report)) << llvm::toString(report.takeError());
  const machine::MemoryNode *sram = model.findMemory("sram.0");
  ASSERT_NE(sram, nullptr);
  ASSERT_EQ(report->capacityViolations.size(), 1u);
  EXPECT_EQ(report->capacityViolations.front(),
            "memory sram.0 requires 40000 bytes but machine has " +
                std::to_string(sram->capacityBytes) + " bytes");
}

//===----------------------------------------------------------------------===//
// Machine-fit diagnostics
//===----------------------------------------------------------------------===//

/// True when some warning contains `needle`.
bool mentions(const std::vector<std::string> &warnings,
              llvm::StringRef needle) {
  return llvm::any_of(warnings, [&](const std::string &warning) {
    return warning.find(needle.str()) != std::string::npos;
  });
}

TEST(MachineFit, UnsupportedLayoutIsReported) {
  constexpr const char *kSwizzled = R"MLIR(
module {
  micro.kernel @swizzled {
    %t = micro.tile_alloc : !micro.tile<16x32xbf16, memory = #micro.memory<sram>, layout = #micro.layout<swizzled, banks = 32, swizzle = "xor">>
    %v = micro.vector "silu" %t : !micro.tile<16x32xbf16, memory = #micro.memory<sram>, layout = #micro.layout<swizzled, banks = 32, swizzle = "xor">> -> !micro.tile<16x32xbf16, memory = #micro.memory<sram>, layout = #micro.layout<swizzled, banks = 32, swizzle = "xor">>
    micro.yield
  }
}
)MLIR";

  auto parsed = parseKernel(kSwizzled);
  ASSERT_TRUE(parsed);
  machine::MachineModel model = avx2Model();

  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());

  // Both the memory that would hold the tile and the engine that would shuffle
  // it are asked, and both can say no.
  EXPECT_TRUE(mentions(dag->layoutWarnings,
                       "memory 'sram' does not support layout 'swizzled'"))
      << (dag->layoutWarnings.empty() ? "no layout warnings"
                                      : dag->layoutWarnings.front());
  EXPECT_TRUE(mentions(dag->layoutWarnings,
                       "engine 'vpu' does not support layout "
                       "'swizzled'"));
}

TEST(MachineFit, UnsizeableElementTypeIsReported) {
  // The micro dialect does not restrict movement operands to its own dtype
  // vocabulary, so an f64 tile can reach the simulator. It has no byte size
  // here, and charging zero for it must not be silent.
  constexpr const char *kF64 = R"MLIR(
module {
  micro.kernel @f64 {
    %a_ext = tensor.empty() : tensor<16x32xf64>
    %a_tile, %a_tok = micro.async_copy %a_ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<16x32xf64> -> tensor<16x32xf64>, !micro.async_token
    micro.wait %a_tok
    micro.yield
  }
}
)MLIR";

  auto parsed = parseKernel(kF64);
  ASSERT_TRUE(parsed);
  machine::MachineModel model = avx2Model();

  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());

  ASSERT_FALSE(dag->events.empty());
  EXPECT_EQ(dag->events.front().bytes, 0u);
  EXPECT_TRUE(mentions(dag->warnings, "not a micro dtype"))
      << (dag->warnings.empty() ? "no warnings" : dag->warnings.front());
}

TEST(MachineFit, UnmodeledOwnerIsReported) {
  constexpr const char *kUnmodeledOwner = R"MLIR(
module {
  micro.kernel @unmodeled_owner {
    %acc = micro.tile_alloc : !micro.tile<16x16xf32, memory = #micro.memory<acc>, owner = #micro.owner<pe>>
    %a_frag = micro.tile_alloc : !micro.tile<16x32xbf16, memory = #micro.memory<sram>>
    %b_frag = micro.tile_alloc : !micro.tile<32x16xbf16, memory = #micro.memory<sram>>
    %result = micro.mma %a_frag, %b_frag, %acc {shape = array<i64: 16, 16, 32>, input = #micro.dtype<bf16>, accumulator = #micro.dtype<f32>} : !micro.tile<16x32xbf16, memory = #micro.memory<sram>>, !micro.tile<32x16xbf16, memory = #micro.memory<sram>>, !micro.tile<16x16xf32, memory = #micro.memory<acc>, owner = #micro.owner<pe>> -> !micro.tile<16x16xf32, memory = #micro.memory<acc>, owner = #micro.owner<pe>>
    micro.yield
  }
}
)MLIR";

  auto parsed = parseKernel(kUnmodeledOwner);
  ASSERT_TRUE(parsed);
  machine::MachineModel model = avx2Model();

  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());

  // 'pe' is a micro owner, but this machine is a thread-parallel CPU and does
  // not declare one, so the occupancy constraint cannot be modeled.
  EXPECT_TRUE(mentions(dag->ownerWarnings,
                       "owner 'pe' is not modeled by machine 'x86-avx2'"))
      << (dag->ownerWarnings.empty() ? "no owner warnings"
                                     : dag->ownerWarnings.front());
}

TEST(L0StaticBound, UnmodeledMemorySpaceIsRejected) {
  constexpr const char *kScratchCopy = R"MLIR(
module {
  micro.kernel @scratch {
    %a_ext = tensor.empty() : tensor<16x32xbf16>
    %a_tile, %a_tok = micro.async_copy %a_ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<scratch>} : tensor<16x32xbf16> -> tensor<16x32xbf16>, !micro.async_token
    micro.wait %a_tok
    micro.yield
  }
}
)MLIR";

  auto parsed = parseKernel(kScratchCopy);
  ASSERT_TRUE(parsed);
  machine::MachineModel model = avx2Model();

  auto dag = buildMicroDAG(parsed->kernel, model);
  ASSERT_FALSE(static_cast<bool>(dag));
  std::string message = llvm::toString(dag.takeError());
  EXPECT_NE(message.find("'scratch'"), std::string::npos) << message;
  EXPECT_NE(message.find("does not model"), std::string::npos) << message;
}

} // namespace
} // namespace mlir::llk::perf
