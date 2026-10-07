//===- candidate_binding.cpp - Candidate -> concrete kernel tests ---------===//
//
// Covers issue #50, binding half. A candidate is a point in the search space;
// binding turns it into a concrete `micro.kernel` the performance model can
// cost. The tests assert three things:
//
//   * the kernel is concrete -- no micro.param, micro.constraint,
//     micro.objective, micro.candidate, or micro.search_space survives
//   * the numeric bindings are visible in the IR -- tile sizes in the spatial
//     loop steps and tensor extents, pipeline depth on micro.pipeline
//   * the symbolic bindings are visible in tile metadata -- layout in
//     #micro.layout, memory in the tile memory spaces, owner in the spatial
//     maps, fragment in micro.tile_partition and the MMA shape
//
// The returned decisions are asserted too: the schedule record persists them,
// so they have to be the values the kernel actually embodies.
//
//===----------------------------------------------------------------------===//

#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Dialect/Micro/MicroEnums.h"
#include "LLK/Perf/CandidateBinding.h"
#include "LLK/Perf/CandidateGenerator.h"
#include "LLK/Perf/SearchSpace.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

// Micro attribute, type, and op declarations.
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace mlir::llk::perf {
namespace {

using llvm::StringRef;

SearchParam integerParam(std::string name, std::vector<SearchChoice> choices) {
  return SearchParam{std::move(name), "integer", std::move(choices)};
}

SearchParam symbolicParam(std::string name, std::string kind,
                          std::vector<SearchChoice> choices) {
  return SearchParam{std::move(name), std::move(kind), std::move(choices)};
}

/// A space shaped like the M11 export's fused_swiglu output, so binding is
/// exercised on the IR the tuner actually receives.
SearchSpace swigluSpace() {
  SearchSpace space;
  space.name = "fused_swiglu_M8_N64_K64";
  space.workload = "fused_swiglu";
  space.params = {
      integerParam("BM", {SearchChoice(int64_t{1}), SearchChoice(int64_t{4}),
                          SearchChoice(int64_t{8}), SearchChoice(int64_t{16})}),
      integerParam("BN", {SearchChoice(int64_t{16}), SearchChoice(int64_t{32}),
                          SearchChoice(int64_t{64})}),
      integerParam("BK",
                   {SearchChoice(int64_t{32}), SearchChoice(int64_t{64})}),
      integerParam("VM", {SearchChoice(int64_t{1})}),
      integerParam("VN", {SearchChoice(int64_t{4})}),
      integerParam("vector_width", {SearchChoice(int64_t{8})}),
      integerParam("num_threads",
                   {SearchChoice(int64_t{1}), SearchChoice(int64_t{8})}),
      integerParam("grain_size", {SearchChoice(int64_t{1})}),
      integerParam("pipeline_stages",
                   {SearchChoice(int64_t{1}), SearchChoice(int64_t{2})}),
      symbolicParam("tile_layout", "layout",
                    {SearchChoice("row_major"), SearchChoice("blocked")}),
      symbolicParam(
          "memory_path", "memory_path",
          {SearchChoice("dram:sram:acc"), SearchChoice("dram:l2:sram:acc")}),
      symbolicParam(
          "owner_mapping", "owner_mapping",
          {SearchChoice("worker/lane"), SearchChoice("worker/vector_engine")}),
      symbolicParam("fragment_shape", "fragment_shape",
                    {SearchChoice("16x16x32"), SearchChoice("8x8x32")}),
      symbolicParam("tail_policy", "tail_policy", {SearchChoice("mask")}),
  };
  return space;
}

/// The space's preferred point: the schedule the search space was anchored on.
Candidate preferred() {
  Candidate candidate;
  candidate.values = {
      {"BM", 8},          {"BN", 64},        {"BK", 64},
      {"VM", 1},          {"VN", 4},         {"vector_width", 8},
      {"num_threads", 8}, {"grain_size", 1}, {"pipeline_stages", 1}};
  candidate.symbolicValues = {{"tile_layout", "row_major"},
                              {"memory_path", "dram:sram:acc"},
                              {"owner_mapping", "worker/vector_engine"},
                              {"fragment_shape", "16x16x32"},
                              {"tail_policy", "mask"}};
  candidate.id = computeCandidateId(candidate.values, candidate.symbolicValues);
  return candidate;
}

WorkloadShape swigluShape() {
  WorkloadShape shape;
  shape.M = 8;
  shape.N = 64;
  shape.K = 64;
  return shape;
}

/// A fresh, detached module for the bound kernel to land in.
struct Scratch {
  std::unique_ptr<MLIRContext> context;
  OwningOpRef<ModuleOp> module;
};

std::unique_ptr<Scratch> makeScratch() {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::micro::MicroDialect>();

  auto scratch = std::make_unique<Scratch>();
  scratch->context = std::make_unique<MLIRContext>(registry);
  scratch->module = ModuleOp::create(UnknownLoc::get(scratch->context.get()));
  return scratch;
}

/// Reads an index constant produced by the binder.
std::optional<int64_t> constantIndex(Value value) {
  auto constant = value.getDefiningOp<arith::ConstantIndexOp>();
  if (!constant)
    return std::nullopt;
  return constant.value();
}

micro::KernelOp onlyKernel(ModuleOp module) {
  micro::KernelOp found;
  module.walk([&](micro::KernelOp kernel) { found = kernel; });
  return found;
}

/// Counts ops of one kind anywhere in `module`, including inside regions.
template <typename Op> unsigned countOps(ModuleOp module) {
  unsigned count = 0;
  module.walk([&](Op) { ++count; });
  return count;
}

//===----------------------------------------------------------------------===//
// Binding decisions
//===----------------------------------------------------------------------===//

TEST(CandidateBinding, ResolvesTheCandidateIntoConcreteTileDecisions) {
  SearchSpace space = swigluSpace();
  Candidate candidate = preferred();
  WorkloadShape shape = swigluShape();

  auto scratch = makeScratch();
  auto bound = bindCandidateToMicroKernel(scratch->module.get(), space,
                                          candidate, shape);
  ASSERT_TRUE(bool(bound)) << llvm::toString(bound.takeError());

  EXPECT_NE(bound->symbolName.find("fused_swiglu"), std::string::npos);
  EXPECT_NE(bound->symbolName.find(candidate.id), std::string::npos)
      << "the kernel symbol identifies the bound candidate";
  EXPECT_EQ(bound->workload, "fused_swiglu");

  const BoundTileDecisions &decisions = bound->decisions;
  EXPECT_EQ(decisions.workerTile, (std::vector<int64_t>{8, 64, 64}));
  EXPECT_EQ(decisions.declaredFragment, (std::vector<int64_t>{16, 16, 32}));
  // BM = 8 clamps the fragment's M to the largest divisor of 8 at most 16.
  EXPECT_EQ(decisions.fragmentShape, (std::vector<int64_t>{8, 16, 32}));
  EXPECT_EQ(decisions.tileLayout, "row_major");
  EXPECT_EQ(decisions.memoryPath,
            (std::vector<std::string>{"dram", "sram", "acc"}));
  EXPECT_EQ(decisions.sourceSpace, "dram");
  EXPECT_EQ(decisions.stagingSpace, "sram");
  EXPECT_EQ(decisions.accumulatorSpace, "acc");
  EXPECT_EQ(decisions.outerOwner, "worker");
  EXPECT_EQ(decisions.fragmentOwner, "vector_engine");
  EXPECT_EQ(decisions.tailPolicy, "mask");
  EXPECT_EQ(decisions.pipelineStages, 1);
  EXPECT_EQ(decisions.vectorWidth, 8);
}

TEST(CandidateBinding, RejectsALayoutTheBinderCannotRealize) {
  // `blocked` is a legal layout *kind*, but a `#micro.layout<blocked>` is only
  // well-formed with a block parameter the binder does not carry. Binding it
  // must be a rejected candidate: building the attribute anyway fails its
  // verifier and aborts the process -- which is exactly how `llk-tune` crashed
  // on a compiler-generated search space before this was checked.
  SearchSpace space = swigluSpace();
  Candidate candidate = preferred();
  candidate.symbolicValues["tile_layout"] = "blocked";

  auto scratch = makeScratch();
  auto bound = bindCandidateToMicroKernel(scratch->module.get(), space,
                                          candidate, swigluShape());
  ASSERT_FALSE(static_cast<bool>(bound));
  EXPECT_NE(llvm::toString(bound.takeError()).find("block/swizzle parameter"),
            std::string::npos);
}

TEST(CandidateBinding, ExtendsAKnownFragmentFromTheMemoryPath) {
  SearchSpace space = swigluSpace();
  Candidate candidate = preferred();
  candidate.symbolicValues["memory_path"] = "dram:l2:sram:acc";

  auto scratch = makeScratch();
  auto bound = bindCandidateToMicroKernel(scratch->module.get(), space,
                                          candidate, swigluShape());
  ASSERT_TRUE(bool(bound)) << llvm::toString(bound.takeError());
  EXPECT_EQ(bound->decisions.stagingSpace, "l2");
  EXPECT_EQ(bound->decisions.accumulatorSpace, "acc");
}

TEST(CandidateBinding, ClampsATileToTheProblemExtent) {
  SearchSpace space = swigluSpace();
  Candidate candidate = preferred();
  candidate.values["BM"] = 64; // the problem has M = 8

  auto scratch = makeScratch();
  auto bound = bindCandidateToMicroKernel(scratch->module.get(), space,
                                          candidate, swigluShape());
  ASSERT_TRUE(bool(bound)) << llvm::toString(bound.takeError());
  EXPECT_EQ(bound->decisions.workerTile[0], 8);
}

//===----------------------------------------------------------------------===//
// Emitted IR
//===----------------------------------------------------------------------===//

TEST(CandidateBinding, EmitsExactlyOneConcreteKernel) {
  SearchSpace space = swigluSpace();
  Candidate candidate = preferred();
  auto scratch = makeScratch();
  auto bound = bindCandidateToMicroKernel(scratch->module.get(), space,
                                          candidate, swigluShape());
  ASSERT_TRUE(bool(bound)) << llvm::toString(bound.takeError());

  EXPECT_EQ(countOps<micro::KernelOp>(scratch->module.get()), 1u);
  EXPECT_EQ(onlyKernel(scratch->module.get()).getSymName(), bound->symbolName);
}

TEST(CandidateBinding, EmitsAVerifiableKernel) {
  SearchSpace space = swigluSpace();
  Candidate candidate = preferred();
  auto scratch = makeScratch();
  auto bound = bindCandidateToMicroKernel(scratch->module.get(), space,
                                          candidate, swigluShape());
  ASSERT_TRUE(bool(bound)) << llvm::toString(bound.takeError());

  // The verifier is the contract the dialect ops hold themselves to; a bound
  // kernel the simulator accepts but the verifier rejects would be a kernel no
  // later lowering could consume.
  EXPECT_TRUE(mlir::succeeded(mlir::verify(scratch->module.get())));
}

TEST(CandidateBinding, BindsAMatmulWithoutTheFusedEpilogue) {
  SearchSpace space = swigluSpace();
  space.workload = "matmul";
  Candidate candidate = preferred();
  candidate.values["BM"] = 16; // M = 16, and a single matmul accumulator fits
  WorkloadShape shape = swigluShape();
  shape.M = 16;

  auto scratch = makeScratch();
  auto bound = bindCandidateToMicroKernel(scratch->module.get(), space,
                                          candidate, shape);
  ASSERT_TRUE(bool(bound)) << llvm::toString(bound.takeError());

  // One weight operand, one accumulator, and no SwiGLU activation. The
  // operands are the kernel's declared inputs, so nothing is materialized for
  // the builder to guess at.
  EXPECT_EQ(countOps<tensor::EmptyOp>(scratch->module.get()), 0u);
  ModuleOp matmul = scratch->module.get();
  auto matmulKernel = *matmul.getOps<micro::KernelOp>().begin();
  ASSERT_TRUE(matmulKernel.getKernelFunctionType());
  EXPECT_EQ(matmulKernel.getKernelFunctionType().getNumInputs(), 2u);
  EXPECT_EQ(countOps<micro::MmaOp>(scratch->module.get()), 1u);
  unsigned silu = 0;
  scratch->module.get().walk([&](micro::VectorOp op) {
    if (op.getOp() == "silu")
      ++silu;
  });
  EXPECT_EQ(silu, 0u);
}

TEST(CandidateBinding, LeavesNoSearchOpBehind) {
  SearchSpace space = swigluSpace();
  Candidate candidate = preferred();
  auto scratch = makeScratch();
  auto bound = bindCandidateToMicroKernel(scratch->module.get(), space,
                                          candidate, swigluShape());
  ASSERT_TRUE(bool(bound)) << llvm::toString(bound.takeError());

  ModuleOp module = scratch->module.get();
  EXPECT_EQ(countOps<micro::SearchSpaceOp>(module), 0u);
  EXPECT_EQ(countOps<micro::ParamOp>(module), 0u);
  EXPECT_EQ(countOps<micro::ConstraintOp>(module), 0u);
  EXPECT_EQ(countOps<micro::ObjectiveOp>(module), 0u);
  EXPECT_EQ(countOps<micro::CandidateOp>(module), 0u);
}

TEST(CandidateBinding, RecordsTheCandidateOnTheKernel) {
  SearchSpace space = swigluSpace();
  Candidate candidate = preferred();
  auto scratch = makeScratch();
  auto bound = bindCandidateToMicroKernel(scratch->module.get(), space,
                                          candidate, swigluShape());
  ASSERT_TRUE(bool(bound)) << llvm::toString(bound.takeError());

  micro::KernelOp kernel = onlyKernel(scratch->module.get());
  ASSERT_TRUE(kernel.getOperation());
  EXPECT_EQ(kernel.getCandidate(), candidate.id);
  EXPECT_EQ(
      kernel->getAttrOfType<DenseI64ArrayAttr>("fragment_shape").asArrayRef(),
      (ArrayRef<int64_t>{8, 16, 32}));
  EXPECT_EQ(kernel->getAttrOfType<DenseI64ArrayAttr>("mma_shape").asArrayRef(),
            (ArrayRef<int64_t>{16, 16, 32}));
  EXPECT_EQ(kernel->getAttrOfType<StringAttr>("memory_path").getValue(),
            "dram:sram:acc");
  EXPECT_EQ(kernel->getAttrOfType<StringAttr>("owner_mapping").getValue(),
            "worker/vector_engine");
  EXPECT_EQ(kernel->getAttrOfType<StringAttr>("tile_layout").getValue(),
            "row_major");
  EXPECT_EQ(kernel->getAttrOfType<StringAttr>("tail_policy").getValue(),
            "mask");
}

TEST(CandidateBinding, BindsTileSizesIntoLoopStepsAndTensorExtents) {
  SearchSpace space = swigluSpace();
  Candidate candidate = preferred();
  auto scratch = makeScratch();
  auto bound = bindCandidateToMicroKernel(scratch->module.get(), space,
                                          candidate, swigluShape());
  ASSERT_TRUE(bool(bound)) << llvm::toString(bound.takeError());

  ModuleOp module = scratch->module.get();

  // Spatial loops: M by BM, N by BN. The walk is pre-order so the outer M loop
  // comes first even though the N loop is nested inside it.
  std::vector<int64_t> spatialSteps;
  module.walk<WalkOrder::PreOrder>([&](micro::SpatialForOp loop) {
    if (auto step = constantIndex(loop.getStep()))
      spatialSteps.push_back(*step);
  });
  EXPECT_EQ(spatialSteps, (std::vector<int64_t>{8, 64}));

  // The K loop steps by BK.
  std::vector<int64_t> forSteps;
  module.walk([&](micro::ForOp loop) {
    if (auto step = constantIndex(loop.getStep()))
      forSteps.push_back(*step);
  });
  EXPECT_EQ(forSteps, (std::vector<int64_t>{64}));

  // External operands: A [M, K], then two B [K, N]. They are the kernel's
  // declared inputs, so their extents are read from the signature rather than
  // from tensors the builder materializes.
  auto kernel = *module.getOps<micro::KernelOp>().begin();
  mlir::FunctionType signature = kernel.getKernelFunctionType();
  ASSERT_TRUE(signature);
  std::vector<std::vector<int64_t>> inputShapes;
  for (mlir::Type input : signature.getInputs()) {
    ArrayRef<int64_t> shape = mlir::cast<ShapedType>(input).getShape();
    inputShapes.push_back(std::vector<int64_t>(shape.begin(), shape.end()));
  }
  EXPECT_EQ(inputShapes,
            (std::vector<std::vector<int64_t>>{{8, 64}, {64, 64}, {64, 64}}));
}

TEST(CandidateBinding, BindsLayoutMemoryAndOwnerIntoTileMetadata) {
  SearchSpace space = swigluSpace();
  Candidate candidate = preferred();
  auto scratch = makeScratch();
  auto bound = bindCandidateToMicroKernel(scratch->module.get(), space,
                                          candidate, swigluShape());
  ASSERT_TRUE(bool(bound)) << llvm::toString(bound.takeError());

  ModuleOp module = scratch->module.get();

  // The accumulator is an acc-memory tile owned by the outer owner.
  bool sawAccumulator = false;
  bool sawStaging = false;
  module.walk([&](Operation *op) {
    for (Value result : op->getResults()) {
      auto tile = dyn_cast<micro::TileType>(result.getType());
      if (!tile)
        continue;
      if (tile.getMemory().getValue() == micro::MemorySpace::acc) {
        sawAccumulator = true;
        EXPECT_EQ(tile.getShape(), (ArrayRef<int64_t>{8, 64}));
        EXPECT_EQ(tile.getOwner().getSymbol().str(), "worker");
        EXPECT_FALSE(tile.getLayout());
      }
      if (tile.getMemory().getValue() == micro::MemorySpace::sram) {
        sawStaging = true;
        ASSERT_TRUE(bool(tile.getLayout()));
        EXPECT_EQ(tile.getLayout().getKind(),
                  static_cast<uint32_t>(micro::LayoutKind::row_major));
        EXPECT_EQ(tile.getLayout().getVector(), 8);
      }
    }
  });
  EXPECT_TRUE(sawAccumulator) << "no accumulator tile in acc memory";
  EXPECT_TRUE(sawStaging) << "no staged tile in sram";

  // The fragment is bound as a partition of the staged tile. A partition is
  // operand-local and two-dimensional; the three-dimensional fragment triple
  // lives on the kernel attributes asserted elsewhere.
  std::vector<std::vector<int64_t>> partitions;
  module.walk([&](micro::TilePartitionOp partition) {
    partitions.push_back(std::vector<int64_t>(partition.getShape().begin(),
                                              partition.getShape().end()));
    ASSERT_TRUE(partition.getOwner().has_value());
    EXPECT_EQ(partition.getOwner()->getSymbol().str(), "vector_engine");
  });
  EXPECT_EQ(partitions,
            (std::vector<std::vector<int64_t>>{{8, 32}, {32, 16}, {32, 16}}));

  // The MMA's shape is the worker tile; the partition is metadata.
  auto mma = *module.getOps<micro::KernelOp>().begin();
  bool sawMma = false;
  mma.walk([&](micro::MmaOp op) {
    sawMma = true;
    EXPECT_EQ(op.getShape(), (ArrayRef<int64_t>{8, 64, 64}));
    EXPECT_FALSE(bool(op.getEngineAttr()));
  });
  EXPECT_TRUE(sawMma) << "no MMA in the bound kernel";
}

TEST(CandidateBinding, BindsPipelineDepth) {
  SearchSpace space = swigluSpace();
  Candidate candidate = preferred();
  candidate.values["pipeline_stages"] = 2;

  auto scratch = makeScratch();
  auto bound = bindCandidateToMicroKernel(scratch->module.get(), space,
                                          candidate, swigluShape());
  ASSERT_TRUE(bool(bound)) << llvm::toString(bound.takeError());

  micro::PipelineOp pipeline;
  scratch->module.get().walk([&](micro::PipelineOp op) { pipeline = op; });
  ASSERT_TRUE(bool(pipeline));
  EXPECT_EQ(pipeline.getStages(), 2u);
}

//===----------------------------------------------------------------------===//
// Rejections
//===----------------------------------------------------------------------===//

TEST(CandidateBinding, RejectsACandidateThatDoesNotBindEveryParameter) {
  SearchSpace space = swigluSpace();
  Candidate candidate = preferred();
  candidate.values.erase("BN");

  auto scratch = makeScratch();
  auto bound = bindCandidateToMicroKernel(scratch->module.get(), space,
                                          candidate, swigluShape());
  ASSERT_FALSE(bool(bound));
  EXPECT_NE(llvm::toString(bound.takeError()).find("BN"), std::string::npos);
}

TEST(CandidateBinding, RejectsAnUnknownLayout) {
  SearchSpace space = swigluSpace();
  Candidate candidate = preferred();
  candidate.symbolicValues["tile_layout"] = "not_a_layout";

  auto scratch = makeScratch();
  auto bound = bindCandidateToMicroKernel(scratch->module.get(), space,
                                          candidate, swigluShape());
  ASSERT_FALSE(bool(bound));
  EXPECT_NE(llvm::toString(bound.takeError()).find("not_a_layout"),
            std::string::npos);
}

TEST(CandidateBinding, RejectsAnUnknownMemorySpaceInThePath) {
  SearchSpace space = swigluSpace();
  Candidate candidate = preferred();
  candidate.symbolicValues["memory_path"] = "dram:hyper:acc";

  auto scratch = makeScratch();
  auto bound = bindCandidateToMicroKernel(scratch->module.get(), space,
                                          candidate, swigluShape());
  ASSERT_FALSE(bool(bound));
  EXPECT_NE(llvm::toString(bound.takeError()).find("hyper"), std::string::npos);
}

TEST(CandidateBinding, RejectsATileThatDoesNotDivideTheProblem) {
  SearchSpace space = swigluSpace();
  Candidate candidate = preferred();
  candidate.values["BM"] = 8;
  WorkloadShape shape = swigluShape();
  shape.M = 12; // 12 % 8 != 0

  auto scratch = makeScratch();
  auto bound = bindCandidateToMicroKernel(scratch->module.get(), space,
                                          candidate, shape);
  ASSERT_FALSE(bool(bound));
  EXPECT_NE(llvm::toString(bound.takeError()).find("divide"),
            std::string::npos);
}

//===----------------------------------------------------------------------===//
// End to end with the real loader
//===----------------------------------------------------------------------===//

const char *kExportedSpace = R"MLIR(
micro.search_space @fused_swiglu_M8_N64_K64 attributes {workload = "fused_swiglu"} {
  micro.param "BM" {kind = "integer", choices = [1 : i64, 4 : i64, 8 : i64]}
  micro.param "BN" {kind = "integer", choices = [16 : i64, 64 : i64]}
  micro.param "BK" {kind = "integer", choices = [32 : i64, 64 : i64]}
  micro.param "pipeline_stages" {kind = "integer", choices = [1 : i64, 2 : i64]}
  micro.param "vector_width" {kind = "integer", choices = [8 : i64]}
  micro.param "tile_layout" {kind = "layout", choices = ["row_major", "blocked"]}
  micro.param "memory_path" {kind = "memory_path", choices = ["dram:sram:acc"]}
  micro.param "owner_mapping" {kind = "owner_mapping", choices = ["worker/vector_engine"]}
  micro.param "fragment_shape" {kind = "fragment_shape", choices = ["16x16x32"]}
  micro.param "tail_policy" {kind = "tail_policy", choices = ["mask"]}
  micro.constraint "sram_capacity" {params = ["BM", "BN", "BK"]}
  micro.objective {direction = "minimize", metric = "latency_cycles", secondary = ["matrix_utilization", "dram_bytes"]}
}
)MLIR";

TEST(CandidateBinding, BindsTheFirstCandidateOfARealSearchSpace) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::micro::MicroDialect>();
  MLIRContext context(registry);

  OwningOpRef<ModuleOp> input = mlir::parseSourceString<ModuleOp>(
      kExportedSpace, mlir::ParserConfig(&context));
  ASSERT_TRUE(bool(input));
  ASSERT_TRUE(mlir::succeeded(mlir::verify(*input)));

  auto space = loadSearchSpace(input.get());
  ASSERT_TRUE(bool(space)) << llvm::toString(space.takeError());

  std::vector<Candidate> candidates = generateGridCandidates(*space);
  ASSERT_FALSE(candidates.empty());

  OwningOpRef<ModuleOp> bound = ModuleOp::create(UnknownLoc::get(&context));
  auto result = bindCandidateToMicroKernel(bound.get(), *space,
                                           candidates.front(), swigluShape());
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());

  EXPECT_EQ(countOps<micro::KernelOp>(bound.get()), 1u);
  EXPECT_EQ(countOps<micro::SearchSpaceOp>(bound.get()), 0u);
  EXPECT_EQ(result->decisions.workerTile, (std::vector<int64_t>{1, 16, 32}));
}

} // namespace
} // namespace mlir::llk::perf
