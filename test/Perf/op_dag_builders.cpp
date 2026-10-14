#include "LLK/Perf/TileInfo.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

namespace mlir::llk::perf {
TEST(OpDAGBuilders, PublicTileDescriptionRetainsTensorSize) {
  mlir::MLIRContext context;
  auto type =
      mlir::RankedTensorType::get({8, 16}, mlir::Float32Type::get(&context));
  TileInfo info = describeType(type);
  EXPECT_FALSE(info.isTile);
  EXPECT_EQ(info.dtype, "f32");
  EXPECT_EQ(info.elements(), 128u);
  EXPECT_EQ(info.bytes(), 512u);
  EXPECT_EQ(shapeString(info.shape), "8x16");
}
TEST(OpDAGBuilders, UnsizableTypesRetainZeroBytes) {
  mlir::MLIRContext context;
  auto dynamic = describeType(mlir::RankedTensorType::get(
      {mlir::ShapedType::kDynamic, 16}, mlir::Float32Type::get(&context)));
  EXPECT_TRUE(dynamic.hasDynamicShape());
  EXPECT_EQ(dynamic.bytes(), 0u);
  EXPECT_EQ(shapeString(dynamic.shape), "?x16");
  auto unsupported = describeType(
      mlir::RankedTensorType::get({8, 16}, mlir::Float64Type::get(&context)));
  EXPECT_EQ(unsupported.elements(), 128u);
  EXPECT_EQ(unsupported.bytes(), 0u);
}
} // namespace mlir::llk::perf

#include "LLK/Perf/OpDAGBuilder.h"
namespace mlir::llk::perf {
namespace {
auto noEvent = [](mlir::Operation &, OpDAGBuildContext &) {
  return llvm::Error::success();
};
void expectError(llvm::Error error, llvm::StringRef fragment) {
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error)).find(fragment.str()),
            std::string::npos);
}
} // namespace
TEST(OpDAGBuilders, DuplicateRegistrationAndAbsentReplacementFail) {
  OpDAGBuilderRegistry builders;
  ASSERT_FALSE(builders.registerBuilder("test.probe", noEvent));
  expectError(builders.registerBuilder("test.probe", noEvent), "test.probe");
  expectError(builders.replaceBuilder("test.absent", noEvent), "test.absent");
  EXPECT_NE(builders.lookup("test.probe"), nullptr);
  EXPECT_EQ(builders.lookup("test.absent"), nullptr);
}
TEST(OpDAGBuilders, RejectsEmptyNamesCallbacksAndStructuralOperations) {
  OpDAGBuilderRegistry builders;
  expectError(builders.registerBuilder("", noEvent), "empty");
  expectError(builders.registerBuilder("test.empty", {}), "callback");
  ASSERT_FALSE(builders.registerBuilder("test.probe", noEvent));
  expectError(builders.replaceBuilder("test.probe", {}), "callback");
  for (llvm::StringRef name :
       {"micro.for", "micro.spatial_for", "micro.pipeline", "micro.yield"}) {
    expectError(builders.registerBuilder(name, noEvent), "structural");
    expectError(builders.replaceBuilder(name, noEvent), "structural");
  }
}
} // namespace mlir::llk::perf

#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Machine/MachineModelLoader.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
namespace mlir::llk::perf {
namespace {
struct Parsed {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::Operation *kernel = nullptr;
};
std::unique_ptr<Parsed> parseKernel(llvm::StringRef source) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::micro::MicroDialect>();
  auto parsed = std::make_unique<Parsed>();
  parsed->context = std::make_unique<mlir::MLIRContext>(registry);
  parsed->module =
      mlir::parseSourceString<mlir::ModuleOp>(source, parsed->context.get());
  if (!parsed->module || failed(mlir::verify(*parsed->module)))
    return nullptr;
  auto kernel = findMicroKernel(parsed->module.get(), "");
  if (!kernel) {
    ADD_FAILURE() << llvm::toString(kernel.takeError());
    return nullptr;
  }
  parsed->kernel = *kernel;
  return parsed;
}
constexpr llvm::StringLiteral probe = R"mlir(
module { micro.kernel @probe {
  %c = arith.constant 0 : index
  micro.yield
} }
)mlir";
auto eventBuilder(uint64_t cycles, std::string name = "probe") {
  return [cycles, name = std::move(name)](mlir::arith::ConstantOp op,
                                          OpDAGBuildContext &context) {
    MicroEvent event;
    event.resourceName = name;
    event.sourceOpName = op->getName().getStringRef().str();
    event.minCycles = cycles;
    context.emitEvent(std::move(event), {}, op->getResults());
    return llvm::Error::success();
  };
}
} // namespace
TEST(OpDAGBuilders, InjectsTypedBuilderWithOwnedConfiguration) {
  auto parsed = parseKernel(probe);
  ASSERT_NE(parsed, nullptr);
  machine::MachineModel model;
  model.target = "probe";
  auto builders = makeDefaultOpDAGBuilders();
  {
    auto name = std::string("owned");
    ASSERT_FALSE(builders.replaceBuilder<mlir::arith::ConstantOp>(
        eventBuilder(7, name)));
  }
  auto dag = buildMicroDAG(parsed->kernel, model, builders);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  ASSERT_EQ(dag->events.size(), 1u);
  EXPECT_EQ(dag->events[0].id, 0u);
  EXPECT_EQ(dag->events[0].minCycles, 7u);
  EXPECT_EQ(dag->events[0].resourceName, "owned");
}
TEST(OpDAGBuilders, RegistriesRemainIndependentAndDefaultFallbackIsFree) {
  auto parsed = parseKernel(probe);
  ASSERT_NE(parsed, nullptr);
  machine::MachineModel model;
  auto first = makeDefaultOpDAGBuilders();
  auto second = makeDefaultOpDAGBuilders();
  ASSERT_FALSE(first.replaceBuilder<mlir::arith::ConstantOp>(eventBuilder(7)));
  ASSERT_FALSE(
      second.replaceBuilder<mlir::arith::ConstantOp>(eventBuilder(11)));
  auto a = buildMicroDAG(parsed->kernel, model, first);
  auto b = buildMicroDAG(parsed->kernel, model, second);
  auto fallback = buildMicroDAG(parsed->kernel, model);
  ASSERT_TRUE(static_cast<bool>(a));
  ASSERT_TRUE(static_cast<bool>(b));
  ASSERT_TRUE(static_cast<bool>(fallback));
  ASSERT_EQ(a->events.size(), 1u);
  ASSERT_EQ(b->events.size(), 1u);
  EXPECT_EQ(a->events[0].minCycles, 7u);
  EXPECT_EQ(b->events[0].minCycles, 11u);
  EXPECT_TRUE(fallback->events.empty());
}
TEST(OpDAGBuilders, HandlerErrorsPropagateAndDoNotPoisonLaterBuilds) {
  auto parsed = parseKernel(probe);
  ASSERT_NE(parsed, nullptr);
  machine::MachineModel model;
  auto builders = makeDefaultOpDAGBuilders();
  ASSERT_FALSE(builders.replaceBuilder<mlir::arith::ConstantOp>(
      [](mlir::arith::ConstantOp op,
         OpDAGBuildContext &context) -> llvm::Error {
        context.emitEvent(MicroEvent{}, {}, op->getResults());
        return llvm::make_error<llvm::StringError>(
            "probe failed", llvm::inconvertibleErrorCode());
      }));
  auto failed = buildMicroDAG(parsed->kernel, model, builders);
  ASSERT_FALSE(static_cast<bool>(failed));
  EXPECT_EQ(llvm::toString(failed.takeError()), "probe failed");
  ASSERT_FALSE(
      builders.replaceBuilder<mlir::arith::ConstantOp>(eventBuilder(3)));
  auto recovered = buildMicroDAG(parsed->kernel, model, builders);
  ASSERT_TRUE(static_cast<bool>(recovered));
  ASSERT_EQ(recovered->events.size(), 1u);
  EXPECT_EQ(recovered->events[0].id, 0u);
}
} // namespace mlir::llk::perf

// Generated operation classes are confined to these integration tests.
#include "LLK/Dialect/Micro/MicroEnums.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/OpImplementation.h"
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"
namespace mlir::llk::perf {
namespace {
machine::MachineModel loadModel(llvm::StringRef file = "x86-avx2-v2.yaml") {
  auto model = machine::loadMachineModel(std::string(LLK_MACHINE_DIR) + "/" +
                                         file.str());
  if (!model) {
    ADD_FAILURE() << llvm::toString(model.takeError());
    return {};
  }
  return *model;
}
constexpr llvm::StringLiteral vectorKernel = R"mlir(
module { micro.kernel @vector {
  %ext = tensor.empty() : tensor<8x8xf32>
  %a, %t = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
  %v = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  %out = micro.vector "silu" %v : !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  micro.wait %t
  micro.yield
} }
)mlir";
void expectEqual(const MicroDAG &a, const MicroDAG &b) {
  ASSERT_EQ(a.events.size(), b.events.size());
  for (size_t i = 0; i < a.events.size(); ++i) {
    const auto &x = a.events[i];
    const auto &y = b.events[i];
    EXPECT_EQ(x.id, y.id);
    EXPECT_EQ(x.kind, y.kind);
    EXPECT_EQ(x.resource, y.resource);
    EXPECT_EQ(x.resourceName, y.resourceName);
    EXPECT_EQ(x.workItems, y.workItems);
    EXPECT_EQ(x.bytes, y.bytes);
    EXPECT_EQ(x.minCycles, y.minCycles);
    EXPECT_EQ(x.deps, y.deps);
    EXPECT_EQ(x.sourceOpName, y.sourceOpName);
    EXPECT_EQ(x.costKind, y.costKind);
    EXPECT_EQ(x.tileShape, y.tileShape);
    EXPECT_EQ(x.tileLayout, y.tileLayout);
    EXPECT_EQ(x.tileMemory, y.tileMemory);
    EXPECT_EQ(x.srcMemory, y.srcMemory);
    EXPECT_EQ(x.tileOwner, y.tileOwner);
    EXPECT_EQ(x.ownerPool, y.ownerPool);
  }
  EXPECT_EQ(a.liveTileBytesByMemory, b.liveTileBytesByMemory);
  EXPECT_EQ(a.warnings, b.warnings);
  EXPECT_EQ(a.layoutWarnings, b.layoutWarnings);
  EXPECT_EQ(a.ownerWarnings, b.ownerWarnings);
  EXPECT_EQ(a.incompleteReasons, b.incompleteReasons);
}
} // namespace
TEST(OpDAGBuilders, ReplacementKeepsUnrelatedBuiltinsAndAliasDependencies) {
  auto parsed = parseKernel(vectorKernel);
  ASSERT_NE(parsed, nullptr);
  auto model = loadModel();
  auto builders = makeDefaultOpDAGBuilders();
  ASSERT_FALSE(builders.replaceBuilder<micro::VectorOp>(
      [](micro::VectorOp op, OpDAGBuildContext &context) {
        MicroEvent event;
        event.resourceName = "replacement";
        event.minCycles = 17;
        context.emitEvent(std::move(event), op.getInputs(), op->getResults(),
                          {0, 0});
        return llvm::Error::success();
      }));
  auto dag = buildMicroDAG(parsed->kernel, model, builders);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  ASSERT_EQ(dag->events.size(), 3u);
  EXPECT_EQ(dag->events[0].kind, EventKind::AsyncCopy);
  EXPECT_EQ(dag->events[1].minCycles, 17u);
  EXPECT_EQ(dag->events[1].deps, (std::vector<uint32_t>{0}));
  EXPECT_EQ(dag->events[2].kind, EventKind::Wait);
  EXPECT_EQ(dag->events[2].deps, (std::vector<uint32_t>{0, 1}));
  EXPECT_EQ(dag->liveTileBytesByMemory.at("sram"), 256u);
}
TEST(OpDAGBuilders, SuppliedRegistryDoesNotMergeBuiltins) {
  auto parsed = parseKernel(vectorKernel);
  ASSERT_NE(parsed, nullptr);
  OpDAGBuilderRegistry builders;
  auto dag = buildMicroDAG(parsed->kernel, loadModel(), builders);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  EXPECT_TRUE(dag->events.empty());
  EXPECT_TRUE(dag->liveTileBytesByMemory.empty());
}
TEST(OpDAGBuilders, TypedAdapterReturnsMismatchError) {
  auto parsed = parseKernel(probe);
  ASSERT_NE(parsed, nullptr);
  OpDAGBuilderRegistry typed;
  ASSERT_FALSE(typed.registerBuilder<mlir::arith::AddIOp>(
      [](mlir::arith::AddIOp, OpDAGBuildContext &) {
        return llvm::Error::success();
      }));
  OpDAGBuilderRegistry builders;
  ASSERT_FALSE(
      builders.registerBuilder("arith.constant", *typed.lookup("arith.addi")));
  auto dag = buildMicroDAG(parsed->kernel, machine::MachineModel{}, builders);
  ASSERT_FALSE(static_cast<bool>(dag));
  expectError(dag.takeError(), "type mismatch");
}
TEST(OpDAGBuilders, DefaultsRemainEquivalentOnBothProfiles) {
  auto parsed = parseKernel(vectorKernel);
  ASSERT_NE(parsed, nullptr);
  const auto builders = makeDefaultOpDAGBuilders();
  for (llvm::StringRef file :
       {"x86-avx2-v2.yaml", "generic-ai-accel-v2.yaml"}) {
    auto model = loadModel(file);
    auto implicit = buildMicroDAG(parsed->kernel, model);
    auto explicitDag = buildMicroDAG(parsed->kernel, model, builders);
    ASSERT_TRUE(static_cast<bool>(implicit))
        << llvm::toString(implicit.takeError());
    ASSERT_TRUE(static_cast<bool>(explicitDag))
        << llvm::toString(explicitDag.takeError());
    expectEqual(*implicit, *explicitDag);
    ASSERT_EQ(explicitDag->events.size(), 3u);
    EXPECT_EQ(explicitDag->events[1].workItems, 64u);
    EXPECT_EQ(explicitDag->events[1].deps, (std::vector<uint32_t>{0}));
  }
}
} // namespace mlir::llk::perf

namespace mlir::llk::perf {
TEST(OpDAGBuilders, PipelineUnrollingCountsAllocationSitesOnce) {
  auto parsed = parseKernel(R"mlir(
module { micro.kernel @pipeline {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  micro.for %i = %c0 to %c4 step %c1 {
    micro.pipeline stages = 2 {
      %buf = micro.tile_alloc : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
      %ext = tensor.empty() : tensor<8x8xf32>
      %a, %t = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
      micro.wait %t
      %v = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
      %out = micro.vector "silu" %v : !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    }
  }
  micro.yield
} }
)mlir");
  ASSERT_NE(parsed, nullptr);
  auto model = loadModel();
  const auto builders = makeDefaultOpDAGBuilders();
  auto a = buildMicroDAG(parsed->kernel, model);
  auto b = buildMicroDAG(parsed->kernel, model, builders);
  ASSERT_TRUE(static_cast<bool>(a)) << llvm::toString(a.takeError());
  ASSERT_TRUE(static_cast<bool>(b)) << llvm::toString(b.takeError());
  expectEqual(*a, *b);
  ASSERT_EQ(b->events.size(), 12u);
  EXPECT_EQ(b->liveTileBytesByMemory.at("sram"), 1024u);
  for (uint32_t i = 0; i < 4; ++i) {
    EXPECT_EQ(b->events[3 * i + 1].deps, (std::vector<uint32_t>{3 * i}));
    EXPECT_EQ(b->events[3 * i + 2].deps,
              (std::vector<uint32_t>{3 * i, 3 * i + 1}));
  }
}
TEST(OpDAGBuilders, MultiHopResultsAndTokensBindToFinalHop) {
  auto parsed = parseKernel(R"mlir(
module { micro.kernel @route attributes {micro.routes = [{value = 0 : i64, kind = "transfer", route = ["dram.0", "l2.0", "sram.0"]}]} {
  %ext = tensor.empty() : tensor<8x8xf32>
  %a, %t = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>, micro.value = 0 : i64} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
  %v = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  %out = micro.vector "silu" %v : !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  micro.wait %t
  micro.yield
} }
)mlir");
  ASSERT_NE(parsed, nullptr);
  const auto builders = makeDefaultOpDAGBuilders();
  auto dag = buildMicroDAG(parsed->kernel, loadModel(), builders);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  ASSERT_EQ(dag->events.size(), 4u);
  EXPECT_EQ(dag->events[0].tileMemory, "l2");
  EXPECT_EQ(dag->events[1].tileMemory, "sram");
  EXPECT_EQ(dag->events[1].deps, (std::vector<uint32_t>{0}));
  EXPECT_EQ(dag->events[2].deps, (std::vector<uint32_t>{1}));
  EXPECT_EQ(dag->events[3].deps, (std::vector<uint32_t>{1, 2}));
}
} // namespace mlir::llk::perf

namespace mlir::llk::perf {
TEST(OpDAGBuilders, RegistrationOrderDoesNotChangeTraversalOrDiagnostics) {
  auto parsed = parseKernel(R"mlir(
module { micro.kernel @order {
  %c = arith.constant 0 : index
  %sum = arith.addi %c, %c : index
  micro.yield
} }
)mlir");
  ASSERT_NE(parsed, nullptr);
  auto constant = eventBuilder(7);
  auto addition = [](mlir::arith::AddIOp op, OpDAGBuildContext &context) {
    MicroEvent event;
    event.minCycles = 11;
    event.resourceName = "sum";
    context.emitEvent(std::move(event), op->getOperands(), op->getResults());
    context.noteWarning("ordered warning");
    context.noteWarning("ordered warning");
    return llvm::Error::success();
  };
  OpDAGBuilderRegistry first, second;
  ASSERT_FALSE(first.registerBuilder<mlir::arith::ConstantOp>(constant));
  ASSERT_FALSE(first.registerBuilder<mlir::arith::AddIOp>(addition));
  ASSERT_FALSE(second.registerBuilder<mlir::arith::AddIOp>(addition));
  ASSERT_FALSE(second.registerBuilder<mlir::arith::ConstantOp>(constant));
  const machine::MachineModel model;
  auto a = buildMicroDAG(parsed->kernel, model, first);
  auto b = buildMicroDAG(parsed->kernel, model, second);
  auto repeated = buildMicroDAG(parsed->kernel, model, first);
  ASSERT_TRUE(static_cast<bool>(a));
  ASSERT_TRUE(static_cast<bool>(b));
  ASSERT_TRUE(static_cast<bool>(repeated));
  expectEqual(*a, *b);
  expectEqual(*a, *repeated);
  ASSERT_EQ(a->events.size(), 2u);
  EXPECT_EQ(a->events[0].minCycles, 7u);
  EXPECT_EQ(a->events[1].minCycles, 11u);
  EXPECT_EQ(a->events[1].deps, (std::vector<uint32_t>{0}));
  EXPECT_EQ(a->warnings, (std::vector<std::string>{"ordered warning"}));
}
} // namespace mlir::llk::perf

namespace mlir::llk::perf {
TEST(OpDAGBuilders, ContextAliasesAndAccountsStorageUnderUnrolling) {
  auto parsed = parseKernel(R"mlir(
module { micro.kernel @context {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  micro.for %i = %c0 to %c2 step %c1 {
    micro.pipeline stages = 3 {
      %c = arith.constant 5 : index
      %sum = arith.addi %c, %c : index
      %out = arith.muli %sum, %sum : index
    }
  }
  micro.yield
} }
)mlir");
  ASSERT_NE(parsed, nullptr);
  OpDAGBuilderRegistry builders;
  ASSERT_FALSE(builders.registerBuilder<mlir::arith::ConstantOp>(
      [](mlir::arith::ConstantOp op, OpDAGBuildContext &context) {
        if (context.operation().getParentOp()->getName().getStringRef() ==
            "micro.pipeline") {
          context.emitEvent(MicroEvent{}, {}, op->getResults());
          context.accountStorage("sram", 64);
        }
        return llvm::Error::success();
      }));
  ASSERT_FALSE(builders.registerZeroCost(
      "arith.addi", [](mlir::Operation &op, OpDAGBuildContext &context) {
        context.aliasValue(op.getResult(0), op.getOperand(0));
        context.accountStorage("sram", 8);
        context.noteWarning("custom_alias_limitation");
        context.noteIncomplete("custom_alias_limitation");
        return llvm::Error::success();
      }));
  ASSERT_FALSE(builders.registerBuilder<mlir::arith::MulIOp>(
      [](mlir::arith::MulIOp op, OpDAGBuildContext &context) {
        context.emitEvent(MicroEvent{}, op->getOperands(), op->getResults());
        return llvm::Error::success();
      }));
  auto dag = buildMicroDAG(parsed->kernel, loadModel(), builders);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  ASSERT_EQ(dag->events.size(), 4u);
  EXPECT_EQ(dag->events[1].deps, (std::vector<uint32_t>{0}));
  EXPECT_EQ(dag->events[3].deps, (std::vector<uint32_t>{2}));
  EXPECT_EQ(dag->liveTileBytesByMemory.at("sram"), 216u);
  EXPECT_EQ(builders.classification("arith.addi"), DAGLeafKind::ZeroCost);
  EXPECT_EQ(dag->incompleteReasons,
            (std::vector<std::string>{"custom_alias_limitation"}));
}
} // namespace mlir::llk::perf

namespace mlir::llk::perf {
TEST(OpDAGBuilders, TypedRegistrationRejectsEmptyCallbacksBeforeDispatch) {
  using TypedCallback =
      std::function<llvm::Error(mlir::arith::ConstantOp, OpDAGBuildContext &)>;
  OpDAGBuilderRegistry builders;
  expectError(
      builders.registerBuilder<mlir::arith::ConstantOp>(TypedCallback{}),
      "callback");
  EXPECT_EQ(builders.lookup("arith.constant"), nullptr);
  ASSERT_FALSE(
      builders.registerBuilder<mlir::arith::ConstantOp>(eventBuilder(7)));
  expectError(builders.replaceBuilder<mlir::arith::ConstantOp>(TypedCallback{}),
              "callback");
  auto parsed = parseKernel(probe);
  ASSERT_NE(parsed, nullptr);
  auto dag = buildMicroDAG(parsed->kernel, machine::MachineModel{}, builders);
  ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
  ASSERT_EQ(dag->events.size(), 1u);
  EXPECT_EQ(dag->events[0].minCycles, 7u);
}
} // namespace mlir::llk::perf

namespace mlir::llk::perf {
namespace {
constexpr const char *kInjectedGemmKernel = R"MLIR(
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

} // namespace
TEST(OpDAGBuilders, DefaultMmaExtractionMatchesInjectedDefaults) {
  auto parsed = parseKernel(kInjectedGemmKernel);
  ASSERT_NE(parsed, nullptr);
  const auto builders = makeDefaultOpDAGBuilders();
  for (llvm::StringRef file :
       {"x86-avx2-v2.yaml", "generic-ai-accel-v2.yaml"}) {
    auto model = loadModel(file);
    auto implicit = buildMicroDAG(parsed->kernel, model);
    auto injected = buildMicroDAG(parsed->kernel, model, builders);
    ASSERT_TRUE(static_cast<bool>(implicit))
        << llvm::toString(implicit.takeError());
    ASSERT_TRUE(static_cast<bool>(injected))
        << llvm::toString(injected.takeError());
    expectEqual(*implicit, *injected);
    ASSERT_EQ(injected->events.size(), 5u);
    EXPECT_EQ(injected->events[3].kind, EventKind::Mma);
    EXPECT_EQ(injected->events[3].workItems, 8192u);
    EXPECT_EQ(injected->events[3].deps, (std::vector<uint32_t>{0, 1, 2}));
    EXPECT_EQ(injected->events[4].kind, EventKind::Store);
    EXPECT_EQ(injected->events[4].deps, (std::vector<uint32_t>{3}));
  }
}
} // namespace mlir::llk::perf

namespace mlir::llk::perf {
TEST(OpDAGBuilders, DeclaredCoverageAuditStrictAndLegacy) {
  auto parsed = parseKernel(R"mlir(module { micro.kernel @coverage {
    %a = arith.constant 1.0 : f32
    %b = arith.addf %a, %a : f32
    %c = arith.addf %b, %a : f32
    micro.yield
  } })mlir");
  ASSERT_NE(parsed, nullptr);
  OpDAGBuilderRegistry builders;
  ASSERT_FALSE(builders.registerZeroCost("arith.constant", noEvent));
  EXPECT_EQ(builders.classification("arith.constant"), DAGLeafKind::ZeroCost);
  ASSERT_FALSE(builders.replaceBuilder("arith.constant", noEvent));
  EXPECT_EQ(builders.classification("arith.constant"), DAGLeafKind::ZeroCost);
  EXPECT_FALSE(builders.classification("arith.addf"));
  auto audit = buildMicroDAG(parsed->kernel, machine::MachineModel{}, builders,
                             DAGBuildOptions{});
  ASSERT_TRUE(bool(audit)) << llvm::toString(audit.takeError());
  EXPECT_EQ(audit->incompleteReasons,
            (std::vector<std::string>{"unmodeled_leaf: arith.addf"}));
  auto strict = buildMicroDAG(parsed->kernel, machine::MachineModel{}, builders,
                              DAGBuildOptions{CoverageMode::Strict});
  ASSERT_FALSE(bool(strict));
  EXPECT_NE(llvm::toString(strict.takeError()).find("arith.addf"),
            std::string::npos);
  auto legacy =
      buildMicroDAG(parsed->kernel, machine::MachineModel{}, builders);
  ASSERT_TRUE(bool(legacy));
  EXPECT_TRUE(legacy->incompleteReasons.empty());
  ASSERT_FALSE(builders.registerBuilder("arith.addf", noEvent));
  EXPECT_EQ(builders.classification("arith.addf"), DAGLeafKind::Modeled);
  auto complete =
      buildMicroDAG(parsed->kernel, machine::MachineModel{}, builders,
                    DAGBuildOptions{CoverageMode::Strict});
  ASSERT_TRUE(bool(complete)) << llvm::toString(complete.takeError());
  EXPECT_TRUE(complete->events.empty());
  ASSERT_FALSE(builders.replaceBuilder(
      "arith.addf", [](mlir::Operation &, OpDAGBuildContext &context) {
        context.noteWarning("callback_partial");
        context.noteIncomplete("callback_partial");
        return llvm::Error::success();
      }));
  auto partial = buildMicroDAG(parsed->kernel, machine::MachineModel{},
                               builders, DAGBuildOptions{});
  ASSERT_TRUE(bool(partial));
  EXPECT_EQ(partial->incompleteReasons,
            (std::vector<std::string>{"callback_partial"}));
  auto refused = buildMicroDAG(parsed->kernel, machine::MachineModel{},
                               builders, DAGBuildOptions{CoverageMode::Strict});
  ASSERT_FALSE(bool(refused));
  EXPECT_NE(llvm::toString(refused.takeError()).find("callback_partial"),
            std::string::npos);
}
} // namespace mlir::llk::perf

namespace mlir::llk::perf {
TEST(OpDAGBuilders, AuditAcceptsIndexScaffoldingWithoutEvents) {
  auto parsed = parseKernel(R"mlir(module { micro.kernel @indices {
    %a = arith.constant 1 : index
    %b = arith.addi %a, %a : index
    %c = arith.muli %b, %a : index
    micro.yield
  } })mlir");
  ASSERT_NE(parsed, nullptr);
  auto builders = makeDefaultOpDAGBuilders();
  auto dag = buildMicroDAG(parsed->kernel, machine::MachineModel{}, builders,
                           DAGBuildOptions{});
  ASSERT_TRUE(bool(dag)) << llvm::toString(dag.takeError());
  EXPECT_TRUE(dag->events.empty());
  EXPECT_TRUE(dag->incompleteReasons.empty());
  for (auto name : {"arith.constant", "arith.addi", "arith.muli"})
    EXPECT_EQ(builders.classification(name), DAGLeafKind::ZeroCost);
}
TEST(OpDAGBuilders, DataArithmeticIsNotControlScaffolding) {
  auto parsed = parseKernel(R"mlir(module { micro.kernel @data {
    %a = arith.constant 1 : i32
    %b = arith.addi %a, %a : i32
    %c = arith.muli %b, %a : i32
    %f = arith.constant 1.0 : f32
    %g = arith.addf %f, %f : f32
    micro.yield
  } })mlir");
  ASSERT_NE(parsed, nullptr);
  auto dag = buildMicroDAG(parsed->kernel, machine::MachineModel{},
                           makeDefaultOpDAGBuilders(), DAGBuildOptions{});
  ASSERT_TRUE(bool(dag)) << llvm::toString(dag.takeError());
  EXPECT_EQ(dag->incompleteReasons,
            (std::vector<std::string>{"unmodeled_leaf: arith.addi",
                                      "unmodeled_leaf: arith.muli",
                                      "unmodeled_leaf: arith.addf"}));
}
TEST(OpDAGBuilders, VectorIndexArithmeticIsNotControlScaffolding) {
  auto parsed = parseKernel(R"mlir(module { micro.kernel @vector_indices {
    %v = arith.constant dense<1> : vector<2xindex>
    %w = arith.addi %v, %v : vector<2xindex>
    micro.yield
  } })mlir");
  ASSERT_NE(parsed, nullptr);
  auto dag = buildMicroDAG(parsed->kernel, machine::MachineModel{},
                           makeDefaultOpDAGBuilders(), DAGBuildOptions{});
  ASSERT_TRUE(bool(dag)) << llvm::toString(dag.takeError());
  EXPECT_EQ(dag->incompleteReasons,
            (std::vector<std::string>{"unmodeled_leaf: arith.addi"}));
}
TEST(OpDAGBuilders, EmptyTensorMetadataDoesNotAllocateTileStorage) {
  auto parsed = parseKernel(R"mlir(module { micro.kernel @metadata {
    %empty = tensor.empty() : tensor<8x8xf32>
    %allocation = micro.tile_alloc : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  } })mlir");
  ASSERT_NE(parsed, nullptr);
  auto dag = buildMicroDAG(parsed->kernel, loadModel(),
                           makeDefaultOpDAGBuilders(), DAGBuildOptions{});
  ASSERT_TRUE(bool(dag)) << llvm::toString(dag.takeError());
  EXPECT_TRUE(dag->events.empty());
  EXPECT_TRUE(dag->incompleteReasons.empty());
  EXPECT_EQ(dag->liveTileBytesByMemory,
            (std::map<std::string, uint64_t>{{"sram", 256}}));
}
TEST(OpDAGBuilders, TensorElementComputationRemainsUnsupported) {
  auto parsed = parseKernel(R"mlir(module { micro.kernel @elements {
    %empty = tensor.empty() : tensor<8xf32>
    %i = arith.constant 0 : index
    %x = tensor.extract %empty[%i] : tensor<8xf32>
    %y = tensor.insert %x into %empty[%i] : tensor<8xf32>
    micro.yield
  } })mlir");
  ASSERT_NE(parsed, nullptr);
  auto dag = buildMicroDAG(parsed->kernel, machine::MachineModel{},
                           makeDefaultOpDAGBuilders(), DAGBuildOptions{});
  ASSERT_TRUE(bool(dag)) << llvm::toString(dag.takeError());
  EXPECT_EQ(dag->incompleteReasons,
            (std::vector<std::string>{"unmodeled_leaf: tensor.extract",
                                      "unmodeled_leaf: tensor.insert"}));
}
TEST(OpDAGBuilders, AuditedLogicalViewsKeepProducerDependencies) {
  auto parsed = parseKernel(R"mlir(module { micro.kernel @logical {
    %ext = tensor.empty() : tensor<8x8xf32>
    %a, %token = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    %v = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %part = micro.tile_partition %v {shape = array<i64: 8, 8>} : !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %out = micro.vector "silu" %part : !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.wait %token
    micro.yield
  } })mlir");
  ASSERT_NE(parsed, nullptr);
  auto dag = buildMicroDAG(parsed->kernel, loadModel(),
                           makeDefaultOpDAGBuilders(), DAGBuildOptions{});
  ASSERT_TRUE(bool(dag)) << llvm::toString(dag.takeError());
  EXPECT_TRUE(dag->incompleteReasons.empty());
  ASSERT_EQ(dag->events.size(), 3u);
  EXPECT_EQ(dag->events[1].deps, (std::vector<uint32_t>{0}));
  EXPECT_EQ(dag->liveTileBytesByMemory.at("sram"), 256u);
}
TEST(OpDAGBuilders, AuditedPhysicalPartitionChargesWorkTrafficAndStorage) {
  auto parsed = parseKernel(R"mlir(module { micro.kernel @physical {
    %buf = micro.tile_alloc : !micro.tile<32x32xbf16, memory = #micro.memory<sram>, layout = #micro.layout<row_major>>
    %t = micro.vector "silu" %buf : !micro.tile<32x32xbf16, memory = #micro.memory<sram>, layout = #micro.layout<row_major>> -> !micro.tile<32x32xbf16, memory = #micro.memory<sram>, layout = #micro.layout<row_major>>
    %frag = micro.tile_partition %t {shape = array<i64: 16, 16>} : !micro.tile<32x32xbf16, memory = #micro.memory<sram>, layout = #micro.layout<row_major>> -> !micro.tile<16x16xbf16, memory = #micro.memory<sram>, layout = #micro.layout<vectorized, vector = 8>>
    micro.yield
  } })mlir");
  ASSERT_NE(parsed, nullptr);
  auto dag = buildMicroDAG(parsed->kernel, loadModel(),
                           makeDefaultOpDAGBuilders(), DAGBuildOptions{});
  ASSERT_TRUE(bool(dag)) << llvm::toString(dag.takeError());
  EXPECT_TRUE(dag->incompleteReasons.empty());
  ASSERT_EQ(dag->events.size(), 2u);
  EXPECT_EQ(dag->events[1].kind, EventKind::TilePartition);
  EXPECT_GT(dag->events[1].minCycles, 0u);
  EXPECT_GT(dag->events[1].bytes, 0u);
  EXPECT_GT(dag->events[1].workItems, 0u);
  EXPECT_EQ(dag->events[1].deps, (std::vector<uint32_t>{0}));
  EXPECT_GT(dag->liveTileBytesByMemory.at("sram"), 2048u);
}
} // namespace mlir::llk::perf

#include "LLK/Perf/SelectedKernelAnalysis.h"
namespace mlir::llk::perf {
TEST(OpDAGBuilders, UnknownCoverageRefusesStrictAnalysis) {
  auto parsed = parseKernel(R"mlir(
module { micro.kernel @unknown {
  %a = arith.constant 1.0 : f32
  %b = arith.addf %a, %a : f32
  micro.yield
} })mlir");
  ASSERT_NE(parsed, nullptr);
  auto partial =
      analyzeSelectedKernel(parsed->kernel, machine::MachineModel{}, false);
  ASSERT_TRUE(bool(partial)) << llvm::toString(partial.takeError());
  EXPECT_FALSE(partial->complete);
  ASSERT_EQ(partial->incompleteReasons.size(), 1u);
  EXPECT_EQ(partial->incompleteReasons[0], "unmodeled_leaf: arith.addf");
  auto strict =
      analyzeSelectedKernel(parsed->kernel, machine::MachineModel{}, true);
  ASSERT_FALSE(bool(strict));
  EXPECT_NE(llvm::toString(strict.takeError()).find("arith.addf"),
            std::string::npos);
}
} // namespace mlir::llk::perf

namespace mlir::llk::perf {
TEST(OpDAGBuilders, RepeatedUnknownCoverageHasOneReason) {
  auto parsed = parseKernel(R"mlir(
module { micro.kernel @loop {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  micro.for %i = %c0 to %c2 step %c1 {
    %a = arith.constant 1.0 : f32
    %b = arith.addf %a, %a : f32
  }
  micro.yield
} })mlir");
  ASSERT_NE(parsed, nullptr);
  auto partial =
      analyzeSelectedKernel(parsed->kernel, machine::MachineModel{}, false);
  ASSERT_TRUE(bool(partial)) << llvm::toString(partial.takeError());
  EXPECT_FALSE(partial->complete);
  EXPECT_TRUE(partial->events.events.empty());
  EXPECT_EQ(partial->incompleteReasons,
            (std::vector<std::string>{"unmodeled_leaf: arith.addf"}));
}
TEST(OpDAGBuilders, AuditedAliasOnlyHandlerRemainsComplete) {
  auto parsed = parseKernel(R"mlir(
module { micro.kernel @alias {
  %a = arith.constant 1.0 : f32
  %b = arith.addf %a, %a : f32
  micro.yield
} })mlir");
  ASSERT_NE(parsed, nullptr);
  auto builders = makeDefaultOpDAGBuilders();
  ASSERT_FALSE(builders.registerZeroCost(
      "arith.addf", [](Operation &op, OpDAGBuildContext &context) {
        context.aliasValue(op.getResult(0), op.getOperand(0));
        return llvm::Error::success();
      }));
  auto dag = buildMicroDAG(parsed->kernel, machine::MachineModel{}, builders,
                           DAGBuildOptions{CoverageMode::Audit});
  ASSERT_TRUE(bool(dag)) << llvm::toString(dag.takeError());
  auto analysis = analyzeSelectedDag(*dag, machine::MachineModel{}, true);
  ASSERT_TRUE(bool(analysis)) << llvm::toString(analysis.takeError());
  EXPECT_TRUE(analysis->complete);
  EXPECT_TRUE(analysis->events.events.empty());
}
} // namespace mlir::llk::perf
