#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Mapping/MappingTarget.h"
#include "selected_group_lowering_test_hooks.h"

#include "gtest/gtest.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace {
namespace mapping = mlir::llk::mapping;

struct RecordedCall {
  mapping::InstanceId instance = 0;
  std::vector<std::string> operationNames;
  std::vector<uint64_t> nodes;
  size_t placements = 0;
  size_t connections = 0;
  std::string computeId;
};

using Calls = std::vector<RecordedCall>;

class RecordingEmitter final : public mapping::TargetEmitter {
public:
  explicit RecordingEmitter(Calls &calls) : calls_(calls) {}

  llvm::StringRef key() const override { return "record"; }
  llvm::Error verify(const mapping::TargetBundle &bundle) const override {
    if (bundle.emitterKey != key())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "wrong recording emitter key");
    return llvm::Error::success();
  }
  bool hasLowering() const override { return true; }
  llvm::Error lower(llvm::ArrayRef<mlir::Operation *> coveredOps,
                    const mapping::TargetBundle &,
                    const mapping::TargetLoweringContext &context,
                    mlir::RewriterBase &) const override {
    RecordedCall call;
    call.instance = context.instance;
    call.placements = context.placements.size();
    call.connections = context.connections.size();
    if (!context.placements.empty())
      call.computeId =
          context.placements.front().computeBindings.lookup("vector_engine");
    for (mlir::Operation *op : coveredOps) {
      call.operationNames.push_back(op->getName().getStringRef().str());
      auto mapping = op->getAttrOfType<mlir::DictionaryAttr>("micro.mapping");
      call.nodes.push_back(
          mapping.getAs<mlir::IntegerAttr>("node").getValue().getZExtValue());
    }
    calls_.push_back(std::move(call));
    return llvm::Error::success();
  }

private:
  Calls &calls_;
};

class RecordingTarget final : public mapping::MappingTarget {
public:
  explicit RecordingTarget(Calls &calls) : calls_(calls) {}
  llvm::StringRef name() const override { return "recording"; }
  const mlir::llk::machine::MachineModel &machine() const override {
    return machine_;
  }
  const mapping::LayoutRegistry &layouts() const override { return layouts_; }
  const mapping::RuleRegistry &rules() const override { return rules_; }
  bool isKnownEmitter(llvm::StringRef key) const override {
    return key == "record";
  }
  std::unique_ptr<mapping::TargetEmitter>
  createEmitter(llvm::StringRef key) const override {
    if (!isKnownEmitter(key))
      return nullptr;
    return std::make_unique<RecordingEmitter>(calls_);
  }

private:
  Calls &calls_;
  mlir::llk::machine::MachineModel machine_;
  mapping::LayoutRegistry layouts_;
  mapping::RuleRegistry rules_;
};

struct Fixture {
  explicit Fixture(mlir::MLIRContext &context)
      : module(mlir::ModuleOp::create(mlir::UnknownLoc::get(&context))),
        builder(&context) {
    context.allowUnregisteredDialects();
    auto type = mlir::Float32Type::get(&context);
    auto functionType = builder.getFunctionType({type, type}, {});
    builder.setInsertionPointToStart(module.getBody());
    auto function = mlir::func::FuncOp::create(builder, module.getLoc(),
                                               "kernel", functionType);
    block = function.addEntryBlock();
    builder.setInsertionPointToStart(block);
  }

  mlir::Operation *addVector(mlir::Value left, mlir::Value right, uint64_t node,
                             uint64_t instance,
                             llvm::StringRef bundleName = "vector_group") {
    mlir::OperationState state(builder.getUnknownLoc(), "micro.vector");
    state.addOperands({left, right});
    state.addTypes(builder.getF32Type());
    auto *op = builder.create(state);
    mlir::NamedAttrList mapping;
    mapping.set("node", builder.getI64IntegerAttr(node));
    mapping.set("instance", builder.getI64IntegerAttr(instance));
    mapping.set("bundle", builder.getStringAttr(bundleName));
    mapping.set("emitter", builder.getStringAttr("record"));
    mapping.set("bundle_parameters", builder.getDictionaryAttr({}));
    op->setAttr("micro.mapping", mapping.getDictionary(builder.getContext()));
    return op;
  }

  mapping::CoveringPlan
  planFor(llvm::ArrayRef<std::pair<uint64_t, uint64_t>> nodes) {
    mapping::CoveringPlan plan;
    for (auto [node, instance] : nodes) {
      mapping::PlanPlacement placement;
      placement.node = node;
      placement.instance = instance;
      placement.rule = "recording_rule";
      placement.bundle.name = "vector_group";
      placement.bundle.emitterKey = "record";
      placement.bundle.parameters = builder.getDictionaryAttr({});
      placement.computeBindings["vector_engine"] = "vpu.b";
      plan.placements.push_back(std::move(placement));
    }
    return plan;
  }

  mlir::ModuleOp module;
  mlir::OpBuilder builder;
  mlir::Block *block = nullptr;
};

TEST(SelectedGroupLowering, SendsOneTopologicalGroupWithSelectedContext) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  context.loadDialect<mlir::func::FuncDialect>();
  Fixture fixture(context);
  auto args = fixture.block->getArguments();
  auto *first = fixture.addVector(args[0], args[1], 10, 77);
  auto *second = fixture.addVector(first->getResult(0), args[1], 20, 77);
  fixture.addVector(second->getResult(0), args[0], 30, 77);

  mapping::CoveringPlan plan = fixture.planFor({{10, 77}, {20, 77}, {30, 77}});
  mapping::PlanConnection connection;
  connection.consumers.push_back(77);
  plan.connectionPlans.push_back(connection);

  Calls calls;
  RecordingTarget target(calls);
  llk::MappedCompilation compilation;
  ASSERT_FALSE(static_cast<bool>(llk::testing::lowerSelectedOperationsForTest(
      fixture.module, target, plan, compilation)));

  ASSERT_EQ(calls.size(), 1u);
  EXPECT_EQ(calls.front().operationNames,
            (std::vector<std::string>{"micro.vector", "micro.vector",
                                      "micro.vector"}));
  EXPECT_EQ(calls.front().nodes, (std::vector<uint64_t>{10, 20, 30}));
  EXPECT_EQ(calls.front().instance, 77u);
  EXPECT_EQ(calls.front().placements, 3u);
  EXPECT_EQ(calls.front().connections, 1u);
  EXPECT_EQ(calls.front().computeId, "vpu.b");
  EXPECT_EQ(compilation.selectedGroupsVerified, 1u);
  EXPECT_EQ(compilation.backendGroupsRealized, 0u);
  EXPECT_EQ(compilation.targetLowered, 3u);
}

TEST(SelectedGroupLowering, CallsIndependentInstancesSeparately) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  context.loadDialect<mlir::func::FuncDialect>();
  Fixture fixture(context);
  auto args = fixture.block->getArguments();
  fixture.addVector(args[0], args[1], 10, 80);
  fixture.addVector(args[0], args[1], 20, 90);
  mapping::CoveringPlan plan = fixture.planFor({{10, 80}, {20, 90}});
  Calls calls;
  RecordingTarget target(calls);
  llk::MappedCompilation compilation;

  ASSERT_FALSE(static_cast<bool>(llk::testing::lowerSelectedOperationsForTest(
      fixture.module, target, plan, compilation)));
  ASSERT_EQ(calls.size(), 2u);
  EXPECT_EQ(calls[0].instance, 80u);
  EXPECT_EQ(calls[1].instance, 90u);
  EXPECT_EQ(calls[0].placements, 1u);
  EXPECT_EQ(calls[1].placements, 1u);
  EXPECT_EQ(compilation.backendGroupsRealized, 0u);
}

TEST(SelectedGroupLowering, PreflightsEveryGroupBeforeLoweringAny) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  context.loadDialect<mlir::func::FuncDialect>();
  Fixture fixture(context);
  auto args = fixture.block->getArguments();
  fixture.addVector(args[0], args[1], 10, 80);
  fixture.addVector(args[0], args[1], 20, 90);
  mapping::CoveringPlan plan = fixture.planFor({{10, 80}, {20, 90}});
  plan.placements.back().bundle.name = "different_bundle";
  Calls calls;
  RecordingTarget target(calls);
  llk::MappedCompilation compilation;

  llvm::Error error = llk::testing::lowerSelectedOperationsForTest(
      fixture.module, target, plan, compilation);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("instance 90"),
            std::string::npos);
  EXPECT_TRUE(calls.empty());
}

TEST(SelectedGroupLowering, RejectsEscapingInternalValuesBeforeRewrite) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  context.loadDialect<mlir::func::FuncDialect>();
  Fixture fixture(context);
  auto args = fixture.block->getArguments();
  auto *first = fixture.addVector(args[0], args[1], 10, 77);
  fixture.addVector(first->getResult(0), args[1], 20, 77);
  mlir::OperationState externalState(fixture.builder.getUnknownLoc(),
                                     "test.external_use");
  externalState.addOperands({first->getResult(0)});
  fixture.builder.create(externalState);
  mapping::CoveringPlan plan = fixture.planFor({{10, 77}, {20, 77}});
  Calls calls;
  RecordingTarget target(calls);
  llk::MappedCompilation compilation;

  llvm::Error error = llk::testing::lowerSelectedOperationsForTest(
      fixture.module, target, plan, compilation);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("live external use"),
            std::string::npos);
  EXPECT_TRUE(calls.empty());
}

TEST(SelectedGroupLowering, RejectsInterveningSideEffectsBeforeRewrite) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  mlir::MLIRContext context(registry);
  context.loadDialect<mlir::func::FuncDialect>();
  Fixture fixture(context);
  auto args = fixture.block->getArguments();
  auto *first = fixture.addVector(args[0], args[1], 10, 77);
  mlir::OperationState effectState(fixture.builder.getUnknownLoc(),
                                   "test.side_effect");
  fixture.builder.create(effectState);
  fixture.addVector(first->getResult(0), args[1], 20, 77);
  mapping::CoveringPlan plan = fixture.planFor({{10, 77}, {20, 77}});
  Calls calls;
  RecordingTarget target(calls);
  llk::MappedCompilation compilation;

  llvm::Error error = llk::testing::lowerSelectedOperationsForTest(
      fixture.module, target, plan, compilation);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("intervening side effect"),
            std::string::npos);
  EXPECT_TRUE(calls.empty());
}
} // namespace
