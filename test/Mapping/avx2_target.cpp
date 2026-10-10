//===- avx2_target.cpp - AVX2 mapping target (D7) ------------------------===//

#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/PlanBinder.h"
#include "LLK/Mapping/WorkloadGraph.h"
#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include "LLK/Dialect/Micro/MicroDialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

using namespace mlir::llk::mapping;

/// Aliased because the tests also have locals named `target`.
namespace avx2_mapping = mlir::llk::target::avx2;

#ifndef LLK_SOURCE_DIR
#error "LLK_SOURCE_DIR must name the repository root"
#endif

namespace {

mlir::DictionaryAttr vectorAttributes(mlir::MLIRContext &context,
                                      llvm::StringRef op) {
  return mlir::DictionaryAttr::get(
      &context, {mlir::NamedAttribute(mlir::StringAttr::get(&context, "op"),
                                      mlir::StringAttr::get(&context, op))});
}

/// One `micro.vector(op = "add")` node with an external input and an output.
WorkloadGraph vectorGraph(mlir::MLIRContext &context) {
  WorkloadGraph graph;
  WorkloadValueId input =
      graph.addValue(WorkloadValue{0, mlir::Type(), "in", /*external=*/true});
  WorkloadValueId output =
      graph.addValue(WorkloadValue{0, mlir::Type(), "out", /*external=*/false});
  WorkloadNode node;
  node.opName = "micro.vector";
  node.attributes = vectorAttributes(context, "add");
  node.inputs.push_back(WorkloadPort{input, mlir::Type(), std::nullopt});
  node.outputs.push_back(WorkloadPort{output, mlir::Type(), std::nullopt});
  graph.addNode(std::move(node));
  graph.finalize();
  return graph;
}

llvm::Expected<std::unique_ptr<MappingTarget>> loadTarget() {
  return avx2_mapping::createMappingTarget(LLK_SOURCE_DIR);
}

/// A kernel in the shape `llk-to-micro` emits for a tile program: a logical
/// view of an external tensor, an async staging copy into SRAM, and a store
/// back to DRAM. Used to carry the tile-movement rules past matching and
/// through the binder, where a wrong port split would show.
constexpr llvm::StringLiteral kTileMovementKernel = R"mlir(
module {
  micro.kernel @tiles {
    %ext = tensor.empty() : tensor<8x32xbf16>
    %v = micro.tile_view %ext {shape = array<i64: 8, 32>} : tensor<8x32xbf16> -> !micro.tile<8x32xbf16, memory = #micro.memory<dram>>
    %t, %tok = micro.tile_async_copy %v {dst_memory = #micro.memory<sram>, owner = #micro.owner<worker>} : !micro.tile<8x32xbf16, memory = #micro.memory<dram>> -> !micro.tile<8x32xbf16, memory = #micro.memory<sram>, owner = #micro.owner<worker>>, !micro.async_token
    micro.tile_store %t {dst_memory = #micro.memory<dram>} : !micro.tile<8x32xbf16, memory = #micro.memory<sram>, owner = #micro.owner<worker>>
    micro.yield
  }
}
)mlir";

/// A bundle naming `emitterKey` with one well-typed integer parameter.
TargetBundle makeBundle(llvm::StringRef emitterKey,
                        mlir::MLIRContext &context) {
  TargetBundle bundle;
  bundle.name = "avx2.vector_add";
  bundle.emitterKey = emitterKey.str();
  bundle.parameters = mlir::DictionaryAttr::get(
      &context,
      {mlir::NamedAttribute(
          mlir::StringAttr::get(&context, "rows"),
          mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), 4))});
  return bundle;
}

} // namespace

TEST(Avx2Target, LoadsAndVerifiesItsConfiguration) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  EXPECT_EQ((*target)->name(), "x86-avx2");
  EXPECT_FALSE((*target)->machine().executors.empty());
  EXPECT_NE((*target)->rules().find("avx2.vector_add"), nullptr);
  EXPECT_NE((*target)->layouts().find("avx2.blocked_2d"), nullptr);
}

TEST(Avx2Target, EveryRuleEmitterIsDeclared) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  for (const RuleDef &rule : (*target)->rules().all())
    EXPECT_TRUE((*target)->isKnownEmitter(rule.emitter)) << rule.id;
  // The plugin's key list is target-private and reachable only from here.
  EXPECT_FALSE(avx2_mapping::emitterKeys().empty());
}

TEST(Avx2Target, ExposesAnEmitterForEachDeclaredKey) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  // The design §19 factory creates a target-owned emitter; its key is one the
  // target declares, so generic code can hand it a bundle without knowing the
  // AVX2 vocabulary.
  std::unique_ptr<TargetEmitter> primary = (*target)->createEmitter();
  ASSERT_NE(primary, nullptr);
  EXPECT_TRUE((*target)->isKnownEmitter(primary->key()));

  for (llvm::StringLiteral key : avx2_mapping::emitterKeys()) {
    std::unique_ptr<TargetEmitter> emitter = (*target)->createEmitter(key);
    ASSERT_NE(emitter, nullptr) << key.data();
    EXPECT_EQ(emitter->key(), key);
  }
  // An undeclared key has no emitter rather than a fabricated one.
  EXPECT_EQ((*target)->createEmitter("avx2_missing"), nullptr);
}

TEST(Avx2Target, EmitterVerifiesBundleCompleteness) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  std::unique_ptr<TargetEmitter> emitter = (*target)->createEmitter();
  ASSERT_NE(emitter, nullptr);
  mlir::MLIRContext context;

  // A bundle for the emitter's own key with well-typed parameters is complete.
  EXPECT_FALSE(
      static_cast<bool>(emitter->verify(makeBundle(emitter->key(), context))));

  // A rule with no bundle parameters produces a bundle whose `parameters` is a
  // null DictionaryAttr; that is the most common shape and must be accepted,
  // not dereferenced.
  TargetBundle parameterless = makeBundle(emitter->key(), context);
  parameterless.parameters = {};
  EXPECT_FALSE(static_cast<bool>(emitter->verify(parameterless)));

  // A bundle naming an emitter the target does not declare is rejected.
  llvm::Error unknown = emitter->verify(makeBundle("avx2_missing", context));
  ASSERT_TRUE(static_cast<bool>(unknown));
  EXPECT_NE(llvm::toString(std::move(unknown)).find("avx2_missing"),
            std::string::npos);

  // A bundle naming a *different but declared* emitter key is rejected too:
  // an emitter handles exactly one key.
  llvm::ArrayRef<llvm::StringLiteral> keys = avx2_mapping::emitterKeys();
  ASSERT_GE(keys.size(), 2u);
  std::unique_ptr<TargetEmitter> first = (*target)->createEmitter(keys[0]);
  ASSERT_NE(first, nullptr);
  llvm::Error otherKey = first->verify(makeBundle(keys[1], context));
  ASSERT_TRUE(static_cast<bool>(otherKey));
  EXPECT_NE(llvm::toString(std::move(otherKey)).find("handles"),
            std::string::npos);

  // A parameter whose value is not the integer/string shape the plugin
  // contract permits is rejected.
  TargetBundle malformed = makeBundle(emitter->key(), context);
  malformed.parameters = mlir::DictionaryAttr::get(
      &context, {mlir::NamedAttribute(mlir::StringAttr::get(&context, "rows"),
                                      mlir::UnitAttr::get(&context))});
  llvm::Error badParameters = emitter->verify(malformed);
  ASSERT_TRUE(static_cast<bool>(badParameters));
  EXPECT_NE(llvm::toString(std::move(badParameters)).find("rows"),
            std::string::npos);
}

TEST(Avx2Target, MatchesItsOwnRuleForTheVectorOperation) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target));
  mlir::MLIRContext context;

  WorkloadNode node;
  node.opName = "micro.vector";
  node.attributes = vectorAttributes(context, "add");
  std::vector<const RuleDef *> matches = matchRules(node, (*target)->rules());
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches[0]->id, "avx2.vector_add");
}

TEST(Avx2Target, LeavesUnruledVectorVariantsUnmatched) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target));
  mlir::MLIRContext context;

  // Matching is per `op` value, not per op name: a variant the target has no
  // implementation for must stay uncovered rather than fall back to a
  // catch-all. `exp` is a `micro.vector` op the AVX2 target does not implement.
  WorkloadNode node;
  node.opName = "micro.vector";
  node.attributes = vectorAttributes(context, "exp");
  EXPECT_TRUE(matchRules(node, (*target)->rules()).empty());
}

TEST(Avx2Target, CoversTheTileMovementOpsTheLoweringEmits) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  // `llk-to-micro` emits these two tile-level movement ops, so a rule must
  // cover each one; without them a compiler-generated tile program fails with
  // `no_matching_rule`. Each rule's `emit` key must be one the target declares.
  for (llvm::StringRef op : {"micro.tile_async_copy", "micro.tile_store"}) {
    WorkloadNode node;
    node.opName = op.str();
    std::vector<const RuleDef *> matches = matchRules(node, (*target)->rules());
    ASSERT_FALSE(matches.empty()) << "no rule matches " << op.str();
    for (const RuleDef *rule : matches)
      EXPECT_TRUE((*target)->isKnownEmitter(rule->emitter)) << rule->id;
  }
}

TEST(Avx2Target, BindsTheTileMovementOpsTheLoweringEmits) {
  // Carried acceptance from the rules that cover `micro.tile_async_copy` and
  // `micro.tile_store`. Two claims, checked separately because different
  // defects falsify them:
  //
  //   * the plan binds and stamps `micro.mapping` on both nodes -- the binder
  //     reads placement identities and value links, never a rule's ports, so
  //     this proves materialization but cannot see a port split;
  //   * each rule's declared port split mirrors the node's own ports and bound
  //     value ids -- the split feeds only the candidate (and its id), so a
  //     wrong split must be caught at the candidate, not at bind time.
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::micro::MicroDialect>();
  context.getOrLoadDialect<mlir::tensor::TensorDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kTileMovementKernel, &context);
  ASSERT_TRUE(module);
  mlir::Operation *kernel = nullptr;
  module->walk([&](mlir::Operation *op) {
    if (!kernel && op->getName().getStringRef() == "micro.kernel")
      kernel = op;
  });
  ASSERT_NE(kernel, nullptr);

  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  llvm::Expected<WorkloadGraph> graph = extractWorkloadGraph(kernel);
  ASSERT_TRUE(static_cast<bool>(graph)) << llvm::toString(graph.takeError());

  LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "bf16";

  // The declared split, against the node the lowering produces. The ports are
  // positional: one input per input port, then one output per output port.
  struct SplitCase {
    llvm::StringLiteral op;
    llvm::StringLiteral rule;
    size_t inputs;
    size_t outputs;
  };
  for (const SplitCase &test :
       {SplitCase{"micro.tile_async_copy", "avx2.tile_async_copy", 1, 1},
        SplitCase{"micro.tile_store", "avx2.tile_store", 1, 0}}) {
    const WorkloadNode *node = nullptr;
    for (const WorkloadNode &candidate : graph->getNodes())
      if (candidate.opName == test.op) {
        node = &candidate;
        break;
      }
    ASSERT_NE(node, nullptr) << test.op.str();
    ASSERT_EQ(node->inputs.size(), test.inputs) << test.op.str();
    ASSERT_EQ(node->outputs.size(), test.outputs) << test.op.str();

    const RuleDef *rule = (*target)->rules().find(test.rule);
    ASSERT_NE(rule, nullptr) << test.rule.str();
    std::string reason;
    std::optional<MappingCandidate> candidate = toMappingCandidate(
        *rule, *node, (*target)->machine(), layoutContext, &reason);
    ASSERT_TRUE(candidate) << test.rule.str() << ": " << reason;
    ASSERT_EQ(candidate->ports.size(), test.inputs + test.outputs)
        << test.rule.str();
    for (size_t i = 0; i < test.inputs; ++i) {
      EXPECT_TRUE(candidate->ports[i].isInput) << test.rule.str();
      EXPECT_EQ(candidate->ports[i].value, node->inputs[i].value)
          << test.rule.str();
    }
    for (size_t i = 0; i < test.outputs; ++i) {
      EXPECT_FALSE(candidate->ports[test.inputs + i].isInput)
          << test.rule.str();
      EXPECT_EQ(candidate->ports[test.inputs + i].value, node->outputs[i].value)
          << test.rule.str();
    }
  }

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(*graph, **target, context, layoutContext, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_EQ(result->plans.size(), 1u);

  llvm::Expected<BoundPlan> bound =
      bindPlan(*module, result->plans.front(), **target);
  ASSERT_TRUE(static_cast<bool>(bound)) << llvm::toString(bound.takeError());
  EXPECT_TRUE(bound->unmaterialized.empty());

  std::string text;
  llvm::raw_string_ostream stream(text);
  bound->module->print(stream);
  llvm::StringRef mapped(stream.str());

  // Materialization: each movement op is stamped with its own rule and emitter,
  // and exactly one placement per node.
  EXPECT_NE(mapped.find("rule = \"avx2.tile_async_copy\""),
            llvm::StringRef::npos);
  EXPECT_NE(mapped.find("emitter = \"avx2_tile_copy\""), llvm::StringRef::npos);
  EXPECT_NE(mapped.find("rule = \"avx2.tile_store\""), llvm::StringRef::npos);
  EXPECT_NE(mapped.find("emitter = \"avx2_tile_store\""),
            llvm::StringRef::npos);
  EXPECT_EQ(mapped.count("micro.mapping"), 2u);
}

TEST(Avx2Target, MapsAVectorNodeEndToEnd) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  mlir::MLIRContext context;
  WorkloadGraph graph = vectorGraph(context);

  LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "f32"; // the profile's lanes for f32 are 8
  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;

  CoveringSearch search(graph, **target, context, layoutContext, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_EQ(result->plans.size(), 1u);
  EXPECT_EQ(result->plans[0].instances.size(), 1u);
  EXPECT_TRUE(result->plans[0].connections.empty());
  EXPECT_FALSE(result->searchTruncated);
}

//===----------------------------------------------------------------------===//
// C2: a selected bundle lowers into AVX2 code
//===----------------------------------------------------------------------===//
//
// Verifying that a bundle is well-formed and lowering it to an implementation
// are different promises: a rule file can declare an emitter key it has no
// code for. These tests pin the second promise -- a selected bundle on its own
// changes the code this target emits.

namespace {

/// A kernel with one `micro.vector(op = "add")`: the shape a selected
/// `avx2.vector_add` candidate covers.
constexpr llvm::StringLiteral kVectorAddKernel = R"mlir(
module {
  micro.kernel @vector_add {
    %ext = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %ext {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)mlir";

/// The same kernel over `i8`, which this backend has no arithmetic path for.
constexpr llvm::StringLiteral kUnsupportedDtypeKernel = R"mlir(
module {
  micro.kernel @vector_add_i8 {
    %ext = tensor.empty() : tensor<8x8xi8>
    %t = micro.tile_view %ext {shape = array<i64: 8, 8>} : tensor<8x8xi8> -> !micro.tile<8x8xi8, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t : !micro.tile<8x8xi8, memory = #micro.memory<sram>>, !micro.tile<8x8xi8, memory = #micro.memory<sram>> -> !micro.tile<8x8xi8, memory = #micro.memory<sram>>
    micro.yield
  }
}
)mlir";

/// A bundle for `avx2_vector_add` as the search persists it: the emitter key,
/// and the `VW` the rule's `param` solved.
TargetBundle vectorAddBundle(mlir::MLIRContext &context, int64_t vectorWidth) {
  TargetBundle bundle;
  bundle.name = "avx2.vector.add.f32";
  bundle.emitterKey = "avx2_vector_add";
  bundle.parameters = mlir::DictionaryAttr::get(
      &context, {mlir::NamedAttribute(
                    mlir::StringAttr::get(&context, "VW"),
                    mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64),
                                           vectorWidth))});
  return bundle;
}

/// The `micro.vector` operations of `module` -- what a selected candidate
/// covers. Found by name so the test needs no Micro op class, the same way
/// generic mapping code never sees one.
llvm::SmallVector<mlir::Operation *> coveredVectorOps(mlir::ModuleOp module) {
  llvm::SmallVector<mlir::Operation *> covered;
  module.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "micro.vector")
      covered.push_back(op);
  });
  return covered;
}

std::string print(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  return text;
}

} // namespace

TEST(Avx2Target, LowersTheSelectedVectorWidthIntoTheTileLayout) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  mlir::MLIRContext context;
  context.loadDialect<mlir::micro::MicroDialect, mlir::tensor::TensorDialect>();

  auto lowerWithWidth = [&](int64_t width) -> std::string {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(kVectorAddKernel, &context);
    EXPECT_TRUE(module);
    TargetLoweringContext lowering{(*target)->machine(), {}, {}};
    mlir::IRRewriter rewriter(&context);
    std::unique_ptr<TargetEmitter> emitter =
        (*target)->createEmitter("avx2_vector_add");
    EXPECT_TRUE(emitter);
    llvm::Error error =
        emitter->lower(coveredVectorOps(*module),
                       vectorAddBundle(context, width), lowering, rewriter);
    EXPECT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));
    return print(*module);
  };

  std::string narrow = lowerWithWidth(4);
  std::string wide = lowerWithWidth(8);

  // The bundle decides the physical implementation: two widths, two programs.
  EXPECT_NE(narrow.find("vector = 4"), std::string::npos) << narrow;
  EXPECT_EQ(narrow.find("vector = 8"), std::string::npos) << narrow;
  EXPECT_NE(wide.find("vector = 8"), std::string::npos) << wide;
  EXPECT_NE(narrow, wide);
}

TEST(Avx2Target, RejectsBundlesItCannotLowerAndLeavesTheIrAlone) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  mlir::MLIRContext context;
  context.loadDialect<mlir::micro::MicroDialect, mlir::tensor::TensorDialect>();

  std::unique_ptr<TargetEmitter> emitter =
      (*target)->createEmitter("avx2_vector_add");
  ASSERT_TRUE(emitter);

  auto refusal = [&](llvm::StringRef source,
                     const TargetBundle &bundle) -> std::string {
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceString<mlir::ModuleOp>(source, &context);
    EXPECT_TRUE(module);
    std::string before = print(*module);
    TargetLoweringContext lowering{(*target)->machine(), {}, {}};
    mlir::IRRewriter rewriter(&context);
    llvm::Error error =
        emitter->lower(coveredVectorOps(*module), bundle, lowering, rewriter);
    EXPECT_TRUE(static_cast<bool>(error))
        << "lowering was expected to be refused";
    std::string message = llvm::toString(std::move(error));
    // A refused bundle must not have rewritten anything: the contract is
    // validated before the first write.
    EXPECT_EQ(print(*module), before);
    return message;
  };

  // A bundle naming an emitter this target does not declare.
  TargetBundle unknown = vectorAddBundle(context, 8);
  unknown.emitterKey = "avx2_missing";
  EXPECT_NE(refusal(kVectorAddKernel, unknown).find("does not declare"),
            std::string::npos);

  // A parameter of the wrong type: the width is a number, not a word.
  TargetBundle mistyped = vectorAddBundle(context, 8);
  mistyped.parameters = mlir::DictionaryAttr::get(
      &context,
      {mlir::NamedAttribute(mlir::StringAttr::get(&context, "VW"),
                            mlir::StringAttr::get(&context, "eight"))});
  EXPECT_NE(refusal(kVectorAddKernel, mistyped).find("not an integer"),
            std::string::npos);

  // A width this backend cannot emit.
  TargetBundle odd = vectorAddBundle(context, 3);
  EXPECT_NE(refusal(kVectorAddKernel, odd).find("power of two"),
            std::string::npos);

  // A dtype with no arithmetic path.
  EXPECT_NE(refusal(kUnsupportedDtypeKernel, vectorAddBundle(context, 8))
                .find("no AVX2 arithmetic implementation"),
            std::string::npos);
}

/// A kernel whose epilogue is the shape the fused rule implements: a convert
/// feeding a SiLU feeding the gating multiply.
constexpr llvm::StringLiteral kFusedVectorKernel = R"mlir(
module {
  micro.kernel @fused {
    %ext = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %ext {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %c = micro.vector "convert" %t : !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %s = micro.vector "silu" %c : !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %m = micro.vector "mul" %s, %t : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.tile_store %m {dst_memory = #micro.memory<dram>} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)mlir";

constexpr llvm::StringLiteral kFusedBf16ToF32Kernel = R"mlir(
module {
  micro.kernel @fused_bf16_to_f32 {
    %input = tensor.empty() : tensor<8x8xbf16>
    %tile = micro.tile_view %input {shape = array<i64: 8, 8>} : tensor<8x8xbf16> -> !micro.tile<8x8xbf16, memory = #micro.memory<sram>>
    %gate = tensor.empty() : tensor<8x8xf32>
    %gate_tile = micro.tile_view %gate {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %converted = micro.vector "convert" %tile : !micro.tile<8x8xbf16, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %activated = micro.vector "silu" %converted : !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %result = micro.vector "mul" %activated, %gate_tile : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.tile_store %result {dst_memory = #micro.memory<dram>} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)mlir";

constexpr llvm::StringLiteral kFusedF16ToF32Kernel = R"mlir(
module {
  micro.kernel @fused_f16_to_f32 {
    %input = tensor.empty() : tensor<8x8xf16>
    %tile = micro.tile_view %input {shape = array<i64: 8, 8>} : tensor<8x8xf16> -> !micro.tile<8x8xf16, memory = #micro.memory<sram>>
    %gate = tensor.empty() : tensor<8x8xf32>
    %gate_tile = micro.tile_view %gate {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %converted = micro.vector "convert" %tile : !micro.tile<8x8xf16, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %activated = micro.vector "silu" %converted : !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %result = micro.vector "mul" %activated, %gate_tile : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.tile_store %result {dst_memory = #micro.memory<dram>} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)mlir";

TEST(Avx2Target, LowersAWholeFusedGroupInOneBundle) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  mlir::MLIRContext context;
  context.loadDialect<mlir::micro::MicroDialect, mlir::tensor::TensorDialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kFusedVectorKernel, &context);
  ASSERT_TRUE(module);

  // The bundle a fused rule emits: the group's own emitter key, and the vector
  // width the rule solved.
  TargetBundle fused;
  fused.name = "avx2.fused.convert_silu_mul";
  fused.emitterKey = "avx2_fused_convert_silu_mul";
  fused.parameters = mlir::DictionaryAttr::get(
      &context,
      {mlir::NamedAttribute(
          mlir::StringAttr::get(&context, "VW"),
          mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), 8))});

  std::unique_ptr<TargetEmitter> emitter =
      (*target)->createEmitter("avx2_fused_convert_silu_mul");
  ASSERT_TRUE(emitter);

  llvm::SmallVector<mlir::Operation *> covered = coveredVectorOps(*module);
  ASSERT_EQ(covered.size(), 3u) << "the group is all three epilogue ops";

  mlir::Operation *kernel = nullptr;
  module->walk([&](mlir::Operation *op) {
    if (!kernel && op->getName().getStringRef() == "micro.kernel")
      kernel = op;
  });
  ASSERT_NE(kernel, nullptr);
  llvm::Expected<WorkloadGraph> graph = extractWorkloadGraph(kernel);
  ASSERT_TRUE(static_cast<bool>(graph)) << llvm::toString(graph.takeError());
  LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "f32";
  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  options.topK = 0;
  CoveringSearch search(*graph, **target, context, layoutContext, options);
  llvm::Expected<MappingSearchResult> selected = search.search();
  ASSERT_TRUE(static_cast<bool>(selected))
      << llvm::toString(selected.takeError());
  ASSERT_FALSE(selected->plans.empty());
  auto selectedFusedPlan =
      llvm::find_if(selected->plans, [](const CoveringPlan &plan) {
        return llvm::any_of(
            plan.placements, [](const PlanPlacement &placement) {
              return placement.rule == "avx2.fused_convert_silu_mul";
            });
      });
  std::string searchDiagnostics;
  for (const Diagnostic &diagnostic : selected->frontier.diagnostics)
    searchDiagnostics += diagnostic.message + "\n";
  ASSERT_NE(selectedFusedPlan, selected->plans.end())
      << "exact search did not select the shipped fused rule; plan count="
      << selected->plans.size() << "\n"
      << searchDiagnostics;
  const CoveringPlan &selectedPlan = *selectedFusedPlan;
  llvm::SmallVector<const PlanPlacement *> fusedPlacements;
  for (const PlanPlacement &placement : selectedPlan.placements)
    if (placement.rule == "avx2.fused_convert_silu_mul")
      fusedPlacements.push_back(&placement);
  ASSERT_EQ(fusedPlacements.size(), 3u);
  const InstanceId fusedInstance = fusedPlacements.front()->instance;
  for (const PlanPlacement *placement : fusedPlacements) {
    EXPECT_EQ(placement->instance, fusedInstance);
    ASSERT_TRUE(placement->bundle.parameters);
    auto bundleWidth =
        placement->bundle.parameters.getAs<mlir::IntegerAttr>("VW");
    ASSERT_TRUE(bundleWidth);
    EXPECT_EQ(bundleWidth.getInt(), 8);
    auto resolvedWidth = placement->resolvedParameters.find("VW");
    ASSERT_NE(resolvedWidth, placement->resolvedParameters.end());
    ASSERT_TRUE(std::holds_alternative<int64_t>(resolvedWidth->second));
    EXPECT_EQ(std::get<int64_t>(resolvedWidth->second), 8);
  }

  TargetLoweringContext lowering{(*target)->machine(), {}, {}};
  mlir::IRRewriter rewriter(&context);
  llvm::Error error = emitter->lower(covered, fused, lowering, rewriter);
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));

  // Every operation the group covers carries the target's physical decision. A
  // fused bundle that rewrote only its anchor would leave the rest of the match
  // at the reference layout, which is a different program from the one the plan
  // selected.
  // Counting *result* annotations rather than every mention: a rewritten value
  // that feeds another rewritten op shows its layout on both sides, so the raw
  // occurrence count measures shared operands as well as rewritten results.
  std::string text = print(*module);
  const std::string marker =
      "-> !micro.tile<8x8xf32, layout = #micro.layout<vectorized";
  size_t rewritten = 0;
  for (size_t at = text.find(marker); at != std::string::npos;
       at = text.find(marker, at + 1))
    ++rewritten;
  EXPECT_EQ(rewritten, 3u) << text;
  EXPECT_NE(text.find("vector = 8"), std::string::npos) << text;

  auto checkConversion = [&](llvm::StringRef source, bool shouldSucceed) {
    mlir::OwningOpRef<mlir::ModuleOp> typedModule =
        mlir::parseSourceString<mlir::ModuleOp>(source, &context);
    ASSERT_TRUE(typedModule);
    llvm::SmallVector<mlir::Operation *> typedOps =
        coveredVectorOps(*typedModule);
    ASSERT_EQ(typedOps.size(), 3u);
    mlir::IRRewriter typedRewriter(&context);
    llvm::Error typedError =
        emitter->lower(typedOps, fused, lowering, typedRewriter);
    EXPECT_EQ(!static_cast<bool>(typedError), shouldSucceed)
        << (typedError ? llvm::toString(std::move(typedError)) : "");
  };
  checkConversion(kFusedBf16ToF32Kernel, /*shouldSucceed=*/true);
  checkConversion(kFusedF16ToF32Kernel, /*shouldSucceed=*/false);

  std::string repeatedOperandSource = kFusedVectorKernel.str();
  size_t repeatedOperand = repeatedOperandSource.find("%s, %t");
  ASSERT_NE(repeatedOperand, std::string::npos);
  repeatedOperandSource.replace(repeatedOperand, 6, "%s, %s");
  mlir::OwningOpRef<mlir::ModuleOp> repeatedOperandModule =
      mlir::parseSourceString<mlir::ModuleOp>(repeatedOperandSource, &context);
  ASSERT_TRUE(repeatedOperandModule);
  llvm::SmallVector<mlir::Operation *> repeatedOperandOps =
      coveredVectorOps(*repeatedOperandModule);
  ASSERT_EQ(repeatedOperandOps.size(), 3u);
  mlir::IRRewriter repeatedOperandRewriter(&context);
  llvm::Error repeatedOperandError = emitter->lower(
      repeatedOperandOps, fused, lowering, repeatedOperandRewriter);
  EXPECT_FALSE(static_cast<bool>(repeatedOperandError))
      << llvm::toString(std::move(repeatedOperandError));

  std::string externalUseSource = kFusedVectorKernel.str();
  size_t terminalStore = externalUseSource.find("    micro.tile_store %m");
  ASSERT_NE(terminalStore, std::string::npos);
  externalUseSource.insert(terminalStore,
                           "    micro.tile_store %c {dst_memory = "
                           "#micro.memory<dram>} : !micro.tile<8x8xf32, "
                           "memory = #micro.memory<sram>>\n");
  mlir::OwningOpRef<mlir::ModuleOp> externalUseModule =
      mlir::parseSourceString<mlir::ModuleOp>(externalUseSource, &context);
  ASSERT_TRUE(externalUseModule);
  llvm::SmallVector<mlir::Operation *> externalUseOps =
      coveredVectorOps(*externalUseModule);
  ASSERT_EQ(externalUseOps.size(), 3u);
  std::string beforeExternalUse = print(*externalUseModule);
  mlir::IRRewriter externalUseRewriter(&context);
  llvm::Error externalUseError =
      emitter->lower(externalUseOps, fused, lowering, externalUseRewriter);
  ASSERT_TRUE(static_cast<bool>(externalUseError));
  EXPECT_NE(llvm::toString(std::move(externalUseError))
                .find("fused intermediate has a live external use"),
            std::string::npos);
  EXPECT_EQ(print(*externalUseModule), beforeExternalUse);

  TargetBundle fusedWidth4 = fused;
  fusedWidth4.parameters = mlir::DictionaryAttr::get(
      &context,
      {mlir::NamedAttribute(
          mlir::StringAttr::get(&context, "VW"),
          mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), 4))});
  mlir::OwningOpRef<mlir::ModuleOp> width4Module =
      mlir::parseSourceString<mlir::ModuleOp>(kFusedVectorKernel, &context);
  ASSERT_TRUE(width4Module);
  llvm::SmallVector<mlir::Operation *> width4Ops =
      coveredVectorOps(*width4Module);
  ASSERT_EQ(width4Ops.size(), 3u);
  mlir::IRRewriter width4Rewriter(&context);
  llvm::Error width4Error =
      emitter->lower(width4Ops, fusedWidth4, lowering, width4Rewriter);
  ASSERT_FALSE(static_cast<bool>(width4Error))
      << llvm::toString(std::move(width4Error));
  EXPECT_NE(print(*width4Module).find("vector = 4"), std::string::npos);

  auto rejectWidth = [&](TargetBundle invalidBundle,
                         bool recordLayout) -> std::string {
    mlir::OwningOpRef<mlir::ModuleOp> invalidModule =
        mlir::parseSourceString<mlir::ModuleOp>(kFusedVectorKernel, &context);
    EXPECT_TRUE(invalidModule);
    if (!invalidModule)
      return {};
    llvm::SmallVector<mlir::Operation *> invalidOps =
        coveredVectorOps(*invalidModule);
    EXPECT_EQ(invalidOps.size(), 3u);
    if (invalidOps.size() != 3)
      return {};
    if (recordLayout) {
      auto i64 = mlir::IntegerType::get(&context, 64);
      auto width = mlir::IntegerAttr::get(i64, 8);
      auto layout = mlir::DictionaryAttr::get(
          &context,
          {mlir::NamedAttribute(mlir::StringAttr::get(&context, "VW"), width)});
      auto layouts = mlir::DictionaryAttr::get(
          &context, {mlir::NamedAttribute(
                        mlir::StringAttr::get(&context, "selected"), layout)});
      auto mapping = mlir::DictionaryAttr::get(
          &context,
          {mlir::NamedAttribute(
              mlir::StringAttr::get(&context, "layout_parameters"), layouts)});
      invalidOps.front()->setAttr("micro.mapping", mapping);
    }
    mlir::IRRewriter invalidRewriter(&context);
    llvm::Error invalidError =
        emitter->lower(invalidOps, invalidBundle, lowering, invalidRewriter);
    EXPECT_TRUE(static_cast<bool>(invalidError));
    return llvm::toString(std::move(invalidError));
  };

  TargetBundle missingWidth = fused;
  missingWidth.parameters = {};
  EXPECT_NE(rejectWidth(missingWidth, /*recordLayout=*/false)
                .find("no vector width was selected"),
            std::string::npos);
  TargetBundle conflictingWidth = fused;
  conflictingWidth.parameters = mlir::DictionaryAttr::get(
      &context,
      {mlir::NamedAttribute(
          mlir::StringAttr::get(&context, "VW"),
          mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), 4))});
  EXPECT_NE(rejectWidth(conflictingWidth, /*recordLayout=*/true)
                .find("conflicts with selected layout width"),
            std::string::npos);
}

TEST(Avx2Target, ReportsEmitterKeysItHasNoLoweringFor) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  mlir::MLIRContext context;
  context.loadDialect<mlir::micro::MicroDialect, mlir::tensor::TensorDialect>();

  // `avx2_mma` is declared and its bundles verify; that is not the same as
  // having an implementation, and the emitter says so rather than pretending.
  std::unique_ptr<TargetEmitter> emitter = (*target)->createEmitter("avx2_mma");
  ASSERT_TRUE(emitter);

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kVectorAddKernel, &context);
  ASSERT_TRUE(module);

  TargetBundle bundle;
  bundle.name = "avx2.mma.bf16";
  bundle.emitterKey = "avx2_mma";
  EXPECT_FALSE(static_cast<bool>(emitter->verify(bundle)));

  TargetLoweringContext lowering{(*target)->machine(), {}, {}};
  mlir::IRRewriter rewriter(&context);
  llvm::Error error =
      emitter->lower(coveredVectorOps(*module), bundle, lowering, rewriter);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("no lowering implementation"),
            std::string::npos);
}
