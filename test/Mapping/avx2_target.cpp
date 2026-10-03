//===- avx2_target.cpp - AVX2 mapping target (D7) ------------------------===//

#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

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
