//===- generic_accelerator_target.cpp - second target (D7) ---------------===//

#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Target/GenericAccelerator/Mapping/GenericAcceleratorMappingTarget.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <gtest/gtest.h>

#include <cctype>
#include <memory>
#include <set>
#include <string>
#include <vector>

using namespace mlir::llk::mapping;

/// Aliased because the tests also have locals named `target`.
namespace accel_mapping = mlir::llk::target::generic_accel;

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
  return accel_mapping::createMappingTarget(LLK_SOURCE_DIR);
}

/// A bundle naming `emitterKey` with one well-typed integer parameter.
TargetBundle makeBundle(llvm::StringRef emitterKey,
                        mlir::MLIRContext &context) {
  TargetBundle bundle;
  bundle.name = "accel.vector_add";
  bundle.emitterKey = emitterKey.str();
  bundle.parameters = mlir::DictionaryAttr::get(
      &context,
      {mlir::NamedAttribute(
          mlir::StringAttr::get(&context, "tile"),
          mlir::IntegerAttr::get(mlir::IntegerType::get(&context, 64), 16))});
  return bundle;
}

/// Whole-word search: `npu` must not match inside `$inputs`, and `npu_engine`
/// must still be caught.
bool mentionsWord(const std::string &text, const std::string &word) {
  size_t position = 0;
  while ((position = text.find(word, position)) != std::string::npos) {
    size_t after = position + word.size();
    bool leftFree =
        position == 0 ||
        !std::isalnum(static_cast<unsigned char>(text[position - 1]));
    bool rightFree = after >= text.size() ||
                     !std::isalnum(static_cast<unsigned char>(text[after]));
    if (leftFree && rightFree)
      return true;
    position = after;
  }
  return false;
}

} // namespace

TEST(GenericAcceleratorTarget, LoadsAndVerifiesItsConfiguration) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  EXPECT_EQ((*target)->name(), "generic-ai-accel");
  EXPECT_NE((*target)->rules().find("accel.vector_add"), nullptr);
  EXPECT_NE((*target)->layouts().find("accel.row_major"), nullptr);
  // A different owner hierarchy: the rules ask for a `pe`, which only this
  // machine declares.
  EXPECT_NE((*target)->machine().findExecutor("pe.0"), nullptr);
  EXPECT_TRUE((*target)->machine().ownerMatches("pe", "pe.0"));
}

TEST(GenericAcceleratorTarget, EveryRuleEmitterIsDeclared) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  for (const RuleDef &rule : (*target)->rules().all())
    EXPECT_TRUE((*target)->isKnownEmitter(rule.emitter)) << rule.id;
  EXPECT_FALSE(accel_mapping::emitterKeys().empty());
}

TEST(GenericAcceleratorTarget, ExposesAnEmitterForEachDeclaredKey) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  std::unique_ptr<TargetEmitter> primary = (*target)->createEmitter();
  ASSERT_NE(primary, nullptr);
  EXPECT_TRUE((*target)->isKnownEmitter(primary->key()));

  for (llvm::StringRef key : accel_mapping::emitterKeys()) {
    std::unique_ptr<TargetEmitter> emitter = (*target)->createEmitter(key);
    ASSERT_NE(emitter, nullptr) << key.data();
    EXPECT_EQ(emitter->key(), key);
  }
  EXPECT_EQ((*target)->createEmitter("accel_missing"), nullptr);
}

TEST(GenericAcceleratorTarget, EmitterVerifiesBundleCompleteness) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  std::unique_ptr<TargetEmitter> emitter = (*target)->createEmitter();
  ASSERT_NE(emitter, nullptr);
  mlir::MLIRContext context;

  EXPECT_FALSE(
      static_cast<bool>(emitter->verify(makeBundle(emitter->key(), context))));

  // A parameterless bundle carries a null DictionaryAttr; that is the common
  // shape and must be accepted, not dereferenced.
  TargetBundle parameterless = makeBundle(emitter->key(), context);
  parameterless.parameters = {};
  EXPECT_FALSE(static_cast<bool>(emitter->verify(parameterless)));

  llvm::Error unknown = emitter->verify(makeBundle("accel_missing", context));
  ASSERT_TRUE(static_cast<bool>(unknown));
  EXPECT_NE(llvm::toString(std::move(unknown)).find("accel_missing"),
            std::string::npos);

  // A bundle naming a different but declared emitter key is rejected.
  llvm::ArrayRef<llvm::StringRef> keys = accel_mapping::emitterKeys();
  ASSERT_GE(keys.size(), 2u);
  std::unique_ptr<TargetEmitter> first = (*target)->createEmitter(keys[0]);
  ASSERT_NE(first, nullptr);
  llvm::Error otherKey = first->verify(makeBundle(keys[1], context));
  ASSERT_TRUE(static_cast<bool>(otherKey));
  EXPECT_NE(llvm::toString(std::move(otherKey)).find("handles"),
            std::string::npos);

  TargetBundle malformed = makeBundle(emitter->key(), context);
  malformed.parameters = mlir::DictionaryAttr::get(
      &context, {mlir::NamedAttribute(mlir::StringAttr::get(&context, "tile"),
                                      mlir::UnitAttr::get(&context))});
  llvm::Error badParameters = emitter->verify(malformed);
  ASSERT_TRUE(static_cast<bool>(badParameters));
  EXPECT_NE(llvm::toString(std::move(badParameters)).find("tile"),
            std::string::npos);
}

TEST(GenericAcceleratorTarget, MapsAVectorNodeEndToEnd) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadTarget();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  mlir::MLIRContext context;
  WorkloadGraph graph = vectorGraph(context);

  LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "f32";
  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;

  CoveringSearch search(graph, **target, context, layoutContext, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_EQ(result->plans.size(), 1u);
  EXPECT_EQ(result->plans[0].instances.size(), 1u);
}

/// The point of the second target: no target vocabulary reached the canonical
/// dialect. If this fails, a target-specific name leaked into generic ODS.
TEST(GenericAcceleratorTarget, TargetVocabularyStaysOutOfGenericMicroOds) {
  // Generated ODS *and* hand-written headers: MicroEnums.h carries the Owner
  // enumerants and is exactly where a target word can hide (CLAUDE.md warns it
  // is hand-written and not tablegen'd).
  const std::vector<std::string> files = {
      "/include/LLK/Dialect/Micro/MicroDialect.td",
      "/include/LLK/Dialect/Micro/MicroTypes.td",
      "/include/LLK/Dialect/Micro/MicroOps.td",
      "/include/LLK/Dialect/Micro/MicroDialect.h",
      "/include/LLK/Dialect/Micro/MicroEnums.h",
  };
  // Target-family words that must never name a generic Micro concept.
  const std::vector<std::string> targetWords = {
      "avx2", "nvidia", "ampere", "sm80", "ttgir",    "npu",  "accel",
      "mxu",  "vpu",    "warp",   "wave", "subgroup", "lane", "pe_group",
  };
  // Debt, not policy: generic Micro already ships warp-class owner names
  // (Owner::warp, wave, subgroup, pe_group, pe, lane). Tracked so the debt is
  // visible; a NEW leak fails, and removing one fails until this set shrinks.
  const std::set<std::string> kKnownGenericLeaks = {
      "warp", "wave", "subgroup", "pe_group", "lane",
  };

  std::set<std::string> found;
  for (const std::string &file : files) {
    std::string path = std::string(LLK_SOURCE_DIR) + file;
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
        llvm::MemoryBuffer::getFile(path);
    ASSERT_TRUE(static_cast<bool>(buffer)) << path;
    std::string lowered = buffer.get()->getBuffer().lower();
    for (const std::string &word : targetWords)
      if (mentionsWord(lowered, word))
        found.insert(word);
  }
  EXPECT_EQ(found, kKnownGenericLeaks)
      << "generic Micro leaks changed; update kKnownGenericLeaks only when the "
         "dialect genuinely changed";
}
