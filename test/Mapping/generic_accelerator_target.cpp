//===- generic_accelerator_target.cpp - second target (D7) ---------------===//

#include "LLK/Dialect/Micro/MicroEnums.h"
#include "LLK/Machine/MachineModelLoader.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Target/GenericAccelerator/Mapping/GenericAcceleratorMappingTarget.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"

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
  // Empty by construction now: the closed owner enumeration that used to ship
  // warp-class names (warp, wave, subgroup, pe, pe_group, lane) is gone -- the
  // dialect keeps only the abstract classes group/worker/vector/matrix/transfer
  // -- and the spatial axis is an open symbol. A target word here would be a
  // genuine leak, so the tracked debt is retired rather than reduced.
  const std::set<std::string> kKnownGenericLeaks = {};

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
      << "a target word reached the canonical Micro ODS; the generic dialect "
         "must name only abstract owner classes";
}

TEST(GenericAcceleratorTarget, DoesNotClaimAnExecutableBackend) {
  // The second target proves the C2 boundary as much as the first one does.
  // Its bundles are well-formed and its emitters verify them, but this package
  // ships no lowering: the accelerator has no code generator here, and saying
  // so is the honest answer. What would be dishonest is a lowering hook that
  // returned success and emitted nothing, because the compiler could not then
  // tell a configured target from an executable one.
  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      accel_mapping::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  llvm::ArrayRef<llvm::StringRef> keys = accel_mapping::emitterKeys();
  ASSERT_FALSE(keys.empty());

  mlir::MLIRContext context;
  mlir::DictionaryAttr noParameters;
  TargetBundle bundle;
  bundle.name = "accel.vector.add.f32";
  bundle.emitterKey = keys.front().str();
  bundle.parameters = noParameters;

  for (llvm::StringRef key : keys) {
    std::unique_ptr<TargetEmitter> emitter = (*target)->createEmitter(key);
    ASSERT_TRUE(emitter) << key.str();
    bundle.emitterKey = key.str();
    EXPECT_FALSE(static_cast<bool>(emitter->verify(bundle)))
        << "a well-formed bundle must still verify for " << key.str();

    TargetLoweringContext lowering{(*target)->machine(), {}, {}};
    mlir::IRRewriter rewriter(&context);
    llvm::Error error =
        emitter->lower(/*coveredOps=*/{}, bundle, lowering, rewriter);
    ASSERT_TRUE(static_cast<bool>(error)) << key.str();
    EXPECT_NE(
        llvm::toString(std::move(error)).find("target_lowering_unsupported"),
        std::string::npos)
        << key.str();
  }
}

//===----------------------------------------------------------------------===//
// C9: two targets, one abstract owner vocabulary
//===----------------------------------------------------------------------===//

/// The same abstract classes serve two machines whose hierarchies and widths
/// have nothing in common. Each profile spells its execution units its own way
/// and declares, in its own data, what those spellings refine; the canonical
/// dialect knows none of those words. This is the neutrality claim C9 makes:
/// generic group/worker/vector/matrix/transfer ownership is enough, and no new
/// ODS enum is needed to describe a second target.
TEST(GenericAcceleratorTarget, OwnerVocabularyIsTargetDataNotAnOdsEnum) {
  auto read = [](llvm::StringRef file) -> std::optional<std::string> {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
        llvm::MemoryBuffer::getFile(std::string(LLK_SOURCE_DIR) + "/machines/" +
                                    file);
    if (!buffer)
      return std::nullopt;
    return buffer.get()->getBuffer().str();
  };

  std::optional<std::string> cpuText = read("x86-avx2-v2.yaml");
  std::optional<std::string> accelText = read("generic-ai-accel-v2.yaml");
  ASSERT_TRUE(cpuText) << "the AVX2 profile must ship with the sources";
  ASSERT_TRUE(accelText)
      << "the accelerator profile must ship with the sources";

  llvm::Expected<mlir::llk::machine::MachineModel> cpu =
      mlir::llk::machine::parseMachineModel(*cpuText, "x86-avx2-v2.yaml");
  ASSERT_TRUE(static_cast<bool>(cpu)) << llvm::toString(cpu.takeError());
  llvm::Expected<mlir::llk::machine::MachineModel> accel =
      mlir::llk::machine::parseMachineModel(*accelText,
                                            "generic-ai-accel-v2.yaml");
  ASSERT_TRUE(static_cast<bool>(accel)) << llvm::toString(accel.takeError());

  // Both identify a worker, but through their own labels: the CPU's finer unit
  // is a `lane`, the accelerator's is a `core` or a `pe`.
  EXPECT_EQ(cpu->ownerClass("lane"), std::optional<std::string>("worker"));
  EXPECT_EQ(cpu->ownerClass("vector_engine"),
            std::optional<std::string>("vector"));
  EXPECT_EQ(accel->ownerClass("core"), std::optional<std::string>("worker"));
  EXPECT_EQ(accel->ownerClass("pe"), std::optional<std::string>("worker"));
  EXPECT_EQ(accel->ownerClass("pe_group"), std::optional<std::string>("group"));

  // The vocabulary is per target: `group` is a class both machines know,
  // `lane` is the CPU's own word and unknown to the accelerator.
  EXPECT_EQ(accel->ownerClass("worker"), std::optional<std::string>("worker"));
  EXPECT_FALSE(accel->ownerClass("lane").has_value());

  // And the dialect knows *none* of those labels -- only the abstract classes
  // are ODS vocabulary, which is exactly the boundary C9 restores.
  EXPECT_FALSE(mlir::micro::symbolizeOwner("lane").has_value());
  EXPECT_FALSE(mlir::micro::symbolizeOwner("pe_group").has_value());
  EXPECT_FALSE(mlir::micro::symbolizeOwner("vector_engine").has_value());
  EXPECT_TRUE(mlir::micro::symbolizeOwner("worker").has_value());
  EXPECT_TRUE(mlir::micro::symbolizeOwner("vector").has_value());
}
