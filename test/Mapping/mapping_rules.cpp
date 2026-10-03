//===- mapping_rules.cpp - LLKMap rule declarations (D4) -----------------===//

#include "LLK/Mapping/MappingRules.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/WorkloadGraph.h"
#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <string>

using namespace mlir::llk::mapping;

namespace {

constexpr llvm::StringLiteral kRules = R"llkmap(
// AVX2 mapping rules.
rule avx2.vector_add v1 {
  match micro.vector(kind = "add", element_type = f32);
  param VW in [4..8];
  require VW == machine.compute("vector_engine").lanes(element_type);
  require executor kind worker;
  require compute kind vector_engine;
  require layout operand0 satisfies avx2.blocked_2d;
  input "operand0";
  output "result";
  bundle "avx2.vector.add.f32";
  emit "avx2_vector_add";
  cost 4;
}

rule avx2.mma_bf16 {
  match micro.mma(element_type = bf16);
  require executor kind worker;
  require compute kind matrix_engine;
  bundle "avx2.mma.bf16";
  emit "avx2_mma_bf16";
}
)llkmap";

llvm::Expected<RuleRegistry> parse(llvm::StringRef text) {
  return parseRuleText(text, "<test>");
}

bool parses(llvm::StringRef text) {
  llvm::Expected<RuleRegistry> registry = parse(text);
  if (!registry) {
    llvm::consumeError(registry.takeError());
    return false;
  }
  return true;
}

std::string predicateText(const RuleDef &rule, size_t index) {
  if (index >= rule.predicates.size())
    return "<missing>";
  const RulePredicate &predicate = rule.predicates[index];
  if (const auto *integer = std::get_if<int64_t>(&predicate.value))
    return predicate.attribute + "=" + std::to_string(*integer);
  return predicate.attribute + "=" + std::get<std::string>(predicate.value);
}

} // namespace

TEST(RuleParse, ParsesRuleDeclarations) {
  llvm::Expected<RuleRegistry> registry = parse(kRules);
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  ASSERT_EQ(registry->all().size(), 2u);

  const RuleDef *rule = registry->find("avx2.vector_add");
  ASSERT_NE(rule, nullptr);
  EXPECT_EQ(rule->version, 1u);
  EXPECT_EQ(rule->matchOp, "micro.vector");
  ASSERT_EQ(rule->predicates.size(), 2u);
  EXPECT_EQ(predicateText(*rule, 0), "kind=add");
  EXPECT_EQ(predicateText(*rule, 1), "element_type=f32");
  EXPECT_EQ(rule->params.size(), 1u);
  EXPECT_EQ(rule->domains.size(), 1u);
  EXPECT_EQ(rule->constraints.size(), 1u);
  ASSERT_EQ(rule->ports.size(), 2u);
  EXPECT_TRUE(rule->ports[0].isInput);
  EXPECT_FALSE(rule->ports[1].isInput);
  EXPECT_EQ(rule->bundle, "avx2.vector.add.f32");
  EXPECT_EQ(rule->emitter, "avx2_vector_add");
  ASSERT_TRUE(rule->costLowerBound.has_value());
  EXPECT_EQ(*rule->costLowerBound, 4u);
}

TEST(RuleParse, ParsesKindAndLayoutRequirements) {
  llvm::Expected<RuleRegistry> registry = parse(kRules);
  ASSERT_TRUE(static_cast<bool>(registry));
  const RuleDef *rule = registry->find("avx2.vector_add");
  ASSERT_NE(rule, nullptr);
  ASSERT_EQ(rule->kindRequirements.size(), 2u);
  EXPECT_EQ(rule->kindRequirements[0].role, "executor");
  EXPECT_EQ(rule->kindRequirements[0].kind, "worker");
  EXPECT_EQ(rule->kindRequirements[1].role, "compute");
  EXPECT_EQ(rule->kindRequirements[1].kind, "vector_engine");
  ASSERT_EQ(rule->layoutRequirements.size(), 1u);
  EXPECT_EQ(rule->layoutRequirements[0].port, "operand0");
  EXPECT_EQ(rule->layoutRequirements[0].layoutId, "avx2.blocked_2d");
}

TEST(RuleParse, DefaultsVersionToOne) {
  llvm::Expected<RuleRegistry> registry = parse(kRules);
  ASSERT_TRUE(static_cast<bool>(registry));
  const RuleDef *rule = registry->find("avx2.mma_bf16");
  ASSERT_NE(rule, nullptr);
  EXPECT_EQ(rule->version, 1u);
  EXPECT_FALSE(rule->costLowerBound.has_value());
}

TEST(RuleParse, RejectsDuplicateRuleId) {
  EXPECT_FALSE(parses(R"llkmap(
rule a.one {
  match micro.vector();
  bundle "b";
  emit "e";
}
rule a.one {
  match micro.vector();
  bundle "b";
  emit "e";
}
)llkmap"));
}

TEST(RuleParse, RejectsMissingBundle) {
  EXPECT_FALSE(parses(R"llkmap(
rule a.one {
  match micro.vector();
  emit "e";
}
)llkmap"));
}

TEST(RuleParse, RejectsMissingMatch) {
  EXPECT_FALSE(parses(R"llkmap(
rule a.one {
  bundle "b";
  emit "e";
}
)llkmap"));
}

TEST(RuleParse, RejectsDuplicateEmit) {
  EXPECT_FALSE(parses(R"llkmap(
rule a.one {
  match micro.vector();
  bundle "b";
  emit "e";
  emit "f";
}
)llkmap"));
}

TEST(RuleParse, RejectsUnknownMicroOperation) {
  EXPECT_FALSE(parses(R"llkmap(
rule a.one {
  match micro.frobnicate();
  bundle "b";
  emit "e";
}
)llkmap"));
}

TEST(RuleParse, RejectsDuplicatePortNames) {
  EXPECT_FALSE(parses(R"llkmap(
rule a.one {
  match micro.vector();
  input "operand0";
  input "operand0";
  bundle "b";
  emit "e";
}
)llkmap"));
}

TEST(RuleParse, RejectsUndeclaredParameterInConstraint) {
  EXPECT_FALSE(parses(R"llkmap(
rule a.one {
  match micro.vector();
  require M > 0;
  bundle "b";
  emit "e";
}
)llkmap"));
}

//===----------------------------------------------------------------------===//
// Target validation
//===----------------------------------------------------------------------===//

namespace {

using mlir::llk::machine::ComputeNode;
using mlir::llk::machine::MachineModel;
using mlir::llk::machine::MemoryNode;

MachineModel ruleMachine() {
  MachineModel model;
  model.target = "rules";
  model.executors = {{"e0", "worker", std::nullopt, {}, 1, {}}};

  ComputeNode vector;
  vector.id = "vec.0";
  vector.kind = "vector_engine";
  vector.attachedTo = "e0";
  vector.shapes = {{8}};
  ComputeNode matrix;
  matrix.id = "mxu.0";
  matrix.kind = "matrix_engine";
  matrix.attachedTo = "e0";
  matrix.shapes = {{16, 16, 32}};
  model.computes = {vector, matrix};

  MemoryNode memory;
  memory.id = "sram.0";
  memory.kind = "sram";
  memory.visibleFrom = "e0";
  memory.capacityBytes = 4096;
  memory.alignmentBytes = 64;
  model.memories = {memory};
  return model;
}

constexpr llvm::StringLiteral kLayouts =
    "layout avx2.blocked_2d(int VW) { param VW in [4..8]; require rank == 2; }";

llvm::Expected<std::unique_ptr<MappingTarget>>
makeTarget(llvm::StringRef rules, std::vector<std::string> emitters) {
  llvm::Expected<LayoutRegistry> layouts = parseLayoutText(kLayouts, "<test>");
  if (!layouts)
    return layouts.takeError();
  llvm::Expected<RuleRegistry> ruleRegistry = parseRuleText(rules, "<test>");
  if (!ruleRegistry)
    return ruleRegistry.takeError();
  return std::make_unique<FileMappingTarget>(
      "test", ruleMachine(), std::move(*layouts), std::move(*ruleRegistry),
      std::move(emitters));
}

bool verifies(MappingTarget &target) {
  if (llvm::Error error = verifyMappingTarget(target)) {
    llvm::consumeError(std::move(error));
    return false;
  }
  return true;
}

constexpr llvm::StringLiteral kGoodRules = R"llkmap(
rule a.vector_add {
  match micro.vector(kind = "add");
  require executor kind worker;
  require compute kind vector_engine;
  require layout operand0 satisfies avx2.blocked_2d;
  bundle "b";
  emit "e1";
}
)llkmap";

} // namespace

TEST(MappingTarget, VerifiesGoodRules) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      makeTarget(kGoodRules, {"e1", "e2"});
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  EXPECT_TRUE(verifies(**target));
  EXPECT_TRUE((*target)->isKnownEmitter("e1"));
  EXPECT_FALSE((*target)->isKnownEmitter("nope"));
}

TEST(MappingTarget, RejectsUnknownLayoutId) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = makeTarget(R"llkmap(
rule a.bad {
  match micro.vector();
  require layout operand0 satisfies avx2.missing;
  bundle "b";
  emit "e1";
}
)llkmap",
                                                                     {"e1"});
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  EXPECT_FALSE(verifies(**target));
}

TEST(MappingTarget, RejectsKindTheMachineDoesNotOffer) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = makeTarget(R"llkmap(
rule a.bad {
  match micro.vector();
  require compute kind tensor_core;
  bundle "b";
  emit "e1";
}
)llkmap",
                                                                     {"e1"});
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  EXPECT_FALSE(verifies(**target));
}

TEST(MappingTarget, RejectsUnknownExecutorKind) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = makeTarget(R"llkmap(
rule a.bad {
  match micro.vector();
  require executor kind pe;
  bundle "b";
  emit "e1";
}
)llkmap",
                                                                     {"e1"});
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  EXPECT_FALSE(verifies(**target));
}

TEST(MappingTarget, RejectsUnknownEmitter) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      makeTarget(kGoodRules, {"different_emitter"});
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  EXPECT_FALSE(verifies(**target));
}

//===----------------------------------------------------------------------===//
// Shipped AVX2 rules
//===----------------------------------------------------------------------===//

namespace {
constexpr llvm::StringLiteral kShippedMachine =
    LLK_MACHINE_DIR "/x86-avx2-v2.yaml";
constexpr llvm::StringLiteral kShippedLayouts =
    LLK_MAPPING_DIR "/x86-avx2/layouts.llkmap";
constexpr llvm::StringLiteral kShippedRules =
    LLK_MAPPING_DIR "/x86-avx2/rules.llkmap";
constexpr llvm::StringLiteral kInvalidRules =
    LLK_MAPPING_DIR "/../test/Mapping/Inputs/invalid-rules.llkmap";

/// The shipped AVX2 rules against the emitter keys the AVX2 package actually
/// declares. Reading them from the package rather than repeating the list
/// here keeps this test from drifting when the target grows a rule.
llvm::Expected<std::unique_ptr<MappingTarget>> loadShippedAvx2() {
  std::vector<std::string> keys;
  for (llvm::StringLiteral key : mlir::llk::target::avx2::emitterKeys())
    keys.push_back(key.str());
  return loadMappingTarget("x86-avx2", kShippedMachine, kShippedLayouts,
                           kShippedRules, std::move(keys));
}
} // namespace

TEST(MappingTarget, LoadsShippedAvx2RulesAndVerifiesThem) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadShippedAvx2();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  EXPECT_EQ((*target)->name(), "x86-avx2");
  // Assert the rules that must be there rather than an exact count, so adding
  // a rule to the shipped set does not fail this test.
  EXPECT_NE((*target)->rules().find("avx2.vector_add"), nullptr);
  EXPECT_NE((*target)->rules().find("avx2.async_copy"), nullptr);
  EXPECT_FALSE((*target)->rules().all().empty());
  EXPECT_FALSE((*target)->machine().executors.empty());
  EXPECT_FALSE((*target)->layouts().all().empty());
  // `loadMappingTarget` verifies before returning; assert it directly too.
  EXPECT_TRUE(verifies(**target));
}

TEST(MappingTarget, ShippedRulesOnlyEmitDeclaredKeys) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadShippedAvx2();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  for (const RuleDef &rule : (*target)->rules().all())
    EXPECT_TRUE((*target)->isKnownEmitter(rule.emitter)) << rule.id;
}

TEST(MappingTarget, ShippedRulesResolveTheirLayouts) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target = loadShippedAvx2();
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  for (const RuleDef &rule : (*target)->rules().all())
    for (const RuleLayoutRequirement &requirement : rule.layoutRequirements)
      EXPECT_NE((*target)->layouts().find(requirement.layoutId), nullptr)
          << rule.id << " -> " << requirement.layoutId;
}

TEST(MappingTarget, RejectsTheInvalidRuleFixture) {
  llvm::Expected<RuleRegistry> registry = loadRuleFile(kInvalidRules);
  EXPECT_FALSE(static_cast<bool>(registry));
  if (!registry)
    llvm::consumeError(registry.takeError());
}

//===----------------------------------------------------------------------===//
// One-op rule matching
//===----------------------------------------------------------------------===//

namespace {

mlir::DictionaryAttr vectorAttributes(mlir::MLIRContext &context,
                                      llvm::StringRef op) {
  return mlir::DictionaryAttr::get(
      &context, {mlir::NamedAttribute(mlir::StringAttr::get(&context, "op"),
                                      mlir::StringAttr::get(&context, op))});
}

/// Builds a `micro.vector` workload node. Only the ids matter here: matching
/// reads the node's attributes, and the rule-to-candidate bridge reads its
/// ports positionally, so the fixture does not set port types.
WorkloadNode vectorNode(mlir::MLIRContext &context, llvm::StringRef op) {
  WorkloadNode node;
  node.id = 7;
  node.opName = "micro.vector";
  node.attributes = vectorAttributes(context, op);
  WorkloadPort inPort;
  inPort.value = 1;
  WorkloadPort outPort;
  outPort.value = 2;
  node.inputs.push_back(inPort);
  node.outputs.push_back(outPort);
  return node;
}

constexpr llvm::StringLiteral kMatchingRules = R"llkmap(
rule r.add {
  match micro.vector(op = "add");
  require executor kind worker;
  require memory kind sram;
  require layout operand0 satisfies avx2.blocked_2d;
  input "operand0";
  output "result";
  bundle "b.add";
  emit "e1";
  cost 9;
}
rule r.relu {
  match micro.vector(op = "relu");
  bundle "b.relu";
  emit "e2";
}
rule r.any {
  match micro.vector();
  bundle "b.any";
  emit "e3";
}
)llkmap";

std::vector<std::string> matchedIds(const WorkloadNode &node,
                                    const RuleRegistry &registry) {
  std::vector<std::string> ids;
  for (const RuleDef *rule : matchRules(node, registry))
    ids.push_back(rule->id);
  return ids;
}

} // namespace

TEST(RuleMatch, MatchesByOperationAndPredicates) {
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry = parse(kMatchingRules);
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());

  WorkloadNode add = vectorNode(context, "add");
  EXPECT_EQ(matchedIds(add, *registry),
            (std::vector<std::string>{"r.add", "r.any"}));

  WorkloadNode sub = vectorNode(context, "sub");
  EXPECT_EQ(matchedIds(sub, *registry), (std::vector<std::string>{"r.any"}));
}

TEST(RuleMatch, DoesNotMatchADifferentOperation) {
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry = parse(R"llkmap(
rule r.mma {
  match micro.mma();
  bundle "b";
  emit "e";
}
)llkmap");
  ASSERT_TRUE(static_cast<bool>(registry));
  WorkloadNode node = vectorNode(context, "add");
  EXPECT_TRUE(matchRules(node, *registry).empty());
}

TEST(RuleMatch, DoesNotMatchAMissingAttribute) {
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry = parse(kMatchingRules);
  ASSERT_TRUE(static_cast<bool>(registry));
  WorkloadNode node;
  node.id = 3;
  node.opName = "micro.vector"; // no attributes at all
  EXPECT_EQ(matchedIds(node, *registry), (std::vector<std::string>{"r.any"}));
}

TEST(RuleMatch, BuildsAMappingCandidate) {
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry = parse(kMatchingRules);
  ASSERT_TRUE(static_cast<bool>(registry));
  const RuleDef *rule = registry->find("r.add");
  ASSERT_NE(rule, nullptr);
  WorkloadNode node = vectorNode(context, "add");

  MappingCandidate candidate = toMappingCandidate(*rule, node);
  EXPECT_EQ(candidate.rule, "r.add");
  EXPECT_EQ(candidate.targetBundle, "b.add");
  ASSERT_EQ(candidate.coveredNodes.size(), 1u);
  EXPECT_EQ(candidate.coveredNodes[0], 7u);
  ASSERT_EQ(candidate.executorRequirements.size(), 1u);
  EXPECT_EQ(candidate.executorRequirements[0].capability, "worker");
  ASSERT_EQ(candidate.memoryRequirements.size(), 1u);
  EXPECT_EQ(candidate.memoryRequirements[0].kind, "sram");
  ASSERT_EQ(candidate.layoutRequirements.size(), 1u);
  EXPECT_EQ(candidate.layoutRequirements[0].layoutClass, "avx2.blocked_2d");
  EXPECT_DOUBLE_EQ(candidate.lowerBound.latencyCycles, 9.0);
}
