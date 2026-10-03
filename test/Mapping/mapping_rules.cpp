//===- mapping_rules.cpp - LLKMap rule declarations (D4) -----------------===//

#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Mapping/MappingRules.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/WorkloadGraph.h"
#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <optional>
#include <string>

using namespace mlir::llk::mapping;

namespace {

constexpr llvm::StringLiteral kRules = R"llkmap(
// AVX2 mapping rules.
rule avx2.vector_add v1 {
  match micro.vector(kind = "add", input[0].element_type = f32);
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
  match micro.mma(input[0].element_type = bf16);
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
  // The second predicate is the port element type, not an `element_type`
  // attribute: the fixture spells it explicitly so the two cannot drift.
  EXPECT_EQ(rule->predicates[1].kind, RulePredicateKind::ElementType);
  EXPECT_TRUE(rule->predicates[1].directionSet);
  EXPECT_TRUE(rule->predicates[1].isInput);
  EXPECT_EQ(rule->predicates[1].portIndex, 0);
  EXPECT_EQ(std::get<std::string>(rule->predicates[1].value), "f32");
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
  // The fixture matches the first operand's dtype, so the predicates cannot
  // silently revert to an `element_type` attribute.
  ASSERT_EQ(rule->predicates.size(), 1u);
  EXPECT_EQ(rule->predicates[0].kind, RulePredicateKind::ElementType);
  EXPECT_EQ(rule->predicates[0].portIndex, 0);
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

  std::optional<MappingCandidate> resolved =
      toMappingCandidate(*rule, node, MachineModel{}, LayoutContext{});
  ASSERT_TRUE(resolved.has_value());
  MappingCandidate candidate = std::move(*resolved);
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

//===----------------------------------------------------------------------===//
// `require <expr>` constraint evaluation (design §14.1)
//===----------------------------------------------------------------------===//

namespace {

/// A `micro.vector` node with one input and one output carrying `inputType` and
/// `outputType`. The port-data predicates read these types, and a rule's
/// constraints read them as `element_type`.
WorkloadNode typedVectorNode(mlir::Type inputType, mlir::Type outputType) {
  WorkloadNode node;
  node.id = 21;
  node.opName = "micro.vector";
  WorkloadPort input;
  input.value = 1;
  input.type = inputType;
  WorkloadPort output;
  output.value = 2;
  output.type = outputType;
  node.inputs.push_back(input);
  node.outputs.push_back(output);
  return node;
}

/// A machine whose `vector_engine` models `f32` at `f32Lanes` elements per
/// instruction, which is what `machine.compute("vector_engine").lanes(...)`
/// queries. Only the capability is needed: `lanesFor` reads `computes`.
MachineModel machineWithVectorLanes(int64_t f32Lanes) {
  MachineModel model;
  model.target = "lanes";
  ComputeNode vector;
  vector.id = "vec.0";
  vector.kind = "vector_engine";
  vector.lanes = {{"f32", f32Lanes}};
  model.computes = {vector};
  return model;
}

/// The shipped shape of the AVX2 vector rule's parameter: a vector width
/// derived from the machine's lane count for the matched element type.
constexpr llvm::StringLiteral kLaneRule = R"llkmap(
rule r.lanes {
  match micro.vector(input[0].element_type = f32);
  param VW in [4..8];
  require VW == machine.compute("vector_engine").lanes(element_type);
  input "operand0";
  output "result";
  bundle "b.lanes";
  emit "e";
}
)llkmap";

} // namespace

TEST(RuleMatch, EvaluatesRequireConstraintsAgainstTheMachine) {
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry = parse(kLaneRule);
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  const RuleDef *rule = registry->find("r.lanes");
  ASSERT_NE(rule, nullptr);

  mlir::Type f32 = mlir::Float32Type::get(&context);
  WorkloadNode node = typedVectorNode(f32, f32);

  LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "f32";

  // The machine models 8 f32 lanes, so VW must be 8 -- inside the declared
  // [4..8] domain, so the rule matches and the derived width is recorded.
  std::string reason;
  std::optional<MappingCandidate> satisfying = toMappingCandidate(
      *rule, node, machineWithVectorLanes(8), layoutContext, &reason);
  ASSERT_TRUE(satisfying.has_value()) << reason;
  auto derived = satisfying->resolvedParameters.find("VW");
  ASSERT_NE(derived, satisfying->resolvedParameters.end());
  EXPECT_EQ(std::get<int64_t>(derived->second), 8);

  // The machine models 16 lanes, which no value in [4..8] can equal: the rule
  // is a non-match, never a candidate.
  std::string rejectingReason;
  EXPECT_FALSE(toMappingCandidate(*rule, node, machineWithVectorLanes(16),
                                  layoutContext, &rejectingReason)
                   .has_value());
  EXPECT_FALSE(rejectingReason.empty());
}

TEST(RuleMatch, AConstraintOnFactsTheNodeCannotSupplyIsANonMatch) {
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry = parse(kLaneRule);
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  const RuleDef *rule = registry->find("r.lanes");
  ASSERT_NE(rule, nullptr);

  // A node that exposes no element type, and a context that names none either:
  // `element_type` cannot be resolved, so the constraint is unevaluable. It is
  // rejected as a non-match with a reason, never silently accepted.
  WorkloadNode bare;
  bare.opName = "micro.vector";
  std::string reason;
  EXPECT_FALSE(toMappingCandidate(*rule, bare, machineWithVectorLanes(8),
                                  LayoutContext{}, &reason)
                   .has_value());
  EXPECT_FALSE(reason.empty());
}

TEST(RuleMatch, ARuleWithoutConstraintsRecordsNoDerivedParameters) {
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry = parse(R"llkmap(
rule r.plain {
  match micro.vector(input[0].element_type = f32);
  param VW in [4..8];
  input "operand0";
  output "result";
  bundle "b.plain";
  emit "e";
}
)llkmap");
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  const RuleDef *rule = registry->find("r.plain");
  ASSERT_NE(rule, nullptr);

  mlir::Type f32 = mlir::Float32Type::get(&context);
  std::optional<MappingCandidate> candidate = toMappingCandidate(
      *rule, typedVectorNode(f32, f32), MachineModel{}, LayoutContext{});
  ASSERT_TRUE(candidate.has_value());
  // An unconstrained parameter is not "derived", so nothing is recorded for it.
  EXPECT_TRUE(candidate->resolvedParameters.empty());
}

TEST(RuleMatch, RecordsOnlyParametersAConstraintDerives) {
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry = parse(R"llkmap(
rule r.two {
  match micro.vector(input[0].element_type = f32);
  param VW in [4..8];
  param T in [1..2];
  require VW == machine.compute("vector_engine").lanes(element_type);
  input "operand0";
  output "result";
  bundle "b.two";
  emit "e";
}
)llkmap");
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  const RuleDef *rule = registry->find("r.two");
  ASSERT_NE(rule, nullptr);

  mlir::Type f32 = mlir::Float32Type::get(&context);
  LayoutContext layoutContext;
  layoutContext.elementType = "f32";
  std::optional<MappingCandidate> candidate =
      toMappingCandidate(*rule, typedVectorNode(f32, f32),
                         machineWithVectorLanes(8), layoutContext);
  ASSERT_TRUE(candidate.has_value());
  // VW is derived by the constraint; T is referenced by no constraint, so it is
  // not "derived" and must not enter the candidate's canonical content.
  ASSERT_EQ(candidate->resolvedParameters.count("VW"), 1u);
  EXPECT_EQ(std::get<int64_t>(candidate->resolvedParameters.lookup("VW")), 8);
  EXPECT_EQ(candidate->resolvedParameters.count("T"), 0u);
}

TEST(RuleMatch, ATruncatedConstraintSearchIsReportedNotSilentlyRejected) {
  mlir::MLIRContext context;
  // A domain larger than the assignment cap, with a constraint no assignment
  // satisfies: the search is cut off before it can prove the rule
  // unsatisfiable, so the non-match reports truncation rather than silently
  // concluding the rule does not apply.
  llvm::Expected<RuleRegistry> registry = parse(R"llkmap(
rule r.big {
  match micro.vector();
  param A in [1..400];
  param B in [1..400];
  require A == 999999;
  input "operand0";
  output "result";
  bundle "b.big";
  emit "e";
}
)llkmap");
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  const RuleDef *rule = registry->find("r.big");
  ASSERT_NE(rule, nullptr);

  mlir::Type f32 = mlir::Float32Type::get(&context);
  bool truncated = false;
  std::string reason;
  std::optional<MappingCandidate> candidate =
      toMappingCandidate(*rule, typedVectorNode(f32, f32), MachineModel{},
                         LayoutContext{}, &reason, &truncated);
  EXPECT_FALSE(candidate.has_value());
  EXPECT_TRUE(truncated);
  EXPECT_FALSE(reason.empty());
}

//===----------------------------------------------------------------------===//
// Port-data predicates: element type, shape, and affine map (design §14.1)
//===----------------------------------------------------------------------===//

namespace {

/// Renders the matched rule ids, in registry order.
llvm::Expected<RuleRegistry> parseOne(llvm::StringRef match) {
  std::string text =
      "rule r { match " + match.str() + "; bundle \"b\"; emit \"e\"; }";
  return parse(text);
}

} // namespace

TEST(RuleMatch, MatchesElementTypeOnAnInputPort) {
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry =
      parseOne("micro.vector(element_type = f32)");
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());

  mlir::Type f32 = mlir::Float32Type::get(&context);
  mlir::Type bf16 = mlir::BFloat16Type::get(&context);

  EXPECT_EQ(matchedIds(typedVectorNode(f32, f32), *registry),
            (std::vector<std::string>{"r"}));
  // A bf16 operand does not satisfy an f32 predicate.
  EXPECT_TRUE(matchRules(typedVectorNode(bf16, f32), *registry).empty());
}

TEST(RuleMatch, MatchesElementTypeOnANamedOutputPort) {
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry =
      parseOne("micro.vector(output[0].element_type = f32)");
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());

  mlir::Type f32 = mlir::Float32Type::get(&context);
  mlir::Type bf16 = mlir::BFloat16Type::get(&context);

  EXPECT_EQ(matchedIds(typedVectorNode(bf16, f32), *registry),
            (std::vector<std::string>{"r"}));
  EXPECT_TRUE(matchRules(typedVectorNode(f32, bf16), *registry).empty());
}

TEST(RuleMatch, MatchesAStaticShape) {
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry =
      parseOne("micro.vector(shape[0] = 64)");
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());

  mlir::Type f32 = mlir::Float32Type::get(&context);
  mlir::Type wide = mlir::RankedTensorType::get({64, 8}, f32);
  mlir::Type narrow = mlir::RankedTensorType::get({32, 8}, f32);

  EXPECT_EQ(matchedIds(typedVectorNode(f32, wide), *registry),
            (std::vector<std::string>{"r"}));
  EXPECT_TRUE(matchRules(typedVectorNode(f32, narrow), *registry).empty());
}

TEST(RuleMatch, MatchesAShapeOnANamedInputPort) {
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry =
      parseOne("micro.vector(input[0].shape[1] = 32)");
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());

  mlir::Type f32 = mlir::Float32Type::get(&context);
  mlir::Type matching = mlir::RankedTensorType::get({8, 32}, f32);
  mlir::Type other = mlir::RankedTensorType::get({8, 16}, f32);

  EXPECT_EQ(matchedIds(typedVectorNode(matching, f32), *registry),
            (std::vector<std::string>{"r"}));
  EXPECT_TRUE(matchRules(typedVectorNode(other, f32), *registry).empty());
}

TEST(RuleMatch, MatchesAnAffineMapByEqualityNotText) {
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry = parse(R"llkmap(
rule r.map {
  match micro.vector(access_map = (d0, d1) -> (d1, d0));
  bundle "b";
  emit "e";
}
rule r.simplified {
  match micro.vector(access_map = (d0, d1) -> (d1 + 0, d0));
  bundle "b";
  emit "e";
}
)llkmap");
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());

  mlir::Type f32 = mlir::Float32Type::get(&context);

  // A semantically equal map written differently still matches, and both rules
  // apply.
  WorkloadNode equivalent = typedVectorNode(f32, f32);
  equivalent.inputs[0].accessMap =
      mlir::AffineMap::get(2, 0,
                           {mlir::getAffineDimExpr(1, &context),
                            mlir::getAffineDimExpr(0, &context)},
                           &context);
  EXPECT_EQ(matchedIds(equivalent, *registry),
            (std::vector<std::string>{"r.map", "r.simplified"}));

  // A different map matches neither.
  WorkloadNode different = typedVectorNode(f32, f32);
  different.inputs[0].accessMap =
      mlir::AffineMap::get(2, 0,
                           {mlir::getAffineDimExpr(0, &context),
                            mlir::getAffineDimExpr(1, &context)},
                           &context);
  EXPECT_TRUE(matchRules(different, *registry).empty());
}

TEST(RuleMatch, PredicatesOnAbsentPortDataDoNotMatch) {
  mlir::MLIRContext context;
  mlir::Type f32 = mlir::Float32Type::get(&context);

  // No access map at all.
  llvm::Expected<RuleRegistry> mapRule =
      parseOne("micro.vector(access_map = (d0, d1) -> (d1, d0))");
  ASSERT_TRUE(static_cast<bool>(mapRule));
  EXPECT_TRUE(matchRules(typedVectorNode(f32, f32), *mapRule).empty());

  // An opaque type exposes no element type.
  llvm::Expected<RuleRegistry> elementRule =
      parseOne("micro.vector(element_type = f32)");
  ASSERT_TRUE(static_cast<bool>(elementRule));
  WorkloadNode opaque = typedVectorNode(mlir::NoneType::get(&context), f32);
  EXPECT_TRUE(matchRules(opaque, *elementRule).empty());

  // A node with no ports at all exposes nothing.
  WorkloadNode bare;
  bare.opName = "micro.vector";
  EXPECT_TRUE(matchRules(bare, *elementRule).empty());

  // A dynamic shape is not a static shape.
  llvm::Expected<RuleRegistry> shapeRule =
      parseOne("micro.vector(shape[0] = 64)");
  ASSERT_TRUE(static_cast<bool>(shapeRule));
  mlir::Type dynamic =
      mlir::RankedTensorType::get({mlir::ShapedType::kDynamic, 8}, f32);
  EXPECT_TRUE(matchRules(typedVectorNode(f32, dynamic), *shapeRule).empty());
}

TEST(RuleMatch, ReadsElementTypeAndShapeFromAMicroTile) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::micro::MicroDialect>();
  mlir::Type tile = mlir::parseType(
      "!micro.tile<16x32xbf16, memory = #micro.memory<sram>>", &context);
  ASSERT_TRUE(static_cast<bool>(tile));

  llvm::Expected<RuleRegistry> registry = parseOne(
      "micro.vector(input[0].element_type = bf16, input[0].shape[0] = 16)");
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  EXPECT_EQ(matchedIds(typedVectorNode(tile, tile), *registry),
            (std::vector<std::string>{"r"}));

  llvm::Expected<RuleRegistry> wrongType =
      parseOne("micro.vector(input[0].element_type = f32)");
  ASSERT_TRUE(static_cast<bool>(wrongType));
  EXPECT_TRUE(matchRules(typedVectorNode(tile, tile), *wrongType).empty());
}

TEST(RuleMatch, MatchesMmaOnItsFirstOperandDtype) {
  mlir::MLIRContext context;
  mlir::Type f32 = mlir::Float32Type::get(&context);
  mlir::Type bf16 = mlir::BFloat16Type::get(&context);

  llvm::Expected<RuleRegistry> registry =
      parseOne("micro.mma(input[0].element_type = bf16)");
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());

  // The shipped `avx2.mma_bf16` rule uses exactly this predicate: it reads the
  // lhs (first operand) dtype, not the operation's `input` attribute.
  WorkloadNode bf16Operands;
  bf16Operands.opName = "micro.mma";
  for (mlir::Type type : {bf16, bf16, f32}) {
    WorkloadPort port;
    port.type = type;
    bf16Operands.inputs.push_back(port);
  }
  EXPECT_EQ(matchedIds(bf16Operands, *registry),
            (std::vector<std::string>{"r"}));

  WorkloadNode f32Operands;
  f32Operands.opName = "micro.mma";
  for (mlir::Type type : {f32, f32, f32}) {
    WorkloadPort port;
    port.type = type;
    f32Operands.inputs.push_back(port);
  }
  EXPECT_TRUE(matchRules(f32Operands, *registry).empty());
}

TEST(RuleMatch, AnOutOfRangePortIndexDoesNotMatch) {
  mlir::MLIRContext context;
  mlir::Type f32 = mlir::Float32Type::get(&context);
  llvm::Expected<RuleRegistry> registry =
      parseOne("micro.vector(input[2].element_type = f32)");
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  // The node has one input, so input[2] is absent, never a wildcard.
  EXPECT_TRUE(matchRules(typedVectorNode(f32, f32), *registry).empty());
}

TEST(RuleParse, RejectsMalformedPortPredicates) {
  // `shape` needs a dimension index.
  EXPECT_FALSE(parses(R"llkmap(
rule a { match micro.vector(shape = 4); bundle "b"; emit "e"; }
)llkmap"));
  // A port subject admits only the port properties.
  EXPECT_FALSE(parses(R"llkmap(
rule a { match micro.vector(input[0].frob = 1); bundle "b"; emit "e"; }
)llkmap"));
  // A port index is required after the direction.
  EXPECT_FALSE(parses(R"llkmap(
rule a { match micro.vector(input[].element_type = f32); bundle "b"; emit "e"; }
)llkmap"));
  // An access map that references an undeclared dimension is rejected.
  EXPECT_FALSE(parses(R"llkmap(
rule a { match micro.vector(access_map = (d0) -> (d1)); bundle "b"; emit "e"; }
)llkmap"));
  // A non-affine access map is rejected at load time.
  EXPECT_FALSE(parses(R"llkmap(
rule a { match micro.vector(access_map = (d0, d1) -> (d0 * d1)); bundle "b"; emit "e"; }
)llkmap"));
  // An element type must be symbolic, a shape must be an integer: a malformed
  // value is a load-time error, never a bad-variant abort on first match.
  EXPECT_FALSE(parses(R"llkmap(
rule a { match micro.vector(element_type = 4); bundle "b"; emit "e"; }
)llkmap"));
  EXPECT_FALSE(parses(R"llkmap(
rule a { match micro.vector(shape[0] = f32); bundle "b"; emit "e"; }
)llkmap"));
}

TEST(RuleParse, ParsesPortPredicates) {
  llvm::Expected<RuleRegistry> registry = parse(R"llkmap(
rule r {
  match micro.vector(kind = "add", element_type = f32, shape[1] = 64,
                     output[0].element_type = bf16,
                     input[2].access_map = (d0, d1) -> (d0 + 1, d1));
  bundle "b";
  emit "e";
}
)llkmap");
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  const RuleDef *rule = registry->find("r");
  ASSERT_NE(rule, nullptr);
  ASSERT_EQ(rule->predicates.size(), 5u);

  EXPECT_EQ(rule->predicates[0].kind, RulePredicateKind::Attribute);
  EXPECT_EQ(rule->predicates[0].attribute, "kind");

  const RulePredicate &element = rule->predicates[1];
  EXPECT_EQ(element.kind, RulePredicateKind::ElementType);
  EXPECT_FALSE(element.directionSet);
  EXPECT_EQ(std::get<std::string>(element.value), "f32");

  const RulePredicate &shape = rule->predicates[2];
  EXPECT_EQ(shape.kind, RulePredicateKind::Shape);
  EXPECT_EQ(shape.dimension, 1);
  EXPECT_EQ(std::get<int64_t>(shape.value), 64);

  const RulePredicate &output = rule->predicates[3];
  EXPECT_EQ(output.kind, RulePredicateKind::ElementType);
  EXPECT_TRUE(output.directionSet);
  EXPECT_FALSE(output.isInput);
  EXPECT_EQ(output.portIndex, 0);

  const RulePredicate &map = rule->predicates[4];
  EXPECT_EQ(map.kind, RulePredicateKind::AccessMap);
  EXPECT_TRUE(map.directionSet);
  EXPECT_TRUE(map.isInput);
  EXPECT_EQ(map.portIndex, 2);
  ASSERT_TRUE(map.accessMap.has_value());
  EXPECT_EQ(map.accessMap->dims.size(), 2u);
}
