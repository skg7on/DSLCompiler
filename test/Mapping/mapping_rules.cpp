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

TEST(RuleParse, RejectsDuplicateKindRequirement) {
  // The same (role, kind) twice would collide in placement's binding map and
  // emit indistinguishable instances, so it is rejected at load.
  EXPECT_FALSE(parses(R"llkmap(
rule a.one {
  match micro.vector();
  require compute kind vector_engine;
  require compute kind vector_engine;
  bundle "b";
  emit "e";
}
)llkmap"));

  // A different kind under the same role, or the same kind under a different
  // role, is not a duplicate.
  EXPECT_TRUE(parses(R"llkmap(
rule a.two {
  match micro.vector();
  require compute kind vector_engine;
  require compute kind matrix_engine;
  require memory kind sram;
  bundle "b";
  emit "e";
}
)llkmap"));
}

TEST(RuleParse, ParsesMemoryRequirementsWithNamedOutputPorts) {
  llvm::Expected<RuleRegistry> registry = parse(R"llkmap(
rule r.two_ports {
  match micro.vector(op = "add");
  require executor kind worker;
  require memory output "small" kind sram;
  require memory output "large" kind dram;
  input "operand0";
  output "small";
  output "large";
  bundle "b";
  emit "e";
}
)llkmap");
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  const RuleDef *rule = registry->find("r.two_ports");
  ASSERT_NE(rule, nullptr);
  ASSERT_EQ(rule->kindRequirements.size(), 3u);
  // The bare executor requirement names no port; the two memory requirements
  // each carry the output port they govern.
  EXPECT_FALSE(rule->kindRequirements[0].port.has_value());
  ASSERT_TRUE(rule->kindRequirements[1].port.has_value());
  EXPECT_EQ(rule->kindRequirements[1].port->name, "small");
  EXPECT_FALSE(rule->kindRequirements[1].port->isInput);
  EXPECT_EQ(rule->kindRequirements[1].kind, "sram");
  ASSERT_TRUE(rule->kindRequirements[2].port.has_value());
  EXPECT_EQ(rule->kindRequirements[2].port->name, "large");
  EXPECT_EQ(rule->kindRequirements[2].kind, "dram");
}

TEST(RuleParse, RejectsANamedPortMemoryRequirementNamingAnUndeclaredPort) {
  EXPECT_FALSE(parses(R"llkmap(
rule r.bad {
  match micro.vector();
  require memory output "ghost" kind sram;
  output "small";
  bundle "b";
  emit "e";
}
)llkmap"));
}

TEST(RuleParse, RejectsANamedPortMemoryRequirementInTheWrongDirection) {
  // "small" is declared as an input, so the output subject does not resolve.
  EXPECT_FALSE(parses(R"llkmap(
rule r.bad {
  match micro.vector();
  require memory output "small" kind sram;
  input "small";
  bundle "b";
  emit "e";
}
)llkmap"));
}

TEST(RuleParse, RejectsAPortSubjectOnANonMemoryRequirement) {
  EXPECT_FALSE(parses(R"llkmap(
rule r.bad {
  match micro.vector();
  require executor output "result" kind worker;
  output "result";
  bundle "b";
  emit "e";
}
)llkmap"));
}

TEST(RuleParse, AllowsOneMemoryKindOnSeveralNamedPorts) {
  // Two requirements of one kind are distinct because they govern distinct
  // ports, so they are not a duplicate.
  EXPECT_TRUE(parses(R"llkmap(
rule r.ok {
  match micro.vector();
  require memory output "a" kind sram;
  require memory output "b" kind sram;
  output "a";
  output "b";
  bundle "b";
  emit "e";
}
)llkmap"));
  // The same port and kind twice would collide in the binding map.
  EXPECT_FALSE(parses(R"llkmap(
rule r.dup {
  match micro.vector();
  require memory output "a" kind sram;
  require memory output "a" kind sram;
  output "a";
  bundle "b";
  emit "e";
}
)llkmap"));
  // A bare requirement and a named one of the same kind overlap ambiguously.
  EXPECT_FALSE(parses(R"llkmap(
rule r.mixed {
  match micro.vector();
  require memory kind sram;
  require memory output "a" kind sram;
  output "a";
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
makeTargetWithLayouts(llvm::StringRef layoutsText, llvm::StringRef rules,
                      std::vector<std::string> emitters) {
  llvm::Expected<LayoutRegistry> layouts =
      parseLayoutText(layoutsText, "<test>");
  if (!layouts)
    return layouts.takeError();
  llvm::Expected<RuleRegistry> ruleRegistry = parseRuleText(rules, "<test>");
  if (!ruleRegistry)
    return ruleRegistry.takeError();
  return std::make_unique<FileMappingTarget>(
      "test", ruleMachine(), std::move(*layouts), std::move(*ruleRegistry),
      std::move(emitters));
}

llvm::Expected<std::unique_ptr<MappingTarget>>
makeTarget(llvm::StringRef rules, std::vector<std::string> emitters) {
  return makeTargetWithLayouts(kLayouts, rules, std::move(emitters));
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
  input "operand0";
  output "result";
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
  // `operand0` is declared, so the port-name check passes and the rejection
  // comes from the unknown layout id -- which is what this test pins.
  llvm::Expected<std::unique_ptr<MappingTarget>> target = makeTarget(R"llkmap(
rule a.bad {
  match micro.vector();
  require layout operand0 satisfies avx2.missing;
  input "operand0";
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

TEST(MappingTarget, RejectsARuleThatDeclaresNoPortsItGoverns) {
  // The rule predicates on the first input but declares no input port, so the
  // boundary it matched can never be wired. A rule's declared ports must cover
  // every port it references (design §14.4, "missing ports"); the rule parses,
  // but loading the target rejects it.
  llvm::Expected<std::unique_ptr<MappingTarget>> target = makeTarget(R"llkmap(
rule a.bad {
  match micro.vector(input[0].element_type = f32);
  bundle "b";
  emit "e1";
}
)llkmap",
                                                                     {"e1"});
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  EXPECT_FALSE(verifies(**target));
}

TEST(MappingTarget, RejectsALayoutRequirementOnAnUndeclaredPort) {
  // The rule declares `operand0` but requires a layout on `operand1`, which it
  // never declares: the requirement can never be discharged.
  llvm::Expected<std::unique_ptr<MappingTarget>> target = makeTarget(R"llkmap(
rule a.bad {
  match micro.vector();
  require layout operand1 satisfies avx2.blocked_2d;
  input "operand0";
  output "result";
  bundle "b";
  emit "e1";
}
)llkmap",
                                                                     {"e1"});
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  EXPECT_FALSE(verifies(**target));
}

TEST(MappingTarget, RejectsAnUnknownMachineQueryAtLoad) {
  // `machine.compute("bogus")` parses -- the grammar knows the call -- but the
  // machine declares no such compute kind. The registry must reject it at load
  // time, not defer to a failure when the rule is first matched.
  llvm::Expected<std::unique_ptr<MappingTarget>> target = makeTarget(R"llkmap(
rule a.bad {
  match micro.vector();
  require machine.compute("bogus").count > 0;
  bundle "b";
  emit "e1";
}
)llkmap",
                                                                     {"e1"});
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  EXPECT_FALSE(verifies(**target));
}

TEST(MappingTarget, RejectsAnUnknownMemoryQueryAtLoad) {
  // The memory half of the same check: `sram.9` is not a declared memory node.
  llvm::Expected<std::unique_ptr<MappingTarget>> target = makeTarget(R"llkmap(
rule a.bad {
  match micro.vector();
  require machine.memory("sram.9").capacity_bytes > 0;
  bundle "b";
  emit "e1";
}
)llkmap",
                                                                     {"e1"});
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  EXPECT_FALSE(verifies(**target));
}

TEST(MappingTarget, RejectsAnUnknownMachineQueryInALayoutAtLoad) {
  // A layout's `require` may query the machine too; an unknown subject there is
  // the same load-time failure, reported against the layout.
  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      makeTargetWithLayouts("layout a.bad_layout(int VW) { param VW in [4..8]; "
                            "require VW == machine.compute(\"bogus\").count; }",
                            R"llkmap(
rule a.ok {
  match micro.vector();
  bundle "b";
  emit "e1";
}
)llkmap",
                            {"e1"});
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
constexpr llvm::StringLiteral kShippedGenericRules =
    LLK_MAPPING_DIR "/generic-ai-accel/rules.llkmap";
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
// Printing and round-trip (design §25.3)
//===----------------------------------------------------------------------===//

namespace {

/// Prints every rule in `registry`, re-parses the text, and returns the
/// re-parsed content hash -- or `std::nullopt` when the printed text does not
/// parse. The hash folds every field that shapes a rule, so hash equality is
/// the structural round-trip comparison.
std::optional<uint64_t> roundTripRules(const RuleRegistry &registry) {
  std::string text;
  for (const RuleDef &def : registry.all())
    text += printRule(def);
  llvm::Expected<RuleRegistry> reparsed = parseRuleText(text, "<round-trip>");
  if (!reparsed) {
    llvm::consumeError(reparsed.takeError());
    return std::nullopt;
  }
  return reparsed->computeContentHash();
}

} // namespace

TEST(RulePrint, RoundTripsEveryConstruct) {
  // A version suffix, attribute/port predicates of every kind, integer and
  // symbolic domains, a machine-query and a quantified constraint, both
  // requirement kinds -- including a memory requirement bound to a named
  // output port -- ports, a bundle with typed parameters, and a cost.
  llvm::Expected<RuleRegistry> registry = parse(R"llkmap(
rule t.everything v3 {
  match micro.mma(op = "mul", input[0].element_type = bf16, output[1].shape[0] = 64, input[0].access_map = (d0, d1) -> (d1, d0), shape[2] = 8);
  param VW in [4..8];
  param policy in {"a", "b"};
  require VW == machine.compute("vector_engine").lanes(element_type);
  require forall d in dimensions : d >= 0;
  require executor kind worker;
  require compute kind vector_engine;
  require layout operand0 satisfies avx2.blocked_2d;
  require memory output "result" kind sram;
  input "lhs";
  input "rhs";
  output "result";
  bundle "t.bundle" { alpha = 1, beta = "two" };
  emit "t_emit";
  cost 9;
}
)llkmap");
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  std::optional<uint64_t> reparsed = roundTripRules(*registry);
  ASSERT_TRUE(reparsed.has_value())
      << "printed rule did not parse back:\n"
      << printRule(*registry->find("t.everything"));
  EXPECT_EQ(*reparsed, registry->computeContentHash());
}

TEST(RulePrint, RoundTripsAVersionlessRule) {
  llvm::Expected<RuleRegistry> registry = parse(R"llkmap(
rule t.plain {
  match micro.async_copy();
  bundle "t.copy";
  emit "t_copy";
}
)llkmap");
  ASSERT_TRUE(static_cast<bool>(registry));
  std::optional<uint64_t> reparsed = roundTripRules(*registry);
  ASSERT_TRUE(reparsed.has_value());
  EXPECT_EQ(*reparsed, registry->computeContentHash());
}

TEST(RulePrint, RoundTripsShippedFiles) {
  for (llvm::StringLiteral path : {kShippedRules, kShippedGenericRules}) {
    llvm::Expected<RuleRegistry> registry = loadRuleFile(path);
    ASSERT_TRUE(static_cast<bool>(registry))
        << path.str() << ": " << llvm::toString(registry.takeError());
    std::optional<uint64_t> reparsed = roundTripRules(*registry);
    ASSERT_TRUE(reparsed.has_value()) << path.str();
    EXPECT_EQ(*reparsed, registry->computeContentHash()) << path.str();
  }
}

TEST(RulePrint, PrintingIsDeterministic) {
  llvm::Expected<RuleRegistry> registry = loadRuleFile(kShippedRules);
  ASSERT_TRUE(static_cast<bool>(registry));
  for (const RuleDef &def : registry->all())
    EXPECT_EQ(printRule(def), printRule(def)) << def.id;
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

//===----------------------------------------------------------------------===//
// Canonical ordering (design §22.1)
//===----------------------------------------------------------------------===//

namespace {

/// Three rules in a deliberately non-canonical file order: the registry must
/// expose them by rule id, never in declaration order.
constexpr llvm::StringLiteral kDeclaredOutOfOrder = R"llkmap(
rule r.zeta {
  match micro.vector();
  bundle "b.zeta";
  emit "e";
}
rule r.alpha {
  match micro.vector();
  bundle "b.alpha";
  emit "e";
}
rule r.middle {
  match micro.vector();
  bundle "b.middle";
  emit "e";
}
)llkmap";

/// The same rules declared in the reverse order, so two registries built from
/// the same content can be compared.
constexpr llvm::StringLiteral kReversedDeclaration = R"llkmap(
rule r.middle {
  match micro.vector();
  bundle "b.middle";
  emit "e";
}
rule r.alpha {
  match micro.vector();
  bundle "b.alpha";
  emit "e";
}
rule r.zeta {
  match micro.vector();
  bundle "b.zeta";
  emit "e";
}
)llkmap";

std::vector<std::string> ruleIds(const RuleRegistry &registry) {
  std::vector<std::string> ids;
  for (const RuleDef &rule : registry.all())
    ids.push_back(rule.id);
  return ids;
}

} // namespace

TEST(RuleOrdering, RegistryIterationIsByRuleId) {
  llvm::Expected<RuleRegistry> registry = parse(kDeclaredOutOfOrder);
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  // `all()` is the deterministic output order, so it is by rule id, not file
  // order.
  EXPECT_EQ(ruleIds(*registry),
            (std::vector<std::string>{"r.alpha", "r.middle", "r.zeta"}));
  // Lookup still resolves every rule after sorting by id.
  EXPECT_NE(registry->find("r.alpha"), nullptr);
  EXPECT_NE(registry->find("r.middle"), nullptr);
  EXPECT_NE(registry->find("r.zeta"), nullptr);
  EXPECT_EQ(registry->find("r.missing"), nullptr);
}

TEST(RuleOrdering, MatchRulesAreCanonicallyOrderedRegardlessOfDeclaration) {
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry = parse(kDeclaredOutOfOrder);
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  WorkloadNode node = vectorNode(context, "add");

  // No rule has a predicate, so all three match. A single-node match fixes the
  // covered-node sequence, and one rule yields at most one candidate for it, so
  // §22.1's (covered-node sequence, rule id, id) order is rule id order here.
  EXPECT_EQ(matchedIds(node, *registry),
            (std::vector<std::string>{"r.alpha", "r.middle", "r.zeta"}));

  // Rebuilding the same content in a different declaration order yields the
  // identical registry and match ordering.
  llvm::Expected<RuleRegistry> reversed = parse(kReversedDeclaration);
  ASSERT_TRUE(static_cast<bool>(reversed))
      << llvm::toString(reversed.takeError());
  EXPECT_EQ(ruleIds(*reversed), ruleIds(*registry));
  EXPECT_EQ(matchedIds(node, *reversed), matchedIds(node, *registry));
}

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
  EXPECT_EQ(candidate.bundle.name, "b.add");
  EXPECT_EQ(candidate.bundle.emitterKey, "e1");
  EXPECT_FALSE(candidate.bundle.parameters);
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

TEST(RuleMatch, ResolvesALayoutRequirementAgainstItsOperandType) {
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry = parse(kMatchingRules);
  ASSERT_TRUE(static_cast<bool>(registry));
  const RuleDef *rule = registry->find("r.add");
  ASSERT_NE(rule, nullptr);

  // `kMatchingRules` requires `layout operand0 satisfies avx2.blocked_2d`. Give
  // the operand an f32 type while the fallback context names a different value
  // (bf16): the requirement is on operand0, so it must be solved against the
  // operand's f32, not the fallback. Solving every requirement against the
  // graph's first value is what made an f32 vector op unplaceable in a bf16
  // tile program.
  WorkloadNode node;
  node.id = 7;
  node.opName = "micro.vector";
  node.attributes = vectorAttributes(context, "add");
  WorkloadPort inPort;
  inPort.value = 1;
  inPort.type =
      mlir::RankedTensorType::get({8, 32}, mlir::Float32Type::get(&context));
  WorkloadPort outPort;
  outPort.value = 2;
  outPort.type =
      mlir::RankedTensorType::get({8, 32}, mlir::BFloat16Type::get(&context));
  node.inputs.push_back(inPort);
  node.outputs.push_back(outPort);

  LayoutContext fallback;
  fallback.rank = 2;
  fallback.elementType = "bf16";
  std::optional<MappingCandidate> resolved =
      toMappingCandidate(*rule, node, MachineModel{}, fallback);
  ASSERT_TRUE(resolved.has_value());
  ASSERT_EQ(resolved->layoutRequirements.size(), 1u);
  EXPECT_EQ(resolved->layoutRequirements[0].elementType, "f32");
  EXPECT_EQ(resolved->layoutRequirements[0].rank, 2);
}

//===----------------------------------------------------------------------===//
// Typed target bundles (design §14.3)
//===----------------------------------------------------------------------===//

TEST(RuleParse, ParsesBundleParameters) {
  llvm::Expected<RuleRegistry> registry = parse(R"llkmap(
rule r.p {
  match micro.vector();
  bundle "b" { tile_m = 8, layout = blocked_2d, note = "wide" };
  emit "e";
}
)llkmap");
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  const RuleDef *rule = registry->find("r.p");
  ASSERT_NE(rule, nullptr);
  // Stored sorted by name, independent of declaration order.
  ASSERT_EQ(rule->bundleParameters.size(), 3u);
  EXPECT_EQ(rule->bundleParameters[0].first, "layout");
  EXPECT_EQ(std::get<std::string>(rule->bundleParameters[0].second),
            "blocked_2d");
  EXPECT_EQ(rule->bundleParameters[1].first, "note");
  EXPECT_EQ(std::get<std::string>(rule->bundleParameters[1].second), "wide");
  EXPECT_EQ(rule->bundleParameters[2].first, "tile_m");
  EXPECT_EQ(std::get<int64_t>(rule->bundleParameters[2].second), 8);
}

TEST(RuleParse, RejectsDuplicateBundleParameter) {
  EXPECT_FALSE(parses(R"llkmap(
rule r.p {
  match micro.vector();
  bundle "b" { tile_m = 8, tile_m = 16 };
  emit "e";
}
)llkmap"));
}

TEST(RuleParse, RejectsMalformedBundleParameter) {
  EXPECT_FALSE(parses(R"llkmap(
rule r.p {
  match micro.vector();
  bundle "b" { tile_m = };
  emit "e";
}
)llkmap"));
}

TEST(RuleBundle, CarriesDeclaredParametersToTheCandidate) {
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry = parse(R"llkmap(
rule r.p {
  match micro.vector(op = "add");
  input "operand0";
  output "result";
  bundle "avx2.vector.add.f32" { tile_m = 8, layout = "blocked_2d" };
  emit "e1";
}
)llkmap");
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  const RuleDef *rule = registry->find("r.p");
  ASSERT_NE(rule, nullptr);
  WorkloadNode node = vectorNode(context, "add");

  std::optional<MappingCandidate> resolved =
      toMappingCandidate(*rule, node, MachineModel{}, LayoutContext{});
  ASSERT_TRUE(resolved.has_value());
  EXPECT_EQ(resolved->bundle.name, "avx2.vector.add.f32");
  EXPECT_EQ(resolved->bundle.emitterKey, "e1");
  ASSERT_TRUE(resolved->bundle.parameters);
  auto tileM = resolved->bundle.parameters.getAs<mlir::IntegerAttr>("tile_m");
  ASSERT_TRUE(tileM);
  EXPECT_EQ(tileM.getInt(), 8);
  auto layout = resolved->bundle.parameters.getAs<mlir::StringAttr>("layout");
  ASSERT_TRUE(layout);
  EXPECT_EQ(layout.getValue(), "blocked_2d");
}

TEST(RuleBundle, FailsLoudlyWhenTheNodeCannotTypeItsParameters) {
  llvm::Expected<RuleRegistry> registry = parse(R"llkmap(
rule r.p {
  match micro.vector();
  bundle "b" { tile_m = 8 };
  emit "e";
}
)llkmap");
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  const RuleDef *rule = registry->find("r.p");
  ASSERT_NE(rule, nullptr);

  // A node with neither attributes nor typed ports has no MLIR context, so the
  // rule's declared parameters cannot be typed. The bridge must fail loudly
  // rather than silently produce a parameterless bundle.
  WorkloadNode node;
  node.id = 5;
  node.opName = "micro.vector";
  EXPECT_DEATH(
      {
        (void)toMappingCandidate(*rule, node, MachineModel{}, LayoutContext{});
      },
      "no MLIR context");
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

/// A search-space binding for `toMappingCandidate`'s trailing `pinned`
/// argument: the map `CoveringSearch` copies off `SearchBinding::values`.
llvm::StringMap<SearchValue> pinnedMap(
    std::initializer_list<std::pair<llvm::StringRef, SearchValue>> entries) {
  llvm::StringMap<SearchValue> map;
  for (const auto &entry : entries)
    map[entry.first] = entry.second;
  return map;
}

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

TEST(RuleMatch, AQuantifiedRequireConstraintDecidesTheMatch) {
  mlir::MLIRContext context;
  mlir::Type f32 = mlir::Float32Type::get(&context);
  WorkloadNode node = typedVectorNode(f32, f32);
  MachineModel machine = machineWithVectorLanes(8);

  llvm::StringLiteral holding = R"llkmap(
rule r.quant {
  match micro.vector(input[0].element_type = f32);
  param VW in [4..8];
  require forall v in domain(VW) : v >= 4;
  input "operand0";
  output "result";
  bundle "b.quant";
  emit "e";
}
)llkmap";
  llvm::Expected<RuleRegistry> registry = parse(holding);
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  const RuleDef *rule = registry->find("r.quant");
  ASSERT_NE(rule, nullptr);

  std::string reason;
  std::optional<MappingCandidate> match =
      toMappingCandidate(*rule, node, machine, {}, &reason);
  ASSERT_TRUE(match.has_value()) << reason;
  // `domain(VW)` names VW's domain, not its value, so the rule derives no
  // parameter value for its bundle.
  EXPECT_TRUE(match->resolvedParameters.empty());

  llvm::StringLiteral failing = R"llkmap(
rule r.quant {
  match micro.vector(input[0].element_type = f32);
  param VW in [4..8];
  require exists v in domain(VW) : v > 8;
  input "operand0";
  output "result";
  bundle "b.quant";
  emit "e";
}
)llkmap";
  llvm::Expected<RuleRegistry> failingRegistry = parse(failing);
  ASSERT_TRUE(static_cast<bool>(failingRegistry));
  const RuleDef *failingRule = failingRegistry->find("r.quant");
  ASSERT_NE(failingRule, nullptr);
  std::string failingReason;
  EXPECT_FALSE(
      toMappingCandidate(*failingRule, node, machine, {}, &failingReason)
          .has_value());
}

TEST(RuleMatch, ATruncatedQuantifierUnderNegationIsNotAMatch) {
  // The domain exceeds the rule's quantifier budget, so the `forall` is
  // undecided and `!undecided` reads as satisfied. The rule must fail closed
  // and report truncation, never match on an undecided constraint.
  llvm::StringLiteral text = R"llkmap(
rule r.trunc {
  match micro.vector(input[0].element_type = f32);
  param VW in [1..200000];
  require !(forall v in domain(VW) : v >= 1);
  input "operand0";
  output "result";
  bundle "b.trunc";
  emit "e";
}
)llkmap";
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry = parse(text);
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  const RuleDef *rule = registry->find("r.trunc");
  ASSERT_NE(rule, nullptr);

  mlir::Type f32 = mlir::Float32Type::get(&context);
  WorkloadNode node = typedVectorNode(f32, f32);

  std::string reason;
  bool truncated = false;
  std::optional<MappingCandidate> candidate = toMappingCandidate(
      *rule, node, machineWithVectorLanes(8), {}, &reason, &truncated);
  EXPECT_FALSE(candidate.has_value());
  EXPECT_TRUE(truncated);
  EXPECT_FALSE(reason.empty());
}

TEST(RuleMatch, ABoundVariableShadowingAParameterIsNotDerived) {
  // The bound `VW` shadows the declared parameter `VW`. Only the domain
  // selector mentions the parameter, so the rule derives no value for its
  // bundle -- a body reference to the bound variable must not be recorded as
  // the parameter.
  llvm::StringLiteral text = R"llkmap(
rule r.shadow {
  match micro.vector(input[0].element_type = f32);
  param VW in [4..8];
  require forall VW in domain(VW) : VW >= 4;
  input "operand0";
  output "result";
  bundle "b.shadow";
  emit "e";
}
)llkmap";
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry = parse(text);
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  const RuleDef *rule = registry->find("r.shadow");
  ASSERT_NE(rule, nullptr);

  mlir::Type f32 = mlir::Float32Type::get(&context);
  WorkloadNode node = typedVectorNode(f32, f32);

  std::string reason;
  std::optional<MappingCandidate> candidate =
      toMappingCandidate(*rule, node, machineWithVectorLanes(8), {}, &reason);
  ASSERT_TRUE(candidate.has_value()) << reason;
  EXPECT_TRUE(candidate->resolvedParameters.empty());
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

//===----------------------------------------------------------------------===//
// A binding constrains rule parameter resolution (phase-4 T2)
//===----------------------------------------------------------------------===//

TEST(RuleMatch, ABindingPinsAParameterItNames) {
  mlir::MLIRContext context;
  // Every value in [4..8] satisfies `VW >= 4`, so the enumeration order -- not
  // the constraint -- decides which assignment is derived today: the first
  // declared value, 4.
  llvm::Expected<RuleRegistry> registry = parse(R"llkmap(
rule r.pin {
  match micro.vector(input[0].element_type = f32);
  param VW in [4..8];
  require VW >= 4;
  input "operand0";
  output "result";
  bundle "b.pin";
  emit "e";
}
)llkmap");
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  const RuleDef *rule = registry->find("r.pin");
  ASSERT_NE(rule, nullptr);

  mlir::Type f32 = mlir::Float32Type::get(&context);
  WorkloadNode node = typedVectorNode(f32, f32);

  std::optional<MappingCandidate> free =
      toMappingCandidate(*rule, node, MachineModel{}, LayoutContext{});
  ASSERT_TRUE(free.has_value());
  EXPECT_EQ(std::get<int64_t>(free->resolvedParameters.lookup("VW")), 4);

  // Pinned to 8, the rule resolves to 8 even though 4 also satisfies the
  // constraint: the binding chooses the value, not the enumeration order.
  llvm::StringMap<SearchValue> pinned = pinnedMap({{"VW", int64_t{8}}});
  std::string reason;
  std::optional<MappingCandidate> bound = toMappingCandidate(
      *rule, node, MachineModel{}, LayoutContext{}, &reason, nullptr, &pinned);
  ASSERT_TRUE(bound.has_value()) << reason;
  ASSERT_EQ(bound->resolvedParameters.count("VW"), 1u);
  EXPECT_EQ(std::get<int64_t>(bound->resolvedParameters.lookup("VW")), 8);
}

TEST(RuleMatch, ABindingWhosePinnedValueFailsTheRequireYieldsNoCandidate) {
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry = parse(R"llkmap(
rule r.pin {
  match micro.vector(input[0].element_type = f32);
  param VW in [4..8];
  require VW == 8;
  input "operand0";
  output "result";
  bundle "b.pin";
  emit "e";
}
)llkmap");
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  const RuleDef *rule = registry->find("r.pin");
  ASSERT_NE(rule, nullptr);

  mlir::Type f32 = mlir::Float32Type::get(&context);
  WorkloadNode node = typedVectorNode(f32, f32);

  // Unpinned, the enumeration finds the one satisfying value, 8.
  EXPECT_TRUE(toMappingCandidate(*rule, node, MachineModel{}, LayoutContext{})
                  .has_value());

  // Pinned to 4 -- a value inside the declared domain -- the only permitted
  // assignment fails `VW == 8`: a non-match, never an error.
  llvm::StringMap<SearchValue> pinned = pinnedMap({{"VW", int64_t{4}}});
  std::string reason;
  EXPECT_FALSE(toMappingCandidate(*rule, node, MachineModel{}, LayoutContext{},
                                  &reason, nullptr, &pinned)
                   .has_value());
  EXPECT_FALSE(reason.empty());
}

TEST(RuleMatch, ABindingOutsideTheDeclaredDomainIsANonMatch) {
  mlir::MLIRContext context;
  // `VW >= 4` would hold for 16, so a pinned value outside [4..8] must not slip
  // through on the strength of its constraint: the declared domain stays the
  // admissible set, and a value it does not contain leaves nothing to try.
  llvm::Expected<RuleRegistry> registry = parse(R"llkmap(
rule r.pin {
  match micro.vector(input[0].element_type = f32);
  param VW in [4..8];
  require VW >= 4;
  input "operand0";
  output "result";
  bundle "b.pin";
  emit "e";
}
)llkmap");
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  const RuleDef *rule = registry->find("r.pin");
  ASSERT_NE(rule, nullptr);

  mlir::Type f32 = mlir::Float32Type::get(&context);
  WorkloadNode node = typedVectorNode(f32, f32);

  llvm::StringMap<SearchValue> pinned = pinnedMap({{"VW", int64_t{16}}});
  std::string reason;
  EXPECT_FALSE(toMappingCandidate(*rule, node, MachineModel{}, LayoutContext{},
                                  &reason, nullptr, &pinned)
                   .has_value());
  EXPECT_FALSE(reason.empty());
}

TEST(RuleMatch, ABindingLeavesUnpinnedParametersEnumerated) {
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry = parse(R"llkmap(
rule r.two {
  match micro.vector(input[0].element_type = f32);
  param VW in [4..8];
  param T in [1..3];
  require VW == 8;
  require T == 3;
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
  WorkloadNode node = typedVectorNode(f32, f32);

  // Pinning VW leaves T unpinned, so T is still enumerated over [1..3] -- past
  // its first value -- and both derived values are recorded.
  llvm::StringMap<SearchValue> pinned = pinnedMap({{"VW", int64_t{8}}});
  std::string reason;
  std::optional<MappingCandidate> candidate = toMappingCandidate(
      *rule, node, MachineModel{}, LayoutContext{}, &reason, nullptr, &pinned);
  ASSERT_TRUE(candidate.has_value()) << reason;
  ASSERT_EQ(candidate->resolvedParameters.count("VW"), 1u);
  ASSERT_EQ(candidate->resolvedParameters.count("T"), 1u);
  EXPECT_EQ(std::get<int64_t>(candidate->resolvedParameters.lookup("VW")), 8);
  EXPECT_EQ(std::get<int64_t>(candidate->resolvedParameters.lookup("T")), 3);
}

TEST(RuleMatch, ABindingNameTheRuleDoesNotDeclareIsIgnored) {
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry = parse(R"llkmap(
rule r.one {
  match micro.vector(input[0].element_type = f32);
  param VW in [4..8];
  require VW >= 4;
  input "operand0";
  output "result";
  bundle "b.one";
  emit "e";
}
)llkmap");
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  const RuleDef *rule = registry->find("r.one");
  ASSERT_NE(rule, nullptr);

  mlir::Type f32 = mlir::Float32Type::get(&context);
  WorkloadNode node = typedVectorNode(f32, f32);

  // The binding names a parameter this rule never declares, so it constrains
  // nothing here and VW still enumerates to its first value, 4.
  llvm::StringMap<SearchValue> pinned = pinnedMap({{"BM", int64_t{64}}});
  std::optional<MappingCandidate> candidate = toMappingCandidate(
      *rule, node, MachineModel{}, LayoutContext{}, nullptr, nullptr, &pinned);
  ASSERT_TRUE(candidate.has_value());
  EXPECT_EQ(std::get<int64_t>(candidate->resolvedParameters.lookup("VW")), 4);
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

// A binding that names the port a layout governs obliges only that port: the
// strategy can constrain `operand0` and `operand1` differently, which a single
// role-less layout value cannot express.
TEST(RuleMatch, APerRoleBindingObligesOnlyItsOwnPort) {
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry = parse(R"llkmap(
rule r.two_ports {
  match micro.vector(input[0].element_type = f32);
  input "operand0";
  input "operand1";
  output "result";
  require layout operand0 satisfies t.plain;
  require layout operand1 satisfies t.blocked;
  bundle "b";
  emit "e";
}
)llkmap");
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  const RuleDef *rule = registry->find("r.two_ports");
  ASSERT_NE(rule, nullptr);
  mlir::Type f32 = mlir::Float32Type::get(&context);

  auto matches = [&](llvm::StringMap<std::string> bound) {
    return toMappingCandidate(*rule, typedVectorNode(f32, f32), MachineModel{},
                              LayoutContext{}, nullptr, nullptr, nullptr,
                              &bound)
        .has_value();
  };

  // Each port bound to what the rule offers there: a match.
  EXPECT_TRUE(matches({{"operand0", "t.plain"}, {"operand1", "t.blocked"}}));
  // `operand0` bound to the value `operand1` requires: no match.
  EXPECT_FALSE(matches({{"operand0", "t.blocked"}}));
  // A role the rule does not mention carries no obligation, so a rule that
  // never mentions a port is not vetoed by a binding for it.
  EXPECT_TRUE(matches({{"operand2", "t.blocked"}}));
  // A value bound with no role governs the axis as a whole: the rule offers it
  // somewhere, so it matches.
  EXPECT_TRUE(matches({{"", "t.blocked"}}));
}

//===----------------------------------------------------------------------===//
// Full rule-legality verification (A3)
//===----------------------------------------------------------------------===//

namespace {

/// A machine offering one `worker` executor with a `vector_engine` attached
/// (f32 lanes), a second worker with no capability, and a `lane` executor.
/// Enough for verification's executor-kind, compute and parameter checks.
MachineModel verificationMachine() {
  MachineModel model;
  model.target = "verify";
  mlir::llk::machine::ExecutorNode worker;
  worker.id = "worker.0";
  worker.kind = "worker";
  mlir::llk::machine::ExecutorNode bare;
  bare.id = "worker.bare";
  bare.kind = "worker";
  mlir::llk::machine::ExecutorNode lane;
  lane.id = "lane.0";
  lane.kind = "lane";
  model.executors = {worker, bare, lane};
  mlir::llk::machine::ComputeNode vector;
  vector.id = "vec.0";
  vector.kind = "vector_engine";
  vector.attachedTo = "worker.0";
  vector.lanes = {{"f32", 8}};
  model.computes = {vector};
  return model;
}

/// A rule exercising every check: an element-type predicate, two
/// machine-derived parameters, an executor kind, and a compute capability.
constexpr llvm::StringLiteral kVerificationRule = R"llkmap(
rule r.verify {
  match micro.vector(op = "add", input[0].element_type = f32);
  param VW in [4..8];
  param U in [1..1];
  require VW == machine.compute("vector_engine").lanes(element_type);
  require U == 1;
  require executor kind worker;
  require compute kind vector_engine;
  input "operand0";
  output "result";
  bundle "b";
  emit "e";
}
)llkmap";

const RuleDef *verificationRule() {
  static llvm::Expected<RuleRegistry> registry = parse(kVerificationRule);
  if (!registry)
    return nullptr;
  return registry->find("r.verify");
}

RecordedRuleSelection recordedOn(llvm::StringRef executor) {
  RecordedRuleSelection recorded;
  recorded.executor = executor.str();
  return recorded;
}

/// A `micro.vector "add"` node carrying the `op` attribute the rule's predicate
/// reads, and typed ports the element-type predicate and constraints read.
WorkloadNode typedAddNode(mlir::MLIRContext &context, mlir::Type inputType,
                          mlir::Type outputType) {
  WorkloadNode node = typedVectorNode(inputType, outputType);
  node.attributes = vectorAttributes(context, "add");
  return node;
}

std::string verifyText(const RuleDef &rule, const WorkloadNode &node,
                       const MachineModel &machine,
                       const RecordedRuleSelection &recorded) {
  return llvm::toString(
      verifyRuleSelection(rule, node, machine, recorded, "op"));
}

} // namespace

TEST(RuleVerify, AcceptsALegalRecordedSelection) {
  const RuleDef *rule = verificationRule();
  ASSERT_NE(rule, nullptr);
  mlir::MLIRContext context;
  mlir::Type f32 = mlir::Float32Type::get(&context);
  RecordedRuleSelection recorded;
  recorded.executor = "worker.0";
  recorded.parameters["VW"] = int64_t(8);
  recorded.parameters["U"] = int64_t(1);
  std::string error = verifyText(*rule, typedAddNode(context, f32, f32),
                                 verificationMachine(), recorded);
  EXPECT_TRUE(error.empty()) << error;
}

TEST(RuleVerify, RejectsADifferentElementType) {
  const RuleDef *rule = verificationRule();
  ASSERT_NE(rule, nullptr);
  mlir::MLIRContext context;
  mlir::Type bf16 = mlir::BFloat16Type::get(&context);
  std::string error = verifyText(*rule, typedAddNode(context, bf16, bf16),
                                 verificationMachine(), recordedOn("worker.0"));
  EXPECT_NE(error.find("no_matching_rule"), std::string::npos) << error;
}

TEST(RuleVerify, RejectsADifferentAccessMap) {
  llvm::Expected<RuleRegistry> registry = parse(R"llkmap(
rule r.map {
  match micro.vector(input[0].access_map = (d0, d1) -> (d0, d1));
  bundle "b";
  emit "e";
}
)llkmap");
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  const RuleDef *mapRule = registry->find("r.map");
  ASSERT_NE(mapRule, nullptr);

  mlir::MLIRContext context;
  mlir::Type f32 = mlir::Float32Type::get(&context);
  MachineModel empty;

  // The node's first operand is addressed by the identity map, which is what
  // the rule declares: a legal selection.
  WorkloadNode identity = typedVectorNode(f32, f32);
  identity.inputs[0].accessMap = mlir::AffineMap::getPermutationMap(
      llvm::ArrayRef<unsigned>{0, 1}, &context);
  EXPECT_TRUE(verifyText(*mapRule, identity, empty, recordedOn("")).empty());

  // The same node with a transposed map no longer satisfies the rule.
  WorkloadNode transposed = typedVectorNode(f32, f32);
  transposed.inputs[0].accessMap = mlir::AffineMap::getPermutationMap(
      llvm::ArrayRef<unsigned>{1, 0}, &context);
  std::string error = verifyText(*mapRule, transposed, empty, recordedOn(""));
  EXPECT_NE(error.find("no_matching_rule"), std::string::npos) << error;
}

TEST(RuleVerify, RejectsAnExecutorOfTheWrongKind) {
  const RuleDef *rule = verificationRule();
  ASSERT_NE(rule, nullptr);
  mlir::MLIRContext context;
  mlir::Type f32 = mlir::Float32Type::get(&context);
  std::string error = verifyText(*rule, typedAddNode(context, f32, f32),
                                 verificationMachine(), recordedOn("lane.0"));
  EXPECT_NE(error.find("no_legal_executor"), std::string::npos) << error;
}

TEST(RuleVerify, RejectsAComputeCapabilityTheExecutorDoesNotOffer) {
  const RuleDef *rule = verificationRule();
  ASSERT_NE(rule, nullptr);
  mlir::MLIRContext context;
  mlir::Type f32 = mlir::Float32Type::get(&context);
  std::string error =
      verifyText(*rule, typedAddNode(context, f32, f32), verificationMachine(),
                 recordedOn("worker.bare"));
  EXPECT_NE(error.find("unsupported_compute_fragment"), std::string::npos)
      << error;
}

TEST(RuleVerify, RejectsARecordedParameterOutsideItsDomain) {
  const RuleDef *rule = verificationRule();
  ASSERT_NE(rule, nullptr);
  mlir::MLIRContext context;
  mlir::Type f32 = mlir::Float32Type::get(&context);
  RecordedRuleSelection recorded = recordedOn("worker.0");
  recorded.parameters["VW"] = int64_t(3); // outside [4..8]
  recorded.parameters["U"] = int64_t(1);
  std::string error = verifyText(*rule, typedAddNode(context, f32, f32),
                                 verificationMachine(), recorded);
  EXPECT_NE(error.find("no_matching_rule"), std::string::npos) << error;
  // The domain check must be what rejects, not the `op = "add"` predicate.
  EXPECT_NE(error.find("outside its declared domain"), std::string::npos)
      << error;
  EXPECT_EQ(error.find("predicate"), std::string::npos) << error;
}

TEST(RuleVerify, RejectsARecordedValueTheConstraintsDoNotAccept) {
  const RuleDef *rule = verificationRule();
  ASSERT_NE(rule, nullptr);
  mlir::MLIRContext context;
  mlir::Type f32 = mlir::Float32Type::get(&context);
  RecordedRuleSelection recorded = recordedOn("worker.0");
  recorded.parameters["VW"] = int64_t(6); // machine models 8 lanes
  recorded.parameters["U"] = int64_t(1);
  std::string error = verifyText(*rule, typedAddNode(context, f32, f32),
                                 verificationMachine(), recorded);
  EXPECT_NE(error.find("no_matching_rule"), std::string::npos) << error;
  // The constraint evaluation must be what rejects, not the predicate.
  EXPECT_NE(error.find("require constraints reject"), std::string::npos)
      << error;
  EXPECT_EQ(error.find("predicate"), std::string::npos) << error;
}

TEST(RuleVerify, RejectsAnUnknownRecordedParameter) {
  const RuleDef *rule = verificationRule();
  ASSERT_NE(rule, nullptr);
  mlir::MLIRContext context;
  mlir::Type f32 = mlir::Float32Type::get(&context);
  RecordedRuleSelection recorded = recordedOn("worker.0");
  recorded.parameters["VW"] = int64_t(8);
  recorded.parameters["U"] = int64_t(1);
  recorded.parameters["XX"] = int64_t(2);
  std::string error = verifyText(*rule, typedAddNode(context, f32, f32),
                                 verificationMachine(), recorded);
  EXPECT_NE(error.find("no_matching_rule"), std::string::npos) << error;
  // The undeclared name must be what rejects, not the predicate.
  EXPECT_NE(error.find("which it does not declare"), std::string::npos)
      << error;
  EXPECT_EQ(error.find("predicate"), std::string::npos) << error;
}

TEST(RuleVerify, RejectsAMissingDerivedParameter) {
  const RuleDef *rule = verificationRule();
  ASSERT_NE(rule, nullptr);
  mlir::MLIRContext context;
  mlir::Type f32 = mlir::Float32Type::get(&context);
  RecordedRuleSelection recorded = recordedOn("worker.0");
  recorded.parameters["U"] = int64_t(1); // VW omitted
  std::string error = verifyText(*rule, typedAddNode(context, f32, f32),
                                 verificationMachine(), recorded);
  EXPECT_NE(error.find("no_matching_rule"), std::string::npos) << error;
  // The missing derived name must be what rejects, not the predicate.
  EXPECT_NE(error.find("does not record the derived parameter 'VW'"),
            std::string::npos)
      << error;
  EXPECT_EQ(error.find("predicate"), std::string::npos) << error;
}

TEST(RuleVerify, RejectsWhenNoAssignmentSatisfiesTheRule) {
  const RuleDef *rule = verificationRule();
  ASSERT_NE(rule, nullptr);
  mlir::MLIRContext context;
  mlir::Type f32 = mlir::Float32Type::get(&context);
  // No parameter assignment was recorded, so verification falls back to
  // generation's satisfiability test -- which must still reject a machine whose
  // lane count no value in [4..8] can equal.
  MachineModel model = verificationMachine();
  model.computes[0].lanes = {{"f32", 16}};
  std::string error = verifyText(*rule, typedAddNode(context, f32, f32), model,
                                 recordedOn("worker.0"));
  EXPECT_NE(error.find("no_matching_rule"), std::string::npos) << error;
}

//===----------------------------------------------------------------------===//
// Stage C7: bounded graph-pattern rules
//===----------------------------------------------------------------------===//

namespace {

/// The fused rule the plan names: a convert feeding a SiLU feeding the gating
/// multiply, implemented as one unit.
constexpr llvm::StringLiteral kFusedRule = R"llkmap(
rule avx2.fused_convert_silu_mul v1 {
  match graph {
    node cv: micro.vector(op = "convert");
    node act: micro.vector(op = "silu");
    node gate: micro.vector(op = "mul");
    edge cv.result -> act.operand0;
    edge act.result -> gate.operand0;
  }
  input "cv.operand0";
  input "gate.operand1";
  output "gate.result";
  bundle "avx2.fused.convert_silu_mul";
  emit "avx2_vector_convert";
  cost 3;
}
)llkmap";

/// A chain of `count` vector nodes, each consuming the previous one's result,
/// with the op of node `i` taken from `ops`. The first input and last output
/// are external, so the chain is a connected subgraph with a two-port boundary.
WorkloadGraph vectorChain(mlir::MLIRContext &context,
                          llvm::ArrayRef<llvm::StringRef> ops,
                          mlir::Type tile) {
  WorkloadGraph graph;
  WorkloadValueId current =
      graph.addValue(WorkloadValue{0, tile, "in", /*external=*/true});
  for (size_t index = 0; index < ops.size(); ++index) {
    bool last = index + 1 == ops.size();
    WorkloadValueId next = graph.addValue(
        WorkloadValue{0, tile, last ? "out" : "v" + std::to_string(index),
                      /*external=*/false});
    WorkloadNode node;
    node.opName = "micro.vector";
    node.sourceOrdinal = static_cast<uint32_t>(index);
    node.attributes = mlir::DictionaryAttr::get(
        &context,
        {mlir::NamedAttribute(mlir::StringAttr::get(&context, "op"),
                              mlir::StringAttr::get(&context, ops[index]))});
    node.inputs.push_back(WorkloadPort{current, tile, std::nullopt});
    node.outputs.push_back(WorkloadPort{next, tile, std::nullopt});
    graph.addNode(std::move(node));
    current = next;
  }
  graph.finalize();
  return graph;
}

} // namespace

TEST(GraphRule, RoundTripsTheDeclaredPattern) {
  llvm::Expected<RuleRegistry> registry = parse(kFusedRule);
  ASSERT_TRUE(bool(registry)) << llvm::toString(registry.takeError());
  const RuleDef *rule = registry->find("avx2.fused_convert_silu_mul");
  ASSERT_TRUE(rule);
  ASSERT_TRUE(rule->pattern);
  ASSERT_EQ(rule->pattern->nodes.size(), 3u);
  ASSERT_EQ(rule->pattern->edges.size(), 2u);
  // The anchor's operation is what a reader that only wants "where does this
  // rule start" gets, whichever form the rule was written in.
  EXPECT_EQ(rule->matchOp, "micro.vector");
  EXPECT_EQ(rule->pattern->nodes.front().name, "cv");
  EXPECT_EQ(rule->pattern->edges[0].producer, "cv");
  EXPECT_EQ(rule->pattern->edges[0].resultIndex, 0u);
  EXPECT_EQ(rule->pattern->edges[0].consumer, "act");
  EXPECT_EQ(rule->pattern->edges[0].operandIndex, 0u);

  // Printing is the round trip's other half: the graph form has to come back
  // out the way it went in, or a rule read from a file would not survive being
  // written to one.
  std::string printed = printRule(*rule);
  EXPECT_NE(printed.find("match graph {"), std::string::npos) << printed;
  EXPECT_NE(printed.find("edge cv.result0 -> act.operand0;"), std::string::npos)
      << printed;

  llvm::Expected<RuleRegistry> reparsed = parse(printed);
  ASSERT_TRUE(bool(reparsed)) << llvm::toString(reparsed.takeError()) << "\n"
                              << printed;
  const RuleDef *again = reparsed->find("avx2.fused_convert_silu_mul");
  ASSERT_TRUE(again);
  ASSERT_TRUE(again->pattern);
  ASSERT_EQ(again->pattern->nodes.size(), 3u);
  ASSERT_EQ(again->pattern->edges.size(), 2u);
  EXPECT_EQ(printRule(*again), printed);
}

TEST(GraphRule, RejectsDuplicateNodeNames) {
  EXPECT_FALSE(parses(R"llkmap(
rule bad v1 {
  match graph {
    node cv: micro.vector(op = "convert");
    node cv: micro.vector(op = "silu");
  }
  bundle "b";
  emit "e";
}
)llkmap"));
}

TEST(GraphRule, RejectsEdgesNamingUndeclaredNodes) {
  EXPECT_FALSE(parses(R"llkmap(
rule bad v1 {
  match graph {
    node cv: micro.vector(op = "convert");
    edge cv.result -> missing.operand0;
  }
  bundle "b";
  emit "e";
}
)llkmap"));
}

TEST(GraphRule, RejectsADuplicateEdge) {
  EXPECT_FALSE(parses(R"llkmap(
rule bad v1 {
  match graph {
    node cv: micro.vector(op = "convert");
    node act: micro.vector(op = "silu");
    edge cv.result -> act.operand0;
    edge cv.result -> act.operand0;
  }
  bundle "b";
  emit "e";
}
)llkmap"));
}

TEST(GraphRule, MatchesAChainAndReportsItsBoundary) {
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry = parse(kFusedRule);
  ASSERT_TRUE(bool(registry)) << llvm::toString(registry.takeError());

  WorkloadGraph graph = vectorChain(context, {"convert", "silu", "mul"},
                                    mlir::IndexType::get(&context));
  std::vector<RulePatternMatch> matches = matchRulePatterns(graph, *registry);
  ASSERT_EQ(matches.size(), 1u);
  ASSERT_EQ(matches.front().coveredNodes.size(), 3u);

  // The covered nodes are in the **pattern's** declaration order, not the
  // graph's: `cv` first even though a finalized graph may order its nodes
  // differently. Comparing by operation rather than by id is what keeps the
  // assertion about the matcher rather than about graph numbering.
  auto opOf = [&](WorkloadNodeId id) -> std::string {
    const WorkloadNode *node = graph.findNode(id);
    EXPECT_NE(node, nullptr) << "a covered node is always in the graph";
    if (!node)
      return {};
    return node->attributes.getAs<mlir::StringAttr>("op").getValue().str();
  };
  EXPECT_EQ(opOf(matches.front().coveredNodes[0]), "convert");
  EXPECT_EQ(opOf(matches.front().coveredNodes[1]), "silu");
  EXPECT_EQ(opOf(matches.front().coveredNodes[2]), "mul");

  // The boundary is what leaves the match: the chain's entry operand and its
  // final result. Each node here has one operand, so the two values *inside*
  // the match are the rule's business and are not ports -- which is the whole
  // point of computing the boundary from the edges rather than from every
  // operand.
  ASSERT_EQ(matches.front().boundary.size(), 2u);
  unsigned inputs = 0, outputs = 0;
  for (const PortRef &ref : matches.front().boundary) {
    if (ref.direction == PortDirection::Input) {
      ++inputs;
      EXPECT_EQ(opOf(ref.node), "convert");
      continue;
    }
    ++outputs;
    // The only result that leaves the match is the last node's.
    EXPECT_EQ(opOf(ref.node), "mul");
  }
  EXPECT_EQ(inputs, 1u);
  EXPECT_EQ(outputs, 1u);
}

TEST(GraphRule, DoesNotMatchWhenAnEdgeDoesNotHold) {
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry = parse(kFusedRule);
  ASSERT_TRUE(bool(registry)) << llvm::toString(registry.takeError());

  // The operations are right and the order is right, but the chain is
  // convert -> mul -> silu: the `act` node does not consume the `cv` node's
  // result, so the pattern's first edge has nothing to bind to. A matcher that
  // only checked the nodes would accept this.
  WorkloadGraph graph = vectorChain(context, {"convert", "mul", "silu"},
                                    mlir::IndexType::get(&context));
  std::vector<RulePatternMatch> matches = matchRulePatterns(graph, *registry);
  EXPECT_TRUE(matches.empty());
}

TEST(GraphRule, RejectsAnEdgeWhoseOccurrenceDoesNotExist) {
  // The syntax is well-formed but `operand3` names an occurrence a one-operand
  // operation cannot have; it is refused at match time rather than binding to
  // whatever operand happens to be there.
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry = parse(R"llkmap(
rule bad v1 {
  match graph {
    node cv: micro.vector(op = "convert");
    node act: micro.vector(op = "silu");
    edge cv.result -> act.operand3;
  }
  bundle "b";
  emit "e";
}
)llkmap");
  ASSERT_TRUE(bool(registry)) << llvm::toString(registry.takeError());

  WorkloadGraph graph =
      vectorChain(context, {"convert", "silu"}, mlir::IndexType::get(&context));
  std::vector<RulePatternMatch> matches = matchRulePatterns(graph, *registry);
  EXPECT_TRUE(matches.empty());
}

TEST(GraphRule, ReportsTheMatchCap) {
  mlir::MLIRContext context;
  llvm::Expected<RuleRegistry> registry = parse(R"llkmap(
rule pair v1 {
  match graph {
    node a: micro.vector(op = "add");
    node b: micro.vector(op = "add");
    edge a.result -> b.operand0;
  }
  bundle "b";
  emit "e";
}
)llkmap");
  ASSERT_TRUE(bool(registry)) << llvm::toString(registry.takeError());

  // Two overlapping pairs in a three-node chain, and a cap of one: the cap is
  // *reported*, because an incomplete match set is an incomplete covering and
  // presenting it as a smaller one would be a lie about what was searched.
  WorkloadGraph graph = vectorChain(context, {"add", "add", "add"},
                                    mlir::IndexType::get(&context));
  bool truncated = false;
  std::vector<RulePatternMatch> matches =
      matchRulePatterns(graph, *registry, /*maxMatches=*/1, &truncated);
  EXPECT_EQ(matches.size(), 1u);
  EXPECT_TRUE(truncated);
}
