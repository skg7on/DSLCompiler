//===- mapping_rules.cpp - LLKMap rule declarations (D4) -----------------===//

#include "LLK/Mapping/MappingRules.h"

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
