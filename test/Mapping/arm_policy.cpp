//===- arm_policy.cpp - Arm SVE/SME LLKMap policy (issue #178) ------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The Arm policy slice loads the two machines/ profiles with
// mapping/arm-{sve,sme}/{layouts,rules}.llkmap through the generic
// `loadMappingTarget`, then checks the three properties the policy exists to
// state: a finite per-dtype layout domain, typed rule predicates, and per-port
// layout requirements for a mixed-dtype program. Target packages and registry
// entries are a later slice; the emitter keys are declared here.
//
//===----------------------------------------------------------------------===//

#include "LLK/Mapping/LayoutConstraints.h"
#include "LLK/Mapping/LlkMap.h"
#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/MappingRules.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/WorkloadGraph.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

using namespace mlir::llk::mapping;

namespace {

#ifndef LLK_MAPPING_DIR
#error "LLK_MAPPING_DIR must name the shipped mapping directory"
#endif
#ifndef LLK_MACHINE_DIR
#error "LLK_MACHINE_DIR must name the shipped machines directory"
#endif

/// The ten emitter keys each Arm package declares (spec §7), in declaration
/// order. The target packages exist from the next slice onward; until then this
/// list is the contract the rules are checked against.
std::vector<std::string> emitterKeys(llvm::StringRef prefix) {
  std::vector<std::string> keys;
  for (llvm::StringRef suffix :
       {"vector_add", "vector_convert", "vector_silu", "vector_mul", "reduce",
        "copy", "tile_copy", "tile_store", "fused_convert_silu_mul"})
    keys.push_back((prefix + "_" + suffix).str());
  keys.push_back((prefix + (prefix == "sve" ? "_mma" : "_mma_za")).str());
  return keys;
}

llvm::Expected<std::unique_ptr<MappingTarget>>
loadArmTarget(llvm::StringRef prefix) {
  std::string base = ("arm-" + prefix).str();
  return loadMappingTarget(
      base, std::string(LLK_MACHINE_DIR) + "/" + base + "-v2.yaml",
      std::string(LLK_MAPPING_DIR) + "/" + base + "/layouts.llkmap",
      std::string(LLK_MAPPING_DIR) + "/" + base + "/rules.llkmap",
      emitterKeys(prefix));
}

/// The rank-two tile type these fixtures put on every port. It is built here
/// and returned as the general `mlir::Type` a workload port stores, so the port
/// assignment is a base-to-base copy rather than a derived-handle slice.
mlir::Type rankTwoTile(mlir::Type elementType) {
  return mlir::RankedTensorType::get({8, 16}, elementType);
}

/// A `micro.vector` / `micro.reduce` node with a rank-two tile on every port.
/// Attribute predicates read `attributes`, port predicates read the port types.
WorkloadNode elementwiseNode(mlir::MLIRContext &context, llvm::StringRef opName,
                             llvm::StringRef opAttribute,
                             const std::vector<mlir::Type> &operandTypes,
                             mlir::Type resultType) {
  WorkloadNode node;
  node.opName = opName.str();
  if (!opAttribute.empty())
    node.attributes = mlir::DictionaryAttr::get(
        &context,
        {mlir::NamedAttribute(mlir::StringAttr::get(&context, "op"),
                              mlir::StringAttr::get(&context, opAttribute))});
  uint32_t value = 1;
  for (mlir::Type type : operandTypes) {
    WorkloadPort port;
    port.value = value++;
    port.type = rankTwoTile(type);
    node.inputs.push_back(port);
  }
  WorkloadPort result;
  result.value = value;
  result.type = rankTwoTile(resultType);
  node.outputs.push_back(result);
  return node;
}

std::vector<std::string> matchedIds(const WorkloadNode &node,
                                    const RuleRegistry &rules) {
  std::vector<std::string> ids;
  for (const RuleDef *rule : matchRules(node, rules))
    ids.push_back(rule->id);
  return ids;
}

/// The single solution's integer value for `name`, or INT64_MIN when the solve
/// did not produce one.
int64_t solutionValue(const LayoutSolveResult &result, llvm::StringRef name) {
  if (result.solutions.size() != 1)
    return INT64_MIN;
  auto it = result.solutions[0].values.find(name.str());
  if (it == result.solutions[0].values.end())
    return INT64_MIN;
  const auto *value = std::get_if<int64_t>(&it->second);
  return value ? *value : INT64_MIN;
}

} // namespace

//===----------------------------------------------------------------------===//
// Configuration
//===----------------------------------------------------------------------===//

TEST(ArmPolicy, LoadsBothTargetsWithExactlyTheDeclaredEmitters) {
  for (llvm::StringRef prefix : {"sve", "sme"}) {
    std::vector<std::string> declared = emitterKeys(prefix);
    ASSERT_EQ(declared.size(), 10u) << prefix.str();
    llvm::Expected<std::unique_ptr<MappingTarget>> target =
        loadArmTarget(prefix);
    ASSERT_TRUE(static_cast<bool>(target))
        << prefix.str() << ": " << llvm::toString(target.takeError());
    EXPECT_EQ((*target)->name(), ("arm-" + prefix).str());
    for (const std::string &key : declared)
      EXPECT_TRUE((*target)->isKnownEmitter(key)) << key;
    // A key the other Arm package (or AVX2) declares is not this target's.
    EXPECT_FALSE((*target)->isKnownEmitter("avx2_vector_add"));
    EXPECT_FALSE((*target)->isKnownEmitter(
        (llvm::StringRef(prefix) == "sve" ? "sme_mma_za" : "sve_mma")));
    // `loadMappingTarget` verifies, so success already proves every rule
    // emitter is declared; assert the converse explicitly for a reader.
    for (const RuleDef &rule : (*target)->rules().all())
      EXPECT_TRUE(llvm::is_contained(declared, rule.emitter)) << rule.id;
  }
}

TEST(ArmPolicy, DeclaresTheSpecifiedRuleIdsAndCosts) {
  for (llvm::StringRef prefix : {"sve", "sme"}) {
    llvm::Expected<std::unique_ptr<MappingTarget>> target =
        loadArmTarget(prefix);
    ASSERT_TRUE(static_cast<bool>(target))
        << llvm::toString(target.takeError());
    for (llvm::StringRef suffix :
         {"vector_add_f32", "vector_add_bf16", "vector_add_i8",
          "vector_mul_f32", "vector_mul_bf16", "vector_mul_i8",
          "vector_convert_bf16_f32", "vector_convert_bf16_f32_row_major",
          "vector_convert_f32_bf16", "vector_convert_f32_bf16_row_major",
          "vector_silu_f32", "reduce_sum_f32", "mma_f32", "mma_bf16",
          "async_copy", "tile_async_copy", "tile_store",
          "fused_convert_silu_mul"}) {
      std::string id = (prefix + "." + suffix).str();
      const RuleDef *rule = (*target)->rules().find(id);
      ASSERT_NE(rule, nullptr) << id;
      EXPECT_TRUE(rule->costLowerBound.has_value()) << id;
      uint64_t expected = 0;
      if (suffix.starts_with("vector_add") || suffix.starts_with("vector_mul"))
        expected = 4;
      else if (suffix.starts_with("vector_convert"))
        expected = 2;
      else if (suffix.starts_with("vector_silu"))
        expected = 4;
      else if (suffix.starts_with("reduce"))
        expected = 3;
      else if (suffix.starts_with("mma"))
        expected = prefix == "sve" ? 16 : 8;
      else if (suffix.starts_with("fused"))
        expected = 3;
      else
        expected = 2; // copy, tile_copy, tile_store
      EXPECT_EQ(*rule->costLowerBound, expected) << id;
    }
  }
}

//===----------------------------------------------------------------------===//
// Layouts: finite per-dtype domains
//===----------------------------------------------------------------------===//

TEST(ArmPolicy, EachTargetDeclaresOneLayoutPerKind) {
  for (llvm::StringRef prefix : {"sve", "sme"}) {
    llvm::Expected<std::unique_ptr<MappingTarget>> target =
        loadArmTarget(prefix);
    ASSERT_TRUE(static_cast<bool>(target))
        << llvm::toString(target.takeError());
    // Exactly one layout realizes each Micro kind, so the binding-to-layout
    // bridge is well defined rather than ambiguous.
    EXPECT_EQ((*target)->layouts().implementing("row_major").size(), 1u)
        << prefix.str();
    EXPECT_EQ((*target)->layouts().implementing("blocked").size(), 1u)
        << prefix.str();

    const LayoutDef *rowMajor =
        (*target)->layouts().find((prefix + ".row_major").str());
    ASSERT_NE(rowMajor, nullptr);
    // Row-major is the identity map with no unused size parameter.
    EXPECT_TRUE(rowMajor->params.empty());
    EXPECT_TRUE(rowMajor->domains.empty());
  }
}

TEST(ArmPolicy, BlockedLayoutSolvesOneWidthPerDtype) {
  mlir::MLIRContext context;
  for (llvm::StringRef prefix : {"sve", "sme"}) {
    llvm::Expected<std::unique_ptr<MappingTarget>> target =
        loadArmTarget(prefix);
    ASSERT_TRUE(static_cast<bool>(target))
        << llvm::toString(target.takeError());
    const LayoutDef *blocked =
        (*target)->layouts().find((prefix + ".blocked_2d").str());
    ASSERT_NE(blocked, nullptr);

    // The domain is the three advertised widths and nothing more: the solve is
    // bounded at three assignments.
    auto domain = blocked->domains.find("VW");
    ASSERT_NE(domain, blocked->domains.end()) << prefix.str();
    std::vector<int64_t> widths;
    for (const LayoutValue &value : domain->second.values)
      widths.push_back(std::get<int64_t>(value));
    EXPECT_EQ(widths, (std::vector<int64_t>{16, 32, 64})) << prefix.str();

    // A 512-bit profile gives f32/bf16/i8 exactly one lane width each, so the
    // solve is unique and never undecided.
    struct DtypeWidth {
      llvm::StringRef elementType;
      int64_t width;
    };
    for (const DtypeWidth &expected :
         {DtypeWidth{"f32", 16}, DtypeWidth{"bf16", 32},
          DtypeWidth{"i8", 64}}) {
      LayoutContext layoutContext;
      layoutContext.rank = 2;
      layoutContext.elementType = expected.elementType.str();
      llvm::Expected<LayoutSolveResult> solved =
          solveLayout(*blocked, (*target)->machine(), context, layoutContext);
      ASSERT_TRUE(static_cast<bool>(solved))
          << prefix.str() << "/" << expected.elementType.str() << ": "
          << llvm::toString(solved.takeError());
      EXPECT_FALSE(solved->undecided)
          << prefix.str() << "/" << expected.elementType.str();
      EXPECT_FALSE(solved->truncated)
          << prefix.str() << "/" << expected.elementType.str();
      ASSERT_EQ(solved->solutions.size(), 1u)
          << prefix.str() << "/" << expected.elementType.str();
      EXPECT_EQ(solutionValue(*solved, "VW"), expected.width)
          << prefix.str() << "/" << expected.elementType.str();
    }

    // f64 has no lane fact in this profile, so the solve errors instead of
    // selecting a width.
    LayoutContext f64Context;
    f64Context.rank = 2;
    f64Context.elementType = "f64";
    llvm::Expected<LayoutSolveResult> f64 =
        solveLayout(*blocked, (*target)->machine(), context, f64Context);
    EXPECT_FALSE(static_cast<bool>(f64)) << prefix.str();
    if (!f64)
      llvm::consumeError(f64.takeError());
  }
}

TEST(ArmPolicy, RowMajorLayoutSolvesWithoutParameters) {
  mlir::MLIRContext context;
  for (llvm::StringRef prefix : {"sve", "sme"}) {
    llvm::Expected<std::unique_ptr<MappingTarget>> target =
        loadArmTarget(prefix);
    ASSERT_TRUE(static_cast<bool>(target))
        << llvm::toString(target.takeError());
    const LayoutDef *rowMajor =
        (*target)->layouts().find((prefix + ".row_major").str());
    ASSERT_NE(rowMajor, nullptr);
    LayoutContext layoutContext;
    layoutContext.rank = 2;
    layoutContext.elementType = "f32";
    llvm::Expected<LayoutSolveResult> solved =
        solveLayout(*rowMajor, (*target)->machine(), context, layoutContext);
    ASSERT_TRUE(static_cast<bool>(solved))
        << llvm::toString(solved.takeError());
    EXPECT_FALSE(solved->undecided);
    EXPECT_FALSE(solved->truncated);
    ASSERT_EQ(solved->solutions.size(), 1u);
    EXPECT_EQ(solved->solutions[0].map.getNumDims(), 2u);
    EXPECT_EQ(solved->solutions[0].map.getNumResults(), 2u);
  }
}

//===----------------------------------------------------------------------===//
// Typed rule predicates
//===----------------------------------------------------------------------===//

TEST(ArmPolicy, TypedPredicatesAcceptOnlyTheSupportedSignatures) {
  mlir::MLIRContext context;
  mlir::Type f32 = mlir::Float32Type::get(&context);
  mlir::Type bf16 = mlir::BFloat16Type::get(&context);
  mlir::Type i8 = mlir::IntegerType::get(&context, 8);

  for (llvm::StringRef prefix : {"sve", "sme"}) {
    llvm::Expected<std::unique_ptr<MappingTarget>> target =
        loadArmTarget(prefix);
    ASSERT_TRUE(static_cast<bool>(target))
        << llvm::toString(target.takeError());
    const RuleRegistry &rules = (*target)->rules();
    std::string p = prefix.str();

    // Elementwise add: one typed rule per advertised dtype.
    EXPECT_EQ(matchedIds(elementwiseNode(context, "micro.vector", "add",
                                         {f32, f32}, f32),
                         rules),
              (std::vector<std::string>{p + ".vector_add_f32"}));
    EXPECT_EQ(matchedIds(elementwiseNode(context, "micro.vector", "add",
                                         {bf16, bf16}, bf16),
                         rules),
              (std::vector<std::string>{p + ".vector_add_bf16"}));
    EXPECT_EQ(matchedIds(
                  elementwiseNode(context, "micro.vector", "add", {i8, i8}, i8),
                  rules),
              (std::vector<std::string>{p + ".vector_add_i8"}));

    // Conversion has a blocked and a row-major initializer variant.
    EXPECT_EQ(
        matchedIds(
            elementwiseNode(context, "micro.vector", "convert", {bf16}, f32),
            rules),
        (std::vector<std::string>{p + ".vector_convert_bf16_f32",
                                  p + ".vector_convert_bf16_f32_row_major"}));

    // SiLU and reduction are f32 only.
    EXPECT_EQ(
        matchedIds(elementwiseNode(context, "micro.vector", "silu", {f32}, f32),
                   rules),
        (std::vector<std::string>{p + ".vector_silu_f32"}));
    EXPECT_EQ(
        matchedIds(elementwiseNode(context, "micro.reduce", "sum", {f32}, f32),
                   rules),
        (std::vector<std::string>{p + ".reduce_sum_f32"}));

    // A contraction matches exactly the rule for its input pair; the f32 and
    // bf16 rules are distinguished by the operands, not the accumulator.
    EXPECT_EQ(matchedIds(elementwiseNode(context, "micro.mma", "",
                                         {f32, f32, f32}, f32),
                         rules),
              (std::vector<std::string>{p + ".mma_f32"}));
    EXPECT_EQ(matchedIds(elementwiseNode(context, "micro.mma", "",
                                         {bf16, bf16, f32}, f32),
                         rules),
              (std::vector<std::string>{p + ".mma_bf16"}));
  }
}

TEST(ArmPolicy, TypedPredicatesRejectUnsupportedSignatures) {
  mlir::MLIRContext context;
  mlir::Type f32 = mlir::Float32Type::get(&context);
  mlir::Type bf16 = mlir::BFloat16Type::get(&context);
  mlir::Type f64 = mlir::Float64Type::get(&context);
  mlir::Type i8 = mlir::IntegerType::get(&context, 8);

  for (llvm::StringRef prefix : {"sve", "sme"}) {
    llvm::Expected<std::unique_ptr<MappingTarget>> target =
        loadArmTarget(prefix);
    ASSERT_TRUE(static_cast<bool>(target))
        << llvm::toString(target.takeError());
    const RuleRegistry &rules = (*target)->rules();

    // f64 vectors are outside the advertised table.
    EXPECT_TRUE(matchedIds(elementwiseNode(context, "micro.vector", "add",
                                           {f64, f64}, f64),
                           rules)
                    .empty());
    EXPECT_TRUE(matchedIds(elementwiseNode(context, "micro.vector", "convert",
                                           {f64}, f32),
                           rules)
                    .empty());
    EXPECT_TRUE(
        matchedIds(elementwiseNode(context, "micro.vector", "silu", {f64}, f64),
                   rules)
            .empty());

    // SiLU is f32 only: a bf16 activation has no rule.
    EXPECT_TRUE(matchedIds(elementwiseNode(context, "micro.vector", "silu",
                                           {bf16}, bf16),
                           rules)
                    .empty());
    // ... and a bf16 reduction has none either.
    EXPECT_TRUE(matchedIds(elementwiseNode(context, "micro.reduce", "sum",
                                           {bf16}, bf16),
                           rules)
                    .empty());

    // i8 contraction is not claimed: no rule matches it.
    std::vector<std::string> i8Mma = matchedIds(
        elementwiseNode(context, "micro.mma", "", {i8, i8, f32}, f32), rules);
    EXPECT_TRUE(llvm::none_of(i8Mma, [](const std::string &id) {
      return id.find(".mma_") != std::string::npos;
    })) << (i8Mma.empty() ? "" : i8Mma[0]);
  }
}

//===----------------------------------------------------------------------===//
// Mixed-dtype per-port layout requirements
//===----------------------------------------------------------------------===//

namespace {

/// convert(bf16) -> silu(f32) -> mul(f32), the fused epilogue's shape.
WorkloadGraph fusedConvertSiluMul(mlir::MLIRContext &context) {
  mlir::Type f32 = mlir::Float32Type::get(&context);
  mlir::Type bf16 = mlir::BFloat16Type::get(&context);

  WorkloadGraph graph;
  WorkloadValueId input = graph.addValue(
      WorkloadValue{0, rankTwoTile(bf16), "in", /*external=*/true});
  WorkloadValueId converted =
      graph.addValue(WorkloadValue{0, rankTwoTile(f32), "converted", false});
  WorkloadValueId activated =
      graph.addValue(WorkloadValue{0, rankTwoTile(f32), "activated", false});
  WorkloadValueId other = graph.addValue(
      WorkloadValue{0, rankTwoTile(f32), "other", /*external=*/true});
  WorkloadValueId result =
      graph.addValue(WorkloadValue{0, rankTwoTile(f32), "result", false});

  auto addVector =
      [&](llvm::StringRef op,
          const std::vector<std::pair<WorkloadValueId, mlir::Type>> &ins,
          WorkloadValueId outValue, mlir::Type outType) {
        WorkloadNode node;
        node.opName = "micro.vector";
        node.attributes = mlir::DictionaryAttr::get(
            &context,
            {mlir::NamedAttribute(mlir::StringAttr::get(&context, "op"),
                                  mlir::StringAttr::get(&context, op))});
        for (const auto &[value, type] : ins) {
          WorkloadPort port;
          port.value = value;
          port.type = type;
          node.inputs.push_back(port);
        }
        WorkloadPort out;
        out.value = outValue;
        out.type = outType;
        node.outputs.push_back(out);
        graph.addNode(std::move(node));
      };

  addVector("convert", {{input, rankTwoTile(bf16)}}, converted,
            rankTwoTile(f32));
  addVector("silu", {{converted, rankTwoTile(f32)}}, activated,
            rankTwoTile(f32));
  addVector("mul", {{activated, rankTwoTile(f32)}, {other, rankTwoTile(f32)}},
            result, rankTwoTile(f32));
  graph.finalize();
  return graph;
}

} // namespace

TEST(ArmPolicy, FusedGraphKeepsPerPortWidths) {
  mlir::MLIRContext context;
  for (llvm::StringRef prefix : {"sve", "sme"}) {
    llvm::Expected<std::unique_ptr<MappingTarget>> target =
        loadArmTarget(prefix);
    ASSERT_TRUE(static_cast<bool>(target))
        << llvm::toString(target.takeError());

    WorkloadGraph graph = fusedConvertSiluMul(context);
    std::vector<RulePatternMatch> matches =
        matchRulePatterns(graph, (*target)->rules());
    const RulePatternMatch *fused = nullptr;
    for (const RulePatternMatch &match : matches)
      if (match.rule->id == (prefix + ".fused_convert_silu_mul").str())
        fused = &match;
    ASSERT_NE(fused, nullptr) << prefix.str();
    ASSERT_EQ(fused->coveredNodes.size(), 3u);

    // The fallback names bf16, the convert operand's dtype. If requirements
    // were solved against the graph's first value, the f32 gate ports would
    // solve to width 32; per-port resolution must keep them at 16.
    LayoutContext fallback;
    fallback.rank = 2;
    fallback.elementType = "bf16";
    std::string reason;
    std::optional<MappingCandidate> candidate = toFusedMappingCandidate(
        *fused->rule, *fused, graph, (*target)->machine(), fallback, &reason);
    ASSERT_TRUE(candidate.has_value()) << prefix.str() << ": " << reason;
    ASSERT_EQ(candidate->layoutRequirements.size(), 3u);

    struct Expected {
      llvm::StringRef port;
      llvm::StringRef elementType;
      int64_t width;
    };
    const Expected expected[] = {{"cv.operand0", "bf16", 32},
                                 {"gate.operand1", "f32", 16},
                                 {"gate.result", "f32", 16}};
    for (size_t index = 0; index < 3; ++index) {
      const LayoutRequirement &requirement =
          candidate->layoutRequirements[index];
      EXPECT_EQ(requirement.layoutClass, (prefix + ".blocked_2d").str());
      EXPECT_EQ(requirement.elementType, expected[index].elementType.str());
      EXPECT_EQ(requirement.rank, 2);

      // The requirement is attributed to the named boundary port.
      auto port = llvm::find_if(candidate->ports, [&](const PortSpec &spec) {
        return spec.name == expected[index].port.str();
      });
      ASSERT_NE(port, candidate->ports.end()) << expected[index].port.str();
      EXPECT_EQ(requirement.port, port->port);

      // Solving that port's own layout yields its own width.
      const LayoutDef *def = (*target)->layouts().find(requirement.layoutClass);
      ASSERT_NE(def, nullptr);
      LayoutContext layoutContext;
      layoutContext.rank = requirement.rank;
      layoutContext.elementType = requirement.elementType;
      llvm::Expected<LayoutSolveResult> solved =
          solveLayout(*def, (*target)->machine(), context, layoutContext);
      ASSERT_TRUE(static_cast<bool>(solved))
          << llvm::toString(solved.takeError());
      EXPECT_FALSE(solved->undecided);
      EXPECT_EQ(solutionValue(*solved, "VW"), expected[index].width)
          << expected[index].port.str();
    }
  }
}

TEST(ArmPolicy, ContractionRequirementsNameStorageAndCompute) {
  mlir::MLIRContext context;
  mlir::Type f32 = mlir::Float32Type::get(&context);

  for (llvm::StringRef prefix : {"sve", "sme"}) {
    llvm::Expected<std::unique_ptr<MappingTarget>> target =
        loadArmTarget(prefix);
    ASSERT_TRUE(static_cast<bool>(target))
        << llvm::toString(target.takeError());
    const RuleDef *rule = (*target)->rules().find((prefix + ".mma_f32").str());
    ASSERT_NE(rule, nullptr);
    WorkloadNode node =
        elementwiseNode(context, "micro.mma", "", {f32, f32, f32}, f32);
    std::string reason;
    std::optional<MappingCandidate> candidate = toMappingCandidate(
        *rule, node, (*target)->machine(), LayoutContext{}, &reason);
    ASSERT_TRUE(candidate.has_value()) << prefix.str() << ": " << reason;

    // Worker scope, and the target's own contraction capability.
    ASSERT_EQ(candidate->executorRequirements.size(), 1u);
    EXPECT_EQ(candidate->executorRequirements[0].capability, "worker");
    ASSERT_EQ(candidate->computeRequirements.size(), 1u);
    EXPECT_EQ(candidate->computeRequirements[0].kind,
              prefix == "sve" ? "vector_engine" : "matrix_engine");

    // The logical accumulator is input `acc` and output `result`, both in `acc`
    // memory.
    ASSERT_EQ(candidate->memoryRequirements.size(), 2u);
    EXPECT_EQ(candidate->memoryRequirements[0].kind, "acc");
    ASSERT_TRUE(candidate->memoryRequirements[0].port.has_value());
    EXPECT_EQ(candidate->memoryRequirements[0].port->direction,
              PortDirection::Input);
    EXPECT_EQ(candidate->memoryRequirements[0].port->index, 2u);
    EXPECT_EQ(candidate->memoryRequirements[1].kind, "acc");
    ASSERT_TRUE(candidate->memoryRequirements[1].port.has_value());
    EXPECT_EQ(candidate->memoryRequirements[1].port->direction,
              PortDirection::Output);
    EXPECT_EQ(candidate->memoryRequirements[1].port->index, 0u);

    // Every contraction operand is row-major, solved against its own f32 type.
    ASSERT_EQ(candidate->layoutRequirements.size(), 3u);
    for (const LayoutRequirement &requirement : candidate->layoutRequirements) {
      EXPECT_EQ(requirement.layoutClass, (prefix + ".row_major").str());
      EXPECT_EQ(requirement.elementType, "f32");
      EXPECT_EQ(requirement.rank, 2);
    }

    EXPECT_DOUBLE_EQ(candidate->lowerBound.latencyCycles,
                     prefix == "sve" ? 16.0 : 8.0);
  }
}
