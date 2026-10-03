//===- layout_constraints.cpp - LLKMap layout language (D3) --------------===//

#include "LLK/Mapping/LayoutConstraints.h"

#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <string>

using namespace mlir::llk::mapping;

namespace {

constexpr llvm::StringLiteral kFile = R"llkmap(
// AVX2 target layouts.
layout avx2.blocked_2d(int M, int N, int VW) {
  param M in [4..16];
  param N in [4..16];
  param VW in [4..8];
  require rank == 2;
  require VW == machine.compute("vector_engine").lanes(element_type);
  require N % VW == 0;
  map (m, n) -> (m, floordiv(n, VW), mod(n, VW));
}

layout avx2.row_major(N) {
  param N in [1..8];
  require rank == 2;
  map (m, n) -> (m, n);
}
)llkmap";

llvm::Expected<LayoutRegistry> parse(llvm::StringRef text) {
  return parseLayoutText(text, "<test>");
}

bool parses(llvm::StringRef text) {
  llvm::Expected<LayoutRegistry> registry = parse(text);
  if (!registry) {
    llvm::consumeError(registry.takeError());
    return false;
  }
  return true;
}

} // namespace

TEST(LayoutParse, ParsesDeclarations) {
  llvm::Expected<LayoutRegistry> registry = parse(kFile);
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  ASSERT_EQ(registry->all().size(), 2u);

  const LayoutDef *def = registry->find("avx2.blocked_2d");
  ASSERT_NE(def, nullptr);
  EXPECT_EQ(def->params.size(), 3u);
  EXPECT_EQ(def->domains.size(), 3u);
  EXPECT_EQ(def->constraints.size(), 3u);
  ASSERT_TRUE(def->map.has_value());
  EXPECT_EQ(def->map->dims.size(), 2u);
  EXPECT_EQ(def->map->results.size(), 3u);
  // `int M` is an integer parameter; a bare name defaults to integer too.
  EXPECT_FALSE(def->params[0].symbolic);
}

TEST(LayoutParse, BareParamDefaultsToInteger) {
  llvm::Expected<LayoutRegistry> registry = parse(kFile);
  ASSERT_TRUE(static_cast<bool>(registry));
  const LayoutDef *def = registry->find("avx2.row_major");
  ASSERT_NE(def, nullptr);
  ASSERT_EQ(def->params.size(), 1u);
  EXPECT_FALSE(def->params[0].symbolic);
}

TEST(LayoutParse, SymbolicParamIsMarked) {
  EXPECT_TRUE(
      parses("layout t.one(sym policy) { param policy in {\"a\", \"b\"}; }"));
}

TEST(LayoutParse, RejectsDuplicateLayoutId) {
  EXPECT_FALSE(parses(R"llkmap(
layout t.dup() { }
layout t.dup() { }
)llkmap"));
}

TEST(LayoutParse, RejectsMissingSemicolon) {
  EXPECT_FALSE(parses(R"llkmap(
layout t.one(N) {
  param N in [1..8]
}
)llkmap"));
}

TEST(LayoutParse, RejectsUnknownTopLevelKeyword) {
  EXPECT_FALSE(parses("mapping t.one() { }"));
}

TEST(LayoutParse, RejectsMapWithUndeclaredDim) {
  EXPECT_FALSE(parses(R"llkmap(
layout t.one(N) {
  param N in [1..8];
  map (m, n) -> (m, z);
}
)llkmap"));
}

TEST(LayoutParse, RejectsUndeclaredIdentifierInRequire) {
  EXPECT_FALSE(parses(R"llkmap(
layout t.one(N) {
  param N in [1..8];
  require M % 2 == 0;
}
)llkmap"));
}

TEST(LayoutParse, RejectsUnknownMachineQuery) {
  EXPECT_FALSE(parses(R"llkmap(
layout t.one(N) {
  param N in [1..8];
  require N == machine.clock_hz();
}
)llkmap"));
}

TEST(LayoutParse, RejectsUnterminatedLayout) {
  EXPECT_FALSE(parses("layout t.one(N) { param N in [1..8];"));
}

//===----------------------------------------------------------------------===//
// Expression evaluation
//===----------------------------------------------------------------------===//

namespace {

using mlir::llk::machine::ComputeNode;
using mlir::llk::machine::MachineModel;

MachineModel evalMachine() {
  MachineModel model;
  model.target = "eval";
  model.executors = {{"e0", "worker", std::nullopt, {}, 1, {}}};
  model.memories = {
      {"sram.0", "sram", "e0", 32768, 64, {"row_major"}, std::nullopt}};
  ComputeNode vector;
  vector.id = "vec.0";
  vector.kind = "vector_engine";
  vector.attachedTo = "e0";
  vector.elementTypes = {"f32"};
  vector.shapes = {{8}};
  vector.lanes = {{"f32", 8}, {"bf16", 16}};
  model.computes.push_back(vector);
  return model;
}

/// Parses `layout t() { require <expr>; }` and evaluates the constraint.
llvm::Expected<EvalValue>
evalExpr(llvm::StringRef expression, const MachineModel &machine,
         LayoutContext context = {},
         const llvm::StringMap<LayoutValue> &bindings = {}) {
  std::string params;
  for (const auto &binding : bindings) {
    if (!params.empty())
      params += ", ";
    params += ("int " + binding.first()).str();
  }
  std::string text =
      ("layout t(" + params + ") { require " + expression + "; }").str();
  llvm::Expected<LayoutRegistry> registry = parseLayoutText(text, "<test>");
  if (!registry)
    return registry.takeError();
  const LayoutDef *def = registry->find("t");
  return evaluateExpr(*def->constraints[0], bindings, machine, context);
}

int64_t evalInt(llvm::StringRef expression, const MachineModel &machine,
                LayoutContext context = {}) {
  llvm::Expected<EvalValue> value = evalExpr(expression, machine, context);
  if (!value || value->kind != EvalValue::Kind::Int)
    return INT64_MIN;
  return value->intValue;
}

} // namespace

TEST(LayoutEval, ArithmeticPrecedence) {
  MachineModel machine = evalMachine();
  EXPECT_EQ(evalInt("1 + 2 * 3 == 7", machine), 1);
  EXPECT_EQ(evalInt("(1 + 2) * 3 == 9", machine), 1);
  EXPECT_EQ(evalInt("-3 + 5 == 2", machine), 1);
}

TEST(LayoutEval, DivisibilityAndComparison) {
  MachineModel machine = evalMachine();
  EXPECT_EQ(evalInt("8 % 4 == 0 && 3 > 2", machine), 1);
  EXPECT_EQ(evalInt("8 % 5 == 0 || 3 >= 3", machine), 1);
  EXPECT_EQ(evalInt("!(1 == 2)", machine), 1);
}

TEST(LayoutEval, StringEquality) {
  MachineModel machine = evalMachine();
  EXPECT_EQ(evalInt("\"a\" == \"a\"", machine), 1);
  EXPECT_EQ(evalInt("\"a\" != \"b\"", machine), 1);
}

TEST(LayoutEval, BuiltinsComeFromContext) {
  MachineModel machine = evalMachine();
  LayoutContext context;
  context.rank = 2;
  context.elementType = "f32";
  EXPECT_EQ(evalInt("rank == 2 && element_type == \"f32\"", machine, context),
            1);
}

TEST(LayoutEval, MachineLanesAndCountQueries) {
  MachineModel machine = evalMachine();
  LayoutContext context;
  context.elementType = "bf16";
  EXPECT_EQ(
      evalInt("machine.compute(\"vector_engine\").lanes(element_type) == 16",
              machine, context),
      1);
  EXPECT_EQ(evalInt("machine.compute(\"vector_engine\").count == 1", machine),
            1);
}

TEST(LayoutEval, MachineMemoryQueries) {
  MachineModel machine = evalMachine();
  EXPECT_EQ(
      evalInt("machine.memory(\"sram.0\").capacity_bytes == 32768", machine),
      1);
  EXPECT_EQ(
      evalInt("machine.memory(\"sram.0\").alignment_bytes == 64", machine), 1);
}

TEST(LayoutEval, RejectsTypeMismatch) {
  MachineModel machine = evalMachine();
  EXPECT_FALSE(static_cast<bool>(evalExpr("1 == \"a\"", machine)));
  EXPECT_FALSE(static_cast<bool>(evalExpr("1 + \"a\" == 2", machine)));
}

TEST(LayoutEval, RejectsDivisionByZero) {
  MachineModel machine = evalMachine();
  EXPECT_FALSE(static_cast<bool>(evalExpr("1 / 0 == 1", machine)));
}

TEST(LayoutEval, RejectsUnknownMachineFacts) {
  MachineModel machine = evalMachine();
  EXPECT_FALSE(static_cast<bool>(
      evalExpr("machine.compute(\"matrix_engine\").count == 0", machine)));
  EXPECT_FALSE(static_cast<bool>(
      evalExpr("machine.memory(\"dram.9\").capacity_bytes > 0", machine)));
  LayoutContext context;
  context.elementType = "f64";
  EXPECT_FALSE(static_cast<bool>(
      evalExpr("machine.compute(\"vector_engine\").lanes(element_type) > 0",
               machine, context)));
}

TEST(LayoutEval, EvaluatesParameterBindings) {
  MachineModel machine = evalMachine();
  llvm::StringMap<LayoutValue> bindings;
  bindings["N"] = int64_t{3};
  llvm::Expected<EvalValue> value =
      evalExpr("N * 2 == 6", machine, {}, bindings);
  ASSERT_TRUE(static_cast<bool>(value));
  EXPECT_EQ(value->intValue, 1);
}

//===----------------------------------------------------------------------===//
// Solving
//===----------------------------------------------------------------------===//

namespace {

constexpr llvm::StringLiteral kSolvable = R"llkmap(
layout t.block(int M, int N, int VW) {
  param M in [1..2];
  param N in [4..8];
  param VW in [2..4];
  require rank == 2;
  require N % VW == 0;
  map (m, n) -> (m, floordiv(n, VW), mod(n, VW));
}
)llkmap";

llvm::Expected<LayoutSolveResult>
solveText(llvm::StringRef text, llvm::StringRef id, const MachineModel &machine,
          mlir::MLIRContext &context, LayoutContext layoutContext = {},
          SolverLimits limits = {}) {
  llvm::Expected<LayoutRegistry> registry = parseLayoutText(text, "<test>");
  if (!registry)
    return registry.takeError();
  const LayoutDef *def = registry->find(id);
  if (!def)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "missing layout '" + id + "'");
  return solveLayout(*def, machine, context, layoutContext, limits);
}

int64_t solutionInt(const LayoutSolution &solution, llvm::StringRef name) {
  auto it = solution.values.find(name.str());
  if (it == solution.values.end())
    return INT64_MIN;
  const auto *value = std::get_if<int64_t>(&it->second);
  return value ? *value : INT64_MIN;
}

std::string mapText(const mlir::AffineMap &map) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  map.print(stream);
  return stream.str();
}

} // namespace

TEST(LayoutSolve, SolvesDeterministicallyInDomainOrder) {
  mlir::MLIRContext context;
  MachineModel machine = evalMachine();
  LayoutContext layoutContext;
  layoutContext.rank = 2;
  // The default solution cap is small; raise it to see the whole space.
  SolverLimits limits;
  limits.maxSolutions = 64;
  llvm::Expected<LayoutSolveResult> result =
      solveText(kSolvable, "t.block", machine, context, layoutContext, limits);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  EXPECT_FALSE(result->truncated);
  // M in {1,2}; (N,VW) in {(4,2),(4,4),(6,2),(6,3),(8,2),(8,4)}.
  ASSERT_EQ(result->solutions.size(), 12u);
  EXPECT_EQ(solutionInt(result->solutions.front(), "M"), 1);
  EXPECT_EQ(solutionInt(result->solutions.front(), "N"), 4);
  EXPECT_EQ(solutionInt(result->solutions.front(), "VW"), 2);
}

TEST(LayoutSolve, BuildsAffineMapFromTheSolution) {
  mlir::MLIRContext context;
  MachineModel machine = evalMachine();
  LayoutContext layoutContext;
  layoutContext.rank = 2;
  llvm::Expected<LayoutSolveResult> result =
      solveText(kSolvable, "t.block", machine, context, layoutContext);
  ASSERT_TRUE(static_cast<bool>(result));
  ASSERT_FALSE(result->solutions.empty());
  EXPECT_EQ(mapText(result->solutions.front().map),
            "(d0, d1) -> (d0, d1 floordiv 2, d1 mod 2)");
}

TEST(LayoutSolve, ReturnsNoSolutionsWhenConstraintsCannotHold) {
  mlir::MLIRContext context;
  MachineModel machine = evalMachine();
  LayoutContext layoutContext;
  layoutContext.rank = 3; // `rank == 2` fails for every assignment
  llvm::Expected<LayoutSolveResult> result =
      solveText(kSolvable, "t.block", machine, context, layoutContext);
  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_TRUE(result->solutions.empty());
  EXPECT_FALSE(result->truncated);
}

TEST(LayoutSolve, RejectsUnboundParameter) {
  mlir::MLIRContext context;
  MachineModel machine = evalMachine();
  llvm::Expected<LayoutSolveResult> result = solveText(R"llkmap(
layout t.x(int M, int N) {
  param M in [1..2];
  require N > 0;
}
)llkmap",
                                                       "t.x", machine, context);
  EXPECT_FALSE(static_cast<bool>(result));
}

TEST(LayoutSolve, ReportsTruncationWhenAssignmentsAreExhausted) {
  mlir::MLIRContext context;
  MachineModel machine = evalMachine();
  SolverLimits limits;
  limits.maxAssignments = 1000;
  llvm::Expected<LayoutSolveResult> result =
      solveText(R"llkmap(
layout t.big(int M) {
  param M in [1..200000];
  require M > 200000;
}
)llkmap",
                "t.big", machine, context, {}, limits);
  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_TRUE(result->solutions.empty());
  EXPECT_TRUE(result->truncated);
}

TEST(LayoutSolve, RespectsMaxSolutionsAndDisclosesIt) {
  mlir::MLIRContext context;
  MachineModel machine = evalMachine();
  LayoutContext layoutContext;
  layoutContext.rank = 2;
  SolverLimits limits;
  limits.maxSolutions = 3;
  llvm::Expected<LayoutSolveResult> result =
      solveText(kSolvable, "t.block", machine, context, layoutContext, limits);
  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_EQ(result->solutions.size(), 3u);
  // Hitting the solution cap is a cap: the caller must be told.
  EXPECT_TRUE(result->truncated);
}

//===----------------------------------------------------------------------===//
// Shipped AVX2 layouts
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_DIR
#error "LLK_MAPPING_DIR must name the shipped mapping directory"
#endif

namespace {
constexpr llvm::StringLiteral kShippedLayouts =
    LLK_MAPPING_DIR "/x86-avx2/layouts.llkmap";
constexpr llvm::StringLiteral kInvalidLayouts =
    LLK_MAPPING_DIR "/../test/Mapping/Inputs/invalid-layouts.llkmap";
} // namespace

TEST(LayoutShipped, LoadsTheAvx2LayoutFile) {
  llvm::Expected<LayoutRegistry> registry = loadLayoutFile(kShippedLayouts);
  ASSERT_TRUE(static_cast<bool>(registry))
      << llvm::toString(registry.takeError());
  EXPECT_NE(registry->find("avx2.blocked_2d"), nullptr);
  EXPECT_NE(registry->find("avx2.row_major"), nullptr);
}

TEST(LayoutShipped, Blocked2dSolvesForF32) {
  llvm::Expected<LayoutRegistry> registry = loadLayoutFile(kShippedLayouts);
  ASSERT_TRUE(static_cast<bool>(registry));
  const LayoutDef *def = registry->find("avx2.blocked_2d");
  ASSERT_NE(def, nullptr);

  mlir::MLIRContext context;
  MachineModel machine = evalMachine();
  LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "f32";
  SolverLimits limits;
  limits.maxSolutions = 64;
  llvm::Expected<LayoutSolveResult> result =
      solveLayout(*def, machine, context, layoutContext, limits);
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->solutions.empty());
  // The lane count forces VW = 8, and N must divide by it.
  for (const LayoutSolution &solution : result->solutions) {
    EXPECT_EQ(solutionInt(solution, "VW"), 8);
    EXPECT_EQ(solutionInt(solution, "N") % 8, 0);
  }
}

TEST(LayoutShipped, Blocked2dDoesNotSolveWhenLanesFallOutsideTheDomain) {
  llvm::Expected<LayoutRegistry> registry = loadLayoutFile(kShippedLayouts);
  ASSERT_TRUE(static_cast<bool>(registry));
  const LayoutDef *def = registry->find("avx2.blocked_2d");
  ASSERT_NE(def, nullptr);

  mlir::MLIRContext context;
  MachineModel machine = evalMachine();
  LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "bf16"; // lanes 16, outside VW in [4..8]
  llvm::Expected<LayoutSolveResult> result =
      solveLayout(*def, machine, context, layoutContext);
  ASSERT_TRUE(static_cast<bool>(result));
  EXPECT_TRUE(result->solutions.empty());
  EXPECT_FALSE(result->truncated);
}

TEST(LayoutShipped, RejectsTheInvalidFixture) {
  llvm::Expected<LayoutRegistry> registry = loadLayoutFile(kInvalidLayouts);
  EXPECT_FALSE(static_cast<bool>(registry));
  if (!registry)
    llvm::consumeError(registry.takeError());
}
