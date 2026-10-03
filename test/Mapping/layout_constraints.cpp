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
