//===- search_binding.cpp - SearchBinding tests (issue #80 / D1) ---------===//

#include "LLK/Mapping/SearchBinding.h"

#include <gtest/gtest.h>

#include <string>

using namespace mlir::llk::mapping;

namespace {
llvm::StringMap<SearchValue>
values(std::initializer_list<std::pair<llvm::StringRef, SearchValue>> entries) {
  llvm::StringMap<SearchValue> map;
  for (const auto &entry : entries)
    map[entry.first] = entry.second;
  return map;
}
} // namespace

TEST(SearchBinding, HashIgnoresInsertionOrder) {
  auto a =
      values({{"BM", int64_t{64}}, {"tile_layout", std::string("blocked")}});
  auto b =
      values({{"tile_layout", std::string("blocked")}, {"BM", int64_t{64}}});
  EXPECT_EQ(computeSearchBindingHash(a), computeSearchBindingHash(b));
}

TEST(SearchBinding, HashDistinguishesValueKind) {
  auto integer = values({{"x", int64_t{1}}});
  auto symbolic = values({{"x", std::string("1")}});
  EXPECT_NE(computeSearchBindingHash(integer),
            computeSearchBindingHash(symbolic));
}

TEST(SearchBinding, HashDistinguishesValue) {
  auto one = values({{"x", int64_t{1}}});
  auto two = values({{"x", int64_t{2}}});
  EXPECT_NE(computeSearchBindingHash(one), computeSearchBindingHash(two));
}

TEST(SearchBinding, CanonicalStringIsSortedAndTyped) {
  auto binding = makeSearchBinding(
      "candidate_0", values({{"b", int64_t{2}}, {"a", std::string("z")}}));
  EXPECT_EQ(canonicalSearchBindingString(binding), "a=s:z;b=i:2");
  EXPECT_NE(binding.stableHash, 0u);
}

TEST(SearchBinding, OrderingUsesCanonicalContentThenId) {
  auto x = makeSearchBinding("c", values({{"a", int64_t{1}}}));
  auto y = makeSearchBinding("c", values({{"a", int64_t{2}}}));
  EXPECT_TRUE(searchBindingLess(x, y));
  EXPECT_FALSE(searchBindingLess(y, x));

  // Identical content: the id breaks the tie deterministically.
  auto first = makeSearchBinding("a", values({{"a", int64_t{1}}}));
  auto second = makeSearchBinding("b", values({{"a", int64_t{1}}}));
  EXPECT_TRUE(searchBindingLess(first, second));
}
