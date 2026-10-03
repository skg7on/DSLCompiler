//===- stable_hash.cpp - Stable id hashing tests (issue #80 / D1) --------===//

#include "LLK/Mapping/StableHash.h"

#include <gtest/gtest.h>

#include <string>

using namespace mlir::llk::mapping;

TEST(StableHash, KnownFnv1aVectors) {
  // FNV-1a 64 reference vectors; pins the algorithm so ids stay reproducible
  // across toolchain versions, which `llvm::hash_value` does not promise.
  EXPECT_EQ(stableHash(""), 0xcbf29ce484222325ULL);
  EXPECT_EQ(stableHash("a"), 0xaf63dc4c8601ec8cULL);
}

TEST(StableHash, CombineDistinguishesValues) {
  EXPECT_NE(stableHashCombine(0, 7), stableHashCombine(0, 8));
  EXPECT_NE(stableHashCombine(0, 7),
            stableHashCombine(stableHashCombine(0, 7), 0));
}

TEST(StableHash, SortedSetIgnoresCollectionOrder) {
  EXPECT_EQ(stableHashSortedSet({9, 7, 3}), stableHashSortedSet({3, 9, 7}));
  EXPECT_EQ(stableHashSortedSet({3, 9, 7}), stableHashSortedSet({7, 9, 3}));
}

TEST(StableHash, SortedSetDistinguishesDifferentMembers) {
  EXPECT_NE(stableHashSortedSet({1, 2}), stableHashSortedSet({1, 3}));
  EXPECT_NE(stableHashSortedSet({1, 2}), stableHashSortedSet({1, 2, 2}));
}

TEST(StableHash, HexIdIsSixteenLowercaseDigits) {
  EXPECT_EQ(hexId(0x0abcULL), "0000000000000abc");
  EXPECT_EQ(hexId(0).size(), 16u);
}
