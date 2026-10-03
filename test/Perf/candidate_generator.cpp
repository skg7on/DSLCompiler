//===- candidate_generator.cpp - Candidate enumeration tests --------------===//
//
// Covers issue #49, generator half:
//   - grid search emits the whole Cartesian product
//   - the order is deterministic: parameters in declaration order, the last
//     parameter varying fastest, so the preferred value of every parameter
//     comes first
//   - maxCandidates keeps a prefix of that order rather than a reshuffle
//   - symbolic choices land in symbolicValues and integer choices in values
//   - candidate ids are stable hashes of the bindings
//   - random search is reproducible from its seed and samples without
//     replacement
//
//===----------------------------------------------------------------------===//

#include "LLK/Perf/Candidate.h"
#include "LLK/Perf/CandidateGenerator.h"

#include "gtest/gtest.h"

#include <set>
#include <string>
#include <vector>

namespace mlir::llk::perf {
namespace {

SearchParam integerParam(std::string name, std::vector<SearchChoice> choices) {
  return SearchParam{std::move(name), "integer", std::move(choices)};
}

SearchParam symbolicParam(std::string name, std::string kind,
                          std::vector<SearchChoice> choices) {
  return SearchParam{std::move(name), std::move(kind), std::move(choices)};
}

/// Two integer parameters with two choices each: the acceptance test's space.
SearchSpace twoByTwo() {
  SearchSpace space;
  space.name = "two_by_two";
  space.workload = "matmul";
  space.params = {
      integerParam("BM", {SearchChoice(int64_t{1}), SearchChoice(int64_t{2})}),
      integerParam("BN",
                   {SearchChoice(int64_t{10}), SearchChoice(int64_t{20})})};
  return space;
}

std::vector<std::string> idsOf(const std::vector<Candidate> &candidates) {
  std::vector<std::string> ids;
  for (const Candidate &candidate : candidates)
    ids.push_back(candidate.id);
  return ids;
}

TEST(CandidateGenerator, GridEmitsTheCartesianProduct) {
  auto candidates = generateGridCandidates(twoByTwo());
  ASSERT_EQ(candidates.size(), 4u);

  ASSERT_EQ(candidates[0].values.at("BM"), 1);
  ASSERT_EQ(candidates[0].values.at("BN"), 10);
  ASSERT_EQ(candidates[1].values.at("BN"), 20);
  ASSERT_EQ(candidates[2].values.at("BM"), 2);
  ASSERT_EQ(candidates[2].values.at("BN"), 10);
  ASSERT_EQ(candidates[3].values.at("BM"), 2);
  ASSERT_EQ(candidates[3].values.at("BN"), 20);
}

TEST(CandidateGenerator, GridOrderIsStableAcrossCalls) {
  EXPECT_EQ(idsOf(generateGridCandidates(twoByTwo())),
            idsOf(generateGridCandidates(twoByTwo())));
}

TEST(CandidateGenerator, MaxCandidatesKeepsAPrefix) {
  auto full = generateGridCandidates(twoByTwo());
  auto capped = generateGridCandidates(twoByTwo(), 3);
  ASSERT_EQ(capped.size(), 3u);
  for (size_t i = 0; i < capped.size(); ++i)
    EXPECT_EQ(capped[i].id, full[i].id);
}

TEST(CandidateGenerator, SymbolicChoicesUseSymbolicValues) {
  SearchSpace space;
  space.params = {
      integerParam("BM", {SearchChoice(int64_t{8})}),
      symbolicParam("tile_layout", "layout",
                    {SearchChoice("row_major"), SearchChoice("blocked")})};

  auto candidates = generateGridCandidates(space);
  ASSERT_EQ(candidates.size(), 2u);
  EXPECT_EQ(candidates[0].values.at("BM"), 8);
  EXPECT_TRUE(candidates[0].values.count("tile_layout") == 0);
  EXPECT_EQ(candidates[0].symbolicValues.at("tile_layout"), "row_major");
  EXPECT_EQ(candidates[1].symbolicValues.at("tile_layout"), "blocked");
}

TEST(CandidateGenerator, CandidateIdIsAStableHashOfTheBindings) {
  std::map<std::string, int64_t> first{{"BM", 8}, {"BN", 64}};
  std::map<std::string, int64_t> reordered{{"BN", 64}, {"BM", 8}};
  std::map<std::string, std::string> symbolic{{"tile_layout", "row_major"}};

  std::string id = computeCandidateId(first, symbolic);
  EXPECT_EQ(id, computeCandidateId(reordered, symbolic));
  EXPECT_EQ(id.substr(0, 10), "candidate_");
  EXPECT_EQ(id.size(), 10u + 16u);
  for (char c : id.substr(10))
    EXPECT_TRUE((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));

  // A changed binding changes the id.
  std::map<std::string, int64_t> other{{"BM", 16}, {"BN", 64}};
  EXPECT_NE(id, computeCandidateId(other, symbolic));
}

TEST(CandidateGenerator, RandomSearchIsReproducibleFromItsSeed) {
  SearchSpace space;
  space.params = {
      integerParam("A", {SearchChoice(int64_t{1}), SearchChoice(int64_t{2}),
                         SearchChoice(int64_t{3}), SearchChoice(int64_t{4})}),
      integerParam("B",
                   {SearchChoice(int64_t{10}), SearchChoice(int64_t{20}),
                    SearchChoice(int64_t{30}), SearchChoice(int64_t{40})})};

  auto first = generateRandomCandidates(space, /*seed=*/1234, /*max=*/5);
  auto second = generateRandomCandidates(space, /*seed=*/1234, /*max=*/5);
  ASSERT_EQ(first.size(), 5u);
  EXPECT_EQ(idsOf(first), idsOf(second));
}

TEST(CandidateGenerator, RandomSearchSamplesWithoutReplacement) {
  SearchSpace space;
  space.params = {
      integerParam("A", {SearchChoice(int64_t{1}), SearchChoice(int64_t{2}),
                         SearchChoice(int64_t{3}), SearchChoice(int64_t{4})}),
      integerParam("B",
                   {SearchChoice(int64_t{10}), SearchChoice(int64_t{20}),
                    SearchChoice(int64_t{30}), SearchChoice(int64_t{40})})};

  auto sample = generateRandomCandidates(space, /*seed=*/7, /*max=*/16);
  ASSERT_EQ(sample.size(), 16u); // the space has 16 points
  std::vector<std::string> sampleIds = idsOf(sample);
  std::set<std::string> unique(sampleIds.begin(), sampleIds.end());
  EXPECT_EQ(unique.size(), 16u);

  // Sampling the whole space yields exactly the grid's point set.
  std::vector<std::string> gridIds = idsOf(generateGridCandidates(space));
  std::set<std::string> grid(gridIds.begin(), gridIds.end());
  EXPECT_EQ(unique, grid);
}

TEST(CandidateGenerator, RandomSearchCannotExceedTheSpace) {
  auto sample = generateRandomCandidates(twoByTwo(), /*seed=*/3, /*max=*/100);
  EXPECT_EQ(sample.size(), 4u);
}

TEST(CandidateGenerator, DispatcherSelectsTheMode) {
  CandidateGeneratorOptions options;
  options.mode = SearchMode::Grid;
  options.maxCandidates = 2;
  EXPECT_EQ(generateCandidates(twoByTwo(), options).size(), 2u);

  options.mode = SearchMode::Random;
  options.seed = 42;
  auto random = generateCandidates(twoByTwo(), options);
  ASSERT_EQ(random.size(), 2u);
  EXPECT_EQ(idsOf(random), idsOf(generateRandomCandidates(twoByTwo(), 42, 2)));
}

} // namespace
} // namespace mlir::llk::perf
