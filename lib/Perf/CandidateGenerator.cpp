//===- CandidateGenerator.cpp - Deterministic candidate enumeration ------===//
//
// Part of the M12 tuning core (issue #49). See CandidateGenerator.h.
//
// Both generators share one `buildCandidate()` so a grid point and a random
// point with the same bindings are byte-identical, ids included. That is what
// lets a random study be compared against the grid without a second notion of
// equality.
//
//===----------------------------------------------------------------------===//

#include "LLK/Perf/CandidateGenerator.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <random>

namespace mlir::llk::perf {

namespace {

/// Builds one candidate from a choice index per parameter.
Candidate buildCandidate(const SearchSpace &space,
                         llvm::ArrayRef<size_t> indices) {
  Candidate candidate;
  for (size_t i = 0; i < space.params.size(); ++i) {
    const SearchParam &param = space.params[i];
    const SearchChoice &choice = param.choices[indices[i]];
    if (choice.isInteger())
      candidate.values.emplace(param.name, choice.integer());
    else
      candidate.symbolicValues.emplace(param.name, choice.symbol().str());
  }
  candidate.id = computeCandidateId(candidate.values, candidate.symbolicValues);
  return candidate;
}

/// The number of points in the space, saturating rather than wrapping. Zero
/// when any parameter declares no choices.
uint64_t domainSize(const SearchSpace &space) {
  if (space.params.empty())
    return 0;
  uint64_t size = 1;
  for (const SearchParam &param : space.params) {
    if (param.choices.empty())
      return 0;
    if (size > UINT64_MAX / param.choices.size())
      return UINT64_MAX;
    size *= param.choices.size();
  }
  return size;
}

} // namespace

std::vector<Candidate>
generateGridCandidates(const SearchSpace &space,
                       std::optional<uint64_t> maxCandidates) {
  std::vector<Candidate> candidates;
  if (domainSize(space) == 0)
    return candidates;
  uint64_t cap = maxCandidates.value_or(UINT64_MAX);

  // Odometer over the choice indices, the last parameter varying fastest, so
  // the first candidate is every parameter's preferred (first) choice.
  std::vector<size_t> indices(space.params.size(), 0);
  while (candidates.size() < cap) {
    candidates.push_back(buildCandidate(space, indices));

    size_t position = indices.size();
    while (position > 0) {
      --position;
      if (++indices[position] < space.params[position].choices.size())
        break;
      indices[position] = 0;
    }
    if (position == 0 && indices[0] == 0)
      break; // wrapped past the first parameter: the product is exhausted
  }
  return candidates;
}

std::vector<Candidate> generateRandomCandidates(const SearchSpace &space,
                                                uint64_t seed,
                                                uint64_t maxCandidates) {
  std::vector<Candidate> candidates;
  uint64_t domain = domainSize(space);
  if (domain == 0 || maxCandidates == 0)
    return candidates;

  const uint64_t target = std::min(maxCandidates, domain);
  std::mt19937_64 rng(seed);
  llvm::StringSet<> seen;
  std::vector<size_t> indices(space.params.size(), 0);

  // Rejection sampling with a bounded attempt count: collecting `target` of
  // `domain` points needs on the order of `domain * ln(...)` draws, far below
  // this cap, and the loop also stops as soon as the space is exhausted. The
  // cap saturates so a saturated domain cannot wrap it to zero attempts.
  const uint64_t maxAttempts =
      target > (UINT64_MAX - 64) / 64 ? UINT64_MAX : 64 * target + 64;
  for (uint64_t attempt = 0;
       attempt < maxAttempts && candidates.size() < target; ++attempt) {
    for (size_t i = 0; i < space.params.size(); ++i)
      // std::mt19937_64's output sequence is fully specified, while
      // std::uniform_int_distribution's is not; taking the modulus keeps the
      // study reproducible across standard libraries.
      indices[i] = rng() % space.params[i].choices.size();

    Candidate candidate = buildCandidate(space, indices);
    if (!seen.insert(candidate.id).second)
      continue;
    candidates.push_back(std::move(candidate));
    if (candidates.size() == domain)
      break;
  }
  return candidates;
}

std::vector<Candidate>
generateCandidates(const SearchSpace &space,
                   const CandidateGeneratorOptions &options) {
  switch (options.mode) {
  case SearchMode::Grid:
    return generateGridCandidates(space, options.maxCandidates);
  case SearchMode::Random:
    // An absent cap means "sample the whole space", which is finite.
    return generateRandomCandidates(
        space, options.seed, options.maxCandidates.value_or(domainSize(space)));
  }
  return {};
}

} // namespace mlir::llk::perf
