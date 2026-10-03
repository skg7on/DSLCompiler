//===- CandidateGenerator.h - Deterministic candidate enumeration --------===//
//
// Part of the M12 tuning core (issue #49).
//
// The MVP generator is a Cartesian grid: every parameter's choices are crossed
// in declaration order, so the first candidate is the schedule's preferred
// value for every parameter and the order never depends on iteration order of
// a hash container. Grid search is what the FileCheck and GTest suites pin.
//
// Random search samples without replacement from the same domain with a
// caller-supplied seed, so a study is reproducible: the same space and the same
// seed produce the same candidates in the same order. It is not a different
// search *distribution* -- it is a way to cap a large space at N samples.
//
// Neither generator performs legality checks. Legality is a property of the
// candidate, the workload, and the machine, and it is evaluated by Legality.h
// so generation stays cheap and side-effect free.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_PERF_CANDIDATEGENERATOR_H
#define LLK_PERF_CANDIDATEGENERATOR_H

#include "LLK/Perf/Candidate.h"
#include "LLK/Perf/SearchSpace.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace mlir::llk::perf {

enum class SearchMode { Grid, Random };

struct CandidateGeneratorOptions {
  SearchMode mode = SearchMode::Grid;
  /// RNG seed; only read in Random mode.
  uint64_t seed = 0;
  /// Stop after this many candidates. Absent means the whole space, for either
  /// mode: grid enumerates it in order, random samples all of it.
  std::optional<uint64_t> maxCandidates;
};

/// Every point of the Cartesian product, parameters in declaration order and
/// choices in declared order. `maxCandidates` caps the result; the prefix of
/// the uncapped sequence is kept, so a cap never reshuffles what remains.
std::vector<Candidate>
generateGridCandidates(const SearchSpace &space,
                       std::optional<uint64_t> maxCandidates = std::nullopt);

/// Up to `maxCandidates` distinct points drawn with `seed`. When the space is
/// no larger than the request, every point is returned. The RNG is seeded only
/// by `seed`, never by time or address, so the sequence is reproducible.
std::vector<Candidate> generateRandomCandidates(const SearchSpace &space,
                                                uint64_t seed,
                                                uint64_t maxCandidates);

/// Dispatches on `options.mode`.
std::vector<Candidate>
generateCandidates(const SearchSpace &space,
                   const CandidateGeneratorOptions &options = {});

} // namespace mlir::llk::perf

#endif // LLK_PERF_CANDIDATEGENERATOR_H
