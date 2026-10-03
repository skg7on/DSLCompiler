//===- Candidate.h - One search-space binding and its metrics -------------===//
//
// Part of the M12 tuning core (issue #49).
//
// A Candidate is one point in the search space: every declared parameter bound
// to exactly one value. Integer bindings (tile sizes, stages, widths) live in
// `values`; symbolic bindings (layout, memory path, owner mapping, fragment
// shape, tail policy) live in `symbolicValues`. Keeping the two maps separate
// means integer handling stays simple while string-valued decisions remain
// first-class.
//
// The id is a stable 64-bit hash of the sorted bindings, not a counter, so the
// same point gets the same name no matter how it was generated or in what
// order it was visited. That is what makes ranking and schedule records
// reproducible across runs.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_PERF_CANDIDATE_H
#define LLK_PERF_CANDIDATE_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>

namespace mlir::llk::perf {

/// One parameter binding. Ids are `candidate_<16 lowercase hex digits>`.
struct Candidate {
  std::string id;
  std::map<std::string, int64_t> values;
  std::map<std::string, std::string> symbolicValues;

  std::optional<int64_t> integer(llvm::StringRef name) const;
  std::optional<llvm::StringRef> symbol(llvm::StringRef name) const;
};

/// Predicted (and optionally measured) cost of one candidate. Filled by the
/// ranking stage (#50) from the L0/L1 models; the tuning core only carries it.
struct CandidateMetrics {
  uint64_t predictedCycles = 0;
  double predictedNs = 0.0;
  double matrixUtilization = 0.0;
  double dmaUtilization = 0.0;
  uint64_t dramBytes = 0;
  uint64_t sramBytes = 0;
  std::string bottleneck;
  std::optional<double> measuredNs;
  std::optional<double> measuredGflops;
};

/// One candidate with its legality verdict and metrics.
struct TuningResult {
  Candidate candidate;
  CandidateMetrics metrics;
  bool legal = false;
  std::string rejectionReason;
};

/// A stable id for the bindings in `values` and `symbolicValues`: FNV-1a over
/// the keys and values in sorted order. Deliberately not `llvm::hash_value`,
/// whose result is not promised to be stable across toolchain versions.
std::string
computeCandidateId(const std::map<std::string, int64_t> &values,
                   const std::map<std::string, std::string> &symbolicValues);

} // namespace mlir::llk::perf

#endif // LLK_PERF_CANDIDATE_H
