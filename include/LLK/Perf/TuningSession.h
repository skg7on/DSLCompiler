//===- TuningSession.h - Generate, check, bind, and rank candidates -------===//
//
// Part of the M12 tuning core (issue #50).
//
// A tuning session is the whole loop the driver runs:
//
//   SearchSpace + WorkloadShape + machine::MachineModel
//     -> generated candidates
//     -> legality verdicts (illegal ones are reported, not run)
//     -> bound concrete micro.kernel per legal candidate
//     -> L0/L1 predicted cost
//     -> ranked top-K
//
// Ranking is by the space's objective. The primary metric is compared first,
// then each secondary metric in the order declared, then the candidate id --
// which is a stable hash of the bindings, so the order never depends on how
// the candidates were generated or on the iteration order of a hash container.
//
// A candidate that is legal but cannot be bound, and a kernel whose analysis
// fails, are rejected with a stable reason rather than silently dropped: the
// report has to account for every generated candidate.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_PERF_TUNINGSESSION_H
#define LLK_PERF_TUNINGSESSION_H

#include "LLK/Machine/MachineModel.h"
#include "LLK/Perf/Candidate.h"
#include "LLK/Perf/CandidateBinding.h"
#include "LLK/Perf/CandidateGenerator.h"
#include "LLK/Perf/ScheduleRecord.h"
#include "LLK/Perf/SearchSpace.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace mlir::llk::perf {

struct TuningSessionOptions {
  CandidateGeneratorOptions generator;
  /// 0 static bound only, 1 also schedule resources.
  unsigned perfLevel = 1;
  /// How many ranked candidates to keep. Zero keeps every legal candidate.
  uint64_t topK = 10;
  /// Path the machine model was read from, recorded in every schedule record.
  std::string machinePath;
};

/// One legal candidate with the cost it was predicted and the decisions it
/// bound to.
struct RankedCandidate {
  TuningResult result;
  BoundTileDecisions decisions;
};

struct TuningSessionReport {
  std::string machinePath;
  std::string machineName;
  unsigned perfLevel = 1;

  /// Every candidate the generator produced.
  uint64_t generated = 0;
  /// Legal candidates, best first, at most `topK` of them.
  std::vector<RankedCandidate> ranked;
  /// Illegal (or unbindable) candidates, in generation order, each with a
  /// stable rejection reason.
  std::vector<TuningResult> rejected;
};

/// True when `a` ranks before `b` under `objective`: primary metric first, then
/// the secondary metrics in order, then the candidate id ascending. The metric
/// name maps to a CandidateMetrics field; an unknown name is neutral, so it
/// never reorders the result.
bool ranksBefore(const TuningResult &a, const TuningResult &b,
                 const SearchObjective &objective);

/// Sorts a copy of `results` with ranksBefore().
std::vector<TuningResult> rankTuningResults(std::vector<TuningResult> results,
                                            const SearchObjective &objective);

/// Runs the loop above. `context` must have the micro, tensor, and arith
/// dialects registered; each candidate is bound into a scratch module the
/// session creates and destroys, so the caller's IR is never touched.
llvm::Expected<TuningSessionReport>
runTuningSession(mlir::MLIRContext &context, const SearchSpace &space,
                 const WorkloadShape &shape,
                 const machine::MachineModel &machine,
                 const TuningSessionOptions &options = {});

/// Turns the ranked candidates into schedule records, carrying the workload
/// identity and tile decisions `llk-tune` persists.
std::vector<ScheduleRecord>
buildScheduleRecords(const TuningSessionReport &report,
                     const SearchSpace &space, const WorkloadShape &shape);

} // namespace mlir::llk::perf

#endif // LLK_PERF_TUNINGSESSION_H
