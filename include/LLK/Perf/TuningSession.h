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

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace mlir {
class MLIRContext;
} // namespace mlir

namespace mlir::llk::perf {

enum class MetricOrigin { Static, Measured, Unavailable };

struct MetricValue {
  std::optional<double> value;
  MetricOrigin origin = MetricOrigin::Unavailable;
};

/// Resolves an objective metric without substituting a different metric when
/// a measurement is unavailable. Unknown or non-finite values are unavailable.
MetricValue resolveMetric(const CandidateMetrics &metrics,
                          llvm::StringRef name);

/// The metrics a tuning objective may name.
bool isKnownMetric(llvm::StringRef name);

/// Checks that every metric an objective names is one the tuner can rank by.
///
/// A name outside that set is a *usage error*, not a neutral zero. An objective
/// naming a metric the model never computes would rank every candidate equal
/// under it and still look like it had decided something; the design names
/// `capacity_spill_bytes`, which is not modelled, so it is refused for exactly
/// that reason rather than quietly ignored.
llvm::Error validateObjective(const SearchObjective &objective);

/// Measures one ranked candidate.
///
/// The session hands over the candidate's bound module; a provider that wants a
/// running kernel compiles it -- `compileMappedKernel` is what a mapping caller
/// passes -- and invokes it. Keeping the compilation on this side is what keeps
/// the tuning core off the JIT while still making a measurement an answer about
/// code that actually ran.
///
/// The two failures are different and are reported differently:
///   * `llvm::Error` -- the candidate could not be compiled or verified, which
///     is a rejection carrying that reason;
///   * `std::nullopt` -- a *miss*: static objectives retain the static score;
///     measured objectives report the legal candidate as unrankable.
using MeasurementProvider =
    std::function<llvm::Expected<std::optional<CandidateMetrics>>(
        mlir::ModuleOp boundModule, const TuningResult &candidate)>;

struct MeasurementOptions {
  MeasurementProvider provider;
  /// How many of the best-ranked candidates to measure. Zero measures none.
  uint64_t measureTop = 1;
  /// Maximum candidates visited while filling `measureTop`. Zero means no
  /// additional cap beyond the static candidate list.
  uint64_t maxAttempts = 0;
  /// Recorded on every measurement so a stored number can be traced to the
  /// target, machine and model that produced it. Production persistence (and
  /// the calibration that reads it back) stays with #51/#52.
  std::string targetIdentity;
  std::string machineIdentity;
  std::string abiIdentity;
  /// Compiler/codegen identity (backend, ISA, math mode and target content).
  std::string codegenIdentity;
};

struct TuningSessionOptions {
  CandidateGeneratorOptions generator;
  /// 0 static bound only, 1 also schedule resources.
  unsigned perfLevel = 1;
  /// How many ranked candidates to keep. Zero keeps every legal candidate.
  uint64_t topK = 10;
  /// Path the machine model was read from, recorded in every schedule record.
  std::string machinePath;
  /// Optional measurement of the best candidates. Empty measures none, so the
  /// default session is exactly the static ranking it always was.
  MeasurementOptions measurement;
};

/// One legal candidate with the cost it was predicted and the decisions it
/// bound to.
struct RankedCandidate {
  TuningResult result;
  BoundTileDecisions decisions;
  /// What a measurement observed, when one ran. Unset for a candidate that was
  /// never measured, and for one the provider could not measure.
  std::optional<CandidateMetrics> measured;
  /// The target, machine and ABI the measurement was taken against. A stored
  /// cycle count is only interpretable with these, which is why they travel
  /// with it rather than being looked up later.
  std::string measuredTarget;
  std::string measuredMachine;
  std::string measuredAbi;
  std::string measuredCodegenIdentity;
};

struct TuningSessionReport {
  std::string machinePath;
  std::string machineName;
  unsigned perfLevel = 1;

  /// Every candidate the generator produced.
  uint64_t generated = 0;
  /// Comparable candidates, best first, at most `topK` of them.
  std::vector<RankedCandidate> ranked;
  /// Legal candidates for which the requested measured objective is missing.
  std::vector<RankedCandidate> unrankable;
  bool measuredCohortOnly = false;
  uint64_t measuredCohortSize = 0;
  uint64_t measurementAttempts = 0;
  bool hasMeasuredResult = false;
  /// Illegal, unbindable, or uncompilable candidates, in generation order,
  /// each with a stable rejection reason.
  std::vector<TuningResult> rejected;

  /// The objective the ranking used, recorded so a reader can tell what "best"
  /// meant rather than having to find the search space again.
  SearchObjective objective;
  /// The report schema, bumped when a field's meaning changes, so a stored
  /// report is never read as a different one.
  uint32_t schemaVersion = 2;
};

/// Compiles and measures the best-ranked candidates.
///
/// Each is bound again into a scratch module -- binding is deterministic, so
/// the module measured is the one the ranking described -- and handed to the
/// provider. A candidate whose compilation fails moves to `rejected` with the
/// compiler's reason, because a plan that cannot be turned into code is not a
/// candidate whatever its predicted cost. A provider miss preserves legality;
/// static objectives keep its static ranking, while measured objectives place
/// it in `TuningSessionReport::unrankable`.
llvm::Error measureTopCandidates(mlir::MLIRContext &context,
                                 const SearchSpace &space,
                                 const WorkloadShape &shape,
                                 TuningSessionReport &report,
                                 const MeasurementOptions &options);

/// True when `a` ranks before `b` under `objective`: primary metric first, then
/// the secondary metrics in order, then the candidate id ascending. The metric
/// name maps to a CandidateMetrics field; `validateObjective` is what rejects a
/// name with no field, so a ranking that runs at all is ranking by something
/// the model produces.
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
