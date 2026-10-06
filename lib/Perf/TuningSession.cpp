//===- TuningSession.cpp - Generate, check, bind, and rank candidates -----===//
//
// Part of the M12 tuning core (issue #50). See TuningSession.h.
//
// Each candidate is bound into its own scratch module: the bound kernel is IR,
// and the session needs one alive only long enough to cost it. A fresh module
// per candidate also means a candidate that fails to bind cannot leave half a
// kernel behind for the next one to trip over.
//
//===----------------------------------------------------------------------===//

#include "LLK/Perf/TuningSession.h"

#include "LLK/Perf/Legality.h"
#include "LLK/Perf/MicroPerfReport.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <string>
#include <vector>

namespace mlir::llk::perf {

using llvm::StringRef;

namespace {

/// The CandidateMetrics field a metric name selects. `validateObjective` is
/// what rejects a name with no field, so this is only reached for names the
/// model produces.
double metricValue(const CandidateMetrics &metrics, StringRef name) {
  if (name == "latency_cycles")
    return static_cast<double>(metrics.predictedCycles);
  if (name == "dram_bytes")
    return static_cast<double>(metrics.dramBytes);
  if (name == "sram_bytes")
    return static_cast<double>(metrics.sramBytes);
  if (name == "matrix_utilization")
    return metrics.matrixUtilization;
  if (name == "dma_utilization")
    return metrics.dmaUtilization;
  // A measured metric falls back to its static counterpart when no measurement
  // was taken: a provider *miss* leaves the static score standing, which is a
  // different statement from scoring the candidate zero.
  if (name == "measured_ns")
    return metrics.measuredNs.value_or(metrics.predictedNs);
  if (name == "measured_gflops")
    return metrics.measuredGflops.value_or(0.0);
  // Every other name is refused by validateObjective before a ranking runs, so
  // reaching here would mean the check was skipped.
  return 0.0;
}

/// Utilization metrics are the ones where more is better; everything else the
/// objective can name is a cost. Secondary metrics declare no direction, so
/// their sense comes from what they measure.
bool lowerIsBetter(StringRef name) {
  return name != "matrix_utilization" && name != "dma_utilization" &&
         name != "measured_gflops";
}

CandidateMetrics metricsFrom(const MicroPerfReport &report,
                             const machine::MachineModel &machine) {
  CandidateMetrics metrics;
  if (report.l1) {
    metrics.predictedCycles = report.l1->predictedCycles;
    metrics.predictedNs = report.l1->predictedNs;
    metrics.matrixUtilization = report.l1->matrixUtilization;
    metrics.dmaUtilization = report.l1->dmaUtilization;
    metrics.bottleneck = report.l1->bottleneck;
  } else {
    metrics.predictedCycles = report.l0.predictedCycles;
    metrics.predictedNs = machine.clockHz
                              ? static_cast<double>(metrics.predictedCycles) *
                                    1e9 / static_cast<double>(*machine.clockHz)
                              : 0.0;
    metrics.bottleneck = report.l0.bottleneck;
  }
  metrics.dramBytes = report.l0.totalBytesDram;
  metrics.sramBytes = report.l0.totalBytesSram;
  return metrics;
}

} // namespace

bool isKnownMetric(StringRef name) {
  return name == "latency_cycles" || name == "dram_bytes" ||
         name == "sram_bytes" || name == "matrix_utilization" ||
         name == "dma_utilization" || name == "measured_ns" ||
         name == "measured_gflops";
}

llvm::Error validateObjective(const SearchObjective &objective) {
  auto check = [](StringRef name) -> llvm::Error {
    if (isKnownMetric(name))
      return llvm::Error::success();
    return llvm::make_error<llvm::StringError>(
        ("objective names metric '" + name +
         "', which the tuner does not produce; it ranks by latency_cycles, "
         "dram_bytes, sram_bytes, matrix_utilization, dma_utilization, "
         "measured_ns or measured_gflops")
            .str(),
        llvm::inconvertibleErrorCode());
  };
  if (llvm::Error error = check(objective.primaryMetric))
    return error;
  for (const std::string &name : objective.secondaryMetrics)
    if (llvm::Error error = check(name))
      return error;
  return llvm::Error::success();
}

bool ranksBefore(const TuningResult &a, const TuningResult &b,
                 const SearchObjective &objective) {
  auto compare = [](double left, double right, bool lower) {
    if (left == right)
      return 0;
    return (left < right) == lower ? -1 : 1;
  };

  int primary = compare(metricValue(a.metrics, objective.primaryMetric),
                        metricValue(b.metrics, objective.primaryMetric),
                        objective.direction == ObjectiveDirection::Minimize);
  if (primary != 0)
    return primary < 0;

  for (const std::string &name : objective.secondaryMetrics) {
    int secondary = compare(metricValue(a.metrics, name),
                            metricValue(b.metrics, name), lowerIsBetter(name));
    if (secondary != 0)
      return secondary < 0;
  }

  return a.candidate.id < b.candidate.id;
}

std::vector<TuningResult> rankTuningResults(std::vector<TuningResult> results,
                                            const SearchObjective &objective) {
  std::stable_sort(results.begin(), results.end(),
                   [&](const TuningResult &a, const TuningResult &b) {
                     return ranksBefore(a, b, objective);
                   });
  return results;
}

llvm::Expected<TuningSessionReport>
runTuningSession(mlir::MLIRContext &context, const SearchSpace &space,
                 const WorkloadShape &shape,
                 const machine::MachineModel &machine,
                 const TuningSessionOptions &options) {
  if (options.perfLevel > 1)
    return llvm::make_error<llvm::StringError>(
        "unsupported performance level; the tuner knows 0 and 1",
        llvm::inconvertibleErrorCode());
  // A ranking under a metric the model does not produce would order every
  // candidate equally and still look like a decision, so the objective is
  // checked before anything is generated.
  if (llvm::Error error = validateObjective(space.objective))
    return std::move(error);

  TuningSessionReport report;
  report.objective = space.objective;
  report.machinePath = options.machinePath;
  report.machineName = machine.target;
  report.perfLevel = options.perfLevel;

  std::vector<Candidate> candidates =
      generateCandidates(space, options.generator);
  report.generated = candidates.size();

  for (const Candidate &candidate : candidates) {
    LegalityResult legality = checkLegality(space, candidate, shape, machine);
    if (!legality.legal) {
      report.rejected.push_back(
          TuningResult{candidate, CandidateMetrics(), false, legality.reason});
      continue;
    }

    OwningOpRef<ModuleOp> scratch = ModuleOp::create(UnknownLoc::get(&context));
    llvm::Expected<BoundKernel> bound =
        bindCandidateToMicroKernel(scratch.get(), space, candidate, shape);
    if (!bound) {
      report.rejected.push_back(
          TuningResult{candidate, CandidateMetrics(), false,
                       "binding: " + llvm::toString(bound.takeError())});
      continue;
    }

    llvm::Expected<MicroPerfReport> perf =
        analyzeKernel(bound->kernel, machine, options.perfLevel);
    if (!perf) {
      report.rejected.push_back(
          TuningResult{candidate, CandidateMetrics(), false,
                       "analysis: " + llvm::toString(perf.takeError())});
      continue;
    }

    RankedCandidate ranked;
    ranked.result =
        TuningResult{candidate, metricsFrom(*perf, machine), true, ""};
    ranked.decisions = std::move(bound->decisions);
    report.ranked.push_back(std::move(ranked));
  }

  std::sort(report.ranked.begin(), report.ranked.end(),
            [&](const RankedCandidate &a, const RankedCandidate &b) {
              return ranksBefore(a.result, b.result, space.objective);
            });
  if (options.topK != 0 && report.ranked.size() > options.topK)
    report.ranked.resize(options.topK);

  // Measurement is optional and comes last: it measures the best candidates the
  // ranking just chose, so it can only ever confirm or reject them, never
  // change which ones they are.
  if (llvm::Error error = measureTopCandidates(context, space, shape, report,
                                               options.measurement))
    return std::move(error);

  return report;
}

llvm::Error measureTopCandidates(mlir::MLIRContext &context,
                                 const SearchSpace &space,
                                 const WorkloadShape &shape,
                                 TuningSessionReport &report,
                                 const MeasurementOptions &options) {
  if (!options.provider || options.measureTop == 0)
    return llvm::Error::success();

  const size_t count = std::min<size_t>(static_cast<size_t>(options.measureTop),
                                        report.ranked.size());

  std::vector<size_t> rejectedIndices;
  for (size_t index = 0; index < count; ++index) {
    RankedCandidate &ranked = report.ranked[index];

    // Binding is deterministic, so binding again gives the module the ranking
    // described. Re-binding is how a candidate reaches the provider without the
    // session holding every module it ever bound.
    OwningOpRef<ModuleOp> scratch = ModuleOp::create(UnknownLoc::get(&context));
    llvm::Expected<BoundKernel> bound = bindCandidateToMicroKernel(
        scratch.get(), space, ranked.result.candidate, shape);
    if (!bound) {
      report.rejected.push_back(TuningResult{
          ranked.result.candidate, ranked.result.metrics, false,
          "measurement binding: " + llvm::toString(bound.takeError())});
      rejectedIndices.push_back(index);
      continue;
    }

    llvm::Expected<std::optional<CandidateMetrics>> observed =
        options.provider(scratch.get(), ranked.result);
    if (!observed) {
      // A candidate that cannot be compiled or verified is rejected rather than
      // ranked: its predicted cost is a claim about code that does not exist.
      report.rejected.push_back(
          TuningResult{ranked.result.candidate, ranked.result.metrics, false,
                       "measurement: " + llvm::toString(observed.takeError())});
      rejectedIndices.push_back(index);
      continue;
    }
    if (*observed) {
      ranked.measured = **observed;
      ranked.measuredTarget = options.targetIdentity;
      ranked.measuredMachine = options.machineIdentity;
      ranked.measuredAbi = options.abiIdentity;
    }
    // A miss leaves the candidate exactly as it was: still ranked, still
    // legal, with its static score standing.
  }

  // Removed back to front so the surviving order is untouched.
  for (auto it = rejectedIndices.rbegin(); it != rejectedIndices.rend(); ++it)
    report.ranked.erase(report.ranked.begin() + *it);
  return llvm::Error::success();
}

std::vector<ScheduleRecord>
buildScheduleRecords(const TuningSessionReport &report,
                     const SearchSpace &space, const WorkloadShape &shape) {
  std::vector<ScheduleRecord> records;
  records.reserve(report.ranked.size());
  for (const RankedCandidate &ranked : report.ranked) {
    ScheduleRecord record;
    record.workload = space.workload;
    record.shape = shape;
    record.target = report.machineName;
    record.machine = report.machinePath;
    record.candidate = ranked.result.candidate;
    record.tile = ranked.decisions;
    record.metrics = ranked.result.metrics;
    record.perfLevel = report.perfLevel;
    records.push_back(std::move(record));
  }
  return records;
}

} // namespace mlir::llk::perf
