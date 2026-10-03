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

/// The CandidateMetrics field a metric name selects. A name the model does not
/// produce reads as zero, which is neutral: it can only tie, never reorder.
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
  // capacity_spill_bytes is named by the spec but not modeled yet; a metric
  // the model cannot produce must not decide an order.
  return 0.0;
}

/// Utilization metrics are the ones where more is better; everything else the
/// objective can name is a cost. Secondary metrics declare no direction, so
/// their sense comes from what they measure.
bool lowerIsBetter(StringRef name) {
  return name != "matrix_utilization" && name != "dma_utilization";
}

CandidateMetrics metricsFrom(const MicroPerfReport &report,
                             const MachineModel &machine) {
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
                                    1e9 / static_cast<double>(machine.clockHz)
                              : 0.0;
    metrics.bottleneck = report.l0.bottleneck;
  }
  metrics.dramBytes = report.l0.totalBytesDram;
  metrics.sramBytes = report.l0.totalBytesSram;
  return metrics;
}

} // namespace

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
                 const WorkloadShape &shape, const MachineModel &machine,
                 const TuningSessionOptions &options) {
  if (options.perfLevel > 1)
    return llvm::make_error<llvm::StringError>(
        "unsupported performance level; the tuner knows 0 and 1",
        llvm::inconvertibleErrorCode());

  TuningSessionReport report;
  report.machinePath = options.machinePath;
  report.machineName = machine.name;
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

  return report;
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
