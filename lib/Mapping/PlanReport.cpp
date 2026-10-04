//===- PlanReport.cpp - Versioned JSON plan report (design §22.2) ---------===//
//
// The report is written with `llvm::json::OStream`'s imperative API rather than
// from a `json::Object`: an Object is backed by a `DenseMap`, whose iteration
// order is not a contract, whereas OStream emits attributes in exactly the
// order the code writes them. That gives the fixed key order and the
// byte-for-byte determinism the report needs (§29.12).
//
//===----------------------------------------------------------------------===//

#include "LLK/Mapping/PlanReport.h"

#include "LLK/Mapping/CostModel.h"
#include "LLK/Mapping/Diagnostics.h"
#include "LLK/Mapping/LatencyProvider.h"
#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/StableHash.h"
#include "LLK/Version.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdio>
#include <string>

namespace mlir::llk::mapping {

namespace {

/// The stable spelling of a search mode. Kept local: `SearchMode` has no
/// stringifier because nothing else needs to print it.
llvm::StringRef stringifySearchMode(SearchMode mode) {
  switch (mode) {
  case SearchMode::Deterministic:
    return "deterministic";
  case SearchMode::Beam:
    return "beam";
  case SearchMode::Exact:
    return "exact";
  }
  return "unknown";
}

/// True when `code` names a *rejection* -- something the search refused -- as
/// opposed to a notice. §22.2 asks for "rejected counts", so a notice never
/// inflates the rejection tally.
///
/// Every code is classified explicitly and there is deliberately no `default`:
/// a code added to the enum without a case here makes this switch incomplete
/// (`-Wswitch`), rather than silently defaulting into the rejection bucket.
/// `AssumedValueSize` is the case that motivated it -- its own documentation
/// says it is not an error, so it must land under notices.
bool isRejection(DiagnosticCode code) {
  switch (code) {
  // Notices: a cap, a provider gap, or an advisory assumption. None is a
  // refusal the search made.
  case DiagnosticCode::SearchTruncated:
  case DiagnosticCode::LatencyCacheMiss:
  case DiagnosticCode::AssumedValueSize:
    return false;
  // Rejections: the search refused a rule, a placement, a pair, a layout, a
  // global constraint, a bundle, or a plan.
  case DiagnosticCode::NoMatchingRule:
  case DiagnosticCode::NoLegalLayout:
  case DiagnosticCode::NoLegalExecutor:
  case DiagnosticCode::MemoryCapacityExceeded:
  case DiagnosticCode::UnsupportedComputeFragment:
  case DiagnosticCode::NoMemoryRoute:
  case DiagnosticCode::NoLayoutTransform:
  case DiagnosticCode::GlobalConstraintFailed:
  case DiagnosticCode::TargetBundleInvalid:
    return true;
  }
  llvm_unreachable("unclassified DiagnosticCode");
}

/// Fixed six-decimal rendering of a double, matching `canonicalCostString`, so
/// a cost component is byte-stable and never printed in exponent form.
/// `snprintf`'s `%f` honours the C locale's decimal separator; the report
/// assumes `LC_NUMERIC=C`, which is the process default and what the rest of
/// the toolchain (including `canonicalCostString`) already relies on.
std::string fixedDouble(double value) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.6f", value);
  return std::string(buffer);
}

} // namespace

llvm::StringRef compilerVersion() { return LLK_COMPILER_VERSION; }

std::string writePlanReport(const MappingSearchResult &result,
                            const machine::MachineModel &machine,
                            const MappingTarget &target,
                            const MappingSearchOptions &options,
                            uint64_t moduleHash) {
  const CoveringPlan *selected =
      result.plans.empty() ? nullptr : &result.plans.front();

  std::string text;
  llvm::raw_string_ostream stream(text);
  llvm::json::OStream json(stream, /*IndentSize=*/2);

  json.object([&] {
    // --- provenance (fixed key order) -----------------------------------
    json.attribute("version", kPlanReportVersion);
    json.attribute("compilerVersion", compilerVersion());
    json.attribute("costModelVersion", kCostModelVersion);
    json.attribute("target", target.name());
    json.attribute("inputModuleHash", hexId(moduleHash));
    json.attribute("sourceBindingHash",
                   hexId(selected ? selected->sourceBindingHash : 0));
    // The source search binding itself, when the search was given one: its
    // sorted, type-tagged parameter values. Empty for a binding-free search,
    // where the hash is zero too.
    json.attribute("sourceBinding", selected ? canonicalSearchValueString(
                                                   selected->globalParameters)
                                             : std::string());
    // Computed rather than read off the model: an in-memory model has no
    // cached hash, and the report must never print a stale or zero one.
    json.attribute("machineHash", hexId(machine::computeContentHash(machine)));
    json.attribute("layoutLibraryHash",
                   hexId(target.layouts().computeContentHash()));
    json.attribute("ruleLibraryHash",
                   hexId(target.rules().computeContentHash()));

    // --- options --------------------------------------------------------
    json.attributeObject("searchOptions", [&] {
      json.attribute("mode", stringifySearchMode(options.mode));
      json.attribute("beamWidth", static_cast<uint64_t>(options.beamWidth));
      json.attribute("topK", static_cast<uint64_t>(options.topK));
      json.attribute("maxCandidatesPerNode",
                     static_cast<uint64_t>(options.maxCandidatesPerNode));
      json.attribute("maxInstancesPerCandidate",
                     static_cast<uint64_t>(options.maxInstancesPerCandidate));
      json.attribute("maxRoutesPerConnection",
                     static_cast<uint64_t>(options.maxRoutesPerConnection));
      json.attribute("memoryBudgetBytes", options.memoryBudgetBytes);
      json.attribute("enableLatencyCache", options.enableLatencyCache);
      json.attribute("enableSymmetryReduction",
                     options.enableSymmetryReduction);
      json.attributeObject("objective", [&] {
        json.attribute("primary",
                       stringifyCostMetric(options.objective.primary));
        json.attribute("minimize", options.objective.minimize);
        json.attributeArray("secondary", [&] {
          for (CostMetric metric : options.objective.secondary)
            json.value(stringifyCostMetric(metric));
        });
      });
    });

    json.attribute("searchTruncated", result.searchTruncated);

    // --- counts ---------------------------------------------------------
    json.attributeObject("counts", [&] {
      json.attribute("candidates", result.candidateCount);
      json.attribute("instances", result.instanceCount);
      json.attribute("routes", result.routeCount);
      json.attribute("plans", result.planCount);
    });

    // --- coded events grouped by stable code ----------------------------
    // `codeCounts` is a `std::map`, so iteration is by code and the grouping is
    // deterministic. §22.2's "rejected counts" are separated from notices: a
    // cap hit or a cache miss is not a rejection, so it never inflates the
    // rejection tally.
    json.attributeArray("rejections", [&] {
      for (const auto &entry : result.frontier.codeCounts) {
        if (!isRejection(entry.first))
          continue;
        json.object([&] {
          json.attribute("code", stringifyDiagnosticCode(entry.first));
          json.attribute("count", entry.second);
        });
      }
    });
    json.attributeArray("notices", [&] {
      for (const auto &entry : result.frontier.codeCounts) {
        if (isRejection(entry.first))
          continue;
        json.object([&] {
          json.attribute("code", stringifyDiagnosticCode(entry.first));
          json.attribute("count", entry.second);
        });
      }
    });

    // --- top-K plans with component costs -------------------------------
    json.attributeArray("plans", [&] {
      for (size_t rank = 0; rank < result.plans.size(); ++rank) {
        const CoveringPlan &plan = result.plans[rank];
        json.object([&] {
          // The plan's content id. It is a snapshot for correlation, *not*
          // something a reader can recompute from this document: the id folds
          // content the summary below does not emit -- since phase-3 T4 the
          // solved layout parameterization of every instance and placement
          // (`CoveringPlan` -> `PlanPlacement::layoutSolutions`), as well as
          // the placements' layouts -- so an id cannot be re-derived from the
          // report's content. (Re-emitting the same search's report does carry
          // every id through verbatim; it just cannot be recomputed from what
          // the document shows.)
          json.attribute("id", hexId(plan.id));
          json.attribute("rank", static_cast<uint64_t>(rank));
          json.attribute("sourceBindingHash", hexId(plan.sourceBindingHash));
          json.attribute("instanceCount",
                         static_cast<uint64_t>(plan.instances.size()));
          json.attribute("placementCount",
                         static_cast<uint64_t>(plan.placements.size()));
          json.attribute("routeCount",
                         static_cast<uint64_t>(plan.connectionPlans.size()));
          // The canonical string is the whole cost vector; the broken-out
          // object is the same numbers per dimension, byte-stable.
          json.attribute("totalCost", canonicalCostString(plan.totalCost));
          json.attributeObject("costComponents", [&] {
            json.attribute("latencyCycles",
                           fixedDouble(plan.totalCost.latencyCycles));
            json.attribute("dramBytes", plan.totalCost.dramBytes);
            json.attribute("localBytes", plan.totalCost.localBytes);
            json.attribute("spillBytes", plan.totalCost.spillBytes);
            json.attribute("computeUtilization",
                           fixedDouble(plan.totalCost.computeUtilization));
            json.attribute("transferUtilization",
                           fixedDouble(plan.totalCost.transferUtilization));
          });
          json.attributeObject("diagnostics", [&] {
            json.attribute("searchTruncated", plan.diagnostics.searchTruncated);
            // `errors`/`warnings` are never written by the search, so these
            // counts are always 0 today; they are emitted so the report shape
            // is stable for a future producer rather than silently omitted.
            json.attribute("errorCount", static_cast<uint64_t>(
                                             plan.diagnostics.errors.size()));
            json.attribute(
                "warningCount",
                static_cast<uint64_t>(plan.diagnostics.warnings.size()));
          });
        });
      }
    });

    json.attribute("selectedPlanId", hexId(selected ? selected->id : 0));
  });

  stream.flush();
  return text;
}

llvm::Error writePlanReportFile(llvm::StringRef path,
                                const MappingSearchResult &result,
                                const machine::MachineModel &machine,
                                const MappingTarget &target,
                                const MappingSearchOptions &options,
                                uint64_t moduleHash) {
  std::string report =
      writePlanReport(result, machine, target, options, moduleHash);
  std::error_code error;
  llvm::raw_fd_ostream out(path, error);
  if (error)
    return llvm::createStringError(
        error, llvm::Twine("cannot write plan report to '") + path + "'");
  out << report;
  out.flush();
  if (out.has_error())
    return llvm::createStringError(
        out.error(), llvm::Twine("cannot write plan report to '") + path + "'");
  return llvm::Error::success();
}

} // namespace mlir::llk::mapping
