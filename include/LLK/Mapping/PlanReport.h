//===- PlanReport.h - Versioned JSON plan report (design §22.2) -----------===//
//
// Part of the target-independent mapping core (epic #67, workstream D7).
//
// A plan report is reproducibility and diagnostic metadata, never executable
// input: it names the exact inputs a search ran on -- the module hash, the
// machine, the layout and rule libraries, and the source search binding -- the
// options it used, how far it got, and the plans it retained with their costs.
// It does not replace `micro.search_space` or the bound Micro-IR.
//
// Determinism is the contract (design §29.12): two runs with identical inputs
// must emit byte-identical JSON. Keys are written in a fixed order (the report
// is built imperatively, so `llvm::json`'s internally unordered object never
// reaches the output), and no timestamp, address, or unordered iteration is
// included. Costs render through `canonicalCostString` / a fixed `%.6f`, both
// byte-stable across runs and platforms.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_PLANREPORT_H
#define LLK_MAPPING_PLANREPORT_H

#include "LLK/Machine/MachineModel.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingTarget.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace mlir::llk::mapping {

/// The report schema version. A reader keys its parser on it; a compatible
/// addition of a field does not change the version, but a change to an existing
/// field's meaning does.
inline constexpr uint64_t kPlanReportVersion = 1;

/// The compiler version string the report was produced by. A placeholder until
/// the project version is threaded through; it is in the report so a consumer
/// can tell two compiler revisions apart. The cost-model version is
/// `kCostModelVersion` (LatencyProvider.h), the same value a measurement's
/// cache key carries.
inline constexpr llvm::StringLiteral kCompilerVersion = "llk-compiler";

/// Serializes `result` to the versioned JSON plan report. `moduleHash` is the
/// content hash of the input module the search ran over (the caller computes it
/// before any binding mutates the module). The returned string is byte-stable
/// for identical inputs.
std::string writePlanReport(const MappingSearchResult &result,
                            const machine::MachineModel &machine,
                            const MappingTarget &target,
                            const MappingSearchOptions &options,
                            uint64_t moduleHash);

/// Writes `writePlanReport(...)` to `path`, creating or truncating it. Returns
/// an error the caller reports rather than silently dropping the report.
llvm::Error writePlanReportFile(llvm::StringRef path,
                                const MappingSearchResult &result,
                                const machine::MachineModel &machine,
                                const MappingTarget &target,
                                const MappingSearchOptions &options,
                                uint64_t moduleHash);

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_PLANREPORT_H
