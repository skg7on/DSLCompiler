//===- MicroPerfReport.h - Analysis entry point and reporting -------------===//
//
// Part of the M10 AVX2 performance simulator (issue #46).
//
// analyzeKernel() is the one call `micro-perf`, the GTests, and any later
// tuning stage share: it extracts the DAG once and derives L0, and L1 when the
// caller asks for it.
//
// Diagnostics live here rather than on L1Report, because capacity, layout, and
// owner problems are properties of the kernel/machine pair, not of a schedule,
// and `--level=0` must be able to report them too.
//
// Both formatters are deterministic: every collection is ordered by key, so the
// YAML is stable enough for FileCheck and for a later `llk-tune` to read.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_PERF_MICROPERFREPORT_H
#define LLK_PERF_MICROPERFREPORT_H

#include "LLK/Machine/MachineModel.h"
#include "LLK/Perf/MicroCostModel.h"
#include "LLK/Perf/MicroDAG.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir::llk::perf {

/// Everything `micro-perf` prints, at either level.
struct MicroPerfReport {
  std::string machine;
  std::string kernel;
  unsigned level = 1;
  uint64_t clockHz = 0;

  L0Report l0;
  /// Present only when level == 1.
  std::optional<L1Report> l1;

  std::vector<std::string> capacityViolations;
  std::vector<std::string> warnings;
  std::vector<std::string> layoutWarnings;
  std::vector<std::string> ownerWarnings;

  /// How many events of each kind the kernel expanded to, keyed by
  /// stringifyEventKind. Loop unrolling is already applied, so these are the
  /// counts the schedule actually ran, not the counts in the source.
  std::map<std::string, uint64_t> operationCounts;
  /// Layout/owner histograms over the simulated tiles, for the `tiles:` block.
  std::map<std::string, uint64_t> layoutHistogram;
  std::map<std::string, uint64_t> ownerHistogram;
};

/// Analyzes one `micro.kernel` against `machine`. `level` selects how far the
/// analysis goes: 0 stops at the static bound, 1 also schedules.
llvm::Expected<MicroPerfReport>
analyzeKernel(mlir::Operation *kernel, const machine::MachineModel &machine,
              unsigned level);

void printMicroPerfYaml(llvm::raw_ostream &os, const MicroPerfReport &report);
void printMicroPerfText(llvm::raw_ostream &os, const MicroPerfReport &report);

} // namespace mlir::llk::perf

#endif // LLK_PERF_MICROPERFREPORT_H
