//===- MicroPerfReport.cpp - Analysis entry point and reporting -----------===//
//
// analyzeKernel() extracts the DAG once and derives everything else from it, so
// `micro-perf --level=0` and `--level=1` can never disagree about what the
// kernel does.
//
// The YAML writer is hand-rolled rather than built on a serializer because the
// shape is fixed and small, and because stable ordering matters more than
// generality: `llk-tune` reads this output, and FileCheck compares it. Every
// collection is a std::map or is sorted, so equal inputs always print equal
// bytes.
//
//===----------------------------------------------------------------------===//

#include "LLK/Perf/MicroPerfReport.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

namespace mlir::llk::perf {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

std::string fixed(double value, int precision) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.*f", precision, value);
  return buffer;
}

/// YAML scalars are printed bare when they cannot be misread, and double-quoted
/// otherwise. Only the quoted form can appear in the diagnostic lists.
std::string quoted(llvm::StringRef text) {
  std::string out = "\"";
  for (char character : text) {
    if (character == '"' || character == '\\')
      out += '\\';
    out += character;
  }
  out += "\"";
  return out;
}

void emitStringList(llvm::raw_ostream &os, llvm::StringRef indent,
                    llvm::StringRef key,
                    const std::vector<std::string> &values) {
  os << indent << key << ":";
  if (values.empty()) {
    os << " []\n";
    return;
  }
  os << "\n";
  for (const std::string &value : values)
    os << indent << "  - " << quoted(value) << "\n";
}

/// Emits a sorted `name: count` block. std::map iteration order is what makes
/// the report byte-stable across runs.
void emitCounterMap(llvm::raw_ostream &os, llvm::StringRef indent,
                    llvm::StringRef key,
                    const std::map<std::string, uint64_t> &counters) {
  os << indent << key << ":";
  if (counters.empty()) {
    os << " {}\n";
    return;
  }
  os << "\n";
  for (const auto &[name, value] : counters)
    os << indent << "  " << name << ": " << value << "\n";
}

void emitTextList(llvm::raw_ostream &os, llvm::StringRef header,
                  const std::vector<std::string> &values) {
  if (values.empty())
    return;
  os << header << "\n";
  for (const std::string &value : values)
    os << "  - " << value << "\n";
}

std::string joinCounters(const std::map<std::string, uint64_t> &counters) {
  if (counters.empty())
    return "none";
  std::string text;
  for (const auto &[name, value] : counters) {
    if (!text.empty())
      text += " ";
    text += name + "=" + std::to_string(value);
  }
  return text;
}

} // namespace

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

llvm::Expected<MicroPerfReport> analyzeKernel(mlir::Operation *kernel,
                                              const MachineModel &machine,
                                              unsigned level) {
  if (level > 1)
    return invalid("unsupported analysis level " + llvm::Twine(level) +
                   "; micro-perf knows level 0 (static bound) and level 1 "
                   "(resource schedule)");

  auto dag = buildMicroDAG(kernel, machine);
  if (!dag)
    return dag.takeError();

  MicroPerfReport report;
  report.machine = machine.name;
  report.level = level;
  report.clockHz = machine.clockHz;
  if (auto symbol = kernel->getAttrOfType<mlir::StringAttr>("sym_name"))
    report.kernel = symbol.getValue().str();
  else
    report.kernel = "<anonymous>";

  report.l0 = computeL0StaticBound(*dag, machine);
  if (level == 1)
    report.l1 = scheduleL1(*dag, machine);
  report.capacityViolations = checkCapacity(*dag, machine);
  report.warnings = dag->warnings;
  report.layoutWarnings = dag->layoutWarnings;
  report.ownerWarnings = dag->ownerWarnings;

  for (const MicroEvent &event : dag->events) {
    ++report.operationCounts[stringifyEventKind(event.kind).str()];
    if (!event.tileLayout.empty())
      ++report.layoutHistogram[event.tileLayout];
    if (!event.tileOwner.empty())
      ++report.ownerHistogram[event.tileOwner];
  }
  return report;
}

//===----------------------------------------------------------------------===//
// YAML
//===----------------------------------------------------------------------===//

void printMicroPerfYaml(llvm::raw_ostream &os, const MicroPerfReport &report) {
  os << "schema_version: 1\n";
  os << "machine: " << report.machine << "\n";
  os << "kernel: " << report.kernel << "\n";
  os << "level: " << report.level << "\n";

  os << "totals:\n";
  os << "  flops: " << report.l0.totalFlops << "\n";
  emitCounterMap(os, "  ", "bytes", report.l0.bytesByMemory);

  emitCounterMap(os, "", "operations", report.operationCounts);

  os << "bounds:\n";
  os << "  compute_cycles: " << report.l0.computeCyclesLowerBound << "\n";
  os << "  memory_cycles: " << report.l0.memoryCyclesLowerBound << "\n";

  if (report.l1) {
    os << "predicted_cycles: " << report.l1->predictedCycles << "\n";
    os << "predicted_ns: " << fixed(report.l1->predictedNs, 1) << "\n";
    os << "utilization:\n";
    os << "  matrix: " << fixed(report.l1->matrixUtilization, 3) << "\n";
    os << "  vector: " << fixed(report.l1->vectorUtilization, 3) << "\n";
    os << "  dma: " << fixed(report.l1->dmaUtilization, 3) << "\n";
    os << "bandwidth:\n";
    os << "  dram: " << fixed(report.l1->dramBandwidthUtilization, 3) << "\n";
    os << "  sram: " << fixed(report.l1->sramBandwidthUtilization, 3) << "\n";
    os << "overlap_efficiency: " << fixed(report.l1->overlapEfficiency, 3)
       << "\n";
  } else {
    os << "predicted_cycles: " << report.l0.predictedCycles << "\n";
  }

  os << "tiles:\n";
  emitCounterMap(os, "  ", "live_bytes", report.l0.liveTileBytesByMemory);
  emitCounterMap(os, "  ", "layouts", report.layoutHistogram);
  emitCounterMap(os, "  ", "owners", report.ownerHistogram);

  os << "bottleneck: "
     << (report.l1 ? report.l1->bottleneck : report.l0.bottleneck) << "\n";

  emitStringList(os, "", "capacity_violations", report.capacityViolations);
  emitStringList(os, "", "layout_warnings", report.layoutWarnings);
  emitStringList(os, "", "owner_warnings", report.ownerWarnings);
  emitStringList(os, "", "warnings", report.warnings);
}

//===----------------------------------------------------------------------===//
// Text
//===----------------------------------------------------------------------===//

void printMicroPerfText(llvm::raw_ostream &os, const MicroPerfReport &report) {
  os << "kernel: " << report.kernel << "\n";
  os << "machine: " << report.machine << " @ " << report.clockHz << " Hz\n";
  os << "level: " << report.level << "\n\n";

  os << "work\n";
  os << "  flops         " << report.l0.totalFlops << "\n";
  os << "  bytes         " << joinCounters(report.l0.bytesByMemory) << "\n";
  os << "  operations    " << joinCounters(report.operationCounts) << "\n";

  os << "bounds\n";
  os << "  compute       " << report.l0.computeCyclesLowerBound << " cycles\n";
  os << "  memory        " << report.l0.memoryCyclesLowerBound << " cycles\n";

  if (report.l1) {
    os << "schedule\n";
    os << "  predicted     " << report.l1->predictedCycles << " cycles ("
       << fixed(report.l1->predictedNs, 1) << " ns)\n";
    os << "  utilization   matrix " << fixed(report.l1->matrixUtilization, 3)
       << "  vector " << fixed(report.l1->vectorUtilization, 3) << "  dma "
       << fixed(report.l1->dmaUtilization, 3) << "\n";
    os << "  bandwidth     dram "
       << fixed(report.l1->dramBandwidthUtilization, 3) << "  sram "
       << fixed(report.l1->sramBandwidthUtilization, 3) << "\n";
    os << "  overlap       " << fixed(report.l1->overlapEfficiency, 3) << "\n";
  } else {
    os << "static bound\n";
    os << "  predicted     " << report.l0.predictedCycles << " cycles\n";
  }

  os << "tiles\n";
  os << "  live bytes    " << joinCounters(report.l0.liveTileBytesByMemory)
     << "\n";
  os << "  layouts       " << joinCounters(report.layoutHistogram) << "\n";
  os << "  owners        " << joinCounters(report.ownerHistogram) << "\n";

  os << "bottleneck: "
     << (report.l1 ? report.l1->bottleneck : report.l0.bottleneck) << "\n";

  os << "\n";
  emitTextList(os, "capacity violations", report.capacityViolations);
  emitTextList(os, "layout warnings", report.layoutWarnings);
  emitTextList(os, "owner warnings", report.ownerWarnings);
  emitTextList(os, "warnings", report.warnings);
}

} // namespace mlir::llk::perf
