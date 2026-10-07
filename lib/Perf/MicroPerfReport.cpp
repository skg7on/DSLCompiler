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

#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/StoragePlan.h"
#include "LLK/Perf/SelectedKernelAnalysis.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cstdio>
#include <limits>
#include <map>
#include <string>
#include <utility>
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

/// The live-storage occupancy a *mapped* kernel's own plan recorded, when the
/// kernel carries one, or an empty map otherwise.
///
/// A mapped kernel's `micro.plan` records the storage allocations the selected
/// plan reserved (`finalizeStoragePlan`'s decisions). Their live peak -- the
/// bytes each memory must hold simultaneously, honoring the aliases the plan
/// proved and the simultaneous residency it recorded -- is *not*
/// `MicroDAG::liveTileBytesByMemory`: that accounting sums every buffer the
/// kernel materializes, charging loop-nested and reused buffers as if they were
/// all live at once, and so reports a footprint far above the plan's proven
/// peak. The plan is the authority for a kernel that carries one.
///
/// The relation used is `computePeakStorage` over the recorded allocations,
/// which is the *fallback* relation `finalizeStoragePlan` summarizes occupancy
/// with when the machine does not model every event resource -- not the primary
/// event-schedule liveness. The two are not structurally the same (live
/// intervals versus scheduled events), though they agree here because the
/// plan's aliases and per-slot residency are exactly what both read. Re-running
/// the primary liveness from the IR alone is not possible: it needs each
/// event's `StorageUse` occurrences, which carry the planner's own allocation
/// identities and are not persisted (that is R8's durable-replay work). An
/// empty map means "no plan recorded": the caller keeps the extraction's own
/// accounting (issue #129, task R7 occupancy reconciliation).
std::map<mapping::MemoryNodeId, uint64_t>
planRecordedLivePeak(mlir::Operation *kernel,
                     const machine::MachineModel &machine) {
  auto plan = kernel->getAttrOfType<mlir::DictionaryAttr>("micro.plan");
  if (!plan)
    return {};
  auto recorded = plan.getAs<mlir::ArrayAttr>("allocations");
  if (!recorded)
    return {};

  std::vector<mapping::StorageAllocation> allocations;
  allocations.reserve(recorded.size());
  for (mlir::Attribute entry : recorded) {
    auto object = mlir::dyn_cast<mlir::DictionaryAttr>(entry);
    if (!object)
      return {};
    mapping::StorageAllocation allocation;
    auto id = object.getAs<mlir::IntegerAttr>("id");
    auto bytes = object.getAs<mlir::IntegerAttr>("bytes");
    auto memory = object.getAs<mlir::StringAttr>("memory");
    if (!id || !bytes || !memory)
      return {};
    allocation.id = id.getInt();
    allocation.bytes = bytes.getInt();
    allocation.memory = memory.getValue().str();
    if (auto alias = object.getAs<mlir::IntegerAttr>("alias_of"))
      allocation.aliasOf = alias.getInt();
    if (auto occurrences =
            object.getAs<mlir::IntegerAttr>("simultaneous_occurrences"))
      allocation.simultaneousOccurrences = occurrences.getInt();
    if (auto borrowed = object.getAs<mlir::BoolAttr>("borrowed"))
      allocation.borrowed = borrowed.getValue();
    // A memory the kernel does not materialize is not charged: the same rule
    // `MicroDAG::noteStorage` applies, so the two accountings agree on what a
    // live byte *is*.
    const machine::MemoryNode *level = machine.findMemory(allocation.memory);
    if (!level || level->kind == "dram")
      continue;
    // Each allocation contributes its footprint once per simultaneously-live
    // occurrence, exactly as `finalizeStoragePlan`'s occupancy probe charges
    // it.
    if (allocation.simultaneousOccurrences != 0 &&
        allocation.bytes > std::numeric_limits<uint64_t>::max() /
                               allocation.simultaneousOccurrences)
      return {};
    allocation.bytes *= allocation.simultaneousOccurrences;
    allocations.push_back(std::move(allocation));
  }
  llvm::Expected<std::map<mapping::MemoryNodeId, uint64_t>> peak =
      mapping::computePeakStorage(allocations);
  if (!peak) {
    llvm::consumeError(peak.takeError());
    return {};
  }
  return std::move(*peak);
}

} // namespace

//===----------------------------------------------------------------------===//
// Analysis
//===----------------------------------------------------------------------===//

llvm::Expected<MicroPerfReport>
analyzeKernel(mlir::Operation *kernel, const machine::MachineModel &machine,
              unsigned level) {
  if (level > 1)
    return invalid("unsupported analysis level " + llvm::Twine(level) +
                   "; micro-perf knows level 0 (static bound) and level 1 "
                   "(resource schedule)");

  auto dag = buildMicroDAG(kernel, machine);
  if (!dag)
    return dag.takeError();

  // The one static analysis (issue #129, task R6): the same normalized stream,
  // schedule and summaries the mapping planner reads. The report derives its
  // L0 byte block and its L1 schedule from it rather than scheduling a second
  // time, so `micro-perf` and the planner cannot disagree about one selected
  // kernel. The DAG is extracted once and handed to the shared analysis.
  llvm::Expected<SelectedKernelAnalysis> analysis =
      analyzeSelectedDag(*dag, machine, /*requireComplete=*/false);
  if (!analysis)
    return analysis.takeError();

  MicroPerfReport report;
  report.machine = machine.target;
  report.level = level;
  report.clockHz = machine.clockHz.value_or(0);
  if (auto symbol = kernel->getAttrOfType<mlir::StringAttr>("sym_name"))
    report.kernel = symbol.getValue().str();
  else
    report.kernel = "<anonymous>";

  report.l0 = computeL0StaticBound(*dag, machine);
  // The byte totals and the live peak are the analysis's own summaries, so the
  // report and a plan consumer read one traffic/occupancy relation.
  report.l0.bytesByMemory = analysis->trafficBytes;
  report.l0.totalBytesDram = analysis->trafficBytes.count("dram")
                                 ? analysis->trafficBytes.at("dram")
                                 : 0;
  report.l0.totalBytesSram = analysis->trafficBytes.count("sram")
                                 ? analysis->trafficBytes.at("sram")
                                 : 0;
  if (level == 1)
    report.l1 = reportL1FromSchedule(*dag, machine, analysis->schedule);
  // Occupancy and the capacity verdict (issue #129, task R7). A mapped kernel
  // carries its plan's own storage decisions, whose live peak is the relation
  // the planner already validated; a kernel without one keeps the extraction's
  // accounting. The verdict is taken against the same number that is reported,
  // so `live_bytes` and `capacity_violations` can never disagree.
  std::map<mapping::MemoryNodeId, uint64_t> planPeak =
      planRecordedLivePeak(kernel, machine);
  if (planPeak.empty()) {
    report.l0.liveTileBytesByMemory = analysis->peakBytes;
    report.capacityViolations = checkCapacity(*dag, machine);
  } else {
    std::map<std::string, uint64_t> byKind;
    for (const auto &[node, bytes] : planPeak) {
      const machine::MemoryNode *memory = machine.findMemory(node);
      if (!memory)
        continue;
      byKind[memory->kind.empty() ? node : memory->kind] += bytes;
      if (bytes > memory->capacityBytes)
        report.capacityViolations.push_back(
            "memory " + node + " requires " + std::to_string(bytes) +
            " bytes but machine has " + std::to_string(memory->capacityBytes) +
            " bytes");
    }
    report.l0.liveTileBytesByMemory = std::move(byKind);
  }
  report.warnings = dag->warnings;
  // The analysis's own completeness verdict, made visible (issue #129, task
  // R6): a reason the extraction already warns about is not repeated, but a
  // strict-modelling gap the stream has (an event resource the machine does not
  // model -- the abstract `dma` pool an unrouted movement names) is reported
  // rather than left to a reader to infer from the numbers.
  for (const std::string &reason : analysis->incompleteReasons)
    if (!llvm::is_contained(report.warnings, reason))
      report.warnings.push_back(reason);
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
