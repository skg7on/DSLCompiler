//===- ScheduleRecord.cpp - Persisted selected schedule (YAML) ------------===//
//
// Part of the M12 tuning core (issue #50). See ScheduleRecord.h.
//
// The YAML is assembled by hand so the key order is the order in the spec and
// byte-identical across runs. `std::map` already iterates in key order, so the
// candidate bindings are deterministic without an explicit sort.
//
//===----------------------------------------------------------------------===//

#include "LLK/Perf/ScheduleRecord.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <string>

namespace mlir::llk::perf {

namespace {

std::string fixed(double value, int precision) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%.*f", precision, value);
  return buffer;
}

/// Writes `key: [a, b, c]` followed by a newline.
void writeStringList(llvm::raw_ostream &os, llvm::StringRef key,
                     llvm::ArrayRef<std::string> values) {
  os << key << ": [";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i)
      os << ", ";
    os << values[i];
  }
  os << "]\n";
}

void writeIntList(llvm::raw_ostream &os, llvm::StringRef key,
                  llvm::ArrayRef<int64_t> values) {
  os << key << ": [";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i)
      os << ", ";
    os << values[i];
  }
  os << "]\n";
}

} // namespace

void writeScheduleYaml(llvm::raw_ostream &os, const ScheduleRecord &record) {
  os << "schema_version: " << record.schemaVersion << "\n";
  os << "workload: " << record.workload << "\n";
  os << "target: " << record.target << "\n";
  os << "machine: " << record.machine << "\n";

  os << "shape:\n";
  os << "  M_bucket: " << classifyMBucket(record.shape.M) << "\n";
  os << "  M: " << record.shape.M << "\n";
  os << "  N: " << record.shape.N << "\n";
  os << "  K: " << record.shape.K << "\n";

  os << "dtype:\n";
  os << "  input: " << record.shape.inputDType << "\n";
  os << "  weight: " << record.shape.weightDType << "\n";
  os << "  output: " << record.shape.outputDType << "\n";
  os << "  accumulator: " << record.shape.accumulatorDType << "\n";

  os << "candidate:\n";
  os << "  id: " << record.candidate.id << "\n";
  os << "  values:\n";
  for (const auto &[name, value] : record.candidate.values)
    os << "    " << name << ": " << value << "\n";

  // The tile decisions are the schedule itself: hierarchy, layout, memory
  // placement, owner mapping, and instruction fragment.
  os << "  tile:\n";
  os << "    hierarchy:\n";
  writeIntList(os, "      worker", record.tile.workerTile);
  writeIntList(os, "      fragment", record.tile.declaredFragment);
  os << "    layout:\n";
  os << "      input: " << record.tile.tileLayout << "\n";
  os << "      weights: " << record.tile.tileLayout << "\n";
  os << "      accumulator: " << record.tile.tileLayout << "\n";
  writeStringList(os, "    memory_path", record.tile.memoryPath);
  os << "    owner_mapping:\n";
  os << "      outer: " << record.tile.outerOwner << "\n";
  os << "      fragment: " << record.tile.fragmentOwner << "\n";
  os << "    tail_policy: " << record.tile.tailPolicy << "\n";
  os << "    pipeline_stages: " << record.tile.pipelineStages << "\n";
  os << "    vector_width: " << record.tile.vectorWidth << "\n";

  os << "metrics:\n";
  os << "  perf_level: " << record.perfLevel << "\n";
  os << "  predicted_cycles: " << record.metrics.predictedCycles << "\n";
  os << "  predicted_ns: " << fixed(record.metrics.predictedNs, 1) << "\n";
  os << "  matrix_utilization: " << fixed(record.metrics.matrixUtilization, 3)
     << "\n";
  os << "  dma_utilization: " << fixed(record.metrics.dmaUtilization, 3)
     << "\n";
  os << "  dram_bytes: " << record.metrics.dramBytes << "\n";
  os << "  sram_bytes: " << record.metrics.sramBytes << "\n";
  os << "  bottleneck: " << record.metrics.bottleneck << "\n";

  os << "measurement:\n";
  if (!record.measured) {
    os << "  measured: false\n";
    return;
  }
  os << "  measured: true\n";
  os << "  warmup: " << record.warmup << "\n";
  os << "  repeat: " << record.repeat << "\n";
  os << "  median_ns: " << fixed(record.medianNs, 1) << "\n";
  os << "  measured_gflops: " << fixed(record.measuredGflops, 1) << "\n";
}

void writeScheduleYaml(llvm::raw_ostream &os,
                       llvm::ArrayRef<ScheduleRecord> records) {
  for (size_t i = 0; i < records.size(); ++i) {
    if (i)
      os << "---\n";
    writeScheduleYaml(os, records[i]);
  }
}

llvm::Error writeScheduleYamlFile(llvm::StringRef path,
                                  llvm::ArrayRef<ScheduleRecord> records) {
  std::error_code ec;
  llvm::raw_fd_ostream os(path, ec, llvm::sys::fs::OF_Text);
  if (ec)
    return llvm::make_error<llvm::StringError>(
        ("cannot open '" + path + "': " + ec.message()).str(), ec);

  writeScheduleYaml(os, records);
  os.flush();
  if (os.has_error())
    return llvm::make_error<llvm::StringError>(
        ("cannot write '" + path + "'").str(), os.error());
  return llvm::Error::success();
}

} // namespace mlir::llk::perf
