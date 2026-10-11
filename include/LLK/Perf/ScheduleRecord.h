//===- ScheduleRecord.h - Persisted selected schedule (YAML) --------------===//
//
// Part of the M12 tuning core (issue #50).
//
// A ScheduleRecord is one row of `llk-tune`'s output: the workload identity,
// the machine it was tuned for, the winning candidate's bindings, the tile
// decisions it resolved to, and the predicted (or measured) cost. It is the
// tile-centric successor to the JSON `schedule_db.json` entries: where those
// stored only loop tile sizes, a record also carries the tile hierarchy,
// layout, memory placement, owner mapping, pipeline decisions, and fragment
// choices, because those are what the Micro-IR schedule actually is.
//
// The YAML is written by hand rather than through a serializer so the field
// order is stable and readable. Every collection is emitted in a deterministic
// order, so two runs over the same candidate produce byte-identical YAML.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_PERF_SCHEDULERECORD_H
#define LLK_PERF_SCHEDULERECORD_H

#include "LLK/Perf/Candidate.h"
#include "LLK/Perf/CandidateBinding.h"
#include "LLK/Perf/SearchSpace.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir::llk::perf {

/// The YAML schema version this build writes. Bump it when a change makes an
/// older record mean something different rather than merely adding a key.
inline constexpr uint32_t kScheduleSchemaVersion = 1;

/// One selected schedule.
struct ScheduleRecord {
  uint32_t schemaVersion = kScheduleSchemaVersion;

  /// Source workload identity: the search space's workload name and the problem
  /// shape it was instantiated for.
  std::string workload;
  WorkloadShape shape;

  /// Target identity: the machine model's `name` and the path it was read from.
  std::string target;
  std::string machine;

  /// The bindings and the tile decisions they resolved to.
  Candidate candidate;
  BoundTileDecisions tile;

  /// Predicted cost at `perfLevel` (0 static bound, 1 resource schedule).
  CandidateMetrics metrics;
  unsigned perfLevel = 1;

  /// Measurement, when an AVX2 run annotated the candidate (#51). Unmeasured
  /// records write `measured: false` and nothing else.
  bool measured = false;
  unsigned warmup = 0;
  unsigned repeat = 0;
  double medianNs = 0.0;
  double measuredGflops = 0.0;

  /// Reference to the frozen mapped choice, present only for target-aware
  /// tuning. Paths are caller-selected and are not part of plan identity.
  uint32_t mappingProvenanceVersion = 0;
  std::string mappingPlanId;
  std::string mappingBindingHash;
  std::string mappingSourceArtifact;
  std::string mappingPlanReport;
};

/// Writes one record as a YAML document.
void writeScheduleYaml(llvm::raw_ostream &os, const ScheduleRecord &record);

/// Writes records as a multi-document YAML stream, `---`-separated.
void writeScheduleYaml(llvm::raw_ostream &os,
                       llvm::ArrayRef<ScheduleRecord> records);

/// Writes the stream to `path`. Fails when the file cannot be opened or the
/// write fails, rather than leaving a truncated file behind silently.
llvm::Error writeScheduleYamlFile(llvm::StringRef path,
                                  llvm::ArrayRef<ScheduleRecord> records);

} // namespace mlir::llk::perf

#endif // LLK_PERF_SCHEDULERECORD_H
