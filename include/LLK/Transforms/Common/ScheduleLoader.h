//===- ScheduleLoader.h - Shared schedule_db.json reader --------*- C++ -*-===//
//
// JSON-based schedule database loader used by ScheduleSelection pass.
// Decoupled from the pass itself so autotuning and benchmarking can reuse it.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_TRANSFORMS_COMMON_SCHEDULELOADER_H
#define LLK_TRANSFORMS_COMMON_SCHEDULELOADER_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llvm {
class StringRef;
} // namespace llvm

namespace mlir {
namespace llk {

/// A single schedule entry from the database.
///
/// The second half of the struct is the metadata the Micro-IR export needs
/// (M11). Every field has a default, and the JSON reader only overwrites the
/// ones an entry actually spells out, so entries written before the fields
/// existed still load and lowering still produces a concrete kernel for them.
struct ScheduleEntry {
  // Tiling and mapping, used by the Linalg pipeline (M5).
  int64_t BM{0}, BN{0}, BK{0};
  int64_t VM{0}, VN{0};
  int64_t vector_width{0};
  int64_t num_threads{1};
  int64_t grain_size{1};
  std::string parallel_axis;

  // Micro-IR export metadata (M11). Only the fields the export actually reads
  // are here: a field nothing consumes would be indistinguishable from one
  // that was forgotten.
  int64_t pipeline_stages{1};
  /// Tiles prefetched ahead of the consuming stage; 0 means no prefetch.
  int64_t prefetch_distance{0};
  std::string memory_path{"dram:sram:acc"};
  std::string mma_shape{"16x16x32"};
  std::string accumulator_space{"acc"};
  std::string tile_layout{"row_major"};
  std::string owner_mapping{"worker"};
  std::string fragment_owner{"vector_engine"};
  /// Defaults to mma_shape when an entry does not name one.
  std::string fragment_shape{"16x16x32"};
  bool enable_tile_masks{true};
};

/// Load matching schedule entries for the given key.
/// Returns all entries that match (M_bucket, N, K) — the caller selects
/// the best among them.
std::vector<ScheduleEntry> loadScheduleDB(llvm::StringRef dbPath, int M_bucket,
                                          int64_t N, int64_t K,
                                          llvm::StringRef opName = "");

/// Pick the best entry among `matches` for a problem of the given shape.
///
/// Shared by ScheduleSelection and the Micro-IR export so the two cannot
/// disagree about which schedule an operation runs under. Returns the built-in
/// conservative schedule when `matches` is empty, which is the documented
/// "no schedule entry" behaviour.
ScheduleEntry selectBestSchedule(const std::vector<ScheduleEntry> &matches,
                                 int64_t N, int64_t K);

/// Classify M into one of 5 buckets: {1}, [2,4], [5,16], [17,64], ≥65.
int classifyM(int64_t M);

} // namespace llk
} // namespace mlir

#endif // LLK_TRANSFORMS_COMMON_SCHEDULELOADER_H
