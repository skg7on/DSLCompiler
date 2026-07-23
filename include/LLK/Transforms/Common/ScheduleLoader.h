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
struct ScheduleEntry {
  int64_t BM{0}, BN{0}, BK{0};
  int64_t VM{0}, VN{0};
  int64_t vector_width{0};
  int64_t num_threads{1};
  int64_t grain_size{1};
  std::string parallel_axis;
};

/// Load matching schedule entries for the given key.
/// Returns all entries that match (M_bucket, N, K) — the caller selects
/// the best among them.
std::vector<ScheduleEntry> loadScheduleDB(llvm::StringRef dbPath, int M_bucket,
                                          int64_t N, int64_t K,
                                          llvm::StringRef opName = "");

/// Classify M into one of 5 buckets: {1}, [2,4], [5,16], [17,64], ≥65.
int classifyM(int64_t M);

} // namespace llk
} // namespace mlir

#endif // LLK_TRANSFORMS_COMMON_SCHEDULELOADER_H
