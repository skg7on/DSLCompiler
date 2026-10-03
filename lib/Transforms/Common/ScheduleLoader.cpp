//===- ScheduleLoader.cpp - Shared schedule_db.json reader ----------------===//
//
// Extracted from ScheduleSelection pass for reuse by autotuning (llk-tune)
// and benchmarking (llk-bench).
//
//===----------------------------------------------------------------------===//

#include "LLK/Transforms/Common/ScheduleLoader.h"

#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

using namespace mlir;

namespace mlir {
namespace llk {

int classifyM(int64_t M) {
  if (M == 1)
    return 0;
  if (M <= 4)
    return 1;
  if (M <= 16)
    return 2;
  if (M <= 64)
    return 3;
  return 4;
}

std::vector<ScheduleEntry> loadScheduleDB(llvm::StringRef dbPath, int M_bucket,
                                          int64_t N, int64_t K,
                                          llvm::StringRef opName) {
  std::vector<ScheduleEntry> matches;

  auto buf = llvm::MemoryBuffer::getFile(dbPath);
  if (!buf) {
    llvm::errs() << "ScheduleLoader: cannot open " << dbPath << "\n";
    return matches;
  }

  auto json = llvm::json::parse(buf->get()->getBuffer());
  if (!json) {
    llvm::errs() << "ScheduleLoader: invalid JSON in " << dbPath << "\n";
    return matches;
  }

  auto *obj = json->getAsObject();
  if (!obj)
    return matches;

  auto *entries = obj->getArray("entries");
  if (!entries)
    return matches;

  for (auto &entry : *entries) {
    auto *entryObj = entry.getAsObject();
    if (!entryObj)
      continue;

    auto opStr = entryObj->getString("operation");
    if (!opStr || *opStr != opName)
      continue;

    auto *shape = entryObj->getObject("shape");
    if (!shape)
      continue;

    auto bucketOpt = shape->getInteger("M_bucket");
    if (!bucketOpt || static_cast<int64_t>(*bucketOpt) != M_bucket)
      continue;

    auto nOpt = shape->getInteger("N");
    auto kOpt = shape->getInteger("K");
    if (!nOpt || !kOpt)
      continue;

    auto *sched = entryObj->getObject("schedule");
    if (!sched)
      continue;

    ScheduleEntry se;
    se.BM = sched->getInteger("BM").value_or(0);
    se.BN = sched->getInteger("BN").value_or(0);
    se.BK = sched->getInteger("BK").value_or(0);
    se.VM = sched->getInteger("VM").value_or(0);
    se.VN = sched->getInteger("VN").value_or(0);
    se.vector_width = sched->getInteger("vector_width").value_or(0);
    se.num_threads = sched->getInteger("num_threads").value_or(1);
    se.grain_size = sched->getInteger("grain_size").value_or(1);
    if (auto axis = sched->getString("parallel_axis"))
      se.parallel_axis = axis->str();

    // Micro-IR export metadata (M11). Every field is optional and keeps its
    // default when absent, so the pre-M11 entries in schedules/schedule_db.json
    // load unchanged.
    se.pipeline_stages = sched->getInteger("pipeline_stages").value_or(1);
    // A negative prefetch distance is not a distance; treat it as "no
    // prefetch" rather than letting it invert the pipeline.
    se.prefetch_distance = std::max<int64_t>(
        0, sched->getInteger("prefetch_distance").value_or(0));
    if (auto path = sched->getString("memory_path"))
      se.memory_path = path->str();
    if (auto shape = sched->getString("mma_shape"))
      se.mma_shape = shape->str();
    if (auto space = sched->getString("accumulator_space"))
      se.accumulator_space = space->str();
    if (auto layout = sched->getString("tile_layout"))
      se.tile_layout = layout->str();
    if (auto owner = sched->getString("owner_mapping"))
      se.owner_mapping = owner->str();
    if (auto owner = sched->getString("fragment_owner"))
      se.fragment_owner = owner->str();
    // fragment_shape is derived from mma_shape unless the entry overrides it.
    se.fragment_shape = se.mma_shape;
    if (auto shape = sched->getString("fragment_shape"))
      se.fragment_shape = shape->str();
    if (auto masks = sched->getBoolean("enable_tile_masks"))
      se.enable_tile_masks = *masks;

    matches.push_back(se);
  }

  return matches;
}

ScheduleEntry selectBestSchedule(const std::vector<ScheduleEntry> &matches,
                                 int64_t N, int64_t K) {
  if (matches.empty()) {
    // Built-in default: conservative small-tile schedule.
    ScheduleEntry def;
    def.BM = 8;
    def.BN = 32;
    def.BK = 32;
    def.VM = 1;
    def.VN = 4;
    def.vector_width = 8;
    def.num_threads = 4;
    def.grain_size = 1;
    def.parallel_axis = "n";
    return def;
  }

  // All entries from loadScheduleDB already match M_bucket.
  // Return the first match; a fuller implementation would
  // compare N and K against the entry values for exact fit.
  return matches.front();
}

} // namespace llk
} // namespace mlir
