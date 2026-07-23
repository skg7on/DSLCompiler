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
    llvm::errs() << "ScheduleSelection: cannot open " << dbPath << "\n";
    return matches;
  }

  auto json = llvm::json::parse(buf->get()->getBuffer());
  if (!json) {
    llvm::errs() << "ScheduleSelection: invalid JSON in " << dbPath << "\n";
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

    matches.push_back(se);
  }

  return matches;
}

} // namespace llk
} // namespace mlir
