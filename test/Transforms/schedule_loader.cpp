//===- schedule_loader.cpp - schedule_db.json loader tests ----------------===//
//
// Covers issue #48:
//   - the shipped schedule database still loads after ScheduleEntry gained the
//     Micro-IR fields
//   - the Micro/tile fields load when an entry spells them out
//   - a missing Micro/tile field falls back to a deterministic default, and
//     fragment_shape follows mma_shape when only the latter is given
//
//===----------------------------------------------------------------------===//

#include "LLK/Transforms/Common/ScheduleLoader.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <string>

namespace mlir::llk {
namespace {

/// Writes `contents` to a uniquely named file under the system temp directory
/// and returns its path, or an empty string when the file cannot be created.
std::string writeTempFile(llvm::StringRef contents) {
  static unsigned counter = 0;
  llvm::SmallString<128> path;
  llvm::sys::path::system_temp_directory(/*erasedOnReboot=*/true, path);
  llvm::sys::path::append(path, "llk-schedule-" + std::to_string(counter++) +
                                    ".json");

  std::error_code error;
  {
    llvm::raw_fd_ostream stream(path, error, llvm::sys::fs::OF_Text);
    if (error)
      return {};
    stream << contents;
  }
  return std::string(path);
}

/// A temporary schedule database on disk, removed when it goes out of scope.
struct TempScheduleFile {
  explicit TempScheduleFile(llvm::StringRef contents)
      : path(writeTempFile(contents)) {}

  ~TempScheduleFile() {
    // Best effort: the system temp directory is cleaned up by the OS anyway.
    if (!path.empty()) {
      std::error_code ignored = llvm::sys::fs::remove(path);
      (void)ignored;
    }
  }

  TempScheduleFile(const TempScheduleFile &) = delete;
  TempScheduleFile &operator=(const TempScheduleFile &) = delete;

  std::string path;
};

/// One entry for `op` with the given schedule body, so each test only spells
/// out the fields it is about.
std::string entryFor(llvm::StringRef op, llvm::StringRef scheduleBody) {
  return ("{\"version\": 1, \"entries\": [{\"operation\": \"" + op +
          "\", \"shape\": {\"M_bucket\": 2, \"N\": 64, \"K\": 64}, "
          "\"schedule\": {" +
          scheduleBody + "}}]}")
      .str();
}

const char *const kPreM11Schedule = R"json(
  "BM": 8, "BN": 32, "BK": 32, "VM": 1, "VN": 4, "vector_width": 8,
  "num_threads": 2, "parallel_axis": "n", "grain_size": 1
)json";

TEST(ScheduleLoader, PreM11EntryGetsDeterministicDefaults) {
  TempScheduleFile file(entryFor("matmul", kPreM11Schedule));
  ASSERT_FALSE(file.path.empty());

  auto matches =
      loadScheduleDB(file.path, /*M_bucket=*/2, /*N=*/64, /*K=*/64, "matmul");
  ASSERT_EQ(matches.size(), 1u);
  const ScheduleEntry &entry = matches.front();

  // The fields that predate the Micro-IR export keep loading.
  EXPECT_EQ(entry.BM, 8);
  EXPECT_EQ(entry.BN, 32);
  EXPECT_EQ(entry.BK, 32);
  EXPECT_EQ(entry.VM, 1);
  EXPECT_EQ(entry.VN, 4);
  EXPECT_EQ(entry.vector_width, 8);
  EXPECT_EQ(entry.num_threads, 2);
  EXPECT_EQ(entry.grain_size, 1);
  EXPECT_EQ(entry.parallel_axis, "n");

  // Every Micro/tile field the entry does not mention falls back to its
  // documented default.
  EXPECT_EQ(entry.pipeline_stages, 1);
  EXPECT_EQ(entry.prefetch_distance, 0);
  EXPECT_EQ(entry.memory_path, "dram:sram:acc");
  EXPECT_EQ(entry.mma_shape, "16x16x32");
  EXPECT_EQ(entry.accumulator_space, "acc");
  EXPECT_EQ(entry.tile_layout, "row_major");
  EXPECT_EQ(entry.owner_mapping, "worker");
  EXPECT_EQ(entry.fragment_owner, "vector_engine");
  EXPECT_EQ(entry.fragment_shape, "16x16x32");
  EXPECT_TRUE(entry.enable_tile_masks);
}

TEST(ScheduleLoader, MicroFieldsLoadWhenPresent) {
  TempScheduleFile file(entryFor("matmul", R"json(
    "BM": 16, "BN": 64, "BK": 64, "VM": 2, "VN": 8, "vector_width": 8,
    "num_threads": 4, "parallel_axis": "n", "grain_size": 2,
    "pipeline_stages": 3, "prefetch_distance": 2,
    "memory_path": "dram:l2:sram", "mma_shape": "16x16x32",
    "accumulator_space": "rf", "tile_layout": "blocked",
    "owner_mapping": "worker/vector_engine",
    "fragment_owner": "matrix_engine", "fragment_shape": "8x8x32",
    "enable_tile_masks": false
  )json"));
  ASSERT_FALSE(file.path.empty());

  auto matches =
      loadScheduleDB(file.path, /*M_bucket=*/2, /*N=*/64, /*K=*/64, "matmul");
  ASSERT_EQ(matches.size(), 1u);
  const ScheduleEntry &entry = matches.front();

  EXPECT_EQ(entry.pipeline_stages, 3);
  EXPECT_EQ(entry.prefetch_distance, 2);
  EXPECT_EQ(entry.memory_path, "dram:l2:sram");
  EXPECT_EQ(entry.mma_shape, "16x16x32");
  EXPECT_EQ(entry.accumulator_space, "rf");
  EXPECT_EQ(entry.tile_layout, "blocked");
  EXPECT_EQ(entry.owner_mapping, "worker/vector_engine");
  EXPECT_EQ(entry.fragment_owner, "matrix_engine");
  EXPECT_EQ(entry.fragment_shape, "8x8x32");
  EXPECT_FALSE(entry.enable_tile_masks);
}

TEST(ScheduleLoader, FragmentShapeFollowsMmaShapeWhenAbsent) {
  TempScheduleFile file(entryFor(
      "matmul",
      R"json("BM": 8, "BN": 32, "BK": 32, "mma_shape": "32x8x16")json"));
  ASSERT_FALSE(file.path.empty());

  auto matches =
      loadScheduleDB(file.path, /*M_bucket=*/2, /*N=*/64, /*K=*/64, "matmul");
  ASSERT_EQ(matches.size(), 1u);
  EXPECT_EQ(matches.front().mma_shape, "32x8x16");
  EXPECT_EQ(matches.front().fragment_shape, "32x8x16");
}

TEST(ScheduleLoader, EntriesForOtherOperationsOrBucketsAreSkipped) {
  TempScheduleFile file(entryFor("fused_swiglu", kPreM11Schedule));
  ASSERT_FALSE(file.path.empty());

  EXPECT_TRUE(
      loadScheduleDB(file.path, /*M_bucket=*/2, 64, 64, "matmul").empty());
  EXPECT_TRUE(loadScheduleDB(file.path, /*M_bucket=*/3, 64, 64, "fused_swiglu")
                  .empty());
  EXPECT_EQ(
      loadScheduleDB(file.path, /*M_bucket=*/2, 64, 64, "fused_swiglu").size(),
      1u);
}

// The database used for real must keep parsing: the loader is what the
// schedule-selection pass and both Micro-IR exports read.
TEST(ScheduleLoader, ShippedDatabaseStillLoads) {
  auto matches = loadScheduleDB(LLK_SCHEDULE_DB, /*M_bucket=*/2, /*N=*/4096,
                                /*K=*/4096, "fused_swiglu");
  ASSERT_FALSE(matches.empty());

  const ScheduleEntry &entry = matches.front();
  EXPECT_EQ(entry.BM, 8);
  EXPECT_EQ(entry.BN, 64);
  EXPECT_EQ(entry.BK, 64);
  EXPECT_EQ(entry.num_threads, 8);
  EXPECT_EQ(entry.grain_size, 2);

  // No shipped entry spells out the Micro fields yet, so the defaults apply.
  EXPECT_EQ(entry.pipeline_stages, 1);
  EXPECT_EQ(entry.prefetch_distance, 0);
  EXPECT_EQ(entry.memory_path, "dram:sram:acc");
  EXPECT_EQ(entry.tile_layout, "row_major");
  EXPECT_EQ(entry.fragment_shape, entry.mma_shape);
}

TEST(ScheduleLoader, UnreadableDatabaseFallsBackToTheBuiltInSchedule) {
  auto matches =
      loadScheduleDB("/nonexistent/llk/schedule_db.json", 2, 64, 64, "matmul");
  EXPECT_TRUE(matches.empty());

  ScheduleEntry entry = selectBestSchedule(matches, /*N=*/64, /*K=*/64);
  EXPECT_EQ(entry.BM, 8);
  EXPECT_EQ(entry.BN, 32);
  EXPECT_EQ(entry.BK, 32);
  EXPECT_EQ(entry.parallel_axis, "n");

  // The built-in schedule is a pre-M11 ScheduleEntry, so the Micro/tile fields
  // are the struct defaults the exports rely on.
  EXPECT_EQ(entry.pipeline_stages, 1);
  EXPECT_EQ(entry.prefetch_distance, 0);
  EXPECT_EQ(entry.memory_path, "dram:sram:acc");
  EXPECT_EQ(entry.owner_mapping, "worker");
  EXPECT_EQ(entry.fragment_shape, entry.mma_shape);
}

} // namespace
} // namespace mlir::llk
