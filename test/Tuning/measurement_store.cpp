//===- measurement_store.cpp - Exact-key measurement store tests ----------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Covers framework D2 (#162). A record survives a process restart only when a
// lookup matches the complete executable identity, environment, protocol and
// verification content. A digest is an index: a file whose full key differs is
// a miss, never a reusable approximation. Corrupt content is an error, not a
// miss, so a truncated record cannot silently become missing evidence.

#include "../Perf/measurement_test_support.h"
#include "LLK/Conversion/MicroMapping/MeasurementStore.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace mlir::llk::tuning {
namespace {

std::vector<std::string> directoryEntries(llvm::StringRef directory) {
  std::vector<std::string> entries;
  std::error_code error;
  for (llvm::sys::fs::directory_iterator it(directory, error), end;
       it != end && !error; it.increment(error))
    entries.push_back(it->path());
  EXPECT_FALSE(error) << error.message();
  std::sort(entries.begin(), entries.end());
  return entries;
}

void writeTextFile(llvm::StringRef path, llvm::StringRef text) {
  std::error_code error;
  llvm::raw_fd_ostream stream(path, error);
  ASSERT_FALSE(error) << error.message();
  stream << text;
  stream.close();
}

void writeJsonFile(llvm::StringRef path, llvm::json::Object object) {
  std::error_code error;
  llvm::raw_fd_ostream stream(path, error);
  ASSERT_FALSE(error) << error.message();
  stream << llvm::json::Value(std::move(object));
  stream.close();
}

class MeasurementStoreTest : public ::testing::Test {
protected:
  llvm::SmallString<128> directory;

  void SetUp() override {
    ASSERT_FALSE(
        llvm::sys::fs::createUniqueDirectory("llk-measurement", directory));
  }

  void TearDown() override {
    llvm::sys::fs::setPermissions(directory, llvm::sys::fs::all_all);
    EXPECT_FALSE(llvm::sys::fs::remove_directories(directory));
  }

  std::string dir() const { return directory.str().str(); }

  static MeasurementLookup lookupFor(const perf::MeasurementRecord &record) {
    return {record.executableIdentity, record.environment, record.protocol,
            record.verification};
  }
};

TEST_F(MeasurementStoreTest, ReopenHitsOnlyTheCompleteIdentity) {
  const auto record = perf::validRecord();
  MeasurementStore writer(dir());
  ASSERT_FALSE(writer.write(record));
  MeasurementStore reader(dir());
  MeasurementLookup key = lookupFor(record);
  auto hit = reader.lookup(key);
  ASSERT_TRUE(bool(hit)) << llvm::toString(hit.takeError());
  ASSERT_TRUE(hit->has_value());
  EXPECT_EQ((*hit)->medianNs, 90.0);
  EXPECT_EQ((*hit)->samplesNs, record.samplesNs);
  EXPECT_EQ((*hit)->executableIdentity, record.executableIdentity);

  key.environment.threads = 2;
  auto miss = reader.lookup(key);
  ASSERT_TRUE(bool(miss)) << llvm::toString(miss.takeError());
  EXPECT_FALSE(miss->has_value());
}

TEST_F(MeasurementStoreTest, MissesEveryChangedIdentityField) {
  const auto record = perf::validRecord();
  MeasurementStore store(dir());
  ASSERT_FALSE(store.write(record));
  const MeasurementLookup base = lookupFor(record);

  auto expectMiss = [&](auto mutate) {
    MeasurementLookup changed = base;
    mutate(changed);
    auto result = store.lookup(changed);
    ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
    EXPECT_FALSE(result->has_value());
  };

  // Source/plan/backend/mode/compiler/ABI content lives in the canonical
  // executable identity.
  expectMiss([](MeasurementLookup &key) {
    key.executableIdentity += "-changed-plan";
  });
  expectMiss(
      [](MeasurementLookup &key) { key.environment.architecture = "aarch64"; });
  expectMiss([](MeasurementLookup &key) {
    key.environment.cpuIdentity = "other-cpu";
  });
  expectMiss([](MeasurementLookup &key) { key.environment.isa = "avx512"; });
  expectMiss([](MeasurementLookup &key) { key.environment.threads = 2; });
  expectMiss(
      [](MeasurementLookup &key) { key.environment.affinity = "pinned-core"; });
  expectMiss(
      [](MeasurementLookup &key) { key.environment.clockPolicy = "pinned"; });
  expectMiss([](MeasurementLookup &key) { key.protocol.version = 2; });
  expectMiss([](MeasurementLookup &key) { key.protocol.warmup = 9; });
  expectMiss([](MeasurementLookup &key) { key.protocol.repeat = 4; });
  expectMiss([](MeasurementLookup &key) { key.protocol.timer = "wall_clock"; });
  expectMiss([](MeasurementLookup &key) { key.protocol.statistic = "mean"; });
  expectMiss(
      [](MeasurementLookup &key) { key.protocol.scope = "whole_kernel"; });
  expectMiss([](MeasurementLookup &key) {
    key.verification.fixtureContent += "-changed";
  });
  expectMiss([](MeasurementLookup &key) {
    key.verification.verifierVersion = "scalar-f32-v2";
  });
  expectMiss([](MeasurementLookup &key) {
    key.verification.numericalContract = "relaxed-f32";
  });
}

TEST_F(MeasurementStoreTest, RejectsWritingAnUnusableRecord) {
  auto record = perf::validRecord();
  record.verified = false;
  MeasurementStore store(dir());
  llvm::Error error = store.write(record);
  ASSERT_TRUE(bool(error));
  const std::string message = llvm::toString(std::move(error));
  EXPECT_NE(message.find("measurement_record:"), std::string::npos) << message;
  EXPECT_TRUE(directoryEntries(dir()).empty());
}

TEST_F(MeasurementStoreTest, ReportsCorruptJsonWithItsPath) {
  const auto record = perf::validRecord();
  MeasurementStore store(dir());
  ASSERT_FALSE(store.write(record));
  const std::vector<std::string> entries = directoryEntries(dir());
  ASSERT_EQ(entries.size(), 1u);
  const std::string path = entries.front();
  writeTextFile(path, "{");

  auto hit = store.lookup(lookupFor(record));
  ASSERT_FALSE(bool(hit)) << "corrupt content must not be a miss";
  const std::string message = llvm::toString(hit.takeError());
  EXPECT_NE(message.find("measurement_store:"), std::string::npos) << message;
  EXPECT_NE(message.find(path), std::string::npos) << message;
}

TEST_F(MeasurementStoreTest, RejectsAValidRecordWithInvalidContent) {
  const auto record = perf::validRecord();
  MeasurementStore store(dir());
  ASSERT_FALSE(store.write(record));
  const std::vector<std::string> entries = directoryEntries(dir());
  ASSERT_EQ(entries.size(), 1u);
  const std::string path = entries.front();
  // A file with a valid schema but unusable content must be reported as an
  // error when read, not treated as an absent record.
  auto invalid = record;
  invalid.verified = false;
  writeJsonFile(path, perf::measurementRecordToJson(invalid));

  auto hit = store.lookup(lookupFor(record));
  ASSERT_FALSE(bool(hit));
  const std::string message = llvm::toString(hit.takeError());
  EXPECT_NE(message.find("measurement_store:"), std::string::npos) << message;
  EXPECT_NE(message.find(path), std::string::npos) << message;
}

TEST_F(MeasurementStoreTest, InterruptedTemporaryFileDoesNotMaskIntactRecord) {
  const auto record = perf::validRecord();
  MeasurementStore store(dir());
  ASSERT_FALSE(store.write(record));
  const std::vector<std::string> entries = directoryEntries(dir());
  ASSERT_EQ(entries.size(), 1u);
  writeTextFile(entries.front() + ".tmp-abandoned", "not a record");

  auto hit = store.lookup(lookupFor(record));
  ASSERT_TRUE(bool(hit)) << llvm::toString(hit.takeError());
  ASSERT_TRUE(hit->has_value());
  EXPECT_EQ((*hit)->medianNs, 90.0);
}

TEST_F(MeasurementStoreTest, DigestCollisionNeverReusesTheWrongRecord) {
  const auto record = perf::validRecord();
  auto other = perf::validRecord();
  other.executableIdentity += "-different";
  other.executableHash = perf::measurementContentHash(other.executableIdentity);

  MeasurementStore store(dir());
  ASSERT_FALSE(store.write(record));
  const std::vector<std::string> entries = directoryEntries(dir());
  ASSERT_EQ(entries.size(), 1u);
  const std::string path = entries.front();
  // Same digest-named path, different complete key.
  writeJsonFile(path, perf::measurementRecordToJson(other));

  auto hit = store.lookup(lookupFor(record));
  ASSERT_TRUE(bool(hit)) << llvm::toString(hit.takeError());
  EXPECT_FALSE(hit->has_value());

  llvm::Error error = store.write(record);
  ASSERT_TRUE(bool(error));
  const std::string message = llvm::toString(std::move(error));
  EXPECT_NE(message.find("measurement_store:"), std::string::npos) << message;
  EXPECT_NE(message.find("canonical key"), std::string::npos) << message;

  auto buffer = llvm::MemoryBuffer::getFile(path);
  ASSERT_TRUE(bool(buffer)) << buffer.getError().message();
  auto parsed = llvm::json::parse((*buffer)->getBuffer());
  ASSERT_TRUE(bool(parsed)) << llvm::toString(parsed.takeError());
  auto stored = perf::measurementRecordFromJson(*parsed);
  ASSERT_TRUE(bool(stored)) << llvm::toString(stored.takeError());
  EXPECT_EQ(stored->executableIdentity, other.executableIdentity);
}

TEST_F(MeasurementStoreTest, FailedWriteLeavesIntactRecordAndNoTemporary) {
  const auto record = perf::validRecord();
  MeasurementStore store(dir());
  ASSERT_FALSE(store.write(record));
  const std::vector<std::string> entries = directoryEntries(dir());
  ASSERT_EQ(entries.size(), 1u);
  const std::string path = entries.front();

  ASSERT_FALSE(llvm::sys::fs::setPermissions(
      directory, llvm::sys::fs::owner_read | llvm::sys::fs::owner_exe));
  auto changed = record;
  changed.samplesNs = {70.0, 60.0, 65.0};
  changed.medianNs = 65.0;
  llvm::Error error = store.write(changed);
  EXPECT_TRUE(bool(error));
  ASSERT_FALSE(
      llvm::sys::fs::setPermissions(directory, llvm::sys::fs::all_all));

  auto hit = store.lookup(lookupFor(record));
  ASSERT_TRUE(bool(hit)) << llvm::toString(hit.takeError());
  ASSERT_TRUE(hit->has_value());
  EXPECT_EQ((*hit)->medianNs, 90.0);
  EXPECT_EQ(directoryEntries(dir()), std::vector<std::string>{path});
}

} // namespace
} // namespace mlir::llk::tuning
