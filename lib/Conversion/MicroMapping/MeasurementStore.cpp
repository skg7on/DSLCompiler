//===- MeasurementStore.cpp - Exact-key verified measurement store --------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/MicroMapping/MeasurementStore.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <utility>

namespace mlir::llk::tuning {
namespace {

llvm::Error storeError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::Twine("measurement_store: ") + message);
}

void appendField(std::string &out, llvm::StringRef value) {
  out += std::to_string(value.size());
  out += ':';
  out += value;
}

/// The complete lookup key. `executableIdentity` is the canonical mapped
/// identity, which already names the source contract, plan, target, backend,
/// ABI and compiler content. Environment, protocol and verification are
/// tagged separately. Samples, median, prediction and timestamp are not part
/// of an identity and never distinguish stored records.
std::string canonicalLookupKey(llvm::StringRef executableIdentity,
                               const perf::MeasurementEnvironment &environment,
                               const perf::MeasurementProtocol &protocol,
                               const perf::VerificationIdentity &verification) {
  std::string key;
  appendField(key, "measurement-lookup-v1");
  appendField(key, executableIdentity);
  appendField(key, environment.architecture);
  appendField(key, environment.cpuIdentity);
  appendField(key, environment.isa);
  appendField(key, std::to_string(environment.threads));
  appendField(key, environment.affinity);
  appendField(key, environment.clockPolicy);
  appendField(key, std::to_string(protocol.version));
  appendField(key, std::to_string(protocol.warmup));
  appendField(key, std::to_string(protocol.repeat));
  appendField(key, protocol.timer);
  appendField(key, protocol.statistic);
  appendField(key, protocol.scope);
  appendField(key, verification.fixtureContent);
  appendField(key, verification.verifierVersion);
  appendField(key, verification.numericalContract);
  return key;
}

std::string canonicalLookupKey(const perf::MeasurementRecord &record) {
  return canonicalLookupKey(record.executableIdentity, record.environment,
                            record.protocol, record.verification);
}

std::string canonicalLookupKey(const MeasurementLookup &lookup) {
  return canonicalLookupKey(lookup.executableIdentity, lookup.environment,
                            lookup.protocol, lookup.verification);
}

/// The digest names the file for indexing only. The full key is compared after
/// reading, so a digest collision cannot supply the wrong record.
std::string filePath(llvm::StringRef directory, llvm::StringRef canonical) {
  llvm::SmallString<128> path(directory);
  llvm::sys::path::append(path,
                          perf::measurementContentHash(canonical) + ".json");
  return path.str().str();
}

llvm::Expected<perf::MeasurementRecord> readRecord(llvm::StringRef path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return storeError(llvm::Twine("cannot read '") + path +
                      "': " + buffer.getError().message());
  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse((*buffer)->getBuffer());
  if (!parsed)
    return storeError(llvm::Twine("invalid JSON in '") + path +
                      "': " + llvm::toString(parsed.takeError()));
  llvm::Expected<perf::MeasurementRecord> record =
      perf::measurementRecordFromJson(*parsed);
  if (!record)
    return storeError(llvm::Twine("invalid record in '") + path +
                      "': " + llvm::toString(record.takeError()));
  return record;
}

void appendJson(llvm::raw_ostream &out, const perf::MeasurementRecord &record) {
  out << llvm::json::Value(perf::measurementRecordToJson(record));
}

} // namespace

MeasurementStore::MeasurementStore(std::string directory)
    : directory(std::move(directory)) {}

llvm::Error MeasurementStore::write(const perf::MeasurementRecord &record) {
  if (llvm::Error error = perf::validateMeasurementRecord(record))
    return error;

  if (std::error_code error = llvm::sys::fs::create_directories(directory))
    return storeError(llvm::Twine("cannot create directory '") + directory +
                      "': " + error.message());

  const std::string canonical = canonicalLookupKey(record);
  const std::string path = filePath(directory, canonical);

  // A file at this digest path may belong to a different complete key. Refuse
  // to overwrite it, and refuse to reuse its content.
  if (llvm::sys::fs::exists(path)) {
    llvm::Expected<perf::MeasurementRecord> existing = readRecord(path);
    if (!existing)
      return existing.takeError();
    if (canonicalLookupKey(*existing) != canonical)
      return storeError(llvm::Twine("refusing to overwrite '") + path +
                        "': stored record has a different canonical key");
  }

  std::string text;
  llvm::raw_string_ostream stream(text);
  appendJson(stream, record);
  stream.flush();

  int fd = -1;
  llvm::SmallString<128> temporary;
  std::error_code error =
      llvm::sys::fs::createUniqueFile(path + ".tmp-%%%%%%", fd, temporary);
  if (error)
    return storeError(llvm::Twine("cannot create a temporary file for '") +
                      path + "': " + error.message());

  bool failed = false;
  std::string failure;
  {
    llvm::raw_fd_ostream out(fd, /*shouldClose=*/true);
    out << text;
    out.flush();
    if (out.has_error()) {
      failed = true;
      failure = "cannot write '" + temporary.str().str() +
                "': " + out.error().message();
      out.clear_error();
    }
  }
  if (!failed) {
    error = llvm::sys::fs::rename(temporary, path);
    if (error) {
      failed = true;
      failure = "cannot replace '" + path + "': " + error.message();
    }
  }
  if (failed) {
    // Remove only the temporary this call created; the intact record, if any,
    // is never touched.
    if (std::error_code cleanup = llvm::sys::fs::remove(temporary))
      failure += " (temporary cleanup failed: " + cleanup.message() + ")";
    return storeError(failure);
  }
  return llvm::Error::success();
}

llvm::Expected<std::optional<perf::MeasurementRecord>>
MeasurementStore::lookup(const MeasurementLookup &key) const {
  const std::string canonical = canonicalLookupKey(key);
  const std::string path = filePath(directory, canonical);
  if (!llvm::sys::fs::exists(path))
    return std::optional<perf::MeasurementRecord>();

  llvm::Expected<perf::MeasurementRecord> record = readRecord(path);
  if (!record)
    return record.takeError();
  if (canonicalLookupKey(*record) != canonical)
    return std::optional<perf::MeasurementRecord>();
  return std::optional<perf::MeasurementRecord>(std::move(*record));
}

} // namespace mlir::llk::tuning
