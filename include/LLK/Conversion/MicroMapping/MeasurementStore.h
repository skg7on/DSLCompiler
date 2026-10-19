//===- MeasurementStore.h - Exact-key verified measurement store ----------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Framework D2 (#162): durable, local, single-writer storage for verified
// whole-kernel measurement records.
//
// A lookup matches the complete executable identity, environment, protocol and
// verification content. A content digest names the file for indexing only: a
// file whose full key differs is a miss, and a write never overwrites it.
// Corrupt content is reported as an error, never as absent evidence.

#ifndef LLK_CONVERSION_MICROMAPPING_MEASUREMENTSTORE_H
#define LLK_CONVERSION_MICROMAPPING_MEASUREMENTSTORE_H

#include "LLK/Perf/MeasurementRecord.h"

#include "llvm/Support/Error.h"

#include <optional>
#include <string>

namespace mlir::llk::tuning {

/// The complete reusable identity of one measured executable. Source, plan,
/// target, backend, ABI and compiler content are part of `executableIdentity`
/// (the canonical mapped identity); environment, protocol and verification are
/// carried separately.
struct MeasurementLookup {
  std::string executableIdentity;
  perf::MeasurementEnvironment environment;
  perf::MeasurementProtocol protocol;
  perf::VerificationIdentity verification;
};

class MeasurementStore {
public:
  explicit MeasurementStore(std::string directory);

  /// Validates the record, then atomically replaces the exact-key file. A file
  /// whose full key differs (a digest collision) is refused. On failure only
  /// the store's own temporary file is removed; existing records are intact.
  llvm::Error write(const perf::MeasurementRecord &record);

  /// Absent path or full-key mismatch -> no value. Invalid JSON, invalid
  /// record content or I/O failure -> Error naming the path.
  llvm::Expected<std::optional<perf::MeasurementRecord>>
  lookup(const MeasurementLookup &key) const;

private:
  std::string directory;
};

} // namespace mlir::llk::tuning

#endif // LLK_CONVERSION_MICROMAPPING_MEASUREMENTSTORE_H
