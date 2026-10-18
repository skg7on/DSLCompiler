//===- MeasurementRecord.h - Verified kernel measurement records ----------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Framework D1 (#161): a durable, reproducible record of one measured
// whole-kernel execution.
//
// A record is evidence, not a hint. It carries the complete executable identity
// (not only a digest), the host environment that produced the timing, the
// measurement protocol, and the fixture/verifier/numerical contract that
// accepted the result. Predicted metrics stay predicted; measured samples and
// their median stay separate. This header deliberately names no runtime or JIT
// type so LLKPerf remains independent of the execution libraries.

#ifndef LLK_PERF_MEASUREMENTRECORD_H
#define LLK_PERF_MEASUREMENTRECORD_H

#include "LLK/Perf/Candidate.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::llk::perf {

/// How the samples were taken. `repeat` must equal the number of recorded
/// samples, and the statistic is the median of those samples.
struct MeasurementProtocol {
  uint32_t version = 1;
  uint64_t warmup = 0;
  uint64_t repeat = 0;
  std::string timer = "steady_clock";
  std::string statistic = "median";
  std::string scope = "per_invocation";
};

/// The host that produced the timing. Any change here invalidates reuse.
struct MeasurementEnvironment {
  std::string architecture;
  std::string cpuIdentity;
  std::string isa;
  uint64_t threads = 0;
  std::string affinity;
  std::string clockPolicy;
};

/// The correctness gate: complete fixture content, the verifier version and
/// the numerical contract under which the measurement was accepted.
struct VerificationIdentity {
  std::string fixtureContent;
  std::string verifierVersion;
  std::string numericalContract;
};

struct MeasurementRecord {
  uint32_t version = 1;
  /// Complete canonical executable identity; the hash below is an index only.
  std::string executableIdentity;
  std::string executableHash;
  std::string targetIdentity;
  std::string backendIdentity;
  MeasurementEnvironment environment;
  MeasurementProtocol protocol;
  VerificationIdentity verification;
  bool verified = false;
  std::vector<double> samplesNs;
  double medianNs = 0.0;
  CandidateMetrics prediction;
  std::string sourceAdapter;
  std::string workloadContract;
  std::string candidateBindings;
  std::string bottleneck;
  std::string timestampUtc;
  std::string compilerVersion;
};

/// Rejects unverified, incomplete, nonpositive, nonfinite or inconsistent
/// records with a `measurement_record:` error naming the offending field.
llvm::Error validateMeasurementRecord(const MeasurementRecord &record);

/// Stable JSON form. Predicted fields stay under `prediction`; measured fields
/// are never relabeled as predictions or vice versa.
llvm::json::Object measurementRecordToJson(const MeasurementRecord &record);

/// Strict decoder: missing keys, wrong JSON types, unknown schema versions and
/// invalid record contents are errors, never a rankable zero metric.
llvm::Expected<MeasurementRecord>
measurementRecordFromJson(const llvm::json::Value &value);

/// Tagged, length-delimited key over every reproducibility field (executable,
/// target, backend, compiler, environment, protocol, verification, source and
/// contract). Sample values, the median, the prediction and the timestamp are
/// deliberately excluded; the digest is only an index into the full key.
std::string canonicalMeasurementKey(const MeasurementRecord &record);

/// The project's audited decimal-rendered FNV-1a content hash. Factoring it
/// here does not change the value used by existing executable identities.
std::string measurementContentHash(llvm::StringRef value);

} // namespace mlir::llk::perf

#endif // LLK_PERF_MEASUREMENTRECORD_H
