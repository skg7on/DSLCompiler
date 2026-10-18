//===- measurement_test_support.h - Valid measurement record fixture ------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// One complete, verified record used by the D1/D2/D4 measurement tests. Every
// identity field is populated and the content hash is derived from the
// canonical executable identity, so a test can change one field and observe
// exactly the validation or lookup behavior that field controls.

#ifndef LLK_TEST_PERF_MEASUREMENT_TEST_SUPPORT_H
#define LLK_TEST_PERF_MEASUREMENT_TEST_SUPPORT_H

#include "LLK/Perf/MeasurementRecord.h"

namespace mlir::llk::perf {

inline MeasurementRecord validRecord() {
  MeasurementRecord record;
  record.executableIdentity = "complete-test-executable-identity";
  record.executableHash = measurementContentHash(record.executableIdentity);
  record.targetIdentity = "complete-test-target-content";
  record.backendIdentity = "selected-target:test-compiler:test-abi";
  record.environment = {
      "x86_64",          "test-cpu", "avx2", 1, "single-thread-unpinned",
      "observed-default"};
  record.protocol = {1, 2, 3, "steady_clock", "median", "per_invocation"};
  record.verification = {"complete-test-fixture", "scalar-f32-v1",
                         "strict-f32"};
  record.verified = true;
  record.samplesNs = {80.0, 100.0, 90.0};
  record.medianNs = 90.0;
  record.prediction.predictedNs = 100.0;
  record.prediction.predictedCycles = 300;
  record.sourceAdapter = "linalg-elementwise-v1";
  record.workloadContract = "complete-test-contract";
  record.candidateBindings = "T0=4,T1=8";
  record.timestampUtc = "2026-10-12T00:00:00Z";
  record.compilerVersion = "test-compiler";
  return record;
}

} // namespace mlir::llk::perf

#endif // LLK_TEST_PERF_MEASUREMENT_TEST_SUPPORT_H
