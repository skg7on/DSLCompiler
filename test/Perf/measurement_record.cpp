//===- measurement_record.cpp - Verified measurement record tests ---------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Covers framework D1 (#161). A record is reusable evidence, not a hint: it
// must carry the complete executable/target/backend/compiler identity, the
// measurement environment and protocol, and the fixture/verifier/numerical
// contract. Missing, malformed or internally inconsistent records are refused
// with a field-specific error so no rankable zero metric is invented.

#include "LLK/Perf/MeasurementRecord.h"
#include "measurement_test_support.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include "gtest/gtest.h"

#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace mlir::llk::perf {
namespace {

using Mutator = std::function<void(MeasurementRecord &)>;

void expectValidationError(const MeasurementRecord &record,
                           llvm::StringRef field) {
  llvm::Error error = validateMeasurementRecord(record);
  ASSERT_TRUE(bool(error)) << "expected rejection naming " << field.str();
  const std::string message = llvm::toString(std::move(error));
  EXPECT_NE(message.find("measurement_record:"), std::string::npos) << message;
  EXPECT_NE(message.find(field.str()), std::string::npos) << message;
}

void expectJsonError(llvm::json::Object object, llvm::StringRef field) {
  auto decoded =
      measurementRecordFromJson(llvm::json::Value(std::move(object)));
  ASSERT_FALSE(bool(decoded)) << "expected malformed JSON to be refused";
  const std::string message = llvm::toString(decoded.takeError());
  EXPECT_NE(message.find("measurement_record:"), std::string::npos) << message;
  EXPECT_NE(message.find(field.str()), std::string::npos) << message;
}

TEST(MeasurementRecord, RoundTripsVerifiedSamplesAndPredictionSeparately) {
  const auto record = validRecord();
  auto decoded = measurementRecordFromJson(
      llvm::json::Value(measurementRecordToJson(record)));
  ASSERT_TRUE(bool(decoded)) << llvm::toString(decoded.takeError());
  EXPECT_TRUE(decoded->verified);
  EXPECT_EQ(decoded->samplesNs, record.samplesNs);
  EXPECT_EQ(decoded->medianNs, 90.0);
  EXPECT_EQ(decoded->prediction.predictedNs, 100.0);
  EXPECT_EQ(decoded->prediction.predictedCycles, 300u);
  // The prediction is not relabeled as a measurement by serialization.
  EXPECT_FALSE(decoded->prediction.measuredNs.has_value());
  EXPECT_FALSE(decoded->prediction.measuredGflops.has_value());
  EXPECT_EQ(canonicalMeasurementKey(*decoded), canonicalMeasurementKey(record));
  EXPECT_EQ(decoded->environment.threads, 1u);
  EXPECT_EQ(decoded->protocol.repeat, 3u);
  EXPECT_EQ(decoded->verification.verifierVersion, "scalar-f32-v1");
}

TEST(MeasurementRecord, AcceptsEvenSampleCountWithMeanMedian) {
  auto record = validRecord();
  record.samplesNs = {110.0, 80.0, 100.0, 90.0};
  record.protocol.repeat = 4;
  record.medianNs = 95.0;
  if (llvm::Error error = validateMeasurementRecord(record))
    FAIL() << llvm::toString(std::move(error));

  auto decoded = measurementRecordFromJson(
      llvm::json::Value(measurementRecordToJson(record)));
  ASSERT_TRUE(bool(decoded)) << llvm::toString(decoded.takeError());
  EXPECT_EQ(decoded->samplesNs, record.samplesNs);
  EXPECT_EQ(decoded->medianNs, 95.0);
}

TEST(MeasurementRecord, RejectsUnverifiedRecord) {
  auto record = validRecord();
  record.verified = false;
  expectValidationError(record, "verified");
}

TEST(MeasurementRecord, RejectsUnknownRecordVersion) {
  auto record = validRecord();
  record.version = 2;
  expectValidationError(record, "version");
}

TEST(MeasurementRecord, RejectsMissingIdentityFields) {
  const std::pair<const char *, Mutator> cases[] = {
      {"executable_identity",
       [](MeasurementRecord &r) { r.executableIdentity.clear(); }},
      {"executable_hash",
       [](MeasurementRecord &r) { r.executableHash.clear(); }},
      {"target_identity",
       [](MeasurementRecord &r) { r.targetIdentity.clear(); }},
      {"backend_identity",
       [](MeasurementRecord &r) { r.backendIdentity.clear(); }},
      {"environment.architecture",
       [](MeasurementRecord &r) { r.environment.architecture.clear(); }},
      {"environment.cpu_identity",
       [](MeasurementRecord &r) { r.environment.cpuIdentity.clear(); }},
      {"environment.isa",
       [](MeasurementRecord &r) { r.environment.isa.clear(); }},
      {"environment.affinity",
       [](MeasurementRecord &r) { r.environment.affinity.clear(); }},
      {"environment.clock_policy",
       [](MeasurementRecord &r) { r.environment.clockPolicy.clear(); }},
      {"verification.fixture_content",
       [](MeasurementRecord &r) { r.verification.fixtureContent.clear(); }},
      {"verification.verifier_version",
       [](MeasurementRecord &r) { r.verification.verifierVersion.clear(); }},
      {"verification.numerical_contract",
       [](MeasurementRecord &r) { r.verification.numericalContract.clear(); }},
      {"source_adapter", [](MeasurementRecord &r) { r.sourceAdapter.clear(); }},
      {"workload_contract",
       [](MeasurementRecord &r) { r.workloadContract.clear(); }},
      {"candidate_bindings",
       [](MeasurementRecord &r) { r.candidateBindings.clear(); }},
      {"compiler_version",
       [](MeasurementRecord &r) { r.compilerVersion.clear(); }},
  };
  for (const auto &[field, mutate] : cases) {
    MeasurementRecord record = validRecord();
    mutate(record);
    expectValidationError(record, field);
  }
}

TEST(MeasurementRecord, RejectsZeroThreads) {
  auto record = validRecord();
  record.environment.threads = 0;
  expectValidationError(record, "environment.threads");
}

TEST(MeasurementRecord, RejectsInvalidProtocolFields) {
  {
    auto record = validRecord();
    record.protocol.version = 2;
    expectValidationError(record, "protocol.version");
  }
  {
    auto record = validRecord();
    record.protocol.timer = "wall_clock";
    expectValidationError(record, "protocol.timer");
  }
  {
    auto record = validRecord();
    record.protocol.statistic = "mean";
    expectValidationError(record, "protocol.statistic");
  }
  {
    auto record = validRecord();
    record.protocol.scope = "whole_kernel";
    expectValidationError(record, "protocol.scope");
  }
}

TEST(MeasurementRecord, RejectsEmptySamples) {
  auto record = validRecord();
  record.samplesNs.clear();
  record.protocol.repeat = 0;
  expectValidationError(record, "samples_ns");
}

TEST(MeasurementRecord, RejectsRepeatThatDoesNotMatchSamples) {
  auto record = validRecord();
  record.protocol.repeat = 4;
  expectValidationError(record, "protocol.repeat");
}

TEST(MeasurementRecord, RejectsNonpositiveAndNonfiniteSamples) {
  const double invalid[] = {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                            std::numeric_limits<double>::infinity()};
  for (double value : invalid) {
    auto record = validRecord();
    record.samplesNs = {80.0, value, 90.0};
    expectValidationError(record, "samples_ns");
  }
}

TEST(MeasurementRecord, RejectsFalseMedian) {
  {
    auto record = validRecord();
    record.medianNs = 91.0;
    expectValidationError(record, "median_ns");
  }
  {
    auto record = validRecord();
    record.medianNs = std::numeric_limits<double>::quiet_NaN();
    expectValidationError(record, "median_ns");
  }
}

TEST(MeasurementRecord, RejectsHashThatDoesNotMatchIdentity) {
  auto record = validRecord();
  record.executableHash = measurementContentHash("some-other-executable");
  expectValidationError(record, "executable_hash");
}

TEST(MeasurementRecord, RejectsMissingJsonKeys) {
  {
    auto object = measurementRecordToJson(validRecord());
    object.erase("executable_identity");
    expectJsonError(std::move(object), "executable_identity");
  }
  {
    auto object = measurementRecordToJson(validRecord());
    object.erase("median_ns");
    expectJsonError(std::move(object), "median_ns");
  }
  {
    auto object = measurementRecordToJson(validRecord());
    object.erase("verification");
    expectJsonError(std::move(object), "verification");
  }
}

TEST(MeasurementRecord, RejectsMalformedJsonTypes) {
  {
    auto object = measurementRecordToJson(validRecord());
    object["version"] = "one";
    expectJsonError(std::move(object), "version");
  }
  {
    auto object = measurementRecordToJson(validRecord());
    object["samples_ns"] = "not-an-array";
    expectJsonError(std::move(object), "samples_ns");
  }
  {
    auto object = measurementRecordToJson(validRecord());
    object["environment"] = 7;
    expectJsonError(std::move(object), "environment");
  }
  {
    auto object = measurementRecordToJson(validRecord());
    auto *samples = object["samples_ns"].getAsArray();
    ASSERT_NE(samples, nullptr);
    (*samples)[0] = "eighty";
    expectJsonError(std::move(object), "samples_ns");
  }
  {
    auto object = measurementRecordToJson(validRecord());
    auto *protocol = object["protocol"].getAsObject();
    ASSERT_NE(protocol, nullptr);
    (*protocol)["warmup"] = -1;
    expectJsonError(std::move(object), "protocol.warmup");
  }
}

TEST(MeasurementRecord, RejectsInvalidRecordCarriedAsJson) {
  auto object = measurementRecordToJson(validRecord());
  object["verified"] = false;
  expectJsonError(std::move(object), "verified");
}

TEST(MeasurementRecord, CanonicalKeyCoversReproducibleIdentity) {
  const auto base = canonicalMeasurementKey(validRecord());

  auto changed = validRecord();
  changed.executableIdentity += "-changed";
  changed.executableHash = measurementContentHash(changed.executableIdentity);
  EXPECT_NE(canonicalMeasurementKey(changed), base);

  changed = validRecord();
  changed.targetIdentity += "-changed";
  EXPECT_NE(canonicalMeasurementKey(changed), base);

  changed = validRecord();
  changed.backendIdentity += "-changed";
  EXPECT_NE(canonicalMeasurementKey(changed), base);

  changed = validRecord();
  changed.environment.threads = 2;
  EXPECT_NE(canonicalMeasurementKey(changed), base);

  changed = validRecord();
  changed.environment.affinity = "pinned-core-3";
  EXPECT_NE(canonicalMeasurementKey(changed), base);

  changed = validRecord();
  changed.protocol.warmup = 5;
  EXPECT_NE(canonicalMeasurementKey(changed), base);

  changed = validRecord();
  changed.verification.fixtureContent += "-changed";
  EXPECT_NE(canonicalMeasurementKey(changed), base);

  changed = validRecord();
  changed.verification.verifierVersion = "scalar-f32-v2";
  EXPECT_NE(canonicalMeasurementKey(changed), base);

  changed = validRecord();
  changed.verification.numericalContract = "relaxed-f32";
  EXPECT_NE(canonicalMeasurementKey(changed), base);

  changed = validRecord();
  changed.sourceAdapter += "-changed";
  EXPECT_NE(canonicalMeasurementKey(changed), base);

  changed = validRecord();
  changed.workloadContract += "-changed";
  EXPECT_NE(canonicalMeasurementKey(changed), base);

  changed = validRecord();
  changed.compilerVersion += "-changed";
  EXPECT_NE(canonicalMeasurementKey(changed), base);
}

TEST(MeasurementRecord, CanonicalKeyExcludesSamplesAndPrediction) {
  const auto base = canonicalMeasurementKey(validRecord());

  auto changed = validRecord();
  changed.samplesNs = {70.0, 60.0, 65.0};
  changed.medianNs = 65.0;
  EXPECT_EQ(canonicalMeasurementKey(changed), base);

  changed = validRecord();
  changed.timestampUtc = "2026-10-18T08:00:00Z";
  EXPECT_EQ(canonicalMeasurementKey(changed), base);

  changed = validRecord();
  changed.prediction.predictedNs = 1234.0;
  changed.prediction.predictedCycles = 999;
  EXPECT_EQ(canonicalMeasurementKey(changed), base);
}

TEST(MeasurementRecord, ContentHashKeepsDecimalFnv1aAlgorithm) {
  EXPECT_EQ(measurementContentHash("abc"), "16654208175385433931");
  EXPECT_EQ(measurementContentHash("complete-test-executable-identity"),
            "13530320208132459473");
  EXPECT_EQ(measurementContentHash(""), "14695981039346656037");
}

} // namespace
} // namespace mlir::llk::perf
