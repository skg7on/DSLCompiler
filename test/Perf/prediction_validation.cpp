//===- prediction_validation.cpp - Prediction error tests --------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Perf/PredictionValidation.h"
#include "measurement_test_support.h"
#include "gtest/gtest.h"
#include <functional>
#include <limits>

using namespace mlir::llk::perf;

namespace {
MeasurementRecord record(llvm::StringRef identity, double measured,
                         double predicted = 100.0) {
  auto r = validRecord();
  r.executableIdentity = identity.str();
  r.executableHash = measurementContentHash(r.executableIdentity);
  r.samplesNs = {measured, measured, measured};
  r.medianNs = measured;
  r.prediction.predictedNs = predicted;
  return r;
}

TEST(PredictionValidation, ComputesRelativeErrorAgainstMeasuredLatency) {
  auto result =
      validatePredictions({record("first", 80), record("second", 200)});
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  ASSERT_EQ(result->rows.size(), 2u);
  EXPECT_EQ(result->rows[0].identity, "first");
  EXPECT_DOUBLE_EQ(result->rows[0].absoluteErrorNs, 20.0);
  EXPECT_DOUBLE_EQ(result->rows[0].signedRelativeError, 0.25);
  EXPECT_DOUBLE_EQ(result->rows[1].signedRelativeError, -0.5);
  EXPECT_DOUBLE_EQ(result->rows[0].absoluteRelativeError, 0.25);
  EXPECT_DOUBLE_EQ(result->rows[1].absoluteRelativeError, 0.5);
  EXPECT_DOUBLE_EQ(result->medianAbsoluteRelativeError, 0.375);
  EXPECT_DOUBLE_EQ(result->p90AbsoluteRelativeError, 0.5);
  auto json = predictionValidationToJson(*result);
  EXPECT_EQ(json.getString("status"), "ok");
  EXPECT_EQ(json.getString("prediction_origin"), "predicted_static");
  EXPECT_EQ(json.getInteger("record_count"), 2);
  EXPECT_EQ(json.getInteger("sample_count"), 6);
}

TEST(PredictionValidation, PreservesInputOrderAndUsesNearestRankP90) {
  std::vector<MeasurementRecord> records;
  for (int i = 10; i >= 1; --i)
    records.push_back(record(std::to_string(i), 100, 100 + i * 10));
  auto result = validatePredictions(records);
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  EXPECT_EQ(result->rows.front().identity, "10");
  EXPECT_EQ(result->rows.back().identity, "1");
  EXPECT_DOUBLE_EQ(result->medianAbsoluteRelativeError, 0.55);
  EXPECT_DOUBLE_EQ(result->p90AbsoluteRelativeError, 0.9);
}

TEST(PredictionValidation, EmptyAndExcludedGroupsHaveNullSummaries) {
  auto empty = validatePredictions({});
  ASSERT_TRUE(bool(empty));
  auto json = predictionValidationToJson(*empty);
  EXPECT_EQ(json.getString("status"), "no_comparable_samples");
  ASSERT_NE(json.get("median_absolute_relative_error"), nullptr);
  EXPECT_EQ(json.get("median_absolute_relative_error")->kind(),
            llvm::json::Value::Null);
  EXPECT_EQ(json.get("p90_absolute_relative_error")->kind(),
            llvm::json::Value::Null);
  auto missing = record("no-clock", 80, 0);
  missing.prediction.predictedCycles = 300;
  auto result = validatePredictions({missing, record("known", 200)});
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  ASSERT_EQ(result->rows.size(), 1u);
  ASSERT_EQ(result->excluded.size(), 1u);
  EXPECT_NE(result->excluded.front().find("no_prediction_time_basis"),
            std::string::npos);
  EXPECT_DOUBLE_EQ(result->medianAbsoluteRelativeError, 0.5);
  auto onlyMissing = validatePredictions({missing});
  ASSERT_TRUE(bool(onlyMissing));
  EXPECT_EQ(predictionValidationToJson(*onlyMissing).getString("status"),
            "no_comparable_samples");
}

TEST(PredictionValidation, RejectsDuplicateIdentities) {
  auto result = validatePredictions({record("same", 80), record("same", 200)});
  ASSERT_FALSE(bool(result));
  EXPECT_NE(llvm::toString(result.takeError()).find("duplicate"),
            std::string::npos);
}

TEST(PredictionValidation, RejectsEveryIncomparableRegimeField) {
  using Change = std::function<void(MeasurementRecord &)>;
  for (const Change &change : std::vector<Change>{
           [](auto &r) { r.targetIdentity += "changed"; },
           [](auto &r) { r.backendIdentity = "reference"; },
           [](auto &r) { r.environment.architecture += "changed"; },
           [](auto &r) { r.environment.cpuIdentity += "changed"; },
           [](auto &r) { r.environment.isa += "changed"; },
           [](auto &r) { r.environment.threads++; },
           [](auto &r) { r.environment.affinity += "changed"; },
           [](auto &r) { r.environment.clockPolicy += "changed"; },
           [](auto &r) { r.protocol.warmup++; },
           [](auto &r) { r.verification.fixtureContent += "changed"; },
           [](auto &r) { r.verification.verifierVersion += "changed"; },
           [](auto &r) { r.verification.numericalContract += "changed"; },
           [](auto &r) { r.sourceAdapter += "changed"; },
           [](auto &r) { r.workloadContract += "dtype/shape-changed"; },
           [](auto &r) { r.candidateBindings += "changed"; },
           [](auto &r) { r.bottleneck += "changed"; },
           [](auto &r) { r.compilerVersion += "changed"; }}) {
    auto changed = record("second", 200);
    change(changed);
    auto result = validatePredictions({record("first", 80), changed});
    ASSERT_FALSE(bool(result));
    EXPECT_NE(llvm::toString(result.takeError()).find("mixed regime"),
              std::string::npos);
  }
}

TEST(PredictionValidation, RejectsInvalidRecordsAndNonfiniteErrorArithmetic) {
  for (double value : {-1.0, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
    auto result = validatePredictions({record("bad", 80, value)});
    ASSERT_FALSE(bool(result));
    llvm::consumeError(result.takeError());
  }
  auto bad = record("unverified", 80);
  bad.verified = false;
  auto invalid = validatePredictions({bad});
  ASSERT_FALSE(bool(invalid));
  llvm::consumeError(invalid.takeError());
  auto overflow = validatePredictions({record("overflow", 1e-300, 1e300)});
  ASSERT_FALSE(bool(overflow));
  EXPECT_NE(llvm::toString(overflow.takeError()).find("nonfinite"),
            std::string::npos);
}
TEST(PredictionValidation,
     RejectsFalseMedianEvenWhenLargeSamplesWouldOverflow) {
  auto large = record("large-samples", 80);
  large.protocol.repeat = 2;
  large.samplesNs = {std::numeric_limits<double>::max(),
                     std::numeric_limits<double>::max()};
  auto result = validatePredictions({large});
  EXPECT_FALSE(bool(result));
  if (!result)
    EXPECT_NE(llvm::toString(result.takeError()).find("median_ns"),
              std::string::npos);
  large.medianNs = std::numeric_limits<double>::max();
  large.prediction.predictedNs = std::numeric_limits<double>::max();
  auto valid = validatePredictions({large});
  ASSERT_TRUE(bool(valid)) << llvm::toString(valid.takeError());
  EXPECT_DOUBLE_EQ(valid->rows.front().absoluteRelativeError, 0.0);
}

} // namespace
