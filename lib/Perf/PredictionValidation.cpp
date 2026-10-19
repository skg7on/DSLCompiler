//===- PredictionValidation.cpp - Comparable prediction error -------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Perf/PredictionValidation.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>

namespace mlir::llk::perf {
namespace {
llvm::Error validationError(const llvm::Twine &message) {
  return llvm::createStringError("prediction_validation: " + message);
}
} // namespace

std::string predictionValidationRegimeKey(const MeasurementRecord &record) {
  auto regime = record;
  regime.executableIdentity = "prediction-regime-v1";
  std::string key = canonicalMeasurementKey(regime);
  key += std::to_string(record.bottleneck.size()) + ":" + record.bottleneck;
  return key;
}

llvm::Expected<PredictionValidation>
validatePredictions(llvm::ArrayRef<MeasurementRecord> records) {
  PredictionValidation result;
  std::set<std::string> identities;
  std::string regimeKey;
  for (const auto &record : records) {
    if (auto error = validateMeasurementRecord(record))
      return std::move(error);
    if (!identities.insert(record.executableIdentity).second)
      return validationError("duplicate executable identity");
    std::string key = predictionValidationRegimeKey(record);
    if (!result.regime) {
      result.regime = record;
      regimeKey = std::move(key);
    } else if (key != regimeKey) {
      return validationError(
          "mixed regime: target, environment, backend, "
          "protocol, verification or workload metadata differ");
    }
    result.sampleCount += record.samplesNs.size();
    if (record.prediction.predictedNs == 0.0) {
      result.excluded.push_back(record.executableIdentity +
                                ": no_prediction_time_basis");
      continue;
    }
    PredictionErrorRow row;
    row.identity = record.executableIdentity;
    row.predictedNs = record.prediction.predictedNs;
    row.measuredNs = record.medianNs;
    row.absoluteErrorNs = std::abs(row.predictedNs - row.measuredNs);
    row.signedRelativeError =
        (row.predictedNs - row.measuredNs) / row.measuredNs;
    row.absoluteRelativeError = std::abs(row.signedRelativeError);
    if (!std::isfinite(row.signedRelativeError))
      return validationError("nonfinite prediction error for '" + row.identity +
                             "'");
    result.rows.push_back(std::move(row));
  }
  if (result.rows.empty())
    return result;

  // Sorting a copy leaves each provenance row in its input order.
  std::vector<double> errors;
  for (const auto &row : result.rows)
    errors.push_back(row.absoluteRelativeError);
  std::sort(errors.begin(), errors.end());
  size_t size = errors.size();
  result.medianAbsoluteRelativeError =
      size % 2 ? errors[size / 2]
               : std::midpoint(errors[size / 2 - 1], errors[size / 2]);
  result.p90AbsoluteRelativeError =
      errors[static_cast<size_t>(std::ceil(0.9 * size)) - 1];
  return result;
}

llvm::json::Object
predictionValidationToJson(const PredictionValidation &result) {
  llvm::json::Object object;
  object["version"] = 1;
  object["status"] = result.rows.empty() ? "no_comparable_samples" : "ok";
  object["prediction_origin"] = "predicted_static";
  object["measurement_origin"] = "verified_measurement_record";
  object["calibrated"] = false;
  object["record_count"] = result.rows.size() + result.excluded.size();
  object["compared_record_count"] = result.rows.size();
  object["excluded_record_count"] = result.excluded.size();
  object["sample_count"] = result.sampleCount;
  object["median_absolute_relative_error"] =
      result.rows.empty()
          ? llvm::json::Value(nullptr)
          : llvm::json::Value(result.medianAbsoluteRelativeError);
  object["p90_absolute_relative_error"] =
      result.rows.empty() ? llvm::json::Value(nullptr)
                          : llvm::json::Value(result.p90AbsoluteRelativeError);
  llvm::json::Array rows;
  for (const auto &row : result.rows)
    rows.push_back(llvm::json::Object{
        {"identity", row.identity},
        {"predicted_ns", row.predictedNs},
        {"measured_ns", row.measuredNs},
        {"absolute_error_ns", row.absoluteErrorNs},
        {"signed_relative_error", row.signedRelativeError},
        {"absolute_relative_error", row.absoluteRelativeError}});
  object["rows"] = std::move(rows);
  llvm::json::Array excluded;
  for (const auto &reason : result.excluded)
    excluded.push_back(reason);
  object["excluded"] = std::move(excluded);
  if (result.regime) {
    auto record = measurementRecordToJson(*result.regime);
    llvm::json::Object regime;
    for (const char *field :
         {"target_identity", "backend_identity", "compiler_version",
          "environment", "protocol", "verification", "source_adapter",
          "workload_contract", "candidate_bindings", "bottleneck"})
      regime[field] = *record.get(field);
    object["regime"] = std::move(regime);
  } else {
    object["regime"] = nullptr;
  }
  return object;
}
} // namespace mlir::llk::perf
