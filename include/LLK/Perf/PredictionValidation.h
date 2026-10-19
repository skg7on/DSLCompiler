//===- PredictionValidation.h - Comparable prediction error ---------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLK_PERF_PREDICTIONVALIDATION_H
#define LLK_PERF_PREDICTIONVALIDATION_H

#include "LLK/Perf/MeasurementRecord.h"
#include <optional>

namespace mlir::llk::perf {

struct PredictionErrorRow {
  std::string identity;
  double predictedNs = 0.0;
  double measuredNs = 0.0;
  double absoluteErrorNs = 0.0;
  double signedRelativeError = 0.0;
  double absoluteRelativeError = 0.0;
};

struct PredictionValidation {
  std::vector<PredictionErrorRow> rows;
  std::vector<std::string> excluded;
  double medianAbsoluteRelativeError = 0.0;
  double p90AbsoluteRelativeError = 0.0;
  /// One validated record supplies the group's complete regime metadata.
  std::optional<MeasurementRecord> regime;
  uint64_t sampleCount = 0;
};

/// Full, length-delimited comparability key. Includes target, environment,
/// backend, compiler, protocol, verification and workload/decision metadata;
/// excludes executable identity, samples, prediction and timestamp.
std::string predictionValidationRegimeKey(const MeasurementRecord &record);

/// Validates one comparable regime, preserving row order. Invalid records,
/// duplicate executable identities and mixed regimes are errors. A zero/missing
/// predicted time is excluded rather than inferred from cycles without a clock.
llvm::Expected<PredictionValidation>
validatePredictions(llvm::ArrayRef<MeasurementRecord> records);

/// Empty/excluded-only groups have null summaries and no_comparable_samples.
llvm::json::Object predictionValidationToJson(const PredictionValidation &);

} // namespace mlir::llk::perf
#endif
