//===- ShapeProfile.h - Explicit concrete samples for dynamic analysis ----===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A dynamic kernel's declared shape domain can be far too large to analyse
// symbolically. Optimization therefore ranks it only on explicit concrete
// samples: each sample binds every symbol the canonical shape contract declares
// to a validated nonnegative extent and carries a positive weight. The
// executable domain is unchanged -- samples are observations used for ranking,
// never replacements for the compiled artifact's runtime dimensions.
//
// `ShapeProfile` is plain data. It depends on LLVM Support and the plain shape
// contract, not on MLIR, adapters, mapping or runtime, so a driver, a report
// reader and the tuning core can all share one validated profile.
//
// The initial schema (dynamic-shape design §7) is:
//
//   {"version":1,
//    "samples":[{"values":{"M":7,"N":13,"K":5},"weight":1.0}],
//    "aggregation":"weighted_mean",
//    "max_variants":1}
//
// Only `weighted_mean` is accepted. Weights must be finite and positive;
// `max_variants` must be positive and defaults to 1.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_PERF_SHAPEPROFILE_H
#define LLK_PERF_SHAPEPROFILE_H

#include "LLK/Shape/ShapeContract.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::llk::perf {

/// One concrete observation of the symbolic shape domain.
struct ShapeSample {
  shape::ShapeValues values;
  /// A finite, positive sampling weight. The default is one observation.
  double weight = 1.0;
};

/// An ordered sample set plus the aggregation policy that combines its costs.
struct ShapeProfile {
  uint32_t version = 1;
  std::vector<ShapeSample> samples;
  std::string aggregation = "weighted_mean";
  /// The maximum number of compiled variants this profile may select. Version 1
  /// ranks against the baseline only; the bound is recorded now so a later
  /// specialization stage cannot silently grow it.
  uint32_t maxVariants = 1;
};

/// Checks a profile against the contract whose symbols it must bind.
///
/// Rejects: an unsupported version, an aggregation other than `weighted_mean`,
/// a zero `maxVariants`, an empty sample list, a non-finite or non-positive
/// weight, a sample that omits a declared symbol or names an undeclared one,
/// and a value that is negative or violates its symbol constraint. An empty
/// sample list is rejected because a profile is the *request* to rank on
/// concrete observations; a caller that has none must not ask for a ranking.
llvm::Error validateShapeProfile(const shape::ShapeContract &contract,
                                 const ShapeProfile &profile);

/// Parses the version-1 JSON schema exactly. Unknown fields, non-integer
/// symbol values, malformed JSON and missing required fields fail with a
/// `shape_profile:`-prefixed diagnostic. Semantic validation is separate:
/// call `validateShapeProfile` with the contract the profile belongs to.
llvm::Expected<ShapeProfile> parseShapeProfile(llvm::StringRef json);

/// Deterministic, length-delimited rendering of a profile. Samples keep their
/// declared order (the profile is an ordered observation set); values and the
/// symbol table are rendered lexically. Used for report identity and for
/// detecting a changed profile across runs.
std::string canonicalShapeProfile(const ShapeProfile &profile);

/// The weight used to combine one sample's cost. Exposed so a caller that
/// already validated the profile can weight per-sample costs without repeating
/// the normalization rule.
double sampleWeight(const ShapeSample &sample);

} // namespace mlir::llk::perf

#endif // LLK_PERF_SHAPEPROFILE_H
