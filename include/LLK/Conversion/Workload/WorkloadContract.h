//===- WorkloadContract.h - Rank-neutral versioned workload contract ------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The rank-neutral workload contract (framework B1). A contract describes what
// a workload source exposes -- ordered named ports, optional named axes, static
// shapes and dtypes, its numerical semantics, and the primitives it requires --
// without assuming every workload has M/N/K contraction axes. Matmul and
// SwiGLU are examples; the contract itself carries no operator-specific label.
//
// This header is plain data: it depends on LLVM Support only and is consumed
// below the source-candidate adapters, performance and mapping integration.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CONVERSION_WORKLOAD_WORKLOADCONTRACT_H
#define LLK_CONVERSION_WORKLOAD_WORKLOADCONTRACT_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mlir {
class DictionaryAttr;
class MLIRContext;
} // namespace mlir

namespace mlir::llk::workload {

/// Whether a port carries data into or out of the workload.
enum class PortDirection { Input, Output };

/// How a port touches the data it names. `None` marks a destination port that
/// is only overwritten (it carries no incoming read).
enum class PortEffect { None, Read, Write, ReadWrite };

llvm::StringRef stringifyPortDirection(PortDirection direction);
std::optional<PortDirection> symbolizePortDirection(llvm::StringRef text);
llvm::StringRef stringifyPortEffect(PortEffect effect);
std::optional<PortEffect> symbolizePortEffect(llvm::StringRef text);

/// One ordered, named port: its direction, effect, static shape and element
/// type. `shape` extents are positive; dynamic (`ShapedType::kDynamic`), zero
/// and negative extents are rejected, as is a rank-zero (empty) shape.
struct WorkloadPort {
  std::string name;
  PortDirection direction = PortDirection::Input;
  PortEffect effect = PortEffect::Read;
  std::vector<int64_t> shape;
  std::string elementType;
};

/// A versioned workload contract. Port order is part of identity and is never
/// sorted; `axes` (a std::map) and `requiredPrimitives` are unordered and are
/// canonicalized. `numericalContract` is opaque canonical semantic metadata.
struct WorkloadContract {
  uint32_t version = 1;
  std::vector<WorkloadPort> ports;
  std::map<std::string, int64_t> axes;
  std::string numericalContract;
  std::vector<std::string> requiredPrimitives;
};

/// Rejects unsupported versions, duplicate port names, rank-zero ports,
/// non-positive (dynamic/zero/negative) extents and unknown dtypes, naming the
/// field in each `workload_contract:` diagnostic.
llvm::Error validateWorkloadContract(const WorkloadContract &contract);

/// The canonical, collision-safe identity of a contract. Every variable-length
/// string is length-delimited, so delimiter-bearing names cannot collide; ports
/// keep declaration order while axes and required primitives are sorted and
/// de-duplicated. A hash is only an index into this full rendering.
std::string canonicalWorkloadContract(const WorkloadContract &contract);

/// Encodes a contract as a `DictionaryAttr` for storage on an MLIR operation.
mlir::DictionaryAttr encodeWorkloadContract(mlir::MLIRContext *context,
                                            const WorkloadContract &contract);

/// Decodes and validates a contract from its attribute encoding. Malformed
/// attributes and semantically invalid contracts fail with field-specific
/// diagnostics.
llvm::Expected<WorkloadContract>
decodeWorkloadContract(mlir::DictionaryAttr attr);

} // namespace mlir::llk::workload

#endif // LLK_CONVERSION_WORKLOAD_WORKLOADCONTRACT_H
