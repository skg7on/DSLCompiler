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

#include "LLK/Shape/ShapeContract.h"

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

/// The plain shape layer's namespace, so contract fields can name its types
/// without spelling the full path at every use.
namespace shape = ::mlir::llk::shape;

/// Whether a port carries data into or out of the workload. The shape layer
/// owns the tag so a workload contract and a bare shape contract agree on it.
using PortDirection = shape::PortDirection;

/// How a port touches the data it names. `None` marks a destination port that
/// is only overwritten (it carries no incoming read).
enum class PortEffect { None, Read, Write, ReadWrite };

llvm::StringRef stringifyPortDirection(PortDirection direction);
std::optional<PortDirection> symbolizePortDirection(llvm::StringRef text);
llvm::StringRef stringifyPortEffect(PortEffect effect);
std::optional<PortEffect> symbolizePortEffect(llvm::StringRef text);

/// One ordered, named port: its direction, effect, shape and element type.
///
/// Version 1 accepts positive constant extents only. Version 2 accepts
/// nonnegative constants and named symbols, so a zero or symbolic extent is a
/// legal port dimension; `symbols` declares the constraints each used symbol
/// must satisfy. A rank-zero (empty) shape is rejected in both versions.
struct WorkloadPort {
  std::string name;
  PortDirection direction = PortDirection::Input;
  PortEffect effect = PortEffect::Read;
  std::vector<shape::Dimension> shape;
  std::string elementType;
};

/// A versioned workload contract. Port order is part of identity and is never
/// sorted; `axes`, `symbols` (both std::maps) and `requiredPrimitives` are
/// unordered and are canonicalized. `numericalContract` is opaque canonical
/// semantic metadata.
///
/// Version 1 is the original integer encoding and keeps its exact bytes and
/// identity. Version 2 tags constants and symbols separately and carries the
/// declared symbol table, giving a symbolic artifact a distinct identity.
struct WorkloadContract {
  uint32_t version = 1;
  std::vector<WorkloadPort> ports;
  std::map<std::string, shape::Dimension> axes;
  std::map<std::string, shape::SymbolConstraint> symbols;
  std::string numericalContract;
  std::vector<std::string> requiredPrimitives;
};

/// Rejects unsupported versions, duplicate port names, rank-zero ports,
/// unknown dtypes, empty symbol names, undeclared or unused symbols and
/// unsatisfiable constraints, naming the field in each `workload_contract:`
/// diagnostic. Version 1 additionally rejects symbolic, zero and negative
/// extents; version 2 accepts zero and symbolic extents but not negatives.
llvm::Error validateWorkloadContract(const WorkloadContract &contract);

/// The canonical, collision-safe identity of a contract. Every variable-length
/// string is length-delimited, so delimiter-bearing names cannot collide; ports
/// keep declaration order while axes and required primitives are sorted and
/// de-duplicated. Version 1 keeps its original rendering byte for byte;
/// version 2 tags constants and symbols separately and appends the symbol
/// table. A hash is only an index into this full rendering.
std::string canonicalWorkloadContract(const WorkloadContract &contract);

/// Encodes a contract as a `DictionaryAttr` for storage on an MLIR operation.
/// Version 1 renders only positive constant extents, exactly as before;
/// version 2 renders integer or symbol-name entries plus a `symbols`
/// dictionary. A contract that does not validate, or a version 1 contract with
/// a symbolic or zero extent, fails rather than encoding an ambiguous artifact.
llvm::Expected<mlir::DictionaryAttr>
encodeWorkloadContract(mlir::MLIRContext *context,
                       const WorkloadContract &contract);

/// Decodes and validates a contract from its attribute encoding. Malformed
/// attributes and semantically invalid contracts fail with field-specific
/// diagnostics.
llvm::Expected<WorkloadContract>
decodeWorkloadContract(mlir::DictionaryAttr attr);

} // namespace mlir::llk::workload

#endif // LLK_CONVERSION_WORKLOAD_WORKLOADCONTRACT_H
