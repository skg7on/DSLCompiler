//===- LLKKernelLowering.h - Registry-driven LLK lowering -------*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CONVERSION_LLKKERNELLOWERING_H
#define LLK_CONVERSION_LLKKERNELLOWERING_H
#include "LLK/Shape/ShapeContract.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include <functional>
#include <memory>
#include <string>
#include <vector>
namespace mlir::llk {
struct KernelContract {
  std::string workload;
  int64_t M = 0, N = 0, K = 0;
  Type inputElemType, accumulatorElemType, outputElemType;
  int64_t mBucket = -1;
};

/// The three contraction axes a source root declares, each either a nonnegative
/// constant or a symbol. M/N/K are adapter facts: shared generic code reads
/// them from this value rather than re-deriving them from operand positions,
/// and never treats MLIR's dynamic sentinel as an extent.
struct ContractionDimensions {
  shape::Dimension M;
  shape::Dimension N;
  shape::Dimension K;
};
struct MicroKernelFragment {
  unsigned armCount = 1;
  bool hasOutputInit = false;
  /// The left operand's source type. The bounded export keeps source port
  /// types, so an unknown extent is never replaced by a guessed constant.
  Type lhsOperandType;
  SmallVector<Type> rhsOperandTypes;
  /// The read initializer operand's source type, when `hasOutputInit`. A
  /// dynamic export keeps the source type instead of fabricating a static one.
  Type initOperandType;
  /// The caller-owned output destination's source type. A contraction family
  /// sets this so the dynamic export can name the destination port: it is the
  /// destination operand for SwiGLU (whose incoming values are not read) and
  /// the result for matmul.
  Type destinationOperandType;
  SmallVector<NamedAttribute> extraAttrs;
  std::function<Value(OpBuilder &, Location, ArrayRef<Value>)> buildEpilogue;
};
class LLKKernelLowering {
public:
  virtual ~LLKKernelLowering() = default;
  virtual StringRef getOpName() const = 0;
  virtual LogicalResult inferContract(Operation *, KernelContract &) const = 0;
  /// The symbolic contraction axes this root declares. The default reports no
  /// contraction content; contraction families override it. Static-known
  /// contradictions are errors, while an unresolved relationship becomes a
  /// symbol shared by every participating port dimension.
  virtual FailureOr<ContractionDimensions> inferDimensions(Operation *) const {
    return failure();
  }
  virtual bool hasLinalgLowering() const { return false; }
  virtual LogicalResult lowerToLinalg(Operation *, OpBuilder &) const {
    return failure();
  }
  virtual bool hasMicroLowering() const { return false; }
  /// Describes the Micro-IR fragment this root emits. `requireStatic` keeps the
  /// pre-dynamic export contract: every extent must be a positive constant.
  /// Shape-contract description passes false so a symbolic root can still
  /// report its fragment while the old export path stays untouched.
  virtual LogicalResult describeMicroKernel(Operation *, const KernelContract &,
                                            MicroKernelFragment &,
                                            bool requireStatic = true) const {
    return failure();
  }
};
class LLKKernelRegistry {
  std::vector<std::unique_ptr<LLKKernelLowering>> owned;
  llvm::StringMap<const LLKKernelLowering *> byName;
  SmallVector<const LLKKernelLowering *> ordered;

public:
  /// Duplicate names preserve the original registration and its order.
  template <typename T, typename... Args> void add(Args &&...args) {
    auto entry = std::make_unique<T>(std::forward<Args>(args)...);
    if (byName.contains(entry->getOpName()))
      return;
    byName[entry->getOpName()] = entry.get();
    ordered.push_back(entry.get());
    owned.push_back(std::move(entry));
  }
  const LLKKernelLowering *lookup(StringRef name) const;
  ArrayRef<const LLKKernelLowering *> entries() const { return ordered; }
};
void populateLLKKernelLowerings(LLKKernelRegistry &);

/// The symbolic axes `root` declares, resolved through the operator-kernel
/// registry so no shared caller branches on an operator name. Fails for a root
/// that is not a registered contraction.
FailureOr<ContractionDimensions> inferContractionDimensions(Operation *root);

/// Resolves M/N/K from a rank-2 contraction. Every static extent that pins an
/// axis must agree; an axis with no static extent becomes a symbol. Reported
/// operand titles are `left-hand side`, `right-hand side` and `result`.
FailureOr<ContractionDimensions>
inferContractionDimensions(Operation *root, Value lhs, Value rhs, Value out);

/// Unifies one axis's candidate extents with an already resolved dimension.
/// MLIR's dynamic sentinel is a wildcard; a disagreeing static extent is an
/// error naming `axis`.
FailureOr<shape::Dimension>
resolveContractionDimension(Operation *root, llvm::StringRef axis,
                            llvm::ArrayRef<int64_t> extents);

LogicalResult inferContractionContract(Operation *, Value lhs, Value rhs,
                                       Value out, Attribute accumulatorType,
                                       KernelContract &);
LogicalResult checkMicroTensor(Operation *, Value, const char *);

/// Validates a contraction's ranks, dtypes, accumulator attribute and
/// symbolic dimensions without requiring static extents. On success `dims`
/// describes M/N/K and `info` carries the element types the contract needs.
LogicalResult validateMicroContractionShape(Operation *root, Value lhs,
                                            Value rhs, Value out,
                                            Attribute accumulatorType,
                                            ContractionDimensions &dims,
                                            KernelContract &info);

/// The still-static export check: the same semantic validation, but every
/// extent must be a positive constant. Retained until the dynamic exporter
/// consumes symbolic dimensions.
LogicalResult validateMicroContraction(Operation *, Value lhs, Value rhs,
                                       Value out, Attribute accumulatorType,
                                       KernelContract &);
} // namespace mlir::llk
#endif
