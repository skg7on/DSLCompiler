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
#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringMap.h"
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
struct MicroKernelFragment {
  unsigned armCount = 1;
  bool hasOutputInit = false;
  SmallVector<Type> rhsOperandTypes;
  SmallVector<NamedAttribute> extraAttrs;
  std::function<Value(OpBuilder &, Location, ArrayRef<Value>)> buildEpilogue;
};
class LLKKernelLowering {
public:
  virtual ~LLKKernelLowering() = default;
  virtual StringRef getOpName() const = 0;
  virtual LogicalResult inferContract(Operation *, KernelContract &) const = 0;
  virtual bool hasLinalgLowering() const { return false; }
  virtual LogicalResult lowerToLinalg(Operation *, OpBuilder &) const {
    return failure();
  }
  virtual bool hasMicroLowering() const { return false; }
  virtual LogicalResult describeMicroKernel(Operation *, const KernelContract &,
                                            MicroKernelFragment &) const {
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
LogicalResult inferContractionContract(Operation *, Value lhs, Value rhs,
                                       Value out, Attribute accumulatorType,
                                       KernelContract &);
LogicalResult checkMicroTensor(Operation *, Value, const char *);
LogicalResult validateMicroContraction(Operation *, Value lhs, Value rhs,
                                       Value out, Attribute accumulatorType,
                                       KernelContract &);
} // namespace mlir::llk
#endif
