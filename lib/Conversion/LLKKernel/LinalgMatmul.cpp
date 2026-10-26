//===- LinalgMatmul.cpp - Linalg matmul operator lowering -----------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Dialect/LLKDialect.h"                  // IWYU pragma: keep
#include "LLK/Dialect/LLKEnums.h"                    // IWYU pragma: keep
#include "LLK/Transforms/Common/MathApproximation.h" // IWYU pragma: keep

#include "mlir/Bytecode/BytecodeOpInterface.h" // IWYU pragma: keep
#include "mlir/Dialect/Arith/IR/Arith.h"       // IWYU pragma: keep
#include "mlir/Dialect/Linalg/IR/Linalg.h"     // IWYU pragma: keep
#include "mlir/Dialect/Math/IR/Math.h"         // IWYU pragma: keep
#include "mlir/Dialect/Tensor/IR/Tensor.h"     // IWYU pragma: keep
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"                        // IWYU pragma: keep
#include "mlir/Interfaces/DestinationStyleOpInterface.h" // IWYU pragma: keep
#include "mlir/Interfaces/TilingInterface.h"             // IWYU pragma: keep
#include "mlir/Pass/Pass.h"                              // IWYU pragma: keep
#include "mlir/Support/LLVM.h"

#include <cmath>  // IWYU pragma: keep
#include <limits> // IWYU pragma: keep

// Generated attribute and op class declarations.
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/LLKAttributes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/LLKOps.h.inc"

using namespace mlir;

#include "LLK/Conversion/LLKKernel/LLKKernelLowering.h"
#include "LLK/Dialect/Micro/MicroDialect.h" // IWYU pragma: keep
#include "LLK/Dialect/Micro/MicroEnums.h"   // IWYU pragma: keep
#include "mlir/IR/BuiltinTypes.h"           // IWYU pragma: keep
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

namespace {
TypeAttr accumulatorType(Operation *root) {
  if (auto attr = root->getAttrOfType<TypeAttr>("accumulator_type"))
    return attr;
  return TypeAttr::get(
      cast<ShapedType>(root->getResult(0).getType()).getElementType());
}
class LinalgMatmulLowering final : public llk::LLKKernelLowering {
public:
  StringRef getOpName() const override { return "linalg.matmul"; }
  LogicalResult inferContract(Operation *root,
                              llk::KernelContract &c) const override {
    c.workload = "matmul";
    return llk::inferContractionContract(
        root, root->getOperand(0), root->getOperand(1), root->getResult(0),
        accumulatorType(root), c);
  }
  FailureOr<llk::ContractionDimensions>
  inferDimensions(Operation *root) const override {
    return llk::inferContractionDimensions(
        root, root->getOperand(0), root->getOperand(1), root->getResult(0));
  }
  bool hasMicroLowering() const override { return true; }
  LogicalResult describeMicroKernel(Operation *root,
                                    const llk::KernelContract &c,
                                    llk::MicroKernelFragment &f,
                                    bool requireStatic = true) const override {
    auto validated = c;
    if (requireStatic) {
      if (failed(llk::validateMicroContraction(
              root, root->getOperand(0), root->getOperand(1),
              root->getResult(0), accumulatorType(root), validated)))
        return failure();
    } else {
      llk::ContractionDimensions dimensions;
      if (failed(llk::validateMicroContractionShape(
              root, root->getOperand(0), root->getOperand(1),
              root->getResult(0), accumulatorType(root), dimensions,
              validated)))
        return failure();
    }
    f.armCount = 1;
    f.hasOutputInit = true;
    f.rhsOperandTypes = {root->getOperand(1).getType()};
    f.buildEpilogue = [](OpBuilder &, Location, ArrayRef<Value> acc) {
      return acc[0];
    };
    return success();
  }
};
} // namespace
namespace mlir::llk {
void registerLinalgMatmulLowering(LLKKernelRegistry &r) {
  r.add<LinalgMatmulLowering>();
}
} // namespace mlir::llk
