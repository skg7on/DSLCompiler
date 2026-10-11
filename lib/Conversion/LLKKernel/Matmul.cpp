#include "LLK/Conversion/LLKToLinalg.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/LLKEnums.h"
#include "LLK/Transforms/Common/MathApproximation.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include <cmath>
#include <limits>

// Generated attribute and op class declarations.
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/LLKAttributes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/LLKOps.h.inc"

using namespace mlir;

#include "LLK/Conversion/LLKKernel/LLKKernelLowering.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Dialect/Micro/MicroEnums.h"
#include "mlir/IR/BuiltinTypes.h"
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

namespace {
class MatmulLowering final : public llk::LLKKernelLowering {
public:
  StringRef getOpName() const override { return "llk.matmul"; }
  LogicalResult inferContract(Operation *root,
                              llk::KernelContract &c) const override {
    c.workload = "matmul";
    return llk::inferContractionContract(
        root, root->getOperand(0), root->getOperand(1), root->getResult(0),
        root->getAttr("accumulator_type"), c);
  }
  bool hasLinalgLowering() const override { return true; }
  LogicalResult lowerToLinalg(Operation *root,
                              OpBuilder &builder) const override {
    auto op = cast<llk::MatmulOp>(root);
    Value result = linalg::MatmulOp::create(builder, op.getLoc(),
                                            ValueRange{op.getA(), op.getB()},
                                            ValueRange{op.getInit()})
                       .getResult(0);
    op.replaceAllUsesWith(result);
    op.erase();
    return success();
  }
  bool hasMicroLowering() const override { return true; }
  LogicalResult
  describeMicroKernel(Operation *root, const llk::KernelContract &c,
                      llk::MicroKernelFragment &f) const override {
    auto validated = c;
    if (failed(llk::validateMicroContraction(
            root, root->getOperand(0), root->getOperand(1), root->getResult(0),
            root->getAttr("accumulator_type"), validated)))
      return failure();
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
void registerMatmulLowering(LLKKernelRegistry &r) { r.add<MatmulLowering>(); }
} // namespace mlir::llk
