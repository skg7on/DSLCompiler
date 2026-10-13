#include "LLK/Conversion/LLKKernel/LLKKernelLowering.h"
#include "LLK/Dialect/Micro/MicroHelpers.h"
#include "LLK/Transforms/Common/ScheduleLoader.h"
#include "mlir/IR/BuiltinTypes.h"
namespace mlir::llk {
LogicalResult checkMicroTensor(Operation *root, Value value, const char *what) {
  auto shaped = dyn_cast<ShapedType>(value.getType());
  if (!shaped || !shaped.hasRank() || shaped.getRank() != 2)
    return root->emitError() << what << " must be a rank-2 tensor";
  if (!shaped.hasStaticShape())
    return root->emitError()
           << what
           << " must have a static shape; dynamic shapes are not supported "
              "by the Micro-IR export";
  if (!micro::dtypeOfElementType(shaped.getElementType()))
    return root->emitError()
           << what << " element type is not a Micro tile dtype "
           << "(expected f32, f16, bf16, i32, or i8)";
  return success();
}

LogicalResult inferContractionContract(Operation *root, Value lhs, Value rhs,
                                       Value out,
                                       Attribute accumulatorAttribute,
                                       KernelContract &info) {
  auto l = dyn_cast<RankedTensorType>(lhs.getType());
  auto r = dyn_cast<RankedTensorType>(rhs.getType());
  auto o = dyn_cast<RankedTensorType>(out.getType());
  if (!l || !r || !o || l.getRank() != 2 || r.getRank() != 2 ||
      o.getRank() != 2)
    return root->emitError("contraction requires rank-2 tensors");
  info.M = l.getDimSize(0);
  info.K = l.getDimSize(1);
  info.N = r.getDimSize(1);
  info.inputElemType = l.getElementType();
  info.outputElemType = o.getElementType();
  auto acc = dyn_cast_or_null<TypeAttr>(accumulatorAttribute);
  info.accumulatorElemType = acc ? acc.getValue() : Type();
  info.mBucket = l.hasStaticShape() ? classifyM(info.M) : -1;
  return success();
}
LogicalResult validateMicroContraction(Operation *root, Value lhs, Value rhs,
                                       Value out,
                                       Attribute accumulatorAttribute,
                                       KernelContract &info) {
  if (failed(checkMicroTensor(root, lhs, "left-hand side")) ||
      failed(checkMicroTensor(root, rhs, "right-hand side")) ||
      failed(checkMicroTensor(root, out, "result")))
    return failure();
  auto lhsShaped = cast<ShapedType>(lhs.getType());
  auto rhsShaped = cast<ShapedType>(rhs.getType());
  auto outShaped = cast<ShapedType>(out.getType());
  info.M = lhsShaped.getDimSize(0);
  info.K = lhsShaped.getDimSize(1);
  info.N = rhsShaped.getDimSize(1);

  if (rhsShaped.getDimSize(0) != info.K)
    return root->emitError()
           << "contraction dimension mismatch: the left-hand side contracts "
              "over "
           << info.K << " but the right-hand side has "
           << rhsShaped.getDimSize(0) << " rows";

  if (rhsShaped.getElementType() != lhsShaped.getElementType())
    return root->emitError()
           << "the concrete Micro-IR exporter requires matching input and "
              "weight element types";

  if (outShaped.getDimSize(0) != info.M || outShaped.getDimSize(1) != info.N)
    return root->emitError()
           << "result shape does not match the contraction: expected M = "
           << info.M << " and N = " << info.N;

  TypeAttr accumulatorTypeAttr;
  if (accumulatorAttribute)
    accumulatorTypeAttr = dyn_cast<TypeAttr>(accumulatorAttribute);
  if (!accumulatorTypeAttr ||
      !micro::dtypeOfElementType(accumulatorTypeAttr.getValue()))
    return root->emitError()
           << "accumulator_type must be a type attribute naming a Micro tile "
              "dtype";

  info.inputElemType = lhsShaped.getElementType();
  info.accumulatorElemType = accumulatorTypeAttr.getValue();
  info.outputElemType = outShaped.getElementType();
  info.mBucket = llk::classifyM(info.M);
  return success();
}

} // namespace mlir::llk
