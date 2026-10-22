//===- LegacyValidation.cpp - legacy contracts ----------------------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LegacyValidation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include <limits>

using namespace mlir;
namespace {
LogicalResult invalid(Operation *op, StringRef message) {
  return op->emitError("triton_invalid_schema: ") << message;
}
LogicalResult unsupported(Operation *op) {
  return op->emitError("triton_unsupported_op: Triton op '")
         << op->getName() << "' is unsupported on CPU. "
         << "See the CPU compatibility guide for supported operations.";
}
bool positiveShape(ArrayRef<int64_t> shape) {
  int64_t count = 1;
  for (int64_t size : shape) {
    if (size <= 0 || size > std::numeric_limits<int64_t>::max() / count)
      return false;
    count *= size;
  }
  return true;
}
bool ptrType(Type type) {
  auto opaque = dyn_cast<OpaqueType>(type);
  return opaque && opaque.getDialectNamespace() == "tt" &&
         opaque.getTypeData() == "ptr";
}
LogicalResult validatePointer(Operation *op) {
  if (op->getNumResults() != 1 || !ptrType(op->getResult(0).getType()) ||
      op->getNumOperands() == 0)
    return invalid(op,
                   "block pointer requires one base and one pointer result");
  auto base = dyn_cast<MemRefType>(op->getOperand(0).getType());
  if (!base || base.getRank() < 1 || base.getRank() > 2 ||
      !base.hasStaticShape() || !positiveShape(base.getShape()) ||
      (base.getMemorySpace() &&
       (!isa<IntegerAttr>(base.getMemorySpace()) ||
        cast<IntegerAttr>(base.getMemorySpace()).getInt() != 0)) ||
      (!isa<FloatType>(base.getElementType()) &&
       !base.getElementType().isSignlessInteger()))
    return invalid(op, "expected a positive static rank-one/two host memref");
  unsigned rank = base.getRank();
  if (op->getNumOperands() != rank + 1)
    return invalid(op, "expected exactly one index offset per dimension");
  auto shape = op->getAttrOfType<DenseI64ArrayAttr>("shape");
  auto block = op->getAttrOfType<DenseI64ArrayAttr>("block_shape");
  auto strides = op->getAttrOfType<DenseI64ArrayAttr>("strides");
  auto order = op->getAttrOfType<DenseI64ArrayAttr>("order");
  if (!shape || !block || !strides || !order || shape.size() != rank ||
      block.size() != rank || strides.size() != rank || order.size() != rank ||
      shape.asArrayRef() != base.getShape() ||
      !positiveShape(block.asArrayRef()))
    return invalid(
        op, "shape, block_shape, strides and order must match base rank/shape");
  auto layout = dyn_cast<StridedLayoutAttr>(base.getLayout());
  if (!base.getLayout().isIdentity() && (!layout || layout.getOffset() != 0))
    return invalid(op, "expected a zero-offset compact base layout");
  int64_t expected = 1;
  llvm::SmallDenseSet<int64_t> seen;
  for (int i = rank - 1; i >= 0; --i) {
    if ((layout && layout.getStrides()[i] != expected) ||
        strides[i] != expected)
      return invalid(
          op, "declared physical strides must match compact base layout");
    expected *= shape[i];
    if (order[i] < 0 || order[i] >= rank || !seen.insert(order[i]).second)
      return invalid(op, "order must be a permutation of dimensions");
    Value offset = op->getOperand(i + 1);
    if (!offset.getType().isIndex())
      return invalid(op, "offsets must have index type");
    APInt constant;
    if (matchPattern(offset, m_ConstantInt(&constant)) &&
        (constant.isNegative() || constant.getSExtValue() >= shape[i]))
      return invalid(op, "offset must be between zero and the base dimension");
  }
  for (OpOperand &use : op->getResult(0).getUses()) {
    StringRef name = use.getOwner()->getName().getStringRef();
    if ((name != "tt.load" && name != "tt.store") ||
        use.getOperandNumber() != 0)
      return invalid(
          op, "block pointers may only feed load/store pointer operands");
  }
  return success();
}
LogicalResult validateAccess(Operation *op, bool load) {
  unsigned required = load ? 1 : 2;
  if (op->getNumOperands() < required || op->getNumOperands() > required + 1 ||
      op->getNumResults() != (load ? 1u : 0u))
    return invalid(op, "incorrect load/store operand or result arity");
  Operation *ptr = op->getOperand(0).getDefiningOp();
  if (!ptr || ptr->getName().getStringRef() != "tt.make_block_ptr")
    return invalid(op, "load/store requires a validated block pointer");
  if (failed(validatePointer(ptr)))
    return failure();
  auto base = cast<MemRefType>(ptr->getOperand(0).getType());
  auto block = ptr->getAttrOfType<DenseI64ArrayAttr>("block_shape");
  auto expected = VectorType::get(block.asArrayRef(), base.getElementType());
  Type actual = load ? op->getResult(0).getType() : op->getOperand(1).getType();
  if (actual != expected)
    return invalid(
        op, "load/store value must match block vector shape and element type");
  if (op->getNumOperands() == required + 1 &&
      op->getOperand(required).getType() !=
          VectorType::get(block.asArrayRef(),
                          IntegerType::get(op->getContext(), 1)))
    return invalid(op, "mask must be an i1 vector of the exact block shape");
  return success();
}
LogicalResult validateDot(Operation *op) {
  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return invalid(op, "dot requires three operands and one result");
  SmallVector<RankedTensorType> types;
  for (Type type : op->getOperandTypes()) {
    auto tensor = dyn_cast<RankedTensorType>(type);
    if (!tensor || tensor.getRank() != 2 || !positiveShape(tensor.getShape()))
      return invalid(op, "dot requires positive static rank-two tensors");
    types.push_back(tensor);
  }
  auto a = types[0], b = types[1], c = types[2];
  Type input = a.getElementType();
  if (a.getDimSize(1) != b.getDimSize(0) ||
      a.getDimSize(0) != c.getDimSize(0) ||
      b.getDimSize(1) != c.getDimSize(1) || b.getElementType() != input ||
      !(input.isF32() || input.isBF16()) || !c.getElementType().isF32() ||
      op->getResult(0).getType() != c)
    return invalid(op, "incompatible dot shapes, storage or accumulator type");
  return success();
}
} // namespace

LogicalResult mlir::llk::validateLegacyTriton(ModuleOp module,
                                              bool rejectResidual) {
  auto result = module.walk([&](Operation *op) -> WalkResult {
    // Old atomic declarations/calls are also prohibited: there is no runtime.
    if (auto call = dyn_cast<func::CallOp>(op)) {
      if (call.getCallee().starts_with("llrt.atomic_")) {
        (void)unsupported(op);
        return WalkResult::interrupt();
      }
    }
    if (auto fn = dyn_cast<func::FuncOp>(op)) {
      if (fn.getSymName().starts_with("llrt.atomic_")) {
        (void)unsupported(op);
        return WalkResult::interrupt();
      }
    }
    StringRef name = op->getName().getStringRef();
    if (!name.starts_with("tt."))
      return WalkResult::advance();
    if (rejectResidual) {
      (void)unsupported(op);
      return WalkResult::interrupt();
    }
    if (op->getNumRegions() || op->getNumSuccessors()) {
      (void)invalid(op,
                    "legacy operations cannot contain regions or successors");
      return WalkResult::interrupt();
    }
    for (NamedAttribute attr : op->getAttrs()) {
      StringRef key = attr.getName().getValue();
      bool allowed = name == "tt.make_block_ptr"
                         ? (key == "shape" || key == "block_shape" ||
                            key == "strides" || key == "order")
                         : (name == "tt.get_program_id" && key == "axis");
      if (!allowed) {
        (void)invalid(op, "unknown or unsupported semantic attribute");
        return WalkResult::interrupt();
      }
    }
    LogicalResult valid = success();
    if (name == "tt.make_block_ptr")
      valid = validatePointer(op);
    else if (name == "tt.load" || name == "tt.store")
      valid = validateAccess(op, name == "tt.load");
    else if (name == "tt.dot")
      valid = validateDot(op);
    else if (name == "tt.get_program_id") {
      auto axis = op->getAttrOfType<IntegerAttr>("axis");
      auto fn = op->getParentOfType<func::FuncOp>();
      if (!axis || axis.getValue().getBitWidth() > 64 || axis.getInt() < 0 ||
          axis.getInt() > 2 || op->getNumOperands() != 0 ||
          op->getNumResults() != 1 || !op->getResult(0).getType().isIndex() ||
          !fn || !llvm::hasSingleElement(fn.getBody()) ||
          op->getBlock() != &fn.getBody().front())
        valid = invalid(op, "program id requires axis 0..2 and one index "
                            "result in a single function block");
      else if (fn.getNumResults())
        valid = fn.emitError(
            "triton_grid_result_unsupported: grid functions must return void");
    } else
      valid = unsupported(op);
    return failed(valid) ? WalkResult::interrupt() : WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}
