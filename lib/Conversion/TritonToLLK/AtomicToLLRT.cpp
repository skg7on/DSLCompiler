//===- AtomicToLLRT.cpp - atomics → llrt runtime calls --------------------===//
//
// Lowers Triton atomic ops (tt.atomic_add, tt.atomic_max, tt.atomic_min,
// tt.atomic_and, tt.atomic_or, tt.atomic_xor, tt.atomic_xchg, tt.atomic_cas)
// to llrt runtime calls wrapping std::atomic or CPU CAS instructions. Each
// callee is declared as a private func.func on demand and called with the
// original operands, preserving the old-value result.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/TritonToLLK/AtomicToLLRT.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {

struct AtomicLowering : public RewritePattern {
  AtomicLowering(MLIRContext *ctx)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    StringRef opName = op->getName().getStringRef();
    if (!opName.starts_with("tt.atomic_"))
      return failure();

    // "tt.atomic_add" → "add"
    StringRef kind = opName.drop_front(strlen("tt.atomic_"));
    if (kind.empty() || op->getNumOperands() < 2)
      return failure();

    Location loc = op->getLoc();
    // Scalar element type drives the runtime specialization suffix.
    Type valueType = op->getOperand(1).getType();

    // Build the callee name: llrt.atomic_<kind>_<typedesc>.
    std::string callee;
    {
      llvm::raw_string_ostream os(callee);
      os << "llrt.atomic_" << kind << "_";
      valueType.print(os);
    }

    // The runtime signature mirrors the atomic op: (ptr, value[, cmp]) -> old.
    SmallVector<Type> argTypes;
    for (Type t : op->getOperandTypes())
      argTypes.push_back(t);
    TypeRange resultTypes = op->getResultTypes();

    // Declare the runtime function once per callee.
    ModuleOp module = op->getParentOfType<ModuleOp>();
    if (module && !module.lookupSymbol<func::FuncOp>(callee)) {
      auto funcType =
          FunctionType::get(rewriter.getContext(), argTypes, resultTypes);
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(module.getBody());
      auto runtimeFunc = func::FuncOp::create(rewriter, loc, callee, funcType);
      runtimeFunc.setPrivate();
    }

    auto call = func::CallOp::create(rewriter, loc, callee, resultTypes,
                                     op->getOperands());
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct AtomicToLLRTPass
    : public PassWrapper<AtomicToLLRTPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AtomicToLLRTPass)

  StringRef getArgument() const override { return "triton-atomic-to-llrt"; }
  StringRef getDescription() const override {
    return "Lower Triton atomic ops to llrt runtime calls";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<AtomicLowering>(ctx);

    if (failed(
#if LLVM_VERSION_MAJOR >= 21
            applyPatternsGreedily(getOperation(), std::move(patterns))
#else
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))
#endif
                ))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::llk::createAtomicToLLRTPass() {
  return std::make_unique<AtomicToLLRTPass>();
}
