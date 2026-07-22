//===- ForallToLLRT.cpp - Lower scf.forall to runtime calls ---------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers remaining scf.forall ops (post-bufferization) to runtime
// parallel_for calls. Outlines the forall body into a worker function.
//
//===----------------------------------------------------------------------===//

#include "LLK/Transforms/ForallToLLRT.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {

struct ForallToLLRTPass
    : public PassWrapper<ForallToLLRTPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ForallToLLRTPass)

  StringRef getArgument() const override { return "forall-to-llrt"; }
  StringRef getDescription() const override {
    return "Lower scf.forall to llrt.parallel_for runtime calls";
  }

  void runOnOperation() override {
    auto module = getOperation();
    int workerCounter = 0;

    // Declare the runtime function if not already present
    auto indexType = mlir::IndexType::get(&getContext());
    if (!module.lookupSymbol<func::FuncOp>("llrtParallelFor")) {
      OpBuilder modBuilder(module.getBodyRegion());
      auto runtimeFuncType =
          FunctionType::get(&getContext(), {indexType, indexType}, {});
      auto runtimeFunc = modBuilder.create<func::FuncOp>(
          module.getLoc(), "llrtParallelFor", runtimeFuncType);
      runtimeFunc.setPrivate();
    }

    SmallVector<scf::ForallOp> toProcess;
    module.walk([&](scf::ForallOp forall) { toProcess.push_back(forall); });

    for (auto forall : toProcess) {
      lowerForall(forall, module, workerCounter++);
    }
  }

private:
  void lowerForall(scf::ForallOp forall, ModuleOp module, int workerId) {
    OpBuilder builder(forall);
    Location loc = forall.getLoc();

    // Collect captured values: operands used in the body that are defined
    // outside the forall
    SmallVector<Value> capturedValues;
    Block *body = forall.getBody();

    // All values used in the body that aren't block arguments are captured
    for (auto &op : body->without_terminator()) {
      for (auto operand : op.getOperands()) {
        if (operand.getParentBlock() != body) {
          // External value — capture it
          if (llvm::is_contained(capturedValues, operand))
            continue;
          capturedValues.push_back(operand);
        }
      }
    }

    // Build worker function type
    auto indexType = builder.getIndexType();
    auto i32Type = builder.getIntegerType(32);

    SmallVector<Type> workerArgTypes;
    workerArgTypes.push_back(indexType); // tid
    workerArgTypes.push_back(i32Type);   // worker_id
    for (auto val : capturedValues) {
      workerArgTypes.push_back(val.getType());
    }

    auto funcType = FunctionType::get(builder.getContext(), workerArgTypes, {});

    // Create worker function
    std::string funcName = "llk_worker_" + std::to_string(workerId);
    auto workerFunc = builder.create<func::FuncOp>(loc, funcName, funcType);
    workerFunc.setPrivate();

    Block *workerBody = workerFunc.addEntryBlock();
    builder.setInsertionPointToStart(workerBody);

    // Map forall body to worker function
    IRMapping mapper;
    // tid → worker arg 0
    mapper.map(body->getArgument(0), workerBody->getArgument(0));
    // captured values → worker args 2+
    for (size_t i = 0; i < capturedValues.size(); i++) {
      mapper.map(capturedValues[i], workerBody->getArgument(2 + i));
    }

    // Clone body ops (except InParallelOp terminator)
    for (auto &op : body->without_terminator()) {
      builder.clone(op, mapper);
    }
    builder.create<func::ReturnOp>(loc);

    // Put worker function before the parent function
    auto parentFunc = forall->getParentOfType<func::FuncOp>();
    if (parentFunc) {
      workerFunc->moveBefore(parentFunc);
    }

    // Replace scf.forall with a call to llrtParallelFor
    builder.setInsertionPoint(forall);

    // Compute total tiles
    Value totalTiles =
        builder.create<arith::ConstantIndexOp>(loc, 100); // placeholder
    Value grainSize = builder.create<arith::ConstantIndexOp>(loc, 1);

    // Create the call
    SmallVector<Value> callOperands;
    callOperands.push_back(totalTiles);
    callOperands.push_back(grainSize);

    // We also need to pass a function pointer and context
    // For now, use a simple placeholder call
    builder.create<func::CallOp>(loc, "llrtParallelFor", TypeRange{},
                                 callOperands);

    forall.erase();
  }
};

} // namespace

namespace mlir {
namespace llk {

std::unique_ptr<Pass> createForallToLLRTPass() {
  return std::make_unique<ForallToLLRTPass>();
}

} // namespace llk
} // namespace mlir
