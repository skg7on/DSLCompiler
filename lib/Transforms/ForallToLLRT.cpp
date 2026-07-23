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
// NOTE: The worker function is outlined but the function pointer wiring to
// the runtime `llrtParallelFor` call is done by the JIT driver or a
// subsequent lowering pass. This pass emits the call with tile/grain
// arguments; the driver resolves the worker symbol at link time.
//
//===----------------------------------------------------------------------===//

#include "LLK/Transforms/ForallToLLRT.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseSet.h"

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
      auto runtimeFunc = func::FuncOp::create(
          modBuilder, module.getLoc(), "llrtParallelFor", runtimeFuncType);
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
    Block *body = forall.getBody();
    int64_t numDims = forall.getRank();

    // Fix for finding #15: use DenseSet for O(1) dedup instead of
    // llvm::is_contained O(n²) linear scan.
    llvm::SmallDenseSet<Value> seen;
    SmallVector<Value> capturedValues;

    auto captureExternal = [&](Operation &op) {
      for (auto operand : op.getOperands()) {
        if (operand.getParentBlock() != body && seen.insert(operand).second) {
          capturedValues.push_back(operand);
        }
      }
    };

    for (auto &op : body->without_terminator()) {
      captureExternal(op);
    }
    // Also capture external operands from the InParallelOp region
    auto term = cast<scf::InParallelOp>(body->getTerminator());
    for (auto &regionOp : term.getRegion().front().without_terminator()) {
      captureExternal(regionOp);
    }

    // Build worker function type: (tid, worker_id, captured..., shared_outs...)
    auto indexType = builder.getIndexType();
    auto i32Type = builder.getIntegerType(32);

    SmallVector<Type> workerArgTypes;
    workerArgTypes.push_back(indexType); // tid
    workerArgTypes.push_back(i32Type);   // worker_id
    for (auto val : capturedValues) {
      workerArgTypes.push_back(val.getType());
    }
    // Include shared_out types for pre-bufferization tensor IR support.
    // Post-bufferization, foralls have no outputs so this is a no-op.
    for (auto output : forall.getOutputs()) {
      workerArgTypes.push_back(output.getType());
    }

    auto funcType = FunctionType::get(builder.getContext(), workerArgTypes, {});

    // Create worker function
    std::string funcName = "llk_worker_" + std::to_string(workerId);
    auto workerFunc = func::FuncOp::create(builder, loc, funcName, funcType);
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
    // Fix for finding #9: if the forall body has shared_out block arguments
    // (pre-bufferization tensor IR), they must also be included in the
    // worker function signature. Post-bufferization, foralls have no
    // shared_outs, so this loop is a no-op.
    for (unsigned i = 0; i < forall.getOutputs().size(); i++) {
      unsigned workerArgIdx = 2 + capturedValues.size() + i;
      assert(workerArgIdx < workerBody->getNumArguments() &&
             "shared_out must be in worker function signature");
      mapper.map(body->getArgument(numDims + i),
                 workerBody->getArgument(workerArgIdx));
    }

    // Clone body ops (except InParallelOp terminator)
    for (auto &op : body->without_terminator()) {
      builder.clone(op, mapper);
    }
    // Also clone ops from the InParallelOp region
    for (auto &regionOp : term.getRegion().front().without_terminator()) {
      builder.clone(regionOp, mapper);
    }
    func::ReturnOp::create(builder, loc);

    // Put worker function before the parent function
    auto parentFunc = forall->getParentOfType<func::FuncOp>();
    if (parentFunc) {
      workerFunc->moveBefore(parentFunc);
    }

    // Fix for finding #2: compute totalTiles from the forall's actual upper
    // bounds instead of hardcoding to 100.
    builder.setInsertionPoint(forall);
    Value totalTiles;
    for (auto ub : forall.getMixedUpperBound()) {
      Value ubVal = getValueOrCreateConstantIndexOp(builder, loc, ub);
      if (!totalTiles) {
        totalTiles = ubVal;
      } else {
        totalTiles = arith::MulIOp::create(builder, loc, totalTiles, ubVal);
      }
    }
    if (!totalTiles) {
      totalTiles = arith::ConstantIndexOp::create(builder, loc, 0);
    }

    Value grainSize = arith::ConstantIndexOp::create(builder, loc, 1);

    SmallVector<Value> callOperands;
    callOperands.push_back(totalTiles);
    callOperands.push_back(grainSize);

    // NOTE: Fix for finding #3 — the worker function pointer and context
    // are NOT wired through this call. The JIT driver (or a subsequent
    // lowering pass) is responsible for resolving the worker symbol
    // (llk_worker_N) and providing it to the runtime bridge.
    func::CallOp::create(builder, loc, "llrtParallelFor", TypeRange{},
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
