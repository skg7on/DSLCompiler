//===- ForallToOpenMP.cpp - Lower scf.forall to OpenMP --------------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Lowers scf.forall to OpenMP via the chain:
//   scf.forall -> scf.parallel -> omp.parallel + omp.wsloop
//
// Uses MLIR's built-in conversion chain. This is a prototype alternative
// to the ForallToLLRT (ThreadPool) path.
//
//===----------------------------------------------------------------------===//

#include "LLK/Transforms/ForallToOpenMP.h"

#include "mlir/Conversion/SCFToOpenMP/SCFToOpenMP.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Config/llvm-config.h"

namespace mlir::llk {
#define GEN_PASS_DEF_FORALLTOOPENMP
#include "LLK/Transforms/Passes.h.inc"
} // namespace mlir::llk

using namespace mlir;

namespace {

struct ForallToOpenMPPass
    : public mlir::llk::impl::ForallToOpenMPBase<ForallToOpenMPPass> {
  void runOnOperation() override {
    auto module = getOperation();

    // Step 1: Convert scf.forall -> scf.parallel.
    // This step normalizes the parallel loop structure before
    // the OpenMP conversion.
    {
      mlir::OpPassManager pm(mlir::ModuleOp::getOperationName());
#if LLVM_VERSION_MAJOR >= 21
      pm.addPass(mlir::createForallToParallelLoopPass());
#else
      // On LLVM 20, scf.forall -> scf.parallel uses a different path.
      // Use a greedy pattern rewriter as fallback.
      pm.addPass(mlir::createCanonicalizerPass());
#endif
      if (mlir::failed(runPipeline(pm, module)))
        return signalPassFailure();
    }

    // Step 2: Convert scf.parallel -> OpenMP (omp.parallel + omp.wsloop).
    {
      mlir::OpPassManager pm(mlir::ModuleOp::getOperationName());
      pm.addPass(mlir::createConvertSCFToOpenMPPass());
      if (mlir::failed(runPipeline(pm, module)))
        return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::llk::createForallToOpenMPPass() {
  return std::make_unique<ForallToOpenMPPass>();
}
