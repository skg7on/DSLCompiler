//===- LLKToLinalg.cpp - Lower LLK ops to Linalg + Arith + Math -----------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Dispatches registered kernel lowerings into Linalg + Arith + Math.

#include "LLK/Conversion/LLKToLinalg.h"
#include "LLK/Conversion/LLKKernel/LLKKernelLowering.h"
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

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// LLKToLinalgPass
//===----------------------------------------------------------------------===//

struct LLKToLinalgPass
    : public PassWrapper<LLKToLinalgPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLKToLinalgPass)

  StringRef getArgument() const override { return "llk-to-linalg"; }
  StringRef getDescription() const override {
    return "Lower LLK ops to Linalg + Arith + Math";
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
    registry.insert<mlir::math::MathDialect>();
    registry.insert<mlir::linalg::LinalgDialect>();
    registry.insert<mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override;
};

void LLKToLinalgPass::runOnOperation() {
  llk::LLKKernelRegistry registry;
  llk::populateLLKKernelLowerings(registry);
  OpBuilder builder(&getContext());
  SmallVector<std::pair<Operation *, const llk::LLKKernelLowering *>> roots;
  // Keep the original per-kind order, collecting before any mutation.
  for (const auto *entry : registry.entries()) {
    if (!entry->hasLinalgLowering())
      continue;
    getOperation().walk([&](Operation *op) {
      if (registry.lookup(op->getName().getStringRef()) == entry)
        roots.emplace_back(op, entry);
    });
  }
  for (auto [op, entry] : roots) {
    builder.setInsertionPoint(op);
    if (failed(entry->lowerToLinalg(op, builder))) {
      signalPassFailure();
      return;
    }
  }
}

static mlir::PassRegistration<LLKToLinalgPass> llkToLinalgPassReg;

} // namespace

namespace mlir {
namespace llk {
std::unique_ptr<Pass> createLLKToLinalgPass() {
  return std::make_unique<LLKToLinalgPass>();
}
} // namespace llk
} // namespace mlir
