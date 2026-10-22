//===- AtomicToLLRT.cpp - checked legacy Triton contracts ----------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Legacy runtime operations remain unsupported until execution and ownership
// contracts are implemented. Keep the pass spelling for explicit diagnostics.
#include "LLK/Conversion/TritonToLLK/AtomicToLLRT.h"
#include "LegacyValidation.h"
#include "mlir/Pass/Pass.h"
using namespace mlir;
namespace {
struct AtomicToLLRTPass
    : PassWrapper<AtomicToLLRTPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AtomicToLLRTPass)
  StringRef getArgument() const override { return "triton-atomic-to-llrt"; }
  StringRef getDescription() const override {
    return "Reject unsupported legacy runtime operations";
  }
  void runOnOperation() override {
    if (failed(mlir::llk::validateLegacyTriton(getOperation())))
      signalPassFailure();
  }
};
} // namespace
std::unique_ptr<Pass> mlir::llk::createAtomicToLLRTPass() {
  return std::make_unique<AtomicToLLRTPass>();
}
