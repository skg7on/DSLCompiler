//===- TritonCPUVerifier.cpp - legacy contracts ---------------------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Schema validation before rewriting; reject-residual is the final boundary.
#include "LLK/Conversion/TritonToLLK/TritonCPUVerifier.h"
#include "LegacyValidation.h"
#include "mlir/Pass/Pass.h"
using namespace mlir;
namespace {
struct TritonCPUVerifierPass
    : PassWrapper<TritonCPUVerifierPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TritonCPUVerifierPass)
  TritonCPUVerifierPass() = default;
  TritonCPUVerifierPass(const TritonCPUVerifierPass &other)
      : PassWrapper(other) {}
  Option<bool> rejectResidual{
      *this, "reject-residual",
      llvm::cl::desc(
          "Reject every remaining experimental tt operation before execution"),
      llvm::cl::init(false)};
  StringRef getArgument() const override { return "triton-cpu-verify"; }
  StringRef getDescription() const override {
    return "Validate experimental Triton schemas and CPU support";
  }
  void runOnOperation() override {
    if (failed(mlir::llk::validateLegacyTriton(getOperation(), rejectResidual)))
      signalPassFailure();
  }
};
} // namespace
std::unique_ptr<Pass> mlir::llk::createTritonCPUVerifierPass() {
  return std::make_unique<TritonCPUVerifierPass>();
}
