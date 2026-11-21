//===- triton_swiglu_staged.cpp - Triton CPU boundary rejection -----------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The staged Triton dialect-to-LLK pipeline is not an executable entry point.
// Its supported CPU boundary is enforced by the CPU verifier, which must reject
// GPU-only constructs with the documented diagnostic instead of lowering them.
// This replaces the former disabled SwiGLU staged-execution skeleton: the
// capability is an explicit rejection, not native execution.

#include "LLK/Conversion/TritonToLLK/TritonCPUVerifier.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallVector.h"

#include <gtest/gtest.h>
#include <string>

TEST(TritonStagedExecution, RejectsUnsupportedTritonInlineAssemblyOnCPU) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect>();
  context.allowUnregisteredDialects();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
func.func @unsupported_inline_asm(%ptr: !tt.ptr) {
  "tt.inline_asm"(%ptr) : (!tt.ptr) -> ()
  return
}
)mlir",
                                                        &context);
  ASSERT_TRUE(module);

  llvm::SmallVector<std::string> diagnostics;
  mlir::ScopedDiagnosticHandler handler(
      &context, [&](mlir::Diagnostic &diagnostic) {
        diagnostics.push_back(diagnostic.str());
      });

  mlir::PassManager passManager(&context);
  passManager.addPass(mlir::llk::createTritonCPUVerifierPass());
  ASSERT_TRUE(mlir::failed(passManager.run(*module)));

  ASSERT_FALSE(diagnostics.empty());
  const std::string &message = diagnostics.front();
  EXPECT_NE(message.find("triton_unsupported_op"), std::string::npos)
      << message;
  EXPECT_NE(message.find("tt.inline_asm"), std::string::npos) << message;
}
