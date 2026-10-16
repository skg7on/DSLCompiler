//===- FrameworkTestSupport.h - Source test helpers -------------*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLK_TEST_TUNING_FRAMEWORKTESTSUPPORT_H
#define LLK_TEST_TUNING_FRAMEWORKTESTSUPPORT_H

#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include "gtest/gtest.h"
#include <memory>
#include <string>

class FrameworkTest : public ::testing::Test {
protected:
  mlir::DialectRegistry dialects;
  std::unique_ptr<mlir::MLIRContext> context;
  void SetUp() override {
    dialects.insert<mlir::micro::MicroDialect, mlir::llk::LLKDialect,
                    mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                    mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
    context = std::make_unique<mlir::MLIRContext>(dialects);
    context->loadAllAvailableDialects();
  }
  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef text) {
    return mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
  }
  std::string print(mlir::Operation *op) {
    std::string text;
    llvm::raw_string_ostream out(text);
    op->print(out);
    return text;
  }
};
#endif
