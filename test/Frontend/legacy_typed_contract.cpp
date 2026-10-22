//===- legacy_typed_contract.cpp - legacy contracts -----------------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/TritonToLLK/AtomicToLLRT.h"
#include "LLK/Conversion/TritonToLLK/BlockPointerToVector.h"
#include "LLK/Conversion/TritonToLLK/GridToForall.h"
#include "LLK/Conversion/TritonToLLK/SharedMemToScratch.h"
#include "LLK/Conversion/TritonToLLK/TritonCPUVerifier.h"
#include "LLK/Conversion/TritonToLLK/TritonToStructured.h"
#if LLK_FRONTEND_ENABLE_EXECUTION
#include "LLK/Runtime/JitCache.h"
#endif
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Error.h"
#include <array>
#include <csignal>
#include <gtest/gtest.h>
#include <string>
#include <vector>

using namespace mlir;
namespace {
struct FrontendLegacy : testing::Test {
  MLIRContext context;
  FrontendLegacy() {
    context.allowUnregisteredDialects();
    context.loadDialect<arith::ArithDialect, func::FuncDialect,
                        memref::MemRefDialect, vector::VectorDialect,
                        LLVM::LLVMDialect>();
  }
  std::string print(ModuleOp module) {
    std::string text;
    llvm::raw_string_ostream os(text);
    module.print(os);
    return text;
  }
  void rejects(StringRef text, std::unique_ptr<Pass> pass, StringRef code) {
    auto module = parseSourceString<ModuleOp>(text, &context);
    ASSERT_TRUE(module);
    std::string before = print(*module), diagnostics;
    ScopedDiagnosticHandler handler(&context, [&](Diagnostic &d) {
      llvm::raw_string_ostream os(diagnostics);
      d.print(os);
      return success();
    });
    PassManager pm(&context);
    pm.addPass(std::move(pass));
    EXPECT_TRUE(failed(pm.run(*module)));
    EXPECT_NE(diagnostics.find(code.str()), std::string::npos) << diagnostics;
    EXPECT_EQ(print(*module), before) << "rejection must precede all mutation";
    EXPECT_TRUE(succeeded(verify(*module)));
  }
};

std::string blockProgram(StringRef replacement = "", StringRef original = "") {
  std::string text = R"mlir(module {
func.func @copy(%src: memref<128x64xf32>, %dst: memref<128x64xf32>) {
  %zero = arith.constant 0 : index
  %p = "tt.make_block_ptr"(%src, %zero, %zero) {shape = array<i64: 128, 64>, strides = array<i64: 64, 1>, block_shape = array<i64: 32, 64>, order = array<i64: 0, 1>} : (memref<128x64xf32>, index, index) -> !tt.ptr
  %v = "tt.load"(%p) : (!tt.ptr) -> vector<32x64xf32>
  %q = "tt.make_block_ptr"(%dst, %zero, %zero) {shape = array<i64: 128, 64>, strides = array<i64: 64, 1>, block_shape = array<i64: 32, 64>, order = array<i64: 0, 1>} : (memref<128x64xf32>, index, index) -> !tt.ptr
  "tt.store"(%q, %v) : (!tt.ptr, vector<32x64xf32>) -> ()
  return
}})mlir";
  if (!original.empty()) {
    size_t pos = text.find(original.str());
    assert(pos != std::string::npos);
    text.replace(pos, original.size(), replacement.str());
  }
  return text;
}

TEST_F(FrontendLegacy, PhysicalStrideIsNotMultiplied) {
  auto module = parseSourceString<ModuleOp>(blockProgram(), &context);
  ASSERT_TRUE(module);
  PassManager pm(&context);
  pm.addPass(mlir::llk::createBlockPointerToVectorPass());
  ASSERT_TRUE(succeeded(pm.run(*module)));
  std::string ir = print(*module);
  EXPECT_NE(ir.find("strided<[64, 1]"), std::string::npos) << ir;
  EXPECT_EQ(ir.find("strided<[4096, 1]"), std::string::npos);
}

struct BadSchema {
  const char *name;
  const char *from;
  const char *to;
};
void PrintTo(const BadSchema &schema, std::ostream *os) { *os << schema.name; }
struct FrontendLegacySchema : FrontendLegacy,
                              testing::WithParamInterface<BadSchema> {};
TEST_P(FrontendLegacySchema, RejectsWithoutMutation) {
  const auto &c = GetParam();
  rejects(blockProgram(c.to, c.from),
          mlir::llk::createBlockPointerToVectorPass(), "triton_invalid_schema");
}
INSTANTIATE_TEST_SUITE_P(
    FrontendLegacy, FrontendLegacySchema,
    testing::Values(BadSchema{"ShortShape", "shape = array<i64: 128, 64>",
                              "shape = array<i64: 128>"},
                    BadSchema{"ShortStride", "strides = array<i64: 64, 1>",
                              "strides = array<i64: 64>"},
                    BadSchema{"BadStride", "strides = array<i64: 64, 1>",
                              "strides = array<i64: 4096, 1>"},
                    BadSchema{"BadOrder", "order = array<i64: 0, 1>",
                              "order = array<i64: 1, 1>"},
                    BadSchema{"NegativeOffset", "constant 0 : index",
                              "constant -1 : index"},
                    BadSchema{"BeyondBaseOffset", "constant 0 : index",
                              "constant 129 : index"},
                    BadSchema{"ZeroBlock", "block_shape = array<i64: 32, 64>",
                              "block_shape = array<i64: 0, 64>"},
                    BadSchema{"BadShape", "shape = array<i64: 128, 64>",
                              "shape = array<i64: 129, 64>"}),
    [](const auto &i) { return i.param.name; });

TEST_F(FrontendLegacy, RejectsInvalidMaskAndLoadArity) {
  for (auto spec : {std::pair{"%p, %zero", "!tt.ptr, index"},
                    std::pair{"%p, %zero, %zero", "!tt.ptr, index, index"}}) {
    std::string replacement =
        "\"tt.load\"(" + std::string(spec.first) + ") : (" + spec.second + ")";
    rejects(blockProgram(replacement, "\"tt.load\"(%p) : (!tt.ptr)"),
            mlir::llk::createBlockPointerToVectorPass(),
            "triton_invalid_schema");
  }
}
TEST_F(FrontendLegacy, RejectsPointerArityTypesAndResultShape) {
  for (
      auto text :
      {R"(module {func.func @bad() { %p = "tt.make_block_ptr"() : () -> !tt.ptr return }})",
       R"(module {func.func @bad(%i: index) { %p = "tt.make_block_ptr"(%i) : (index) -> !tt.ptr return }})",
       R"(module {func.func @bad(%b: memref<8x8xf32>, %i: index) { %p = "tt.make_block_ptr"(%b,%i) : (memref<8x8xf32>,index) -> !tt.ptr return }})",
       R"(module {func.func @bad(%b: memref<8x8xf32>, %i: f32) { %p = "tt.make_block_ptr"(%b,%i,%i) {shape=array<i64:8,8>,block_shape=array<i64:4,4>,strides=array<i64:8,1>,order=array<i64:0,1>} : (memref<8x8xf32>,f32,f32) -> !tt.ptr return }})"})
    rejects(text, mlir::llk::createBlockPointerToVectorPass(),
            "triton_invalid_schema");
  auto wrongType = blockProgram();
  size_t at = 0;
  while ((at = wrongType.find("vector<32x64xf32>", at)) != std::string::npos) {
    wrongType.replace(at, std::string("vector<32x64xf32>").size(),
                      "vector<32x64xi32>");
    at += std::string("vector<32x64xi32>").size();
  }
  rejects(wrongType, mlir::llk::createBlockPointerToVectorPass(),
          "triton_invalid_schema");
}
TEST_F(FrontendLegacy, SignedElementTypesAreRejectedBeforeMutation) {
  for (const auto *type : {"si32", "ui32"}) {
    std::string text = blockProgram();
    size_t at = 0;
    while ((at = text.find("f32", at)) != std::string::npos) {
      text.replace(at, 3, type);
      at += std::string(type).size();
    }
    rejects(text, mlir::llk::createBlockPointerToVectorPass(),
            "triton_invalid_schema");
  }
}
TEST_F(FrontendLegacy, RejectsInvalidStoreAndMaskShapes) {
  rejects(blockProgram("\"tt.store\"(%q) : (!tt.ptr)",
                       "\"tt.store\"(%q, %v) : (!tt.ptr, vector<32x64xf32>)"),
          mlir::llk::createBlockPointerToVectorPass(), "triton_invalid_schema");
  auto text =
      blockProgram("%mask = arith.constant dense<true> : vector<32x63xi1>\n  "
                   "%v = \"tt.load\"(%p, %mask) : (!tt.ptr, vector<32x63xi1>)",
                   "%v = \"tt.load\"(%p) : (!tt.ptr)");
  rejects(text, mlir::llk::createBlockPointerToVectorPass(),
          "triton_invalid_schema");
}
TEST_F(FrontendLegacy, GridResultsCannotEscape) {
  rejects(R"(module {func.func @grid(%M: index, %BM: index) -> index {
    %p = "tt.get_program_id"() {axis=0:i32} : () -> index
    return %p : index
  }})",
          mlir::llk::createTritonGridToForallPass(),
          "triton_grid_result_unsupported");
}
TEST_F(FrontendLegacy, InvalidGridAxisAndMissingDimensions) {
  for (auto axis : {"-1", "3", "0"}) {
    std::string text =
        "module {func.func @grid() { %p = \"tt.get_program_id\"() {axis=" +
        std::string(axis) + ":i32} : () -> index return }}";
    rejects(text, mlir::llk::createTritonGridToForallPass(),
            "triton_invalid_schema");
  }
}
TEST_F(FrontendLegacy, RejectsDotBeforeOperandAccess) {
  rejects(R"(module {func.func @bad() {
    %x = "tt.dot"() : () -> f32
    return
  }})",
          mlir::llk::createTritonToStructuredPass(), "triton_invalid_schema");
}
TEST_F(FrontendLegacy, RejectsUnsupportedOperations) {
  for (auto name :
       {"tt.await", "tt.reduce", "tt.broadcast", "tt.trans", "tt.make_range",
        "tt.advance", "tt.splat", "tt.inline_asm"}) {
    std::string text = "module {func.func @bad() { \"" + std::string(name) +
                       "\"() : () -> () return }}";
    rejects(text, mlir::llk::createTritonCPUVerifierPass(),
            "triton_unsupported_op");
  }
}
TEST_F(FrontendLegacy, AtomicsHaveNoRuntimeImplementation) {
  rejects(R"(module {func.func @bad(%p: !llvm.ptr, %v: f32) {
    %x = "tt.atomic_add"(%p,%v) : (!llvm.ptr,f32) -> f32
    return
  }})",
          mlir::llk::createAtomicToLLRTPass(), "triton_unsupported_op");
}
TEST_F(FrontendLegacy, ScratchOwnershipIsUnsupported) {
  rejects(R"(module {func.func @bad() {
    %x = "tt.alloc"() : () -> memref<32x64xf32,3>
    return
  }})",
          mlir::llk::createSharedMemToScratchPass(), "triton_unsupported_op");
}

TEST_F(FrontendLegacy, ExecutableBoundaryRejectsResidualTriton) {
  auto module = parseSourceString<ModuleOp>(blockProgram(), &context);
  ASSERT_TRUE(module);
  auto pass = mlir::llk::createTritonCPUVerifierPass();
  ASSERT_TRUE(succeeded(pass->initializeOptions(
      "reject-residual=true", [](const llvm::Twine &) { return failure(); })));
  rejects(blockProgram(), std::move(pass), "triton_unsupported_op");
  PassManager pm(&context);
  pm.addPass(mlir::llk::createBlockPointerToVectorPass());
  auto finalCheck = mlir::llk::createTritonCPUVerifierPass();
  ASSERT_TRUE(succeeded(finalCheck->initializeOptions(
      "reject-residual=true", [](const llvm::Twine &) { return failure(); })));
  pm.addPass(std::move(finalCheck));
  EXPECT_TRUE(succeeded(pm.run(*module)));
}
TEST_F(FrontendLegacy, RuntimeAtomicCallsAreRejected) {
  rejects(R"(module {
    func.func private @llrt.atomic_add_f32(!llvm.ptr, f32) -> f32
    func.func @bad(%p: !llvm.ptr, %v: f32) {
      %x = func.call @llrt.atomic_add_f32(%p,%v) : (!llvm.ptr,f32) -> f32
      return
    }
  })",
          mlir::llk::createTritonCPUVerifierPass(), "triton_unsupported_op");
}
TEST_F(FrontendLegacy, WholeModulePreflightBeforeGridMutation) {
  rejects(R"(module {
    func.func @valid(%M: index, %BM: index) {
      %p = "tt.get_program_id"() {axis=0:i32} : () -> index
      return
    }
    func.func @bad() {
      %p = "tt.get_program_id"() {axis=0:i32} : () -> index
      return
    }
  })",
          mlir::llk::createTritonGridToForallPass(), "triton_invalid_schema");
}
#if LLK_FRONTEND_ENABLE_EXECUTION
TEST_F(FrontendLegacy, DynamicOffsetsAreCheckedBeforeMemoryAccess) {
  std::string text = blockProgram();
  text.replace(text.find("%dst: memref<128x64xf32>)"),
               std::string("%dst: memref<128x64xf32>)").size(),
               "%dst: memref<128x64xf32>, %row: index, %col: index)");
  size_t at = 0;
  while ((at = text.find("%zero, %zero", at)) != std::string::npos) {
    text.replace(at, std::string("%zero, %zero").size(), "%row, %col");
    at += std::string("%row, %col").size();
  }
  auto module = parseSourceString<ModuleOp>(text, &context);
  ASSERT_TRUE(module);
  PassManager pm(&context);
  pm.addPass(mlir::llk::createBlockPointerToVectorPass());
  ASSERT_TRUE(succeeded(pm.run(*module)));
  ::llk::JitCache cache;
  using Fn = void (*)(void *, void *, int64_t, int64_t, int64_t, int64_t,
                      int64_t, void *, void *, int64_t, int64_t, int64_t,
                      int64_t, int64_t, int64_t, int64_t);
  auto fn = cache.lookupTyped<Fn>("dynamic_legacy_copy", *module);
  ASSERT_TRUE(static_cast<bool>(fn)) << llvm::toString(fn.takeError());
  constexpr int guard = 16, size = 128 * 64;
  std::vector<float> input(size + 2 * guard, -12345),
      output(size + 2 * guard, -12345);
  for (int i = 0; i < size; ++i)
    input[guard + i] = float(i + 1);
  auto invoke = [&](int64_t row, int64_t col) {
    (*fn)(input.data() + guard, input.data() + guard, 0, 128, 64, 64, 1,
          output.data() + guard, output.data() + guard, 0, 128, 64, 64, 1, row,
          col);
  };
  invoke(126, 62);
  for (int i = 0; i < size + 2 * guard; ++i) {
    int row = (i - guard) / 64, col = (i - guard) % 64;
    bool copied = i >= guard && i < guard + size && row >= 126 && col >= 62;
    EXPECT_EQ(output[i], copied ? float(i - guard + 1) : -12345) << i;
  }
  EXPECT_EXIT(invoke(-1, 0), testing::KilledBySignal(SIGABRT), "");
  EXPECT_EXIT(invoke(128, 0), testing::KilledBySignal(SIGABRT), "");
}
struct CopyCase {
  const char *name;
  int row;
  int col;
  int maskRows;
  int maskCols;
};
void PrintTo(const CopyCase &c, std::ostream *os) { *os << c.name; }
struct FrontendLegacyCopy : FrontendLegacy,
                            testing::WithParamInterface<CopyCase> {};
TEST_P(FrontendLegacyCopy, NativeRowsAndCanaries) {
  const auto &c = GetParam();
  std::string text = blockProgram();
  auto replaceAll = [&](StringRef from, const std::string &to) {
    size_t at = 0;
    while ((at = text.find(from.str(), at)) != std::string::npos) {
      text.replace(at, from.size(), to);
      at += to.size();
    }
  };
  replaceAll("%zero = arith.constant 0 : index",
             "%row = arith.constant " + std::to_string(c.row) +
                 " : index\n %col = arith.constant " + std::to_string(c.col) +
                 " : index");
  replaceAll("%zero, %zero", "%row, %col");
  bool masked = c.maskRows >= 0;
  if (masked) {
    replaceAll("%v = \"tt.load\"(%p) : (!tt.ptr)",
               "%mr = arith.constant " + std::to_string(c.maskRows) +
                   " : index\n %mc = arith.constant " +
                   std::to_string(c.maskCols) +
                   " : index\n %mask = vector.create_mask %mr, %mc : "
                   "vector<32x64xi1>\n %v = \"tt.load\"(%p, %mask) : (!tt.ptr, "
                   "vector<32x64xi1>)");
    replaceAll("\"tt.store\"(%q, %v) : (!tt.ptr, vector<32x64xf32>)",
               "\"tt.store\"(%q, %v, %mask) : (!tt.ptr, vector<32x64xf32>, "
               "vector<32x64xi1>)");
  }
  auto module = parseSourceString<ModuleOp>(text, &context);
  ASSERT_TRUE(module) << text;
  PassManager pm(&context);
  pm.addPass(mlir::llk::createBlockPointerToVectorPass());
  ASSERT_TRUE(succeeded(pm.run(*module)));
  // Do not execute the historically unsafe addressing, including in RED.
  ASSERT_EQ(print(*module).find("strided<[4096, 1]"), std::string::npos);
  ::llk::JitCache cache;
  using Fn =
      void (*)(void *, void *, int64_t, int64_t, int64_t, int64_t, int64_t,
               void *, void *, int64_t, int64_t, int64_t, int64_t, int64_t);
  auto fn = cache.lookupTyped<Fn>("legacy_copy", *module);
  ASSERT_TRUE(static_cast<bool>(fn)) << llvm::toString(fn.takeError());
  constexpr int elements = 128 * 64, guard = 16;
  std::vector<float> src(elements + 2 * guard, -12345),
      dst(elements + 2 * guard, -12345);
  for (int i = 0; i < elements; ++i)
    src[guard + i] = float(i + 1);
  auto srcBefore = src;
  (*fn)(src.data() + guard, src.data() + guard, 0, 128, 64, 64, 1,
        dst.data() + guard, dst.data() + guard, 0, 128, 64, 64, 1);
  EXPECT_EQ(src, srcBefore);
  for (int i = 0; i < elements + 2 * guard; ++i) {
    int row = (i - guard) / 64, col = (i - guard) % 64;
    bool copied = i >= guard && i < elements + guard && row >= c.row &&
                  row < c.row + 32 && col >= c.col;
    if (masked)
      copied = copied && row < c.row + c.maskRows && col < c.col + c.maskCols;
    EXPECT_EQ(dst[i], copied ? float(i - guard + 1) : -12345) << i;
  }
}
INSTANTIATE_TEST_SUITE_P(
    FrontendLegacy, FrontendLegacyCopy,
    testing::Values(CopyCase{"ZeroOffsets", 0, 0, -1, -1},
                    CopyCase{"NonzeroOffsets", 2, 3, -1, -1},
                    CopyCase{"TailWithAllTrueUserMask", 126, 62, 32, 64},
                    CopyCase{"PartialUserMask", 3, 4, 4, 8},
                    CopyCase{"AllFalseUserMask", 0, 0, 0, 0}),
    [](const auto &i) { return i.param.name; });

#endif // LLK_FRONTEND_ENABLE_EXECUTION
} // namespace
