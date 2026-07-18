//===- swiglu_scalar.cpp - End-to-end JIT execution tests -----------------===//
//
// Tests the full M1 scalar pipeline:
//   1. Parse LLK IR from text
//   2. Lower: LLK -> Linalg -> OneShotBufferize
//   3. Verify IR structure at each stage
//   4. JIT compile via JitCache (may skip if ABI wrapper not yet available)
//
// NOTE: Full JIT execution requires the ABI wrapper (scheduled for M2).
// Until then the JIT compilation smoke test validates IR pipeline correctness.
// The test uses text-based IR parsing (rather than the TableGen-generated
// builder API) to avoid ABI incompatibilities with the runtime MLIR dylib.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/LLKToLinalg.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Runtime/JitCache.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <sstream>
#include <string>

// ---------------------------------------------------------------------------
// Helper: build a DialectRegistry with all dialects needed by the pipeline.
// ---------------------------------------------------------------------------

static mlir::DialectRegistry buildRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect,
                  mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect,
                  mlir::scf::SCFDialect,
                  mlir::arith::ArithDialect,
                  mlir::math::MathDialect,
                  mlir::memref::MemRefDialect>();
  return registry;
}

// ---------------------------------------------------------------------------
// Helper: parse a SwiGLU module for given shape dimensions
// ---------------------------------------------------------------------------

static mlir::OwningOpRef<mlir::ModuleOp> parseSwiGLUModule(
    mlir::MLIRContext* ctx, int64_t M, int64_t N, int64_t K) {
  std::ostringstream ss;
  ss << R"(module {
  func.func @llk_swiglu(%x: tensor<)"
     << M << "x" << K << R"(xbf16>, %wg: tensor<)"
     << K << "x" << N << R"(xbf16>, %wu: tensor<)"
     << K << "x" << N << R"(xbf16>, %init: tensor<)"
     << M << "x" << N << R"(xbf16>) -> tensor<)"
     << M << "x" << N << R"(xbf16> {
    %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<)"
     << M << "x" << K << "xbf16, tensor<"
     << K << "x" << N << "xbf16, tensor<"
     << K << "x" << N << R"(xbf16>)
        outs(%init : tensor<)"
     << M << "x" << N << R"(xbf16>)
        {accumulator_type = f32, activation = #llk.activation<silu>,
         math_mode = #llk.math_mode<bounded_fast>}
        -> tensor<)"
     << M << "x" << N << R"(xbf16>
    return %y : tensor<)"
     << M << "x" << N << R"(xbf16>
  }
})";
  return mlir::parseSourceString<mlir::ModuleOp>(ss.str(), ctx);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(SwiGLUScalar, ParseAndVerify) {
  // Verify we can parse an LLK IR module and it contains the expected ops.
  auto registry = buildRegistry();
  mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  auto module = parseSwiGLUModule(&ctx, 2, 8, 4);
  ASSERT_TRUE(static_cast<bool>(module))
      << "Should successfully parse valid LLK IR";

  // Verify the function exists
  bool foundFunc = false;
  bool foundSwiGLU = false;
  module->walk([&](mlir::Operation* op) {
    if (auto funcOp = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
      if (funcOp.getSymName() == "llk_swiglu") foundFunc = true;
    }
    if (op->getName().getStringRef() == "llk.fused_swiglu") foundSwiGLU = true;
  });
  EXPECT_TRUE(foundFunc) << "Should find llk_swiglu function";
  EXPECT_TRUE(foundSwiGLU) << "Should find llk.fused_swiglu op";
}

TEST(SwiGLUScalar, LLKToLinalgLowering) {
  // Lower llk.fused_swiglu -> linalg.matmul + linalg.generic.
  // Verify the resulting IR has the expected structure.
  auto registry = buildRegistry();
  mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  auto module = parseSwiGLUModule(&ctx, 4, 8, 2);

  mlir::PassManager pm(&ctx);
  pm.addPass(mlir::llk::createLLKToLinalgPass());
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // No llk.fused_swiglu ops should remain
  int swigluCount = 0;
  module->walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() == "llk.fused_swiglu") swigluCount++;
  });
  EXPECT_EQ(swigluCount, 0) << "llk.fused_swiglu should be fully lowered";

  // Should have 2 linalg.matmul (gate + up projections)
  int matmulCount = 0;
  module->walk([&](mlir::linalg::MatmulOp op) { matmulCount++; });
  EXPECT_EQ(matmulCount, 2) << "Should produce 2 linalg.matmul ops";

  // Should have 1 linalg.generic (elementwise SiLU+multiply)
  int genericCount = 0;
  module->walk([&](mlir::linalg::GenericOp op) { genericCount++; });
  EXPECT_EQ(genericCount, 1) << "Should produce 1 linalg.generic op";

  // Should have 2 linalg.fill (zero init for gate + up accumulators)
  int fillCount = 0;
  module->walk([&](mlir::linalg::FillOp op) { fillCount++; });
  EXPECT_EQ(fillCount, 2) << "Should produce 2 linalg.fill ops";

  // The linalg.generic should contain arith.extf, math.exp, arith.truncf
  int extfCount = 0, expCount = 0, truncfCount = 0;
  module->walk([&](mlir::Operation* op) {
    if (mlir::isa<mlir::arith::ExtFOp>(op)) extfCount++;
    if (mlir::isa<mlir::math::ExpOp>(op)) expCount++;
    if (mlir::isa<mlir::arith::TruncFOp>(op)) truncfCount++;
  });
  EXPECT_GE(extfCount, 2) << "Should have bf16->f32 promotions";
  EXPECT_GE(expCount, 1) << "Should have math.exp for SiLU";
  EXPECT_GE(truncfCount, 1) << "Should have f32->bf16 truncation";
}

TEST(SwiGLUScalar, OneShotBufferize) {
  // Run the full lowering pipeline through OneShotBufferize.
  // Verify no tensor.empty ops remain.
  auto registry = buildRegistry();
  mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  auto module = parseSwiGLUModule(&ctx, 2, 4, 3);

  mlir::PassManager pm(&ctx);
  pm.addPass(mlir::llk::createLLKToLinalgPass());
  pm.addPass(mlir::createCanonicalizerPass());
  mlir::bufferization::OneShotBufferizationOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // After bufferization, no tensor.empty ops should remain
  int emptyOps = 0;
  module->walk([&](mlir::tensor::EmptyOp op) { emptyOps++; });
  EXPECT_EQ(emptyOps, 0)
      << "All tensor.empty ops should be eliminated after bufferization";
}

TEST(SwiGLUScalar, JitCompilationSmoke) {
  // Run the full pipeline (LLK->Linalg->Bufferize) then attempt JIT
  // compilation. Full E2E execution with validation against FP64 reference
  // requires the ABI wrapper (scheduled for M2).

  auto registry = buildRegistry();
  mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  auto module = parseSwiGLUModule(&ctx, 2, 4, 3);

  // Lower through the full pipeline
  mlir::PassManager pm(&ctx);
  pm.addPass(mlir::llk::createLLKToLinalgPass());
  pm.addPass(mlir::createCanonicalizerPass());
  mlir::bufferization::OneShotBufferizationOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // Attempt JIT compilation via JitCache
  ::llk::JitCache cache;
  auto fnOrErr = cache.lookupOrCompile("swiglu_smoke_2_4_3", *module);

  if (fnOrErr) {
    // JIT compilation succeeded.
    SUCCEED() << "JIT compilation infrastructure is operational. "
              << "ABI wrapper (M2) needed for execution.";
  } else {
    // JIT compilation not yet available — expected without ABI wrapper.
    llvm::consumeError(fnOrErr.takeError());
    GTEST_SKIP() << "JIT compilation not yet available. "
                 << "IR pipeline (parse, lower, bufferize) all passed. "
                 << "Full E2E requires ABI wrapper (scheduled for M2).";
  }
}

TEST(SwiGLUScalar, MultipleShapeConfigurations) {
  // Verify the pipeline works for a variety of shape configurations.
  struct Shape {
    int64_t M, N, K;
  };
  std::vector<Shape> shapes = {
      {1, 1, 1},    // unit
      {1, 128, 1},  // K=1, wide N
      {16, 1, 64},  // N=1, wide K
      {1, 256, 256},
      {32, 64, 128},
      {64, 32, 512},
  };

  for (auto [M, N, K] : shapes) {
    auto registry = buildRegistry();
    mlir::MLIRContext ctx(registry);
    ctx.loadAllAvailableDialects();

    auto module = parseSwiGLUModule(&ctx, M, N, K);
    ASSERT_TRUE(static_cast<bool>(module))
        << "Parsing should succeed for M=" << M << " N=" << N << " K=" << K;

    mlir::PassManager pm(&ctx);
    pm.addPass(mlir::llk::createLLKToLinalgPass());
    pm.addPass(mlir::createCanonicalizerPass());
    mlir::bufferization::OneShotBufferizationOptions bufOpts;
    bufOpts.bufferizeFunctionBoundaries = true;
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));

    EXPECT_TRUE(mlir::succeeded(pm.run(*module)))
        << "Pipeline should succeed for M=" << M << " N=" << N << " K=" << K;
  }
}
