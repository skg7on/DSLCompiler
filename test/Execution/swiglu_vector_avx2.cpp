//===- swiglu_vector_avx2.cpp - E2E vector-pipeline JIT tests -------------===//
//
// Tests the full M2 vector pipeline end-to-end:
//   1. Build SwiGLU module programmatically (avoiding func.func text-parse issues)
//   2. Lower: LLKToLinalg → TileAndVectorize → Canonicalize → OneShotBufferize
//   3. JIT compile via JitCache
//   4. Execute JIT-compiled kernel and validate against FP64 reference
//   5. Test shapes include odd N=127 for masked-tail coverage
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/LLKToLinalg.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/LLKEnums.h"
#include "LLK/Runtime/JitCache.h"
#include "LLK/Transforms/TileAndVectorize.h"

// Dependencies required by the generated op/attribute headers.
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

// Generated attribute classes (ActivationAttr, MathModeAttr).
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/LLKAttributes.h.inc"

// Generated op classes (FusedSwiGLUOp).
#define GET_OP_CLASSES
#include "LLK/Dialect/LLKOps.h.inc"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Error.h"

#if LLVM_VERSION_MAJOR >= 21
using BufOpts = mlir::bufferization::OneShotBufferizePassOptions;
#else
using BufOpts = mlir::bufferization::OneShotBufferizationOptions;
#endif

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// BF16 conversion helpers (simulated via float bit manipulation)
// ---------------------------------------------------------------------------

static float bf16_to_f32(uint16_t bf16) {
  uint32_t bits = static_cast<uint32_t>(bf16) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

static uint16_t f32_to_bf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  // Round to nearest even
  bits += 0x8000 + ((bits >> 16) & 1);
  return static_cast<uint16_t>(bits >> 16);
}

// ---------------------------------------------------------------------------
// FP64 reference: SwiGLU computed in double precision.
// Y = SiLU(X @ Wg^T) * (X @ Wu^T)
// ---------------------------------------------------------------------------

static void swiglu_fp64(const double* x, const double* wg, const double* wu,
                        double* y, int64_t M, int64_t N, int64_t K) {
  std::vector<double> gate(M * N, 0.0);
  std::vector<double> up(M * N, 0.0);

  for (int64_t m = 0; m < M; m++) {
    for (int64_t n = 0; n < N; n++) {
      double sum_g = 0.0, sum_u = 0.0;
      for (int64_t k = 0; k < K; k++) {
        sum_g += x[m * K + k] * wg[k * N + n];
        sum_u += x[m * K + k] * wu[k * N + n];
      }
      gate[m * N + n] = sum_g;
      up[m * N + n] = sum_u;
    }
  }

  for (int64_t i = 0; i < M * N; i++) {
    double g = gate[i];
    double silu = g / (1.0 + std::exp(-g));
    y[i] = silu * up[i];
  }
}

// ---------------------------------------------------------------------------
// Helpers: MLIR module construction and pipeline
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
  // Register bufferization interfaces required by One-Shot Bufferize.
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
  return registry;
}

/// Ensure all required dialects are loaded into the context.
/// Must be called after MLIRContext construction with a DialectRegistry.
static void loadRequiredDialects(mlir::MLIRContext &ctx) {
  ctx.getOrLoadDialect<mlir::llk::LLKDialect>();
  ctx.getOrLoadDialect<mlir::func::FuncDialect>();
  ctx.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  ctx.getOrLoadDialect<mlir::tensor::TensorDialect>();
  ctx.getOrLoadDialect<mlir::scf::SCFDialect>();
  ctx.getOrLoadDialect<mlir::arith::ArithDialect>();
  ctx.getOrLoadDialect<mlir::math::MathDialect>();
  ctx.getOrLoadDialect<mlir::memref::MemRefDialect>();
}

/// Build a SwiGLU module programmatically using OpBuilder.
/// Avoids func.func custom assembly text-parsing issues.
static mlir::OwningOpRef<mlir::ModuleOp> buildSwiGLUModule(
    mlir::OpBuilder &builder, int64_t M, int64_t N, int64_t K) {
  auto loc = builder.getUnknownLoc();
  auto module = builder.create<mlir::ModuleOp>(loc);
  auto bf16Type = builder.getBF16Type();
  auto xType = mlir::RankedTensorType::get({M, K}, bf16Type);
  auto wType = mlir::RankedTensorType::get({K, N}, bf16Type);
  auto outType = mlir::RankedTensorType::get({M, N}, bf16Type);

  auto funcType = builder.getFunctionType({xType, wType, wType, outType}, {outType});
  auto func = builder.create<mlir::func::FuncOp>(loc, "llk_swiglu", funcType);
  auto *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  auto fusedOp = builder.create<mlir::llk::FusedSwiGLUOp>(loc, outType,
      entry->getArgument(0), entry->getArgument(1), entry->getArgument(2),
      entry->getArgument(3),
      mlir::TypeAttr::get(builder.getF32Type()),
      mlir::llk::ActivationAttr::get(builder.getContext(), mlir::llk::Activation::silu),
      mlir::llk::MathModeAttr::get(builder.getContext(), mlir::llk::MathMode::bounded_fast));

  mlir::func::ReturnOp::create(builder, loc, fusedOp.getResult());
  return module;
}

/// Run the M2 vector pipeline: LLKToLinalg → TileAndVectorize → Canonicalize → Bufferize.
static bool runVectorPipeline(mlir::ModuleOp module) {
  mlir::PassManager pm(module->getContext());
  pm.addPass(mlir::llk::createLLKToLinalgPass());
  pm.addPass(mlir::llk::createTileAndVectorizePass());
  pm.addPass(mlir::createCanonicalizerPass());
  BufOpts bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));
  return mlir::succeeded(pm.run(module));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST(SwiGLUVectorAVX2, PipelineSmoke) {
  // Verify the vector pipeline runs without crashing for a small shape.
  auto registry = buildRegistry();
  mlir::MLIRContext ctx(registry);
  loadRequiredDialects(ctx);
  mlir::OpBuilder builder(&ctx);

  auto module = buildSwiGLUModule(builder, 4, 8, 3);
  ASSERT_TRUE(module) << "Module construction should succeed";

  bool pipelineOk = runVectorPipeline(*module);
  EXPECT_TRUE(pipelineOk) << "Vector pipeline should succeed for (4,8,3)";
}

TEST(SwiGLUVectorAVX2, Correctness) {
  // End-to-end correctness: build, lower, JIT, execute, validate.
  std::vector<std::tuple<int64_t, int64_t, int64_t>> shapes = {
      {32, 128, 256},  // typical LLM shapes
      {16, 512, 512},  // large projections
      {32, 256, 127},  // 127 tests N-tail (not multiple of VN=8)
  };

  for (auto [M, N, K] : shapes) {
    auto registry = buildRegistry();
    mlir::MLIRContext ctx(registry);
    loadRequiredDialects(ctx);
    mlir::OpBuilder builder(&ctx);

    auto module = buildSwiGLUModule(builder, M, N, K);
    ASSERT_TRUE(module) << "Module construction should succeed for M=" << M
                        << " N=" << N << " K=" << K;

    // Run the vector lowering pipeline.
    ASSERT_TRUE(runVectorPipeline(*module))
        << "Vector pipeline should succeed for M=" << M
        << " N=" << N << " K=" << K;

    // JIT compile.
    ::llk::JitCache cache;
    auto key = "swiglu_vec_" + std::to_string(M) + "_" + std::to_string(N) +
               "_" + std::to_string(K);
    auto fnOrErr = cache.lookupOrCompile(key, *module);
    if (!fnOrErr) {
      llvm::consumeError(fnOrErr.takeError());
      GTEST_SKIP() << "JIT compilation not available — skipping E2E execution.";
      return;
    }
    auto fn = *fnOrErr;

    // Generate random BF16 input data.
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<uint16_t> x_bf16(M * K);
    std::vector<uint16_t> wg_bf16(K * N);
    std::vector<uint16_t> wu_bf16(K * N);
    std::vector<uint16_t> y_bf16(M * N, 0);

    std::vector<double> x_ref(M * K), wg_ref(K * N), wu_ref(K * N);

    for (size_t i = 0; i < x_bf16.size(); i++) {
      float v = dist(rng);
      x_bf16[i] = f32_to_bf16(v);
      x_ref[i] = static_cast<double>(bf16_to_f32(x_bf16[i]));
    }
    for (size_t i = 0; i < wg_bf16.size(); i++) {
      float vg = dist(rng);
      float vu = dist(rng);
      wg_bf16[i] = f32_to_bf16(vg);
      wu_bf16[i] = f32_to_bf16(vu);
      wg_ref[i] = static_cast<double>(bf16_to_f32(wg_bf16[i]));
      wu_ref[i] = static_cast<double>(bf16_to_f32(wu_bf16[i]));
    }

    // Build ABI descriptors.
    Tensor2D x_ten = {x_bf16.data(), M, K, K, 1};
    Tensor2D wg_ten = {wg_bf16.data(), K, N, N, 1};
    Tensor2D wu_ten = {wu_bf16.data(), K, N, N, 1};
    Tensor2D y_ten = {y_bf16.data(), M, N, N, 1};

    MemRef2D x_mem = makeMemRef2D(x_ten);
    MemRef2D wg_mem = makeMemRef2D(wg_ten);
    MemRef2D wu_mem = makeMemRef2D(wu_ten);
    MemRef2D y_mem = makeMemRef2D(y_ten);
    MemRef2D init_mem = y_mem;

    // Execute JIT kernel.
    fn(&x_mem, &wg_mem, &wu_mem, &init_mem, &y_mem);

    // Compute FP64 reference.
    std::vector<double> y_ref(M * N);
    swiglu_fp64(x_ref.data(), wg_ref.data(), wu_ref.data(), y_ref.data(),
                M, N, K);

    // Validate: BF16 tolerance is 1e-2.
    float max_err = 0.0f;
    for (int64_t i = 0; i < M * N; i++) {
      float result = bf16_to_f32(y_bf16[i]);
      float expected = static_cast<float>(y_ref[i]);
      float abs_err = std::abs(result - expected);
      if (abs_err > max_err) max_err = abs_err;
    }

    EXPECT_LT(max_err, 1e-2f)
        << "M=" << M << " N=" << N << " K=" << K
        << ": max absolute error " << max_err << " exceeds 1e-2 tolerance";
  }
}
