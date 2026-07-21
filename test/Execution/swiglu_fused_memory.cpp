//===- swiglu_fused_memory.cpp - Fused memory execution test -------------===//
//
// Tests the full M3 fused memory pipeline:
//   1. Build SwiGLU module with M=128,N=4096,K=4096 (programmatic, avoids
//      func.func text-parsing issues)
//   2. Run: LLKToLinalg -> FuseDoubleContraction -> Canonicalize ->
//      TileAndVectorize -> Canonicalize -> OneShotBufferize
//   3. Verify no [M,N] sized memref.alloc survives the pipeline
//
// Without fusion, the gate and up projections each produce a [M,N] result
// tensor that bufferization converts to a memref.alloc: 2 * 128*4096*4B ~=
// 4 MB. With FuseDoubleContraction, the dual matmul is fused into a single
// generic op, eliminating the intermediate [M,N] allocations.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/LLKToLinalg.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/LLKEnums.h"
#include "LLK/Runtime/JitCache.h"
#include "LLK/Transforms/FuseDoubleContraction.h"
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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Config/llvm-config.h"

#if LLVM_VERSION_MAJOR >= 21
using BufOpts = mlir::bufferization::OneShotBufferizePassOptions;
#else
using BufOpts = mlir::bufferization::OneShotBufferizationOptions;
#endif

#include <gtest/gtest.h>

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers: MLIR module construction and pipeline
// ---------------------------------------------------------------------------

static mlir::DialectRegistry buildRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                  mlir::scf::SCFDialect, mlir::arith::ArithDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect>();
  // Register bufferization interfaces required by One-Shot Bufferize.
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  return registry;
}

/// Ensure all required dialects are loaded into the context.
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
static mlir::OwningOpRef<mlir::ModuleOp>
buildSwiGLUModule(mlir::OpBuilder &builder, int64_t M, int64_t N, int64_t K) {
  auto loc = builder.getUnknownLoc();
  auto module = builder.create<mlir::ModuleOp>(loc);
  auto bf16Type = builder.getBF16Type();
  auto xType = mlir::RankedTensorType::get({M, K}, bf16Type);
  auto wType = mlir::RankedTensorType::get({K, N}, bf16Type);
  auto outType = mlir::RankedTensorType::get({M, N}, bf16Type);

  auto funcType =
      builder.getFunctionType({xType, wType, wType, outType}, {outType});
  auto func = builder.create<mlir::func::FuncOp>(loc, "llk_swiglu", funcType);
  auto *entry = func.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  auto fusedOp = builder.create<mlir::llk::FusedSwiGLUOp>(
      loc, outType, entry->getArgument(0), entry->getArgument(1),
      entry->getArgument(2), entry->getArgument(3),
      mlir::TypeAttr::get(builder.getF32Type()),
      mlir::llk::ActivationAttr::get(builder.getContext(),
                                     mlir::llk::Activation::silu),
      mlir::llk::MathModeAttr::get(builder.getContext(),
                                   mlir::llk::MathMode::bounded_fast));

  builder.create<mlir::func::ReturnOp>(loc, fusedOp.getResult());
  return module;
}

/// Run the fused memory pipeline:
///   LLKToLinalg -> FuseDoubleContraction -> Canonicalize ->
///   TileAndVectorize -> Canonicalize -> Bufferize
static bool runFusedPipeline(mlir::ModuleOp module) {
  mlir::PassManager pm(module->getContext());
  pm.addPass(mlir::llk::createLLKToLinalgPass());
  pm.addPass(mlir::llk::createFuseDoubleContractionPass());
  pm.addPass(mlir::createCanonicalizerPass());
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

TEST(SwiGLUFusedMemory, NoFullIntermediates) {
  // Large shape: M=128,N=4096,K=4096
  // Without fusion: gate + up = 2 * [128, 4096] BF16 allocations ~ 2 MB each.
  // With fusion: only tile scratch + packed weights remain.
  int64_t M = 128, N = 4096, K = 4096;

  auto registry = buildRegistry();
  mlir::MLIRContext ctx(registry);
  loadRequiredDialects(ctx);
  mlir::OpBuilder builder(&ctx);

  auto module = buildSwiGLUModule(builder, M, N, K);
  ASSERT_TRUE(module) << "Module construction should succeed";

  // Run the full fused lowering pipeline.
  ASSERT_TRUE(runFusedPipeline(*module))
      << "Fused memory pipeline should succeed for M=" << M << " N=" << N
      << " K=" << K;

  // Verify no [M, N] sized memref.alloc appears in the bufferized output.
  bool foundFullSizeAlloc = false;
  module->walk([&](mlir::memref::AllocOp alloc) {
    auto ty = alloc.getType();
    if (ty.hasStaticShape() && ty.getShape().size() == 2) {
      if (ty.getShape()[0] == M && ty.getShape()[1] == N) {
        foundFullSizeAlloc = true;
      }
    }
  });
  EXPECT_FALSE(foundFullSizeAlloc) << "Full-size [" << M << "," << N
                                   << "] allocation detected -- fusion failed";
}
