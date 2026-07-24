//===- llk-compile.cpp - LLK kernel compiler driver -----------------------===//
//
// CLI tool that parses an MLIR module containing LLK dialect ops,
// runs the full progressive lowering pipeline (LLK→Linalg→...→LLVM IR),
// JIT-compiles the result, and reports success.
//
// Usage: llk-compile <input.mlir> [--M=<dim>] [--N=<dim>] [--K=<dim>]
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/LLKToLinalg.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Runtime/JitCache.h"
#include "LLK/Transforms/ForallToLLRT.h"
#include "LLK/Transforms/FuseDoubleContraction.h"
#include "LLK/Transforms/LinearizeForall.h"
#include "LLK/Transforms/PackWeights.h"
#include "LLK/Transforms/ScheduleSelection.h"
#include "LLK/Transforms/ScratchAnalysis.h"
#include "LLK/Transforms/SerialParallelDispatch.h"
#include "LLK/Transforms/ShapeSpecialization.h"
#include "LLK/Transforms/TileAndVectorize.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/SubsetOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Config/llvm-config.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#if LLVM_VERSION_MAJOR >= 21
using BufOpts = mlir::bufferization::OneShotBufferizePassOptions;
#else
using BufOpts = mlir::bufferization::OneShotBufferizationOptions;
#endif

namespace cl = llvm::cl;

static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input .mlir>"),
                                      cl::Required);

static cl::opt<int64_t> optM("M", cl::desc("M dimension (rows)"), cl::init(0));
static cl::opt<int64_t> optN("N", cl::desc("N dimension (cols)"), cl::init(0));
static cl::opt<int64_t> optK("K", cl::desc("K dimension (hidden)"),
                             cl::init(0));

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

static mlir::LogicalResult runCompilationPipeline(mlir::ModuleOp module) {
  mlir::PassManager pm(module->getContext());

  // Stage 1: Lower LLK dialect ops to Linalg + Arith + Math.
  pm.addPass(mlir::llk::createLLKToLinalgPass());

  // Stage 2: Canonicalize to clean up the lowered IR.
  pm.addPass(mlir::createCanonicalizerPass(mlir::GreedyRewriteConfig()));

  // Stage 3: Shape specialization — classify M bucket and emit dispatch guards.
  pm.addPass(mlir::llk::createShapeSpecializationPass());

  // Stage 4: Select schedule from schedule_db.json based on (op, M, N, K, ISA).
  pm.addPass(mlir::llk::createScheduleSelectionPass());

  // Stage 5: Fuse double contraction (SwiGLU-specific; no-op for
  // RoPE/Attention).
  pm.addPass(mlir::llk::createFuseDoubleContractionPass());

  // Stage 6: Canonicalize after fusion.
  pm.addPass(mlir::createCanonicalizerPass(mlir::GreedyRewriteConfig()));

  // Stage 7: Annotate weights with packing layout.
  pm.addPass(mlir::llk::createPackWeightsPass());

  // Stage 8: Tile and vectorize using Transform dialect schedule.
  pm.addPass(mlir::llk::createTileAndVectorizePass());

  // Stage 9: Canonicalize after tiling/vectorization.
  pm.addPass(mlir::createCanonicalizerPass(mlir::GreedyRewriteConfig()));

  // Stage 10: Linearize 2D scf.forall into 1D (Parallel Decompose).
  pm.addPass(mlir::llk::createLinearizeForallPass());

  // Stage 11: Select serial vs parallel dispatch based on shape thresholds.
  pm.addPass(mlir::llk::createSerialParallelDispatchPass());

  // Stage 12: One-Shot Bufferize (tensor → memref).
  BufOpts bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));

  // Stage 13: Audit scratch allocations post-bufferization.
  pm.addPass(mlir::llk::createScratchAnalysisPass());

  // Stage 14: Lower residual linalg ops on memref to scf.for loops.
  // This handles linalg.generic and linalg.fill ops that survive past
  // TileAndVectorize (which only tiles matmul/contraction ops).
  // Uses the built-in MLIR pass "convert-linalg-to-loops".
  {
    const auto *passInfo = mlir::PassInfo::lookup("convert-linalg-to-loops");
    if (passInfo) {
      if (mlir::failed(
              passInfo->addToPipeline(pm, {}, [](const llvm::Twine &msg) {
                llvm::errs() << "convert-linalg-to-loops: " << msg << "\n";
                return mlir::failure();
              }))) {
        llvm::errs()
            << "ERROR: convert-linalg-to-loops failed to add to pipeline\n";
        return mlir::failure();
      }
    }
  }

  // Stage 15: Lower scf.forall to llrt.parallel_for (ThreadPool runtime).
  pm.addPass(mlir::llk::createForallToLLRTPass());

  // Stage 16: Convert Vector → LLVM (SIMD intrinsics).
  pm.addPass(mlir::createConvertVectorToLLVMPass());

  // Stage 17: Convert SCF → CF.
#if LLVM_VERSION_MAJOR >= 21
  pm.addPass(mlir::createSCFToControlFlowPass());
#else
  pm.addPass(mlir::createConvertSCFToCFPass());
#endif

  // Stage 18: Convert Arith → LLVM.
  pm.addPass(mlir::createArithToLLVMConversionPass());

  // Stage 19: Convert Math → LLVM.
  pm.addPass(mlir::createConvertMathToLLVMPass());

  // Stage 20: Convert Func → LLVM.
  pm.addPass(mlir::createConvertFuncToLLVMPass());

  // Stage 21: Convert MemRef → LLVM.
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());

  // Stage 22: Reconcile unrealized casts from partial conversions.
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  // NOTE: JitCache::lookupOrCompile runs its own lowering passes
  // inside the cache miss path; these are no-ops when the module
  // is already fully lowered to LLVM dialect.

  return pm.run(module);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Register all built-in MLIR passes.
  mlir::registerAllPasses();

  // Register all LLK passes.
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createLLKToLinalgPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createShapeSpecializationPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createScheduleSelectionPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createFuseDoubleContractionPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createPackWeightsPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createTileAndVectorizePass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createLinearizeForallPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createSerialParallelDispatchPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createScratchAnalysisPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createForallToLLRTPass();
  });

  // Parse command line.
  cl::ParseCommandLineOptions(argc, argv, "LLK kernel compiler\n");

  // Build dialect registry with all required dialects.
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                  mlir::scf::SCFDialect, mlir::arith::ArithDialect,
                  mlir::math::MathDialect, mlir::memref::MemRefDialect,
                  mlir::LLVM::LLVMDialect>();

  // Register TilingInterface external models so Linalg ops can be tiled.
  mlir::linalg::registerTilingInterfaceExternalModels(registry);

  // Register BufferizableOpInterface external models for One-Shot Bufferize.
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);

  // Register SubsetOpInterface external models for bufferization analysis.
  mlir::vector::registerSubsetOpInterfaceExternalModels(registry);

  // Register ValueBoundsOpInterface external models for bufferization
  // analysis (needed when the IR contains scf.for, scf.forall,
  // arith, and linalg ops with dynamic shapes).
  mlir::arith::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::linalg::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);

  mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  // Parse the input MLIR file.
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(inputFile, &ctx);
  if (!module) {
    llvm::errs() << "Failed to parse " << inputFile << "\n";
    return 1;
  }

  // Run the compilation pipeline.
  if (mlir::failed(runCompilationPipeline(*module))) {
    llvm::errs() << "Compilation pipeline failed\n";
    return 1;
  }

  // Derive a cache key from the module content and shape parameters.
  // Hash the module source to get a stable key that varies by operation.
  std::string moduleStr;
  {
    llvm::raw_string_ostream os(moduleStr);
    module->print(os);
  }
  std::string cacheKey = "llk_" +
                         std::to_string(std::hash<std::string>{}(moduleStr)) +
                         "_M" + std::to_string(optM) + "_N" +
                         std::to_string(optN) + "_K" + std::to_string(optK);

  // JIT-compile and look up the kernel entry point.
  llk::JitCache cache;
  auto fnOrErr = cache.lookupOrCompile(cacheKey, *module);
  if (!fnOrErr) {
    llvm::errs() << "JIT compilation failed: "
                 << llvm::toString(fnOrErr.takeError()) << "\n";
    return 1;
  }

  llvm::outs() << "Compilation successful\n";
  return 0;
}
