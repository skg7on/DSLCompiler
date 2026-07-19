//===- llk-compile.cpp - LLK kernel compiler driver -----------------------===//
//
// CLI tool that parses an MLIR module containing llk.fused_swiglu,
// runs the progressive lowering pipeline (LLK→Linalg→Bufferize),
// JIT-compiles the result, and reports success.
//
// Usage: llk-compile <input.mlir> [--M=<dim>] [--N=<dim>] [--K=<dim>]
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/LLKToLinalg.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Runtime/JitCache.h"

#include "mlir/Conversion/Passes.h"
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
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace cl = llvm::cl;

static cl::opt<std::string>
    inputFile(cl::Positional, cl::desc("<input .mlir>"), cl::Required);

static cl::opt<int64_t> optM("M", cl::desc("M dimension (rows)"), cl::init(0));
static cl::opt<int64_t> optN("N", cl::desc("N dimension (cols)"), cl::init(0));
static cl::opt<int64_t> optK("K", cl::desc("K dimension (hidden)"), cl::init(0));

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

static mlir::LogicalResult runCompilationPipeline(mlir::ModuleOp module) {
    mlir::PassManager pm(module->getContext());

    // Step 1: Lower LLK dialect ops to Linalg + Arith + Math.
    pm.addPass(mlir::llk::createLLKToLinalgPass());

    // Step 2: Canonicalize to clean up the lowered IR.
    pm.addPass(mlir::createCanonicalizerPass(mlir::GreedyRewriteConfig()));

    // Step 3: One-Shot Bufferize (tensor → memref).
    mlir::bufferization::OneShotBufferizeOptions bufOpts;
    bufOpts.bufferizeFunctionBoundaries = true;
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));

    return pm.run(module);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    llvm::InitLLVM y(argc, argv);

    // Register all built-in MLIR passes.
    mlir::registerAllPasses();

    // Register the LLK-to-Linalg pass with the global registry.
    mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
        return mlir::llk::createLLKToLinalgPass();
    });

    // Parse command line.
    cl::ParseCommandLineOptions(argc, argv, "LLK kernel compiler\n");

    // Build dialect registry with all required dialects.
    mlir::DialectRegistry registry;
    registry.insert<mlir::llk::LLKDialect,
                    mlir::func::FuncDialect,
                    mlir::linalg::LinalgDialect,
                    mlir::tensor::TensorDialect,
                    mlir::scf::SCFDialect,
                    mlir::arith::ArithDialect,
                    mlir::math::MathDialect,
                    mlir::memref::MemRefDialect>();

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

    // JIT-compile and look up the kernel entry point.
    llk::JitCache cache;
    std::string cacheKey = "swiglu_test";
    auto fnOrErr = cache.lookupOrCompile(cacheKey, *module);
    if (!fnOrErr) {
        llvm::errs() << "JIT compilation failed: "
                     << llvm::toString(fnOrErr.takeError()) << "\n";
        return 1;
    }

    llvm::outs() << "Compilation successful\n";
    return 0;
}
