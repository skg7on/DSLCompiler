//===- llk-opt.cpp - LLK optimizer driver ---------------------------------===//
//
// Main entry point for the LLK optimizer tool. Parses MLIR input, runs
// optimization passes, and emits the result.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

// Include the LLK dialect public header to register it.
#include "LLK/Conversion/LLKToLinalg.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Transforms/FuseDoubleContraction.h"
#include "LLK/Transforms/PackWeights.h"
#include "LLK/Transforms/ScratchAnalysis.h"
#include "LLK/Transforms/TileAndVectorize.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register all built-in MLIR dialects.
  mlir::registerAllDialects(registry);

  // Register the LLK dialect.
  registry.insert<mlir::llk::LLKDialect>();

  // Register Linalg transform dialect extensions (e.g.
  // transform.structured.match).
  mlir::linalg::registerTransformDialectExtension(registry);

  // Register all built-in MLIR passes.
  mlir::registerAllPasses();

  // Register the LLK-to-Linalg lowering pass.
  mlir::PassPipelineRegistration<> llkToLinalgPipeline(
      "llk-to-linalg-pipeline", "Full LLK-to-Linalg lowering pipeline",
      [](mlir::OpPassManager &pm) {
        pm.addPass(mlir::llk::createLLKToLinalgPass());
      });

  // Register the FuseDoubleContraction pass.
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createFuseDoubleContractionPass();
  });

  // Register the TileAndVectorize pass.
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createTileAndVectorizePass();
  });

  // Register the PackWeights pass.
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createPackWeightsPass();
  });

  // Register the ScratchAnalysis pass.
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createScratchAnalysisPass();
  });

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "LLK optimizer driver\n", registry));
}
