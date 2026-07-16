//===- llk-opt.cpp - LLK optimizer driver ---------------------------------===//
//
// Main entry point for the LLK optimizer tool. Parses MLIR input, runs
// optimization passes, and emits the result.
//
//===----------------------------------------------------------------------===//

#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

// Include the LLK dialect declaration to register it.
#include "LLK/Dialect/LLKDialect.h.inc"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register all built-in MLIR dialects.
  mlir::registerAllDialects(registry);

  // Register the LLK dialect.
  registry.insert<mlir::llk::LLKDialect>();

  // Register all built-in MLIR passes.
  mlir::registerAllPasses();
  // LLKToLinalg pass registration will be added in Task 1.5.

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "LLK optimizer driver\n", registry));
}
