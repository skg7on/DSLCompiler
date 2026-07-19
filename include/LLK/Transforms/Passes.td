//===----------------------------------------------------------------------===//
// LLK Transform Passes
//===----------------------------------------------------------------------===//

#ifndef LLK_TRANSFORMS_PASSES_TD
#define LLK_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def TileAndVectorize : Pass<"tile-and-vectorize", "mlir::ModuleOp"> {
    let summary = "Apply tiling and vectorization schedule to LLK ops";
    let description = [{
        Applies a Transform dialect schedule to tile, fuse, and vectorize
        LLK operations. Reads schedule from a .mlir transform file or uses
        built-in defaults keyed by target ISA.
    }];
    let constructor = "mlir::llk::createTileAndVectorizePass()";
    let options = [
        Option<"scheduleFile", "schedule-file", "std::string", /*default=*/"\"\"",
               "Path to Transform dialect schedule file">,
        Option<"targetIsa", "target-isa", "std::string", /*default=*/"\"avx2\"",
               "Target ISA: avx2, avx512, sve">,
    ];
}

#endif
