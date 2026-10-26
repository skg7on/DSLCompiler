# Plain symbolic shape data, checked binding and deterministic encoding. It
# depends on LLVM Support only, so workload contracts, generic mapping and the
# runtime can all share it without depending on one another or on MLIR.
add_llk_library(LLKShape
    ShapeContract.cpp
    LINK_LIBRARIES LLVM
)
