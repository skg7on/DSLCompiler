# The rank-neutral workload contract: plain port/axis/semantics data plus
# MLIR attribute serialization. Depends only on LLVM Support and MLIR IR so it
# can sit below the source-candidate adapters, perf and mapping integration.
add_llk_library(LLKWorkloadContracts
    WorkloadContract.cpp
    LINK_LIBRARIES LLVM LLK::MLIR
)
