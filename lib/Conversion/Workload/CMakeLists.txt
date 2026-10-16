# The rank-neutral workload contract: plain port/axis/semantics data plus
# MLIR attribute serialization. Depends only on LLVM Support and MLIR IR so it
# can sit below the source-candidate adapters, perf and mapping integration.
add_llk_library(LLKWorkloadContracts
    WorkloadContract.cpp
    LINK_LIBRARIES LLVM LLK::MLIR
)

# The source-adapter boundary: registry, root selection, parameter ownership
# and realization receipts. It depends on the contract data and the typed
# search space, so the adapter above it never re-parses MLIR attributes.
add_llk_library(LLKSourceCandidates
    SourceAdapter.cpp
    BindingSchema.cpp
    LINK_LIBRARIES LLKWorkloadContracts LLKPerf LLK::MLIR
)
