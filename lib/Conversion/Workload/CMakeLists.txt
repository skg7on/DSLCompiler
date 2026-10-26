# The rank-neutral workload contract: plain port/axis/semantics data plus
# MLIR attribute serialization. Depends only on LLVM Support and MLIR IR so it
# can sit below the source-candidate adapters, perf and mapping integration.
add_llk_library(LLKWorkloadContracts
    WorkloadContract.cpp
    LINK_LIBRARIES LLKShape LLVM LLK::MLIR
)

# The source-adapter boundary: registry, root selection, parameter ownership
# and realization receipts. It depends on the contract data and the typed
# search space, so the adapter above it never re-parses MLIR attributes. The
# built-in contraction family is a compatibility wrapper around the current
# LLK-to-Micro export, which is why this library also sits above LLKToMicro and
# the operator-kernel registry. LLKPerf consumes the contract header; it must
# never depend on this library.
add_llk_library(LLKSourceCandidates
    SourceAdapter.cpp
    BindingSchema.cpp
    ContractionSourceAdapter.cpp
    ElementwiseSourceAdapter.cpp
    LINK_LIBRARIES LLKWorkloadContracts LLKPerf LLKKernel LLKToMicro
                   MicroDialect LLKDialect LLK::MLIR
)
