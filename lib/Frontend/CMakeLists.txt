add_llk_library(LLKFrontend
    Program.cpp
    Importer.cpp
    LINK_LIBRARIES LLKSourceCandidates LLKWorkloadContracts LLKPerf LLK::MLIR LLVM
)
