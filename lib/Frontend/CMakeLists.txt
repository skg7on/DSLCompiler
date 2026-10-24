add_llk_library(LLKFrontend
    Program.cpp
    Importer.cpp
    Compiler.cpp
    LINK_LIBRARIES LLKSourceCandidates LLKWorkloadContracts LLKPerf LLKMappedTuning LLKTargetMapping LLKRuntime LLK::MLIR LLVM
)
