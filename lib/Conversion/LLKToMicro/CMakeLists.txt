add_llk_library(LLKToMicro
    LLKToMicro.cpp
    MicroKernelDriver.cpp
    LINK_LIBRARIES LLKDialect MicroDialect LLKCommon LLKKernel LLK::MLIR
)
