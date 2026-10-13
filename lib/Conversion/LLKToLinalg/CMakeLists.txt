add_llk_library(LLKToLinalg
    LLKToLinalg.cpp
    LINK_LIBRARIES LLKDialect LLKCommon LLKKernel LLK::MLIR
)
