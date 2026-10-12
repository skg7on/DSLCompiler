add_llk_library(LLKToMicro
    LLKToMicro.cpp
    LINK_LIBRARIES LLKDialect MicroDialect LLKCommon LLK::MLIR
)
