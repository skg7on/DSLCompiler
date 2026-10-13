add_llk_library(LLKKernel
    Registry.cpp
    Contract.cpp
    SwiGLU.cpp
    RoPE.cpp
    Attention.cpp
    Matmul.cpp
    LinalgMatmul.cpp
    LINK_LIBRARIES LLKDialect MicroDialect LLKCommon LLK::MLIR
)
