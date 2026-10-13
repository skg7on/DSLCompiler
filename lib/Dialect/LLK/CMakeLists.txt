add_llk_library(LLKDialect
    LLKDialect.cpp
    LLKOps.cpp
    DEPENDS LLKDialectGen LLKOpsGen
    LINK_LIBRARIES LLK::MLIR
)
