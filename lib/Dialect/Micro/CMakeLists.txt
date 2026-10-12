add_llk_library(MicroDialect
    MicroDialect.cpp
    MicroOps.cpp
    DEPENDS MicroDialectGen MicroTypesGen MicroOpsGen
    LINK_LIBRARIES LLK::MLIR
)
