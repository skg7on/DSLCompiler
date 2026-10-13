add_llk_library(LLKCommon
    Common/TilingUtils.cpp
    Common/VectorizationUtils.cpp
    Common/ScheduleLoader.cpp
    Common/MaskGeneration.cpp
    Common/MathApproximation.cpp
    LINK_LIBRARIES LLKDialect LLK::MLIR
)

add_llk_library(LLKTransforms
    TileAndVectorize.cpp
    DEPENDS LLKTransformsGen
    LINK_LIBRARIES LLKToLinalg LLKCommon LLK::MLIR
)

add_llk_library(LLKFuseDoubleContraction
    FuseDoubleContraction.cpp
    LINK_LIBRARIES LLKToLinalg LLK::MLIR
)

add_llk_library(LLKPackWeights
    PackWeights.cpp
    LINK_LIBRARIES LLKDialect LLK::MLIR
)

add_llk_library(LLKScratchAnalysis
    ScratchAnalysis.cpp
    LINK_LIBRARIES LLK::MLIR
)

add_llk_library(LLKLinearizeForall
    LinearizeForall.cpp
    LINK_LIBRARIES LLKDialect LLK::MLIR
)

add_llk_library(LLKSerialParallelDispatch
    SerialParallelDispatch.cpp
    LINK_LIBRARIES LLK::MLIR
)

add_llk_library(LLKForallToLLRT
    ForallToLLRT.cpp
    LINK_LIBRARIES LLK::MLIR
)

add_llk_library(LLKForallToOpenMP
    ForallToOpenMP.cpp
    DEPENDS LLKTransformsGen
    LINK_LIBRARIES LLK::MLIR
)

add_llk_library(LLKShapeSpecialization
    ShapeSpecialization.cpp
    LINK_LIBRARIES LLKDialect LLK::MLIR
)

add_llk_library(LLKScheduleSelection
    ScheduleSelection.cpp
    LINK_LIBRARIES LLKDialect LLK::MLIR
)
