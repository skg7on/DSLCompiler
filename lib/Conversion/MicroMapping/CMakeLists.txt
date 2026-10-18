add_llk_library(LLKMappedKernelAbi
    MappedKernelAbi.cpp
    LINK_LIBRARIES LLK::MLIR
)

add_llk_library(LLKMicroMapping
    MicroMapPass.cpp
    MicroBindPlanPass.cpp
    MicroVerifyMappingPass.cpp
    SearchBindingLoader.cpp
    CandidateInstantiation.cpp
    PlanMaterialization.cpp
    CompletePlanEvaluation.cpp
    TargetCoverageValidation.cpp
    LINK_LIBRARIES LLKMapping LLKTargetMapping LLKMachine LLKPerf LLKToMicro LLKWorkloadContracts LLKSourceCandidates LLK::MLIR
)

add_llk_library(LLKMappedCompilation
    MappedCompilation.cpp
    LINK_LIBRARIES LLKMicroMapping LLKMicroToLinalg LLKMappedKernelAbi
                   LLKWorkloadContracts LLKRuntime LLKMapping LLKTargetMapping
                   LLK::MLIR
)

add_llk_library(LLKMappedTuning
    MappedTuningSession.cpp
    LINK_LIBRARIES LLKMicroMapping LLKMappedCompilation
)

target_include_directories(LLKMappedCompilation PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}" "${PROJECT_SOURCE_DIR}/test/Execution")
target_include_directories(LLKMappedTuning PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
