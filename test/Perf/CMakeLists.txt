add_llk_test_category(Perf
    SOURCES
        ../Mapping/resource_regression_fixture.cpp
        measurement_record.cpp
        prediction_validation.cpp
        op_dag_builders.cpp
        l0_static_bound.cpp
        l1_resource_dag.cpp
        route_identity.cpp
        search_space_loader.cpp
        candidate_generator.cpp
        legality.cpp
        candidate_binding.cpp
        tuning_session.cpp
    LIBRARIES
        LLKPerf
        MicroDialect
        LLKMapping
        LLKMicroMapping
        LLKMappedCompilation
        LLK::MLIR
    DEFINITIONS
        LLK_MACHINE_DIR="${CMAKE_SOURCE_DIR}/machines"
    INCLUDE_DIRECTORIES
        ${CMAKE_SOURCE_DIR}/lib/Conversion/MicroMapping
        ${CMAKE_SOURCE_DIR}/test/Mapping
)

# Sampled dynamic analysis: the shared DAG/plan analysis evaluated on explicit
# concrete shapes. It needs the real export and the shipped AVX2 machine, but
# no JIT.
add_executable(DynamicShapeAnalysisTest dynamic_shape_analysis.cpp)
target_link_libraries(DynamicShapeAnalysisTest PRIVATE LLKBuildOptions
    GTest::gtest_main
    LLKPerf
    MicroDialect
    LLKDialect
    LLKKernel
    LLKMapping
    LLKMicroMapping
    LLKMappedCompilation
    LLKTargetMapping
    LLKToMicro
    LLKSourceCandidates
    LLKWorkloadContracts
    LLK::MLIR)
target_compile_definitions(DynamicShapeAnalysisTest PRIVATE
    LLK_MACHINE_DIR="${CMAKE_SOURCE_DIR}/machines"
    LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
target_include_directories(DynamicShapeAnalysisTest PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/Conversion/MicroMapping)
add_test(NAME DynamicShapeAnalysisTest COMMAND DynamicShapeAnalysisTest)

if(LLK_BUILD_TOOLS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME AnalysisCoverageCLI
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/analysis_coverage_cli.py
            $<TARGET_FILE:micro-perf> ${CMAKE_SOURCE_DIR})
endif()
