add_llk_test_category(Perf
    SOURCES
        $<TARGET_OBJECTS:LLKTestFixtures>
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
        dynamic_shape_analysis.cpp
    LIBRARIES
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
        LLK::MLIR
    DEFINITIONS
        LLK_MACHINE_DIR="${CMAKE_SOURCE_DIR}/machines"
        LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    INCLUDE_DIRECTORIES
        ${CMAKE_SOURCE_DIR}/lib/Conversion/MicroMapping
        ${CMAKE_SOURCE_DIR}/test/Mapping
)

# Historical gate name: the sampled dynamic-analysis acceptance is part of the
# category binary, while the plan still cites this binary as its gate.
add_test(NAME DynamicShapeAnalysisTest COMMAND LLKPerfTests
    --gtest_filter=DynamicShapeAnalysis.*)
set_tests_properties(DynamicShapeAnalysisTest PROPERTIES
    LABELS "Perf"
    TIMEOUT 60
    SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]")

if(LLK_BUILD_TOOLS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME AnalysisCoverageCLI
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/analysis_coverage_cli.py
            $<TARGET_FILE:micro-perf> ${CMAKE_SOURCE_DIR})
endif()
