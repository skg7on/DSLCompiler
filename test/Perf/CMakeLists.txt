add_llk_test_category(Perf
    SOURCES
        ../Mapping/resource_regression_fixture.cpp
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

if(LLK_BUILD_TOOLS)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_test(NAME AnalysisCoverageCLI
        COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/analysis_coverage_cli.py
            $<TARGET_FILE:micro-perf> ${CMAKE_SOURCE_DIR})
endif()
