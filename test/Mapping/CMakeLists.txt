# Dynamic Shape Task 3 (issue #194): dynamic plan legality and resource
# certification. A standalone binary so its CTest entry is the named
# `DynamicShapeMappingTest` the task's verification gate runs, and so it can
# reuse the issue #129 resource fixtures without joining that category's binary.
add_executable(DynamicShapeMappingTest
    dynamic_shape_mapping.cpp
    $<TARGET_OBJECTS:LLKTestFixtures>
)
target_link_libraries(DynamicShapeMappingTest PRIVATE
    LLKMapping
    LLKTargetMapping
    LLKMicroMapping
    LLKPerf
    MicroDialect
    LLKShape
    LLK::MLIR
    GTest::gtest_main
)
target_compile_definitions(DynamicShapeMappingTest PRIVATE
    LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)
target_include_directories(DynamicShapeMappingTest PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/Conversion/MicroMapping
)
add_test(NAME DynamicShapeMappingTest COMMAND DynamicShapeMappingTest)

add_llk_test_category(Mapping
    SOURCES
        stable_hash.cpp
        codegen_requirements.cpp
        cost_model.cpp
        search_binding.cpp
        workload_graph.cpp
        tile_facts.cpp
        mapping_plan.cpp
        layout_constraints.cpp
        mapping_rules.cpp
        placement.cpp
        connections.cpp
        covering_search.cpp
        storage_plan.cpp
        avx2_target.cpp
        arm_policy.cpp
        arm_targets.cpp
        arm_sve_backend.cpp
        arm_sme_backend.cpp
        backend_pipeline_hooks.cpp
        arm_target_fixture.cpp
        arm_placement.cpp
        generic_accelerator_target.cpp
        plan_binder.cpp
        e2e_workflow.cpp
        routing.cpp
        plan_report.cpp
        properties.cpp
        exact_resource_oracle.cpp
        $<TARGET_OBJECTS:LLKTestFixtures>
        storage_liveness.cpp
    LIBRARIES
        LLKMapping
        LLKTargetMapping
        LLKMicroMapping
        LLKMappedCompilation
        LLKPerf
        LLKMicroToLinalg
        MicroDialect
        LLK::MLIR
    DEFINITIONS
        LLK_MAPPING_DIR="${CMAKE_SOURCE_DIR}/mapping"
        LLK_MACHINE_DIR="${CMAKE_SOURCE_DIR}/machines"
        LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        LLK_TEST_BINARY_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    INCLUDE_DIRECTORIES
        ${CMAKE_SOURCE_DIR}/lib/Conversion/MicroMapping
)
