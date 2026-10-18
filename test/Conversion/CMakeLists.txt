add_llk_test_category(Conversion
    SOURCES
        LLKKernel/registry.cpp
        LLKKernel/contract.cpp
        LLKKernel/fragments.cpp
        MicroMapping/search_binding_loader.cpp
        MicroMapping/complete_plan_evaluation.cpp
        framework_capabilities.cpp
        framework_conformance.cpp
        ../Mapping/resource_regression_fixture.cpp
    LIBRARIES
        LLKKernel
        LLKMappedCompilation
        LLKMicroMapping
        LLKTargetMapping
        MicroDialect
        LLK::MLIR
    DEFINITIONS
        LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
    INCLUDE_DIRECTORIES
        ${CMAKE_SOURCE_DIR}/lib/Conversion/MicroMapping
        ${CMAKE_SOURCE_DIR}/test/Mapping
)

# C3: selected-lowering evidence and a versioned manifest. This runs static
# lowering and analysis only; numeric selected execution is the separate
# FrameworkPrimitivesAVX2Test gate.
add_test(NAME FrameworkSelectedEvidence
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/framework_selected_evidence.py
        $<TARGET_FILE:llk-compile> $<TARGET_FILE:micro-perf>
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/framework_selected_evidence)
