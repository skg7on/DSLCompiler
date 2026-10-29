add_llk_test_category(Conversion
    SOURCES
        LLKKernel/registry.cpp
        LLKKernel/contract.cpp
        LLKKernel/fragments.cpp
        MicroMapping/search_binding_loader.cpp
        MicroMapping/complete_plan_evaluation.cpp
        framework_capabilities.cpp
        framework_conformance.cpp
        framework_target_portability.cpp
        $<TARGET_OBJECTS:LLKTestFixtures>
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
if(TARGET llk-compile AND TARGET micro-perf)
  add_test(NAME FrameworkSelectedEvidence
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/framework_selected_evidence.py
          $<TARGET_FILE:llk-compile> $<TARGET_FILE:micro-perf>
          ${CMAKE_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}/framework_selected_evidence)
  set_tests_properties(FrameworkSelectedEvidence PROPERTIES
      FIXTURES_SETUP FrameworkSelectedEvidenceFiles)

  add_test(NAME FrameworkSelectedEvidenceRegression
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/framework_selected_evidence_test.py
          $<TARGET_FILE:llk-compile> $<TARGET_FILE:micro-perf> ${CMAKE_SOURCE_DIR})

  # Fixtures regenerate evidence when the validator is selected by itself and
  # prevent a failed producer from leaving stale evidence counted as passing.
  add_test(NAME FrameworkEvidenceManifest
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/framework_manifest_validator.py
          ${CMAKE_CURRENT_BINARY_DIR}/framework_selected_evidence)
  set_tests_properties(FrameworkEvidenceManifest PROPERTIES
      FIXTURES_REQUIRED FrameworkSelectedEvidenceFiles)
endif()

# C4: generic mapping/machine libraries stay target-neutral and do not link the
# performance model, runtime or JIT.
add_test(NAME FrameworkBoundaryScan
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/framework_boundary_scan.py
        ${CMAKE_SOURCE_DIR})

add_test(NAME FrameworkBoundaryScanRegression
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/framework_boundary_scan_test.py)
add_test(NAME FrameworkEvidenceManifestRegression
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/framework_manifest_validator_test.py)
if(TARGET llk-compile AND TARGET llk-opt)
  add_test(NAME FrameworkCoverageCli
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/framework_coverage_cli.py
          $<TARGET_FILE:llk-compile> $<TARGET_FILE:llk-opt> ${CMAKE_SOURCE_DIR})
endif()
