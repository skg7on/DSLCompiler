# One binary for the whole category. Two suites are also reachable under their
# historical binary names because the dynamic-shape plan cites them as gates.
set(_llk_tuning_sources
    workload_contract.cpp
    generic_source_candidates.cpp
    source_adapter_registry.cpp
    elementwise_source_adapter.cpp
    measurement_store.cpp
    dynamic_shape_contract.cpp)
if(LLK_BUILD_E2E_TESTS)
  list(APPEND _llk_tuning_sources
      mapped_tuning_session.cpp
      candidate_instantiation.cpp)
endif()

add_llk_test_category(Tuning
    SOURCES ${_llk_tuning_sources}
    LIBRARIES
        LLKWorkloadContracts
        LLKSourceCandidates
        LLKMappedTuning
        LLKMicroMapping
        LLKMicroToLinalg
        LLKShape
        LLKKernel
        LLKToMicro
        LLKDialect
        MicroDialect
        LLK::MLIR
    DEFINITIONS
        LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

# Historical gate names. See docs/superpowers/plans/2026-10-17-dynamic-shape-llk-microir.md.
add_test(NAME FrameworkSourceTest COMMAND LLKTuningTests
    --gtest_filter=WorkloadContract.*:FrameworkTest.*:BindingSchema.*:SourceAdapterRegistry.*:SourceSelection.*:ElementwiseSourceAdapter.*:ElementwiseSourceAdapterContext.*:MeasurementStoreTest.*)
add_test(NAME DynamicShapeContractTest COMMAND LLKTuningTests
    --gtest_filter=ShapeContract.*)
set_tests_properties(FrameworkSourceTest DynamicShapeContractTest
    PROPERTIES
        LABELS "Tuning"
        TIMEOUT 60
        SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]")

if(LLK_BUILD_E2E_TESTS)
  add_test(NAME MappedTuneCLI
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/mapped_tune_cli.py
          $<TARGET_FILE:llk-tune> $<TARGET_FILE:llk-compile> ${CMAKE_SOURCE_DIR})

  add_test(NAME MeasurementStoreCLI
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/measurement_store_cli.py
          $<TARGET_FILE:llk-tune> ${CMAKE_SOURCE_DIR})

  add_test(NAME GenericTuneCLI
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/generic_tune_cli.py
          $<TARGET_FILE:llk-tune> $<TARGET_FILE:llk-compile>
          ${CMAKE_SOURCE_DIR})
endif()

if(LLK_BUILD_TOOLS)
  add_test(NAME PredictionValidationCLI
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/prediction_validation_cli.py
          $<TARGET_FILE:llk-validate>)
endif()

if(LLK_BUILD_TOOLS AND LLK_BUILD_E2E_TESTS)
  add_test(NAME PredictionValidationMappedCLI
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/prediction_validation_mapped_cli.py
          $<TARGET_FILE:llk-tune> $<TARGET_FILE:llk-validate> ${CMAKE_SOURCE_DIR}
          ${CMAKE_BINARY_DIR}/prediction-validation-evidence)
endif()
