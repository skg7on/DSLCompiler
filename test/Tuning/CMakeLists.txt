# The framework source tests must run without an E2E/JIT guard.
add_executable(FrameworkSourceTest workload_contract.cpp
    generic_source_candidates.cpp source_adapter_registry.cpp
    elementwise_source_adapter.cpp measurement_store.cpp)
target_link_libraries(FrameworkSourceTest PRIVATE LLKWorkloadContracts
    LLKSourceCandidates LLKMappedTuning LLKMicroMapping LLKMicroToLinalg
    LLKDialect MicroDialect GTest::gtest_main LLK::MLIR)
target_compile_definitions(FrameworkSourceTest PRIVATE
    LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME FrameworkSourceTest COMMAND FrameworkSourceTest)

# Symbolic shape data, checked binding and the versioned workload contract.
# This target needs no JIT or E2E guard: it exercises data and MLIR attribute
# encoding only.
add_executable(DynamicShapeContractTest dynamic_shape_contract.cpp)
target_link_libraries(DynamicShapeContractTest PRIVATE LLKShape
    LLKWorkloadContracts LLKKernel LLKDialect MicroDialect
    GTest::gtest_main LLK::MLIR)
add_test(NAME DynamicShapeContractTest COMMAND DynamicShapeContractTest)

if(LLK_BUILD_E2E_TESTS)
  add_executable(MappedTuningSessionTest mapped_tuning_session.cpp)
  target_link_libraries(MappedTuningSessionTest PRIVATE LLKMappedTuning
      GTest::gtest_main LLK::MLIR)
  target_compile_definitions(MappedTuningSessionTest PRIVATE
      LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  add_test(NAME MappedTuningSessionTest COMMAND MappedTuningSessionTest)

  add_executable(CandidateInstantiationTest candidate_instantiation.cpp)
  target_link_libraries(CandidateInstantiationTest PRIVATE LLKToMicro
      LLKMicroMapping LLKDialect MicroDialect GTest::gtest_main
      LLK::MLIR)
  target_compile_definitions(CandidateInstantiationTest PRIVATE
      LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  add_test(NAME CandidateInstantiationTest COMMAND CandidateInstantiationTest)

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
  # Dynamic plans are ranked only on explicit concrete samples; the tool records
  # the profile and every observation, and `micro-perf` analyzes one sample
  # through the same specialization.
  add_test(NAME DynamicShapeProfileCLI
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/dynamic_profile_cli.py
          $<TARGET_FILE:llk-tune> $<TARGET_FILE:micro-perf> ${CMAKE_SOURCE_DIR})

  add_test(NAME PredictionValidationMappedCLI
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/prediction_validation_mapped_cli.py
          $<TARGET_FILE:llk-tune> $<TARGET_FILE:llk-validate> ${CMAKE_SOURCE_DIR}
          ${CMAKE_BINARY_DIR}/prediction-validation-evidence)
endif()
