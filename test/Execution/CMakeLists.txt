# LLVM permits these sources to be omitted when JIT tests are disabled.
set(LLVM_OPTIONAL_SOURCES
      swiglu_scalar.cpp
      triton_swiglu_staged.cpp
      swiglu_vector_avx2.cpp
      mapped_acceptance.cpp
      micro_kernel_execution.cpp
      swiglu_fused_memory.cpp)

if(LLK_BUILD_E2E_TESTS)
  set(_llk_execution_e2e_sources ${LLVM_OPTIONAL_SOURCES})
  set(_llk_execution_e2e_libs
      LLKPasses LLKTransforms LLKFuseDoubleContraction LLKPackWeights
      LLKToMicro LLKMicroToLinalg LLKMappedCompilation LLKMapping
      LLKTargetMapping MicroDialect LLKDialect TritonToLLKPasses
      ${LLK_MLIR_EXEC_LIBS})
endif()

add_llk_test_category(Execution
    PARTIAL_SOURCES_INTENDED
    SOURCES
        swiglu_reference.cpp
        edge_cases.cpp
        rope_correctness.cpp
        attention_correctness.cpp
        attention_stability.cpp
        shared_pipeline.cpp
        cpu_features_test.cpp
        packed_vs_unpacked.cpp
        specialization_dispatch.cpp
        schedule_determinism.cpp
        affinity_test.cpp
        jit_cache_miss.cpp
        thread_pool_stress.cpp
        swiglu_parallel_scaling.cpp
        swiglu_small_shape.cpp
        jit_cache_hit.cpp
        cache_eviction.cpp
        ${_llk_execution_e2e_sources}
    LIBRARIES
        LLKTargetX86
        LLKRuntime
        ${_llk_execution_e2e_libs}
    DEFINITIONS
        LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
)

# Separate mapped binaries keep the portable and required-selected-target
# contracts independently invokable by CI. Place the acceptance executable at
# the build root because the pinned workflow runs it directly from build/.
if(LLK_BUILD_E2E_TESTS)
  set(_llk_mapped_libs
      LLKMappedCompilation LLKToMicro LLKMicroToLinalg LLKMapping
      LLKTargetMapping MicroDialect LLKDialect GTest::gtest_main
      ${LLK_MLIR_EXEC_LIBS})

  add_llvm_executable(MappedAcceptance mapped_acceptance.cpp
      PARTIAL_SOURCES_INTENDED)
  target_link_libraries(MappedAcceptance PRIVATE ${_llk_mapped_libs})
  target_compile_definitions(MappedAcceptance PRIVATE
      LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  set_target_properties(MappedAcceptance PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_test(NAME MappedAcceptance COMMAND MappedAcceptance)
  set_tests_properties(MappedAcceptance PROPERTIES
      LABELS "mapped-reference"
      SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]")

  add_llvm_executable(MappedAVX2Acceptance mapped_acceptance.cpp
      PARTIAL_SOURCES_INTENDED)
  target_link_libraries(MappedAVX2Acceptance PRIVATE ${_llk_mapped_libs})
  target_compile_definitions(MappedAVX2Acceptance PRIVATE
      LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}" LLK_REQUIRE_SELECTED_TARGET=1)
  set_target_properties(MappedAVX2Acceptance PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_test(NAME MappedAVX2Acceptance COMMAND MappedAVX2Acceptance
      --require-selected-target --gtest_filter=MappedAcceptance.*)
  set_tests_properties(MappedAVX2Acceptance PROPERTIES LABELS "selected-avx2")

  add_test(NAME SelectedTargetFlagContract
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/check_selected_target_flag.py
          $<TARGET_FILE:MappedAVX2Acceptance> $<TARGET_FILE:MappedAcceptance>)
  set_tests_properties(SelectedTargetFlagContract PROPERTIES LABELS "selected-avx2")

  add_llvm_executable(MappedInvocationTest mapped_invocation.cpp
      PARTIAL_SOURCES_INTENDED)
  target_link_libraries(MappedInvocationTest PRIVATE LLKRuntime GTest::gtest_main
      ${LLK_MLIR_EXEC_LIBS})
  add_test(NAME MappedInvocationTest COMMAND MappedInvocationTest)

  add_llvm_executable(MappedAllocationLifetimeTest mapped_allocation_lifetime.cpp
      PARTIAL_SOURCES_INTENDED)
  target_link_libraries(MappedAllocationLifetimeTest PRIVATE LLKRuntime
      GTest::gtest_main ${LLK_MLIR_EXEC_LIBS})
  target_compile_definitions(MappedAllocationLifetimeTest PRIVATE
      LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  add_test(NAME MappedAllocationLifetimeTest COMMAND MappedAllocationLifetimeTest)
  set_tests_properties(MappedAllocationLifetimeTest PROPERTIES
      LABELS "mapped-reference")

  add_llvm_executable(MappedCodegenRequirementsTest
      mapped_codegen_requirements.cpp PARTIAL_SOURCES_INTENDED)
  target_link_libraries(MappedCodegenRequirementsTest PRIVATE LLKRuntime
      GTest::gtest_main)
  add_test(NAME MappedCodegenRequirementsTest
      COMMAND MappedCodegenRequirementsTest)

  add_llvm_executable(SelectedGroupLoweringTest selected_group_lowering.cpp
      PARTIAL_SOURCES_INTENDED)
  target_link_libraries(SelectedGroupLoweringTest PRIVATE LLKMappedCompilation
      LLKMapping GTest::gtest_main ${LLK_MLIR_EXEC_LIBS})
  add_test(NAME SelectedGroupLoweringTest COMMAND SelectedGroupLoweringTest)

  add_test(NAME RequiredExecutionEvidenceParser
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/check_required_execution_test.py)
  set_tests_properties(RequiredExecutionEvidenceParser PROPERTIES
      LABELS "mapped-reference")
  add_test(NAME SelectedExecutionEvidenceValidator
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/check_selected_execution_evidence.py
          --self-test)
  set_tests_properties(SelectedExecutionEvidenceValidator PROPERTIES
      LABELS "selected-avx2")
endif()
