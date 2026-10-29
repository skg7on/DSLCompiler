# These sources are compiled only when JIT tests are enabled.
set(_llk_execution_jit_sources
      swiglu_scalar.cpp
      triton_swiglu_staged.cpp
      swiglu_vector_avx2.cpp
      micro_kernel_execution.cpp
      swiglu_fused_memory.cpp)

if(LLK_BUILD_E2E_TESTS)
  set(_llk_execution_e2e_sources ${_llk_execution_jit_sources})
  set(_llk_execution_e2e_libs
      LLKToLinalg LLKTransforms LLKFuseDoubleContraction LLKPackWeights
      LLKToMicro LLKMicroToLinalg LLKMappedCompilation LLKMapping
      LLKTargetMapping MicroDialect LLKDialect TritonToLLKPasses
      LLK::MLIR)
endif()

add_llk_test_category(Execution
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
      LLK::MLIR)

  add_executable(MappedAcceptance mapped_acceptance.cpp)
  target_link_libraries(MappedAcceptance PRIVATE ${_llk_mapped_libs})
  target_compile_definitions(MappedAcceptance PRIVATE
      LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  set_target_properties(MappedAcceptance PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
  add_test(NAME MappedAcceptance COMMAND MappedAcceptance)
  set_tests_properties(MappedAcceptance PROPERTIES
      LABELS "mapped-reference"
      SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]")

  add_executable(MappedAVX2Acceptance mapped_acceptance.cpp)
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

  add_executable(MappedInvocationTest mapped_invocation.cpp)
  target_link_libraries(MappedInvocationTest PRIVATE LLKRuntime GTest::gtest_main
      LLK::MLIR)
  add_test(NAME MappedInvocationTest COMMAND MappedInvocationTest)

  add_executable(MappedAllocationLifetimeTest mapped_allocation_lifetime.cpp)
  target_link_libraries(MappedAllocationLifetimeTest PRIVATE LLKRuntime
      LLKMappedCompilation LLKWorkloadContracts MicroDialect LLKDialect
      GTest::gtest_main LLK::MLIR)
  target_compile_definitions(MappedAllocationLifetimeTest PRIVATE
      LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  add_test(NAME MappedAllocationLifetimeTest COMMAND MappedAllocationLifetimeTest)
  set_tests_properties(MappedAllocationLifetimeTest PROPERTIES
      LABELS "mapped-reference")

  add_executable(MappedCodegenRequirementsTest
      mapped_codegen_requirements.cpp)
  target_link_libraries(MappedCodegenRequirementsTest PRIVATE LLKRuntime
      GTest::gtest_main)
  add_test(NAME MappedCodegenRequirementsTest
      COMMAND MappedCodegenRequirementsTest)

  set(_llk_generic_elementwise_libs
      LLKMappedCompilation LLKMappedTuning LLKSourceCandidates
      LLKWorkloadContracts LLKMapping LLKTargetMapping MicroDialect LLKDialect
      GTest::gtest LLK::MLIR)
  add_executable(GenericElementwiseTest generic_elementwise.cpp)
  target_link_libraries(GenericElementwiseTest PRIVATE
      ${_llk_generic_elementwise_libs})
  target_compile_definitions(GenericElementwiseTest PRIVATE
      LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  add_test(NAME GenericElementwiseTest COMMAND GenericElementwiseTest)
  set_tests_properties(GenericElementwiseTest PROPERTIES
      LABELS "mapped-reference")

  # The required-selected-target binary makes a missing AVX2 lane a failure,
  # not a skip, so CI evidence cannot silently pass on the wrong host.
  add_executable(GenericElementwiseAVX2Test generic_elementwise.cpp)
  target_link_libraries(GenericElementwiseAVX2Test PRIVATE
      ${_llk_generic_elementwise_libs})
  target_compile_definitions(GenericElementwiseAVX2Test PRIVATE
      LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}" LLK_REQUIRE_SELECTED_TARGET=1)
  add_test(NAME GenericElementwiseAVX2Test COMMAND GenericElementwiseAVX2Test
      --require-selected-target
      --gtest_filter=GenericElementwise.SelectedAvx2*)
  set_tests_properties(GenericElementwiseAVX2Test PROPERTIES
      LABELS "selected-avx2")

  set(_llk_framework_primitives_libs
      LLKMappedCompilation LLKMapping LLKTargetMapping LLKPerf MicroDialect
      LLKDialect GTest::gtest LLK::MLIR)
  add_executable(FrameworkPrimitivesTest framework_primitives.cpp)
  target_link_libraries(FrameworkPrimitivesTest PRIVATE
      ${_llk_framework_primitives_libs})
  target_compile_definitions(FrameworkPrimitivesTest PRIVATE
      LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  target_include_directories(FrameworkPrimitivesTest PRIVATE
      ${CMAKE_SOURCE_DIR}/lib/Conversion/MicroMapping)
  add_test(NAME FrameworkPrimitivesTest COMMAND FrameworkPrimitivesTest)
  set_tests_properties(FrameworkPrimitivesTest PROPERTIES
      LABELS "mapped-reference")

  add_executable(FrameworkPrimitivesAVX2Test framework_primitives.cpp)
  target_link_libraries(FrameworkPrimitivesAVX2Test PRIVATE
      ${_llk_framework_primitives_libs})
  target_compile_definitions(FrameworkPrimitivesAVX2Test PRIVATE
      LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}" LLK_REQUIRE_SELECTED_TARGET=1)
  target_include_directories(FrameworkPrimitivesAVX2Test PRIVATE
      ${CMAKE_SOURCE_DIR}/lib/Conversion/MicroMapping)
  add_test(NAME FrameworkPrimitivesAVX2Test COMMAND FrameworkPrimitivesAVX2Test
      --require-selected-target
      --gtest_filter=FrameworkPrimitives.SelectedAvx2*)
  set_tests_properties(FrameworkPrimitivesAVX2Test PROPERTIES
      LABELS "selected-avx2")

  add_executable(SelectedGroupLoweringTest selected_group_lowering.cpp)
  target_link_libraries(SelectedGroupLoweringTest PRIVATE LLKMappedCompilation
      LLKMapping GTest::gtest_main LLK::MLIR)
  add_test(NAME SelectedGroupLoweringTest COMMAND SelectedGroupLoweringTest)

  add_test(NAME RequiredExecutionEvidenceParser
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/check_required_execution_test.py)
  set_tests_properties(RequiredExecutionEvidenceParser PROPERTIES
      LABELS "mapped-reference")
  # The SVE artifact checks are pure text/hash predicates, so their regression
  # test needs neither qemu nor a cross toolchain and runs on every host.
  add_test(NAME ArmSVEEvidenceParser
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/arm_sve_evidence_test.py)
  set_tests_properties(ArmSVEEvidenceParser PROPERTIES
      LABELS "mapped-reference")
  add_test(NAME ArmSMEEvidenceParser
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/arm_sme_evidence_test.py)
  set_tests_properties(ArmSMEEvidenceParser PROPERTIES LABELS "mapped-reference")
  add_test(NAME SelectedExecutionEvidenceValidator
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/check_selected_execution_evidence.py
          --self-test)
  set_tests_properties(SelectedExecutionEvidenceValidator PROPERTIES
      LABELS "selected-avx2")
endif()

# ABI v2 and output-destination preparation are IR and descriptor contracts, so
# they are exercised without a JIT: descriptor resolution binds symbols before
# entry, and the prepared callable signature is inspected directly. Runtime
# allocation and ownership behavior stays in the JIT lifetime tests.
add_executable(DynamicMappedInvocationTest dynamic_mapped_invocation.cpp)
target_link_libraries(DynamicMappedInvocationTest PRIVATE LLKMappedCompilation
    LLKToMicro LLKTargetMapping LLKMapping LLKWorkloadContracts MicroDialect
    LLKDialect GTest::gtest_main LLK::MLIR)
target_compile_definitions(DynamicMappedInvocationTest PRIVATE
    LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME DynamicMappedInvocationTest COMMAND DynamicMappedInvocationTest)
set_tests_properties(DynamicMappedInvocationTest PROPERTIES LABELS "Execution")
