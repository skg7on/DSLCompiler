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
