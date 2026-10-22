# Keep native execution checks in the E2E lane; schema tests always build.
if(LLK_BUILD_E2E_TESTS)
  set(_llk_frontend_execution_libs LLKRuntime)
endif()
add_llk_test_category(Frontend
    SOURCES legacy_typed_contract.cpp
    LIBRARIES TritonToLLKPasses ${_llk_frontend_execution_libs} LLK::MLIR
    DEFINITIONS LLK_FRONTEND_ENABLE_EXECUTION=$<BOOL:${LLK_BUILD_E2E_TESTS}>
)
