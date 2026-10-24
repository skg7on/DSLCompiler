# Keep native execution checks in the E2E lane; schema tests always build.
find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
if(LLK_BUILD_E2E_TESTS)
  set(_llk_frontend_execution_libs LLKRuntime)
endif()
set(_llk_capture_dir "${CMAKE_CURRENT_BINARY_DIR}/captured")
file(GLOB_RECURSE _llk_capture_sources CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/python/llk/*.py")
add_custom_command(
    OUTPUT "${_llk_capture_dir}/add.json" "${_llk_capture_dir}/mul.json"
           "${_llk_capture_dir}/loop.json"
    COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${PROJECT_SOURCE_DIR}/python"
            ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/capture_fixture.py"
            "${_llk_capture_dir}"
    DEPENDS capture_fixture.py ${_llk_capture_sources}
    VERBATIM)
add_custom_target(LLKFrontendCaptureFixtures DEPENDS
    "${_llk_capture_dir}/add.json" "${_llk_capture_dir}/mul.json"
    "${_llk_capture_dir}/loop.json")
add_llk_test_category(Frontend
    SOURCES legacy_typed_contract.cpp program.cpp importer.cpp compiler.cpp
    LIBRARIES LLKFrontend TritonToLLKPasses LLKMappedTuning LLKTargetMapping ${_llk_frontend_execution_libs} LLK::MLIR
    DEFINITIONS LLK_FRONTEND_ENABLE_EXECUTION=$<BOOL:${LLK_BUILD_E2E_TESTS}>
                LLK_FRONTEND_FIXTURE="${PROJECT_SOURCE_DIR}/test/python/frontend_add_v1.json"
                LLK_FRONTEND_CAPTURE_DIR="${_llk_capture_dir}"
                LLK_FRONTEND_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)
add_dependencies(LLKFrontendTests LLKFrontendCaptureFixtures)
