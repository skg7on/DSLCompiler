# A frontend request carries its frozen target configuration inside the
# canonical JSON, so the C consumer needs a self-contained fixture rather than a
# path into the checkout. Materialize it from the PF03 serialized fixture.
find_package(Python3 3.11 REQUIRED COMPONENTS Interpreter)
set(_llk_capi_fixture "${CMAKE_CURRENT_BINARY_DIR}/frontend_add_request.json")
add_custom_command(
    OUTPUT "${_llk_capi_fixture}"
    COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/make_fixture.py"
            "${PROJECT_SOURCE_DIR}" "${_llk_capi_fixture}"
    DEPENDS make_fixture.py
            "${PROJECT_SOURCE_DIR}/test/python/frontend_add_v1.json"
    VERBATIM)
add_custom_target(LLKCAPIFrontendFixtures DEPENDS "${_llk_capi_fixture}")

# Plain C consumer: the ABI must be usable without C++ or LLVM headers.
add_executable(LLKCAPIFrontendTests frontend.c)
target_link_libraries(LLKCAPIFrontendTests PRIVATE LLKFrontendCAPI)
target_compile_definitions(LLKCAPIFrontendTests PRIVATE
    LLK_CAPI_FIXTURE="${_llk_capi_fixture}")
add_dependencies(LLKCAPIFrontendTests LLKCAPIFrontendFixtures)
add_test(NAME FrontendCAPI.Consumer COMMAND LLKCAPIFrontendTests)
set_tests_properties(FrontendCAPI.Consumer PROPERTIES
    LABELS "CAPI" TIMEOUT 120 WORKING_DIRECTORY "${CMAKE_BINARY_DIR}")

# C++ consumers cover the error, ownership and concurrency contract; the suite
# names start with FrontendCAPI so -R FrontendCAPI selects every case.
add_llk_test_category(CAPI
    SOURCES frontend_errors.cpp
    LIBRARIES LLKFrontendCAPI
    DEFINITIONS LLK_CAPI_FIXTURE="${_llk_capi_fixture}"
)
add_dependencies(LLKCAPITests LLKCAPIFrontendFixtures)
