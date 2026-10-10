# One GTest executable per category; CTest still runs each case separately.
include(GoogleTest)

function(add_llk_test_category category)
  cmake_parse_arguments(ARG "" "" "SOURCES;LIBRARIES;DEFINITIONS;INCLUDE_DIRECTORIES" ${ARGN})
  set(test_target "LLK${category}Tests")
  add_llvm_executable(${test_target} ${ARG_SOURCES})
  target_link_libraries(${test_target} PRIVATE GTest::gtest_main ${ARG_LIBRARIES})
  if(ARG_DEFINITIONS)
    target_compile_definitions(${test_target} PRIVATE ${ARG_DEFINITIONS})
  endif()
  if(ARG_INCLUDE_DIRECTORIES)
    target_include_directories(${test_target} PRIVATE ${ARG_INCLUDE_DIRECTORIES})
  endif()
  # Runtime discovery handles parameterized and disabled tests and avoids
  # guessing GTest names from source text. Preserve the old working directory.
  gtest_discover_tests(${test_target}
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
      PROPERTIES LABELS "${category}"
                 TIMEOUT 60
                 SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]")
  set_property(GLOBAL APPEND PROPERTY LLK_TEST_TARGETS ${test_target})
endfunction()

add_subdirectory(Execution)
add_subdirectory(Numerical)
add_subdirectory(Transforms)
add_subdirectory(Perf)
add_subdirectory(Machine)
add_subdirectory(Mapping)
add_subdirectory(Conversion)
add_subdirectory(Benchmark)

include(FileCheck.cmake)

# Tests registered here are the IR/CLI tests; GTests are discovered after build.
get_property(_llk_ir_tests DIRECTORY PROPERTY TESTS)
if(_llk_ir_tests)
  set_tests_properties(${_llk_ir_tests} PROPERTIES
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
      SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]")
endif()

get_property(_llk_check_deps GLOBAL PROPERTY LLK_TEST_TARGETS)
foreach(_llk_tool llk-opt micro-perf llk-tune llk-compile llk-bench)
  if(TARGET ${_llk_tool})
    list(APPEND _llk_check_deps ${_llk_tool})
  endif()
endforeach()
# Multi-config add_test entries require an active configuration. A single-
# config tree may have no build type, in which case omit the -C argument.
set(_llk_ctest_config_args)
if(CMAKE_CONFIGURATION_TYPES OR CMAKE_BUILD_TYPE)
  set(_llk_ctest_config_args -C "$<CONFIG>")
endif()
add_custom_target(check-llk
    COMMAND ${CMAKE_CTEST_COMMAND} ${_llk_ctest_config_args} --output-on-failure
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    DEPENDS ${_llk_check_deps}
    VERBATIM
    USES_TERMINAL
    COMMENT "Running the LLK test suite"
)
