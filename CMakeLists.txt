cmake_minimum_required(VERSION 3.20)

# Static libraries may repeat on link lines so traditional one-pass linkers
# (GNU ld) can resolve them. Apple's ld64 and LLD record every symbol and warn
# about the duplicates, so let CMake apply linker-capability-aware
# de-duplication. The guards keep older CMake versions on their legacy behavior.
if(POLICY CMP0156)
  cmake_policy(SET CMP0156 NEW)
endif()
if(POLICY CMP0179)
  cmake_policy(SET CMP0179 NEW)
endif()

# Compiler selection must happen before project() enables the languages.
include(cmake/LLKCompiler.cmake)
project(llk-compiler VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(CMAKE_EXPORT_COMPILE_COMMANDS STREQUAL "")
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL
      "Enable/Disable output of compile commands during generation" FORCE)
endif()

option(LLK_BUILD_TOOLS "Build the LLK command-line tools" ON)
option(LLK_BUILD_E2E_TESTS "Build end-to-end JIT tests" ON)

include(cmake/LLKDependencies.cmake)
include(cmake/LLKBuildOptions.cmake)
include(cmake/LLKVersion.cmake)
include(cmake/LLKTableGen.cmake)

add_subdirectory(lib)
add_subdirectory(runtime)
if(LLK_BUILD_TOOLS)
  add_subdirectory(tools)
endif()

# CTest must be enabled at the root for ctest to find the entire suite.
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(third_party)
  add_subdirectory(test)
endif()

# Documentation is opt-in; compiler builds do not require Sphinx.
set(LLK_DOCS_PYTHON "python3" CACHE STRING "Python interpreter for the Sphinx manual")
add_custom_target(docs-html
    COMMAND ${LLK_DOCS_PYTHON} -m sphinx -b html -n -W --keep-going
            -d ${PROJECT_BINARY_DIR}/docs/doctrees
            ${PROJECT_SOURCE_DIR}/docs/manual/source ${PROJECT_BINARY_DIR}/docs/html
    USES_TERMINAL
    VERBATIM
    COMMENT "Building the MicroIR HTML developer manual"
)
