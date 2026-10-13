# TableGen reads directory includes; compilers use the same package paths.
include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS}
    "${PROJECT_SOURCE_DIR}/include" "${PROJECT_BINARY_DIR}/include")
add_definitions(${LLVM_DEFINITIONS})

# Match the LLVM build's ABI for every project library and its consumers.
add_library(LLKBuildOptions INTERFACE)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(LLKBuildOptions INTERFACE -Werror=deprecated-declarations)
  if(NOT LLVM_ENABLE_RTTI)
    target_compile_options(LLKBuildOptions INTERFACE -fno-rtti)
  endif()
  if(NOT LLVM_ENABLE_EH)
    target_compile_options(LLKBuildOptions INTERFACE -fno-exceptions)
  endif()
endif()

# MLIR's shared target does not always export LLVM as a public dependency.
# Consumers also call LLVM APIs directly, so put both shared libraries on the
# link line (required by ELF linkers and macOS's two-level namespace).
add_library(LLKMLIR INTERFACE)
add_library(LLK::MLIR ALIAS LLKMLIR)
target_link_libraries(LLKMLIR INTERFACE MLIR LLVM)

# Project libraries are static; their LLVM/MLIR dependencies are shared.
# DEPENDS is for generated headers. LINK_LIBRARIES expresses the public graph.
function(add_llk_library target)
  cmake_parse_arguments(ARG "" "" "DEPENDS;LINK_LIBRARIES" ${ARGN})
  add_library(${target} STATIC ${ARG_UNPARSED_ARGUMENTS})
  target_link_libraries(${target} PUBLIC LLKBuildOptions ${ARG_LINK_LIBRARIES})
  if(ARG_DEPENDS)
    add_dependencies(${target} ${ARG_DEPENDS})
  endif()
endfunction()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
foreach(_config IN LISTS CMAKE_CONFIGURATION_TYPES)
  string(TOUPPER "${_config}" _config)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_config} "${PROJECT_BINARY_DIR}/bin")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_config} "${PROJECT_BINARY_DIR}/lib")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${_config} "${PROJECT_BINARY_DIR}/lib")
endforeach()
