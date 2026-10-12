# One LLVM build supplies the packages, source/generated headers, tools and
# libraries. Source headers may live beside the build tree; use the paths
# exported by its packages rather than guessing an llvm-project source layout.
set(LLVM_PROJECT_BUILD_DIR "" CACHE PATH "Path to the llvm-project build directory")
if(NOT LLVM_PROJECT_BUILD_DIR)
  message(FATAL_ERROR
      "LLVM_PROJECT_BUILD_DIR is required. Configure with "
      "-DLLVM_PROJECT_BUILD_DIR=/absolute/path/to/llvm-project/build.")
endif()
get_filename_component(LLVM_PROJECT_BUILD_DIR "${LLVM_PROJECT_BUILD_DIR}"
    REALPATH BASE_DIR "${PROJECT_SOURCE_DIR}")
set(LLVM_PROJECT_BUILD_DIR "${LLVM_PROJECT_BUILD_DIR}" CACHE PATH
    "Path to the llvm-project build directory" FORCE)

# FORCE also replaces stale package locations when switching LLVM builds in
# an existing CMake cache. NO_DEFAULT_PATH alone does not override LLVM_DIR.
foreach(_package IN ITEMS LLVM MLIR)
  string(TOLOWER "${_package}" _directory)
  set(${_package}_DIR "${LLVM_PROJECT_BUILD_DIR}/lib/cmake/${_directory}"
      CACHE PATH "${_package} package from LLVM_PROJECT_BUILD_DIR" FORCE)
  if(NOT EXISTS "${${_package}_DIR}/${_package}Config.cmake")
    message(FATAL_ERROR
        "LLVM_PROJECT_BUILD_DIR (${LLVM_PROJECT_BUILD_DIR}) must contain "
        "lib/cmake/${_directory}/${_package}Config.cmake. Build LLVM with "
        "-DLLVM_ENABLE_PROJECTS=mlir.")
  endif()
endforeach()

include("${CMAKE_CURRENT_LIST_DIR}/LLKImportedTargets.cmake")

find_package(LLVM REQUIRED CONFIG PATHS "${LLVM_DIR}" NO_DEFAULT_PATH)
llk_require_build_target(LLVM SHARED_LIBRARY)
if(NOT LLVM_LINK_LLVM_DYLIB OR NOT LLVM_DYLIB_COMPONENTS STREQUAL "all")
  message(FATAL_ERROR "LLVM must export shared linkage for all components. "
      "Rebuild with -DLLVM_LINK_LLVM_DYLIB=ON -DLLVM_DYLIB_COMPONENTS=all.")
endif()

# LLVM is loaded first so MLIR's nested find_package(LLVM) uses this build.
find_package(MLIR REQUIRED CONFIG PATHS "${MLIR_DIR}" NO_DEFAULT_PATH)
llk_require_build_target(MLIR SHARED_LIBRARY)
if(NOT MLIR_LINK_MLIR_DYLIB)
  message(FATAL_ERROR "MLIR must export shared linkage. "
      "Rebuild LLVM with -DMLIR_LINK_MLIR_DYLIB=ON.")
endif()
llk_require_build_target(mlir_c_runner_utils SHARED_LIBRARY)

# Use the exported executable so CMake resolves the tool for each build
# configuration. LLVM multi-config builds keep tools in e.g. Release/bin.
llk_require_build_target(mlir-tblgen EXECUTABLE _llk_mlir_tablegen)
set(MLIR_TABLEGEN_EXE mlir-tblgen)
get_filename_component(LLVM_TOOLS_BINARY_DIR "${_llk_mlir_tablegen}" DIRECTORY)

list(PREPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}" "${MLIR_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# SDK root includes sort before libc++ when propagated explicitly by these
# system dependencies. The compiler already searches them via its sysroot.
if(APPLE)
  foreach(_dependency IN ITEMS ZLIB::ZLIB LibEdit::LibEdit)
    if(TARGET ${_dependency})
      get_target_property(_includes ${_dependency} INTERFACE_INCLUDE_DIRECTORIES)
      if(_includes)
        list(FILTER _includes EXCLUDE REGEX "^/.*[.]sdk/usr/include$")
        set_property(TARGET ${_dependency} PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${_includes}")
      endif()
    endif()
  endforeach()
endif()

message(STATUS "LLVM build: ${LLVM_PROJECT_BUILD_DIR} (${LLVM_PACKAGE_VERSION})")
message(STATUS "LLVM/MLIR linkage: shared LLVM + MLIR")
