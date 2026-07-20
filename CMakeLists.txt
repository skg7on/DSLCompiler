cmake_minimum_required(VERSION 3.20)
project(llk-compiler LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(LLK_BUILD_TOOLS "Build llk-opt and llk-compile tools" ON)
option(LLK_BUILD_E2E_TESTS "Build end-to-end JIT tests (requires LLVMOrcJIT)" ON)

# Path to llvm-project build directory.
# All other LLVM/MLIR paths are inferred from this.
# When set, find_package searches ONLY this directory (not system paths).
set(LLVM_PROJECT_BUILD_DIR "" CACHE PATH "Path to llvm-project build directory")
if(LLVM_PROJECT_BUILD_DIR)
  find_package(MLIR REQUIRED CONFIG NO_DEFAULT_PATH
    PATHS "${LLVM_PROJECT_BUILD_DIR}/lib/cmake/mlir")
  find_package(LLVM REQUIRED CONFIG NO_DEFAULT_PATH
    PATHS "${LLVM_PROJECT_BUILD_DIR}/lib/cmake/llvm")
else()
  find_package(MLIR REQUIRED CONFIG)
  find_package(LLVM REQUIRED CONFIG)
endif()

message(STATUS "LLVM version: ${LLVM_PACKAGE_VERSION}")
message(STATUS "MLIR include dir: ${MLIR_INCLUDE_DIRS}")

# Ensure MLIR/LLVM include dirs are available for tablegen and compilation
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})

# Add project source and binary dirs for headers and generated .inc files
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_BINARY_DIR}/include)

# Set up LLVM/MLIR tablegen macros
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# TableGen for LLK dialect
set(LLVM_TARGET_DEFINITIONS include/LLK/Dialect/LLKDialect.td)
mlir_tablegen(include/LLK/Dialect/LLKDialect.h.inc -gen-dialect-decls -dialect=llk)
mlir_tablegen(include/LLK/Dialect/LLKDialect.cpp.inc -gen-dialect-defs -dialect=llk)
mlir_tablegen(include/LLK/Dialect/LLKAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=llk)
mlir_tablegen(include/LLK/Dialect/LLKAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=llk)
add_public_tablegen_target(LLKDialectGen)

set(LLVM_TARGET_DEFINITIONS include/LLK/Dialect/LLKOps.td)
mlir_tablegen(include/LLK/Dialect/LLKOps.h.inc -gen-op-decls
  EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include/LLK/Dialect)
mlir_tablegen(include/LLK/Dialect/LLKOps.cpp.inc -gen-op-defs
  EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include/LLK/Dialect)
mlir_tablegen(include/LLK/Dialect/LLKOpsDialect.h.inc -gen-dialect-decls -dialect=llk
  EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include/LLK/Dialect)
mlir_tablegen(include/LLK/Dialect/LLKOpsDialect.cpp.inc -gen-dialect-defs -dialect=llk
  EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include/LLK/Dialect)
add_public_tablegen_target(LLKOpsGen)

# TableGen for LLK transform passes
set(LLVM_TARGET_DEFINITIONS include/LLK/Transforms/Passes.td)
mlir_tablegen(include/LLK/Transforms/Passes.h.inc -gen-pass-decls -name=LLK)
add_public_tablegen_target(LLKTransformsGen)

# Library: LLK dialect
add_mlir_dialect_library(LLKDialect
    lib/Dialect/LLK/LLKDialect.cpp
    lib/Dialect/LLK/LLKOps.cpp
    DEPENDS LLKDialectGen LLKOpsGen
    LINK_LIBS PUBLIC MLIRIR MLIRDialect MLIRSupport MLIRArithDialect
                     MLIRTensorDialect
)

# Library: LLKToLinalg conversion
add_mlir_library(LLKToLinalg
    lib/Conversion/LLKToLinalg/LLKToLinalg.cpp
    DEPENDS LLKDialect
    LINK_LIBS PUBLIC LLKDialect MLIRLinalgDialect MLIRArithDialect MLIRMathDialect
                     MLIRFuncDialect MLIRTensorDialect MLIRPass MLIRTransforms
)

# Library: JIT cache runtime
add_library(LLKRuntime STATIC
    runtime/JitCache.cpp
)
target_include_directories(LLKRuntime PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${LLVM_INCLUDE_DIRS}
)
target_link_libraries(LLKRuntime PUBLIC
    LLVMOrcJIT MLIRExecutionEngine
    LLVMAArch64CodeGen LLVMAArch64Info LLVMAArch64Desc
    LLVMAArch64AsmParser LLVMAArch64Disassembler
)

# Library: X86 target feature detection
add_library(LLKTargetX86 STATIC
    lib/Target/X86/TargetAVX2.cpp
)
target_include_directories(LLKTargetX86 PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Library: LLK Transforms (TileAndVectorize)
add_mlir_library(LLKTransforms
    lib/Transforms/TileAndVectorize.cpp
    DEPENDS LLKDialect LLKToLinalg LLKTransformsGen
    LINK_LIBS PUBLIC LLKToLinalg MLIRLinalgDialect MLIRLinalgTransforms
                     MLIRVectorDialect MLIRSCFDialect MLIRSCFTransforms
                     MLIRFuncDialect MLIRTensorDialect MLIRArithDialect MLIRPass
)

# Library: FuseDoubleContraction
add_mlir_library(LLKFuseDoubleContraction
    lib/Transforms/FuseDoubleContraction.cpp
    DEPENDS LLKDialect LLKToLinalg
    LINK_LIBS PUBLIC LLKToLinalg MLIRLinalgDialect MLIRArithDialect MLIRMathDialect
                     MLIRFuncDialect MLIRTensorDialect MLIRPass MLIRTransforms
)

# Tool: llk-opt
add_mlir_library(LLKPasses
    lib/Conversion/LLKToLinalg/LLKToLinalg.cpp
    DEPENDS LLKDialect
    LINK_LIBS PUBLIC LLKToLinalg
)

# MLIR libraries needed by executables that use our project libraries.
# Listed explicitly rather than using the global MLIR_DIALECT_LIBS /
# MLIR_CONVERSION_LIBS to avoid linking 200+ libraries when using
# pre-built LLVM binary releases (which may have missing libs).
set(LLK_MLIR_EXEC_LIBS
    MLIRArithDialect
    MLIRArithToLLVM
    MLIRArithTransforms
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
    MLIRControlFlowToLLVM
    MLIRFuncDialect
    MLIRFuncToLLVM
    MLIRFuncTransforms
    MLIRIR
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRMathDialect
    MLIRMathToLLVM
    MLIRMemRefDialect
    MLIRMemRefToLLVM
    MLIRParser
    MLIRPass
    MLIRReconcileUnrealizedCasts
    MLIRSCFDialect
    MLIRSCFToControlFlow
    MLIRSCFTransforms
    MLIRSupport
    MLIRTensorDialect
    MLIRTensorTransforms
    MLIRTransforms
    MLIRVectorToLLVMPass
)

if(LLK_BUILD_TOOLS)
add_llvm_executable(llk-opt tools/llk-opt/llk-opt.cpp)
target_link_libraries(llk-opt PRIVATE LLKPasses LLKTransforms LLKFuseDoubleContraction MLIRMlirOptMain ${LLK_MLIR_EXEC_LIBS})

# Tool: llk-compile
add_llvm_executable(llk-compile tools/llk-compile/llk-compile.cpp)
target_link_libraries(llk-compile PRIVATE LLKPasses LLKRuntime ${LLK_MLIR_EXEC_LIBS})
endif()

# Tests
include(CTest)
find_package(GTest REQUIRED)
enable_testing()

function(add_llk_test test_name test_file)
    add_llvm_executable(${test_name} ${test_file})
    target_link_libraries(${test_name} PRIVATE GTest::gtest_main)
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

add_llk_test(SwigluReference test/Execution/swiglu_reference.cpp)

if(LLK_BUILD_E2E_TESTS)
add_llvm_executable(SwigluScalar test/Execution/swiglu_scalar.cpp)
target_link_libraries(SwigluScalar PRIVATE LLKPasses LLKRuntime GTest::gtest_main ${LLK_MLIR_EXEC_LIBS})
add_test(NAME SwigluScalar COMMAND SwigluScalar)

add_llvm_executable(SwiGLUVector test/Execution/swiglu_vector_avx2.cpp)
target_link_libraries(SwiGLUVector PRIVATE LLKPasses LLKTransforms LLKRuntime GTest::gtest_main ${LLK_MLIR_EXEC_LIBS})
add_test(NAME SwiGLUVector COMMAND SwiGLUVector)
endif()

add_llk_test(VectorVsScalar test/Numerical/vector_vs_scalar.cpp)
add_llk_test(EdgeCases test/Execution/edge_cases.cpp)
add_llk_test(CpuFeaturesTest test/Execution/cpu_features_test.cpp)
target_link_libraries(CpuFeaturesTest PRIVATE LLKTargetX86)

# FileCheck tests
find_program(FILECHECK_BIN FileCheck)
# llk-opt is built as part of this project; referenced at test time via $<TARGET_FILE>

function(add_llk_filecheck_test test_name input_file)
    add_test(NAME ${test_name}
        COMMAND ${FILECHECK_BIN} ${input_file} < ${input_file}
    )
    set_tests_properties(${test_name} PROPERTIES
        ENVIRONMENT "LLK_OPT=$<TARGET_FILE:llk-opt>"
    )
endfunction()

if(FILECHECK_BIN)
  add_llk_filecheck_test(DialectOps test/Dialect/ops.mlir)
  # LLKToLinalgConversion test: run llk-opt --llk-to-linalg then FileCheck
  add_test(NAME LLKToLinalgConversion
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg ${CMAKE_SOURCE_DIR}/test/Conversion/llk_to_linalg.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/llk_to_linalg.mlir"
  )
  # Schedule parse test: verify the transform dialect schedule parses correctly
  # and the named sequence round-trips through llk-opt.
  add_test(NAME ScheduleBF16Check
      COMMAND sh -c "$<TARGET_FILE:llk-opt> ${CMAKE_SOURCE_DIR}/test/Transforms/schedule_bf16_check.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/schedule_bf16_check.mlir"
  )
  # TileAndVectorize test: run llk-opt with LLK-to-Linalg + TileAndVectorize, then FileCheck
  add_test(NAME TileAndVectorize
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg --tile-and-vectorize \
        ${CMAKE_SOURCE_DIR}/test/Transforms/tile_and_vectorize.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/tile_and_vectorize.mlir"
  )
  # MaskGen test: verify vector.create_mask and vector.mask for N=127 tail
  add_test(NAME MaskGen
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg --tile-and-vectorize \
        ${CMAKE_SOURCE_DIR}/test/Transforms/mask_gen.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/mask_gen.mlir"
  )
  # FuseDoubleContraction test
  add_test(NAME FuseDoubleContraction
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg --fuse-double-contraction \
        ${CMAKE_SOURCE_DIR}/test/Transforms/fuse_double_contraction.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/fuse_double_contraction.mlir"
  )
  # FuseReject test: different X operands -> no fusion
  add_test(NAME FuseReject
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg --fuse-double-contraction \
        ${CMAKE_SOURCE_DIR}/test/Transforms/fuse_reject.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/fuse_reject.mlir"
  )
endif()

# Test for ops_invalid.mlir: runs llk-opt with --verify-diagnostics.
# Only registered when llk-opt is built (LLK_BUILD_TOOLS=ON).
if(LLK_BUILD_TOOLS)
add_test(NAME DialectOpsInvalid
    COMMAND $<TARGET_FILE:llk-opt> --verify-diagnostics --split-input-file
            ${CMAKE_SOURCE_DIR}/test/Dialect/ops_invalid.mlir
)
endif()
