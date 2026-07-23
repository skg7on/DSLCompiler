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

# Library: JIT cache runtime + weight packing
add_library(LLKRuntime STATIC
    runtime/JitCache.cpp
    runtime/PackedWeights.cpp
    runtime/ThreadPool.cpp
)
target_include_directories(LLKRuntime PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${LLVM_INCLUDE_DIRS}
)
# Resolve native target libraries dynamically for the build host.
# Equivalent to: llvm-config --libs native
# On x86_64 → LLVMX86{CodeGen,Info,Desc,AsmParser,AsmPrinter,Disassembler}
# On AArch64 → LLVMAArch64{CodeGen,Info,Desc,AsmParser,AsmPrinter,Disassembler}
llvm_map_components_to_libnames(LLVM_NATIVE_TARGET_LIBS native)
target_link_libraries(LLKRuntime PUBLIC
    LLVMOrcJIT MLIRExecutionEngine
    ${LLVM_NATIVE_TARGET_LIBS}
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

# Library: PackWeights weight annotation pass
add_mlir_library(LLKPackWeights
    lib/Transforms/PackWeights.cpp
    DEPENDS LLKDialect
    LINK_LIBS PUBLIC MLIRLinalgDialect MLIRFuncDialect MLIRTensorDialect MLIRPass
)

# Library: ScratchAnalysis memory auditing pass
add_mlir_library(LLKScratchAnalysis
    lib/Transforms/ScratchAnalysis.cpp
    LINK_LIBS PUBLIC MLIRMemRefDialect MLIRFuncDialect MLIRPass
)

# Library: LinearizeForall pass
add_mlir_library(LLKLinearizeForall
    lib/Transforms/LinearizeForall.cpp
    DEPENDS LLKDialect
    LINK_LIBS PUBLIC MLIRSCFDialect MLIRArithDialect MLIRFuncDialect MLIRTensorDialect MLIRPass
)

# Library: SerialParallelDispatch pass
add_mlir_library(LLKSerialParallelDispatch
    lib/Transforms/SerialParallelDispatch.cpp
    LINK_LIBS PUBLIC MLIRSCFDialect MLIRArithDialect MLIRFuncDialect MLIRPass
)

# Library: ForallToLLRT pass
add_mlir_library(LLKForallToLLRT
    lib/Transforms/ForallToLLRT.cpp
    LINK_LIBS PUBLIC MLIRSCFDialect MLIRArithDialect MLIRFuncDialect MLIRPass
)

# Library: ShapeSpecialization pass
add_mlir_library(LLKShapeSpecialization
    lib/Transforms/ShapeSpecialization.cpp
    DEPENDS LLKDialect
    LINK_LIBS PUBLIC MLIRFuncDialect MLIRPass
)

# Library: ScheduleSelection pass
add_mlir_library(LLKScheduleSelection
    lib/Transforms/ScheduleSelection.cpp
    DEPENDS LLKDialect
    LINK_LIBS PUBLIC MLIRFuncDialect MLIRPass
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
    MLIRRegisterAllPasses
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
target_link_libraries(llk-opt PRIVATE LLKPasses LLKTransforms LLKFuseDoubleContraction LLKPackWeights LLKScratchAnalysis LLKLinearizeForall LLKSerialParallelDispatch LLKForallToLLRT LLKShapeSpecialization LLKScheduleSelection MLIRMlirOptMain ${LLK_MLIR_EXEC_LIBS})

# Tool: llk-compile
add_llvm_executable(llk-compile tools/llk-compile/llk-compile.cpp)
target_link_libraries(llk-compile PRIVATE LLKPasses LLKRuntime ${LLK_MLIR_EXEC_LIBS})

# Tool: llk-bench (benchmarking harness)
add_llvm_executable(llk-bench tools/llk-bench/llk-bench.cpp)
target_link_libraries(llk-bench PRIVATE LLKRuntime)

# Tool: llk-tune (autotuning grid search)
add_llvm_executable(llk-tune tools/llk-tune/llk-tune.cpp)
target_link_libraries(llk-tune PRIVATE LLKRuntime)
endif()

# Tests
include(CTest)
enable_testing()

# Build GoogleTest from vendored source (never the system package) to avoid
# ABI incompatibilities with prebuilt system libgtest packages.
# googletest's cxx_library() macro auto-creates the GTest::gtest_main ALIAS.
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/googletest)

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

add_llvm_executable(SwiGLUFusedMemory test/Execution/swiglu_fused_memory.cpp)
target_link_libraries(SwiGLUFusedMemory PRIVATE LLKPasses LLKTransforms LLKFuseDoubleContraction LLKPackWeights LLKRuntime GTest::gtest_main ${LLK_MLIR_EXEC_LIBS})
add_test(NAME SwiGLUFusedMemory COMMAND SwiGLUFusedMemory)
endif()

add_llk_test(VectorVsScalar test/Numerical/vector_vs_scalar.cpp)
add_llk_test(EdgeCases test/Execution/edge_cases.cpp)
add_llk_test(CpuFeaturesTest test/Execution/cpu_features_test.cpp)
target_link_libraries(CpuFeaturesTest PRIVATE LLKTargetX86)

add_llvm_executable(ThreadPoolStress test/Execution/thread_pool_stress.cpp)
target_link_libraries(ThreadPoolStress PRIVATE LLKRuntime GTest::gtest_main)
add_test(NAME ThreadPoolStress COMMAND ThreadPoolStress)

add_llvm_executable(SwiGLUParallelScaling test/Execution/swiglu_parallel_scaling.cpp)
target_link_libraries(SwiGLUParallelScaling PRIVATE LLKRuntime GTest::gtest_main)
add_test(NAME SwiGLUParallelScaling COMMAND SwiGLUParallelScaling)

add_llvm_executable(SwiGLUSmallShape test/Execution/swiglu_small_shape.cpp)
target_link_libraries(SwiGLUSmallShape PRIVATE LLKRuntime GTest::gtest_main)
add_test(NAME SwiGLUSmallShape COMMAND SwiGLUSmallShape)

add_llvm_executable(JitCacheHit test/Execution/jit_cache_hit.cpp)
target_link_libraries(JitCacheHit PRIVATE LLKRuntime GTest::gtest_main)
add_test(NAME JitCacheHit COMMAND JitCacheHit)

# FileCheck tests
find_program(FILECHECK_BIN FileCheck)
# llk-opt is built as part of this project; referenced at test time via $<TARGET_FILE>

function(add_llk_filecheck_test test_name input_file)
    # Use sh -c to properly handle input redirection (CMake's add_test
    # COMMAND mode does not support < redirection natively).
    # Execute the RUN line: llk-opt %s | llk-opt | FileCheck %s
    # This round-trips the IR through llk-opt twice to verify parse+print stability.
    add_test(NAME ${test_name}
        COMMAND sh -c "$<TARGET_FILE:llk-opt> ${CMAKE_SOURCE_DIR}/${input_file} | $<TARGET_FILE:llk-opt> | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/${input_file}"
    )
endfunction()

if(FILECHECK_BIN)
  add_llk_filecheck_test(DialectOps test/Dialect/ops.mlir)
  # LLKToLinalgConversion test: run llk-opt --llk-to-linalg then FileCheck
  add_test(NAME LLKToLinalgConversion
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg ${CMAKE_SOURCE_DIR}/test/Conversion/llk_to_linalg.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/llk_to_linalg.mlir"
  )
  # RoPE lowering: verify llk.rope → cos/sin + even/odd interleave
  add_test(NAME RoPELowering
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg ${CMAKE_SOURCE_DIR}/test/Conversion/rope_lowering.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/rope_lowering.mlir"
  )
  # Attention lowering: verify llk.attention → Q@K^T + softmax + @V
  add_test(NAME AttentionLowering
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg ${CMAKE_SOURCE_DIR}/test/Conversion/attention_lowering.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/attention_lowering.mlir"
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
  # FuseIndependent test: independent swiglus with different X are each fused
  add_test(NAME FuseIndependent
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg --fuse-double-contraction \
        ${CMAKE_SOURCE_DIR}/test/Transforms/fuse_independent.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/fuse_independent.mlir"
  )
  # FuseReject test: two matmuls sharing X with non-SiLU consumer -> no fusion
  add_test(NAME FuseReject
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --fuse-double-contraction \
        ${CMAKE_SOURCE_DIR}/test/Transforms/fuse_reject.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/fuse_reject.mlir"
  )
  # PackWeights test: annotate weight operands with packed_layout attribute
  add_test(NAME PackWeights
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg --pack-weights \
        ${CMAKE_SOURCE_DIR}/test/Transforms/pack_weights.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/pack_weights.mlir"
  )
  # LinearizeForall test: multi-D scf.forall → 1D
  add_test(NAME LinearizeForall
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --linearize-forall \
        ${CMAKE_SOURCE_DIR}/test/Transforms/linearize_forall.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/linearize_forall.mlir"
  )
  # SerialParallelDispatch test: threshold-based serial/parallel
  add_test(NAME SerialDispatch
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --serial-parallel-dispatch \
        ${CMAKE_SOURCE_DIR}/test/Transforms/serial_dispatch.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/serial_dispatch.mlir"
  )
  # ForallToLLRT test: scf.forall → runtime call with worker outline
  add_test(NAME ForallToLLRT
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --forall-to-llrt \
        ${CMAKE_SOURCE_DIR}/test/Transforms/forall_to_llrt.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/forall_to_llrt.mlir"
  )
  # ShapeSpecialization test: check M bucket classification
  add_test(NAME ShapeSpecialization
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --shape-specialize \
        ${CMAKE_SOURCE_DIR}/test/Transforms/shape_specialization.mlir \
        2>&1 | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/shape_specialization.mlir"
  )
  # ScheduleSelection test: check correct schedule is chosen
  add_test(NAME ScheduleSelection
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --select-schedule \
        ${CMAKE_SOURCE_DIR}/test/Transforms/schedule_selection.mlir \
        2>&1 | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/schedule_selection.mlir"
  )
  # ScheduleFallback test: check fallback behavior for unknown (N,K)
  add_test(NAME ScheduleFallback
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --select-schedule \
        ${CMAKE_SOURCE_DIR}/test/Transforms/schedule_fallback.mlir \
        2>&1 | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/schedule_fallback.mlir"
  )
  # Online softmax: verify max reduction + exp + div pattern in attention lowering
  add_test(NAME OnlineSoftmax
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg ${CMAKE_SOURCE_DIR}/test/Transforms/online_softmax.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/online_softmax.mlir"
  )
  # RoPE vector shuffle: verify even/odd slice + interleave
  add_test(NAME RoPEVectorShuffle
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg ${CMAKE_SOURCE_DIR}/test/Transforms/rope_vector_shuffle.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/rope_vector_shuffle.mlir"
  )
  # Causal mask: verify cmp + select pattern for upper-triangular -inf
  add_test(NAME CausalMask
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg ${CMAKE_SOURCE_DIR}/test/Transforms/causal_mask.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/causal_mask.mlir"
  )
endif()

# Test for ops_invalid.mlir: runs llk-opt with --verify-diagnostics.
# Only registered when llk-opt is built (LLK_BUILD_TOOLS=ON).
if(LLK_BUILD_TOOLS)
add_test(NAME DialectOpsInvalid
    COMMAND $<TARGET_FILE:llk-opt> --verify-diagnostics --split-input-file
            ${CMAKE_SOURCE_DIR}/test/Dialect/ops_invalid.mlir
)
# ScratchAnalysis test: verify-diagnostics validates expected-warning comments
add_test(NAME ScratchAnalysis
    COMMAND $<TARGET_FILE:llk-opt> --verify-diagnostics --scratch-analysis --split-input-file
            ${CMAKE_SOURCE_DIR}/test/Transforms/scratch_analysis.mlir
)
endif()
