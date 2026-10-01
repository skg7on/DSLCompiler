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

# Treat deprecated API usage as errors
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  add_compile_options(-Werror=deprecated-declarations)
endif()

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

# Link MLIR dependencies through mlir_target_link_libraries() instead of
# naming MLIR components in LINK_LIBS. add_mlir_library() passes LINK_LIBS
# straight to llvm_add_library(), so those component archives reach the link
# interface of every consumer and put a second copy of MLIR next to the dylib
# (see the LLK_MLIR_EXEC_LIBS comment below for why that breaks verifiers).
# mlir_target_link_libraries() substitutes the dylib when it exists and the
# component archives otherwise, so it is correct for both configurations.

# Library: LLK dialect
add_mlir_dialect_library(LLKDialect
    lib/Dialect/LLK/LLKDialect.cpp
    lib/Dialect/LLK/LLKOps.cpp
    DEPENDS LLKDialectGen LLKOpsGen
)
mlir_target_link_libraries(LLKDialect PUBLIC
    MLIRIR MLIRDialect MLIRSupport MLIRArithDialect MLIRTensorDialect
)

# TableGen for Micro dialect
set(LLVM_TARGET_DEFINITIONS include/LLK/Dialect/Micro/MicroDialect.td)
mlir_tablegen(include/LLK/Dialect/Micro/MicroDialect.h.inc -gen-dialect-decls -dialect=micro)
mlir_tablegen(include/LLK/Dialect/Micro/MicroDialect.cpp.inc -gen-dialect-defs -dialect=micro)
mlir_tablegen(include/LLK/Dialect/Micro/MicroAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=micro)
mlir_tablegen(include/LLK/Dialect/Micro/MicroAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=micro)
add_public_tablegen_target(MicroDialectGen)

set(LLVM_TARGET_DEFINITIONS include/LLK/Dialect/Micro/MicroTypes.td)
mlir_tablegen(include/LLK/Dialect/Micro/MicroTypes.h.inc -gen-typedef-decls -typedefs-dialect=micro
  EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include/LLK/Dialect/Micro)
mlir_tablegen(include/LLK/Dialect/Micro/MicroTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=micro
  EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include/LLK/Dialect/Micro)
add_public_tablegen_target(MicroTypesGen)

set(LLVM_TARGET_DEFINITIONS include/LLK/Dialect/Micro/MicroOps.td)
mlir_tablegen(include/LLK/Dialect/Micro/MicroOps.h.inc -gen-op-decls
  EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include/LLK/Dialect/Micro)
mlir_tablegen(include/LLK/Dialect/Micro/MicroOps.cpp.inc -gen-op-defs
  EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include/LLK/Dialect/Micro)
add_public_tablegen_target(MicroOpsGen)

# Library: Micro dialect
add_mlir_dialect_library(MicroDialect
    lib/Dialect/Micro/MicroDialect.cpp
    lib/Dialect/Micro/MicroOps.cpp
    DEPENDS MicroDialectGen MicroTypesGen MicroOpsGen
)
mlir_target_link_libraries(MicroDialect PUBLIC
    MLIRIR MLIRDialect MLIRSupport MLIRArithDialect
)

# Library: MachineModel, YAML loader, and the L0/L1 performance models (M10).
# The model and loader are plain data; the simulator walks micro IR, so LLKPerf
# depends on MicroDialect. That dependency is not named through LINK_LIBS, it
# goes through mlir_target_link_libraries() inside MicroDialect itself, so the
# dylib/archive split described below stays correct for every consumer.
add_library(LLKPerf STATIC
    lib/Perf/MachineModel.cpp
    lib/Perf/MachineModelLoader.cpp
    lib/Perf/MicroDAG.cpp
    lib/Perf/MicroCostModel.cpp
    lib/Perf/MicroPerfReport.cpp
)
target_include_directories(LLKPerf PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${LLVM_INCLUDE_DIRS}
)
target_link_libraries(LLKPerf PUBLIC LLVMSupport MicroDialect)

# Library: LLKToLinalg conversion
add_mlir_library(LLKToLinalg
    lib/Conversion/LLKToLinalg/LLKToLinalg.cpp
    DEPENDS LLKDialect LLKCommon
    LINK_LIBS PUBLIC LLKDialect LLKCommon
)
mlir_target_link_libraries(LLKToLinalg PUBLIC
    MLIRLinalgDialect MLIRArithDialect MLIRMathDialect MLIRFuncDialect
    MLIRTensorDialect MLIRPass MLIRTransforms
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
    LLVMOrcJIT
    ${LLVM_NATIVE_TARGET_LIBS}
)
mlir_target_link_libraries(LLKRuntime PUBLIC
    MLIRExecutionEngine
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    MLIRMathToLLVM
    MLIRMemRefToLLVM
    MLIRReconcileUnrealizedCasts
    MLIRSCFToControlFlow
    MLIRVectorToLLVMPass
)

# find_package(ZLIB) and find_package(LibEdit) run from LLVMConfig.cmake resolve
# to the macOS SDK, and CMake propagates their include directories as *system*
# includes onto every target that links LLVM. <sdk>/usr/include is already
# searched implicitly through the sysroot, but as an explicit -isystem it sorts
# ahead of the C++ standard library's own headers. Apple's clang ships libc++
# inside that SDK so it is unaffected; any other clang on macOS fails to build
# LLKRuntime, the only target linking LLVM libraries directly, with
#   <cstdint> tried including <stdint.h> but didn't find libc++'s <stdint.h>
# Drop just those SDK roots so the include order stays valid for both.
foreach(_llk_sysdep IN ITEMS ZLIB::ZLIB LibEdit::LibEdit)
  if(NOT TARGET ${_llk_sysdep})
    continue()
  endif()
  get_target_property(_llk_sysdep_inc ${_llk_sysdep} INTERFACE_INCLUDE_DIRECTORIES)
  if(NOT _llk_sysdep_inc)
    continue()
  endif()
  set(_llk_sysdep_kept "")
  foreach(_llk_sysdep_dir IN LISTS _llk_sysdep_inc)
    if(NOT _llk_sysdep_dir MATCHES "^/.*\\.sdk/usr/include$")
      list(APPEND _llk_sysdep_kept ${_llk_sysdep_dir})
    endif()
  endforeach()
  set_property(TARGET ${_llk_sysdep}
               PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${_llk_sysdep_kept}")
endforeach()

# Library: X86 target feature detection
add_library(LLKTargetX86 STATIC
    lib/Target/X86/TargetAVX2.cpp
)
target_include_directories(LLKTargetX86 PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Library: AArch64 target feature detection (SVE/SVE2/NEON)
add_library(LLKTargetAArch64 STATIC
    lib/Target/AArch64/TargetSVE.cpp
)
target_include_directories(LLKTargetAArch64 PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)

# Library: Common utilities shared across all kernels
add_mlir_library(LLKCommon
    lib/Transforms/Common/TilingUtils.cpp
    lib/Transforms/Common/VectorizationUtils.cpp
    lib/Transforms/Common/ScheduleLoader.cpp
    lib/Transforms/Common/MaskGeneration.cpp
    lib/Transforms/Common/MathApproximation.cpp
    DEPENDS LLKDialect
)
mlir_target_link_libraries(LLKCommon PUBLIC
    MLIRArithDialect MLIRMathDialect MLIRVectorDialect MLIRPass
)

# Library: LLK Transforms (TileAndVectorize)
add_mlir_library(LLKTransforms
    lib/Transforms/TileAndVectorize.cpp
    DEPENDS LLKDialect LLKToLinalg LLKTransformsGen
    LINK_LIBS PUBLIC LLKToLinalg
)
mlir_target_link_libraries(LLKTransforms PUBLIC
    MLIRLinalgDialect MLIRLinalgTransforms MLIRVectorDialect MLIRSCFDialect
    MLIRSCFTransforms MLIRFuncDialect MLIRTensorDialect MLIRArithDialect MLIRPass
)

# Library: FuseDoubleContraction
add_mlir_library(LLKFuseDoubleContraction
    lib/Transforms/FuseDoubleContraction.cpp
    DEPENDS LLKDialect LLKToLinalg
    LINK_LIBS PUBLIC LLKToLinalg
)
mlir_target_link_libraries(LLKFuseDoubleContraction PUBLIC
    MLIRLinalgDialect MLIRArithDialect MLIRMathDialect MLIRFuncDialect
    MLIRTensorDialect MLIRPass MLIRTransforms
)

# Library: PackWeights weight annotation pass
add_mlir_library(LLKPackWeights
    lib/Transforms/PackWeights.cpp
    DEPENDS LLKDialect
)
mlir_target_link_libraries(LLKPackWeights PUBLIC
    MLIRLinalgDialect MLIRFuncDialect MLIRTensorDialect MLIRPass
)

# Library: ScratchAnalysis memory auditing pass
add_mlir_library(LLKScratchAnalysis
    lib/Transforms/ScratchAnalysis.cpp
)
mlir_target_link_libraries(LLKScratchAnalysis PUBLIC
    MLIRMemRefDialect MLIRFuncDialect MLIRPass
)

# Library: LinearizeForall pass
add_mlir_library(LLKLinearizeForall
    lib/Transforms/LinearizeForall.cpp
    DEPENDS LLKDialect
)
mlir_target_link_libraries(LLKLinearizeForall PUBLIC
    MLIRSCFDialect MLIRArithDialect MLIRFuncDialect MLIRTensorDialect MLIRPass
)

# Library: SerialParallelDispatch pass
add_mlir_library(LLKSerialParallelDispatch
    lib/Transforms/SerialParallelDispatch.cpp
)
mlir_target_link_libraries(LLKSerialParallelDispatch PUBLIC
    MLIRSCFDialect MLIRArithDialect MLIRFuncDialect MLIRPass
)

# Library: ForallToLLRT pass
add_mlir_library(LLKForallToLLRT
    lib/Transforms/ForallToLLRT.cpp
)
mlir_target_link_libraries(LLKForallToLLRT PUBLIC
    MLIRSCFDialect MLIRArithDialect MLIRFuncDialect MLIRPass
)

# Library: ForallToOpenMP pass (prototype)
add_mlir_library(LLKForallToOpenMP
    lib/Transforms/ForallToOpenMP.cpp
    DEPENDS LLKTransformsGen
)
mlir_target_link_libraries(LLKForallToOpenMP PUBLIC
    MLIRSCFDialect MLIRSCFTransforms MLIRSCFToOpenMP MLIROpenMPDialect MLIRPass
)

# Library: ShapeSpecialization pass
add_mlir_library(LLKShapeSpecialization
    lib/Transforms/ShapeSpecialization.cpp
    DEPENDS LLKDialect
)
mlir_target_link_libraries(LLKShapeSpecialization PUBLIC
    MLIRFuncDialect MLIRPass
)

# Library: ScheduleSelection pass
add_mlir_library(LLKScheduleSelection
    lib/Transforms/ScheduleSelection.cpp
    DEPENDS LLKDialect
)
mlir_target_link_libraries(LLKScheduleSelection PUBLIC
    MLIRFuncDialect MLIRPass
)

# Library: TritonToLLK lowering passes
add_subdirectory(lib/Conversion/TritonToLLK)

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
    MLIROpenMPDialect
    MLIRParser
    MLIRPass
    MLIRReconcileUnrealizedCasts
    MLIRRegisterAllDialects
    MLIRRegisterAllPasses
    MLIRSCFDialect
    MLIRSCFToControlFlow
    MLIRSCFToOpenMP
    MLIRSCFTransforms
    MLIRSupport
    MLIRTensorDialect
    MLIRTensorTransforms
    MLIRTransforms
    MLIRVectorToLLVMPass
)

# When MLIR is built with a single libMLIR dylib, MLIRConfig.cmake exports
# MLIR_LINK_MLIR_DYLIB and the component libraries above are static archives.
# Naming them individually puts those archives in the executable alongside the
# dylib, so the process holds two copies of MLIR whose dialect classes have
# different TypeIDs. Startup then aborts during context construction with:
#   LLVM ERROR: Trying to register different dialects for the same namespace: builtin
# Link the dylib target instead, as MLIR's own mlir_target_link_libraries() does.
if(MLIR_LINK_MLIR_DYLIB)
  set(LLK_MLIR_EXEC_LIBS MLIR)
endif()

if(LLK_BUILD_TOOLS)
add_llvm_executable(llk-opt tools/llk-opt/llk-opt.cpp)
target_link_libraries(llk-opt PRIVATE LLKPasses LLKTransforms LLKFuseDoubleContraction LLKPackWeights LLKScratchAnalysis LLKLinearizeForall LLKSerialParallelDispatch LLKForallToLLRT LLKForallToOpenMP LLKShapeSpecialization LLKScheduleSelection TritonToLLKPasses MicroDialect MLIRMlirOptMain ${LLK_MLIR_EXEC_LIBS})

# Tool: llk-compile
add_llvm_executable(llk-compile tools/llk-compile/llk-compile.cpp)
target_link_libraries(llk-compile PRIVATE LLKPasses LLKTransforms LLKFuseDoubleContraction LLKPackWeights LLKScratchAnalysis LLKLinearizeForall LLKSerialParallelDispatch LLKForallToLLRT LLKShapeSpecialization LLKScheduleSelection TritonToLLKPasses LLKRuntime ${LLK_MLIR_EXEC_LIBS})

# Tool: llk-bench (benchmarking harness)
add_llvm_executable(llk-bench tools/llk-bench/llk-bench.cpp)
target_link_libraries(llk-bench PRIVATE LLKRuntime)

# Tool: llk-tune (autotuning grid search)
add_llvm_executable(llk-tune tools/llk-tune/llk-tune.cpp)
target_link_libraries(llk-tune PRIVATE LLKRuntime)

# Tool: micro-perf (Micro-IR performance simulator, M10)
add_llvm_executable(micro-perf tools/micro-perf/micro-perf.cpp)
target_link_libraries(micro-perf PRIVATE LLKPerf MicroDialect ${LLK_MLIR_EXEC_LIBS})
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
target_link_libraries(SwigluScalar PRIVATE LLKPasses LLKTransforms LLKRuntime GTest::gtest_main ${LLK_MLIR_EXEC_LIBS})
add_test(NAME SwigluScalar COMMAND SwigluScalar)

# Triton IR → staged pipeline → JIT integration test (M8b). Test body is
# currently a DISABLED stub; links the staged-pipeline pass libs it will
# exercise once the full Triton lowering path is integrated.
add_llvm_executable(TritonSwigluStaged test/Execution/triton_swiglu_staged.cpp)
target_link_libraries(TritonSwigluStaged PRIVATE LLKPasses TritonToLLKPasses LLKRuntime GTest::gtest_main ${LLK_MLIR_EXEC_LIBS})
add_test(NAME TritonSwigluStaged COMMAND TritonSwigluStaged)

add_llvm_executable(SwiGLUVector test/Execution/swiglu_vector_avx2.cpp)
target_link_libraries(SwiGLUVector PRIVATE LLKPasses LLKTransforms LLKRuntime GTest::gtest_main ${LLK_MLIR_EXEC_LIBS})
add_test(NAME SwiGLUVector COMMAND SwiGLUVector)

add_llvm_executable(SwiGLUFusedMemory test/Execution/swiglu_fused_memory.cpp)
target_link_libraries(SwiGLUFusedMemory PRIVATE LLKPasses LLKTransforms LLKFuseDoubleContraction LLKPackWeights LLKRuntime GTest::gtest_main ${LLK_MLIR_EXEC_LIBS})
add_test(NAME SwiGLUFusedMemory COMMAND SwiGLUFusedMemory)
endif()

add_llk_test(VectorVsScalar test/Numerical/vector_vs_scalar.cpp)
add_llk_test(EdgeCases test/Execution/edge_cases.cpp)
add_llk_test(RoPECorrectness test/Execution/rope_correctness.cpp)
add_llk_test(AttentionCorrectness test/Execution/attention_correctness.cpp)
add_llk_test(AttentionStability test/Execution/attention_stability.cpp)
add_llk_test(RoPEPrecision test/Numerical/rope_precision.cpp)
add_llk_test(SharedPipeline test/Execution/shared_pipeline.cpp)
add_llk_test(CpuFeaturesTest test/Execution/cpu_features_test.cpp)
target_link_libraries(CpuFeaturesTest PRIVATE LLKTargetX86)

add_llk_test(PackedVsUnpacked test/Execution/packed_vs_unpacked.cpp)
add_llk_test(SpecializationDispatch test/Execution/specialization_dispatch.cpp)
add_llk_test(ScheduleDeterminism test/Execution/schedule_determinism.cpp)

add_llvm_executable(AffinityTest test/Execution/affinity_test.cpp)
target_link_libraries(AffinityTest PRIVATE LLKRuntime GTest::gtest_main)
add_test(NAME AffinityTest COMMAND AffinityTest)

add_llvm_executable(JitCacheMiss test/Execution/jit_cache_miss.cpp)
target_link_libraries(JitCacheMiss PRIVATE LLKRuntime GTest::gtest_main)
add_test(NAME JitCacheMiss COMMAND JitCacheMiss)

add_llk_test(ScheduleRegression benchmark/schedule_regression.cpp)

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

add_llvm_executable(CacheEviction test/Execution/cache_eviction.cpp)
target_link_libraries(CacheEviction PRIVATE LLKRuntime GTest::gtest_main)
add_test(NAME CacheEviction COMMAND CacheEviction)

# The loader test loads the shipped machine files, not just inline fixtures, so
# the models used for real must parse and verify.
add_llvm_executable(MachineModelLoaderTest test/Perf/machine_model_loader.cpp)
target_link_libraries(MachineModelLoaderTest PRIVATE LLKPerf GTest::gtest_main)
target_compile_definitions(MachineModelLoaderTest PRIVATE
    LLK_MACHINE_DIR="${CMAKE_SOURCE_DIR}/machines")
add_test(NAME MachineModelLoaderTest COMMAND MachineModelLoaderTest)

# Performance-model tests. Unlike the loader test these parse micro IR. They
# must not name MLIR components directly: LLK_MLIR_EXEC_LIBS already carries
# MLIRParser, and resolves to the MLIR dylib when there is one. Naming a
# component alongside it is the mistake described above -- the executable then
# holds two copies of MLIR whose attribute classes have different TypeIDs, and
# `isa<StringAttr>` starts failing on perfectly good IR. Each test loads the
# shipped machine files by path, for the same reason the loader test does: the
# model used for real must be the one tested.
function(add_llk_perf_test test_name test_file)
  add_llvm_executable(${test_name} ${test_file})
  target_link_libraries(${test_name} PRIVATE
      LLKPerf MicroDialect GTest::gtest_main ${LLK_MLIR_EXEC_LIBS})
  target_compile_definitions(${test_name} PRIVATE
      LLK_MACHINE_DIR="${CMAKE_SOURCE_DIR}/machines")
  add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

add_llk_perf_test(L0StaticBoundTest test/Perf/l0_static_bound.cpp)
add_llk_perf_test(L1ResourceDAGTest test/Perf/l1_resource_dag.cpp)

# FileCheck tests require llk-opt and must use the FileCheck built alongside
# the LLVM/MLIR packages consumed by this project. Clear the cached result on
# every configure so changing LLVM_PROJECT_BUILD_DIR cannot retain FileCheck
# from an older or unrelated LLVM build.
if(LLK_BUILD_TOOLS)
  unset(FILECHECK_BIN CACHE)
  unset(FILECHECK_BIN)
  find_program(FILECHECK_BIN NAMES FileCheck
      HINTS "${LLVM_TOOLS_BINARY_DIR}"
      NO_DEFAULT_PATH)
  if(NOT FILECHECK_BIN)
    message(FATAL_ERROR
        "FileCheck is required when LLK_BUILD_TOOLS=ON, but it was not found "
        "in LLVM_TOOLS_BINARY_DIR (${LLVM_TOOLS_BINARY_DIR}). Rebuild LLVM "
        "with -DLLVM_BUILD_UTILS=ON.")
  endif()

  function(add_llk_filecheck_test test_name input_file)
      # Use sh -c to properly handle input redirection (CMake's add_test
      # COMMAND mode does not support < redirection natively).
      # Execute the RUN line: llk-opt %s | llk-opt | FileCheck %s
      # This round-trips the IR through llk-opt twice to verify parse+print stability.
      add_test(NAME ${test_name}
          COMMAND sh -c "$<TARGET_FILE:llk-opt> ${CMAKE_SOURCE_DIR}/${input_file} | $<TARGET_FILE:llk-opt> | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/${input_file}"
      )
  endfunction()

  add_llk_filecheck_test(DialectOps test/Dialect/ops.mlir)
  add_llk_filecheck_test(MicroDialectOps test/Dialect/Micro/ops.mlir)
  add_llk_filecheck_test(MicroDialectTileOps test/Dialect/Micro/tile_ops.mlir)
  add_llk_filecheck_test(MicroDialectSearchSpace test/Dialect/Micro/search_space.mlir)
  # LLKToLinalgConversion test: run llk-opt --llk-to-linalg then FileCheck
  add_test(NAME LLKToLinalgConversion
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg ${CMAKE_SOURCE_DIR}/test/Conversion/llk_to_linalg.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/llk_to_linalg.mlir"
  )
  # RoPE lowering: verify llk.rope → bounded-fast trig + even/odd interleave
  add_test(NAME RoPELowering
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg ${CMAKE_SOURCE_DIR}/test/Conversion/rope_lowering.mlir | ${FILECHECK_BIN} --implicit-check-not=math.cos --implicit-check-not=math.sin ${CMAKE_SOURCE_DIR}/test/Conversion/rope_lowering.mlir"
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
  # ForallToOpenMP test: scf.forall → omp.parallel + omp.wsloop
  add_test(NAME ForallToOpenMP
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --forall-to-openmp \
        ${CMAKE_SOURCE_DIR}/test/Transforms/forall_to_openmp.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/forall_to_openmp.mlir"
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
  # RoPE broadcast: verify trig tables broadcast across BxH dimensions
  add_test(NAME RoPEBroadcast
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg ${CMAKE_SOURCE_DIR}/test/Transforms/rope_broadcast.mlir | ${FILECHECK_BIN} --implicit-check-not=math.cos --implicit-check-not=math.sin ${CMAKE_SOURCE_DIR}/test/Transforms/rope_broadcast.mlir"
  )
  # GridToForall: verify tt.get_program_id -> scf.forall lowering
  add_test(NAME GridToForall
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --allow-unregistered-dialect --triton-grid-to-forall \
        ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/grid_to_forall.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/grid_to_forall.mlir"
  )
  # GridToForallReturnOperands: regression for func.return referencing a cloned
  # op -- the pass must remap the return operand instead of crashing on erase.
  add_test(NAME GridToForallReturnOperands
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --allow-unregistered-dialect --verify-each=false --triton-grid-to-forall \
        ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/grid_to_forall_return.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/grid_to_forall_return.mlir"
  )
  # TritonToStructured: verify tt.dot -> linalg.matmul lowering
  add_test(NAME TritonToStructured
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --triton-to-structured --allow-unregistered-dialect \
        ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/triton_to_structured.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/triton_to_structured.mlir"
  )
  # BlockPointerToVector: verify tt.make_block_ptr + load/store -> vector.transfer
  add_test(NAME BlockPointerToVector
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --allow-unregistered-dialect --triton-block-ptr-to-vector \
        ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/block_ptr_to_vector.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/block_ptr_to_vector.mlir"
  )
  # BlockPointer1D: verify 1D block pointer lowering
  add_test(NAME BlockPointer1D
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --allow-unregistered-dialect --triton-block-ptr-to-vector \
        ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/block_ptr_1d.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/block_ptr_1d.mlir"
  )
  # BlockPointerMasked: verify boundary mask for partial tiles (N=127 not /8)
  add_test(NAME BlockPointerMasked
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --allow-unregistered-dialect --triton-block-ptr-to-vector \
        ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/block_ptr_masked.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/block_ptr_masked.mlir"
  )
  # SharedMemToScratch: verify tt.alloc -> memref.alloc, tt.async_copy -> memref.copy
  add_test(NAME SharedMemToScratch
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --allow-unregistered-dialect --triton-shared-mem-to-scratch \
        ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/shared_mem_to_scratch.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/shared_mem_to_scratch.mlir"
  )
  # AtomicToLLRT: verify tt.atomic_* -> llrt.atomic_* runtime calls
  add_test(NAME AtomicToLLRT
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --allow-unregistered-dialect --triton-atomic-to-llrt \
        ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/atomic_to_llrt.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/atomic_to_llrt.mlir"
  )
  # TritonCPUVerifierPass: whitelisted ops pass through with no diagnostics
  add_test(NAME TritonCPUVerifierPass
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --allow-unregistered-dialect --triton-cpu-verify \
        ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/cpu_verifier_pass.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/cpu_verifier_pass.mlir"
  )
  # TritonCPUVerifierReject: unsupported tt.* ops rejected with an error
  add_test(NAME TritonCPUVerifierReject
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --allow-unregistered-dialect --triton-cpu-verify \
        ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/cpu_verifier_reject.mlir \
        2>&1 | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/cpu_verifier_reject.mlir"
  )
  # TritonFullPipeline: all 5 Triton->LLK stages in one invocation
  add_test(NAME TritonFullPipeline
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --allow-unregistered-dialect \
        --triton-to-structured --canonicalize --triton-cpu-verify \
        --triton-block-ptr-to-vector --triton-shared-mem-to-scratch \
        --triton-atomic-to-llrt --triton-grid-to-forall \
        ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/triton_full_pipeline.mlir \
        | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/triton_full_pipeline.mlir"
  )
  # Causal mask: verify cmp + select pattern for upper-triangular -inf
  add_test(NAME CausalMask
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg ${CMAKE_SOURCE_DIR}/test/Transforms/causal_mask.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Transforms/causal_mask.mlir"
  )
  # micro-perf: the simulator report on a concrete tile GEMM must contain the
  # counts, the schedule, the tile summary, and a bottleneck.
  add_test(NAME MicroPerfCli
      COMMAND sh -c "$<TARGET_FILE:micro-perf> --machine=${CMAKE_SOURCE_DIR}/machines/x86-avx2-cpu.yaml --level=1 ${CMAKE_SOURCE_DIR}/test/Perf/micro_perf_cli.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Perf/micro_perf_cli.mlir"
  )
  # The help text has to say what the MVP is, so nobody reads "simulator" as
  # "functional emulator". Same fixture, other check prefix.
  add_test(NAME MicroPerfHelp
      COMMAND sh -c "$<TARGET_FILE:micro-perf> --help 2>&1 | ${FILECHECK_BIN} --check-prefix=HELP ${CMAKE_SOURCE_DIR}/test/Perf/micro_perf_cli.mlir"
  )
  # TritonToLLK: verify llk.matmul / llk.make_tensor parse and verify
  add_test(NAME TritonToLLKNewOps
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --verify-diagnostics --split-input-file ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/llk_new_ops.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/TritonToLLK/llk_new_ops.mlir"
  )
endif()

# Test for ops_invalid.mlir: runs llk-opt with --verify-diagnostics.
# Only registered when llk-opt is built (LLK_BUILD_TOOLS=ON).
if(LLK_BUILD_TOOLS)
add_test(NAME DialectOpsInvalid
    COMMAND $<TARGET_FILE:llk-opt> --verify-diagnostics --split-input-file
            ${CMAKE_SOURCE_DIR}/test/Dialect/ops_invalid.mlir
)
add_test(NAME MicroDialectOpsInvalid
    COMMAND $<TARGET_FILE:llk-opt> --verify-diagnostics --split-input-file
            ${CMAKE_SOURCE_DIR}/test/Dialect/Micro/ops_invalid.mlir
)
add_test(NAME MicroDialectTileOpsInvalid
    COMMAND $<TARGET_FILE:llk-opt> --verify-diagnostics --split-input-file
            ${CMAKE_SOURCE_DIR}/test/Dialect/Micro/tile_ops_invalid.mlir
)
add_test(NAME MicroDialectSearchSpaceInvalid
    COMMAND $<TARGET_FILE:llk-opt> --verify-diagnostics --split-input-file
            ${CMAKE_SOURCE_DIR}/test/Dialect/Micro/search_space_invalid.mlir
)
# ScratchAnalysis test: verify-diagnostics validates expected-warning comments
add_test(NAME ScratchAnalysis
    COMMAND $<TARGET_FILE:llk-opt> --verify-diagnostics --scratch-analysis --split-input-file
            ${CMAKE_SOURCE_DIR}/test/Transforms/scratch_analysis.mlir
)
endif()

# The canonical local test command, as documented in README.md. It is defined
# once, over the CTest registry, so `check-llk`, `ctest`, and CI cannot drift
# apart. Test binaries are derived from the registry rather than listed by
# hand: every test whose name is also a build target is built first, and
# llk-opt covers the FileCheck and verifier tests.
get_property(_llk_test_names DIRECTORY PROPERTY TESTS)
set(_llk_check_deps "")
foreach(_llk_test_name IN LISTS _llk_test_names)
  if(TARGET ${_llk_test_name})
    list(APPEND _llk_check_deps ${_llk_test_name})
  endif()
  # A gtest that calls GTEST_SKIP still exits 0, so CTest would otherwise
  # report an unsupported host as a pass. Match gtest's skip banner so the
  # summary separates skipped from executed tests.
  set_tests_properties(${_llk_test_name}
                       PROPERTIES SKIP_REGULAR_EXPRESSION "\\[  SKIPPED \\]")
endforeach()
if(TARGET llk-opt)
  list(APPEND _llk_check_deps llk-opt)
endif()
if(TARGET micro-perf)
  list(APPEND _llk_check_deps micro-perf)
endif()

add_custom_target(check-llk
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS ${_llk_check_deps}
    USES_TERMINAL
    COMMENT "Running the LLK test suite"
)
