cmake_minimum_required(VERSION 3.20)

# Compiler selection. This has to run before project(), because that is where
# CMake consults $CC/$CXX and detects the compiler.
#
# $CC/$CXX take precedence over CMake's own compiler search, and CMake aborts
# configuration when the named file is missing. A shell profile that exports a
# toolchain from `brew --prefix llvm@NN` leaves exactly that behind once the keg
# is removed, because brew keeps printing the path after uninstalling. Drop such
# values. Absolute paths only: bare names like CC=gcc are resolved through PATH
# by CMake and must be preserved.
foreach(_llk_var IN ITEMS CC CXX)
  if(DEFINED ENV{${_llk_var}} AND IS_ABSOLUTE "$ENV{${_llk_var}}"
     AND NOT EXISTS "$ENV{${_llk_var}}")
    message(STATUS "Ignoring ${_llk_var}=$ENV{${_llk_var}} (no such file); "
                   "falling back to the compiler on PATH")
    unset(ENV{${_llk_var}})
  endif()
endforeach()

# On macOS, CMake's default search lands on /usr/bin/cc (Apple clang), while the
# intended compiler is the clang on PATH: Homebrew's llvm kegs are keg-only yet
# still symlink clang/clang++ into <prefix>/bin, which PATH places ahead of
# /usr/bin. NO_DEFAULT_PATH restricts the search to PATH order. Pre-seeding the
# cache is what actually selects it -- CMake otherwise prefers its own /usr/bin
# search over PATH. An explicit -D or a still-valid $CC/$CXX wins.
#
# These only apply to a fresh build tree. Once CMAKE_C_COMPILER/CMAKE_CXX_COMPILER
# are cached they are kept, so an already-configured build/ must be removed to
# re-resolve the compiler.
if(APPLE)
  if(NOT DEFINED CMAKE_C_COMPILER AND NOT DEFINED ENV{CC})
    find_program(_llk_clang NAMES clang HINTS ENV PATH NO_DEFAULT_PATH)
    if(_llk_clang)
      set(CMAKE_C_COMPILER "${_llk_clang}" CACHE FILEPATH "C compiler")
    endif()
  endif()
  if(NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
    find_program(_llk_clangxx NAMES clang++ HINTS ENV PATH NO_DEFAULT_PATH)
    if(_llk_clangxx)
      set(CMAKE_CXX_COMPILER "${_llk_clangxx}" CACHE FILEPATH "C++ compiler")
    endif()
  endif()
endif()

# On macOS, pin the deployment target when the user has not chosen one.
#
# Left to itself the compiler reads it off the target triple -- arm64-apple-
# darwin27.0.0 on a current host -- through a Darwin-to-macOS version table, and
# a clang built before the host's Darwin major existed falls back to the old
# pre-macOS-11 formula "Darwin - 9". Homebrew's llvm@20 therefore announces
# macOS 18.0 for a macOS 27 host, which is old enough that the linker warns
# about every Homebrew dylib built for a newer macOS ("building for macOS-18.0,
# but linking with dylib '/opt/homebrew/opt/zstd/lib/libzstd.1.dylib' which was
# built for newer version 26.0"), and it is wrong regardless: the objects are
# compiled against the installed SDK. The host version is what the triple was
# trying to express, so pin to that. An explicit -DCMAKE_OSX_DEPLOYMENT_TARGET
# wins, and a value already in the cache is left alone.
if(APPLE AND NOT CMAKE_OSX_DEPLOYMENT_TARGET)
  execute_process(COMMAND sw_vers -productVersion
                  OUTPUT_VARIABLE _llk_host_macos
                  OUTPUT_STRIP_TRAILING_WHITESPACE
                  ERROR_QUIET)
  # "major.minor" only: the value reaches the compiler as
  # -mmacosx-version-min, which takes no patch component.
  if(_llk_host_macos MATCHES "^([0-9]+\\.[0-9]+)")
    set(CMAKE_OSX_DEPLOYMENT_TARGET "${CMAKE_MATCH_1}" CACHE STRING
        "Minimum OS X version to target for deployment (at runtime); newer"
        FORCE)
    message(STATUS "Pinning CMAKE_OSX_DEPLOYMENT_TARGET to host macOS "
                   "${CMAKE_MATCH_1}")
  else()
    message(STATUS "Could not read a macOS version from `sw_vers`; leaving "
                   "CMAKE_OSX_DEPLOYMENT_TARGET to the compiler default")
  endif()
endif()

project(llk-compiler VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# The compiler version the plan report (§22.2) records. It is the project
# version plus a `git describe` suffix when the checkout is a git tree, so two
# compiler revisions never print the same string. A git failure leaves the
# bare project version, which is still a real version rather than a name.
set(LLK_COMPILER_VERSION "${PROJECT_VERSION}")
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(
    COMMAND "${GIT_EXECUTABLE}" describe --tags --always --dirty
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    OUTPUT_VARIABLE LLK_GIT_DESCRIBE
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE LLK_GIT_RESULT)
  if(LLK_GIT_RESULT EQUAL 0 AND LLK_GIT_DESCRIBE)
    set(LLK_COMPILER_VERSION "${PROJECT_VERSION}+${LLK_GIT_DESCRIBE}")
  endif()
endif()
configure_file(include/LLK/Version.h.in
    "${CMAKE_BINARY_DIR}/include/LLK/Version.h" @ONLY)
message(STATUS "LLK compiler version: ${LLK_COMPILER_VERSION}")

# Emit compile_commands.json for clangd and other editors. Honoured by the Ninja
# and Makefile generators; other generators ignore it.
#
# CMake pre-seeds this as an *empty* cache entry -- CMakeGenericSystem.cmake
# runs before this file does -- so "defined" is always true and only the value
# carries intent. An explicit -DCMAKE_EXPORT_COMPILE_COMMANDS=OFF is non-empty
# and is therefore preserved.
if(CMAKE_EXPORT_COMPILE_COMMANDS STREQUAL "")
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL
      "Enable/Disable output of compile commands during generation" FORCE)
endif()

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
    lib/Perf/MicroDAG.cpp
    lib/Perf/MicroCostModel.cpp
    lib/Perf/MicroPerfReport.cpp
    lib/Perf/SearchSpace.cpp
    lib/Perf/Candidate.cpp
    lib/Perf/CandidateGenerator.cpp
    lib/Perf/Legality.cpp
    lib/Perf/CandidateBinding.cpp
    lib/Perf/ScheduleRecord.cpp
    lib/Perf/TuningSession.cpp
)
target_include_directories(LLKPerf PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${LLVM_INCLUDE_DIRS}
)
# LLKPerf consumes mapped events, so it shares LLKMapping's cost and event
# vocabulary (design §17.2).
target_link_libraries(LLKPerf PUBLIC LLVMSupport MicroDialect LLKMachine LLKMapping)
# Candidate binding builds concrete micro IR, which references tensor.empty for
# the kernel's external tensors. MLIRTensorDialect is named through
# mlir_target_link_libraries() so it resolves to the MLIR dylib when one exists,
# keeping a single copy of MLIR in every consumer.
# LLVM builds without RTTI or exceptions; a plain add_library does not inherit
# that, and the mismatch makes llvm::Error vtable emission reference a typeinfo
# the LLVM build does not contain.
target_compile_options(LLKPerf PRIVATE -fno-rtti -fno-exceptions)
mlir_target_link_libraries(LLKPerf PRIVATE MLIRTensorDialect)

# Library: versioned machine topology (issue #82, epic #67).
# Target facts only -- executor/memory/compute/transfer nodes, their edges, and
# the normalized queries placement and routing ask of them. It must not depend
# on LLKPerf, LLKMapping, or any backend; a target package describes a machine,
# it does not change this code.
add_library(LLKMachine STATIC
    lib/Machine/MachineModel.cpp
    lib/Machine/MachineModelLoader.cpp
)
target_include_directories(LLKMachine PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${LLVM_INCLUDE_DIRS}
)
target_link_libraries(LLKMachine PUBLIC LLVMSupport)
# LLVM builds without RTTI or exceptions; a plain add_library does not inherit
# that, and the mismatch makes llvm::Error vtable emission reference a typeinfo
# the LLVM build does not contain.
target_compile_options(LLKMachine PRIVATE -fno-rtti -fno-exceptions)

# Library: target-independent mapping data model (issue #80, epic #67 D1).
# Generic planning types only -- workload graphs, search bindings, plan
# structures, costs, and stable ids. It must not depend on LLKPerf or any
# backend: mapping is below performance modeling, and a target package plugs
# into it rather than the reverse. Routing (D2) additionally consumes
# LLKMachine's topology.
add_library(LLKMapping STATIC
    lib/Mapping/StableHash.cpp
    lib/Mapping/Diagnostics.cpp
    lib/Mapping/TileFacts.cpp
    lib/Mapping/StoragePlan.cpp
    lib/Mapping/CostModel.cpp
    lib/Mapping/CostEvent.cpp
    lib/Mapping/EventSchedule.cpp
    lib/Mapping/LatencyProvider.cpp
    lib/Mapping/SearchBinding.cpp
    lib/Mapping/WorkloadGraph.cpp
    lib/Mapping/MappingPlan.cpp
    lib/Mapping/LlkMap.cpp
    lib/Mapping/LayoutConstraints.cpp
    lib/Mapping/MappingRules.cpp
    lib/Mapping/MappingTarget.cpp
    lib/Mapping/Placement.cpp
    lib/Mapping/CoveringSearch.cpp
    lib/Mapping/PlanBinder.cpp
    lib/Mapping/MappingMetadata.cpp
    lib/Mapping/PlanVerification.cpp
    lib/Mapping/Routing.cpp
    lib/Mapping/PlanReport.cpp
)
target_include_directories(LLKMapping PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${LLVM_INCLUDE_DIRS}
)
target_link_libraries(LLKMapping PUBLIC LLVMSupport LLKMachine)
# LLVM and MLIR are built without RTTI or exceptions. A plain add_library does
# not inherit those, and without -fno-rtti the vtable LLVM emits for
# llvm::ErrorInfo (used when llvm::Expected carries a non-trivial payload)
# references a typeinfo the LLVM build does not contain -- an undefined symbol
# at link time. Match the toolchain.
target_compile_options(LLKMapping PRIVATE -fno-rtti -fno-exceptions)
# AffineMap, OperationName, DictionaryAttr, and Type are MLIR IR values used by
# the workload graph. They are named through mlir_target_link_libraries() so the
# dylib/archive split stays correct, exactly as LLKPerf does.
# The binder builds `#micro.memory<...>` and `!micro.async_token` without the
# dialect's generated classes, so it reads them from assembly. Named through
# mlir_target_link_libraries() like every other MLIR dependency here.
mlir_target_link_libraries(LLKMapping PRIVATE MLIRIR MLIRAsmParser)

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

# Library: LLK -> concrete micro.kernel export (M11)
add_mlir_library(LLKToMicro
    lib/Conversion/LLKToMicro/LLKToMicro.cpp
    DEPENDS LLKDialect MicroDialect LLKCommon
    LINK_LIBS PUBLIC LLKDialect MicroDialect LLKCommon
)
mlir_target_link_libraries(LLKToMicro PUBLIC
    MLIRArithDialect MLIRFuncDialect MLIRTensorDialect MLIRPass MLIRTransforms
)

# Library: micro -> Linalg bridge. `micro` was a sink: nothing lowered it to
# something a backend consumes, so a mapped or exported micro.kernel could not
# reach the LLVM/JIT pipeline. This is that bridge.
add_mlir_library(LLKMicroToLinalg
    lib/Conversion/MicroToLinalg/MicroToLinalg.cpp
    DEPENDS MicroDialect
    LINK_LIBS PUBLIC MicroDialect
)
mlir_target_link_libraries(LLKMicroToLinalg PUBLIC
    MLIRArithDialect MLIRFuncDialect MLIRLinalgDialect MLIRMemRefDialect
    MLIRSCFDialect MLIRTensorDialect MLIRPass MLIRTransforms
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

# Library: target mapping packages (issue #67, workstream D7). Each target owns
# its configuration paths and emitter keys; generic mapping code sees only the
# MappingTarget interface. Kept out of LLKTargetX86 so CPU feature detection
# does not drag in MLIR.
add_library(LLKTargetMapping STATIC
    lib/Target/X86/Mapping/AVX2MappingTarget.cpp
    lib/Target/GenericAccelerator/Mapping/GenericAcceleratorMappingTarget.cpp
)
target_include_directories(LLKTargetMapping PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(LLKTargetMapping PUBLIC LLKMapping LLKMachine)
target_compile_options(LLKTargetMapping PRIVATE -fno-rtti -fno-exceptions)

# Library: mapping passes (design §21). The `--micro-map` and
# `--micro-bind-plan` passes are the compiler front door to the mapping engine.
# They are strictly target-neutral: a target is loaded from files on disk, so
# the library depends on the mapping core and the target packages, never the
# reverse.
add_library(LLKMicroMapping STATIC
    lib/Conversion/MicroMapping/MicroMapPass.cpp
    lib/Conversion/MicroMapping/MicroBindPlanPass.cpp
    lib/Conversion/MicroMapping/MicroVerifyMappingPass.cpp
    lib/Conversion/MicroMapping/SearchBindingLoader.cpp
    lib/Conversion/MicroMapping/PlanMaterialization.cpp
)
target_include_directories(LLKMicroMapping PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${LLVM_INCLUDE_DIRS}
)
# The SearchBinding loader reads a space's parameters through
# llk::perf::loadSearchSpace rather than re-parsing micro.param, so this library
# needs LLKPerf. That direction is legal: LLKPerf already depends on LLKMapping,
# and LLKMapping must never depend on LLKPerf, so the pass library -- above both
# -- is the only layer that may link them together.
target_link_libraries(LLKMicroMapping PUBLIC LLKMapping LLKTargetMapping LLKMachine LLKPerf)
# LLVM and MLIR are built without RTTI or exceptions; a plain add_library does
# not inherit those. Match the toolchain, exactly as LLKMapping does.
target_compile_options(LLKMicroMapping PRIVATE -fno-rtti -fno-exceptions)
# PassWrapper/OperationPass come from MLIRPass; ShapedType and Operation from
# MLIRIR. Named through mlir_target_link_libraries() so the dylib/archive split
# stays correct.
mlir_target_link_libraries(LLKMicroMapping PRIVATE MLIRIR MLIRParser MLIRPass)

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
target_link_libraries(llk-opt PRIVATE LLKPasses LLKTransforms LLKFuseDoubleContraction LLKPackWeights LLKScratchAnalysis LLKLinearizeForall LLKSerialParallelDispatch LLKForallToLLRT LLKForallToOpenMP LLKShapeSpecialization LLKScheduleSelection LLKToMicro LLKMicroToLinalg LLKMicroMapping TritonToLLKPasses MicroDialect MLIRMlirOptMain ${LLK_MLIR_EXEC_LIBS})

# Tool: llk-compile
add_llvm_executable(llk-compile tools/llk-compile/llk-compile.cpp)
target_link_libraries(llk-compile PRIVATE LLKPasses LLKTransforms LLKFuseDoubleContraction LLKPackWeights LLKScratchAnalysis LLKLinearizeForall LLKSerialParallelDispatch LLKForallToLLRT LLKShapeSpecialization LLKScheduleSelection LLKToMicro TritonToLLKPasses LLKRuntime ${LLK_MLIR_EXEC_LIBS})

# Tool: llk-bench (benchmarking harness)
add_llvm_executable(llk-bench tools/llk-bench/llk-bench.cpp)
target_link_libraries(llk-bench PRIVATE LLKRuntime)

# Tool: llk-tune (autotuning: Micro search space, or the legacy grid)
add_llvm_executable(llk-tune tools/llk-tune/llk-tune.cpp)
target_link_libraries(llk-tune PRIVATE
    LLKPerf MicroDialect LLKDialect LLKRuntime ${LLK_MLIR_EXEC_LIBS})

# Tool: micro-perf (Micro-IR performance simulator, M10)
add_llvm_executable(micro-perf tools/micro-perf/micro-perf.cpp)
target_link_libraries(micro-perf PRIVATE LLKPerf MicroDialect LLKDialect ${LLK_MLIR_EXEC_LIBS})
endif()

# Tests
include(CTest)
enable_testing()

add_subdirectory(third_party)

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

# MicroKernelExecution: the acceptance chain for the micro->Linalg bridge. A
# `micro.kernel` is lowered, buffered, JIT-compiled and *called*, and its result
# checked -- which is also what pins the kernel ABI (entry tensors become
# parameters, the stored value becomes the bufferized out-parameter).
add_llvm_executable(MicroKernelExecution test/Execution/micro_kernel_execution.cpp)
target_link_libraries(MicroKernelExecution PRIVATE
    LLKMicroToLinalg LLKTransforms LLKRuntime GTest::gtest_main
    ${LLK_MLIR_EXEC_LIBS})
add_test(NAME MicroKernelExecution COMMAND MicroKernelExecution)

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


# The schedule loader test loads the shipped database as well as inline
# fixtures, so the file the passes read for real is covered by the suite.
add_llvm_executable(ScheduleLoaderTest test/Transforms/schedule_loader.cpp)
target_link_libraries(ScheduleLoaderTest PRIVATE LLKCommon GTest::gtest_main ${LLK_MLIR_EXEC_LIBS})
target_compile_definitions(ScheduleLoaderTest PRIVATE
    LLK_SCHEDULE_DB="${CMAKE_SOURCE_DIR}/schedules/schedule_db.json")
add_test(NAME ScheduleLoaderTest COMMAND ScheduleLoaderTest)

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

# Binder -> perf route identity: builds a real plan, binds it through the
# canonical PlanMaterializer (which stamps the copies the perf model charges),
# and charges the bound kernel. Crosses the mapping/perf boundary, so it links
# LLKMapping and the Micro mapping pass library that owns the materializer.
add_llk_perf_test(RouteIdentityTest test/Perf/route_identity.cpp)
target_link_libraries(RouteIdentityTest PRIVATE LLKMapping LLKMicroMapping)
target_include_directories(RouteIdentityTest PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/Conversion/MicroMapping)

# Tuning-core tests (issue #49): the search-space loader, candidate
# generation, and machine-aware legality. The loader test parses micro IR;
# the other two build typed spaces directly, so a failure points at one layer.
add_llk_perf_test(SearchSpaceLoaderTest test/Perf/search_space_loader.cpp)
add_llk_perf_test(CandidateGeneratorTest test/Perf/candidate_generator.cpp)
add_llk_perf_test(LegalityTest test/Perf/legality.cpp)

# Candidate binding (issue #50): the point where a search-space candidate
# becomes a concrete micro.kernel. The test builds the kernel and asserts both
# the emitted IR and the decisions the schedule record persists.
add_llk_perf_test(CandidateBindingTest test/Perf/candidate_binding.cpp)

# Tuning session and schedule YAML (issue #50): ranking order, the session loop,
# and the persisted record. Uses the shipped machine model, so the model the
# driver tunes against is the one under test.
add_llk_perf_test(TuningSessionTest test/Perf/tuning_session.cpp)

# Machine topology tests (issue #82, epic #67): the v2 model, its normalized
# queries, validation, and the YAML loader. They are plain-data tests -- no
# micro IR -- so they link only LLKMachine and gtest.
function(add_llk_machine_test test_name test_file)
  add_llvm_executable(${test_name} ${test_file})
  target_link_libraries(${test_name} PRIVATE LLKMachine GTest::gtest_main)
  target_compile_definitions(${test_name} PRIVATE
      LLK_MACHINE_DIR="${CMAKE_SOURCE_DIR}/machines")
  add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

add_llk_machine_test(MachineModelTest test/Machine/machine_model.cpp)
add_llk_machine_test(MachineModelV2LoaderTest test/Machine/machine_model_loader.cpp)

# Mapping-engine tests (epic #67, workstream D1). They parse micro IR to build
# workload graphs, so they follow the same MLIR exec-lib rule as the perf tests:
# name MLIR components only through LLK_MLIR_EXEC_LIBS, never alongside it.
function(add_llk_mapping_test test_name test_file)
  add_llvm_executable(${test_name} ${test_file})
  target_link_libraries(${test_name} PRIVATE
      LLKMapping LLKTargetMapping MicroDialect GTest::gtest_main ${LLK_MLIR_EXEC_LIBS})
  target_compile_definitions(${test_name} PRIVATE
      LLK_MAPPING_DIR="${CMAKE_SOURCE_DIR}/mapping"
      LLK_MACHINE_DIR="${CMAKE_SOURCE_DIR}/machines"
      LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

add_llk_mapping_test(MappingStableHashTest test/Mapping/stable_hash.cpp)
add_llk_mapping_test(MappingCostModelTest test/Mapping/cost_model.cpp)
add_llk_mapping_test(MappingSearchBindingTest test/Mapping/search_binding.cpp)
add_llk_mapping_test(MappingWorkloadGraphTest test/Mapping/workload_graph.cpp)
add_llk_mapping_test(MappingTileFactsTest test/Mapping/tile_facts.cpp)
add_llk_mapping_test(MappingPlanTest test/Mapping/mapping_plan.cpp)
add_llk_mapping_test(MappingLayoutTest test/Mapping/layout_constraints.cpp)
add_llk_mapping_test(MappingRulesTest test/Mapping/mapping_rules.cpp)
add_llk_mapping_test(MappingPlacementTest test/Mapping/placement.cpp)
add_llk_mapping_test(MappingConnectionsTest test/Mapping/connections.cpp)
add_llk_mapping_test(MappingCoveringSearchTest test/Mapping/covering_search.cpp)
add_llk_mapping_test(MappingStoragePlanTest test/Mapping/storage_plan.cpp)
add_llk_mapping_test(MappingAvx2TargetTest test/Mapping/avx2_target.cpp)
add_llk_mapping_test(MappingGenericAcceleratorTargetTest test/Mapping/generic_accelerator_target.cpp)
# MappingPlanBinderTest binds through the canonical PlanMaterializer, whose
# construction lives in the Micro mapping pass library (LLKMicroMapping) -- the
# target-neutral mapping core has no dialect, so the test reaches the factory
# through MicroMappingCommon.h exactly as the CLI does.
add_llvm_executable(MappingPlanBinderTest test/Mapping/plan_binder.cpp)
target_link_libraries(MappingPlanBinderTest PRIVATE
    LLKMapping LLKTargetMapping LLKMicroMapping MicroDialect GTest::gtest_main
    ${LLK_MLIR_EXEC_LIBS})
target_include_directories(MappingPlanBinderTest PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/Conversion/MicroMapping)
target_compile_definitions(MappingPlanBinderTest PRIVATE
    LLK_MAPPING_DIR="${CMAKE_SOURCE_DIR}/mapping"
    LLK_MACHINE_DIR="${CMAKE_SOURCE_DIR}/machines"
    LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME MappingPlanBinderTest COMMAND MappingPlanBinderTest)
add_llk_mapping_test(MappingE2EWorkflowTest test/Mapping/e2e_workflow.cpp)
add_llk_mapping_test(MappingRoutingTest test/Mapping/routing.cpp)
add_llk_mapping_test(MappingPlanReportTest test/Mapping/plan_report.cpp)
add_llk_mapping_test(MappingPropertiesTest test/Mapping/properties.cpp)

# SearchBinding loader (phase-4 task 1): micro.candidate -> mapping::SearchBinding.
# The loader lives in the pass library, not LLKMapping, because it reuses
# llk::perf::loadSearchSpace to read the parameters -- and LLKPerf depends on
# LLKMapping, so the loader cannot live in the reverse direction. The test reads
# the checked-in fixture by path, so the same file the FileCheck round-trip
# covers is the one the loader is driven against.
add_llvm_executable(SearchBindingLoaderTest
    test/Conversion/MicroMapping/search_binding_loader.cpp)
target_link_libraries(SearchBindingLoaderTest PRIVATE
    LLKMicroMapping MicroDialect GTest::gtest_main ${LLK_MLIR_EXEC_LIBS})
target_include_directories(SearchBindingLoaderTest PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/Conversion/MicroMapping)
target_compile_definitions(SearchBindingLoaderTest PRIVATE
    LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
add_test(NAME SearchBindingLoaderTest COMMAND SearchBindingLoaderTest)

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
      # Arguments after the input file are placed on the first llk-opt
      # invocation only, so a test can select a pass; the second invocation
      # stays the bare parse/print round-trip.
      add_test(NAME ${test_name}
          COMMAND sh -c "$<TARGET_FILE:llk-opt> ${ARGN} ${CMAKE_SOURCE_DIR}/${input_file} | $<TARGET_FILE:llk-opt> | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/${input_file}"
      )
  endfunction()

  add_llk_filecheck_test(DialectOps test/Dialect/ops.mlir)
  add_llk_filecheck_test(MicroDialectOps test/Dialect/Micro/ops.mlir)
  add_llk_filecheck_test(MicroDialectTileOps test/Dialect/Micro/tile_ops.mlir)
  add_llk_filecheck_test(MicroDialectSearchSpace test/Dialect/Micro/search_space.mlir)
  add_llk_filecheck_test(MicroMappingSearchBinding test/Conversion/MicroMapping/search_binding.mlir)

  # MicroMappingMap: the --micro-map pass runs the mapping chain (extract ->
  # search -> bind) and writes the selected plan's metadata onto the kernel.
  # The pass options are quoted so sh hands llk-opt one argv token; the whole
  # option string is the user-facing contract from design §21.
  add_llk_filecheck_test(MicroMappingMap test/Conversion/MicroMapping/micro_map.mlir
      "\"--micro-map=target=x86-avx2 machine=${CMAKE_SOURCE_DIR}/machines/x86-avx2-v2.yaml layouts=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/layouts.llkmap rules=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store mode=deterministic\"")

  # MicroMappingVerify: phase 2 of the layered verification (design §18.3).
  # The pipeline chains `--micro-map` (which binds a plan and records its ids)
  # into `--micro-verify-mapping` (which loads the same target and resolves every
  # recorded id). A successful verify leaves the IR byte-identical, so the
  # FileCheck assertions are the plan/mapping metadata MicroMappingMap checks --
  # a verifier that mutated would have to break them. Raw add_test (rather than
  # add_llk_filecheck_test) because two passes are chained with their own option
  # strings; the helper's ARGN list would hand sh a command separator.
  add_test(NAME MicroMappingVerify
      COMMAND sh -c "$<TARGET_FILE:llk-opt> \"--micro-map=target=x86-avx2 machine=${CMAKE_SOURCE_DIR}/machines/x86-avx2-v2.yaml layouts=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/layouts.llkmap rules=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store mode=deterministic\" ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/verify_mapping.mlir | $<TARGET_FILE:llk-opt> \"--micro-verify-mapping=target=x86-avx2 machine=${CMAKE_SOURCE_DIR}/machines/x86-avx2-v2.yaml layouts=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/layouts.llkmap rules=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store\" | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/verify_mapping.mlir"
  )

  # MicroMappingVerifyInvalid: the negative half of phase 2. Each chunk of the
  # fixture is mapped IR with one recorded id that does not resolve (or a module
  # with no kernel at all), and `--verify-diagnostics` asserts the stable §22.3
  # code *and* the specific detail on the diagnostic. The same test also pins
  # that a fully resolved chunk produces no diagnostic.
  add_test(NAME MicroMappingVerifyInvalid
      COMMAND $<TARGET_FILE:llk-opt> --verify-diagnostics --split-input-file
              "--micro-verify-mapping=target=x86-avx2 machine=${CMAKE_SOURCE_DIR}/machines/x86-avx2-v2.yaml layouts=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/layouts.llkmap rules=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store"
              ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/verify_mapping_invalid.mlir
  )

  # MicroMappingUnmaterialized: the canonical tile-transfer fixture is now a
  # *materialized* plan (task B4): the CLI supplies the canonical
  # PlanMaterializer, so the vector->tile-store edge becomes a typed
  # `micro.tile_async_copy` + `micro.wait` into the store's memory, and the
  # store's operand is rewired to the destination-memory tile. The fixture
  # runs with require-executable=1, so a warning about an unmaterialized
  # connection is itself a failure. stderr is merged because the pass's
  # diagnostics go there while FileCheck reads stdout. Raw add_test because
  # add_llk_filecheck_test() pipes stdout only and cannot set --check-prefix.
  add_test(NAME MicroMappingUnmaterialized
      COMMAND sh -c "$<TARGET_FILE:llk-opt> \"--micro-map=target=probe machine=${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/probe_machine.yaml layouts=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/layouts.llkmap rules=${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/unmaterialized_rules.llkmap emitters=e1 mode=deterministic require-executable=1\" ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/micro_map_unmaterialized.mlir 2>&1 | ${FILECHECK_BIN} --check-prefix=MAP ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/micro_map_unmaterialized.mlir"
  )

  # MicroMappingRequireExecutable: the executable binding contract (design
  # §18.2) is satisfied by the canonical materializer -- the same plan is now
  # fully executable, so the backend-facing run succeeds and emits the
  # materialized tile movement rather than refusing an omitted decision. A
  # null materializer still fails Executable (covered by the unit tests in
  # MappingPlanBinderTest), but the CLI always supplies one.
  add_test(NAME MicroMappingRequireExecutable
      COMMAND sh -c "$<TARGET_FILE:llk-opt> \"--micro-map=target=probe machine=${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/probe_machine.yaml layouts=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/layouts.llkmap rules=${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/unmaterialized_rules.llkmap emitters=e1 mode=deterministic require-executable=1\" ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/micro_map_unmaterialized.mlir 2>&1 | grep -q 'micro.tile_async_copy'"
  )

  # MicroMappingBarrier: the storage plan (task B3) is now run on the selected
  # plan (task C1), so a movement that needs more than one transfer engine is
  # barrier-synchronized and the canonical materializer emits a `micro.barrier`.
  # The fixture routes SRAM -> L2 -> DRAM (two links, two engines) on
  # barrier_machine.yaml, so `requiresBarrier` is set and the barrier is
  # observable in the IR. Raw add_test because FileCheck needs --check-prefix and
  # stderr must be merged (the pass's diagnostics go to stderr).
  add_test(NAME MicroMappingBarrier
      COMMAND sh -c "$<TARGET_FILE:llk-opt> \"--micro-map=target=barrier machine=${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/barrier_machine.yaml layouts=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/layouts.llkmap rules=${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/unmaterialized_rules.llkmap emitters=e1 mode=deterministic require-executable=1\" ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/barrier_sync.mlir 2>&1 | ${FILECHECK_BIN} --check-prefix=MAP ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/barrier_sync.mlir"
  )

  # MicroMappingBarrierStorage: the reported and verified half of C1 -- the
  # report's selectedState carries populated allocations, synchronization and the
  # plan-step DAG, and B6's barrier verification fires (deleting the barrier
  # makes verification fail). A script because it compares the IR, the report
  # file, and two verification outcomes.
  add_test(NAME MicroMappingBarrierStorage
      COMMAND sh ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/barrier_storage.sh
          $<TARGET_FILE:llk-opt> ${CMAKE_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}/MicroMappingBarrierStorage
  )

  # MicroMappingReport: the `report=<path>` option's contract -- the report file
  # is produced, the mapped IR is byte-identical with and without it (the report
  # is metadata), and two separate llk-opt runs produce identical reports. A
  # script because FileCheck only speaks about one stdout stream, while this
  # compares two IR outputs and two report files.
  add_test(NAME MicroMappingReport
      COMMAND sh ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/micro_map_report.sh
          $<TARGET_FILE:llk-opt> ${CMAKE_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}/MicroMappingReport
  )

  # MicroMappingReportOnly: `report-only=1` runs the search and writes the
  # report but leaves the module untouched (design §21: "emitting a plan report
  # without modifying input IR"). A script because the assertions span two
  # files -- the report and the printed IR -- and one of them is a diff of the
  # output against the input kernel itself, byte for byte.
  add_test(NAME MicroMappingReportOnly
      COMMAND sh ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/micro_map_report_only.sh
          $<TARGET_FILE:llk-opt> ${CMAKE_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}/MicroMappingReportOnly
  )

  # MicroMappingCandidate: `candidate=<sym>` drives `--micro-map` (and, for the
  # replay half, `--micro-bind-plan`) from a persistent `micro.candidate`
  # (phase-4 task 4). The candidate's values pin rule parameters and its
  # `layout`-kind parameter -- resolved by kind, not by name -- selects the
  # bound layout, so the plan the pass binds is traceable to the search-space
  # point it came from. Because the plan id folds in the binding's hash, the
  # round trip needs the same `candidate=` on both runs (ruling S8); this test
  # proves it, and that the id is genuinely absent from a binding-free replay.
  # A script because the binding's hash is read back out of the report file and
  # compared against the bound IR, and because the rejection and namespace cases
  # are separate invocations whose diagnostics are asserted.
  add_test(NAME MicroMappingCandidate
      COMMAND sh ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/micro_map_candidate.sh
          $<TARGET_FILE:llk-opt> ${CMAKE_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}/MicroMappingCandidate
  )

  # MicroMappingBindPlanId: `--micro-bind-plan` accepts the id the report prints
  # (bare hex), plus 0x-hex and signed/unsigned decimal, and rejects anything
  # else. End-to-end because the id is obtained by running `--micro-map report=`
  # first -- the documented report -> bind workflow. A script because the id is
  # read out of a file and fed back into a second invocation.
  add_test(NAME MicroMappingBindPlanId
      COMMAND sh ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/micro_bind_plan_id.sh
          $<TARGET_FILE:llk-opt> ${CMAKE_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}/MicroMappingBindPlanId
  )

  # MicroMappingBindPlanMode: a plan id is a content hash, so reproducing one
  # means replaying the search -- same mode, beam-width, and top-k. This pins
  # that `--micro-bind-plan` takes those options and honours them: a beam id
  # round-trips in beam mode, and a deterministic id is absent from a beam
  # search with the same cap (the diagnostic names the mode it searched).
  add_test(NAME MicroMappingBindPlanMode
      COMMAND sh ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/micro_bind_plan_mode.sh
          $<TARGET_FILE:llk-opt> ${CMAKE_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}/MicroMappingBindPlanMode
  )

  # MicroMappingBindPlanRoundtrip: the strongest form of the report -> bind
  # contract -- binding the reported id reproduces the *same* plan, not merely
  # *a* plan, so the bound IR is byte-identical to the `--micro-map report=`
  # output. The id-parsing and mode tests above prove the id is accepted and the
  # search is replayed; this one proves the replayed search selects the plan the
  # report named. Three cases: deterministic and beam over the hand-authored
  # kernel, and exact over IR that `--llk-to-micro` produced, so the round-trip
  # is covered on compiler-generated IR too. A script because it compares two
  # IR streams the way FileCheck cannot.
  #
  # The ids here are all binding-free, so the replay passes no `candidate=`. A
  # plan id folds the search point in, so a binding-derived id needs the same
  # `candidate=` on both runs; MicroMappingCandidate covers that round trip
  # (ruling S8).
  add_test(NAME MicroMappingBindPlanRoundtrip
      COMMAND sh ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/micro_bind_plan_roundtrip.sh
          $<TARGET_FILE:llk-opt> ${CMAKE_SOURCE_DIR}
          ${CMAKE_CURRENT_BINARY_DIR}/MicroMappingBindPlanRoundtrip
  )

  # MicroMappingMatmulE2E / MicroMappingSwiGLUE2E: the gap-#3 acceptance. Each
  # runs the two-pass pipeline `--llk-to-micro | --micro-map mode=exact` over
  # the LLK source, so the kernel being mapped is genuinely compiler-generated
  # (the input is the same LLK program as the matching LLKToMicro fixture), and
  # asserts the shipped AVX2 target binds a complete plan onto it: a
  # `micro.plan` on the kernel and a `micro.mapping` on every covered node,
  # including the `micro.vector` op variants the lowering emits and the
  # `micro.tile_async_copy`/`micro.tile_store` movement ops.
  add_test(NAME MicroMappingMatmulE2E
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-micro=\"schedule-db=missing.json\" ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/matmul_e2e.mlir | $<TARGET_FILE:llk-opt> \"--micro-map=target=x86-avx2 machine=${CMAKE_SOURCE_DIR}/machines/x86-avx2-v2.yaml layouts=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/layouts.llkmap rules=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store mode=exact\" | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/matmul_e2e.mlir"
  )
  add_test(NAME MicroMappingSwiGLUE2E
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-micro=\"schedule-db=missing.json\" ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/swiglu_e2e.mlir | $<TARGET_FILE:llk-opt> \"--micro-map=target=x86-avx2 machine=${CMAKE_SOURCE_DIR}/machines/x86-avx2-v2.yaml layouts=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/layouts.llkmap rules=${CMAKE_SOURCE_DIR}/mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store mode=exact\" | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/MicroMapping/swiglu_e2e.mlir"
  )

  # LLKToMicro lowering tests. The schedule database is pointed at a path that
  # does not exist on purpose: the export then uses its built-in conservative
  # schedule, so the expected tile sizes do not depend on the working directory
  # or on the contents of schedules/schedule_db.json.
  add_test(NAME LLKToMicroSwiGLU
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-micro=\"schedule-db=missing.json\" ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/swiglu_to_micro.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/swiglu_to_micro.mlir"
  )
  add_test(NAME LLKToMicroMatmul
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-micro=\"schedule-db=missing.json\" ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/matmul_to_micro.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/matmul_to_micro.mlir"
  )
  # The search-space export, anchored on a fixture database that spells out the
  # Micro/tile schedule fields for one operation and leaves the other on the
  # pre-M11 fields, so one run covers both loaded and defaulted values.
  add_test(NAME LLKToMicroSearchSpace
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-micro-search-space=\"schedule-db=${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/search_space_schedule.json\" ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/search_space_from_schedule.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/search_space_from_schedule.mlir"
  )
  # The M-bucket rules prune BM choices: bucket 0 allows only BM = 1, and the
  # large-M buckets reject tiles of 4 rows or fewer.
  add_test(NAME LLKToMicroSearchSpaceMBucket
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-micro-search-space=\"schedule-db=missing.json\" ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/search_space_m_bucket.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/search_space_m_bucket.mlir"
  )
  # `llk-compile --emit=micro-search` prints the search space and exits without
  # reaching the JIT: the input's llk ops are never lowered to LLVM.
  add_test(NAME LLKCompileMicroSearch
      COMMAND sh -c "$<TARGET_FILE:llk-compile> --emit=micro-search ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/search_space_from_schedule.mlir | ${FILECHECK_BIN} --check-prefix=COMPILE ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/search_space_from_schedule.mlir"
  )
  # The acceptance check that the exported Micro-IR is self-describing: the
  # performance simulator evaluates the kernel the export produced, using only
  # the metadata carried by the IR.
  add_test(NAME LLKToMicroPerfRoundTrip
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-micro=\"schedule-db=missing.json\" ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/swiglu_to_micro.mlir | $<TARGET_FILE:micro-perf> --machine=${CMAKE_SOURCE_DIR}/machines/x86-avx2-v2.yaml --level=1 - | ${FILECHECK_BIN} --check-prefix=PERF ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/swiglu_to_micro.mlir"
  )
  # LLKToLinalgConversion test: run llk-opt --llk-to-linalg then FileCheck
  add_test(NAME LLKToLinalgConversion
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg ${CMAKE_SOURCE_DIR}/test/Conversion/llk_to_linalg.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/llk_to_linalg.mlir"
  )

  # MicroToLinalg: the bridge from canonical Micro-IR to the backend pipeline.
  # A `micro.kernel` becomes a `func.func` over memrefs and Linalg, which is
  # what lets a mapped or exported kernel reach LLVM and the JIT.
  add_test(NAME MicroToLinalg
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --micro-to-linalg ${CMAKE_SOURCE_DIR}/test/Conversion/MicroToLinalg/micro_to_linalg.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/MicroToLinalg/micro_to_linalg.mlir"
  )
  # MicroToLinalgInvalid: an op with no lowering must fail the conversion
  # loudly rather than being silently left in the module. Each rejection needs
  # its own chunk: the conversion aborts at the first illegal op, so one module
  # can only ever report one of them.
  add_test(NAME MicroToLinalgInvalid
      COMMAND $<TARGET_FILE:llk-opt> --verify-diagnostics --split-input-file
              --micro-to-linalg
              ${CMAKE_SOURCE_DIR}/test/Conversion/MicroToLinalg/micro_to_linalg_invalid.mlir
  )
  # MicroToLinalgJit: the lowered kernel reaches the JIT. This is the check that
  # the bridge produces IR the backend pipeline can finish -- which is why the
  # lowering stays in tensor land: an already-bufferized module leaves an
  # unrealized vector cast that LLVM translation rejects.
  add_test(NAME MicroToLinalgJit
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --micro-to-linalg ${CMAKE_SOURCE_DIR}/test/Conversion/MicroToLinalg/micro_to_linalg_jit.mlir | $<TARGET_FILE:llk-compile> - | grep -q 'Compilation successful'"
  )
  # math_mode=triton_fast on the SiLU path: the sigmoid's exp must take the
  # bounded-fast (math.exp2) path, like bounded_fast and unsafe_fast do.
  add_test(NAME LLKToLinalgTritonFast
      COMMAND sh -c "$<TARGET_FILE:llk-opt> --llk-to-linalg ${CMAKE_SOURCE_DIR}/test/Conversion/llk_to_linalg_triton_fast.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Conversion/llk_to_linalg_triton_fast.mlir"
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
      COMMAND sh -c "$<TARGET_FILE:micro-perf> --machine=${CMAKE_SOURCE_DIR}/machines/x86-avx2-v2.yaml --level=1 ${CMAKE_SOURCE_DIR}/test/Perf/micro_perf_cli.mlir | ${FILECHECK_BIN} ${CMAKE_SOURCE_DIR}/test/Perf/micro_perf_cli.mlir"
  )
  # The help text has to say what the MVP is, so nobody reads "simulator" as
  # "functional emulator". Same fixture, other check prefix.
  add_test(NAME MicroPerfHelp
      COMMAND sh -c "$<TARGET_FILE:micro-perf> --help 2>&1 | ${FILECHECK_BIN} --check-prefix=HELP ${CMAKE_SOURCE_DIR}/test/Perf/micro_perf_cli.mlir"
  )
  # llk-tune in Micro mode: a search space and the AVX2 model go in, a ranked
  # schedule YAML comes out. The fixture's YAML checks describe that output.
  add_test(NAME LLKTuneSearchSpace
      COMMAND sh -c "$<TARGET_FILE:llk-tune> --input=${CMAKE_SOURCE_DIR}/test/Perf/llk_tune_search_space.mlir --machine=${CMAKE_SOURCE_DIR}/machines/x86-avx2-v2.yaml --M=8 --N=64 --K=64 --top-k=2 --output=${CMAKE_CURRENT_BINARY_DIR}/llk_tune_out.yaml && ${FILECHECK_BIN} --check-prefix=YAML ${CMAKE_SOURCE_DIR}/test/Perf/llk_tune_search_space.mlir < ${CMAKE_CURRENT_BINARY_DIR}/llk_tune_out.yaml"
  )
  # Compatibility: with no --input the driver keeps the pre-Micro grid search
  # and writes the JSON schedule_db entry the existing pipeline consumes.
  add_test(NAME LLKTuneLegacyJson
      COMMAND sh -c "$<TARGET_FILE:llk-tune> -M=8 -N=64 -K=64 -o=${CMAKE_CURRENT_BINARY_DIR}/llk_tune_legacy.json && ${FILECHECK_BIN} --check-prefix=LEGACY ${CMAKE_SOURCE_DIR}/test/Perf/llk_tune_search_space.mlir < ${CMAKE_CURRENT_BINARY_DIR}/llk_tune_legacy.json"
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
# LLKToMicro rejection tests: shapes and schedules the export must refuse with
# a diagnostic naming the offending axis.
add_test(NAME LLKToMicroLoweringInvalid
    COMMAND $<TARGET_FILE:llk-opt> --verify-diagnostics --split-input-file
            --llk-to-micro=schedule-db=missing.json
            ${CMAKE_SOURCE_DIR}/test/Conversion/LLKToMicro/lowering_invalid.mlir
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
if(TARGET llk-tune)
  list(APPEND _llk_check_deps llk-tune)
endif()

add_custom_target(check-llk
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS ${_llk_check_deps}
    USES_TERMINAL
    COMMENT "Running the LLK test suite"
)
