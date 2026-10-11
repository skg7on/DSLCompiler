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

# LLVM's target helpers use these variables instead of CMake's defaults.
# Set both families before creating any project or third-party targets.
set(LLVM_RUNTIME_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/bin")
set(LLVM_LIBRARY_OUTPUT_INTDIR "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
foreach(_llk_config IN LISTS CMAKE_CONFIGURATION_TYPES)
  string(TOUPPER "${_llk_config}" _llk_config_upper)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_llk_config_upper} "${CMAKE_BINARY_DIR}/bin")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${_llk_config_upper} "${CMAKE_BINARY_DIR}/lib")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${_llk_config_upper} "${CMAKE_BINARY_DIR}/lib")
endforeach()

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
    lib/Perf/SelectedKernelAnalysis.cpp
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
    lib/Mapping/StorageLiveness.cpp
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
    DEPENDS LLKDialect MicroDialect LLKCommon MLIRLinalgDialect
    LINK_LIBS PUBLIC LLKDialect MicroDialect LLKCommon
)
mlir_target_link_libraries(LLKToMicro PUBLIC
    MLIRArithDialect MLIRFuncDialect MLIRLinalgDialect MLIRTensorDialect
    MLIRPass MLIRTransforms
)

# Library: micro -> Linalg bridge. `micro` was a sink: nothing lowered it to
# something a backend consumes, so a mapped or exported micro.kernel could not
# reach the LLVM/JIT pipeline. This is that bridge.
add_mlir_library(LLKMicroToLinalg
    lib/Conversion/MicroToLinalg/MicroToLinalg.cpp
    DEPENDS MicroDialect
    LINK_LIBS PUBLIC MicroDialect
)

add_library(LLKMappedKernelAbi STATIC
    lib/Conversion/MicroMapping/MappedKernelAbi.cpp
)
target_include_directories(LLKMappedKernelAbi PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${LLVM_INCLUDE_DIRS}
)
target_compile_options(LLKMappedKernelAbi PRIVATE -fno-rtti -fno-exceptions)
mlir_target_link_libraries(LLKMappedKernelAbi PUBLIC
    MLIRBufferizationPipelines MLIRBufferizationTransforms MLIRSCFTransforms
    MLIRFuncDialect MLIRMemRefDialect MLIRSCFDialect MLIRPass MLIRIR
)
mlir_target_link_libraries(LLKMicroToLinalg PUBLIC
    MLIRArithDialect MLIRFuncDialect MLIRLinalgDialect MLIRMemRefDialect
    MLIRSCFDialect MLIRTensorDialect MLIRPass MLIRTransforms
)

# Library: JIT cache runtime + weight packing
add_library(LLKRuntime STATIC
    runtime/JitCache.cpp
    runtime/MappedExecutable.cpp
    runtime/MappedInvocation.cpp
    runtime/PackedWeights.cpp
    runtime/ThreadPool.cpp
)
target_include_directories(LLKRuntime PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${LLVM_INCLUDE_DIRS}
    PRIVATE ${CMAKE_SOURCE_DIR}/test/Execution
)
# Resolve native target libraries dynamically for the build host.
# Equivalent to: llvm-config --libs native
# On x86_64 → LLVMX86{CodeGen,Info,Desc,AsmParser,AsmPrinter,Disassembler}
# On AArch64 → LLVMAArch64{CodeGen,Info,Desc,AsmParser,AsmPrinter,Disassembler}
llvm_map_components_to_libnames(LLVM_NATIVE_TARGET_LIBS native)
target_link_libraries(LLKRuntime PUBLIC
    LLKMappedKernelAbi
    LLVMOrcJIT
    ${LLVM_NATIVE_TARGET_LIBS}
)
# LLVM and MLIR are built without RTTI or exceptions; a plain add_library does
# not inherit those. Linking `llvm::Error` values through the ORC interfaces
# needs the same flags, or the link fails on `typeinfo for llvm::ErrorInfoBase`.
target_compile_options(LLKRuntime PRIVATE -fno-rtti -fno-exceptions)
# MLIR's C runtime: a lowered `memref.copy` calls `memrefCopy`, which the
# finalize pass emits as a call into the runtime rather than inlining. Every
# consumer of this library needs it, so it is named outside the MLIR-target
# helper, which only closes over MLIR's own components.
target_link_libraries(LLKRuntime PUBLIC mlir_c_runner_utils)
mlir_target_link_libraries(LLKRuntime PUBLIC
    # `createLowerAffinePass` lives here. It reaches the binary through MLIR's
    # dylib in a shared-MLIR build, so a local build links it by accident; a
    # static-MLIR build needs it named, exactly like the rest of this list.
    MLIRAffineToStandard
    MLIRExecutionEngine
    MLIRArithToLLVM
    MLIRControlFlowToLLVM
    MLIRFuncToLLVM
    MLIRMathToLLVM
    MLIRMemRefToLLVM
    MLIRMemRefTransforms
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

# Library: target mapping packages (issue #67, workstreams D7 and C2). Each
# target owns its configuration paths, emitter keys, and -- since C2 -- the
# lowering of a selected bundle. Generic mapping code sees only the
# MappingTarget interface. Kept out of LLKTargetX86 so CPU feature detection
# does not drag in MLIR; the lowering does, because a target lowerer rewrites
# Micro operations and so is the layer that knows the dialect.
add_library(LLKTargetMapping STATIC
    lib/Target/MappingTargets.cpp
    lib/Target/X86/Mapping/AVX2MappingTarget.cpp
    lib/Target/X86/Mapping/AVX2BundleLowering.cpp
    lib/Target/X86/Mapping/AVX2BackendLowering.cpp
    lib/Target/GenericAccelerator/Mapping/GenericAcceleratorMappingTarget.cpp
)
target_include_directories(LLKTargetMapping PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)
target_link_libraries(LLKTargetMapping PUBLIC LLKMapping LLKMachine MicroDialect)
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
    lib/Conversion/MicroMapping/CandidateInstantiation.cpp
    lib/Conversion/MicroMapping/PlanMaterialization.cpp
    lib/Conversion/MicroMapping/CompletePlanEvaluation.cpp
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
target_link_libraries(LLKMicroMapping PUBLIC LLKMapping LLKTargetMapping LLKMachine LLKPerf LLKToMicro)
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
# Library: the shared mapped-compilation path (issue #67, stage C5). The one
# sequence from a selected plan to runnable code, taken by both the compiler and
# the tuner so a schedule's provenance does not depend on which tool asked for
# it. Kept out of LLKMicroMapping because it reaches all the way to the JIT: a
# caller that only wants to map a kernel should not link an execution engine.
add_library(LLKMappedCompilation STATIC
    lib/Conversion/MicroMapping/MappedCompilation.cpp
)
target_include_directories(LLKMappedCompilation PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${LLVM_INCLUDE_DIRS}
)
# The canonical plan materializer lives beside the source file; the include is
# private because it is an implementation detail of the Micro mapping library.
target_include_directories(LLKMappedCompilation PRIVATE
    ${CMAKE_SOURCE_DIR}/lib/Conversion/MicroMapping
    ${CMAKE_SOURCE_DIR}/test/Execution
)
target_link_libraries(LLKMappedCompilation PUBLIC
    LLKMicroMapping LLKMicroToLinalg LLKMappedKernelAbi LLKRuntime LLKMapping
    LLKTargetMapping
)
# LLVM and MLIR are built without RTTI or exceptions; a plain add_library does
# not inherit those. Match the toolchain, exactly as LLKMapping does.
target_compile_options(LLKMappedCompilation PRIVATE -fno-rtti -fno-exceptions)
mlir_target_link_libraries(LLKMappedCompilation PUBLIC
    MLIRBufferizationTransforms MLIRLinalgTransforms MLIRPass MLIRIR
)

# Target-aware tuning sits above both mapping and mapped compilation.
add_library(LLKMappedTuning STATIC
    lib/Conversion/MicroMapping/MappedTuningSession.cpp
)
target_include_directories(LLKMappedTuning PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${LLVM_INCLUDE_DIRS}
    PRIVATE ${CMAKE_SOURCE_DIR}/lib/Conversion/MicroMapping
)
target_link_libraries(LLKMappedTuning PUBLIC LLKMicroMapping LLKMappedCompilation)
target_compile_options(LLKMappedTuning PRIVATE -fno-rtti -fno-exceptions)

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
target_link_libraries(llk-compile PRIVATE mlir_c_runner_utils LLKPasses LLKTransforms LLKFuseDoubleContraction LLKPackWeights LLKScratchAnalysis LLKLinearizeForall LLKSerialParallelDispatch LLKForallToLLRT LLKShapeSpecialization LLKScheduleSelection LLKToMicro LLKMappedCompilation LLKTargetMapping LLKMapping LLKRuntime TritonToLLKPasses ${LLK_MLIR_EXEC_LIBS})

# Tool: llk-bench (benchmarking harness)
add_llvm_executable(llk-bench tools/llk-bench/llk-bench.cpp)
target_link_libraries(llk-bench PRIVATE LLKRuntime)

# Tool: llk-tune (autotuning: Micro search space, or the legacy grid)
add_llvm_executable(llk-tune tools/llk-tune/llk-tune.cpp)
target_link_libraries(llk-tune PRIVATE
    LLKPerf LLKMappedTuning LLKTargetMapping LLKMappedCompilation LLKMapping
    LLKMicroMapping LLKToMicro MicroDialect LLKDialect LLKRuntime
    ${LLK_MLIR_EXEC_LIBS})

# Tool: micro-perf (Micro-IR performance simulator, M10)
add_llvm_executable(micro-perf tools/micro-perf/micro-perf.cpp)
target_link_libraries(micro-perf PRIVATE LLKPerf MicroDialect LLKDialect ${LLK_MLIR_EXEC_LIBS})
endif()

# CTest must be enabled at the build root so ctest finds the test subdirectory.
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(third_party)
  add_subdirectory(test)
endif()

# Documentation is opt-in: normal builds and compiler tests need no Sphinx.
# Point this at a virtualenv with docs/manual/requirements.txt installed.
# Output lands in the build tree; the repository holds sources only, and the
# published copy is rendered by CI.
set(LLK_DOCS_PYTHON "python3" CACHE STRING "Python interpreter for the Sphinx manual")
add_custom_target(docs-html
    COMMAND ${LLK_DOCS_PYTHON} -m sphinx -b html -n -W --keep-going
            -d ${CMAKE_BINARY_DIR}/docs/doctrees
            ${CMAKE_SOURCE_DIR}/docs/manual/source ${CMAKE_BINARY_DIR}/docs/html
    USES_TERMINAL
    VERBATIM
    COMMENT "Building the MicroIR HTML developer manual"
)
