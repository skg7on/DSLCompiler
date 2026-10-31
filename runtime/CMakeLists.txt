add_llk_library(LLKRuntime
    JitCache.cpp
    CompiledVariantIdentity.cpp
    MappedExecutable.cpp
    MappedInvocation.cpp
    MappedObject.cpp
    PackedWeights.cpp
    ThreadPool.cpp
    LINK_LIBRARIES LLKMappedKernelAbi LLKMapping LLKShape LLVM LLK::MLIR
                   mlir_c_runner_utils
)

# The JIT uses the shared invocation helpers from the execution tests.
target_include_directories(LLKRuntime PRIVATE "${PROJECT_SOURCE_DIR}/test/Execution")
