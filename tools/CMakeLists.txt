add_executable(llk-opt llk-opt/llk-opt.cpp)
target_link_libraries(llk-opt PRIVATE LLKBuildOptions
    LLKToLinalg
    LLKTransforms
    LLKFuseDoubleContraction
    LLKPackWeights
    LLKScratchAnalysis
    LLKLinearizeForall
    LLKSerialParallelDispatch
    LLKForallToLLRT
    LLKForallToOpenMP
    LLKShapeSpecialization
    LLKScheduleSelection
    LLKToMicro
    LLKMicroToLinalg
    LLKMicroMapping
    TritonToLLKPasses
    MicroDialect
    MLIRMlirOptMain
    LLK::MLIR
)

# Tool: llk-compile
add_executable(llk-compile llk-compile/llk-compile.cpp)
target_link_libraries(llk-compile PRIVATE LLKBuildOptions
    mlir_c_runner_utils
    LLKToLinalg
    LLKTransforms
    LLKFuseDoubleContraction
    LLKPackWeights
    LLKScratchAnalysis
    LLKLinearizeForall
    LLKSerialParallelDispatch
    LLKForallToLLRT
    LLKShapeSpecialization
    LLKScheduleSelection
    LLKToMicro
    LLKMappedCompilation
    LLKMappedKernelAbi
    LLKTargetMapping
    LLKMapping
    LLKRuntime
    TritonToLLKPasses
    LLK::MLIR
)

# Tool: llk-bench (benchmarking harness)
add_executable(llk-bench llk-bench/llk-bench.cpp)
target_link_libraries(llk-bench PRIVATE LLKBuildOptions
    LLKRuntime
)

# Tool: llk-tune (autotuning: Micro search space, or the legacy grid)
add_executable(llk-tune llk-tune/llk-tune.cpp)
target_link_libraries(llk-tune PRIVATE LLKBuildOptions
    LLKPerf LLKMappedTuning LLKTargetMapping LLKMappedCompilation LLKMapping
    LLKMicroMapping LLKToMicro MicroDialect LLKDialect LLKRuntime
    LLK::MLIR)

# Tool: micro-perf (Micro-IR performance simulator, M10)
add_executable(micro-perf micro-perf/micro-perf.cpp)
target_link_libraries(micro-perf PRIVATE LLKBuildOptions
    LLKPerf
    MicroDialect
    LLKDialect
    LLK::MLIR
)

# Tool: llk-validate (verified-record statistics; no runtime/JIT integration)
add_executable(llk-validate llk-validate/llk-validate.cpp)
target_link_libraries(llk-validate PRIVATE LLKBuildOptions LLKPerf)
