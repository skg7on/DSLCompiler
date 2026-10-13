add_llk_test_category(Benchmark
    SOURCES
        ${CMAKE_SOURCE_DIR}/benchmark/schedule_regression.cpp
)

if(LLK_BUILD_TOOLS)
  # libLLVM also registers a ThinLTO --threads option. The benchmark must
  # start successfully and route this flag to its own worker count.
  add_test(NAME BenchmarkThreadOption
      COMMAND llk-bench -M=2 -N=16 -K=16 --threads=2
          --warmup-ms=0 --measure-ms=0 --reps=1)
  set_tests_properties(BenchmarkThreadOption PROPERTIES
      LABELS "Benchmark" PASS_REGULAR_EXPRESSION "threads=2")
endif()
