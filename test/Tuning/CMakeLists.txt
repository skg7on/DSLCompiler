if(LLK_BUILD_E2E_TESTS)
  add_executable(MappedTuningSessionTest mapped_tuning_session.cpp)
  target_link_libraries(MappedTuningSessionTest PRIVATE LLKMappedTuning
      GTest::gtest_main LLK::MLIR)
  target_compile_definitions(MappedTuningSessionTest PRIVATE
      LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  add_test(NAME MappedTuningSessionTest COMMAND MappedTuningSessionTest)

  add_executable(CandidateInstantiationTest candidate_instantiation.cpp)
  target_link_libraries(CandidateInstantiationTest PRIVATE LLKToMicro
      LLKMicroMapping LLKDialect MicroDialect GTest::gtest_main
      LLK::MLIR)
  target_compile_definitions(CandidateInstantiationTest PRIVATE
      LLK_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
  add_test(NAME CandidateInstantiationTest COMMAND CandidateInstantiationTest)

  add_test(NAME MappedTuneCLI
      COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/mapped_tune_cli.py
          $<TARGET_FILE:llk-tune> $<TARGET_FILE:llk-compile> ${CMAKE_SOURCE_DIR})
endif()
