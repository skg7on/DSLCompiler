add_test(NAME CMakeConfiguration
    COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/check_configuration.py"
        --source-dir "${PROJECT_SOURCE_DIR}"
        --llvm-build "${LLVM_PROJECT_BUILD_DIR}")
set_tests_properties(CMakeConfiguration PROPERTIES LABELS "CMake" TIMEOUT 120)

add_test(NAME CMakeImportedTargets
    COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/check_imported_targets.py"
        --source-dir "${PROJECT_SOURCE_DIR}")
set_tests_properties(CMakeImportedTargets PROPERTIES LABELS "CMake" TIMEOUT 120)

add_test(NAME CMakeCompilerSelection
    COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/check_compiler_selection.py"
        --source-dir "${PROJECT_SOURCE_DIR}")
set_tests_properties(CMakeCompilerSelection PROPERTIES LABELS "CMake" TIMEOUT 120)
