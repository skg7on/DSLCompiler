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
