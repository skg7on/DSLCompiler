# Assert LLKFrontendCAPI exports exactly the declared C entry points.
#
# The project archives are built with the default visibility, so the export set
# comes from a linker export list, not from source annotations. That makes this a
# property of the build rather than of the code, and it differs between Mach-O
# and ELF, so it is checked on whichever platform is running the tests. Loading
# an extra symbol into a Python process could interpose another MLIR copy.
if(NOT DEFINED LIBRARY OR NOT DEFINED NM)
  message(FATAL_ERROR "LIBRARY and NM are required")
endif()

set(_declared
    llk_frontend_api_version
    llk_frontend_compile_v1
    llk_frontend_invoke_v1
    llk_frontend_report_v1
    llk_frontend_inspect_v1
    llk_frontend_error_json_v1
    llk_frontend_error_free_v1
    llk_frontend_release_v1)

if(APPLE)
  # External symbols only, undefined references excluded; Mach-O names carry a
  # leading underscore.
  set(_nm_arguments -gU)
  set(_prefix "_")
else()
  set(_nm_arguments -D --defined-only)
  set(_prefix "")
endif()

execute_process(COMMAND "${NM}" ${_nm_arguments} "${LIBRARY}"
                OUTPUT_VARIABLE _output RESULT_VARIABLE _result)
if(NOT _result EQUAL 0)
  message(FATAL_ERROR "nm ${_nm_arguments} failed on ${LIBRARY}")
endif()

set(_exported "")
string(REPLACE "\n" ";" _lines "${_output}")
foreach(_line IN LISTS _lines)
  string(STRIP "${_line}" _line)
  if(_line STREQUAL "")
    continue()
  endif()
  string(REGEX MATCH "[^ \t]+$" _symbol "${_line}")
  if(_prefix)
    string(REGEX REPLACE "^${_prefix}" "" _symbol "${_symbol}")
  endif()
  list(APPEND _exported ${_symbol})
endforeach()

list(SORT _exported)
list(SORT _declared)
if(NOT _exported STREQUAL _declared)
  message(FATAL_ERROR
          "unexpected export set\n  actual:   ${_exported}\n"
          "  declared: ${_declared}")
endif()
message(STATUS "exports exactly: ${_declared}")
