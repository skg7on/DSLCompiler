# Shared C ABI loaded by the Python frontend through ctypes. LLKFrontend and its
# dependencies stay statically linked; only the declared entry points are
# exported, so no MLIR/C++ symbol leaks into the dynamic symbol table.
add_library(LLKFrontendCAPI SHARED Frontend.cpp)
target_link_libraries(LLKFrontendCAPI PRIVATE LLKBuildOptions LLKFrontend)
target_compile_definitions(LLKFrontendCAPI PRIVATE LLK_FRONTEND_CAPI_EXPORTS)
set_target_properties(LLKFrontendCAPI PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

# Visibility attributes only cover this translation unit. The project archives
# it links were built with the default visibility, so a linker export list is
# what actually keeps LLK and LLVM symbols out of the dynamic symbol table --
# otherwise loading this library into Python would interpose another copy of
# MLIR. The JIT hands memrefCopy to LLJIT by address, so hiding it is safe.
set(_llk_capi_symbols
    llk_frontend_api_version
    llk_frontend_compile_v1
    llk_frontend_invoke_v1
    llk_frontend_report_v1
    llk_frontend_inspect_v1
    llk_frontend_error_json_v1
    llk_frontend_error_free_v1
    llk_frontend_release_v1)

if(APPLE)
  # Mach-O records a leading underscore on every symbol name.
  set(_llk_capi_exports "${CMAKE_CURRENT_BINARY_DIR}/LLKFrontendCAPI.exports")
  set(_llk_capi_export_lines "")
  foreach(_symbol IN LISTS _llk_capi_symbols)
    string(APPEND _llk_capi_export_lines "_${_symbol}\n")
  endforeach()
  file(WRITE "${_llk_capi_exports}" "${_llk_capi_export_lines}")
  target_link_options(LLKFrontendCAPI PRIVATE
      "LINKER:-exported_symbols_list,${_llk_capi_exports}")
elseif(UNIX)
  set(_llk_capi_version_script
      "${CMAKE_CURRENT_BINARY_DIR}/LLKFrontendCAPI.map")
  set(_llk_capi_global "")
  foreach(_symbol IN LISTS _llk_capi_symbols)
    string(APPEND _llk_capi_global "    ${_symbol};\n")
  endforeach()
  file(WRITE "${_llk_capi_version_script}"
      "{\n  global:\n${_llk_capi_global}  local: *;\n};\n")
  target_link_options(LLKFrontendCAPI PRIVATE
      "LINKER:--version-script=${_llk_capi_version_script}")
endif()
