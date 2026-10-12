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
