# Vendored third-party dependencies.
#
# These are pulled in through a subdirectory of their own so that the warning
# suppressions below stay scoped to third-party code: a directory-scoped
# setting reaches this directory and its children, and nothing else.

# Build GoogleTest from vendored source (never the system package) to avoid
# ABI incompatibilities with prebuilt system libgtest packages.
# googletest's cxx_library() macro auto-creates the GTest::gtest_main ALIAS.
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)

# HandleLLVMOptions() appends LLVM's warning set to CMAKE_CXX_FLAGS for the
# whole project, and googletest's sources are not written to it:
# gtest-death-test.cc has a `default:` arm on a switch that already covers every
# enumerator, which -Wcovered-switch-default flags. Upstream LLVM suppresses it
# the same way, in third-party/unittest/CMakeLists.txt.
if(CXX_SUPPORTS_COVERED_SWITCH_DEFAULT_FLAG)
  add_definitions("-Wno-covered-switch-default")
endif()

add_subdirectory(googletest)
