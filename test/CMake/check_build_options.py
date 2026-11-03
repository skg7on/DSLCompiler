"""Compile consumers of the build options using LLVM-style definition flags."""

import argparse
import pathlib
import subprocess
import tempfile
import unittest


class BuildOptionsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name).resolve()

    def compile_consumer(self, abi):
        (self.root / "probe.cpp").write_text(
            '#if !defined(__STDC_CONSTANT_MACROS) || '
            '!defined(__STDC_FORMAT_MACROS) || !defined(__STDC_LIMIT_MACROS)\n'
            '#error LLVM definitions were not passed as separate arguments\n'
            '#endif\n'
            f'static_assert(_GLIBCXX_USE_CXX11_ABI == {abi}, "LLVM ABI changed");\n'
            'int main() { return 0; }\n'
        )
        (self.root / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.20)\n'
            'project(BuildOptionsConsumer LANGUAGES CXX)\n'
            # HandleLLVMOptions first sets directory definitions, then exports
            # them again in a single LLVM_DEFINITIONS command-line string.
            f'add_compile_definitions(_GLIBCXX_USE_CXX11_ABI={abi})\n'
            f'set(LLVM_DEFINITIONS "-D_GLIBCXX_USE_CXX11_ABI={abi} '
            '-D__STDC_CONSTANT_MACROS -D__STDC_FORMAT_MACROS '
            '-D__STDC_LIMIT_MACROS")\n'
            'set(LLVM_ENABLE_RTTI ON)\n'
            'set(LLVM_ENABLE_EH ON)\n'
            'add_library(LLVM INTERFACE IMPORTED)\n'
            'add_library(MLIR INTERFACE IMPORTED)\n'
            f'include("{SOURCE}/cmake/LLKBuildOptions.cmake")\n'
            'add_executable(probe probe.cpp)\n'
            'target_link_libraries(probe PRIVATE LLKBuildOptions)\n'
            'if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")\n'
            '  target_compile_options(probe PRIVATE -Werror)\n'
            'endif()\n'
        )
        result = subprocess.run(
            ["cmake", "-S", str(self.root), "-B", str(self.root / "build"),
             "-G", "Ninja", *COMPILER_OPTIONS],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        result = subprocess.run(
            ["cmake", "--build", str(self.root / "build")],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_new_abi_and_all_llvm_definitions_compile_without_redefinition(self):
        self.compile_consumer(1)

    def test_old_abi_and_all_llvm_definitions_compile_without_redefinition(self):
        self.compile_consumer(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=pathlib.Path, required=True)
    parser.add_argument("--cxx-compiler")
    args, remaining = parser.parse_known_args()
    SOURCE = args.source_dir.resolve()
    COMPILER_OPTIONS = ([f"-DCMAKE_CXX_COMPILER={args.cxx_compiler}"]
                        if args.cxx_compiler else [])
    unittest.main(argv=[__file__, *remaining])
