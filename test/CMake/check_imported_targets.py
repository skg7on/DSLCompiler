"""Check imported configuration selection with a real exported CMake build."""

import argparse
import pathlib
import subprocess
import tempfile
import unittest


def run(*command):
    return subprocess.run(command, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)


class ImportedTargetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = pathlib.Path(cls.temp.name).resolve()
        producer = cls.root / "producer"
        producer.mkdir()
        (producer / "library.cpp").write_text("int probe() { return 0; }\n")
        (producer / "tool.cpp").write_text("int main() { return 0; }\n")
        (producer / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.20)\n'
            'project(producer LANGUAGES CXX)\n'
            'set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE "${CMAKE_BINARY_DIR}/Release/bin")\n'
            'add_library(LLVM SHARED library.cpp)\n'
            'add_executable(mlir-tblgen tool.cpp)\n'
            'export(TARGETS LLVM mlir-tblgen FILE "${CMAKE_BINARY_DIR}/Targets.cmake")\n'
        )
        cls.llvm_build = cls.root / "llvm-build"
        result = run("cmake", "-S", str(producer), "-B", str(cls.llvm_build),
                     "-G", "Ninja Multi-Config")
        if result.returncode:
            raise AssertionError(result.stdout)
        result = run("cmake", "--build", str(cls.llvm_build), "--config", "Release")
        if result.returncode:
            raise AssertionError(result.stdout)

    def configure(self, config, mapping="", generator="Ninja"):
        consumer = self.root / self.id().split(".")[-1]
        consumer.mkdir()
        (consumer / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.20)\n'
            'project(consumer LANGUAGES NONE)\n'
            f'set(LLVM_PROJECT_BUILD_DIR "{self.llvm_build}")\n'
            f'include("{self.llvm_build}/Targets.cmake")\n'
            f'include("{SOURCE}/cmake/LLKImportedTargets.cmake")\n'
            + mapping + '\n'
            'llk_require_build_target(LLVM SHARED_LIBRARY)\n'
            'llk_require_build_target(mlir-tblgen EXECUTABLE tool_path)\n'
            'add_custom_target(probe ALL COMMAND mlir-tblgen)\n'
            'file(WRITE "${CMAKE_BINARY_DIR}/tool-path.txt" "${tool_path}")\n'
        )
        self.consumer_build = consumer / "build"
        return run("cmake", "-S", str(consumer), "-B", str(self.consumer_build),
                   "-G", generator, f"-DCMAKE_BUILD_TYPE={config}")

    def test_release_does_not_require_unbuilt_debug(self):
        result = self.configure("Release")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual((self.consumer_build / "tool-path.txt").read_text(),
                         str(self.llvm_build / "Release/bin/mlir-tblgen"))
        result = run("cmake", "--build", str(self.consumer_build))
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_configuration_mapping_selects_built_release(self):
        result = self.configure("RelWithDebInfo", mapping=
            'set_target_properties(LLVM mlir-tblgen PROPERTIES '
            'MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release)')
        self.assertEqual(result.returncode, 0, result.stdout)
        result = run("cmake", "--build", str(self.consumer_build))
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_requested_unbuilt_debug_is_rejected(self):
        result = self.configure("Debug")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("LLVM has not been built", result.stdout)

    def test_multiconfig_consumer_accepts_built_release(self):
        result = self.configure("", generator="Ninja Multi-Config")
        self.assertEqual(result.returncode, 0, result.stdout)
        result = run("cmake", "--build", str(self.consumer_build), "--config", "Release")
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_empty_single_config_validates_cmake_fallback(self):
        # Without a build type CMake falls back to the first exported variant
        # (Debug), rather than searching the filesystem for a built variant.
        result = self.configure("")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("LLVM has not been built", result.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=pathlib.Path, required=True)
    args, remaining = parser.parse_known_args()
    SOURCE = args.source_dir.resolve()
    unittest.main(argv=[__file__, *remaining])
