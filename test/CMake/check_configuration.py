"""Exercise the build's package-selection contract through real CMake runs."""

import argparse
import pathlib
import subprocess
import tempfile
import unittest


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)

    def configure(self, *options, build=None):
        return subprocess.run(
            ["cmake", "-S", str(SOURCE), "-B", str(build or self.root / "build"),
             "-G", "Ninja", "-DBUILD_TESTING=OFF", *options],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

    def test_build_directory_is_required_even_with_package_hints(self):
        result = self.configure(
            f"-DLLVM_DIR={LLVM_BUILD}/lib/cmake/llvm",
            f"-DMLIR_DIR={LLVM_BUILD}/lib/cmake/mlir",
        )
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("LLVM_PROJECT_BUILD_DIR is required", result.stdout)

    def test_invalid_build_directory_has_actionable_error(self):
        result = self.configure(f"-DLLVM_PROJECT_BUILD_DIR={self.root / 'missing'}")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("LLVM_PROJECT_BUILD_DIR", result.stdout)
        self.assertIn("LLVMConfig.cmake", result.stdout)

    def test_explicit_build_directory_overrides_foreign_cached_packages(self):
        foreign = self.root / "foreign"
        foreign.mkdir()
        for package in ("LLVM", "MLIR"):
            (foreign / f"{package}Config.cmake").write_text(
                'message(FATAL_ERROR "foreign package loaded")\n'
            )
        result = self.configure(
            f"-DLLVM_PROJECT_BUILD_DIR={LLVM_BUILD}",
            f"-DLLVM_DIR={foreign}", f"-DMLIR_DIR={foreign}",
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        cache = (self.root / "build/CMakeCache.txt").read_text()
        self.assertIn(f"LLVM_DIR:PATH={LLVM_BUILD}/lib/cmake/llvm", cache)
        self.assertIn(f"MLIR_DIR:PATH={LLVM_BUILD}/lib/cmake/mlir", cache)

    def test_static_llvm_is_rejected(self):
        package = self.root / "llvm/lib/cmake/llvm"
        package.mkdir(parents=True)
        (package / "LLVMConfig.cmake").write_text(
            'set(LLVM_PACKAGE_VERSION 22.1.8)\n'
            'add_library(LLVM STATIC IMPORTED)\n'
        )
        # A matching MLIR package exists, but LLVM must be rejected first.
        mlir = self.root / "llvm/lib/cmake/mlir"
        mlir.mkdir(parents=True)
        (mlir / "MLIRConfig.cmake").write_text("")
        result = self.configure(f"-DLLVM_PROJECT_BUILD_DIR={self.root / 'llvm'}")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("LLVM must be a shared library", result.stdout)

    def shared_llvm_package(self, library):
        build = self.root / "llvm"
        package = build / "lib/cmake/llvm"
        package.mkdir(parents=True)
        (package / "LLVMConfig.cmake").write_text(
            'add_library(LLVM SHARED IMPORTED)\n'
            f'set_target_properties(LLVM PROPERTIES IMPORTED_LOCATION "{library}")\n'
            'set(LLVM_LINK_LLVM_DYLIB ON)\n'
            'set(LLVM_DYLIB_COMPONENTS all)\n'
        )
        mlir = build / "lib/cmake/mlir"
        mlir.mkdir(parents=True)
        (mlir / "MLIRConfig.cmake").write_text("")
        return build

    def test_llvm_library_from_foreign_build_is_rejected(self):
        library = self.root / "foreign/libLLVM.so"
        library.parent.mkdir()
        library.touch()
        build = self.shared_llvm_package(library)
        result = self.configure(f"-DLLVM_PROJECT_BUILD_DIR={build}")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("LLVM points outside LLVM_PROJECT_BUILD_DIR", result.stdout)

    def test_missing_shared_library_artifact_is_rejected(self):
        build = self.shared_llvm_package(self.root / "llvm/lib/libLLVM.so")
        result = self.configure(f"-DLLVM_PROJECT_BUILD_DIR={build}")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("LLVM has not been built", result.stdout)

    def configure_mlir_export(self, library_type):
        # LLVM 22 exports the shared target but not MLIR_LINK_MLIR_DYLIB.
        # Exercise dependency discovery independently of dialect generation.
        build = self.shared_llvm_package(self.root / "llvm/lib/libLLVM.so")
        (build / "lib/libLLVM.so").touch()
        modules = build / "cmake"
        modules.mkdir()
        for name in ("TableGen", "AddLLVM", "AddMLIR", "HandleLLVMOptions"):
            (modules / f"{name}.cmake").write_text("")
        with (build / "lib/cmake/llvm/LLVMConfig.cmake").open("a") as config:
            config.write(f'set(LLVM_CMAKE_DIR "{modules}")\n')
        (build / "lib/libMLIR.so").touch()
        (build / "lib/libmlir_c_runner_utils.so").touch()
        (build / "bin").mkdir()
        (build / "bin/mlir-tblgen").touch()
        (build / "lib/cmake/mlir/MLIRConfig.cmake").write_text(
            f'add_library(MLIR {library_type} IMPORTED)\n'
            f'set_target_properties(MLIR PROPERTIES IMPORTED_LOCATION "{build}/lib/libMLIR.so")\n'
            'add_library(mlir_c_runner_utils SHARED IMPORTED)\n'
            f'set_target_properties(mlir_c_runner_utils PROPERTIES IMPORTED_LOCATION "{build}/lib/libmlir_c_runner_utils.so")\n'
            'add_executable(mlir-tblgen IMPORTED)\n'
            f'set_target_properties(mlir-tblgen PROPERTIES IMPORTED_LOCATION "{build}/bin/mlir-tblgen")\n'
        )
        consumer = self.root / "consumer"
        consumer.mkdir()
        (consumer / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.20)\nproject(ExportConsumer NONE)\n'
            f'include("{SOURCE}/cmake/LLKDependencies.cmake")\n'
        )
        return subprocess.run(
            ["cmake", "-S", str(consumer), "-B", str(self.root / "build"),
             f"-DLLVM_PROJECT_BUILD_DIR={build}"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )

    def test_shared_mlir_without_linkage_variable_is_accepted(self):
        result = self.configure_mlir_export("SHARED")
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_static_mlir_is_rejected(self):
        result = self.configure_mlir_export("STATIC")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("MLIR must be a shared library", result.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=pathlib.Path, required=True)
    parser.add_argument("--llvm-build", type=pathlib.Path, required=True)
    args, remaining = parser.parse_known_args()
    SOURCE = args.source_dir.resolve()
    LLVM_BUILD = args.llvm_build.resolve()
    unittest.main(argv=[__file__, *remaining])
