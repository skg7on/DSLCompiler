# Build organization

The root `CMakeLists.txt` sets project options and adds the library, runtime,
tool, and test directories. Each library directory lists its sources and public
dependencies with `add_llk_library()`.

| Module | Responsibility |
| --- | --- |
| `LLKCompiler.cmake` | Compiler selection and macOS deployment target, before `project()` |
| `LLKDependencies.cmake` | Required LLVM build, package discovery, shared linkage and TableGen tool |
| `LLKImportedTargets.cmake` | Imported artifact provenance and configuration selection |
| `LLKBuildOptions.cmake` | Headers, ABI flags, library helper and output directories |
| `LLKVersion.cmake` | Generated compiler version header |
| `LLKTableGen.cmake` | Generated dialect and pass headers under `build/include/LLK` |

`LLVM_PROJECT_BUILD_DIR` is the only LLVM/MLIR discovery input. The packages in
its `lib/cmake/` directories export the source and generated header paths, so
the source checkout need not have a particular location relative to the build.
Cached `LLVM_DIR` and `MLIR_DIR` values are replaced on every configure.

Use `LLVM` for LLVM-only dependencies and `LLK::MLIR` for MLIR dependencies.
`LLK::MLIR` links the shared `MLIR` and `LLVM` targets together: project code
calls both APIs, and `MLIR` does not always propagate `LLVM` publicly. There is
no LLVM component archive fallback. `MLIRMlirOptMain` supplies the optimizer
entry point, and `mlir_c_runner_utils` supplies the JIT's C runtime helpers;
these remain separate upstream targets from the same LLVM build.

Project libraries are static. `LLKBuildOptions` matches the LLVM build's RTTI
and exception settings for their consumers. Generated headers use explicit
`DEPENDS`; linked project libraries propagate their own dependencies.

TableGen runs from the root so generated include paths remain stable. It uses
the exported `mlir-tblgen` executable, including configuration-specific paths.
FileCheck is found beside that tool. Tests live under `test/`, with build-system
regressions under `test/CMake/`.
