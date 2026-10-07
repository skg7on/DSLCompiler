# Getting started

Build the tools, parse a small kernel, and export its execution representation. This guide assumes a Unix shell and runs DSLCompiler commands from the repository root. Linux is the CI platform; macOS also supports local development. AVX2 profile analysis does not require an AVX2 host. Selected AVX2 machine-code execution requires an appropriate x86 host and stronger evidence than a successful portable JIT compile.

## Prerequisites

- Git, CMake 3.20 or newer, Ninja, Python 3 for registered script tests, and a C++20 compiler.
- A matching LLVM/MLIR **development build**, including headers, CMake package files, TableGen tools, and `FileCheck`. A standalone LLVM executable installation is insufficient.
- Enough memory and disk for LLVM/MLIR. Limit build parallelism if compilation exhausts memory.

The project's [CI configuration](../.github/workflows/ci.yml) pins LLVM **22.1.8**. Use that version for a reproducible first build. Source compatibility guards exist for LLVM 20+, and local development also uses LLVM main; this does not guarantee every intermediate LLVM version is tested. See the upstream [MLIR build instructions](https://mlir.llvm.org/getting_started/) and [LLVM CMake reference](https://llvm.org/docs/CMake.html) for general toolchain configuration.

## Build LLVM and MLIR once

Run these commands in a parent directory where you want both repositories:

```bash
git clone --branch llvmorg-22.1.8 --depth 1 https://github.com/llvm/llvm-project.git
cmake -S llvm-project/llvm -B llvm-project/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=Native \
  -DLLVM_BUILD_UTILS=ON \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DLLVM_INCLUDE_EXAMPLES=OFF \
  -DLLVM_INCLUDE_DOCS=OFF \
  -DLLVM_INCLUDE_BENCHMARKS=OFF \
  -DMLIR_INCLUDE_TESTS=OFF \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF
cmake --build llvm-project/build --parallel 4
```

`LLVM_BUILD_UTILS=ON` makes `FileCheck` available even with LLVM's own tests disabled. Keep LLVM and MLIR from the same checkout and build. `Native` enables the current host backend; it does not turn an AArch64 host into an AVX2 execution host. You may use an existing development build instead of repeating this step.

## Configure DSLCompiler

```bash
git clone https://github.com/skg7on/DSLCompiler.git
cd DSLCompiler
cmake -S . -B build -G Ninja \
  -DLLVM_PROJECT_BUILD_DIR="$PWD/../llvm-project/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLK_BUILD_TOOLS=ON \
  -DLLK_BUILD_E2E_TESTS=ON
cmake --build build --parallel 4
```

Adjust `LLVM_PROJECT_BUILD_DIR` to your LLVM build's absolute path. It must contain `LLVM_BUILD/lib/cmake/mlir` and `LLVM_BUILD/lib/cmake/llvm`, where `LLVM_BUILD` denotes that directory; this setting prevents accidental mixing with system packages. Without it, CMake's normal `MLIR_DIR`/`LLVM_DIR` discovery applies. The project supplies its GTest dependency from [third_party](../third_party/CMakeLists.txt).

Useful configuration choices:

| Setting | Default | Effect |
|---|---|---|
| `LLK_BUILD_TOOLS` | `ON` | Builds `llk-opt` and `llk-compile`; their registered CLI tests require matching `FileCheck` |
| `LLK_BUILD_E2E_TESTS` | `ON` | Builds the JIT end-to-end test executables |
| `CMAKE_EXPORT_COMPILE_COMMANDS` | enabled for supported generators | Produces `build/compile_commands.json` for clangd |
| `CMAKE_BUILD_TYPE` | no project default | Choose `Release` for timings or `RelWithDebInfo` for debugging |

The tuner, performance tool, and benchmark are separate CMake targets. For a smaller initial build:

```bash
cmake --build build --target llk-opt llk-compile llk-tune micro-perf llk-bench --parallel 4
```

This builds the tutorial tools, not every test executable.

## Check the build

```bash
build/llk-opt --help
build/llk-compile --help
build/micro-perf --help
build/llk-tune --help
build/llk-bench --help
cmake --build build --target check-llk --parallel 4
```

`check-llk` builds the registered test binaries and runs CTest. After building them, `ctest --test-dir build --output-on-failure` runs the suite again without building. Use `ctest --test-dir build -N` to see what your configuration registered. Some legacy tests may skip; report the actual skips when sharing evidence. A green suite with skips is not proof of selected AVX2 execution.

## First useful result

```bash
mkdir -p build/tutorial
build/llk-opt docs/examples/matmul.mlir > build/tutorial/matmul.parsed.mlir
build/llk-compile --emit=micro docs/examples/matmul.mlir \
  > build/tutorial/matmul.micro.mlir
```

The first output is parsed and printed semantic MLIR. The second includes a concrete `micro.kernel` beside the original function. Neither command invokes the kernel on input tensors. Continue with [your first kernel](tutorials/01-first-kernel.md), then [mapping](tutorials/02-mapping.md).

## Common setup failures

| Symptom | Check or remedy |
|---|---|
| MLIR package not found | Verify the LLVM build path and the two CMake package directories; a runtime-only package lacks them |
| `FileCheck is required` | Reconfigure that LLVM build with `LLVM_BUILD_UTILS=ON`, build `FileCheck`, and reconfigure DSLCompiler |
| API errors or duplicate MLIR linkage | Check that headers, archives/dylib, TableGen, and CMake packages come from one LLVM revision |
| Compiler path no longer exists | Check `CC`/`CXX` and cached `CMAKE_CXX_COMPILER`; use a new build directory after changing toolchains |
| macOS compiler unexpectedly selected | CMake honors explicit compiler settings and valid `CC`/`CXX`; a fresh configuration otherwise looks for clang on `PATH` |
| Linking fails or compilation is killed | Lower parallelism and check memory; retain the complete configure/build diagnostic for an issue |
| Test executable not found | Build `check-llk`; building only the five tools does not build the full suite |

For code changes, follow the repository's [isolated worktree policy](../.claude/rules/worktree-isolation.md). The LLVM build can be shared across worktrees, but each worktree needs its own project build directory. Continue to the [contributor guide](contributing.md) for component ownership and tests.
