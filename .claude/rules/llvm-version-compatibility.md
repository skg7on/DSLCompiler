# LLVM Version Compatibility

**Category:** implementation
**Enforcement:** mandatory — all MLIR/LLVM API usage MUST support multiple LLVM versions

## Rule

This project builds against two different LLVM/MLIR environments and MUST compile and run correctly on both:

| Environment | LLVM Version | Source |
|---|---|---|
| Local development (macOS) | LLVM 24+ | `/Users/skg7on/Workspace/Projects/llvm-project/build` (built from `main`) |
| CI (Ubuntu 24.04) | LLVM 20.1.8 | Pre-built release tarball from GitHub releases |

## Conditional Compilation

When LLVM/MLIR APIs differ between versions, use `LLVM_VERSION_MAJOR` from `llvm/Config/llvm-config.h` to gate the code:

```cpp
#include "llvm/Config/llvm-config.h"

// Pass names that changed between LLVM 20 and 21+
#if LLVM_VERSION_MAJOR >= 21
  pm.addPass(mlir::createSCFToControlFlowPass());
#else
  pm.addPass(mlir::createConvertSCFToCFPass());
#endif
```

For type names, use a version-conditional alias:

```cpp
#if LLVM_VERSION_MAJOR >= 21
using BufOpts = mlir::bufferization::OneShotBufferizePassOptions;
#else
using BufOpts = mlir::bufferization::OneShotBufferizationOptions;
#endif
```

## Known API Differences (LLVM 20 vs 21+)

| API | LLVM 20 | LLVM 21+ |
|-----|---------|----------|
| SCF→CF pass factory | `createConvertSCFToCFPass()` | `createSCFToControlFlowPass()` |
| Bufferization options type | `OneShotBufferizationOptions` | `OneShotBufferizePassOptions` |
| Bufferization pass factory | `createOneShotBufferizePass(const OneShotBufferizationOptions&)` | `createOneShotBufferizePass(OneShotBufferizePassOptions)` |
| Value type cast | `v.getType().cast<T>()` | `mlir::cast<T>(v.getType())` |

## Process

1. Write code against the local LLVM 24 API first
2. Before committing, add `#if LLVM_VERSION_MAJOR >= 21` / `#else` guards for any API that differs in LLVM 20
3. Push and verify CI passes (both the CI workflow and CodeQL workflow)
4. If CI fails with API errors, add the missing guard and push again
