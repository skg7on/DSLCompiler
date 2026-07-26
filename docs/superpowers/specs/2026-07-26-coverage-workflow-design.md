# Code Coverage Workflow — Design Spec

**Date:** 2026-07-26
**Status:** approved

## Overview

Add a GitHub Actions `coverage.yml` workflow that builds the project with GCC `--coverage` instrumentation, runs the test suite, and uploads an HTML coverage report as a workflow artifact. Line, branch, and function coverage are tracked.

## Decisions

| Decision | Choice |
|---|---|
| Metrics | Line + branch + function |
| Output | HTML report via `gcovr --html-details`, uploaded as artifact |
| Workflow | Separate `coverage.yml`, independent from `ci.yml` |
| Build mode | Debug + `--coverage` |
| Compiler | GCC (Ubuntu 24.04 system default, explicitly set via CC/CXX) |
| LLVM | Release, reuse cache key from CI (`llvm-build-22.1.8-v1`) |
| Threshold | None — purely informational |
| Triggers | push to main, PR to main, `workflow_dispatch` |

## Workflow Structure

```yaml
name: Coverage

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  coverage:
    runs-on: ubuntu-24.04
    timeout-minutes: 120
    env:
      LLVM_VERSION: 22.1.8
      LLVM_SOURCE: ${{ github.workspace }}/llvm-project
      LLVM_BUILD: ${{ github.workspace }}/llvm-build
    steps:
      - Checkout
      - Install deps (ninja-build, libgtest-dev, gcovr via pip)
      - Download LLVM source
      - Cache/restore LLVM build (same key as ci.yml)
      - Build LLVM Release (if cache miss)
      - Configure project: Debug + --coverage, CC=gcc CXX=g++
      - Build project
      - Run ctest
      - Generate gcovr HTML report
      - Upload artifact (name: coverage-report, retention: 7 days)
```

## Compiler & Coverage Flags

Explicitly set `CC=gcc` and `CXX=g++` in the workflow environment. Ubuntu 24.04 ships GCC 14.x.

CMake configuration for the project (LLVM stays Release as-is):

```
-DCMAKE_BUILD_TYPE=Debug
-DCMAKE_CXX_FLAGS="--coverage"
-DCMAKE_C_FLAGS="--coverage"
-DCMAKE_EXE_LINKER_FLAGS="--coverage"
-DCMAKE_SHARED_LINKER_FLAGS="--coverage"
```

`--coverage` is a GCC shorthand equivalent to `-fprofile-arcs -ftest-coverage -lgcov`.

## gcovr Configuration

Run after `ctest` completes (tests produce `.gcda` files):

```bash
gcovr \
  --root . \
  --exclude 'test/.*' \
  --exclude 'third_party/.*' \
  --exclude 'build/.*' \
  --exclude '.*\.inc\..*' \
  --html-details coverage.html \
  --print-summary
```

Exclusions:
- `test/` — test code
- `third_party/` — vendored dependencies (googletest)
- `build/` — CMake build artifacts
- `.*\.inc\..*` — MLIR tablegen-generated `.inc` files

Output: `coverage.html` (index with per-directory rollups) + `coverage.<source>.html` per source file with line-by-line annotated hit counts.

All files zipped into a single artifact: `coverage-report`.

## gcovr Installation

Install via pip for the latest version (apt's `gcovr` is often stale):

```bash
pip install gcovr
```

## LLVM Build Sharing

The coverage workflow reuses the same LLVM cache key as `ci.yml` (`llvm-build-${LLVM_VERSION}-v1`). Since both build LLVM in Release mode with identical flags, the cache is shared. This avoids building LLVM twice when both workflows run on the same PR.

## No Concurrency Conflict

Separate workflow means coverage doesn't slow down the CI feedback loop. If LLVM cache is cold and both workflows start simultaneously, one restores/rebuilds first; the other either gets a cache hit or queues briefly behind the first LLVM build. This is rare (cache is warm 99% of the time) and acceptable.

## What's NOT Included

- **No coverage threshold enforcement** — report is purely informational.
- **No PR comment** — HTML artifact only.
- **No scheduled cron run** — coverage on every push/PR is sufficient.
- **No LLVM/MLIR coverage** — only the project's own source (`lib/`, `include/`, `tools/`, `runtime/`).
