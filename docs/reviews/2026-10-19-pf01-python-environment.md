# PF01 Python frontend environment and verification

**Date:** 2026-10-19 (Asia/Shanghai).

**Implementation:** [`ffc55f5788cfd1d801836b9aae20035708c65c02`](https://github.com/skg7on/DSLCompiler/commit/ffc55f5788cfd1d801836b9aae20035708c65c02)
in [PR #233](https://github.com/skg7on/DSLCompiler/pull/233), implementing
[#212](https://github.com/skg7on/DSLCompiler/issues/212). The documentation update
does not change Python runtime behavior. PR #233 is pending merge at this update.

## Current development environment

Use the project-level `.venv` interpreter, including when running source checks
from an isolated worktree. For this checkout it is
`/Users/skg7on/Workspace/Projects/DSLCompiler/.venv/bin/python`; portable setup and
worktree commands are in the [frontend guide](../manual/source/python-frontend.rst).
Pipx environments and earlier scratch environments are not the active local
development environment.

| Package | Observed version |
| --- | --- |
| Python | 3.14.8 |
| Torch | 2.14.1 |
| NumPy | 2.5.4 |
| pytest | 9.1.1 |
| build | 1.6.1 |
| setuptools | 84.0.0 |
| wheel | 0.48.0 |
| torchvision, optional for PF01 | 0.29.1 |
| torchaudio, optional for PF01 | 2.11.0 |

Torchvision and torchaudio imports succeed. This is import evidence only; no
image/audio operator corpus or native LLK lowering is qualified by those imports.

## PF01 verification

From the PF01 worktree root, with `LLK_PYTHON` set to the project interpreter:

```sh
PYTHONDONTWRITEBYTECODE=1 "$LLK_PYTHON" -m pytest -q -p no:cacheprovider
```

**Result: 21 passed, zero skips.** Root `pytest.ini` selects `test/python` and
adds the current worktree's `python/` source path. Package tests build a real wheel
and install it offline into a clean Torch-absent environment outside the checkout.
Real Torch subprocesses cover registered/public singleton identity, repeated
registration and reload, zero Dynamo resets, foreign-owner preservation, broken
optional dependency propagation, direct native-unavailable errors and the same
error through actual `torch.compile`.

Independent review of the implementation found no issues and independently ran
the same 21 tests on Python 3.12.12/Torch 2.9.0, plus import-order/reload probes.
Earlier reference runs on Python 3.11.14 and 3.12.12 with Torch 2.9.0 each passed
21 tests. These remain the reference matrix in
[compatibility.json](../../python/llk/compatibility.json); the current `.venv` is
additional PF01 compatibility evidence.

## Runtime probes and limits

Both Torch-first and NumPy-first imports passed CPU f32 matrix multiplication,
autograd gradient checks and NumPy/Torch interoperability. The dyld image list
contained exactly one OpenMP runtime, Torch's bundled `libomp.dylib`, present as
a regular file inside `.venv`. The previous global-install OpenMP conflict was not reproduced.
No duplicate-runtime bypass or Homebrew runtime symlink was used for these checks.

These probes execute Torch, not LLK-generated kernels. PF01 reports
`native_backend_unavailable`, including the structured underlying error in
`BackendCompilerFailed.inner_exception`. Native frontend compilation, checked
buffers, executable ownership, numerical qualification, Inductor/training and
upstream Triton integration remain later tasks under #203. Required native
frontend CMake/CI execution lanes remain PF20.

## Documentation and dependency checks

- Documentation checker regressions: 13 passed.
- Sphinx source-link extension regression: 1 passed.
- Strict HTML build (`-b html -n -W --keep-going`): passed without warnings.
- Repository path checks: all 28 configured documents passed; rendered guide
  headings, versions, capability boundary and manual navigation were checked.
- `python -m pip check` using the project interpreter: no broken requirements.

These documentation checks do not replace native compiler acceptance or the
future PF20 frontend CI lanes.
