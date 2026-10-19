# LLK Python frontend prototype

The `llk-compiler` distribution provides the `llk` Python package. Its initial
validation baseline is Python 3.11/3.12 with Torch 2.9.0. Torch is optional for
package import; the package does not depend on upstream Triton.

Use the project-level `.venv` for development. The local environment checked on
2026-10-19 is Python 3.14.8 / Torch 2.14.1 / NumPy 2.5.4, with pytest 9.1.1,
build 1.6.1, setuptools 84.0.0 and wheel 0.48.0. All 21 PF01 tests pass using
that interpreter. The reference Python 3.11/3.12 + Torch 2.9.0 matrix also passed
21 tests per configuration; it remains the manifest's reference baseline.

From your isolated worktree, use the existing project environment by its absolute
interpreter path:

```sh
export LLK_PYTHON=/absolute/path/to/DSLCompiler/.venv/bin/python
"$LLK_PYTHON" -m pip install -e './python[test]'
"$LLK_PYTHON" -m pytest
```

If creating a fresh environment, run `python3 -m venv .venv` in an isolated
worktree and use that interpreter. For the current local version pair, install
`torch==2.14.1` and `numpy==2.5.4` before running the commands above. The `[test]`
extra preserves an installed Torch version; `[torch-test]` pins Torch 2.9.0 for
the reference Python 3.11/3.12 environments. Torchvision and torchaudio are optional
and are not required by PF01. The [manual guide](../docs/manual/source/python-frontend.rst)
has the full environment and scope record.

The root pytest configuration selects `test/python`. The package tests build an
actual wheel and install it without network access into a fresh environment
outside the checkout. Registration tests require real Torch and fail when that
dependency is missing; they do not substitute a mock registry or skip the lane.

`import llk.torch` registers the `llk` Dynamo backend when Torch is installed.
Repeated registration and module reload preserve the registered callable,
available through `llk.torch.get_backend()`. Registration does not reset Dynamo.
If another package already owns the name, import raises a structured
`UnsupportedFeatureError` with code `backend_registration_conflict`.

Native frontend compilation and execution remain unavailable. The Dynamo backend
raises `UnsupportedFeatureError` with code `native_backend_unavailable`;
`torch.compile` wraps that diagnostic in `BackendCompilerFailed.inner_exception`.
It does not return the captured graph for eager execution. Inspect
`llk.capabilities()` for the package's explicit capability report and packaged
compatibility baseline. The existing decorator/AST prototype remains experimental;
its captured strings are not proof of native compilation or execution.
