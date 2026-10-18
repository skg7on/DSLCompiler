# LLK Python frontend prototype

The `llk-compiler` distribution provides the `llk` Python package. Its initial
validation baseline is Python 3.11/3.12 with Torch 2.9.0. Torch is optional for
package import; the package does not depend on upstream Triton.

Run these commands from an isolated repository worktree with Python 3.12:

```sh
python3.12 -m venv build/python-venv
build/python-venv/bin/python -m pip install -e './python[test,torch-test]'
build/python-venv/bin/python -m pytest
```

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
