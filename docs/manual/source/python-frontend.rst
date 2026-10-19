.. _python-frontend:

Python frontend
===============

The ``llk-compiler`` distribution provides the ``llk`` package. PF01, tracked by
`#212 <https://github.com/skg7on/DSLCompiler/issues/212>`__ and implemented in
`PR #233 <https://github.com/skg7on/DSLCompiler/pull/233>`__, delivers packaging,
optional Torch imports, idempotent Dynamo registration, structured errors and
capability reporting. The
:source:`environment verification record <docs/reviews/2026-10-19-pf01-python-environment.md>`
pins the implementation and validation evidence.

Native frontend compilation and invocation remain unavailable. Calling the
``llk`` Dynamo backend raises ``UnsupportedFeatureError`` with code
``native_backend_unavailable``. Real ``torch.compile`` wraps this in
``BackendCompilerFailed.inner_exception``. The registration tests verify this
diagnostic; their passing result does not establish native kernel execution.
Typed capture, executable decorators, Inductor composition, training and genuine
upstream Triton launches remain later tasks under
`#203 <https://github.com/skg7on/DSLCompiler/issues/203>`__.

.. _python-frontend-environment:

Use the project environment
---------------------------

Use the existing project-level ``.venv`` rather than a pipx application
environment. Pipx environments for Torch and pytest can be separate, while a
frontend test run needs its libraries and test tools in the same environment.
When entering an isolated worktree, retain the absolute path to the project
environment's interpreter. Source changes and generated artifacts stay in the
worktree; an existing root-checkout ``.venv`` can supply the interpreter.

From the isolated worktree root, after setting the interpreter path:

.. code-block:: sh

   export LLK_PYTHON=/absolute/path/to/DSLCompiler/.venv/bin/python
   "$LLK_PYTHON" -m pip install -e './python[test]'
   "$LLK_PYTHON" -m pytest

If there is no environment yet, create ``.venv`` in an isolated worktree with
``python3 -m venv .venv`` and set ``LLK_PYTHON`` to its absolute interpreter path.
For the verified local version pair, install ``torch==2.14.1`` and
``numpy==2.5.4``. The ``test`` extra in
:source:`python/pyproject.toml <python/pyproject.toml>` supplies pytest and the
wheel-build tools without selecting or downgrading Torch.

The separate ``torch-test`` extra selects Torch 2.9.0 for the original Python
3.11/3.12 reference matrix. Use that extra in those reference environments,
rather than applying its pin to the current Python 3.14 environment. Torch is
optional for importing the package, but required for the real registration
tests. Upstream Triton, torchvision and torchaudio are not PF01 requirements.

.. _python-frontend-verified-versions:

Verified local snapshot
-----------------------

The project ``.venv`` was checked on **2026-10-19** on macOS arm64:

.. list-table::
   :header-rows: 1
   :widths: 32 24 44

   * - Component
     - Version
     - Evidence
   * - Python
     - 3.14.8
     - Project ``.venv/bin/python``
   * - Torch / NumPy
     - 2.14.1 / 2.5.4
     - Both import orders, CPU matmul, autograd and NumPy interoperability pass
   * - pytest / build
     - 9.1.1 / 1.6.1
     - Test and package-build imports pass
   * - setuptools / wheel
     - 84.0.0 / 0.48.0
     - Clean wheel build/install tests pass
   * - torchvision / torchaudio
     - 0.29.1 / 2.11.0
     - Optional imports pass; image/audio operator behavior is not qualified

Only one OpenMP runtime, Torch's bundled ``libomp.dylib``, was loaded in the
Torch/NumPy probes. No shared-runtime symlink or duplicate-runtime bypass was
needed in this isolated environment. These probes exercise Torch itself; they
are not LLK native-execution evidence.

The full PF01 Python suite passes **21/21, with zero skips**, using this project
interpreter against the PR worktree. Earlier Python 3.11.14 and 3.12.12 runs with
Torch 2.9.0 each passed the same 21 tests. The
:source:`packaged compatibility manifest <python/llk/compatibility.json>` retains
that reference matrix; the current local snapshot is additional PF01 evidence,
not a promise of arbitrary-version or later-milestone compatibility.

.. _python-frontend-testing:

Run focused tests
------------------

.. code-block:: sh

   "$LLK_PYTHON" -m pytest test/python/test_torch_registration.py \
     test/python/test_package_install.py -q

The root :source:`pytest.ini <pytest.ini>` restricts default collection to
``test/python`` and adds the local ``python/`` source path. This avoids collecting
vendored GoogleTest's own Python tests, which use a different harness. Run from
the worktree containing PF01; an older root checkout may not have this config.

The package tests build the actual wheel and install it offline into a fresh
Torch-absent virtual environment outside the checkout. Real-Torch subprocesses
test registration/reload identity, zero Dynamo resets, existing-owner protection,
broken optional-dependency propagation and explicit execution rejection. Missing
test/build dependencies are setup failures, not successful or skipped coverage.

These Python checks are separate from the compiler's ``check-llk`` target. Native
frontend CMake registration and required execution CI lanes remain PF20. Continue
to use :doc:`contributing` for C++/MLIR and selected-target acceptance.

.. _python-frontend-inspect:

Inspect the package boundary
----------------------------

After installing the package into the chosen environment:

.. code-block:: sh

   "$LLK_PYTHON" -c 'import llk; print(llk.capabilities())'
   "$LLK_PYTHON" -c 'import llk.torch; print(llk.torch.get_backend())'

``get_backend()`` returns the exact registered callable. Repeated registration
and module reload preserve it without clearing Dynamo's global state. A foreign
owner of the ``llk`` backend name is preserved and produces
``backend_registration_conflict``. The experimental decorator's captured strings
do not establish verified MLIR, real upstream TTIR or a native JIT invocation.
