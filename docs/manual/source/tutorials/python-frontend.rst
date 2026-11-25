.. _tutorial-python-frontend:

Run a Python kernel and enforce the frontend lanes
==================================================

This tutorial builds the checked C ABI, installs the ``llk`` package and executes
a typed kernel on the reference backend. It also shows how to run the required
frontend test lanes on a fully provisioned Linux x86_64/AVX2 host. Run every
command from the isolated worktree root; generated files stay under ``build/``.

The reference example needs Python 3.11 or newer and no Torch or Triton.
Use the project ``.venv`` interpreter, or create one in the worktree:

.. code-block:: sh

   python3 -m venv .venv
   export LLK_PYTHON="$PWD/.venv/bin/python"
   "$LLK_PYTHON" -m pip install -e './python[test]'

An existing project environment can be shared by setting ``LLK_PYTHON`` to its
absolute interpreter path instead. Torch integration is qualified against the
Python 3.11/3.12 + Torch 2.9.0 reference matrix; upstream Triton requires the
exact :ref:`toolchain qualification <python-frontend-triton>`.

Build the C ABI
---------------

Configure with the Python C ABI enabled and use the same C++ toolchain for
LLVM/MLIR and this project, as described in :doc:`/getting-started`:

.. code-block:: sh

   cmake -S . -B build -G Ninja \
     -DLLVM_PROJECT_BUILD_DIR=/path/to/llvm-project/build \
     -DLLK_BUILD_PYTHON_CAPI=ON
   cmake --build build --target LLKFrontendCAPI

Keep ``LLK_ENABLE_PYTHON_TESTS`` off for this reference-only example. The
full-lane configuration is shown below after its prerequisites.

Select the native library
--------------------------

On Linux, select the shared library built in this worktree:

.. code-block:: sh

   export LLK_FRONTEND_LIBRARY="$PWD/build/lib/libLLKFrontendCAPI.so"

On macOS use ``libLLKFrontendCAPI.dylib`` instead. The Python wheel does not
include this native library.

The wheel carries ``llk/config/machines/*.yaml`` and
``llk/config/mapping/*/*.llkmap`` together with ``llk/resources.json`` hashes,
so ``@llk.jit`` and ``CompileOptions()`` load target policy from the
installation without a repository checkout. A missing or modified packaged
file fails before compilation with ``native_configuration_missing`` or
``native_configuration_corrupt``.

Execute a kernel
----------------

Save the following as ``build/python-add.py``, then run it with
``"$LLK_PYTHON" build/python-add.py``. Source capture needs a real file rather
than an interactive definition. The native frontend proves the program,
selects a plan, compiles it and invokes it over caller-owned buffers.

.. code-block:: python

   import array
   import llk
   import llk.language as tl
   from llk.frontend import LaunchSpec, TensorArgument, TensorSpec

   M, N, BLOCK_M, BLOCK_N = 16, 16, 8, 16

   @llk.jit
   def add(a, b, out, M: tl.constexpr, N: tl.constexpr,
           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
       rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
       cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
       offsets = rows[:, None] * N + cols[None, :]
       mask = (rows[:, None] < M) & (cols[None, :] < N)
       x = tl.load(a + offsets, mask=mask, other=0.0)
       y = tl.load(b + offsets, mask, 0.0)
       tl.store(out + offsets, x + y, mask=mask)

   def buffer(rows, columns, fill):
       storage = array.array("f", [fill] * (rows * columns))
       view = memoryview(storage).cast("B").cast("f", (rows, columns))
       return view, storage

   a, a_storage = buffer(M, N, 1.0)
   b, b_storage = buffer(M, N, 2.0)
   out, out_storage = buffer(M, N, 0.0)

   def spec(name, direction, effect):
       return TensorSpec(name, (M, N), "f32", (N, 1), direction, effect)

   compiled = add.compile(
       TensorArgument(spec("a", "input", "read"), a),
       TensorArgument(spec("b", "input", "read"), b),
       TensorArgument(spec("out", "output", "write"), out),
       M, N, BLOCK_M, BLOCK_N,
       grid=LaunchSpec((M // BLOCK_M, N // BLOCK_N, 1), (BLOCK_M, BLOCK_N, 1)),
   )
   compiled.invoke((a, b), (out,))
   assert list(out_storage) == [3.0] * (M * N)
   compiled.close()

Run focused lanes
------------------

The required lanes are declared once in
``test/Frontend/acceptance_manifest.json`` and selected by pytest markers:
``frontend_unit`` (compiler-free), ``frontend_native`` (reference execution),
``torch_integration``, ``triton_integration`` and ``selected_avx2``.
The first two can be run with the reference setup above:

.. code-block:: sh

   export LLK_PYTHON_EVIDENCE_DIR="$PWD/build/frontend-evidence"
   mkdir -p "$LLK_PYTHON_EVIDENCE_DIR/junit"

   LLK_PYTHON_LANE=frontend_unit "$LLK_PYTHON" -m pytest test/python \
     -m "frontend_unit and not frontend_native and not torch_integration and not triton_integration" \
     -q --junitxml="$LLK_PYTHON_EVIDENCE_DIR/junit/frontend_unit.xml"

   LLK_PYTHON_LANE=frontend_native "$LLK_PYTHON" -m pytest test/python \
     -m "frontend_native and not torch_integration and not triton_integration" \
     -q --junitxml="$LLK_PYTHON_EVIDENCE_DIR/junit/frontend_native.xml"

Each lane writes ``<evidence-dir>/<lane>.json`` with the number of native calls
made through ``LLKFrontendCAPI``. These two focused runs do not satisfy the
five-lane acceptance manifest.

Enforce all required lanes
--------------------------

Full acceptance requires Linux x86_64 with AVX2, the native library, real Torch,
Inductor's host C++ compiler and the pinned built Triton source checkout. Use a
Python 3.11 or 3.12 environment for the Torch 2.9.0 reference matrix:

.. code-block:: sh

   "$LLK_PYTHON" -m pip install -e './python[test,torch-test]'
   export LLK_TRITON_PYTHONPATH=/absolute/path/to/triton/python
   "$LLK_PYTHON" -c 'import llk.triton; print(llk.triton.qualify_toolchain())'

After the two focused lanes above, run the other three in the same evidence
directory:

.. code-block:: sh

   LLK_PYTHON_LANE=torch_integration "$LLK_PYTHON" -m pytest test/python \
     -m torch_integration \
     -q --junitxml="$LLK_PYTHON_EVIDENCE_DIR/junit/torch_integration.xml"

   LLK_PYTHON_LANE=triton_integration "$LLK_PYTHON" -m pytest test/python \
     -m triton_integration \
     -q --junitxml="$LLK_PYTHON_EVIDENCE_DIR/junit/triton_integration.xml"

   LLK_PYTHON_LANE=selected_avx2 "$LLK_PYTHON" -m pytest test/python \
     -m selected_avx2 \
     -q --junitxml="$LLK_PYTHON_EVIDENCE_DIR/junit/selected_avx2.xml"

The Triton corpus reports native calls made in its isolated worker. The checker
requires JUnit and evidence files from **all five** lanes; missing lanes fail.
Use a fresh evidence directory for each complete run to avoid stale results.
Enforce the declared results with:

.. code-block:: sh

   "$LLK_PYTHON" test/python/check_required_python_execution.py \
     --manifest test/Frontend/acceptance_manifest.json \
     --junit-dir "$LLK_PYTHON_EVIDENCE_DIR/junit" \
     --evidence-dir "$LLK_PYTHON_EVIDENCE_DIR"

The checker prints one stable code per failure and exits nonzero:

``required_execution_missing``
   A required lane collected or executed nothing.
``required_lane_skipped``
   A required lane skipped a case.
``required_execution_failed``
   A required lane reported failures or errors.
``native_invocation_missing``
   A lane that must call native code recorded no native-call counter.
``selected_reference_fallback``
   The selected lane did not report ``selected-target`` execution, or it
   reported reference groups.

As an alternative to the manual pytest commands, register all five lanes and
the checker with CTest after provisioning the same dependencies. Export
``LLK_TRITON_PYTHONPATH`` before configuration; CMake records it for the tests.
``LLK_ENABLE_PYTHON_TESTS`` requires both the C API and ``BUILD_TESTING``:

.. code-block:: sh

   cmake -S . -B build \
     -DBUILD_TESTING=ON \
     -DLLK_BUILD_PYTHON_CAPI=ON \
     -DLLK_ENABLE_PYTHON_TESTS=ON \
     -DPython3_EXECUTABLE="$LLK_PYTHON"
   cmake --build build --target check-llk -j 4

Selected AVX2 execution
-----------------------

The ``selected_avx2`` lane runs the upstream Triton corpus on the AVX2
selected-target backend. It requires a Linux x86_64 host with AVX2, the pinned
Triton source checkout named by ``LLK_TRITON_PYTHONPATH``, and
``LLK_FRONTEND_LIBRARY`` naming the shared library. On other hosts the corpus
reports ``selected_target_supported: false`` and the checker rejects the lane
with ``selected_reference_fallback`` instead of accepting reference results.

Troubleshooting
---------------

``native_backend_unavailable``
   ``LLK_FRONTEND_LIBRARY`` is unset, missing, or cannot be loaded. Build
   ``LLKFrontendCAPI`` and point the variable at the shared library.

``frontend_api_version_mismatch``
   The loaded library reports a different C API version than this package.
   Rebuild the package and library from the same revision.

``native_configuration_missing`` / ``native_configuration_corrupt``
   The installed ``llk/config`` tree does not match ``llk/resources.json``.
   Reinstall the wheel; do not edit installed target policy in place.
