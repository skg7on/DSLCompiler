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
PF04 provides typed AST capture for the subset described below. Executable
decorators, Inductor composition, training and genuine upstream Triton launches
remain later tasks under
`#203 <https://github.com/skg7on/DSLCompiler/issues/203>`__.

Typed program and binding contract
----------------------------------

PF03 (`#214 <https://github.com/skg7on/DSLCompiler/issues/214>`__) adds the
``llk.frontend`` metadata API. It provides frozen ``TensorSpec``, ``LaunchSpec``,
``SourceSpan``, ``Value``, ``Operation``, ``Region``, ``FrontendProgram``,
``NumericalContract``, ``CompileOptions``, ``CompileRequest`` and
``CompilationReport`` records. Mappings and sequences are copied into immutable
snapshots. This API validates authored programs and binds arguments; PF04 captures
supported Python source into those records. Native compilation remains a later
milestone.

``bind_kernel(fn, args, kwargs, grid, options)`` returns a ``KernelBinding`` with
ports in signature order, bound constexprs, accepted immutable closure/global
constants, source text/location, launch metadata and options. Parameters annotated
with the actual ``llk.language.constexpr`` sentinel are compile-time values.
Renamed imports, module attributes and resolvable postponed annotations work by
symbol identity. Annotation call expressions are rejected without evaluation,
including Python 3.14 deferred annotations. An annotation-only local alias lost
by postponed annotation storage on older Python versions fails explicitly.
Binding errors carry ``file``, ``line`` and ``column`` attributes.

Tensor arguments require an explicit ``TensorSpec`` or
``TensorArgument(spec, provider)``. Tensor metadata describes positive static
rank-two compact row-major storage and an explicit input/read or output/write
effect. Binding retains providers strongly and never inspects their addresses,
converts their data or infers output effects from argument names. Actual provider
layout, allocation, storage offset and alias validation belongs to PF08 before
invocation. A bare ``TensorSpec`` supports metadata-only binding with no provider.

Tuple grids have one to three positive integer axes, padded with ones to a
three-tuple. A callable grid receives immutable constexpr metadata and runs
exactly once after argument validation. Such grids use block ``(1, 1, 1)``;
pass an explicit ``LaunchSpec`` to carry other block dimensions. Boolean extents,
nonpositive dimensions and signed 64-bit overflow fail. Affine access and
whole-domain coverage proofs belong to the PF05 native importer.

Schema v1 and canonical bytes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :source:`JSON schema <docs/design/python-frontend-v1.schema.json>` fixes the
wire fields, supported opcodes and per-op attribute/arity grammar. The
:source:`literal add request <test/python/frontend_add_v1.json>` pins a complete
32-by-64 f32 add program with two read inputs, one write output, masks and explicit
load padding. ``serialize_request`` returns compact UTF-8 JSON bytes with sorted
object keys; ordered ports, operations and region arguments retain their order.
It never includes providers or runtime addresses. ``deserialize_request`` rejects
unknown/missing fields, duplicate keys, unknown versions and non-finite numbers.

Python validation additionally checks SSA definition/use, region scope, exact
mask types/shapes, read/write effects, checked allocation products and dependent
operation types. Integer indices and extents must be actual signed 64-bit
integers; booleans and floating values such as ``1.0`` fail even where JSON
Schema's mathematical integer definition accepts them.

A ``counted_loop`` has operands ``(lower, upper, init...)``, scalar index bounds,
a positive constant ``step`` attribute, and one result per init value. Its one
``Region`` has arguments ``(iv, iter...)`` and a terminal ``yield`` matching the
carried result types and order. Nested loop values stay local. Top-level programs
end with a void ``return``; other region kinds are rejected.

``NumericalContract`` records storage, compute and accumulator dtypes, ordered
``CastSite`` records keyed by cast result id, rounding, activation policy,
reduction order, reassociation, FMA and subnormal policy. Program and compile
options must agree on this contract. Explicit cast sites must match the program
exactly, including casts inside nested loops.

``CompileOptions`` defaults to ``x86-avx2``, ``reference``, deterministic search,
64 candidates and top-1 retention. Its default ``config_root`` is the installed
package resource directory. Construction snapshots YAML and LLKMap file contents
under that root; an explicit ``config_files`` mapping supplies an existing
snapshot. Canonical identity includes source, accepted constants, launch,
numerical policy, options and configuration contents. Wire decoding requires an
explicit snapshot object and does not read local files. PF03 does not bundle a
native target package or establish executable support for an options value.

Run the compiler-independent contract checks with:

.. code-block:: sh

   "$LLK_PYTHON" -m pytest test/python/test_frontend_binding.py \
     test/python/test_frontend_schema.py -q

The :source:`PF03 verification record <docs/reviews/2026-10-23-pf03-typed-frontend.md>`
records versions, nonzero test counts and the reviewed boundary.

Typed AST capture
------------------

PF04 (`#215 <https://github.com/skg7on/DSLCompiler/issues/215>`__) provides
``llk.language._ast_compiler.compile_kernel(fn, binding) -> FrontendProgram``.
The explicit ``KernelBinding`` supplies the source and frozen argument/capture
values. Capture never executes the function body or arbitrary Python calls.
Renamed module and directly imported language symbols resolve by identity.
Calling a language operation eagerly raises ``outside_frontend_capture``.

Supported source includes name assignments and aliases, recursive numeric
arithmetic, exact floating negation, comparisons, full slices with ``None`` axes, mask ``&``/``|``/``~``,
``where``, static ``arange``, ``program_id``, ``cdiv``, explicit memory operations,
zero tiles, dot products and explicit conversions. Memory calls preserve
positional/keyword operands; duplicate, unpacked and unknown keywords fail.
Loads require explicit ``other`` padding. Masks must be i1 and broadcast to the
memory tile; stores require the exact value shape/type. No implicit conversion
between SSA value dtypes or storage providers occurs.

Only literal/constexpr conditions choose a branch during capture. Runtime
branches, ``while``, arbitrary calls, arbitrary loop-carried state and value
returns fail explicitly. Counted ``range`` loops retain typed induction and
accumulator arguments, a region body and matching yields; the body is not
unrolled. Accepted carried values are rank-two f32 dot accumulators, including
nested loops and explicit intermediate rounding casts. Other control flow
requires a separate extension.

Induction names must be fresh within their enclosing environment; reusing a
preexisting value or an enclosing loop's induction name fails explicitly.
Floating unary minus uses the checked ``neg`` opcode, preserving signed zero
and sign/payload semantics without substituting arithmetic subtraction.

Tensor-pointer arithmetic retains its declared tensor base and element offsets.
Capture expresses a flat offset as row ``offset div width`` and column
``offset rem width`` using the declared compact physical shape; it never invents
a rank-one runtime ABI. Bounds, affine bijection and full-domain single-writer
coverage remain checks for the PF05 importer. A captured program alone is not
evidence that such a proof or an execution succeeded.

Every emitted operation has an original file/line/column, including indentation
removed to parse nested functions. Unknown source, calls and mutable captures
produce structured diagnostics. The exact recognized expression
``x / (1 + tl.exp(-x))`` and ``tl.silu(x)`` capture explicit exact SiLU metadata;
general ``exp`` is unsupported. A relaxed activation requires matching explicit
numerical permissions. Cast result ids and rounding sites remain ordered, even
inside loops. Source-specific facts extend the program's numerical contract;
construct a request using that full contract:

.. code-block:: python

   from dataclasses import replace
   from llk.frontend import CompileRequest
   from llk.language._ast_compiler import compile_kernel

   program = compile_kernel(fn, binding)
   options = replace(binding.options,
                     numerical_contract=program.numerical_contract)
   request = CompileRequest(program, options)

``@llk.jit`` definitions expose ``frontend_ir(binding)`` for the same metadata.
The former ``ttir()`` method raises ``upstream_toolchain_unqualified``; private
frontend records are never labeled upstream TTIR. The old string builder raises
``legacy_ir_builder_unavailable``. Invocation still raises
``native_backend_unavailable`` until the native service/runtime milestones land.

.. code-block:: sh

   "$LLK_PYTHON" -m pytest test/python/test_ast_semantics.py \
     test/python/test_frontend_diagnostics.py -q

See the :source:`PF04 verification record <docs/reviews/2026-10-23-pf04-typed-ast.md>`
for the tested source/diagnostic contract and its capability limits.

Legacy Triton safety boundary
-----------------------------

PF02 (`#213 <https://github.com/skg7on/DSLCompiler/issues/213>`__) repairs the
experimental quoted ``tt.*`` format consumed by the native passes. It is not
upstream TTIR or a Python compilation path. Each pass validates the complete
legacy input before rewriting. Malformed schemas report
``triton_invalid_schema``; unsupported operations report ``triton_unsupported_op``;
grid functions returning values report ``triton_grid_result_unsupported``.

Block pointers accept static positive rank-one/two compact host memrefs with
zero storage offset. Declared physical strides must match the base layout.
Logical subview steps are one; view sizes are clipped to the remaining base
extent. Offsets must be nonnegative and strictly below the base extent; constant
violations fail compilation and dynamic violations abort through ``cf.assert``
before memory access. Transfer boundary masks are combined with explicit i1
vector masks. Loads pad masked lanes with zero.

Atomics and their old ``llrt.atomic_*`` declarations/calls, scratch allocation,
asynchronous copy/await, advance, reductions, broadcasts and transposes remain
unsupported. The final experimental executable boundary must run
``--triton-cpu-verify=reject-residual=true`` after supported lowering to reject
any remaining ``tt.*`` operations. ``TritonFullPipeline`` tests that sequence
with MLIR verification enabled; native load/store tests validate numerical rows
and guard buffers. They do not establish Python or upstream Triton execution.

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
``backend_registration_conflict``. Decorator inspection returns checked frontend
metadata; it does not establish verified MLIR, real upstream TTIR or a native
JIT invocation.
