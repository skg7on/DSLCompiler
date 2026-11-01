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

PF06 exposes the native C++ compilation/invocation service, PF07 the versioned C
ABI and PF08 the Python decorator path that executes through it. ``@llk.jit``
kernels invoke native code synchronously over caller-owned buffers for the
strict-f32 rank-two add/multiply subset described below. The ``llk`` Dynamo
backend still raises ``UnsupportedFeatureError`` with code
``native_backend_unavailable``; real ``torch.compile`` wraps this in
``BackendCompilerFailed.inner_exception``. The registration tests verify that
diagnostic; their passing result does not establish native kernel execution
through Dynamo. Inductor composition, training and genuine upstream Triton
launches remain later tasks under
`#203 <https://github.com/skg7on/DSLCompiler/issues/203>`__.

Typed program and binding contract
----------------------------------

PF03 (`#214 <https://github.com/skg7on/DSLCompiler/issues/214>`__) adds the
``llk.frontend`` metadata API. It provides frozen ``TensorSpec``, ``LaunchSpec``,
``SourceSpan``, ``Value``, ``Operation``, ``Region``, ``FrontendProgram``,
``NumericalContract``, ``CompileOptions``, ``CompileRequest`` and
``CompilationReport`` records. Mappings and sequences are copied into immutable
snapshots. This API validates authored programs and binds arguments; PF04 captures
supported Python source into those records. PF06 adds native C++ compilation;
the Python bridge remains a later milestone.

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
``legacy_ir_builder_unavailable``. Capture itself performs no compilation; PF08
adds synchronous invocation to the decorated kernel.

.. code-block:: sh

   "$LLK_PYTHON" -m pytest test/python/test_ast_semantics.py \
     test/python/test_frontend_diagnostics.py -q

See the :source:`PF04 verification record <docs/reviews/2026-10-23-pf04-typed-ast.md>`
for the tested source/diagnostic contract and its capability limits.

Checked native structured import
--------------------------------

PF05 (`#216 <https://github.com/skg7on/DSLCompiler/issues/216>`__) adds the C++
``LLKFrontend`` library. ``llk::frontend::parseRequest(StringRef)`` returns an
owned, immutable ``FrontendRequest`` after LLVM JSON decoding, strict schema-v1
checks and typed SSA/effect/region validation. Duplicate keys (including escaped
aliases), fractional integer tokens, signed-64 overflow, non-finite numbers and
unknown fields fail with ``invalid_frontend_schema``. Requests are bounded to
16 MiB and 64 JSON nesting levels. Configuration contents must be an explicit
snapshot; decoding never reads files or executes Python.

``importProgram(request, context)`` proves accesses before constructing MLIR.
The initial accepted envelope is two read-only inputs and one write-only output
of the same positive static rank-two compact f32 shape, computing exactly one
strict add or multiply per output element. Checked affine expressions over
program ids and two lane axes establish in-bounds reads, matching input/output
coordinates, all-active masks and exact single-writer coverage. Constant-divisor
quotient/remainder expressions from PF04 flat offsets are accepted when the
residue bounds prove their affine form. Grid times BLOCK must equal each tensor
axis; the third grid/BLOCK axes must be singleton. Tails, flattened lane tiles,
axis permutations, non-affine indexing, loops, casts, activations and relaxed
numerical policies fail explicitly in this elementwise importer. Counted loops
and activations are accepted only by the contraction family described below,
which proves its own structural predicates rather than relaxing this one.

The resulting ``ImportedProgram`` owns a verified module with registered
``func``, ``arith``, ``tensor`` and ``linalg`` operations. Its source selection
uses the existing ``linalg-elementwise-v1`` adapter and its canonical workload
contract. The entire canonical request remains a module attribute, preserving
source, launch, ordered metadata and configuration identity. Source BLOCK sizes
become singleton search domains with source-owned roles ``axis:d0`` and
``axis:d1``; realization emits the adapter's receipt and cannot retune them.

The explicit public/native port map preserves signature order. Native adapter
ports are two read inputs, an internal no-read destination (port 2), and the
write output (port 3). The destination is not a public buffer. PF06 allocates separate owned scratch
for it. The MLIR context must outlive the
imported module. This API establishes structured import and source realization;
Python native compilation/invocation is delivered separately by PF06–PF08.

.. code-block:: sh

   cmake --build build --target LLKFrontendTests
   build/bin/LLKFrontendTests --gtest_filter='FrontendProgram.*:FrontendImporter.*'

Native tests consume both the literal PF03 request and live PF04 add, multiply
and nested-loop capture requests generated with Python 3.11+ during the test
build. No Torch or upstream Triton is needed for these fixtures. See the
:source:`PF05 verification record <docs/reviews/2026-10-23-pf05-native-import.md>`.

Shared native compilation and invocation
----------------------------------------

PF06 (`#217 <https://github.com/skg7on/DSLCompiler/issues/217>`__) adds
``llk::frontend::compileFrontend(request) -> Expected<FrontendCompilation>``.
The service imports the checked request, uses the workload-contract overload of
``runMappedTuningSession`` and retains one candidate under explicit deterministic
search limits. It never translates elementwise work into synthetic contraction
extents. Initial retention is ``top_k=1``; wider requests fail explicitly.

Target packages load only the request's frozen ``config_files`` snapshot,
materialized in a private temporary directory for the existing registered target
factory. ``config_root`` remains part of request identity; its live contents are
never a fallback. Supply the target's machine/layout/rule files in their package
relative paths. Empty or invalid snapshots fail before native compilation.

The shared ``compileMappedTuningSelection`` completion helper re-instantiates the
retained source and checks source, binding, graph, target and plan identities,
plus complete source/mapping ownership receipts, before calling
``compileMappedKernel``. Changed source, binding or target content and stale plan
or receipt records reject. Existing CLI/replay and capability formats remain the
native contracts; no separate search or optimizer is introduced.

``FrontendCompilation`` owns the executable, report/inspection strings and
independent internal destination storage. ``invoke(inputs, outputs)`` takes
typed ``InvocationBuffer2D`` records in public signature order, filtered by input
and output direction. It adds the private no-read destination to native input
port 2, then calls ``MappedExecutable::invoke`` with the existing allocation,
shape, dtype, compact-layout and disjointness checks. Calls on one compilation
are serialized. The caller retains public buffers until the synchronous call
returns; no input copy or dtype conversion occurs.

``report()`` contains the full native request identity, compiler/workload/target/
plan/ABI identities, actual native group and search counters, source bindings,
ownership receipts, public/native adaptation and CapabilityReceipt v2. It is a
compilation snapshot: invocation remains ``not_checked`` there. Reference and
selected-target execution have different identities. Selected mode checks the
target's declared ISA and never substitutes reference execution. Native counters
are preserved as defined by the compiler; a reference backend does not imply a
positive ``referenceGroupsLowered`` counter when all semantic emitter groups
lowered before the portable backend.

``inspect("frontend")``, ``inspect("llk")`` and ``inspect("micro")`` return owned
request, structured source and verified bound MicroIR text. The bound text is
captured once inside the shared compiler before lowering consumes the module.
Unavailable stages, including ``asm``, report ``inspection_unavailable``. No MLIR
object escapes the service's context lifetime. PF07 adds the versioned C ABI and
PF08 the Python launch described below.

.. code-block:: sh

   cmake --build build --target LLKFrontendTests
   build/bin/LLKFrontendTests --gtest_filter='FrontendCompiler.*'

See the :source:`PF06 verification record <docs/reviews/2026-10-24-pf06-native-compilation.md>`
for actual reference/selected lanes and native guard results.

Counted contractions and fused SwiGLU
-------------------------------------

PF11 (`#222 <https://github.com/skg7on/DSLCompiler/issues/222>`__) recognizes a
canonical counted-K loop with one or two ``tl.dot`` accumulators and emits
``llk.matmul`` or ``llk.fused_swiglu``. The tile program is only accepted when it
is exactly the whole-tensor contraction:

* the launch tiles the output domain exactly, with ``grid * block == shape`` and
  singleton third axes;
* the loop starts at zero and traverses exactly the declared ``K`` extent in
  whole ``BK`` blocks;
* every load reads the compact row-major tensor it names at
  ``row = program_id * block + lane`` and ``col = k + lane``;
* accumulators are explicit loop-carried values, seeded either by a zero tile or
  by a caller tensor; and
* the two-projection form reads the *same* left-hand tile, uses two distinct
  weight tensors, and applies exactly ``silu(gate) * up``.

The frontend's contraction contract is uniform f32 and static. A bf16 captured
source declines, because every explicit cast is a numerical site the fused f32
epilogue does not realize. A caller accumulator is preserved for matmul: it
becomes the public native ``init`` port and is read-only. A zero accumulator has
no public buffer; the compiler allocates a zeroed init port. Fused SwiGLU reads
no accumulator, so every public input is caller-owned.

The source family is realized with the static ``pad`` boundary policy on an
anchor derived from the target machine: the largest divisor tile of ``M x N x K``
whose whole-operand materialization fits the machine's staging capacity. A shape
with no feasible tile fails explicitly (``unsupported_frontend_program``) rather
than allocating a padded copy that exceeds the machine. Layout, memory path and
owner hierarchy are not source choices; the mapping stage solves them.

``silu``'s ``approximation="exact"`` maps to ``#llk.math_mode<strict>`` and
``"native"`` to ``#llk.math_mode<bounded_fast>``. The policy is part of the
source contract, the Micro-IR fact (``micro.numerical_contract``) and the cache
identity, so a relaxed activation is a different program.

Patterns outside the family fail by name: a second writer, a different
left-hand tile per projection, a reduction extent that is not ``K``, a
non-shared left operand, or any explicit intermediate cast
(``unsupported_numerical_contract``).

.. code-block:: sh

   cmake --build build --target LLKFrontendTests
   build/bin/LLKFrontendTests --gtest_filter='FrontendContractions.*'
   LLK_FRONTEND_LIBRARY=build/lib/libLLKFrontendCAPI.dylib \
       python -m pytest test/python/test_frontend_contractions.py -q

See the :source:`PF11 verification record <docs/reviews/2026-11-01-pf11-python-frontend-contractions.md>`.

Native decorator execution
--------------------------

PF07 (`#218 <https://github.com/skg7on/DSLCompiler/issues/218>`__) adds the
versioned ``LLKFrontendCAPI`` shared library and PF08 (`#219
<https://github.com/skg7on/DSLCompiler/issues/219>`__) loads it from ctypes.
Set ``LLK_FRONTEND_LIBRARY`` to a development build of
``libLLKFrontendCAPI``; otherwise the installed ``llk`` package looks for the
library at its resource root or ``lib/`` subdirectory. The bridge checks
``llk_frontend_api_version() == 1`` before any other entry point. A missing or
incompatible library raises ``UnsupportedFeatureError`` with
``native_backend_unavailable`` or ``frontend_api_version_mismatch``; it never
falls back to eager execution.

Both block-programming forms are synchronous and equivalent::

   grid = LaunchSpec((1, 1, 1), (32, 64, 1))
   launch = add_kernel[grid]
   launch(a, b, out, 32, 64, 32, 64)
   add_kernel(a, b, out, 32, 64, 32, 64, grid=grid)

Omission of ``grid`` fails with ``invalid_launch_coverage`` before binding or
native entry. Grid times BLOCK must equal every public tensor extent, and the
third grid/block axes must be singleton. A callable grid runs exactly once after
argument validation and must return a ``LaunchSpec`` (or tuple) that satisfies
the same coverage proof. ``@llk.jit`` and ``@llk.jit()`` both work; keyword
options become a frozen ``CompileOptions``, for example
``@llk.jit(backend="reference", config_root=..., config_files=...)``.

``kernel.compile(*args, grid=...)`` returns a ``CompiledKernelLease`` without
invoking. A lease is a ``CompiledKernel`` and exposes ``report`` (the checked
``CompilationReport`` including ``backend``, ``source_adapter``,
``inspection_stages``, ``native_counters`` and ``native_environment``),
``invoke(inputs, outputs)``, ``inspect(stage)``, the ``frontend_ir``, ``llkir``
and ``micro_ir`` inspection properties, and idempotent ``close``; it is also a
context manager. ``asm`` is optional and raises ``UnsupportedFeatureError``
with ``inspection_unavailable``. ``llk.compile_request(request)`` compiles an
explicit ``CompileRequest`` directly and bypasses the cache. Every
``invoke`` holds a lock for the whole native call, so ``close`` waits for an
in-flight invocation and later calls raise ``ExecutionError`` with
``invalid_frontend_handle``. The caller must close a lease; a plain decorator
call obtains a lease, invokes it and closes it.

Each ``JitFunction`` owns a process-local bounded LRU specialization cache
(default 128 entries). A lookup identity contains the canonical request bytes
(source, captured constants, concrete tensor signature, constexprs, launch,
numerical policy and the target configuration snapshot by content) plus the
loaded native environment: ABI version, library path and file identity. A
SHA-256 digest indexes the buckets but a hit always compares the full identity,
so a digest collision compiles a separate artifact. Concurrent misses for one
identity share a single compilation; a failed compilation is discarded rather
than cached, so a later call retries. ``kernel.cache_info()`` returns an
immutable record with ``hits``, ``misses``, ``compilations`` and ``evictions``.
Leases own a strong reference independently of cache membership: eviction drops
only the cache's reference, so an active lease keeps invoking the same native
executable safely, and changing the target configuration or native library
cannot reuse a handle compiled for different content.

Inputs and outputs are caller-owned providers wrapped in
``TensorArgument(TensorSpec, provider)``. A provider must expose the Python
buffer protocol (or an explicit ``__array_interface__``) with a positive rank-two
shape, compact row-major strides, an element type matching the declared
``TensorSpec``, enough allocation bytes for the declared extent, and writability
for outputs. Array-interface views whose ``base`` names a different data pointer,
negative or otherwise non-compact strides, unaligned pointers, mismatched
shapes/dtypes/extents, read-only outputs, non-buffer objects, and outputs that
overlap inputs or other outputs are rejected with ``TensorContractError`` and
``invalid_frontend_buffer`` before the JIT is entered. A raw byte provider must
be a unit-stride contiguous rank-one region; reversed or strided byte views are
rejected. The exposed buffer pointer is otherwise treated as the base of the
caller-owned allocation, matching the C ABI, so storage outside the exported
payload (for example canaries) is not part of the declared allocation. No input
copy, dtype conversion or provider caching occurs. The adapter holds a live
buffer export for every provider through the native call, so a resizable
``bytearray``/``array`` cannot move its storage mid-invocation, and the provider
objects and native executable stay strongly referenced until the call finishes.
``ttir()`` remains unavailable with ``upstream_toolchain_unqualified``;
the checked metadata is exposed through ``frontend_ir(binding)``.

The initial executable envelope is the PF05/PF06 one: positive static rank-two
compact row-major f32 add or multiply with exact two-input/one-output coverage,
ABI v1, and explicit ``reference`` or ``selected-target`` execution. Selected
mode checks the host ISA and never substitutes reference execution. The stock
``x86-avx2-v2`` machine profile cannot realize the literal 32×64 launch with
8×16 blocks: its shared storage analysis needs 147,456 B of ``sram.0`` for 16
simultaneous program occurrences against 32,768 B, so such a request fails with
``frontend_compilation_failed``/``memory_capacity_exceeded``. The acceptance
tests exercise the same kernel, adapter and launch path at 16×16/8×16 tiled and
32×64 full domain; broader tilings need a separate mapping-capacity follow-up.

.. code-block:: sh

   export LLK_FRONTEND_LIBRARY=$PWD/build/lib/libLLKFrontendCAPI.dylib
   "$LLK_PYTHON" -m pytest test/python/test_native_jit.py \
     test/python/test_buffer_contract.py test/python/test_launch_contract.py -q
   "$LLK_PYTHON" -m pytest test/python/test_specialization_cache.py \
     test/python/test_jit_lifetime.py test/python/test_inspection.py -q

Use ``libLLKFrontendCAPI.so`` on Linux. The library requires a build configured
with ``-DLLK_BUILD_PYTHON_CAPI=ON``. See the
:source:`PF08 verification record <docs/reviews/2026-11-01-pf08-python-native-jit.md>`
for exact commands, counts, buffer guards and native identities, and the
:source:`PF09 verification record <docs/reviews/2026-11-01-pf09-python-specialization-cache.md>`
for cache, lease and inspection evidence.

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
``backend_registration_conflict``. ``capabilities()`` remains compiler-free and
conservative about the Torch/Dynamo backend, which is still unavailable; it is
not evidence either way for a decorator invocation. ``frontend_ir`` returns
checked metadata only, while a decorated call with a loaded CAPI library
compiles and invokes native code as described above.
