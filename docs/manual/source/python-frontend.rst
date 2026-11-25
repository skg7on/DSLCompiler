.. _python-frontend:

Python frontend
===============

The ``llk-compiler`` distribution provides the ``llk`` package: a typed Python
kernel frontend feeding the same source adapters, mapping search and native
runtime as the compiler tools. Python capture, semantic proofs, target selection
and buffer invocation are separate checked boundaries. Capture does not execute
the kernel body, and the frontend does not introduce a second optimizer.

``@llk.jit`` executes positive static rank-two compact row-major f32 kernels:
elementwise add/multiply, canonical counted matmul and fused SwiGLU. Elementwise
kernels use two read inputs and one write output; rounded edges require a proven
in-bounds mask. Contractions have their own exact-coverage and numerical grammar. Execution is
synchronous through ``LLKFrontendCAPI`` over caller-owned buffers. The native
library must be supplied alongside the Python package (``LLK_FRONTEND_LIBRARY``
or a library next to the package). Target configuration is packaged: the wheel
carries ``llk/config/machines/*.yaml`` and ``llk/config/mapping/*/*.llkmap``
with ``llk/resources.json`` content hashes, and ``CompileOptions()`` validates
and snapshots those installed files. A missing or modified resource fails with
``native_configuration_missing`` or ``native_configuration_corrupt`` before a
compile request is built.

Other Python entry points share this native service:

* :ref:`Torch integration <python-frontend-torch>` provides strict
  ``torch.compile(backend="llk")``, checked functional operators,
  ``llk_inductor`` composition and bounded first-order autograd support.
* :ref:`Upstream Triton integration <python-frontend-triton>` exports genuine
  specialized TTIR and binds supported ``@triton.jit`` kernels to CPU execution.
  It requires the pinned upstream toolchain; in-process launches also require
  compatible native-library loading.

Native execution through these entry points uses concrete positive rank-two
f32 signatures. Native symbolic contractions have a separate
:doc:`dynamic-shapes` contract.

.. _python-frontend-pipeline:

Architecture and support boundaries
-----------------------------------

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Boundary
     - Responsibility
     - Current limit
   * - Bind and capture
     - Signature-ordered metadata and accepted captures → typed SSA ``FrontendProgram``
     - Recognized language operations only; capture coverage is wider than executable import
   * - Decode and prove
     - Canonical schema-v1 request → checked native request → structured Linalg or LLK contraction
     - Recognized f32 elementwise or contraction grammar, proven accesses and one writer per output element
   * - Realize and compile
     - ``linalg-elementwise-v1`` or ``llk-contraction-v1`` → shared mapped compilation → ``MappedExecutable``
     - Pinned elementwise blocks or machine-derived contraction anchor; deterministic search, ``top_k=1`` and complete storage analysis
   * - Invoke
     - Versioned C API → typed rank-two buffers → synchronous native call
     - ABI v1, compact zero-offset buffers and disjoint caller-owned outputs
   * - Reuse and inspect
     - Per-kernel bounded specialization cache, owned leases, reports and IR text
     - Process-local concrete signatures; no persistent cache or assembly inspection

Read :source:`Python records <python/llk/frontend/_ir.py>`,
:source:`AST capture <python/llk/language/_ast_compiler.py>`,
:source:`native importer <lib/Frontend/Importer.cpp>`,
:source:`compiler service <lib/Frontend/Compiler.cpp>` and
:source:`Python runtime <python/llk/runtime>` to follow a request.

.. _python-frontend-shapes:

Static Python signatures and dynamic native kernels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Python ``TensorSpec`` contains concrete positive extents. Shape, constexpr and
launch changes enter specialization identity and can compile another artifact.
Rounded tiles handle a non-divisible **static** shape; they do not make the
Python signature symbolic. One Python artifact cannot be reinvoked with different
descriptor extents.

The native contraction path instead supports symbolic workload contracts and
mapped ABI v2: one matmul/SwiGLU executable can accept independently changing
M/N/K, including zero extents, under its guards. The Python importer does not yet
expose that symbolic contract, dynamic rank, general broadcasting or arbitrary
reductions. Its matmul/SwiGLU family uses static f32 signatures and ABI v1.
See :doc:`dynamic-shapes` for compile-once native reuse.

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
   "$LLK_PYTHON" -m pytest test/python/test_frontend_binding.py \
     test/python/test_frontend_schema.py test/python/test_ast_semantics.py -q

If there is no environment yet, create ``.venv`` in an isolated worktree with
``python3 -m venv .venv`` and set ``LLK_PYTHON`` to its absolute interpreter path.
The package requires Python 3.11 or newer and has no mandatory Torch or NumPy
dependency. The ``test`` extra in
:source:`python/pyproject.toml <python/pyproject.toml>` supplies pytest and the
wheel-build tools without selecting or downgrading Torch.

The separate ``torch-test`` extra selects Torch 2.9.0 for the original Python
3.11/3.12 reference matrix. Use that extra in those reference environments,
rather than applying its pin to another Python environment. Torch is
optional for importing the package, but required for the real registration
tests. Upstream Triton, torchvision and torchaudio are not decorator execution
requirements.

.. _python-frontend-verified-versions:

Compatibility and capability reporting
--------------------------------------

The :source:`packaged compatibility manifest <python/llk/compatibility.json>`
records the Python 3.11/3.12 + Torch 2.9.0 registration baseline. Its
``native_execution: unavailable`` and ``upstream_triton: ttir-export-only``
entries are conservative package/registration metadata, as is
``llk.capabilities()``. They do not probe the loaded C API or qualify the local
Triton toolchain. Use ``CompiledKernel.report`` and checked invocation results
for native execution, and the qualification and launch APIs below for Triton.

.. _python-frontend-native-library:

Build the native bridge
------------------------

For native execution, configure and build in the worktree with the matching
LLVM/MLIR development build from :doc:`getting-started`:

.. code-block:: sh

   cmake -S . -B build -G Ninja \
     -DLLVM_PROJECT_BUILD_DIR=/absolute/path/to/llvm-project/build \
     -DLLK_BUILD_PYTHON_CAPI=ON
   cmake --build build --target LLKFrontendCAPI LLKFrontendTests
   export LLK_FRONTEND_LIBRARY=$PWD/build/lib/libLLKFrontendCAPI.dylib

See :source:`C API declarations <include/LLK/CAPI/Frontend.h>` and
:source:`Python launch tests <test/python/test_native_jit.py>` for ownership and
error translation examples.

Use ``libLLKFrontendCAPI.so`` on Linux. Python installation alone does not
supply this library. The bridge checks C API version 1 before using handles.

Typed program and binding contract
----------------------------------

The ``llk.frontend`` metadata API provides frozen ``TensorSpec``, ``LaunchSpec``,
``SourceSpan``, ``Value``, ``Operation``, ``Region``, ``FrontendProgram``,
``NumericalContract``, ``CompileOptions``, ``CompileRequest`` and
``CompilationReport`` records. Mappings and sequences are copied into immutable
snapshots. Binding validates authored metadata, capture creates these records
from supported source, and the native service independently checks and compiles
the resulting request.

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
layout, allocation, storage offset and alias validation runs before invocation.
A bare ``TensorSpec`` supports metadata-only binding with no provider.

Tuple grids have one to three positive integer axes, padded with ones to a
three-tuple and use block ``(1, 1, 1)``. A callable grid receives immutable
constexpr metadata and runs exactly once after argument validation. Pass or
return an explicit ``LaunchSpec`` to carry other block dimensions. Boolean extents,
nonpositive dimensions and signed 64-bit overflow fail. Affine access and
whole-domain coverage proofs belong to the native importer.

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
explicit snapshot object and does not read local files. The Python package does
not bundle the native shared library; options alone do not establish executable
support. Installed machine/layout/rule resources are sufficient for default
configuration. Set ``config_root`` to the worktree root when developing target
policy, or provide the required file contents explicitly in ``config_files``.

Run the compiler-independent contract checks with:

.. code-block:: sh

   "$LLK_PYTHON" -m pytest test/python/test_frontend_binding.py \
     test/python/test_frontend_schema.py -q

Typed AST capture
------------------

``llk.language._ast_compiler.compile_kernel(fn, binding) -> FrontendProgram``
performs typed AST capture.
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
unrolled. Accepted captured values are rank-two f32 dot accumulators, including nested
loops and explicit intermediate rounding casts. Native execution recognizes
only the canonical counted contraction family below; nesting or casts can still
capture successfully and then fail import. Other control flow requires an
extension.

Induction names must be fresh within their enclosing environment; reusing a
preexisting value or an enclosing loop's induction name fails explicitly.
Floating unary minus uses the checked ``neg`` opcode, preserving signed zero
and sign/payload semantics without substituting arithmetic subtraction.

Tensor-pointer arithmetic retains its declared tensor base and element offsets.
Capture expresses a flat offset as row ``offset div width`` and column
``offset rem width`` using the declared compact physical shape; it never invents
a rank-one runtime ABI. Bounds, affine bijection and full-domain single-writer
coverage remain checks for the native importer. A captured program alone is not
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
``legacy_ir_builder_unavailable``. Capture itself performs no compilation;
decorated calls proceed through native compilation and synchronous invocation.

.. code-block:: sh

   "$LLK_PYTHON" -m pytest test/python/test_ast_semantics.py \
     test/python/test_frontend_diagnostics.py -q

Checked native structured import
--------------------------------

The C++ ``LLKFrontend`` library independently checks serialized requests.
``llk::frontend::parseRequest(StringRef)`` returns an
owned, immutable ``FrontendRequest`` after LLVM JSON decoding, strict schema-v1
checks and typed SSA/effect/region validation. Duplicate keys (including escaped
aliases), fractional integer tokens, signed-64 overflow, non-finite numbers and
unknown fields fail with ``invalid_frontend_schema``. Requests are bounded to
16 MiB and 64 JSON nesting levels. Configuration contents must be an explicit
snapshot; decoding never reads files or executes Python.

``importProgram`` selects an elementwise or contraction proof before constructing
MLIR. The elementwise envelope is two read-only inputs and one write-only output
of the same positive static rank-two compact f32 shape, computing exactly one
strict add or multiply per output element. Checked affine expressions over
program ids and two lane axes establish in-bounds reads, matching input/output
coordinates and exact single-writer coverage. Constant-divisor quotient/remainder
expressions from captured flat offsets are accepted when residue bounds or the
exact consuming mask discharge their affine requirements. The third grid/BLOCK
axes must be singleton; each other grid axis must equal ``ceildiv(extent, BLOCK)``.
Flattened lane tiles, axis permutations, non-affine indexing, loops, casts,
activations and relaxed numerical policies fail explicitly in this elementwise
family. The separate contraction proof below recognizes counted dot loops and
the SwiGLU activation under its own predicates.

For divisible tiles, masks must be true throughout the tile. Rounded tiles must
prove exactly the in-bounds predicate ``(row < M) & (column < N)`` for the rounded
axes; constant-true terms on divisible axes disappear during mask normalization.
Missing, all-false, too-short, too-wide or otherwise unproved masks fail with
``unsupported_tail_policy``. Invalid address coverage fails with
``invalid_launch_coverage``. This is a bounded proof grammar, not an arbitrary
mask theorem prover.

A proved tail adds ``llk.tail_policy = "masked-edge"`` and ``llk.tail_block``
source facts plus a source-owned ``tail_policy`` search choice. The elementwise
adapter checks that the chosen block matches the proof, then emits interior and
static edge rectangles with bounded loads and clipped stores. The realization
receipt records these decisions; a retuned block cannot inherit another block's
proof. This extends static tiling, not symbolic shape support. See
:source:`ElementwiseSourceAdapter.cpp <lib/Conversion/Workload/ElementwiseSourceAdapter.cpp>`
and :source:`native tail tests <test/Frontend/tails.cpp>`.

The elementwise ``ImportedProgram`` owns a verified module with registered
``func``, ``arith``, ``tensor`` and ``linalg`` operations. Its source selection
uses the existing ``linalg-elementwise-v1`` adapter and its canonical workload
contract. The entire canonical request remains a module attribute, preserving
source, launch, ordered metadata and configuration identity. Source BLOCK sizes
become singleton search domains with source-owned roles ``axis:d0`` and
``axis:d1``; realization emits the adapter's receipt and cannot retune them.

The explicit public/native port map preserves signature order. For elementwise
kernels, native ports are two read inputs, an internal no-read destination
(port 2), and the write output (port 3). The destination is not public; the
compiler owns its storage. Contraction adaptation depends on the seed and family
as described below. The MLIR context must outlive the
imported module. This API establishes structured import and source realization. Compilation and
invocation follow through the shared service and C API described below.

.. code-block:: sh

   cmake --build build --target LLKFrontendTests
   build/bin/LLKFrontendTests --gtest_filter='FrontendProgram.*:FrontendImporter.*'

Native tests consume literal and live add, multiply, tail and nested-loop
capture requests generated with Python 3.11+ during the test build. Nested-loop
fixtures exercise capture and explicit import rejection. No Torch or upstream
Triton is needed for these fixtures.

Shared native compilation and invocation
----------------------------------------

``llk::frontend::compileFrontend(request) -> Expected<FrontendCompilation>``
provides the shared compilation service.
The service loads the frozen target package before import so contraction anchors
use its machine facts. It then uses the workload-contract overload of
``runMappedTuningSession`` and retains one candidate under explicit deterministic
search limits. It never translates elementwise work into synthetic contraction
extents. Retention is ``top_k=1``; wider requests fail explicitly.

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

``FrontendCompilation`` owns the executable, report/inspection strings and any
required internal destination storage. ``invoke(inputs, outputs)`` takes
typed ``InvocationBuffer2D`` records in public signature order, filtered by input
and output direction. When needed it inserts compiler-owned native input port 2:
a no-read destination for elementwise or a zeroed initializer for zero-seeded
matmul. Caller-initialized matmul and SwiGLU require no internal input allocation.
It then calls ``MappedExecutable::invoke`` with the existing allocation,
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
object escapes the service's context lifetime. The versioned C ABI exposes
this service to the Python launch described below.

.. code-block:: sh

   cmake --build build --target LLKFrontendTests
   build/bin/LLKFrontendTests --gtest_filter='FrontendCompiler.*:FrontendTails.*'

Counted contractions and fused SwiGLU
-------------------------------------

The native importer recognizes a canonical counted-K loop with one or two ``tl.dot`` accumulators and emits
``llk.matmul`` or ``llk.fused_swiglu``. The tile program is only accepted when it
is exactly the whole-tensor contraction:

* the launch tiles the output domain exactly, with ``grid * block == shape`` and
  singleton third axes;
* the loop starts at zero and traverses exactly the declared ``K`` extent in
  whole ``BK`` blocks;
* compact row-major addressing reads ``A[rows, k+rk]`` and ``B[k+rk, cols]``
  (both weights in SwiGLU), with output rows/columns derived from program IDs
  and their lane axes;
* accumulators are explicit loop-carried values, seeded either by a zero tile or
  by a caller tensor; and
* the two-projection form reads the *same* left-hand tile, uses two distinct
  weight tensors, and applies exactly ``silu(gate) * up``.

The frontend's contraction contract is uniform f32 and static. BF16 captured
sources are rejected; explicit casts introduce numerical sites this family does
not realize. A caller accumulator is preserved for matmul: it
becomes the public native ``init`` port and is read-only. A zero accumulator has
no public buffer; the compiler allocates a zeroed init port. Fused SwiGLU reads
no accumulator, so every public input is caller-owned.

The source family is realized with the static ``pad`` boundary policy on an
anchor derived from the target machine. It searches descending exact M/N
divisors with ``BK=K`` and selects a tile whose whole-operand materialization fits
the machine's staging and accumulator capacities. The instruction fragment is
the machine's largest-volume declared rank-three matrix shape; a machine without
such a fragment cannot supply this anchor. A shape
with no feasible tile fails explicitly (``unsupported_frontend_program``) rather
than allocating a padded copy that exceeds the machine. Layout, memory path and
owner hierarchy are not source choices; the mapping stage solves them.

``silu``'s ``approximation="exact"`` maps to ``#llk.math_mode<strict>`` and
``"native"`` to ``#llk.math_mode<bounded_fast>``. The policy is part of the
source contract, the Micro-IR fact (``micro.numerical_contract``) and the cache
identity, so a relaxed activation is a different program.

Contraction memory masks must be constant true. Rounded M/N or K blocks are
not accepted by this family, even though static masked edges are supported for
elementwise kernels. More general loop nests, extra writers, noncanonical
addresses, a different left tile per projection and a reduction extent other
than K fail with ``unsupported_frontend_program``. Explicit intermediate casts
and caller-seeded SwiGLU fail with ``unsupported_numerical_contract``.

Storage, computation and accumulation are f32 with RTNE rounding, source
reduction order, preserved subnormals and no reassociation or FMA permission.
Exact/native SiLU is the explicit activation choice above; this does not admit
arbitrary relaxed arithmetic. The static ``pad`` anchor uses exact divisor tiles
and ``BK=K``, so it introduces no additional reduction terms. Backend realizability
and full storage/capacity checks still apply after the anchor is chosen.

See :source:`Python contraction examples <test/python/test_frontend_contractions.py>`
and :source:`native contraction tests <test/Frontend/contractions.cpp>` for
canonical loops, public accumulator binding, NaN/Inf checks, numerical identity
and rejected patterns.

.. code-block:: sh

   cmake --build build --target LLKFrontendTests
   build/bin/LLKFrontendTests --gtest_filter='FrontendContractions.*'
   "$LLK_PYTHON" -m pytest test/python/test_frontend_contractions.py -q


Native decorator execution
--------------------------

Download :download:`python-add.py <examples/python-add.py>` for a complete
standard-library buffer example. After building and selecting the C API library
above, run it from the worktree root:

.. code-block:: sh

   "$LLK_PYTHON" docs/manual/source/examples/python-add.py

.. literalinclude:: examples/python-add.py
   :language: python

The example uses a static 1×7 domain and one 8×16 block. The mask proves the
seven active lanes, native invocation is checked against expected output values,
and the repeated call reuses the cached artifact. This small case fits the stock
machine profile; it is not a performance benchmark.

Python loads the versioned ``LLKFrontendCAPI`` shared library through ctypes.
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
native entry. Each grid axis must equal ``ceildiv(extent, BLOCK)`` for every
public output; extra whole blocks and partial coverage fail. Input extents may
differ for contractions, whose accesses the native importer proves separately. Rounded edge lanes
require the elementwise mask proof above; the contraction family instead
requires exact output and K coverage. The third grid/block axes must be singleton.
A callable grid runs exactly once after argument validation and must return a ``LaunchSpec`` (or tuple) that satisfies
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

Specialization cache
~~~~~~~~~~~~~~~~~~~~

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

Configuration contents are frozen when options are constructed. To pick up
changed target files, construct new options or a new decorated kernel; an existing
snapshot does not reread them. Providers and addresses are absent from request
identity and are checked on every invocation. This concrete-signature cache is
separate from native symbolic variant and measurement identities.

Buffer ownership and validation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
For ``@llk.jit``, ``frontend_ir(binding)`` exposes checked metadata and
``ttir()`` raises ``upstream_toolchain_unqualified``. Genuine upstream TTIR is
available through the separate :ref:`Triton exporter <python-frontend-triton>`.

The executable envelope is positive static rank-two compact row-major f32 with
ABI v1 and explicit ``reference`` or ``selected-target`` execution. Elementwise
kernels need exact two-input/one-output coverage with proved rounded edges;
matmul and fused SwiGLU need the counted contraction proof. Selected
mode checks the host ISA and never substitutes reference execution. For the
elementwise add example, the stock ``x86-avx2-v2`` profile rejects the 32×64 launch with
8×16 blocks: its shared storage analysis needs 147,456 B of ``sram.0`` for 16
simultaneous program occurrences against 32,768 B, so such a request fails with
``frontend_compilation_failed``/``memory_capacity_exceeded``. The acceptance
tests exercise the same kernel, adapter and launch path at 16×16/8×16 tiled and
32×64 full domain; other tilings still need complete mapping-capacity checks.

.. code-block:: sh

   export LLK_FRONTEND_LIBRARY=$PWD/build/lib/libLLKFrontendCAPI.dylib
   "$LLK_PYTHON" -m pytest test/python/test_native_jit.py \
     test/python/test_buffer_contract.py test/python/test_launch_contract.py \
     test/python/test_frontend_tails.py test/python/test_frontend_contractions.py -q
   "$LLK_PYTHON" -m pytest test/python/test_specialization_cache.py \
     test/python/test_jit_lifetime.py test/python/test_inspection.py -q

.. _python-frontend-torch:

Torch compilation and operators
-------------------------------

Import ``llk.torch`` to register the ``llk`` and ``llk_inductor`` backends and
the functional ``torch.ops.llk`` operators. Torch is optional for package
import, but these execution paths require real Torch and the native C API.
Use CPU tensors with positive concrete rank-two shapes, f32 storage, compact
row-major strides and zero storage offset. Outputs are freshly allocated;
unsupported inputs are rejected before a data pointer is requested.

Strict Dynamo compilation
~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``backend="llk"`` when every computational node must be represented by the
native frontend. AOTAutograd normalizes the captured graph to the recognized
ATen subset: same-shape add/multiply, matrix multiplication and fused
``silu(x @ wg) * (x @ wu)``. For example, with the native library selected:

.. code-block:: python

   import torch
   import llk.torch

   compiled_add = torch.compile(
       lambda a, b: a + b, backend="llk", fullgraph=True, dynamic=False
   )
   a = torch.ones((4, 5), dtype=torch.float32)
   b = torch.full((4, 5), 2.0, dtype=torch.float32)
   torch.testing.assert_close(compiled_add(a, b), a + b)

``dynamic=False`` requests concrete specialization when shapes change. It
does not expose the native symbolic ABI. ``options={...}`` accepts
``CompileOptions`` fields for that compilation, including
``backend="reference"`` or ``"selected-target"``. The default is reference
execution; selected mode requires the target ISA and permits no reference
fallback.

An unsupported graph raises ``UnsupportedFXError`` with its ``node_name``;
``torch.compile`` wraps backend compilation errors in
``BackendCompilerFailed.inner_exception``. A missing native library reports
``native_backend_unavailable``. The strict backend never substitutes
``gm.forward`` for native compilation.

Functional operators and Inductor composition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The registered operators are ``torch.ops.llk.add(a, b)``, ``mul(a, b)``,
``matmul(a, b)`` and ``swiglu(x, wg, wu)``. They use the shared request builder,
bounded specialization cache and checked Torch buffer adapter. Add/multiply
require matching shapes; matmul uses ``[M, K] @ [K, N]``; SwiGLU uses one
``[M, K]`` input and two ``[K, N]`` weights. FakeTensor implementations inspect
metadata only: they do not read pointers, compile or invoke native code.
Foreign ownership of the ``llk`` namespace is reported rather than replaced.

Use ``backend="llk_inductor"`` to place supported regions inside an
Inductor-compiled graph:

.. code-block:: python

   composed = torch.compile(
       lambda a, b: torch.sin(a + b),
       backend="llk_inductor", fullgraph=True, dynamic=False,
   )
   torch.testing.assert_close(composed(a, b), torch.sin(a + b))

Here the addition uses an LLK operator and Inductor compiles the remainder.
The partitioner clones the graph and replaces only effect-safe regions with
native-compatible operands. Symbolic extents, bf16 storage, non-compact
operands and graphs with in-place tensor effects decline LLK partitioning.
Declined nodes remain for Inductor to compile; there is no implicit input copy
or dtype conversion to make a region eligible. Fusion stops at the external
LLK call. ``options`` for this backend are scoped Inductor configuration
patches, rather than LLK ``CompileOptions`` fields.

For an FX ``GraphModule``,
``llk.torch.rewrite_supported_subgraphs(gm, example_inputs)`` returns the
rewritten graph and a frozen ``PartitionReport``. Its ``matched_nodes`` names
accepted region roots; ``declined`` pairs node names with stable reasons.
See :source:`partitioning implementation <python/llk/torch/_inductor_backend.py>`
and :source:`composition examples <test/python/test_inductor_composition.py>`.

First-order training
~~~~~~~~~~~~~~~~~~~~

Training support depends on the entry point:

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Entry point
     - Supported gradients
     - Boundary
   * - Functional ``llk::add/mul/matmul``
     - First-order registered autograd formulas
     - Saved operands stay alive; non-compact incoming gradients and required transpose copies are materialized explicitly before native calls
   * - Strict ``backend="llk"``
     - Add passes gradients through; multiply runs two native backward kernels
     - Matmul backward is rejected because the strict importer cannot realize its transpose copies
   * - ``backend="llk_inductor"``
     - Add, multiply and matmul regions, including native matmul backward calls
     - Inductor owns transpose copies and graph orchestration

``llk::swiglu`` rejects ``requires_grad`` inputs with
``unsupported_training_contract``; strict SwiGLU training is also rejected.
Higher-order gradients are outside this contract.

Wrap a training step in ``with llk.torch.record_native_calls() as recorder:``
and inspect ``recorder.report`` afterward. Its ``TrainingReport`` separates
forward/backward native ``CompilationReport`` identities and explicit ATen
materializations. Recorders are process-visible so AOTAutograd worker threads
are observed; unrelated concurrent training steps should use separate runs.
See :source:`training tests <test/python/test_torch_training.py>` for gradient
comparisons and native-call checks.

.. _python-frontend-triton:

Upstream Triton export and CPU launch
------------------------------------------

``llk.triton`` accepts genuine upstream ``@triton.jit`` sources through two
checked stages: specialized TTIR export into an unbound template, then binding
of concrete tensors, scalar values and a runtime grid into the shared native
frontend. Template export can succeed for constructs that native binding does
not support. Importing ``llk.triton`` itself does not import upstream Triton.

Qualify the upstream toolchain
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :source:`toolchain lock <python/llk/triton/toolchain-lock.json>` pins Triton
3.6.0 at source revision ``93f05e19a8024938c320d82854e307d602b5a05d``, LLVM
revision ``27d654c4c4e6eb7c19e46af20500200e793da7c7``, export target
``cuda:80`` and template/protocol version 1. The export target selects upstream
frontend semantics; CPU execution uses LLK's ``llk_cpu`` target and requires no
GPU. Set ``LLK_TRITON_PYTHONPATH`` to the directory containing the built
upstream ``triton`` package, for example a qualified source checkout's
``python/`` directory:

.. code-block:: sh

   export LLK_TRITON_PYTHONPATH=/absolute/path/to/triton/python
   "$LLK_PYTHON" -c 'import llk.triton; print(llk.triton.qualify_toolchain())'

The ``triton-test`` extra selects version 3.6.0 where a wheel is available,
but a version match alone does not satisfy the exact source/LLVM checks.
Missing, unreadable or mismatched toolchains raise
``upstream_toolchain_unqualified``. ``LLK_TRITON_TOOLCHAIN_LOCK`` selects an
explicit lock for an embedded or test environment; it does not bypass
qualification.

Export a neutral template
~~~~~~~~~~~~~~~~~~~~~~~~~

``capture_kernel(module_path, function, signature, ...)`` is the convenience
form of ``export_module(ExportRequest(...))``. For the checked-in add kernel:

.. code-block:: python

   from llk.triton import capture_kernel

   exported = capture_kernel(
       "test/python/compat/kernels/elementwise.py", "add_kernel",
       {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32",
        "n": "i32", "BLOCK": "constexpr"},
       constexprs={"BLOCK": 64},
   )
   print(exported.ttir)
   print(exported.template.identity())

The isolated worker runs upstream's own TTIR compiler stage and typed MLIR
bindings, returning only JSON. ``TritonExport`` contains genuine ``ttir``, an
``UnboundFrontendTemplate`` and observed ``toolchain`` identity. The template
records parameter kinds/access, constexprs, alignment constraints, typed
dataflow and source/TTIR hashes. Runtime tensor shapes and grid are deliberately
absent. No native kernel is compiled or invoked during export.

Launch with the CPU driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With Torch, the qualified upstream package and ``LLK_FRONTEND_LIBRARY``
available, the same fixture can be launched over CPU tensors:

.. code-block:: python

   import torch
   import llk.triton
   from llk.frontend import CompileOptions

   with llk.triton.activate(CompileOptions(backend="reference")) as active:
       from test.python.compat.kernels.elementwise import add_kernel

       x = torch.ones((1, 64), dtype=torch.float32)
       y = torch.full((1, 64), 2.0, dtype=torch.float32)
       out = torch.empty_like(x)
       launch = add_kernel[(1,)]
       launch(x, y, out, x.numel(), BLOCK=64)
       torch.testing.assert_close(out, x + y)
       report = active.last_launch_report()
       assert report is not None and report.native_invocations > 0

Run this example from the repository root so the fixture module is importable.
``activate`` serializes access to Triton's process-global driver and restores
the previous driver on success or error. It first probes native-library
coexistence in subprocesses in both load orders. Conflicting LLVM images can
prevent in-process launches, including on macOS; the diagnostic is
``upstream_toolchain_unqualified``. ``coexistence_status()`` exposes the probe
results. Process-isolated export and native request execution are separate
paths; their success does not establish in-process driver compatibility.

At launch, the driver checks artifact/template identity, CPU tensor metadata,
pointer alignment, scalar types and coverage before native compilation. Buffers
are caller-owned compact rank-two f32 with zero storage offset and disjoint
outputs. Calls are synchronous. The native cache includes concrete shape,
scalars, launch, template identity, numerical/configuration contents and loaded
library identity; upstream pointer specialization alone cannot reuse a native
handle for another shape.

``active.last_launch_report()`` and ``llk.triton.last_launch_report()`` return
launch evidence, or ``None`` before a launch. ``TritonLaunchReport`` records
template and native request identities, plan/ABI/backend identity, runtime and
native grids, block shape and native-call count. To require AVX2 execution,
select ``CompileOptions(backend="selected-target")`` on a compatible x86_64
host. Reference execution does not establish selected-target support.

Supported source patterns
~~~~~~~~~~~~~~~~~~~~~~~~~

The :source:`compatibility corpus <docs/design/triton-compatibility-v1.json>`
defines concrete executable fixtures for these patterns:

* Flat f32 add/multiply with full or tiled coverage, including a flattened
  lane block over declared rank-two storage.
* Two-axis f32 add/multiply with proved
  ``(rows < M) & (cols < N)`` masks for rounded static edges.
* Counted f32 matmul, zero or caller-seeded accumulation, and fused
  ``silu(gate) * up`` with a common left operand. Dot precision must be
  ``input_precision="ieee"``; output tiling is exact, K is a positive multiple
  of the reduction block, and contraction loads are fully in bounds.

Rank views from ``expand_dims``/``broadcast`` and the exact SiLU epilogue are
recognized within these proofs. General broadcasting, arbitrary layouts,
symbolic shapes and reduced-precision dot modes are not admitted. Machine
capacity and backend realizability still constrain each concrete launch.

Unsupported source reports a structured diagnostic:

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Construct
     - Diagnostic
   * - Atomics, read/write alias effects or asynchronous staging
     - ``unsupported_triton_memory_effect``
   * - Reductions, runtime control flow or GPU-only inline assembly
     - ``unsupported_triton_operation``
   * - Transposes or arbitrary reshapes
     - ``unsupported_triton_layout``
   * - Non-IEEE dot precision
     - ``unsupported_triton_numerics``
   * - Runtime strides or an unproved launch/access pattern
     - ``unsupported_triton_launch``

Use :source:`upstream kernel fixtures <test/python/compat/kernels>` and
:source:`compatibility tests <test/python/compat/test_triton_corpus.py>` for
complete sources and exact checked shapes. The experimental quoted ``tt.*``
passes described below have a separate input contract.

.. _python-frontend-testing:

Run focused tests
------------------

.. code-block:: sh

   "$LLK_PYTHON" -m pytest test/python/test_torch_registration.py \
     test/python/test_package_install.py -q
   LLK_FRONTEND_LIBRARY="$PWD/build/lib/libLLKFrontendCAPI.dylib" \
     "$LLK_PYTHON" -m pytest test/python/test_dynamo_backend.py \
     test/python/test_fx_importer.py -q
   LLK_FRONTEND_LIBRARY="$PWD/build/lib/libLLKFrontendCAPI.dylib" \
     "$LLK_PYTHON" -m pytest test/python/test_torch_ops.py \
     test/python/test_fake_tensor_contract.py -q
   LLK_FRONTEND_LIBRARY="$PWD/build/lib/libLLKFrontendCAPI.dylib" \
     "$LLK_PYTHON" -m pytest test/python/test_torch_training.py -q

The root :source:`pytest.ini <pytest.ini>` restricts default collection to
``test/python`` and adds the local ``python/`` source path. This avoids collecting
vendored GoogleTest's own Python tests, which use a different harness. Run from
the current worktree so imports and fixtures use the edited source.

The package tests build the actual wheel and install it offline into a fresh
Torch-absent virtual environment outside the checkout. Real-Torch subprocesses
test registration/reload identity, zero Dynamo resets, existing-owner protection,
broken optional-dependency propagation and the missing-library diagnostic. The
Torch compilation suites execute real ``torch.compile`` graphs when the CAPI
library is loaded and count every native invocation. Operator suites run
``torch.library.opcheck`` schema/fake checks and direct native operator calls,
and spy on ``data_ptr``/compilation during FakeTensor execution. Training tests
compare gradients with independent f64 finite differences and analytic f32
references, count distinct forward/backward native kernels, and run the strict
and Inductor training routes. Missing test/build dependencies are setup
failures, not successful or skipped coverage.

Required execution lanes
------------------------

The lanes whose execution is required are declared once in
``test/Frontend/acceptance_manifest.json`` and selected by registered pytest
markers:

``frontend_unit``
   Compiler-free capture, schema, packaging and error contracts.
``frontend_native``
   Reference-backend native compilation and invocation through
   ``LLKFrontendCAPI``.
``torch_integration``
   Real Torch registration, Dynamo, Inductor, operators and training.
``triton_integration``
   Pinned upstream Triton qualification, TTIR export and launch.
``selected_avx2``
   Linux x86_64/AVX2 selected-target execution. Reference results are never
   accepted for this lane.

A lane run sets ``LLK_PYTHON_EVIDENCE_DIR`` and ``LLK_PYTHON_LANE``; the test
session records its JUnit XML and one evidence JSON file containing the native
call counter (and, for the selected lane, the executed backend and group
counts). ``test/python/check_required_python_execution.py`` joins the two and
fails with stable codes:

``required_execution_missing``
   A required lane collected or executed nothing.
``required_lane_skipped``
   A required lane skipped a case.
``required_execution_failed``
   A required lane reported failures or errors.
``native_invocation_missing``
   A lane that must execute native code recorded no call counter.
``selected_reference_fallback``
   The selected lane did not report ``selected-target`` execution, or it
   reported reference groups.

.. code-block:: sh

   export LLK_PYTHON_EVIDENCE_DIR="$PWD/build/frontend-evidence"
   mkdir -p "$LLK_PYTHON_EVIDENCE_DIR/junit"
   LLK_PYTHON_LANE=frontend_native "$LLK_PYTHON" -m pytest test/python \
     -m "frontend_native and not torch_integration and not triton_integration" \
     -q --junitxml="$LLK_PYTHON_EVIDENCE_DIR/junit/frontend_native.xml"

This runs one focused lane. The checker requires matching JUnit and evidence
from all five lanes in a fresh directory; a partial run fails with
``required_execution_missing``. Follow the
:doc:`Python frontend tutorial </tutorials/python-frontend>` for the complete
sequence and the final checker command.

With ``-DLLK_ENABLE_PYTHON_TESTS=ON`` (which requires
``-DLLK_BUILD_PYTHON_CAPI=ON``) the five lanes and the checker are registered
with CTest, and ``check-llk`` builds the shared library and enforces them. The
option is off for ordinary compiler-only builds. The required CI jobs run the
lanes on Python 3.11 and 3.12 with Torch 2.9.0, once without Torch or Triton to
prove the optional-dependency-free import boundary, and once on Linux x86_64
AVX2 with the pinned upstream Triton checkout for the Triton and selected
lanes. The :doc:`Python frontend tutorial </tutorials/python-frontend>` gives
the exact commands.

C++ frontend tests remain registered with CTest. Continue to use
:doc:`contributing` for C++/MLIR and selected-target acceptance.

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
``backend_registration_conflict``. Importing ``llk.torch`` also registers
``llk_inductor`` (exposed as ``llk.torch.inductor_backend``) and exposes the
checked functional operators as ``torch.ops.llk.add``, ``mul``, ``matmul`` and
``swiglu``. ``capabilities()`` remains the
compiler-free package report; it does not probe the native library, so it is
not execution evidence either way. A decorated call with a loaded CAPI library
compiles and invokes native code as described above, and the strict Dynamo
backend uses the same service when the library is loaded.

Native tail fixtures cover 33×65 and 1×7 add/multiply, canaries, unchanged inputs,
mask/address rejection and source receipts. Selected-target checks require a
compatible host; an unavailable-host diagnostic does not prove selected execution.
Read :source:`Python tail tests <test/python/test_frontend_tails.py>` for the
concrete launch/mask contract. A compilation report remains a snapshot with
invocation ``not_checked``; it is not updated by later calls or an output checker.

Legacy Triton safety boundary
-----------------------------

The experimental quoted ``tt.*`` format is consumed by the native passes. It is not
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
