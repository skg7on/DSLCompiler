.. _dynamic-shapes:

Dynamic shapes
==============

LLK matmul and fused SwiGLU support independently varying runtime M, N and K,
including runtime weight dimensions. A single compiled dynamic kernel can run
multiple valid shapes without recompilation. Rank stays known at compilation;
tile capacities, instruction fragments, vector widths, pipeline depths and
selected resources stay fixed. Mixed static/dynamic tensor types use the same
checked path.

This guide covers the implementation delivered by issue #191's eight tasks
(PRs #260–#265, #270 and #271). The older
:source:`design proposal <docs/superpowers/specs/2026-10-17-dynamic-shape-llk-microir-design.md>`
and :source:`implementation plan <docs/superpowers/plans/2026-10-17-dynamic-shape-llk-microir.md>`
record the original design; current interfaces and registered tests define the
implemented boundary.

.. _dynamic-shapes-contracts:

Shapes and semantic guards
--------------------------

A dimension is a nonnegative constant or a named symbol. Repeated symbol names
assert equality across ports; two unrelated dynamic axes are not assumed equal.
Symbols can declare a minimum, an optional maximum and a positive
``multipleOf`` divisor. These contracts are rank-neutral; M/N/K are facts supplied
by the contraction adapter.

.. list-table::
   :header-rows: 1

   * - Kernel
     - Read-only inputs
     - Caller-owned output
   * - Matmul
     - A[M,K], B[K,N], init[M,N]
     - out[M,N]
   * - Fused SwiGLU
     - X[M,K], Wg[K,N], Wu[K,N]
     - out[M,N]

Matmul computes ``init + A × B`` with the source accumulator type and output
conversion. SwiGLU computes two zero-initialized projections and applies the
source SiLU/multiply/conversion epilogue; its destination's old contents are not
an initializer. Static contradictions fail during compilation. Unresolved
relationships become runtime checks, including agreement between both SwiGLU
weights and the output. An empty M or N does not suppress a conflicting K check.

The version-2 workload contract records symbolic ports, axes, constraints and
numerical semantics. The exported kernel records explicit argument/result
bindings for caller-owned destinations. Static version-1 contracts remain
readable with their existing canonical encoding and identity.
See :source:`ShapeContract.h <include/LLK/Shape/ShapeContract.h>` and
:source:`WorkloadContract.h <include/LLK/Conversion/Workload/WorkloadContract.h>`.

.. _dynamic-shapes-export:

Export a dynamic kernel
-----------------------

Download :download:`dynamic-matmul.mlir <examples/dynamic-matmul.mlir>`:

.. literalinclude:: examples/dynamic-matmul.mlir
   :language: mlir

Run from your worktree root with built tools:

.. code-block:: sh

   mkdir -p build/tutorial
   build/bin/llk-compile --emit=micro \
     docs/manual/source/examples/dynamic-matmul.mlir \
     > build/tutorial/dynamic-matmul.micro.mlir
   build/bin/llk-opt build/tutorial/dynamic-matmul.micro.mlir \
     -o build/tutorial/dynamic-matmul.roundtrip.mlir
   build/bin/llk-compile --emit=micro-search \
     docs/manual/source/examples/dynamic-matmul.mlir \
     > build/tutorial/dynamic-matmul.search.mlir

The concrete export adds ``micro.kernel @matmul_dynamic`` beside the source
function. Its signature includes a separate output destination as well as the
read initializer. Inspect the shape-check prologue, ``tail_policy = "bounded"``,
``micro.tile_load``, ``micro.mma`` with ``valid_k``, and clipped
``micro.tile_store``. The search export emits finite compile-time choices; it
does not enumerate runtime M/N/K values or compile a kernel per sample.

Any unresolved contraction axis selects an explicit bounded baseline instead
of a fixed-shape schedule-database entry, even when M alone is known. Custom
candidate schedules still have to prove legality for their declared domain.
``llk-compile --M/--N/--K`` only affect the legacy cache key; runtime dimensions
come from invocation descriptors.

.. _dynamic-shapes-boundaries:

Boundaries, empty dimensions and local storage
----------------------------------------------

``micro.tile_load`` copies only its valid rectangle into a fixed-capacity tile
and fills the remaining lanes with typed padding. ``micro.tile_store`` writes
only the valid destination rectangle. ``micro.mma valid_k`` reduces exactly
the valid K elements: it does not execute a full padded BK reduction. This
preserves the source's initialization, conversion and math-mode semantics,
including the strict signed-zero/nonfinite cases covered by acceptance tests.

M=0 or N=0 produces an empty output without compute or movement. K=0 still
performs initialization and stores the specified result: matmul preserves its
initializer and SwiGLU evaluates its epilogue from zero projections. Null data
pointers are allowed for genuinely empty buffers after descriptor and shape
validation.

External tensor dimensions may be dynamic, but local tiles and maximum
concurrent residency must be bounded at compilation. Complete-plan evaluation
certifies materialized storage, lifetimes and selected capacity before retaining
an executable candidate. A large external output is not a whole-output SRAM
allocation. Unbounded dynamic local allocation is rejected. An unknown
divisibility predicate requires a runtime guard or bounded movement; it is not
evidence that a full tile is legal.

.. _dynamic-shapes-invocation:

Invoke through mapped ABI v2
----------------------------

Use ``compileMappedKernel`` and ``MappedExecutable::invoke`` from C++.
``llk-compile`` exports/compiles but does not supply buffers or invoke code.
ABI v2 retains the descriptor transport and adds symbolic dimensions,
constraints and explicit argument/result bindings. Read ``KernelAbi`` from the
prepared executable for port order; do not infer callable arguments from the
original semantic function. ABI v1 remains the positive-static compatibility
format with its original hash.

Supply typed ``InvocationBuffer2D`` values carrying a ``MemRef2D``, dtype and
backing allocation size. Invocation binds symbols from the actual descriptors
and checks arity, rank, dimension equality, dtype, sizes, arithmetic overflow,
allocation bounds, strides and forbidden aliases before machine-code access.
The executable boundary remains rank two, compact row-major, zero offset and at
most 12 descriptors. Inputs are read-only and outputs must be disjoint from
inputs and one another. The caller owns outputs; per-call compiler scratch is
released after invocation. The deprecated untagged descriptor overload rejects
ABI v2 because it cannot validate dtype or allocation size.

For complete compile-once/invoke-many examples, use
:source:`dynamic_mapped_acceptance.cpp <test/Execution/dynamic_mapped_acceptance.cpp>`;
:source:`dynamic_mapped_invocation.cpp <test/Execution/dynamic_mapped_invocation.cpp>`
covers descriptor rejection. The public interfaces are
:source:`MappedCompilation.h <include/LLK/Conversion/MappedCompilation.h>`,
:source:`KernelAbi.h <include/LLK/Runtime/KernelAbi.h>` and
:source:`MappedExecutable.h <include/LLK/Runtime/MappedExecutable.h>`.

.. _dynamic-shapes-analysis:

Analyze concrete samples
------------------------

Dynamic tuning requires ``llk-tune --shape-profile=<path>`` (or an inline JSON
object) on the target-aware semantic-source path. For the contraction adapter,
samples replace exact-shape ``-M/-N/-K`` tuning. A profile supplies a nonempty
ordered set of complete symbol bindings with finite positive weights:

.. literalinclude:: examples/dynamic-shape-profile.json
   :language: json

Download :download:`dynamic-shape-profile.json <examples/dynamic-shape-profile.json>`.
Schema version 1 supports ``weighted_mean`` aggregation. Unknown fields,
duplicate keys, missing/unknown symbols, invalid extents or constraints and
invalid weights fail with ``shape_profile:`` diagnostics. ``max_variants`` is a
positive budget. Omitted weights default to 1.0, aggregation to
``weighted_mean`` and ``max_variants`` to 1; this example records all explicitly.
The current tuner retains the full-domain baseline.
Increasing the budget does not automatically generate specializations.

Each sample specializes an analysis clone, evaluates the complete selected plan
and records costs such as latency, DRAM/local bytes and padding fill elements.
Only complete analysis for every sample can rank a dynamic plan; absent profiles
or incomplete samples cannot become a one-iteration or zero-cost score. The JSON
report retains ``shape_profile``, per-candidate ``shape_samples`` and
``shape_aggregation``. Samples guide ranking; they do not narrow the baseline's
valid shape domain. Predictions remain uncalibrated model outputs.

A machine profile must describe the selected movement paths and enough local
capacity for the fixed tiles and simultaneous owners. The shipped CPU profile
is not automatically sufficient for every dynamic schedule. For this example,
prepare the same enlarged local capacities and explicit DRAM/accumulator paths
used by the CLI acceptance fixture. This is an illustrative analysis model,
not a measured CPU memory configuration. Run this Python from the worktree root:

.. code-block:: python

   from pathlib import Path

   output = Path("build/tutorial")
   output.mkdir(parents=True, exist_ok=True)
   text = Path("machines/x86-avx2-v2.yaml").read_text()
   text = text.replace("capacity_bytes: 8192", "capacity_bytes: 1048576")
   text = text.replace("capacity_bytes: 32768", "capacity_bytes: 1048576")
   for name, source, destination in (
       ("dram_to_acc", "dram.0", "acc.0"),
       ("acc_to_dram", "acc.0", "dram.0"),
   ):
       text += (
           f"  - id: {name}.0\n"
           f"    source: {source}\n"
           f"    destination: {destination}\n"
           "    bandwidth_bytes_per_cycle: 64\n"
           "    latency_cycles: 220\n"
           "    transaction_bytes: 64\n"
           "    transfer_engines: [dma.0]\n"
       )
   (output / "dynamic-machine.yaml").write_text(text)

Tune the checked-in source-plus-search-space fixture:

.. code-block:: sh

   build/bin/llk-tune --input=test/Tuning/Inputs/issue198/dynamic_matmul.mlir \
     --machine=build/tutorial/dynamic-machine.yaml \
     --shape-profile=docs/manual/source/examples/dynamic-shape-profile.json \
     --mapping-target=x86-avx2 --mapping-root=. --mapping-mode=exact \
     --mapping-backend=reference --candidate-source=semantic --source-symbol=matmul \
     --top-k=1 --mapping-report=build/tutorial/dynamic-tune.json \
     --candidate-artifacts=build/tutorial/dynamic-candidates \
     --output=build/tutorial/dynamic.schedule.yaml

The retained ``source.mlir`` includes ``micro.workload_contract``, which names the
symbols. Plain ``llk-compile --emit=micro`` export does not attach that contract;
use the tuner's artifact for symbol-based analysis. With one retained candidate,
the glob below selects its source:

.. code-block:: sh

   build/bin/micro-perf --machine=build/tutorial/dynamic-machine.yaml \
     --shape-values=M=7,N=13,K=5 \
     build/tutorial/dynamic-candidates/*/source.mlir

``--shape-values`` binds one observation on an analysis clone. A sampled report
records ``shape_values`` and ``sampled: true``. Missing bindings, unresolved loops
or resources, unsupported operations and event-expansion limits produce explicit
incomplete analysis. A standalone report can exit successfully while
``complete: false``; inspect completeness and diagnostics before using its costs.
The tuner's unbound source artifact can still have unresolved resource pools;
binding M/N/K alone does not establish complete physical analysis.

The registered CLI smoke test also exercises full/partial ``--shape-values``
and invalid weights:

.. code-block:: sh

   python3 test/Tuning/dynamic_profile_cli.py \
     build/bin/llk-tune build/bin/micro-perf .

The :source:`CLI test <test/Tuning/dynamic_profile_cli.py>` prepares its own
temporary model and artifacts using the same source/search fixture.
See :doc:`tools/performance` and :doc:`tools/tuning` for the other options.

.. _dynamic-shapes-replay:

Reuse code and replay a selection
---------------------------------

The plain compiled-variant identity API separates code inputs from observations.
Its code inputs include source content, symbolic contract and version, numerical
contract, guard domain, ABI version/hash, static schedule and selected resources,
target/backend, execution mode and compiler version. Measurement identity adds
concrete shape values or the ordered profile/weights. Reusing code across shapes
never licenses reuse of another shape's latency. Packed-weight keys also include
the actual weight K/N so different runtime weights cannot share an incompatible
packing.

Keep the tuner's original ``source.mlir`` and frozen ``plan.json`` together and
replay with the same machine/target/backend, as described in
:ref:`the compiler replay guide <tools-compiler-replay-a-saved-selection>`.
Changed source, contract, guards, ABI, numerical mode, schedule or target content
invalidates the corresponding reusable identity. An already compiled baseline
accepts different valid descriptor shapes without recompilation.

The current tuner also includes the frozen plan report in its compiled-variant
schedule identity. That report contains the sampling profile and costs, so
changing the profile can change the reported ``compiled_variant`` even when the
baseline code is equivalent. Cross-profile compiled-cache reuse is therefore
not guaranteed by this workflow; measurement identity also retains the profile.

The ``DynamicVariantSet`` API validates an ordered set of guarded variants plus
exactly one unguarded full-domain baseline last, all within an explicit budget.
The first matching guard wins; declaration order and budget are frozen replay
data. This is a validated API/data contract; the current tuning workflow selects
the baseline and does not build an automatic specializing dispatcher. See
:source:`CompiledVariantIdentity.h <include/LLK/Runtime/CompiledVariantIdentity.h>`
and :source:`dynamic_shape_replay.cpp <test/Tuning/dynamic_shape_replay.cpp>`.

.. _dynamic-shapes-coverage:

Coverage and retained limits
----------------------------

Compile-once acceptance exercises matmul and SwiGLU at
``(M,N,K) = (1,1,1), (7,13,5), (8,32,32), (9,33,35), (17,65,67)``, mixed-static
forms and each zero dimension. Tests check nonzero matmul initializers, exact K
tails, output canaries, descriptor failures and scratch release. Direct dynamic
elementwise MicroIR execution is also covered; the semantic
``linalg-elementwise-v1`` source adapter remains static-only.

Reference execution and selected AVX2 execution are separate evidence. Selected
AVX2 acceptance requires a compatible x86_64 AVX2 host and zero reference
fallback. Runtime-K contractions use shared exact-K loop lowering within that
selected path; static contractions retain their native vector path. Export or
successful compilation alone does not establish numerical correctness.

Dynamic rank, ragged/data-dependent shapes, arbitrary strides/offsets,
unbounded local tiles, general Linalg import and automatic guarded
specialization are outside this increment. The narrow Arm executable subsets
do not gain dynamic contraction execution from this work. Native Python
frontend execution remains a separate milestone (see :doc:`python-frontend`).
