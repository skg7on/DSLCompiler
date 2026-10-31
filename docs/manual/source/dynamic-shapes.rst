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

.. _dynamic-shapes-scenarios:

Supported scenarios
-------------------

Choose the entry path as well as the shape pattern. Dynamic contraction
execution is delivered through the mapped compiler and typed C++ invocation;
Python frontend execution and arbitrary semantic import have separate limits.

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Scenario
     - Supported behavior
     - Boundary
   * - Fully dynamic matmul or fused SwiGLU
     - One executable handles independently changing M, N and K, including
       runtime weight dimensions
     - Rank-two signatures; both SwiGLU weight arms must agree on K/N
   * - Mixed static/dynamic contraction
     - Fix any subset of M/N/K and vary the remaining dimensions
     - Every invocation must retain the declared constant extents
   * - Partial tiles and K tails
     - Non-divisible dimensions use bounded loads, clipped stores and exact
       reduction lengths
     - The selected target emitters must support those boundary requirements
   * - Empty output or empty reduction
     - M=0/N=0 is empty; K=0 retains initialization and epilogue semantics
     - Shape relationships and descriptors are still validated
   * - External tensors larger than local memory
     - Caller-owned inputs/outputs remain external; fixed local tiles stream
       through the computation
     - Local storage and simultaneous residency must be certified; descriptor
       arithmetic and backing allocations must fit
   * - Direct dynamic elementwise MicroIR
     - The acceptance suite also executes directly authored dynamic elementwise
       kernels
     - This does not extend the static-only ``linalg-elementwise-v1`` semantic
       source adapter
   * - Dynamic performance analysis and tuning
     - Analyze explicit bindings and rank complete plans over weighted samples
     - A model prediction for those observations, with explicit completeness;
       no calibrated timing or domain-wide latency claim
   * - Baseline reuse and frozen replay
     - Reinvoke a compiled baseline at valid new shapes and replay its saved
       source/plan
     - Automatic guarded specialization is not generated; cross-profile tuner
       cache reuse is not guaranteed

The numerical acceptance examples use BF16 contractions with FP32 accumulation,
plus direct f32 elementwise coverage. Export recognizes additional Micro tile
dtypes, but parsing/export support alone does not establish a matching target
rule, executable lowering or numerical acceptance for every dtype. Reference
execution and selected AVX2 execution are separate modes; selected AVX2 requires
a compatible x86_64 AVX2 host. See :ref:`coverage and limits <dynamic-shapes-coverage>`.

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

.. _dynamic-shapes-binding-algorithm:

How dimension resolution and guards work
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The contraction adapter resolves each semantic axis from all of its occurrences.
For K, for example, it examines A's columns and B's rows, and SwiGLU includes
both weight rows. Conflicting known constants fail immediately. A known constant
pins the axis; if no occurrence pins it, the adapter introduces the symbol
``K``. The same process resolves M and N. This keeps mixed signatures consistent
without treating an unknown extent as an integer sentinel in shape arithmetic.

At invocation, typed descriptor validation and symbol binding form a gate before
kernel entry:

#. Validate the buffer envelope, dtype, extent/stride/offset arithmetic, allocation
   bounds and output disjointness.
#. Match every ordered input/output port to its declared shape. Check constant
   axes directly; bind a symbol on its first occurrence and compare subsequent
   occurrences with that value.
#. Check every bound symbol against its minimum, maximum and ``multipleOf``.
   Only a consistent complete binding may enter generated code. Exported IR also
   carries assertions for unresolved semantic dimension equalities.

Thus ``X[7,5], Wg[5,13], Wu[5,13], out[7,13]`` binds
``{M:7, N:13, K:5}``. Replacing Wu with ``[6,13]`` is an error rather than a
different workload. Making X/output empty in M does not make that K disagreement
legal. The binding implementation is :source:`ShapeContract.cpp <lib/Shape/ShapeContract.cpp>`;
the descriptor gate is :source:`MappedInvocation.cpp <runtime/MappedInvocation.cpp>`.

The reusable predicate classifier exposes three conservative verdicts against
the declared domain; one example input cannot establish a domain-wide fact:

.. list-table::
   :header-rows: 1

   * - Verdict
     - Meaning
     - Example for ``K % 32 == 0``
   * - ``Proven``
     - Every admitted extent satisfies the predicate
     - The contract declares ``K.multipleOf = 32``
   * - ``Disproven``
     - No admitted extent satisfies it
     - K is constrained to exactly 35
   * - ``RuntimeGuard``
     - The declared facts do not settle it
     - K is an otherwise unconstrained nonnegative symbol

A path requiring a disproven predicate is illegal. An unresolved predicate
requires a checked guard or a bounded realization; it cannot silently justify a
full-tile access. This classifier supports divisibility, equality and lower-bound
predicates over range/congruence constraints. It does not solve arbitrary
symbolic expressions or automatically create guarded specializations.

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

.. _dynamic-shapes-tiling-algorithm:

Fixed capacities with runtime tile counts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The exporter fixes ``BM``, ``BN`` and ``BK`` in the schedule and computes the
number of tiles from runtime extents. It implements ``ceildiv(E,T)`` as
``E // T + (E % T != 0)`` instead of forming ``E + T - 1``, avoiding that addition's
overflow near the index limit. Each visited tile uses
``valid = min(capacity, extent - offset)``. Offsets are generated only for tiles
inside the extent, so empty axes visit no tile.

The following is semantic pseudocode for matmul. The actual MicroIR expresses
M/N work with spatial loops, K with a temporal loop, and staging with
``micro.pipeline`` and completion tokens:

.. code-block:: text

   for mi in [0, ceildiv(M, BM)):
     m0 = mi * BM; vm = min(BM, M - m0)
     for ni in [0, ceildiv(N, BN)):
       n0 = ni * BN; vn = min(BN, N - n0)
       acc = load_and_convert_init_tile(m0, n0, valid=[vm, vn])
       for ki in [0, ceildiv(K, BK)):
         k0 = ki * BK; vk = min(BK, K - k0)
         a = bounded_load(A, [m0, k0], valid=[vm, vk], capacity=[BM, BK])
         b = bounded_load(B, [k0, n0], valid=[vk, vn], capacity=[BK, BN])
         wait_for_loads()
         acc = mma(a, b, acc, valid_k=vk)
       result = output_conversion(acc)
       clipped_store(out, [m0, n0], result, valid=[vm, vn])

SwiGLU shares the X tile across two weight loads and two accumulators. Both
accumulators start at zero; after the K loop finishes, the output tile evaluates
``silu(gate) * up`` with the source math mode and output conversion. Applying the
epilogue once after the full reduction is essential: applying SiLU per K tile
would change the function being computed. See
:source:`MicroKernelDriver.cpp <lib/Conversion/LLKToMicro/MicroKernelDriver.cpp>`.

``micro.tile_load`` copies only its valid rectangle into a fixed-capacity tile
and fills the remaining lanes with typed padding. ``micro.tile_store`` writes
only the valid destination rectangle. ``micro.mma valid_k`` reduces exactly
the valid K elements: it does not execute a full padded BK reduction. This
preserves the source's initialization, conversion and math-mode semantics,
including the strict signed-zero/nonfinite cases covered by acceptance tests.

For example, with ``BM=8, BN=32, BK=32`` and ``M=9, N=33, K=35``, the grid is
2×2 output tiles with two K iterations each. The last output tile has
``validM=1, validN=1`` and the last reduction tile has ``validK=3``. Its local
capacities remain 8×32 for A, 32×32 for B and 8×32 for the accumulator. Only the
valid external rectangles are read/written, and only three K terms are reduced
in the tail. Padding makes local tiles well-defined; it does not add semantic
reduction terms. Under strict arithmetic, extra padded ``+0`` contributions can
even change an accumulated ``-0`` to ``+0``.

M=0 or N=0 produces an empty output without compute or movement. K=0 still
performs initialization and stores the specified result: matmul preserves its
initializer and SwiGLU evaluates its epilogue from zero projections. Null data
pointers are allowed for genuinely empty buffers after descriptor and shape
validation.

.. _dynamic-shapes-storage-algorithm:

Certify storage independently of total work
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

External tensor dimensions may be dynamic, but local tiles and maximum
concurrent residency must be bounded at compilation. Complete-plan evaluation
certifies materialized storage, lifetimes and selected capacity before retaining
an executable candidate. A large external output is not a whole-output SRAM
allocation. Unbounded dynamic local allocation is rejected. An unknown
divisibility predicate requires a runtime guard or bounded movement; it is not
evidence that a full tile is legal.

The storage planner separates **execution multiplicity** (how many times an
operation eventually runs) from **simultaneous residency** (how many tile
instances may be live together). A long serial K loop increases work without
making every iteration's storage live at once. Spatial owners and pipeline
overlap can increase residency and must be included.

For a complete selected plan, certification:

#. Resolves concrete placements and memory routes, including intermediate
   movement buffers, then computes each local allocation's physical footprint
   from its static tile shape, element size and selected layout.
#. Bounds simultaneous occurrences from recovered structural facts. When the
   dynamic spatial grid is not a static count, the selected executor pool's
   concurrency supplies the bound; the logical grid size is not a concurrency
   count. Strict planning rejects a missing residency bound.
#. Accounts for allocation lifetimes, per-hop staging and required reuse
   ordering, and checks peak live bytes against the selected memories' capacities
   before executable candidate retention. Borrowed external tensors do not
   become local scratch allocations.

For one allocation the contribution is its physical bytes times its resident
occurrences; the peak also depends on which lifetimes overlap. This is why neither
``M*N*sizeof(dtype)`` nor the sum of all sequential tile iterations is the local
capacity requirement. Larger runtime tensors can keep the same bounded local
plan, while increasing pipeline depth or owner concurrency can make that plan
illegal. See :source:`StoragePlan.cpp <lib/Mapping/StoragePlan.cpp>` and the
:source:`dynamic mapping tests <test/Mapping/dynamic_shape_mapping.cpp>`.

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

.. _dynamic-shapes-sampling-algorithm:

How sampled ranking works
~~~~~~~~~~~~~~~~~~~~~~~~~

The evaluator keeps the executable's symbolic source and the candidate's fixed
mapping decisions, then evaluates each profile observation separately:

#. Validate the sample's complete symbol binding and clone the source module.
   Replace dimension queries with the sample values and canonicalize the clone
   so shape guards, ceil-div tile counts and boundary sizes become concrete.
#. Extract the sampled workload graph, bind/materialize the same selected plan,
   and verify target operation coverage. Run shared selected-kernel analysis
   with completeness required, including the plan's storage liveness and
   capacity checks. Reject the candidate if any sample fails.
#. Retain per-sample costs and combine comparable cost fields with the weighted
   mean. For latency this is ``sum(weight[i] * cycles[i]) / sum(weight[i])``;
   byte-count means are rounded to integer counters. Weights need not sum to one.

The profile shipped here has weights 1, 2 and 1, so the middle observation
contributes half the latency score. That score is not worst-case latency and is
not one physical execution trace. A single-sample plan may attach its event
snapshot; a multi-sample plan omits that snapshot because no individual stream
represents the weighted mean. Per-sample results remain available.

This establishes a ranking for the declared observations, not numerical
correctness or a performance guarantee at unobserved shapes. Domain legality
still relies on symbolic guards, bounded movement and storage certification;
actual timings require verified invocation and measurement. The implementation
is :source:`ShapeSpecialization.cpp <lib/Conversion/MicroMapping/ShapeSpecialization.cpp>`
and :source:`CompletePlanEvaluation.cpp <lib/Conversion/MicroMapping/CompletePlanEvaluation.cpp>`.

.. _dynamic-shapes-analysis-walkthrough:

Run the sampling workflow
~~~~~~~~~~~~~~~~~~~~~~~~~

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

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Area
     - Current limit
     - Practical consequence
   * - Rank and shape relationships
     - Rank is compile-time known; dimensions are constants or symbols with
       equality, range and multiple-of constraints
     - Dynamic rank, ragged/data-dependent extents and arbitrary symbolic
       relations such as ``N = 2*M`` are not supplied by this shape contract
   * - Callable buffers
     - Rank two, compact row-major, zero offset, at most 12 descriptors;
       per-port dtype fixed by the compiled ABI
     - Transposed/strided views, nonzero-offset descriptors and runtime dtype
       changes are rejected; a new shape does not change the buffer format
   * - Ownership and effects
     - Read-only inputs and disjoint caller-owned outputs; v2 requires typed
       descriptors with allocation sizes
     - Allocate the correctly sized output before invocation. In-place aliases
       and untagged v2 invocation are not supported
   * - Local storage and target boundaries
     - Fixed tile footprints, provable simultaneous residency and emitters that
       accept bounded-load/store/partial-K requirements
     - An unbounded local allocation, missing residency proof, capacity overflow
       or unsupported boundary realization prevents executable retention
   * - Semantic import and frontend
     - Dynamic LLK export covers matmul/SwiGLU; the semantic elementwise adapter
       remains static-only
     - General Linalg import and native Python frontend execution are separate
       capabilities (see :doc:`python-frontend`)
   * - Target execution
     - Reference and selected-target modes are distinct; the demonstrated
       dynamic selected execution path is AVX2
     - Existing Arm executable subsets and generic-accelerator analysis do not
       establish dynamic contraction execution on those targets
   * - Numerical semantics
     - Accumulator type, initializer, exact K extent, math mode and output
       conversion points remain source semantics
     - Dynamic shapes do not authorize extra padded reduction terms, moving an
       epilogue into the K loop or silently relaxing strict math
   * - Performance evidence
     - Explicit samples and complete selected-plan analysis are required for
       ranking; standalone incomplete reports can still contain advisory costs
     - Do not interpret ``complete: false`` as a usable ranking or a weighted
       model score as measured/worst-case latency
   * - Specialization and reuse
     - Guarded variants have validated ordering and a budget, while tuning
       currently selects the baseline and retains sampled report provenance
     - No automatic specialization dispatcher or guaranteed cross-profile
       compiled-cache hit; never reuse incompatible measurements or packed weights

When a workload is refused, first identify the failing boundary. A dimension
conflict is a semantic/descriptor error; a capacity or boundary-emitter failure
is plan legality; missing samples/resources are analysis completeness; an
incompatible ISA is host eligibility. Changing M/N/K flags or lowering to a
reference backend does not repair an invalid shape or descriptor.
