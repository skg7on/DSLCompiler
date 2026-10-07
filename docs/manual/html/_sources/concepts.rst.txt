.. _concepts-microir-concepts:

MicroIR concepts
================

MicroIR describes how a tensor kernel executes: which tiles it computes, where
those tiles live, which resources own the work, and when computation and movement
occur. The project uses the ``micro`` MLIR dialect as its canonical execution IR.
Its purpose is to make schedules inspectable and machine models usable for
mapping and performance exploration.

Read this guide before the :doc:`architecture </architecture>`, or follow the
:doc:`first-kernel tutorial </tutorials/01-first-kernel>` alongside it. The
:doc:`feature guide </features>` distinguishes the current implementation from the
larger design.

.. _concepts-semantics-schedule-and-target:

Semantics, schedule, and target
-------------------------------

These are three different decisions:

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Question
     - Representation
     - Example
   * - What is the mathematical operation?
     - LLK and structured MLIR
     - ``Y = SiLU(X · Wg) * (X · Wu)``
   * - How is the work divided and ordered?
     - Schedule data and concrete MicroIR
     - Output tiles, reduction tiles, staging copies, pipeline stages
   * - Where can that schedule run?
     - MachineModel, LLKMap, target package
     - Worker hierarchy, vector width, legal layouts, transfer links, emitter

The small ``llk`` dialect retains domain information such as fused SwiGLU, rotary
position embedding, and attention. Linalg expresses structured tensor
computation and supports tiling, fusion, and vectorization. MicroIR expresses a
scheduled execution program. A concrete ``micro.kernel`` should contain compute,
movement, and scheduling operations rather than an opaque DNN semantic operator.

For example, the SwiGLU formula alone says nothing about whether two projections
share an input tile, whether their accumulators fit a local memory, or whether a
copy can overlap computation. Those are schedule questions. Mapping then decides
which target rules implement the work and how producers connect to consumers.

The current ``LLKToMicro`` exporter recognizes LLK matmul and fused SwiGLU roots.
RoPE and attention have the established CPU path, but they are not general MicroIR
export roots today. Conversely, the legacy ``--llk-to-linalg`` pass leaves
``llk.matmul`` untouched; its supported compiler route is Micro export and the
mapped path. See :source:`LLKToMicro.cpp <lib/Conversion/LLKToMicro/LLKToMicro.cpp>`
and :source:`LLKOps.td <include/LLK/Dialect/LLKOps.td>`.

.. _concepts-the-tile-is-the-execution-value:

The tile is the execution value
-------------------------------

Think of a tile as:

.. code-block:: text

   Tile = (shape, element type, layout, memory space, owner)

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Property
     - Meaning
     - Why it matters
   * - Shape
     - The logical region being processed
     - Work volume, transfer bytes, fragment compatibility, tails
   * - Element type
     - Input, output, or accumulator format
     - Storage size and legal compute capability
   * - Layout
     - How logical indices correspond to physical storage
     - Contiguous vectors, blocking, alignment, transforms
   * - Memory space
     - An abstract storage level
     - Capacity, visibility, movement and bandwidth
   * - Owner
     - A resource class or target-resolved symbol
     - Placement, execution hierarchy and concurrency

The dialect represents this with ``!micro.tile`` and attributes such as
``#micro.layout``, ``#micro.memory``, and ``#micro.owner``. A tile is more informative
than a tensor shape alone: two ``8x8xf32`` values can require different movement
and implementation choices if their layouts or storage levels differ.

Four conceptual tile roles help explain cost:

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Role
     - Example
     - Cost interpretation
   * - Logical tile
     - A view or partition of a larger value
     - The index relation itself does not copy bytes
   * - Memory tile
     - An allocated or copied tile
     - Storage and movement must be accounted for
   * - Execution tile
     - Work associated with an owner
     - Compute work and available parallelism matter
   * - Instruction fragment
     - A matrix fragment or vector-width chunk
     - The target capability constrains granularity

These are roles in the programming model, not four separate MLIR types. A
logical view can later require materialization to satisfy a consumer. An
instruction fragment is smaller than, or otherwise compatible with, the
execution tile that contains it. Do not equate tile size with SIMD width.

The implemented logical operations include ``micro.tile_view`` and
``micro.tile_partition``; materialized operations include ``micro.tile_alloc``,
``micro.tile_async_copy``, and ``micro.tile_store``. Compute uses ``micro.mma``,
``micro.vector``, and ``micro.reduce``. The definitive operation set is in
:source:`MicroOps.td <include/LLK/Dialect/Micro/MicroOps.td>`, with checks in
:source:`MicroOps.cpp <lib/Dialect/Micro/MicroOps.cpp>`.

.. _concepts-three-conceptual-layers-one-dialect:

Three conceptual layers, one dialect
------------------------------------

The design uses the names **uTile**, **uMap**, and **uHW** to describe increasing
execution detail:

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Layer
     - Responsibility
     - Typical facts
   * - uTile
     - Tile dataflow
     - MMA inputs, elementwise dependencies, reductions, partitions
   * - uMap
     - Placement and schedule
     - Memory, owner, copies, pipeline, synchronization
   * - uHW
     - Machine-visible granularity
     - Fragments, vector widths, transfer events, resource costs

They share the ``micro`` namespace. There are no separate ``uTile``, ``uMap``, or ``uHW``
dialects, and users do not run a fixed three-pass pipeline with those names.
The labels explain which decisions a representation contains. Target lowering
eventually realizes selected facts in backend IR or code; the current mapped
backend still has realization gaps described in :doc:`features </features>`.

.. _concepts-time-space-and-dependencies:

Time, space, and dependencies
-----------------------------

``micro.for`` expresses temporal iteration: work repeats over time.
``micro.spatial_for`` expresses spatial decomposition: iterations are associated
with a mapping axis. ``micro.pipeline`` records a staging/overlap structure.
Copies return dependency tokens that ``micro.wait`` consumes; ``micro.barrier``
expresses synchronization between owners.

For tiled GEMM, a spatial decomposition can distribute output tiles while a
temporal reduction loop advances through K. A pipeline can stage the next input
tiles while a compute operation consumes the current ones. A legal dependence
graph permits overlap; actual overlap depends on resource availability and the
backend. Merely writing a pipeline or async copy does not demonstrate hardware
concurrency.

The performance analyzer recovers structural work and dependencies from these
operations. Mapping also extracts a workload graph, but the planner and bound
kernel analyzer do not yet agree on all structural multiplicity and resource
costs. Their estimates must be compared explicitly.

.. _concepts-search-ir-and-concrete-ir:

Search IR and concrete IR
-------------------------

MicroIR has two forms with different consumers:

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Form
     - Main operations
     - Use
   * - Search
     - ``micro.search_space``, ``micro.param``, ``micro.constraint``, ``micro.objective``, ``micro.candidate``
     - Declare choices, legality rules and an optimization objective
   * - Concrete
     - ``micro.kernel`` plus tile, compute, movement and structure operations
     - Represent a particular schedule for analysis, mapping and supported lowering

A search space can offer choices for ``BM``, ``BN``, ``BK``, vector width, pipeline
stages, owner mapping, memory path, fragment shape, or layout. A candidate binds
parameters. A constraint rules out incompatible choices, such as excess staging
storage or a fragment that does not fit the machine. An objective describes how
legal candidates are ranked, commonly by latency cycles.

Search IR is not executable. Binding parameters is also distinct from selecting
a target mapping plan: a tile size can be fixed before the compiler decides which
executor, layout rule, or transfer route to use. The current tuner generates,
checks, binds and ranks synthetic GEMM/SwiGLU kernels; integration that maps and
compiles every original-source candidate is planned under issue #129. See
:doc:`tuning </tools/tuning>` for the supported workflow.

.. _concepts-a-kernels-signature-is-its-contract:

A kernel's signature is its contract
------------------------------------

An executable ``micro.kernel`` declares inputs and results explicitly. Its entry
block arguments are inputs, and ``micro.yield`` identifies results. A yielded tile
can satisfy a tensor result of the same logical shape and element type; placement
attributes are schedule facts rather than part of that logical result contract.

The legacy argumentless syntax still parses for analysis and round-tripping.
It does not supply an execution ABI. A lowerer must not guess that an internal
``tensor.empty`` is a caller buffer, or infer outputs from whichever stores happen
to appear in the body. The :source:`kernel definition <include/LLK/Dialect/Micro/MicroOps.td>`
and :source:`kernel contract tests <test/Dialect/Micro/ops.mlir>`
show the explicit form.

The logical signature should also be distinguished from the runtime interface.
The mapped runtime uses buffer descriptor pointers; the established CPU runtime
uses its existing C-struct wrapper. The mapped descriptor and buffer ownership
contracts still need repairs before broad repeated-invocation use.

.. _concepts-abstract-symbols-and-concrete-machine-nodes:

Abstract symbols and concrete machine nodes
-------------------------------------------

Memory spaces such as ``dram``, ``sram``, and ``acc`` identify abstract levels. A
MachineModel provides concrete nodes such as ``dram.0`` or ``sram.0``, with capacities,
visibility, alignment, and links. Several concrete nodes may have the same kind.
Selecting a memory node must therefore preserve its identity, not just its kind.

Owners and spatial map axes are open symbols. The canonical classes are ``group``,
``worker``, ``vector``, ``matrix``, and ``transfer``; target vocabulary can refine those
classes. For example, a target profile can declare that its ``pe`` is a ``worker`` or
its ``vector_engine`` is a ``vector``. The dialect accepts the symbol, while target
resolution determines whether execution can use it. An unknown spelling can
remain analysis data; parse success does not prove a valid target placement.

See the shipped :source:`AVX2 profile <machines/x86-avx2-v2.yaml>` and
:source:`generic accelerator profile <machines/generic-ai-accel-v2.yaml>`. Both use the
same topology schema and canonical IR. The latter is an architecture exploration
target rather than an accelerator hardware backend.

.. _concepts-layouts-and-movement-are-separate-decisions:

Layouts and movement are separate decisions
-------------------------------------------

A canonical ``#micro.layout`` describes a layout kind and its parameters. LLKMap
defines target-owned layout IDs, finite parameter domains, legality constraints,
and logical-to-physical affine maps. A target ID such as ``avx2.blocked_2d`` is not a
new canonical layout enum. An ``implements`` clause can connect a target layout to
a canonical kind.

Producer and consumer ports may require:

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Connection
     - Meaning
   * - Direct
     - Existing placement and layout are compatible
   * - Transfer
     - Bytes move across one or more machine links
   * - Layout transform
     - A value is re-represented in another layout
   * - Transfer and transform
     - Both movement and re-representation are required

``micro.transform`` names source and destination affine maps rather than a target
layout ID. It represents work, so it is not a free type cast. Likewise, a
multi-hop route requires accounting for every link and intermediate storage;
the existence of a route alone does not establish a feasible storage lifetime.
The current implementation can emit routed movement and transforms, with known
physical storage limitations listed in :doc:`features </features>`.

.. _concepts-a-prediction-is-evidence-for-a-decision:

A prediction is evidence for a decision
---------------------------------------

Three claims should be evaluated independently:

#. **Legal representation:** the IR and selected metadata pass their applicable
   structural, target and semantic checks.

#. **Correct execution:** a supported backend invokes the kernel and matches a
   numerical reference on stated inputs.

#. **Accurate prediction:** modeled costs agree with measured behavior for a
   specified machine and workload.

The project has useful IR, mapping, performance analysis, and portable numerical
execution today. Complete selected target realization, static planner/analyzer
parity, and calibrated prediction are separate unfinished contracts. Shipped
profile costs are seeds for modeling, not measured claims about a CPU SKU.

Continue with :doc:`architecture </architecture>`, the
:doc:`mapping tutorial </tutorials/02-mapping>`, or the detailed
:source:`tile programming model <docs/superpowers/specs/2026-08-13-micro-ir-tile-programming-model-spec.md>`.
