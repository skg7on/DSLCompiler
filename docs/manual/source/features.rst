.. _features-features-and-current-limits:

Features and current limits
===========================

The project goal is a generic MicroIR-based, mapping- and search-based DNN compiler
framework. Matmul and fused SwiGLU are example kernels used to validate reusable
contracts. MicroIR provides a common representation for scheduled tensor kernels,
declarative target mapping and performance exploration. The established CPU compiler remains
available while the newer mapped path is being strengthened. Use this page to
choose a workflow and understand what a successful run proves.

This summary reflects merged PR #133 (2026-10-11, merge ``5ee61b1``).
The :source:`acceptance matrix <docs/reviews/issue67-final-acceptance.md>` records
revision-pinned tests, execution artifacts and retained limitations. The October 7
gap assessment is historical; its resource/runtime/tuner repairs are implemented.

.. _features-what-you-can-do-today:

What you can do today
---------------------

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Capability
     - Available behavior
     - Important limit
     - Guide
   * - Write and inspect MicroIR
     - Tile types; kernel signatures; temporal/spatial loops; pipelines; MMA/vector/reduction; movement, transforms, waits and barriers; search-space operations
     - Syntax and structural verification do not establish a legal target mapping or executable backend support
     - :doc:`Concepts </concepts>`, :doc:`first kernel </tutorials/01-first-kernel>`
   * - Compile the established CPU path
     - LLK-to-Linalg lowering for supported semantic operations, scheduling, tiling/fusion, explicit Vector IR, bufferization and LLVM/JIT infrastructure
     - ``llk.matmul`` is left untouched by the legacy conversion and uses the Micro route; kernel, shape, dtype, math-mode and host support remain specific
     - :doc:`Compiler </tools/compiler>`
   * - Export supported semantic kernels to MicroIR
     - Concrete kernel and search-space export for LLK matmul and fused SwiGLU
     - General Linalg, RoPE and attention export are not implemented by ``LLKToMicro``
     - :doc:`Compiler </tools/compiler>`
   * - Describe a machine
     - Versioned YAML topology with executor, memory, compute, transfer nodes and links; owner refinements and model validation
     - Shipped numerical costs are modeling seeds; a profile is not a hardware execution backend
     - :doc:`Architecture </architecture>`, :doc:`performance tutorial </tutorials/03-performance>`
   * - Author target policy
     - LLKMap layouts, affine maps, finite constraints, port predicates, capability requirements, target bundles and emitter keys
     - Bounded solving can truncate or remain undecided; unsupported facts must be diagnosed
     - :source:`Formal syntax <docs/design/llkmap-syntax.md>`
   * - Match single and fused operations
     - Single-operation rules and bounded fused-group matching
     - Selected fused groups bind, lower and execute in the verified SwiGLU chain; unsupported effects/regions or emitters are rejected
     - :doc:`Mapping tutorial </tutorials/02-mapping>`
   * - Search mappings
     - Deterministic, beam and exact covering modes; placement, route and connection alternatives; search bounds and diagnostics
     - Complete storage/materialized analysis runs before top-K retention; beam mode has no optimality guarantee and exact search remains bounded
     - :doc:`Optimizer </tools/optimizer>`
   * - Bind, report and replay selections
     - Versioned plan reports and selected metadata; canonical movement/transform/sync materialization; layered verification
     - Concrete compute/memory/DMA identities and per-hop storage/lifetimes are retained and verified; unsupported execution-affecting decisions are rejected
     - :doc:`Optimizer </tools/optimizer>`, :doc:`architecture </architecture>`
   * - Estimate performance
     - Level-0 work/bandwidth bounds and level-1 dependency/resource scheduling; cycles, traffic, utilization, bottleneck and capacity reporting
     - Selected-plan and materialized static analysis share resource/event accounting with parity controls; predictions remain uncalibrated
     - :doc:`Performance analyzer </tools/performance>`
   * - Tune schedule parameters
     - Original-source candidate instantiation, mapped selection/analysis/compilation, static ranking, candidate artifacts, frozen replay and optional verified measurement
     - Requires explicit mapping target/source mode for original-source tuning; production measurement persistence/calibration remain separate
     - :doc:`Tuner </tools/tuning>`, :doc:`tuning tutorial </tutorials/04-tuning>`
   * - Run supported mapped kernels
     - Shared mapped compilation, typed descriptor invocation, caller-buffer ownership and per-call scratch release; selected AVX2 numeric fixtures
     - Selected-target execution requires x86_64 AVX2; rank-two supported dtypes/strides/bounds are checked by typed invocation; deprecated untagged descriptors cannot check dtype/allocation size
     - :doc:`Compiler </tools/compiler>`
   * - Exercise the benchmark harness
     - ``llk-bench`` runs a synthetic host loop and reports timing
     - It does not parse MLIR, load weights, invoke a compiled kernel or use the JIT; thread-pool setup occurs per invocation
     - :doc:`Benchmark </tools/benchmark>`
   * - Explore a second target
     - Generic accelerator topology, layouts, rules and mapping through the same canonical IR
     - No generic accelerator hardware backend or directly interpreting MicroIR emulator is delivered
     - :doc:`Mapping tutorial </tutorials/02-mapping>`

The broader goal does not imply arbitrary DNN support today. Semantic export and
source-search templates remain contraction-shaped, and typed mapped invocation
is rank-two. The :source:`framework design spec <docs/superpowers/specs/2026-10-12-generic-microir-dnn-compiler-framework-design.md>`
records these genericity gaps and capability-based acceptance requirements.

The implemented operation inventory is
:source:`MicroOps.td <include/LLK/Dialect/Micro/MicroOps.td>`. Broad operation lists in
design specifications describe direction as well as delivered features; they
should not be used as an executable support matrix.

.. _features-choose-a-starting-point:

Choose a starting point
-----------------------

**Learn the IR:** parse a small kernel with ``llk-opt``, inspect its tile values,
and follow :doc:`tutorial 1 </tutorials/01-first-kernel>`. This is the fastest way to
understand the model without needing a semantic frontend or JIT.

**Explore a mapping:** use a small concrete kernel, the shipped AVX2 package,
and a versioned plan report. Follow :doc:`tutorial 2 </tutorials/02-mapping>`, inspect
the selected rules and routes, then compare the bound IR with the source. A
matching plan ID is evidence of selection consistency, not proof of complete
physical feasibility or target code generation.

**Explore a machine:** change a profile in an isolated worktree and compare
analyzer reports for the same kernel. Follow :doc:`tutorial 3 </tutorials/03-performance>`.
State which profile and analysis level produced the result, and keep modeled
cycles separate from measured time.

**Explore schedules:** export or load a search space, cap enumeration while
experimenting, inspect rejected candidates, and follow
:doc:`tutorial 4 </tutorials/04-tuning>`. Use explicit target-aware source tuning for replayable selected plans; compatibility tuning remains a separate synthetic mode. Static ranking does not prove the fastest measured executable.

**Explore the timing harness:** read the :doc:`benchmark guide </tools/benchmark>`
before using ``llk-bench``. Its synthetic host loop measures the harness rather
than a compiled MLIR kernel, and its per-invocation thread-pool setup is part of
that timing. For actual supported kernel invocation, use the C++
:source:`mapped compilation API <include/LLK/Conversion/MappedCompilation.h>` and
:source:`MappedExecutable <include/LLK/Runtime/MappedExecutable.h>`, with
:source:`execution tests <test/Execution/mapped_acceptance.cpp>` as examples. Observe
the mapped runtime limits above and record host, inputs, schedule and timing
conditions; these APIs do not constitute the planned calibrated measurement loop.

.. _features-work-planned-separately:

Work planned separately
-----------------------

PR #133 completes the #129 resource, runtime, mapped tuning and acceptance
repair series within the supported bounded contract. Production measured-record
persistence, calibration fitting and prediction-error validation remain #51/#52.
The generic accelerator package has static two-hop conformance, not hardware
execution. A direct MicroIR emulator and broad Python/model frontend remain deferred.

Selected AVX2 acceptance passed 13/13 without required skips. The implementation
CI suite recorded 1,215 entries, 1,211 passed, three legacy JIT skips and one
pre-existing disabled test. The legacy skips do not establish numerical legacy
JIT execution and are not host AVX2 capability skips. See the acceptance matrix
for their diagnostics and exact revision.

.. _features-reading-status-evidence:

Reading status evidence
-----------------------

Historical milestone documents record substantial delivered work, but a merged
component or passing portable numerical test does not settle every end-to-end
contract. In particular:

* Parsing proves the representation can be read; target verification resolves
  selected facts.

* A materialized route proves copies were emitted; complete intermediate
  allocation lifetimes and occupancy need their own checks.

* A portable native-host JIT result proves numerical behavior for its inputs;
  selected AVX2 lowering needs explicit vector-width/ISA evidence.

* Shared event primitives do not by themselves establish planner/analyzer cost
  parity, and static parity does not establish calibrated hardware prediction.

For current verified scope and retained limits, read the
:source:`acceptance matrix <docs/reviews/issue67-final-acceptance.md>`.
To help close a gap or extend a capability, start with
:doc:`contributing </contributing>`.
