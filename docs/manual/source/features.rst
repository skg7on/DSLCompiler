.. _features-features-and-current-limits:

Features and current limits
===========================

The project goal is a generic MicroIR-based, mapping- and search-based DNN compiler
framework. Matmul and fused SwiGLU are example kernels used to validate reusable
contracts. MicroIR provides a common representation for scheduled tensor kernels,
declarative target mapping and performance exploration. The established CPU compiler remains
available while the newer mapped path is being strengthened. Use this page to
choose a workflow and understand what a successful run proves.

This summary reflects merged PR #133 (2026-10-11, merge ``5ee61b1``), the
generic framework increments A–D, and the bounded Arm SVE/SME static mapping
increment E1–E7. The revision-pinned
:source:`framework conformance review <docs/reviews/2026-10-12-framework-conformance.md>`
records capability-based acceptance, exact test counts and retained limits.
The :source:`Arm static acceptance ledger <docs/reviews/arm-mapping-targets-acceptance.md>`
records the target-specific evidence and its exact accepted domain. The
:source:`PR #133 acceptance matrix <docs/reviews/issue67-final-acceptance.md>`
records the earlier resource/runtime/tuner repairs. The October 7 gap assessment
is historical.

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
     - :doc:`Machine model </models/machine-model>`, :doc:`performance tutorial </tutorials/03-performance>`
   * - Author target policy
     - LLKMap layouts, affine maps, finite constraints, port predicates, capability requirements, target bundles and emitter keys
     - Bounded solving can truncate or remain undecided; unsupported facts must be diagnosed
     - :doc:`Mapping policy </models/mapping-policy>`, :source:`formal syntax <docs/design/llkmap-syntax.md>`
   * - Choose a schedule
     - JSON schedule database keyed by operation and M bucket; concrete or search-space Micro export; YAML schedule records from the tuner
     - Selection matches operation and M bucket only; the schedule database does not configure the fixed legacy tiling pass
     - :doc:`Schedules </models/schedules>`, :doc:`optimizer </tools/optimizer>`
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
     - ``--source-adapter`` selects or infers the source contract; contraction M/N/K options remain supported only for the contraction adapter; record persistence and exact reuse are opt-in; calibration remains separate
     - :doc:`Tuner </tools/tuning>`, :doc:`tuning tutorial </tutorials/04-tuning>`
   * - Validate measured predictions
     - Verified raw-sample records, exact reproducible reuse, comparable-regime JSON/text error reports and disjoint full-identity holdout manifests
     - Whole-kernel timing remains separate from event costs; unknown time bases are excluded; no automatic fitting or calibrated accuracy threshold
     - :doc:`Validation </tools/validation>`, :doc:`Tuner </tools/tuning>`
   * - Search a non-contraction source
     - Versioned workload contract, source adapter registry, bounded rank-two f32 elementwise add/multiply with tiled spatial schedules
     - The bounded elementwise adapter is the first delivered non-contraction family; arbitrary Linalg import, broadcasting, reductions, dynamic shapes and approximation modes are rejected
     - :doc:`Tuner </tools/tuning>`, :doc:`Compiler </tools/compiler>`
   * - Report stage-specific capabilities
     - Capability receipts for structure, interpretation, mapping, physical plan, reference lowering, selected lowering, ABI eligibility, preparation, host compatibility and invocation, keyed by source/contract/target/plan identity
     - ``supported``, ``unsupported`` and ``not_checked`` are distinct; printing a static receipt never runs lowering or JIT; a reference-backend run leaves selected lowering ``not_checked``
     - :doc:`Architecture </architecture>`, :source:`conformance review <docs/reviews/2026-10-12-framework-conformance.md>`
   * - Validate composed primitives and reduction
     - Direct 8x8→8 sum and add→multiply; selected AVX2 Vector reduction/vector IR; operation-aware target coverage before ranking
     - Rank-one invocation is outside mapped ABI v1; generic-accelerator mul/sum are analysis-only; Linalg reduction import is not implemented
     - :doc:`Performance analyzer </tools/performance>`, conformance review
   * - Extend a target without generic dispatch
     - Renamed-target portability, literal topology variants and a generic-library boundary scan
     - A target still needs its own machine/layout/rule/emitter package; no physical accelerator backend is delivered
     - :source:`conformance review <docs/reviews/2026-10-12-framework-conformance.md>`
   * - Run supported mapped kernels
     - Shared mapped compilation, typed descriptor invocation, caller-buffer ownership and per-call scratch release; reference and selected AVX2 numeric fixtures including generic elementwise add/multiply
     - Selected-target execution requires x86_64 AVX2; rank-two supported dtypes/strides/bounds are checked by typed invocation; analysis/lowering are rank-neutral but executable ABI v1 rejects other ranks with ``runtime_capability:`` diagnostics; deprecated untagged descriptors cannot check dtype/allocation size
     - :doc:`Compiler </tools/compiler>`
   * - Exercise the benchmark harness
     - ``llk-bench`` runs a synthetic host loop and reports timing
     - It does not parse MLIR, load weights, invoke a compiled kernel or use the JIT; thread-pool setup occurs per invocation
     - :doc:`Benchmark </tools/benchmark>`
   * - Install and inspect the Python frontend
     - ``llk-compiler`` packaging, optional Torch imports, stable Dynamo registration, structured errors and explicit capability reporting (PF01)
     - Backend calls report ``native_backend_unavailable``; native Python JIT, Inductor and upstream Triton execution remain later milestones
     - :doc:`Python frontend </python-frontend>`
   * - Explore a second target
     - Generic accelerator topology, layouts, rules and mapping through the same canonical IR; static f32 add/mul/sum coverage with operation-aware validation
     - No generic accelerator hardware backend or directly interpreting MicroIR emulator is delivered; accelerator mul/sum emitters verify only
     - :doc:`Mapping tutorial </tutorials/02-mapping>`
   * - Map Arm SVE and SME statically
     - Both target names load a bounded machine profile and typed LLKMap policy, then map direct-MicroIR fixtures: f32/bf16 M=N=K=16 contraction, typed f32/bf16/i8 add and multiply, both bf16↔f32 conversions, f32 SiLU and reduce sum, and the fused convert→SiLU→multiply graph. Selected plans report concrete vector or matrix engine and accumulator identities, verify, replay from the reported plan id and analyze
     - Static mapping and analysis only, at modeled VL=SVL=512: no executable SVE/SME, physical ZA realization, calibrated timing or automatic retargeting of CPU schedules. Emitters refuse lowering, and selected-target compilation is rejected
     - :source:`Acceptance ledger <docs/reviews/arm-mapping-targets-acceptance.md>`, :doc:`Optimizer </tools/optimizer>`

The broader goal does not imply arbitrary DNN support today. The LLK-to-Micro
export and contraction source family remain contraction-shaped, while source
search also admits the bounded rank-two elementwise family. Typed mapped
invocation is still rank-two: contracts may describe other ranks, and analysis
and lowering remain rank-neutral, but executable preparation refuses them. The
:source:`framework design spec <docs/superpowers/specs/2026-10-12-generic-microir-dnn-compiler-framework-design.md>`
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

Analysis completeness and ranked plans
--------------------------------------

Ranked mapping and mapped-tuning results require complete audited analysis
before top-K retention. Unknown computational leaves, unresolved physical facts
and selected-analysis failures reject a proposal with diagnostics. Complete
analysis-only targets can still rank modeled plans without an executable
backend; executable binding and selected lowering remain separate requirements.

The integration API defaults to ``AnalysisRequirement::Complete``. Explicit
``PartialReport`` with partial binding returns an inspectable artifact with
ordered warnings and ``materialized=false``; such artifacts cannot enter ranked
results, even through a custom completion callback. Only this report mode may
fall back to synthesized-event estimates after extraction fails. Capacity and
whole-plan byte budgets still apply. Executable binding always requires complete
analysis, regardless of the requested report mode.

``micro-perf`` uses audited extraction and reports ``complete`` and
``incomplete_reasons`` in YAML (and the corresponding text output). Direct
MicroDAG users may choose ``Audit`` for annotated partial results or ``Strict``
for rejection. Existing custom-builder overloads retain ``LegacyPermissive``
behavior. Registered alias-only or explicit zero-cost handlers remain supported;
an unknown effect-free arithmetic operation is not automatically zero cost.
See the :source:`mapping workflow <docs/design/micro-ir-mapping-workflow.md>`
for the integration boundary and supported modes.

Capability receipts and stage evidence
--------------------------------------

A known emitter key and a complete static mapping do not prove that a target
lowers or runs a kernel. ``CapabilityReceipt`` v2 reports one verdict per stage:
structure, interpretation, mapping, physical plan, reference lowering, selected
lowering, ABI eligibility, preparation, host compatibility and invocation.
Each is ``supported`` (established for this artifact),
``unsupported`` (a concrete blocker was found) or ``not_checked`` (the stage was
never attempted). A receipt is keyed by the source graph content, canonical
workload contract, target content and plan id.

Preparation and host compatibility require an actual executable; invocation
requires a call that returned successfully. Neither preparation nor a successful
call establishes numerical correctness.

Receipts never run a backend lowering or JIT merely to print. A target without a
codegen contract reports selected lowering unsupported; a selected-backend
``Lowered`` result is supported only with realized groups and zero reference
fallback; a reference-backend run leaves selected lowering ``not_checked``
rather than misreporting the deliberate mode as a failure. ``llk-tune`` mapping
reports carry a per-candidate ``capabilities`` object, and ``llk-compile`` prints
a ``capabilities:`` JSON line on stderr so stdout remains a re-parsable
artifact.

Operation coverage complements the receipt: before complete-plan analysis and
scoring, the bound IR is grouped by selected instance and each emitter validates
the covered operation's variant and operand/result types without rewriting the
IR. A mismatch is a candidate rejection, not a late lowering error. This is how
the analysis-only accelerator accepts f32 add/multiply/sum while refusing bf16.

Target names are provenance, not policy. Renaming a target preserves legal
plans, analysis verdicts and its base backend capability while changing the
receipt's ``targetIdentity``; changing machine resource or capacity data changes
the legal covering set and rejection reasons through the same generic code. The
conformance review records these checks and the library link boundary.

.. _features-work-planned-separately:

Work planned separately
-----------------------

PR #133 completes the #129 resource, runtime, mapped tuning and acceptance
repair series within the supported bounded contract. Phase D adds opt-in verified
measurement persistence and exact reuse, plus held-out prediction-error reports;
see :doc:`tools/tuning` and :doc:`tools/validation`. Calibration fitting and
independently measured event attribution remain separate work.
The generic accelerator package has static topology, movement and f32
add/mul/sum conformance, not hardware execution. A direct MicroIR emulator and a
broad Python/model execution frontend remain deferred. PF01's Python package and
registration foundation is available separately; see :doc:`python-frontend` for
its tested environment and explicit execution boundary.

Selected AVX2 acceptance for PR #133 passed 13/13 without required skips. The
current framework review records the later increments and their exact local and
CI counts; required selected execution remains a Linux ``x86_64`` AVX2 gate.
Legacy JIT skips do not establish numerical legacy execution and are not host
AVX2 capability skips. See the
:source:`framework conformance review <docs/reviews/2026-10-12-framework-conformance.md>`
for current evidence and the older
:source:`acceptance matrix <docs/reviews/issue67-final-acceptance.md>` for the
#133 diagnostics and revision.

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
