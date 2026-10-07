.. _tools-optimizer-llk-opt-inspect-transform-and-map-mlir:

``llk-opt``: inspect, transform, and map MLIR
=============================================

``llk-opt`` is DSLCompiler's ``mlir-opt`` driver. It parses and verifies MLIR, runs the passes you select, and prints the resulting module. It registers MLIR's built-in dialects and passes, the ``llk`` and ``micro`` dialects, and the project's lowering/mapping passes. It does not JIT-compile or invoke kernels.

Use the :doc:`first-kernel tutorial </tutorials/01-first-kernel>` for a guided export and the :doc:`mapping tutorial </tutorials/02-mapping>` for a complete mapping example. Commands below run from the repository root with tools in ``build/``; see the :doc:`documentation index </index>` and :doc:`compiler manual </tools/compiler>`.

.. _tools-optimizer-parse-print-and-select-passes:

Parse, print, and select passes
-------------------------------

.. code-block:: sh

   mkdir -p build/tutorial
   build/llk-opt docs/manual/source/examples/matmul.mlir
   build/llk-opt docs/manual/source/examples/matmul.mlir -o build/tutorial/matmul.parsed.mlir
   build/llk-opt --show-dialects
   build/llk-opt --help

With no passes, the tool checks the input and prints normalized MLIR. It reads stdin when no file is supplied or the input is ``-``. ``-o <file>`` saves IR; omitted output goes to stdout. Errors and remarks go to stderr, and failed parsing, verification, or passes return a nonzero exit status.

Pass-specific options belong inside one quoted argument, separated by spaces:

.. code-block:: sh

   build/llk-opt --llk-to-micro="schedule-db=schedules/schedule_db.json target=x86-avx2-cpu" \
     docs/manual/source/examples/matmul.mlir -o build/tutorial/matmul.micro.mlir

Do not separate pass option keys with commas. Commas are used **inside** values such as an emitter CSV, and between passes in textual pipelines. Multiple pass flags run in command-line order. The equivalent explicit pipeline syntax is:

.. code-block:: sh

   build/llk-opt --pass-pipeline='builtin.module(llk-to-linalg,canonicalize)' \
     docs/manual/source/examples/swiglu.mlir -o build/tutorial/swiglu.linalg.mlir

The tool also registers the Linalg Transform-dialect extension, including operations such as ``transform.structured.match``. Registration makes syntax and passes available; it does not apply a transform schedule automatically.

Useful inherited MLIR driver options:

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Option
     - Use
   * - ``--verify-each``
     - Verify IR between transformation passes; useful when assembling a new pipeline.
   * - ``--verify-roundtrip``
     - Verify that parsed IR can be printed and parsed again.
   * - ``--mlir-print-ir-before-all`` / ``--mlir-print-ir-after-all``
     - Inspect pass-boundary IR in diagnostics.
   * - ``--mlir-print-ir-after-failure``
     - Print the state left by a failing pass.
   * - ``--mlir-disable-threading``
     - Make diagnostic/IR debugging easier to follow.
   * - ``--split-input-file``
     - Process test chunks separated by the split marker independently.
   * - ``--verify-diagnostics``
     - Match ``expected-*`` annotations in diagnostic test files.
   * - ``--allow-unregistered-dialect``
     - Parse generic-form operations for staged frontend experiments; this does not supply their semantics or lowering.

The built-in pass list depends on the LLVM/MLIR revision used for the build. ``--help`` is the authority for that build's inherited options; the project-specific surface below is registered by :source:`llk-opt.cpp <tools/llk-opt/llk-opt.cpp>`.

.. _tools-optimizer-project-pass-reference:

Project pass reference
----------------------

.. _tools-optimizer-semantic-and-micro-conversion:

Semantic and Micro conversion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Pass
     - Behavior and options
   * - ``--llk-to-linalg``
     - Lower SwiGLU, RoPE, and Attention to Linalg/Arith/Math. No pass options. Currently leaves ``llk.matmul`` untouched.
   * - ``--llk-to-linalg-pipeline``
     - Registered pipeline alias containing **only** ``llk-to-linalg``; it is not the full compiler pipeline.
   * - ``--llk-to-micro``
     - Add concrete tile-centric kernels for ``llk.matmul`` and ``llk.fused_swiglu``. ``schedule-db`` defaults to ``schedules/schedule_db.json``; ``target`` defaults to the recorded label ``x86-avx2-cpu``.
   * - ``--llk-to-micro-search-space``
     - Add a search space around the same selected schedule. ``schedule-db`` defaults to ``schedules/schedule_db.json``. Has no ``target`` option.
   * - ``--micro-to-linalg``
     - Reference bridge: turn explicit-signature kernels and tile operations into tensor/Linalg/SCF functions. No pass options. Does not run target emitters, preserve selected SIMD width as explicit Vector ops, bufferize, or JIT-compile.

The export passes leave their semantic source functions intact. They support static rank-two matmul/SwiGLU roots and Micro tile dtypes ``f32``, ``f16``, ``bf16``, ``i32``, ``i8``; the accumulator attribute must name a supported dtype. SwiGLU requires SiLU. Shape mismatches, unsupported dynamic/rank types, or schedules the exporter cannot realize fail with diagnostics. A module with no supported root is unchanged. Read the :ref:`compiler export details <tools-compiler-export-a-concrete-kernel-and-a-search-space>` for the output contract.

To make experiments independent of the shipped database, point at an intentionally absent file:

.. code-block:: sh

   build/llk-opt --llk-to-micro="schedule-db=build/tutorial/absent-schedule.json" \
     docs/manual/source/examples/matmul.mlir -o build/tutorial/matmul.micro.mlir

   build/llk-opt --llk-to-micro-search-space="schedule-db=build/tutorial/absent-schedule.json" \
     docs/manual/source/examples/swiglu.mlir -o build/tutorial/swiglu.search.mlir

Both commands warn and use the conservative fallback. The matmul artifact contains ``@matmul_M16_N64_K64``, spatial loops, staged tile copies, an accumulator-carrying K loop, ``micro.mma``, and writeback. The search artifact describes parameters and legality/objective metadata; exporting it does not create a mapped executable.

.. _tools-optimizer-legacy-scheduling-and-runtime-passes:

Legacy scheduling and runtime passes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Pass
     - Current behavior
   * - ``--shape-specialize``
     - Report the M-bucket classification for SwiGLU. The current pass emits remarks rather than dispatch guards.
   * - ``--select-schedule``
     - Read ``schedules/schedule_db.json`` for supported LLK operations and emit a selected-schedule remark. Does not configure the fixed constants in ``TileAndVectorize``.
   * - ``--fuse-double-contraction``
     - Recognize the SwiGLU-style pair of matmuls sharing X and a SiLU consumer.
   * - ``--pack-weights``
     - Annotate weight operands with ``llk.packed_layout``; the default block value is 64. Does not by itself materialize a packed weight buffer.
   * - ``--tile-and-vectorize``
     - First lower supported LLK operations, then tile Linalg matmul/generic operations and emit Vector IR. Current fixed sizes: ``BM=32``, ``BN=64``, ``BK=64``, ``VM=4``, ``VN=8``.
   * - ``--linearize-forall``
     - Linearize multidimensional ``scf.forall`` for one-dimensional dispatch.
   * - ``--serial-parallel-dispatch``
     - Serialize statically small result-free foralls using the compile host's hardware-concurrency threshold; tensor-result foralls are left for later handling.
   * - ``--forall-to-llrt``
     - Lower foralls to the project's thread-pool/runtime dispatch path.
   * - ``--forall-to-openmp``
     - Prototype lowering via SCF parallel loops to OpenMP.
   * - ``--scratch-analysis``
     - Audit bufferized ``memref`` allocations for full-size intermediates.

``tile-and-vectorize`` exposes ``schedule-file`` (default empty) and ``target-isa`` (default ``avx2``, help lists ``avx512``/``sve``). **Those options are currently inert:** the implementation does not read the file or ISA value and uses the fixed constants above. Do not use their names as evidence that a Transform file ran or that another ISA was selected. The Micro export's ``schedule-db`` is an active option with a different implementation.

The shared schedule selector currently matches operation and M bucket, then chooses the first entry; exact N/K, target, dtype, and math-mode fields do not participate in selection. Buckets are ``{1}``, ``[2,4]``, ``[5,16]``, ``[17,64]``, and ``≥65``. With no match, the export uses a conservative ``BM=8``, ``BN=32``, ``BK=32``, ``VM=1``, ``VN=4`` schedule (vector width 8, four threads, grain 1), with tiles resolved against the actual extents.

.. _tools-optimizer-staged-triton-passes:

Staged Triton passes
~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Pass
     - Purpose
   * - ``--triton-to-structured``
     - Convert supported Triton compute operations to structured Linalg.
   * - ``--triton-grid-to-forall``
     - Convert supported program-id/grid dispatch to ``scf.forall``.
   * - ``--triton-block-ptr-to-vector``
     - Convert supported block-pointer accesses to vector transfers.
   * - ``--triton-shared-mem-to-scratch``
     - Convert shared-memory allocations to per-worker scratch.
   * - ``--triton-atomic-to-llrt``
     - Lower supported atomics to runtime calls.
   * - ``--triton-cpu-verify``
     - Reject unsupported GPU operations before CPU lowering.

These passes have no project pass options. Arbitrary Triton dumps are not a supported input promise; use the fixtures under :source:`test/Conversion/TritonToLLK <test/Conversion/TritonToLLK>` to understand the supported generic-form operations and stage order.

.. _tools-optimizer-inspect-scalar-and-vector-lowering:

Inspect scalar and vector lowering
----------------------------------

For scalar loop IR, lower a supported semantic workload and omit the vectorization pass:

.. code-block:: sh

   build/llk-opt --llk-to-linalg \
     --one-shot-bufferize="bufferize-function-boundaries" \
     --convert-linalg-to-loops docs/manual/source/examples/swiglu.mlir \
     -o build/tutorial/swiglu.scalar.mlir

The artifact contains ``memref`` operations and scalar ``scf.for`` loops, with no explicit ``vector.*`` operations. This is an IR inspection pipeline; ``llk-opt`` does not compile it or time it.

For the legacy vector stage:

.. code-block:: sh

   build/llk-opt --tile-and-vectorize docs/manual/source/examples/swiglu.mlir \
     -o build/tutorial/swiglu.vector.mlir

Expect tiled SCF structure and operations such as ``vector.transfer_read``, ``vector.contract``, and ``vector.transfer_write``. Use SwiGLU or a real Linalg matmul here: ``llk.matmul`` is left unchanged by the current LLK-to-Linalg pass. The full legacy compiler always schedules its own passes; there is no ``llk-compile --scalar`` switch.

For the Micro reference bridge:

.. code-block:: sh

   build/llk-opt --micro-to-linalg build/tutorial/matmul.micro.mlir \
     -o build/tutorial/matmul.reference.mlir

The generated kernel becomes an ordinary function with the signature declared by the kernel. The source LLK function may still be present. Micro spatial loops become serial SCF structure, pipeline overlap is not realized, and tiles become tensors. This is useful for inspecting numerical lowering, but it is not evidence that selected hardware concurrency or AVX2 width reached machine code. The :ref:`mapped compiler <tools-compiler-map-through-a-registered-package>` owns target emitter dispatch and backend compilation.

.. _tools-optimizer-mapping-configuration-and-options:

Mapping configuration and options
---------------------------------

``micro-map``, ``micro-bind-plan``, and ``micro-verify-mapping`` all load a target description from **five required keys**:

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Key
     - Meaning
   * - ``target=<label>``
     - Opaque target identity recorded/checked by generic mapping code. It does not load a registered compiler package by itself.
   * - ``machine=<path>``
     - MachineModel v2 YAML profile.
   * - ``layouts=<path>``
     - Layout declarations in LLKMap format.
   * - ``rules=<path>``
     - Mapping rules in LLKMap format.
   * - ``emitters=<csv>``
     - Emitter keys declared available to this target description.

The complete target is validated before searching or verifying a kernel. The emitter CSV must include every emitter referenced by the loaded rule library, even if the particular workload does not select it. For the current shipped AVX2 rule file that includes ``avx2_fused_convert_silu_mul``. An older nine-key CSV fails with ``unknown emitter``, including on a simple add workload.

Set a reusable option string in the shell:

.. code-block:: sh

   AVX2_MAP='target=x86-avx2 machine=machines/x86-avx2-v2.yaml layouts=mapping/x86-avx2/layouts.llkmap rules=mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store,avx2_fused_convert_silu_mul'
   SEARCH='mode=exact top-k=8 beam-width=64'

All three passes require exactly one ``micro.kernel``. They do not export semantic input themselves or offer a kernel-selection option. Retain the unbound concrete module as your search/replay input.

``micro-map`` searches and binds the best retained plan. Its additional keys are:

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Key
     - Default
     - Contract
   * - ``mode=deterministic\|beam\|exact``
     - ``beam``
     - Deterministic selection, bounded beam search, or exact search over the admitted candidates/instances/connections.
   * - ``top-k=<unsigned>``
     - ``8``
     - Maximum complete plans retained.
   * - ``beam-width=<unsigned>``
     - ``64``
     - Beam frontier width.
   * - ``report=<path>``
     - Empty
     - Write the versioned JSON search/plan report before binding.
   * - ``report-only=0\|1``
     - ``0``
     - With ``1``, require ``report=``, write it, and leave the module unbound.
   * - ``candidate=<symbol>``
     - Empty
     - Constrain search to a ``micro.candidate`` in this same module; omit ``@``.
   * - ``require-executable=0\|1``
     - ``0``
     - With ``1``, reject decisions the binder cannot materialize instead of binding a partial plan with warnings.

The module's supported ``micro.objective`` controls comparison order; absent one, latency minimization is used. Inspect report provenance, cap/truncation fields, and diagnostics rather than assuming the selected plan is globally optimal. Exact mode still works within internal enumeration/resource caps. Current storage validation can reject plans **after** top-K retention, so increasing top-K may expose a legal plan but does not establish a complete feasibility guarantee. The :source:`current gap assessment <docs/reviews/2026-10-07-issue67-current-gap-assessment.md>` explains this and other resource-model limits.

.. _tools-optimizer-save-an-analysis-report-without-binding:

Save an analysis report without binding
---------------------------------------

After creating ``build/tutorial/matmul.micro.mlir`` above:

.. code-block:: sh

   build/llk-opt "--micro-map=$AVX2_MAP $SEARCH report=build/tutorial/matmul.plan.json report-only=1" \
     build/tutorial/matmul.micro.mlir -o build/tutorial/matmul.analysis.mlir

   diff build/tutorial/matmul.micro.mlir build/tutorial/matmul.analysis.mlir

The JSON report records schema version 2, compiler/cost-model versions, source-module and source-graph hashes, target/machine/rule/layout hashes, search options/caps, scores, selected plan id, diagnostics, and frozen ``selectedState``. The output MLIR remains unbound. The ``diff`` compares printer-normalized files here; the pass's semantic contract is that it does not mutate the module.

Inspect the report as structured data:

.. code-block:: sh

   python3 - <<'PY'
   import json
   with open('build/tutorial/matmul.plan.json') as stream:
       report = json.load(stream)
   for key in ('version', 'compilerVersion', 'target', 'selectedPlanId', 'searchTruncated'):
       print(key, report[key])
   print('search options:', report['searchOptions'])
   PY

``report-only=1`` skips binding and thus does not test the binder's executable contract. ``require-executable=1`` has no binding to enforce in that mode. A report can also describe skipped storage planning or other analysis limitations; report generation is not a certificate of executable legality.

.. _tools-optimizer-bind-and-verify-a-plan:

Bind and verify a plan
----------------------

.. code-block:: sh

   build/llk-opt "--micro-map=$AVX2_MAP $SEARCH require-executable=1 report=build/tutorial/matmul.bound.plan.json" \
     --verify-each build/tutorial/matmul.micro.mlir \
     -o build/tutorial/matmul.mapped.mlir

   build/llk-opt "--micro-verify-mapping=$AVX2_MAP" \
     build/tutorial/matmul.mapped.mlir -o build/tutorial/matmul.verified.mlir

   diff build/tutorial/matmul.mapped.mlir build/tutorial/matmul.verified.mlir

The mapped kernel carries ``micro.plan``; covered operations carry ``micro.mapping``; selected connections are recorded in ``micro.routes``, with materialized movement/synchronization where supported. Unlike the compiler's mapped stdout, this output is pure MLIR and can be piped into another ``llk-opt`` invocation.

Verification has three boundaries:

#. The ordinary dialect verifier checks syntax, types, and structural metadata constraints.

#. ``micro-verify-mapping`` loads the same target data and resolves selected rules, bundles, emitters, executors, memories/visibility, layouts, and route topology/metadata. It changes nothing and fails on the first coded violation.

#. A target emitter validates/interprets its selected bundle during mapped compilation.

``micro-verify-mapping`` has only the five configuration keys. It accepts no ``mode``, ``top-k``, ``beam-width``, ``report``, ``candidate``, or ``plan-id``, and it does not search. Verifying a saved module without loading its target cannot establish layer two. Passing layer two does not prove target code generation, measured performance, or complete physical-resource checking: shipped plans can report skipped storage finalization, and selected width/resource/fused execution gaps remain open.

.. _tools-optimizer-reproduce-a-plan-id-with-micro-bind-plan:

Reproduce a plan id with ``micro-bind-plan``
--------------------------------------------

``micro-bind-plan`` re-runs search and binds one retained plan whose content-derived id matches ``plan-id``. It supports the same five configuration keys plus ``mode``, ``top-k``, ``beam-width``, ``candidate``, and ``require-executable``; it has no report-file input or ``report-only`` mode.

.. code-block:: sh

   PLAN_ID=$(python3 - <<'PY'
   import json
   with open('build/tutorial/matmul.bound.plan.json') as stream:
       print(json.load(stream)['selectedPlanId'])
   PY
   )

   build/llk-opt "--micro-bind-plan=$AVX2_MAP $SEARCH plan-id=$PLAN_ID require-executable=1" \
     build/tutorial/matmul.micro.mlir -o build/tutorial/matmul.bound-again.mlir

   diff build/tutorial/matmul.mapped.mlir build/tutorial/matmul.bound-again.mlir

The id accepts the report's bare 16-digit hexadecimal spelling, ``0x``-prefixed hex, or decimal. Preserve it as text; shell integer conversion can mishandle a hash above the signed 64-bit range.

Use the same unbound source, target files/emitter keys, mode, top-K, beam width, and candidate used to produce the id. A binding-derived id includes the source binding hash; omitting ``candidate=`` is a different search point. A missing retained id fails explicitly instead of selecting another plan.

For frozen-state replay that avoids searching, use :ref:`llk-compile --plan-report <tools-compiler-replay-a-saved-selection>`. The distinction matters: reproducing an id depends on retaining it in the same search; frozen-state replay depends on report schema and graph/target-content compatibility.

.. _tools-optimizer-map-at-an-explicit-candidate:

Map at an explicit candidate
----------------------------

The candidate must exist in the same module as the concrete kernel. The fixture below declares an integer ``VW`` and a layout-kind parameter, and includes both satisfiable and deliberately unsatisfiable candidates:

.. code-block:: sh

   build/llk-opt "--micro-map=$AVX2_MAP mode=deterministic candidate=candidate_17 report=build/tutorial/candidate.plan.json" \
     test/Conversion/MicroMapping/micro_map_candidate.mlir \
     -o build/tutorial/candidate.mapped.mlir

The report records ``sourceBindingCandidate``, ``sourceBindingHash``, and the bound parameter values. The candidate pins matching rule parameters and constrains declared layout requirements; layout kinds are resolved through target declarations such as ``implements blocked``. Unknown candidate symbols, violated global constraints, ambiguous kind-to-layout bridges, and contradictory rule requirements fail with a reason.

This is not general schedule instantiation: the mapper searches the concrete workload graph already in the module. It does not regenerate that graph's tiling for every schedule parameter, and a rule without a layout requirement does not gain one merely because a candidate binds a layout. Check the selected rules/parameters in the report.

The generic accelerator's shipped policy can also be inspected with a workload its rules cover:

.. code-block:: sh

   build/llk-opt --micro-map="target=generic-ai-accel machine=machines/generic-ai-accel-v2.yaml layouts=mapping/generic-ai-accel/layouts.llkmap rules=mapping/generic-ai-accel/rules.llkmap emitters=accel_vector_add,accel_mxu,accel_copy mode=exact" \
     test/Conversion/MicroMapping/micro_map.mlir \
     -o build/tutorial/add.accelerator.mlir

This demonstrates different target policy over the same tensor-copy/vector-add fixture. The fixture has no explicit executable ABI; it is for mapping inspection. It does not establish accelerator hardware execution or support for the compiler's tile-level matmul export.

.. _tools-optimizer-troubleshooting-and-next-steps:

Troubleshooting and next steps
------------------------------

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Diagnostic or unexpected result
     - Action
   * - ``missing required key``
     - Supply all five configuration keys inside the quoted pass argument. ``machine=`` alone is insufficient for mapping verification.
   * - ``unknown emitter``
     - Include every emitter referenced by the entire loaded rules file. Use the complete current CSV above.
   * - No kernel / several kernels
     - Export first and use one-kernel input; these mapping passes have no symbol selector.
   * - ``report-only requires report=<path>``
     - Supply a report filename and ensure its parent directory exists.
   * - No complete plan / ``no_matching_rule``
     - Read the frontier diagnostics; check operation variant, dtype, layout and candidate requirements. Inspect truncation before concluding the workload is unmappable.
   * - No retained satisfiable storage plan
     - Inspect physical footprint/capacity and the report/search limits. Top-K filtering and storage finalization currently have known ordering limitations.
   * - ``no plan ... has id``
     - Reproduce the original source, configuration, mode, top-K, beam width, and candidate. Do not search already-bound IR.
   * - Structural verification passes but mapping verification fails
     - Restore the matching target data; ordinary parsing cannot resolve external rule/resource/layout ids.
   * - ``micro-to-linalg`` rejects a signature or store
     - Declare kernel input/result types and thread destination values through stores/yields. An analysis-only kernel need not have an executable ABI.
   * - No change after ``--llk-to-linalg``
     - Confirm the operation is supported; current ``llk.matmul`` lowering belongs to the Micro export path.
   * - ``schedule-file``/``target-isa`` changes no output
     - These ``tile-and-vectorize`` options are currently unused. Use the active Micro schedule-database path for schedule experiments.

Evaluate a concrete or mapped kernel with :doc:`micro-perf </tools/performance>`, explore schedules with :doc:`llk-tune </tools/tuning>`, and compile an explicit kernel ABI with :doc:`llk-compile </tools/compiler>`. :doc:`llk-bench </tools/benchmark>` measures a synthetic host harness; timing a compiled kernel requires a C++ invocation/measurement caller. Consult the :source:`current gap assessment <docs/reviews/2026-10-07-issue67-current-gap-assessment.md>` when interpreting mapping, resource, and execution claims.
