.. _tutorials-02-mapping-tutorial-2-map-a-kernel-and-replay-its-plan:

Tutorial 2: map a kernel and replay its plan
============================================

**Goal:** turn a concrete Micro kernel into a selected, placed, connected plan, inspect the report, and reproduce the same bound IR. **Prerequisite:** :doc:`Tutorial 1 </tutorials/01-first-kernel>`. All commands run from the repository root in the same shell.

.. _tutorials-02-mapping-1-freeze-the-input-and-configuration:

1. Freeze the input and configuration
-------------------------------------

.. code-block:: bash

   mkdir -p build/tutorial
   build/llk-opt --llk-to-micro="schedule-db=build/tutorial/no-schedule.json" \
     docs/manual/source/examples/matmul.mlir > build/tutorial/matmul.fallback.micro.mlir

   EMITTERS="avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store,avx2_fused_convert_silu_mul"
   TARGET_OPTIONS="target=x86-avx2 machine=machines/x86-avx2-v2.yaml layouts=mapping/x86-avx2/layouts.llkmap rules=mapping/x86-avx2/rules.llkmap emitters=$EMITTERS"
   MAP_OPTIONS="$TARGET_OPTIONS mode=exact top-k=8"

These configuration files serve different purposes:

* :source:`Machine profile <machines/x86-avx2-v2.yaml>`: physical topology, capability, capacity, and cost parameters.

* :source:`Layouts <mapping/x86-avx2/layouts.llkmap>`: declarative layout definitions and constraints.

* :source:`Rules <mapping/x86-avx2/rules.llkmap>`: operation patterns, target bundles, placement requirements, and emitter names.

* ``emitters``: implementations this caller permits the mapper to select. This is an availability list; it does not prove each emitter produces the selected ISA faithfully.

``exact`` searches the modeled covering alternatives with explicit budgets; it is not a claim that every physical resource constraint or global optimality condition is already implemented. Start with small fixtures and inspect truncation/diagnostics. See :doc:`current limits </features>`.

.. _tutorials-02-mapping-2-search-bind-and-report:

2. Search, bind, and report
---------------------------

.. code-block:: bash

   build/llk-opt "--micro-map=$MAP_OPTIONS report=build/tutorial/matmul.plan.json" \
     build/tutorial/matmul.fallback.micro.mlir > build/tutorial/matmul.mapped.mlir
   build/llk-opt "--micro-verify-mapping=$TARGET_OPTIONS" build/tutorial/matmul.mapped.mlir \
     > build/tutorial/matmul.verified.mlir

The mapped kernel gains ``micro.plan``, placement metadata on covered operations, and route/movement metadata where needed. The report includes hashes of the input and target configuration, search options, diagnostic counts, retained plans, and a selected plan ID. The verifier checks the implemented metadata contract; it does not close the complete physical-feasibility gaps tracked in #129.

Inspect the report using Python, without an extra JSON utility:

.. code-block:: bash

   python3 -m json.tool build/tutorial/matmul.plan.json

Questions to answer before trusting the selection:

#. Did the report retain a complete selected plan?

#. Which rules, executors, layouts, and routes were selected?

#. Is the search truncated? Were candidates rejected, and why?

#. Does storage analysis report a complete footprint or an explicit skipped/incomplete case?

#. Does the plan describe the target behavior you intended to test?

.. _tutorials-02-mapping-3-replay-by-plan-id:

3. Replay by plan ID
--------------------

Extract the reported ID and rerun the same search settings through the binding pass:

.. code-block:: bash

   PLAN_ID=$(python3 -c 'import json; print(json.load(open("build/tutorial/matmul.plan.json"))["selectedPlanId"])')
   build/llk-opt "--micro-bind-plan=plan-id=$PLAN_ID $MAP_OPTIONS" \
     build/tutorial/matmul.fallback.micro.mlir > build/tutorial/matmul.replayed.mlir
   diff -u build/tutorial/matmul.mapped.mlir build/tutorial/matmul.replayed.mlir

Expected: ``diff`` prints nothing and exits successfully. This pass **searches again** using the same input and options, then binds the matching ID. It is different from the compiler's frozen-report replay via ``--plan-report``. The latter validates provenance and consumes a recorded selection; see the :doc:`compiler manual </tools/compiler>`.

Record the original input file, machine/layout/rule files, emitter list, compiler revision, mode, beam width if used, and top-K/budgets. A plan ID alone is not enough to reproduce a different input or configuration. Keep both sides of a replay comparison on the same path spelling and configuration.

.. _tutorials-02-mapping-4-inspect-a-later-compilation-stage:

4. Inspect a later compilation stage
------------------------------------

.. code-block:: bash

   build/llk-compile --mapping-target=x86-avx2 --mapping-root=. \
     --mapping-mode=exact --mapping-stop=lowered \
     docs/manual/source/examples/matmul.mlir > build/tutorial/matmul.lowered.txt

The output starts with mapping/realization status followed by lowered MLIR. This driver independently exports and maps the source using its normal schedule settings; it is not a replay of the fallback export above. Review the status as well as the IR. Selected emitters can delegate tensor realization to a portable reference bridge; successful lowering does not establish selected AVX2 width or instruction fidelity.

To request JIT compilation through the mapped path:

.. code-block:: bash

   build/llk-compile --mapping-target=x86-avx2 --mapping-root=. \
     --mapping-mode=exact docs/manual/source/examples/matmul.mlir

Expected: a compilation-success status if this host/build supports the path. The CLI does not accept tensor data or invoke the resulting function. Numerical invocation uses ``MappedExecutable`` in the C++ runtime and is demonstrated in the :source:`mapped acceptance tests <test/Execution/mapped_acceptance.cpp>`. Repeated invocation ownership, richer ABI validation, and selected-target realization remain work in #129.

.. _tutorials-02-mapping-5-optional-second-target-experiment:

5. Optional second-target experiment
------------------------------------

The :source:`generic accelerator package <mapping/generic-ai-accel>` is for mapping/model exploration, not an accelerator runtime. Use the hand-authored :source:`copy/add fixture <test/Conversion/MicroMapping/micro_map.mlir>`:

.. code-block:: bash

   build/llk-opt --micro-map="target=generic-ai-accel machine=machines/generic-ai-accel-v2.yaml layouts=mapping/generic-ai-accel/layouts.llkmap rules=mapping/generic-ai-accel/rules.llkmap emitters=accel_vector_add,accel_mxu,accel_copy mode=deterministic report=build/tutorial/accelerator.plan.json" \
     test/Conversion/MicroMapping/micro_map.mlir > build/tutorial/accelerator.mapped.mlir

If emitter names or supported rules change, consult the package and :doc:`optimizer manual </tools/optimizer>`. Compare selected placements and routes with AVX2 rather than assuming the same cost model has been calibrated for both machines.

**What you established:** mapping and report generation, metadata verification, and reproducible plan-ID binding. Continue to :doc:`Tutorial 3 </tutorials/03-performance>` to inspect costs and capacity diagnostics independently.
