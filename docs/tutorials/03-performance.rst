.. _tutorials-03-performance-tutorial-3-read-a-performance-prediction:

Tutorial 3: read a performance prediction
=========================================

**Goal:** understand the units and evidence in a cost report, starting with one tile whose arithmetic and bytes are easy to check. **Prerequisite:** :doc:`getting started </getting-started>`. Mapping is optional for the first exercise. ``micro-perf`` predicts execution costs; it never computes numerical tensor outputs.

.. _tutorials-03-performance-1-analyze-one-explicit-tile:

1. Analyze one explicit tile
----------------------------

:download:`gemm-tile.micro.mlir </examples/gemm-tile.micro.mlir>` contains a ``16×32`` BF16 A tile, ``32×16`` BF16 B tile, and ``16×16`` FP32 accumulator. Two async copies stage inputs into SRAM, a wait establishes readiness, an MMA performs the work, and a store accounts for output movement.

.. code-block:: bash

   mkdir -p build/tutorial
   build/micro-perf --machine=machines/x86-avx2-v2.yaml --level=0 \
     docs/examples/gemm-tile.micro.mlir > build/tutorial/gemm.l0.yaml
   build/micro-perf --machine=machines/x86-avx2-v2.yaml --level=1 \
     docs/examples/gemm-tile.micro.mlir > build/tutorial/gemm.l1.yaml

L0 computes static compute/memory bounds. L1 also schedules modeled dependencies and resources. Use numeric ``0`` and ``1``, not ``l0``/``l1``, for the current CLI's level values.

.. _tutorials-03-performance-2-check-work-before-cycles:

2. Check work before cycles
---------------------------

For one matrix multiply, the model counts two FLOPs per multiply-accumulate:

.. code-block:: text

   FLOPs = 2 × 16 × 16 × 32 = 16,384
   A bytes = 16 × 32 × 2 = 1,024
   B bytes = 32 × 16 × 2 = 1,024
   C bytes = 16 × 16 × 4 = 1,024

The current fixture reports 3,072 DRAM bytes, 2,048 SRAM bytes, and 1,024 accumulator bytes. These are per-memory traffic accounts; summing all memory rows counts activity at multiple hierarchy levels, not unique input/output bytes. With this fixture, current live storage is 2,048 bytes in SRAM and 1,024 in the accumulator.

Read fields in this order:

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Field
     - What to learn
   * - Kernel, machine, level
     - Verify you analyzed the intended artifact
   * - FLOPs and operation counts
     - Check workload expansion and loop multiplicity
   * - Per-memory bytes
     - Identify where movement is charged
   * - Compute/memory bounds
     - See the simple limits before scheduling effects
   * - Predicted cycles and nanoseconds
     - Use the modeled machine's clock; these are not measured host times
   * - Utilization and bandwidth
     - Locate modeled resource pressure
   * - Live bytes and capacity violations
     - Check the model's storage accounting
   * - Warnings and bottleneck
     - Understand missing paths, assumptions, and dominant modeled costs

The shipped profile has no direct ``acc → dram`` copy path for this tile fixture. Its report warns that the store cost is composed from the endpoints. Keep that warning with the result; a positive cycle number does not make the missing path disappear.

For a compact human-readable report:

.. code-block:: bash

   build/micro-perf --machine=machines/x86-avx2-v2.yaml --level=1 --format=text \
     docs/examples/gemm-tile.micro.mlir

.. _tutorials-03-performance-3-analyze-a-whole-exported-workload:

3. Analyze a whole exported workload
------------------------------------

.. code-block:: bash

   build/llk-compile --emit=micro docs/examples/swiglu.mlir \
     > build/tutorial/swiglu.micro.mlir
   build/micro-perf --machine=machines/x86-avx2-v2.yaml --level=1 \
     build/tutorial/swiglu.micro.mlir > build/tutorial/swiglu.perf.yaml

Two projections give ``2 × M × N × K × 2 = 262,144`` FLOPs for ``M=16,N=64,K=64``, before epilogue work. The model also expands loop/spatial multiplicity. Inspect storage diagnostics: the current owner-count/liveness model can overestimate concurrency. A reported violation is an actionable model result, not a independently proven physical footprint.

For automation, add ``--fail-on-capacity-violation``. Exit code **2** means the analyzed kernel exceeded modeled capacity. Normal reports can still contain capacity violations without failing; CI should choose that policy explicitly.

.. _tutorials-03-performance-4-evaluate-the-bound-artifact:

4. Evaluate the bound artifact
------------------------------

After :doc:`Tutorial 2 </tutorials/02-mapping>`:

.. code-block:: bash

   build/micro-perf --machine=machines/x86-avx2-v2.yaml --level=1 \
     build/tutorial/matmul.mapped.mlir > build/tutorial/matmul.mapped.perf.yaml

Keep this report beside ``matmul.plan.json``. They currently use shared infrastructure but do **not** guarantee equal planner/perf cycles, DRAM traffic, or complete physical resource accounting. The :source:`gap assessment <docs/reviews/2026-10-07-issue67-current-gap-assessment.md>` explains the disagreement and `#129 <https://github.com/skg7on/DSLCompiler/issues/129>`__ plans common bound-kernel analysis.

.. _tutorials-03-performance-5-compare-a-machine-change-responsibly:

5. Compare a machine change responsibly
---------------------------------------

Copy a machine YAML into ``build/tutorial/``, change one capacity or clock/cost parameter while preserving the schema, and rerun ``micro-perf --machine=<your-file>`` on the same IR. Record the original profile, edited profile, compiler revision, IR, and both reports. For mapped IR, changing topology invalidates the old selection's provenance; remap before claiming the new machine can realize the same plan.

Use predicted changes to formulate an experiment. To claim real latency or accuracy, invoke a compiled kernel with defined input data, compare outputs to a numerical reference, and time repeated invocations with a documented methodology. The current :doc:`benchmark tool </tools/benchmark>` does not measure your compiled Micro kernel, and the current CLI tuner has no production measurement provider. Calibration remains separate work in #51/#52.

**What you established:** interpretable static/resource predictions and a reproducible report, with assumptions preserved. See the :doc:`performance manual </tools/performance>` for exact options, report fields, multi-kernel selection, and error behavior.
