.. _tutorials-01-first-kernel-tutorial-1-inspect-your-first-kernel:

Tutorial 1: inspect your first kernel
=====================================

**Goal:** understand how semantic tensor computation becomes a tile execution program. **Prerequisite:** the five tools from :doc:`getting started </getting-started>`. Run every command from the repository root. This tutorial inspects compilation artifacts; it does not invoke a kernel on tensor data.

.. _tutorials-01-first-kernel-1-read-the-source:

1. Read the source
------------------

Open :download:`matmul.mlir </examples/matmul.mlir>`. Its function accepts A with shape ``16×64``, B with shape ``64×64``, and a destination operand with shape ``16×64``. Elements are BF16; the arithmetic accumulates in FP32 and returns BF16.

.. code-block:: mlir

   %y = llk.matmul ins(%a, %b : tensor<16x64xbf16>, tensor<64x64xbf16>)
       outs(%init : tensor<16x64xbf16>)
       {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
       -> tensor<16x64xbf16>

``llk.matmul`` describes the operation. It does not choose a memory node, worker, DMA route, or hardware implementation. ``bounded_fast`` expresses a numerical contract in IR; choosing another math mode should be a deliberate semantic change.

.. _tutorials-01-first-kernel-2-parse-and-verify:

2. Parse and verify
-------------------

.. code-block:: bash

   mkdir -p build/tutorial
   build/llk-opt docs/manual/source/examples/matmul.mlir > build/tutorial/matmul.parsed.mlir

Expected: an equivalent ``func.func @matmul`` containing ``llk.matmul``. Change one dimension so the operands no longer agree and the verifier will explain the mismatch. Revert that experiment before continuing.

.. _tutorials-01-first-kernel-3-inspect-structured-lowering:

3. Inspect structured lowering
------------------------------

.. code-block:: bash

   build/llk-opt --llk-to-linalg docs/manual/source/examples/swiglu.mlir \
     > build/tutorial/swiglu.linalg.mlir

For this step use the :download:`SwiGLU source </examples/swiglu.mlir>`, whose LLK-to-Linalg lowering is implemented. Open the result: the semantic operation becomes generic tensor/Linalg computation, including two projections and an elementwise epilogue. This is the structured CPU path's starting point. It is useful for inspecting transformations before bufferization. In contrast, the current ``--llk-to-linalg`` pass leaves ``llk.matmul`` untouched, so the plain LLK matmul example must use the Micro/mapped path for compilation. The current Micro exporter is a separate pass over supported LLK operations; do not feed this lowered file to it expecting a general Linalg importer.

.. _tutorials-01-first-kernel-4-export-microir:

4. Export MicroIR
-----------------

.. code-block:: bash

   build/llk-compile --emit=micro docs/manual/source/examples/matmul.mlir \
     > build/tutorial/matmul.micro.mlir
   build/llk-opt build/tutorial/matmul.micro.mlir \
     > build/tutorial/matmul.micro.roundtrip.mlir

Find ``micro.kernel @matmul_M16_N64_K64``. The source function remains alongside it because export is non-destructive. Inside the kernel, look for:

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Construct
     - What it tells you
   * - Kernel arguments and result
     - Explicit external operands and output contract
   * - ``micro.spatial_for``
     - Output tile iteration and spatial mapping intent
   * - ``micro.for`` with ``iter_args``
     - Sequential K accumulation carried through SSA values
   * - ``micro.pipeline``
     - Schedule structure and declared pipeline depth
   * - ``micro.tile_view``
     - Logical tile selection from a larger tensor
   * - ``micro.tile_async_copy`` / ``micro.wait``
     - Staging and readiness dependencies
   * - ``micro.mma``
     - Matrix work over a materialized tile
   * - ``micro.vector "convert"``
     - Output element conversion
   * - ``micro.tile_store`` / ``micro.yield``
     - Output write-back and loop/result propagation

The exporter uses a schedule database when a matching entry exists, with a conservative fallback otherwise. Exact tile shapes can differ with the database and compiler revision. To deliberately select the built-in fallback for the remaining tutorials, use a filename that does not exist:

.. code-block:: bash

   build/llk-opt --llk-to-micro="schedule-db=build/tutorial/no-schedule.json" \
     docs/manual/source/examples/matmul.mlir > build/tutorial/matmul.fallback.micro.mlir

Keep ``no-schedule.json`` absent. A missing database is intentional in this exercise; production work should supply and record its schedule configuration.

.. _tutorials-01-first-kernel-5-see-a-fused-workload:

5. See a fused workload
-----------------------

.. code-block:: bash

   build/llk-compile --emit=micro docs/manual/source/examples/swiglu.mlir \
     > build/tutorial/swiglu.micro.mlir

:download:`SwiGLU </examples/swiglu.mlir>` computes ``SiLU(X·Wg) ⊙ (X·Wu)``. Inspect the two accumulator/MMA arms and the vector epilogue. A fused semantic operation decomposes into execution primitives; the kernel body does not retain a high-level ``llk.fused_swiglu`` operation. The adjacent source function still does.

For a search-space view of the same source:

.. code-block:: bash

   build/llk-compile --emit=micro-search docs/manual/source/examples/swiglu.mlir \
     > build/tutorial/swiglu.exported.search.mlir

Look for ``micro.search_space``, parameters, constraints, and an objective. It describes possible schedules rather than one executable kernel. :doc:`Tutorial 4 </tutorials/04-tuning>` explains how the current tuner consumes this kind of input.

**What you established:** parsing, semantic lowering, concrete Micro export, and search export. For placement and selected implementations, continue to :doc:`Tutorial 2 </tutorials/02-mapping>`. For compiler flags and the JIT-versus-invocation distinction, use the :doc:`compiler manual </tools/compiler>`.
