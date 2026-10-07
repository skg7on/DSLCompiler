.. _examples-index-tutorial-inputs:

Tutorial inputs
===============

These files are small, checked-in inputs for the :doc:`developer tutorials </index>`. Run commands from the repository root and keep generated artifacts in ``build/tutorial/``.

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Input
     - Purpose
     - Observable result
   * - :download:`matmul.mlir </examples/matmul.mlir>`
     - Static BF16 matrix multiplication with FP32 accumulation
     - Concrete MicroIR with copies, MMA, conversion, and output write-back
   * - :download:`swiglu.mlir </examples/swiglu.mlir>`
     - Two BF16 projections with a SiLU/multiply epilogue
     - Two MMA arms and vector epilogue operations
   * - :download:`gemm-tile.micro.mlir </examples/gemm-tile.micro.mlir>`
     - One explicit tile MMA, independent of the source exporter
     - Cost accounting for 16,384 FLOPs and explicit transfer traffic
   * - :download:`swiglu.search.mlir </examples/swiglu.search.mlir>`
     - A deliberately small, one-candidate search space
     - A predicted schedule YAML record from ``llk-tune``

The first two are semantic source programs, not tensor data files. Their destination operands express the LLK interface. The current Micro exporter declares the computation's external input operands and creates its result internally; it does not preserve an arbitrary destination accumulator's initial contents. The tile example is an analysis fixture with ``tensor.empty``, not a numerical invocation example. The search example is for the current synthetic candidate binder, not a mapping-driven tuner.

The tutorial commands are exercised by :download:`run_tutorials.py </tutorials/run_tutorials.py>`, which checks generated IR, plan replay, prediction reports, and schedule output. It does not claim numerical execution or selected AVX2 instruction verification.

Source fixtures
---------------

The listings below are included from the files used by the tools, so a manual
build always displays the checked-in inputs. Download links above preserve the
original filenames. Follow :doc:`/tutorials/index` for the commands and expected
artifacts.

Matrix multiplication
~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: matmul.mlir
   :language: mlir
   :caption: matmul.mlir

Fused SwiGLU
~~~~~~~~~~~~

.. literalinclude:: swiglu.mlir
   :language: mlir
   :caption: swiglu.mlir

One explicit GEMM tile
~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: gemm-tile.micro.mlir
   :language: mlir
   :caption: gemm-tile.micro.mlir

One-candidate search space
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: swiglu.search.mlir
   :language: mlir
   :caption: swiglu.search.mlir
