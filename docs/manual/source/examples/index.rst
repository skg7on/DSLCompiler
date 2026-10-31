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
   * - :download:`dynamic-matmul.mlir </examples/dynamic-matmul.mlir>`
     - Independently dynamic M/N/K with BF16 inputs/output and FP32 accumulation
     - Bounded MicroIR with a read initializer and separate output destination
   * - :download:`dynamic-shape-profile.json </examples/dynamic-shape-profile.json>`
     - Explicit weighted M/N/K observations for dynamic tuning
     - Per-sample costs and weighted ranking; see :doc:`/dynamic-shapes`

The static matmul and SwiGLU files are semantic source programs, not tensor data
files. Static Micro export retains its legacy internally created result form;
dynamic export preserves matmul's initializer as a read input and adds a
separate caller-owned output. SwiGLU uses zero-initialized projections. The tile
example is an analysis fixture with ``tensor.empty``, not a numerical invocation
example. The search-only example exercises compatibility binding. Target-aware
tuning uses a source-plus-space fixture and saves frozen mapping artifacts.

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
