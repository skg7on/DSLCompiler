Tool reference
==============

The tools share MLIR inputs and project libraries, but have different output
contracts. Use ``llk-opt`` to inspect an IR boundary, ``llk-compile`` to export
or compile, and ``micro-perf`` to estimate the cost of a concrete kernel.
``llk-tune`` explores schedule candidates. ``llk-bench`` times the current
synthetic host workload.

.. toctree::
   :maxdepth: 1

   compiler
   optimizer
   performance
   tuning
   benchmark

For a guided workflow, start with :doc:`/tutorials/index`. All examples assume
the repository root as the working directory and tool binaries in ``build/``.
Use each binary's ``--help`` to inspect the build you are actually running.
