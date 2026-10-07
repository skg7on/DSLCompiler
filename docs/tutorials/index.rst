Tutorials
=========

Build the tools using :doc:`/getting-started`, then run the tutorials in order.
They use the :doc:`checked-in inputs </examples/index>` and write artifacts
under ``build/tutorial/``. Commands run from the repository root.

.. toctree::
   :maxdepth: 1

   01-first-kernel
   02-mapping
   03-performance
   04-tuning

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Step
     - Result to inspect
     - What you learn
   * - First kernel
     - Semantic, structured, concrete, and search IR
     - How a tensor computation becomes tile operations
   * - Mapping
     - Bound IR and versioned JSON plan report
     - How a machine and target policy select a mapping; how replay works
   * - Performance
     - Work, traffic, cycles, and resource fields
     - How to assess a static prediction before comparing machines
   * - Tuning
     - Predicted schedule YAML records
     - How to bound a search and interpret synthetic candidate ranking

Run the automated companion after building the five tools:

.. code-block:: bash

   python3 docs/tutorials/run_tutorials.py --build-dir build

The :download:`tutorial runner </tutorials/run_tutorials.py>` checks 26 commands
and their artifacts. It verifies parsing, export, mapping/replay, static work
and traffic, and schedule persistence. It does not invoke tensor computations
or prove selected ISA fidelity. See :doc:`/features` for those remaining contracts.
