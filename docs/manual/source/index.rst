.. _index-microir-developer-guide:

MicroIR Developer Manual
========================

DSLCompiler aims to **design a generic MicroIR-based, mapping- and search-based DNN compiler framework**. The ``micro`` dialect describes tile computation, placement, movement and schedule; reusable mapping and search select a plan for lowering or analysis. **Matmul and fused SwiGLU are example operator kernels for testing the framework.** AVX2 is the first executable validation target; the generic accelerator package demonstrates policy/topology extension through the same mapping core.

The :source:`framework goal and design specification <docs/superpowers/specs/2026-10-12-generic-microir-dnn-compiler-framework-design.md>` defines project scope, extension contracts and capability-based acceptance. Current implementation coverage is narrower and is recorded in :doc:`features`.

Start with the concepts, build the five tools, then follow the four tutorials.
Each tutorial states what its result establishes; the tool reference explains
the options, output contracts, and failure modes in detail.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Introduction

   concepts
   architecture
   features
   dynamic-shapes
   getting-started
   python-frontend

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Target models and data

   models/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Use the tools

   tools/index
   tutorials/index
   examples/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Get involved

   contributing
   building-docs

.. _index-choose-a-starting-point:

Choose a starting point
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 40 36 24
   :class: wide-table

   * - Your goal
     - Read first
     - Try next
   * - Understand the project in a few minutes
     - :doc:`Concepts </concepts>`, :doc:`features and current limits </features>`
     - :doc:`Architecture </architecture>`
   * - Build and inspect a kernel
     - :doc:`Getting started </getting-started>`
     - :doc:`Your first kernel </tutorials/01-first-kernel>`
   * - Choose a hardware mapping
     - :doc:`Mapping tutorial </tutorials/02-mapping>`
     - :doc:`Optimizer manual </tools/optimizer>`
   * - Estimate hardware costs
     - :doc:`Performance tutorial </tutorials/03-performance>`
     - :doc:`Performance manual </tools/performance>`
   * - Explore schedule choices
     - :doc:`Tuning tutorial </tutorials/04-tuning>`
     - :doc:`Tuner manual </tools/tuning>`
   * - Reuse a kernel across runtime shapes
     - :doc:`Dynamic shapes </dynamic-shapes>`
     - Symbolic export, checked invocation, sampled analysis and replay
   * - Develop the Python frontend
     - :doc:`Python frontend </python-frontend>`
     - Project ``.venv`` setup, PF01 tests and current capability boundaries
   * - Describe a machine or write target policy
     - :doc:`Machine models </models/machine-model>`
     - :doc:`Mapping policy </models/mapping-policy>` and
       :doc:`Schedules </models/schedules>`
   * - Extend a compiler component or target
     - :doc:`Architecture </architecture>`
     - :doc:`Contributor guide </contributing>`

.. _index-tool-manuals:

Tool manuals
------------

.. list-table::
   :header-rows: 1
   :widths: 18 62 20
   :class: wide-table

   * - Tool
     - Main responsibility
     - Manual
   * - ``llk-compile``
     - Export IR or compile a kernel, including the shared mapped compilation path
     - :doc:`Compiler </tools/compiler>`
   * - ``llk-opt``
     - Parse, verify, transform, map, and inspect MLIR
     - :doc:`Optimizer </tools/optimizer>`
   * - ``micro-perf``
     - Estimate cycles, traffic, capacity, and resource pressure
     - :doc:`Performance </tools/performance>`
   * - ``llk-tune``
     - Generate and rank schedule candidates; write schedule records
     - :doc:`Tuning </tools/tuning>`
   * - ``llk-bench``
     - Run the current synthetic host benchmark harness
     - :doc:`Benchmarking </tools/benchmark>`

The tutorials run from the repository root, use binaries in ``build/``, and write generated files under ``build/tutorial/``. The checked-in :doc:`examples </examples/index>` contain real MLIR inputs. The sequence is: semantic source → concrete MicroIR → mapped MicroIR → cost report. Search-space export and tuning are a related workflow, with the current integration limits explained explicitly.

.. _index-reading-the-project-accurately:

Reading the project accurately
------------------------------

This guide describes the implementation after merged PR #133 (``5ee61b1``, 2026-10-11). It introduces the architecture's intended contracts and identifies where implementation evidence is narrower. A successful parse, mapping, cost estimate, JIT compilation, and numerical invocation establish different things.

.. important::

   Mapping success, portable numerical correctness, selected-target execution,
   and calibrated performance predictions need different evidence. Read
   :doc:`features` before treating a successful command as a broader guarantee.

The :doc:`feature guide </features>` summarizes current capabilities. Merged
`PR #133 <https://github.com/skg7on/DSLCompiler/pull/133>`__ delivers #129's resource,
execution and original-source tuning repairs. The
:source:`acceptance matrix <docs/reviews/issue67-final-acceptance.md>` records the
supported contract, selected AVX2 evidence and retained legacy limits. Production
measurement/calibration remains separate in
`#51 <https://github.com/skg7on/DSLCompiler/issues/51>`__ and
`#52 <https://github.com/skg7on/DSLCompiler/issues/52>`__.

.. _index-deeper-references:

Deeper references
-----------------

* :source:`Micro dialect semantics <docs/design/m9-micro-ir-core-concepts.md>` and :source:`tile programming model <docs/superpowers/specs/2026-08-13-micro-ir-tile-programming-model-spec.md>`.

* :source:`Mapping workflow <docs/design/micro-ir-mapping-workflow.md>`, :source:`layout grammar <docs/design/llkmap-layout-grammar.md>`, and :source:`rule grammar <docs/design/llkmap-rule-grammar.md>`.

* :source:`Current framework architecture <ARCHITECTURE.md>`, :source:`normative mapping design <docs/superpowers/specs/2026-09-18-microir-inspired-dslcompiler-enhancement-design.md>` and :source:`historical CPU validation architecture <docs/design/legacy-cpu-architecture.md>`.

* :source:`Milestone design documents <docs/design>` and :source:`implementation plans <docs/superpowers/plans>` explain decisions and task history; start with this guide for current usage.
