.. _index-microir-developer-guide:

MicroIR Developer Manual
========================

DSLCompiler is an MLIR compiler and hardware evaluation project built around **MicroIR**, the ``micro`` dialect. It describes a kernel's tile computation, placement, movement, and schedule so developers can inspect how it executes on a machine profile. AVX2 is the first validation target; the generic accelerator package demonstrates how target policy can change without changing the mapping core.

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
   getting-started

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

This guide describes the implementation on ``main`` after PRs #128 and #130, using their source baseline ``bb82038``. It introduces the architecture's intended contracts and identifies where implementation evidence is narrower. A successful parse, mapping, cost estimate, JIT compilation, and numerical invocation establish different things.

.. important::

   Mapping success, portable numerical correctness, selected-target execution,
   and calibrated performance predictions need different evidence. Read
   :doc:`features` before treating a successful command as a broader guarantee.

The :doc:`feature guide </features>` summarizes current capabilities. `Issue #129 <https://github.com/skg7on/DSLCompiler/issues/129>`__, its :source:`implementation plan <docs/superpowers/plans/2026-10-07-issue129-gap-closure.md>`, and the :source:`gap assessment <docs/reviews/2026-10-07-issue67-current-gap-assessment.md>` track remaining resource, execution, tuning, and acceptance work. Those plans are not user-facing features already delivered. Measurement calibration is separate work in `#51 <https://github.com/skg7on/DSLCompiler/issues/51>`__ and `#52 <https://github.com/skg7on/DSLCompiler/issues/52>`__.

.. _index-deeper-references:

Deeper references
-----------------

* :source:`Micro dialect semantics <docs/design/m9-micro-ir-core-concepts.md>` and :source:`tile programming model <docs/superpowers/specs/2026-08-13-micro-ir-tile-programming-model-spec.md>`.

* :source:`Mapping workflow <docs/design/micro-ir-mapping-workflow.md>`, :source:`layout grammar <docs/design/llkmap-layout-grammar.md>`, and :source:`rule grammar <docs/design/llkmap-rule-grammar.md>`.

* :source:`Normative mapping design <docs/superpowers/specs/2026-09-18-microir-inspired-dslcompiler-enhancement-design.md>` and :source:`historical CPU architecture <ARCHITECTURE.md>`.

* :source:`Milestone design documents <docs/design>` and :source:`implementation plans <docs/superpowers/plans>` explain decisions and task history; start with this guide for current usage.
