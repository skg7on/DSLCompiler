.. _models-index:

Machine models, mapping policy, and schedules
=============================================

The compiler keeps three kinds of target knowledge outside its C++ code, in
files a target author edits:

.. list-table::
   :header-rows: 1
   :widths: 18 22 32 28
   :class: wide-table

   * - Input
     - Files
     - Declares
     - Decides
   * - Machine model
     - ``machines/<target>-v2.yaml``
     - Topology, capabilities, capacities, link and access costs
     - Which resources exist, what each can run or hold, and what movement costs
   * - Mapping policy
     - ``mapping/<target>/layouts.llkmap`` and ``rules.llkmap``
     - Legal layouts, operation rules, capability requirements, bundles
     - Which implementation of an operation is legal on this target
   * - Schedule data
     - ``schedules/schedule_db.json`` and Micro schedule records
     - Tile, vector, pipeline and owner choices
     - Which concrete program shape an operation is exported with

The separation is a contract, not a filing convention. Generic code in
``lib/Mapping`` and ``lib/Machine`` never branches on a target name; it reads
these files and asks normalized questions. A new target supplies new data and a
target plugin, not edits to the mapping core.

.. _models-three-questions:

Three questions, three files
----------------------------

The files answer different questions, and confusing them is the usual source of
an unsupported claim:

.. list-table::
   :header-rows: 1
   :widths: 30 34 36
   :class: wide-table

   * - Question
     - Answer comes from
     - It does **not** establish
   * - Can this executor run the work, and can it see this memory?
     - Machine model
     - That an emitter exists, or that the resulting code runs on hardware
   * - Which implementation is legal for this operation?
     - LLKMap rules and layouts
     - That the implementation is fast, or calibrated
   * - Which tile sizes and loop structure should be used?
     - Schedule data
     - That the schedule is legal on a different target or machine

A machine profile describes a *class* of machine. Its numbers are modeling
seeds; declaring ``latency_cycles: 220`` does not measure a DRAM access. An
LLKMap rule names a target bundle and emitter key; it does not implement either.
A schedule-database entry records a choice; it does not prove that choice was
measured.

.. _models-how-they-combine:

How they combine during compilation
-----------------------------------

The MicroIR path uses all three in a fixed order. The legacy CPU validation
path uses none of them to choose code: its loop structure is compiled in, and
its schedule-database lookup is reported as a remark rather than applied.

.. code-block:: text

   micro.kernel (concrete tile program)
        |
        |  extract
        v
   WorkloadGraph --- nodes, values, boundary ports
        |
        |  match          <-- LLKMap rules (operation patterns, ports, bundles)
        v
   MappingCandidate
        |
        |  place          <-- Machine model (executor/compute/memory kinds)
        |                       LLKMap layouts (parameter domains + machine queries)
        v
   CandidateInstance
        |
        |  connect        <-- Machine model (directed links, transfer engines)
        v
   Connections: direct | transfer | transform | both
        |
        |  cover + rank   <-- cost model (machine latency/bandwidth/capacity)
        v
   CoveringPlan --- complete materialized plan, storage feasibility checked
        |
        |  bind
        v
   mapped micro.kernel + plan metadata
        |
        |  target emitters (LLKMap bundle + emitter key)
        v
   lowered / executable code

Search-space binding adds one more input: a ``micro.candidate`` can pin rule
parameters and project its ``owner_mapping`` and ``memory_path`` choices onto
explicit requirements. That projection is described in
:doc:`/models/mapping-policy`.

.. _models-what-the-files-cannot-do:

Boundaries worth remembering
----------------------------

* **A profile is not a backend.** ``machines/arm-sve-v2.yaml`` supports static
  mapping and analysis; it does not execute SVE instructions. Only the AVX2
  package has a selected-target execution path, and only on a compatible
  x86_64 host.
* **A rule is not a cost proof.** Selecting a bundle says the target declares an
  implementation and an emitter. Whether that emitter lowers, and whether the
  code matches the modeled cost, are separate questions answered by target
  lowering and measurement.
* **A schedule entry is not portable IR.** ``schedule_db.json`` entries carry
  legacy contraction-shaped fields (``BM``/``BN``/``BK``, five M buckets). They
  are input to the LLK-to-Micro exporter, not a general schedule language.
  Future workloads need their own parameter schema.
* **Missing measurement is not zero cost.** Predicted metrics and measured
  timings are recorded separately; an unmeasured candidate is reported as
  unmeasured, not as fast.

.. _models-pages:

The three inputs in detail
--------------------------

.. toctree::
   :maxdepth: 1

   machine-model
   mapping-policy
   schedules

For a guided tour, see :doc:`/tutorials/02-mapping` (mapping) and
:doc:`/tutorials/04-tuning` (schedule search). For the pass and option
reference, see :doc:`/tools/optimizer` and :doc:`/tools/compiler`.
