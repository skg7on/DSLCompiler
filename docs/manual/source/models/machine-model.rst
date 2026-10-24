.. _models-machine-model:

Machine model (``machines/*.yaml``)
===================================

A machine model is a versioned YAML file describing a machine as a graph of
stable-ID resources. Mapping, layout solving, routing, storage analysis and
performance prediction all read it. They do not embed a target name; target
differences are meant to arrive as model data. One limitation applies today:
several cost and analysis paths expect the conventional capability kinds
``vector_engine`` and ``matrix_engine`` (see :ref:`models-machine-kinds`).

The schema is ``llk.machine.v2``. The loader is
:source:`MachineModelLoader.cpp <lib/Machine/MachineModelLoader.cpp>`; the typed
model and its queries are :source:`MachineModel.h <include/LLK/Machine/MachineModel.h>`.

.. _models-machine-shipped:

Shipped profiles
----------------

.. list-table::
   :header-rows: 1
   :widths: 22 16 34 28
   :class: wide-table

   * - File
     - ``target``
     - Shape
     - Status
   * - :source:`x86-avx2-v2.yaml <machines/x86-avx2-v2.yaml>`
     - ``x86-avx2``
     - 8 workers, AVX2 lanes and FMA units, DRAM/L2/SRAM/accumulator, one DMA
     - First executable validation backend (x86_64 AVX2 hosts)
   * - :source:`generic-ai-accel-v2.yaml <machines/generic-ai-accel-v2.yaml>`
     - ``generic-ai-accel``
     - chip/core/PE-group/PE hierarchy, 16x16x32 MXU, multi-hop DRAM→SRAM→ACC
     - Mapping and analysis only; no hardware backend
   * - :source:`arm-sve-v2.yaml <machines/arm-sve-v2.yaml>`
     - ``arm-sve``
     - 8 workers with 512-bit vector engines; no matrix engine
     - Static mapping/analysis at modeled VL=512
   * - :source:`arm-sme-v2.yaml <machines/arm-sme-v2.yaml>`
     - ``arm-sme``
     - 512-bit SVE plus a modeled matrix engine
     - Static mapping/analysis at modeled VL=SVL=512

Every number in these files is a calibration seed, not a SKU measurement. The
equivalence between a v1 and v2 profile is what preserves old cost baselines;
it is not evidence of accuracy.

.. _models-machine-schema:

File structure
--------------

.. code-block:: yaml

   schema: llk.machine.v2
   target: example
   description: One worker with a vector unit over a two-level memory.
   clock_hz: 3000000000
   worker_threads: 1
   sync:
     barrier_cycles: 100
     wait_cycles: 0

   executors:
     - id: worker.0
       kind: worker
       concurrency: 1

   memories:
     - id: dram.0
       kind: dram
       visible_from: worker.0
       capacity_bytes: 68719476736
       alignment_bytes: 64
       supported_layouts: [row_major]
       bandwidth_bytes_per_cycle: 32
       latency_cycles: 220
     - id: sram.0
       kind: sram
       visible_from: worker.0
       capacity_bytes: 32768
       alignment_bytes: 64
       supported_layouts: [row_major, vectorized, blocked]
       bandwidth_bytes_per_cycle: 64
       latency_cycles: 4

   compute:
     - id: vpu
       kind: vector_engine
       refines: [vector]
       attached_to: worker.0
       element_types: [f32]
       supported_layouts: [row_major, vectorized]
       shapes: [[8]]
       lanes: {f32: 8}
       issue_cycles: 1
       latency_cycles: 4

   transfer_engines:
     - id: dma.0
       kind: dma
       refines: [transfer]
       attached_to: worker.0
       count: 1
       max_outstanding: 1
       setup_cycles: 0

   links:
     - id: dram_to_sram.0
       source: dram.0
       destination: sram.0
       bandwidth_bytes_per_cycle: 64
       latency_cycles: 220
       transaction_bytes: 64
       transfer_engines: [dma.0]

Top-level keys
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 12 16 50
   :class: wide-table

   * - Key
     - Required
     - Type
     - Meaning
   * - ``schema``
     - yes
     - string
     - ``llk.machine.v2``, optionally ``llk.machine.v2.N`` for minor ``N``.
   * - ``target``
     - yes
     - string
     - Target label recorded in plans and reports. Must be non-empty.
   * - ``description``
     - no
     - string
     - Free text; not interpreted.
   * - ``clock_hz``
     - no
     - positive int
     - Enables nanosecond estimates. Cycles remain meaningful when absent.
   * - ``worker_threads``
     - no
     - positive int (default 1)
     - Host workers; the denominator of the compute-utilization metric.
   * - ``sync``
     - no
     - mapping
     - ``barrier_cycles`` and ``wait_cycles`` charged for synchronization.
   * - ``executors``
     - no
     - list
     - Places work can execute.
   * - ``memories``
     - no
     - list
     - Places data can reside.
   * - ``compute``
     - no
     - list
     - Capabilities attached to executors.
   * - ``transfer_engines``
     - no
     - list
     - Resources that move data over links.
   * - ``links``
     - no
     - list
     - Directed memory-to-memory data paths.

All node IDs share one namespace: an executor and a memory cannot both be
``sram.0``.

.. _models-machine-executors:

Executors
~~~~~~~~~

An executor is a place work can execute. Placement matches an abstract Micro
owner against an executor's **kind**, never against a concrete ID.

.. list-table::
   :header-rows: 1
   :widths: 20 12 14 54
   :class: wide-table

   * - Key
     - Required
     - Type
     - Meaning
   * - ``id``
     - yes
     - string
     - Concrete resource identity, e.g. ``worker.0``.
   * - ``kind``
     - yes
     - string
     - An abstract owner class or a target label the profile refines.
   * - ``parent``
     - no
     - string
     - Containment edge to another executor. Must be acyclic.
   * - ``coordinates``
     - no
     - int list
     - Logical position; contributes to symmetry detection.
   * - ``concurrency``
     - no
     - positive int (default 1)
     - Slots sharing this executor. This is how a kind's multiplicity is
       expressed in v2.
   * - ``refines``
     - no
     - string list
     - Abstract owner classes this target label denotes.
   * - ``scheduling_class``
     - no
     - ``in_order`` / ``out_of_order``
     - How the executor's slots order work. Defaults to ``in_order``.
   * - ``equivalent_to``
     - no
     - string list
     - Executors the target declares interchangeable.

``equivalent_to`` must be mutual, name existing executors, and join executors of
the same kind. Placement's symmetry reducer collapses a declared group to one
representative. The switch is ``PlacementOptions::reduceSymmetry``; without a
declaration it falls back to comparing kind, parent, coordinates, concurrency,
scheduling class, and attached compute and visible memory nodes.

``scheduling_class`` is loaded, validated, hashed, and consulted by that
symmetry comparison. The event scheduler does not yet overlap work on
``out_of_order`` executors; today both classes schedule conservatively.

.. _models-machine-kinds:

Kinds and ``refines``
~~~~~~~~~~~~~~~~~~~~~

``refines`` is the alias mechanism. The canonical owner classes are ``group``,
``worker``, ``vector``, ``matrix`` and ``transfer``. A profile may spell its CPU
worker ``lane`` and declare ``refines: [worker]``; a kernel that asks for a
``worker`` then matches it. Refinement targets must be canonical classes, not
other labels, so resolution cannot cycle.

The alias mechanism is applied in different places, and the difference matters
when authoring policy:

.. list-table::
   :header-rows: 1
   :widths: 30 70
   :class: wide-table

   * - Requirement
     - Matching rule
   * - Executor
     - The executor's kind equals the requirement **or** its ``refines`` names
       the requirement (``ownerMatches``).
   * - Compute
     - The capability's ``kind`` must equal the requirement **exactly**.
       ``refines`` is not applied, so write ``vector_engine``, not ``vector``.
   * - Memory
     - The memory's ``kind`` must equal the requirement exactly.
   * - Transfer engine
     - Rules do not carry transfer-kind requirements; validation resolves the
       kind through ``refines``.

Several cost and analysis paths also look up the literal kinds
``vector_engine`` and ``matrix_engine`` rather than resolving a capability's
class. A profile that spells its vector or matrix capability differently will
load and validate, but those paths may fail to find it. Keep the conventional
labels unless you are extending those paths too.

.. _models-machine-memories:

Memories
~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 12 16 46
   :class: wide-table

   * - Key
     - Required
     - Type
     - Meaning
   * - ``id``
     - yes
     - string
     - Concrete node identity, e.g. ``sram.0``.
   * - ``kind``
     - yes
     - string
     - Canonical memory space: ``dram``, ``l2``, ``sram``, ``rf``, ``acc``, or
       ``scratch``.
   * - ``visible_from``
     - yes
     - string
     - Executor scope that can address this memory. Descendants inherit it.
   * - ``capacity_bytes``
     - yes
     - positive int
     - Storage budget checked by plan and storage analysis.
   * - ``alignment_bytes``
     - no
     - positive int (default 1)
     - Address alignment; a route hop must satisfy it.
   * - ``supported_layouts``
     - no
     - string list
     - Layout classes this memory can hold. A memory that declares none supports
       none.
   * - ``banks``
     - no
     - unsigned int
     - Modeled banking degree. Loaded and hashed; not yet consulted by routing
       or the cost model. A declared zero is accepted.
   * - ``bandwidth_bytes_per_cycle``
     - no
     - number
     - Access bandwidth. Zero means not modeled; the loader does not validate
       the sign.
   * - ``latency_cycles``
     - no
     - int
     - Access latency. Zero means not modeled.
   * - ``access_granularity_bytes``
     - no
     - positive int
     - Smallest unit of one access. Absent means not modeled; consumers fall
       back to alignment. Not yet consulted by the router or simulator.

Memory *kinds* name spaces (``sram``); node IDs name instances (``sram.0``). A
plan that selects a memory preserves the concrete ID, because two nodes of the
same kind are different resources with different capacities and links.

.. _models-machine-compute:

Compute capabilities
~~~~~~~~~~~~~~~~~~~~

A compute node is a capability attached to an executor.

.. list-table::
   :header-rows: 1
   :widths: 24 12 16 48
   :class: wide-table

   * - Key
     - Required
     - Type
     - Meaning
   * - ``id``
     - yes
     - string
     - Concrete capability identity.
   * - ``kind``
     - yes
     - string
     - Must resolve to the ``matrix`` or ``vector`` owner class.
   * - ``refines``
     - no
     - string list
     - Alias declarations. Consulted by class resolution; rule and placement
       requirements still match the literal capability kind.
   * - ``attached_to``
     - yes
     - string
     - Executor that hosts the capability.
   * - ``element_types``
     - no
     - string list
     - Input element types the capability accepts.
   * - ``supported_layouts``
     - no
     - string list
     - Layout classes the capability consumes or produces.
   * - ``shapes``
     - yes
     - list of int lists
     - Legal instruction shapes, e.g. ``[[8], [16], [32]]`` or
       ``[[16, 16, 32]]``. Non-empty; every extent positive.
   * - ``lanes``
     - no
     - mapping dtype→int
     - Elements per instruction. Read by layouts through
       ``machine.compute(<kind>).lanes(<dtype>)``.
   * - ``issue_cycles``
     - no
     - unsigned int (default 1)
     - Cycles to issue one work item. The cost model clamps a declared zero
       to 1.
   * - ``latency_cycles``
     - no
     - int (default 0)
     - Completion latency. Loaded and hashed; not yet consulted by the cost
       model.
   * - ``throughput_per_cycle``
     - no
     - number
     - Modeled instruction throughput, used by the MicroDAG estimate.
   * - ``concurrency``
     - no
     - positive int (default 1)
     - Issue slots sharing the capability.
   * - ``accumulator_dtypes``
     - no
     - string list
     - Element types the capability accumulates into.
   * - ``occupancy_limit``
     - no
     - positive int
     - Resident-work limit. Absent means only slot count bounds occupancy.
       Loaded and hashed; not yet consulted by the simulator.

.. _models-machine-transfer:

Transfer engines and links
~~~~~~~~~~~~~~~~~~~~~~~~~~

A transfer engine is a resource that moves data; a link is a directed memory
path that names the engines able to serve it.

.. list-table::
   :header-rows: 1
   :widths: 24 12 16 48
   :class: wide-table

   * - Transfer engine key
     - Required
     - Type
     - Meaning
   * - ``id``
     - yes
     - string
     - Concrete engine identity.
   * - ``kind``
     - yes
     - string
     - Must resolve to the ``transfer`` owner class.
   * - ``refines``
     - no
     - string list
     - Alias declarations.
   * - ``attached_to``
     - yes
     - string
     - Executor the engine is attached to.
   * - ``count``
     - no
     - positive int (default 1)
     - Number of engines this node represents; summed by
       ``transferEngineCount()``.
   * - ``max_outstanding``
     - no
     - positive int (default 1)
     - Outstanding transfers. Consulted by tuner pipeline-legality analysis;
       not by the event simulator.
   * - ``setup_cycles``
     - no
     - int (default 0)
     - Fixed start cost, independent of size.

.. list-table::
   :header-rows: 1
   :widths: 24 12 16 48
   :class: wide-table

   * - Link key
     - Required
     - Type
     - Meaning
   * - ``id``
     - yes
     - string
     - Concrete edge identity.
   * - ``source``, ``destination``
     - yes
     - string
     - Existing memory IDs. Direction is ``source`` → ``destination``.
   * - ``bandwidth_bytes_per_cycle``
     - yes
     - positive number
     - Transfer bandwidth.
   * - ``latency_cycles``
     - no
     - int
     - Per-hop latency.
   * - ``transaction_bytes``
     - no
     - positive int (default 1)
     - Granularity: a copy's size must tile into whole transactions.
   * - ``transfer_engines``
     - no
     - string list
     - Existing transfer engines that can serve the hop.
   * - ``concurrency``
     - no
     - positive int (default 1)
     - Simultaneous transfers over the link. Loaded and hashed; not yet consulted
       by routing or scheduling.
   * - ``directionality``
     - no
     - ``unidirectional`` / ``bidirectional``
     - Defaults to unidirectional. The router currently follows ``source`` →
       ``destination`` only; model a reverse path with an explicit reverse link.

Links join concrete memory nodes. A policy that names spaces (``dram`` to
``sram``) is resolved to a node pair by ``findLinkByKinds``.

.. _models-machine-visibility:

Containment, visibility, and routing
------------------------------------

Three relations answer different questions:

.. list-table::
   :header-rows: 1
   :widths: 24 34 42
   :class: wide-table

   * - Relation
     - Encoded by
     - Query
   * - Containment
     - Executor ``parent`` chain
     - ``isWithin(node, ancestor)``
   * - Memory visibility
     - ``visible_from`` plus containment
     - ``isVisible(memory, executor)``
   * - Data movement
     - ``links`` between memory nodes
     - ``findLinkByKinds(sourceKind, destinationKind)`` and route enumeration

A link is legal only if some engine attached to an executor that can reach the
link's source serves it. Route enumeration checks link existence, engine reach,
memory alignment and layout support, transaction tiling, and intermediate
capacity. Cost is summed per hop; ties break by hop count, then by the
lexicographic link-ID sequence, so two runs on the same machine agree exactly.

.. _models-machine-queries:

Queries the compiler asks
-------------------------

The mapping core talks to the model through these normalized queries:

.. list-table::
   :header-rows: 1
   :widths: 38 62
   :class: wide-table

   * - Query
     - Used to decide
   * - ``ownerMatches(ownerKind, executor)``
     - Whether an abstract owner can run on an executor (kind or refinement).
   * - ``ownerClass(spelling)``
     - Which abstract class a target spelling denotes; nullopt when unknown or
       when nodes disagree.
   * - ``ownerCount(ownerKind)``
     - How many owner slots the machine offers, summing ``concurrency``.
   * - ``isVisible(memory, executor)``
     - Whether a placement may bind a memory at an executor.
   * - ``lanesFor(computeKind, dtype)``
     - Vector width used by a layout requirement.
   * - ``findMemoryOfKind(kind)``
     - First memory of a space, in declaration order.
   * - ``computesOfKind``, ``computesFor``
     - Candidate vector/matrix capabilities for a placement.
   * - ``transferEngineCount``, ``primaryTransferEngine``,
       ``transferEnginesFor``
     - Transfer capacity and per-engine setup/outstanding facts.
   * - ``findLinkByKinds(sourceKind, destinationKind)``
     - The space-level path a rule or connection may use.

``MachineModel`` exposes non-owning pointers into its vectors. A caller must keep
the model alive while it holds results.

.. _models-machine-cost:

From facts to a prediction
--------------------------

The same file supplies every input to the mapping cost model and the performance
analyzer:

.. list-table::
   :header-rows: 1
   :widths: 30 34 36
   :class: wide-table

   * - Fact
     - Cost contribution
     - Where consumed
   * - Compute ``lanes``, ``issue_cycles``
     - Issues = work items / lanes; issue cycles
     - Placement cost, event scheduling, MicroDAG
   * - Compute ``throughput_per_cycle``
     - MMA cycles = flops / throughput
     - MicroDAG
   * - Compute ``latency_cycles``
     - Declared, validated and hashed
     - Not yet consulted by the cost model
   * - Memory ``latency_cycles``, ``bandwidth_bytes_per_cycle``
     - Access cost
     - MicroDAG, utilization
   * - Link ``latency_cycles``, ``bandwidth_bytes_per_cycle``
     - Hop cost = latency + bytes / bandwidth
     - Routing, event schedule
   * - Link ``transaction_bytes``
     - Legality: bytes must tile into whole transactions
     - Routing, plan verification
   * - Transfer ``setup_cycles``, ``count``
     - Fixed start cost; available transfer units
     - MicroDAG, utilization denominator
   * - ``sync.barrier_cycles``, ``sync.wait_cycles``
     - Barrier and wait charges
     - Event schedule, utilization window
   * - ``worker_threads``, ``clock_hz``
     - Utilization denominator; cycles→nanoseconds
     - Cost model, tuner prediction

A prediction is assembled from these numbers and then ranked by the objective
declared in ``micro.objective`` when the program has one, otherwise by the
default minimize-latency order. It is not calibrated against hardware; see
:doc:`/tools/validation` for how measured and predicted values are kept apart.

.. _models-machine-identity:

Identity and replay
-------------------

``canonicalMachineString`` renders the model deterministically, with nodes
sorted by ID inside each section, so YAML key order cannot change the content
hash. ``computeContentHash`` produces the stable ID used by plan reports, the
latency cache and replay checks.

``schemaMinor`` is schema metadata, not machine content: it is not part of the
hash. A plan report records the machine hash, target ID and search options; a
replay requires all of them to match.

.. _models-machine-validation:

Validation
----------

``verifyMachineModel`` checks the invariants a loaded model must satisfy and
returns the first violation in declaration order:

* schema major is 2; ``target`` is non-empty;
* ``clock_hz``, when declared, is positive; ``worker_threads`` is positive;
* every node ID is non-empty and globally unique;
* executor kinds resolve to an owner class; ``refines`` names canonical owner
  classes; ``concurrency`` is positive; ``parent`` exists and containment is
  acyclic;
* ``equivalent_to`` entries are mutual, existing, same-kind, and not self;
* memory kinds are canonical memory spaces; ``visible_from`` exists;
  capacity and alignment are positive;
* compute kinds resolve to ``matrix`` or ``vector``; ``attached_to`` exists;
  ``shapes`` is non-empty with positive extents; ``concurrency`` is positive;
* transfer kinds resolve to ``transfer``; ``attached_to`` exists;
  ``count`` and ``max_outstanding`` are positive;
* link endpoints and transfer engines exist; bandwidth, transaction size and
  concurrency are positive.

Schema versioning follows one rule:

* A file declaring a major other than 2 is rejected, never reinterpreted.
* Unknown keys are rejected when the declared minor is at or below the build's
  supported minor (currently 0), because at the current minor an unknown key is
  a typo.
* A file declaring a *higher* minor may carry keys this build does not know;
  those keys are ignored. Structure, required fields and kinds are still
  enforced.

The minor is self-declared, so a profile can opt out of unknown-key rejection by
claiming a higher minor. That is deliberate forward compatibility, not a
validation bypass for structure.

.. _models-machine-authoring:

Authoring checklist
-------------------

#. Pick canonical kinds where they exist and use ``refines`` for target
   vocabulary. Keep every refinement pointing at a canonical owner class.
#. Model multiplicity with ``concurrency`` / ``count``, not by repeating a node
   for every lane.
#. Give every memory an explicit ``visible_from``; descendant executors inherit
   visibility, ancestors do not.
#. Make every path a directed link, including the reverse of a modeled path.
   Use ``transaction_bytes`` as a granularity, not a size cap.
#. Keep capacities honest: storage analysis rejects plans whose live tiles
   exceed them.
#. Record where numbers came from. A profile's numbers should be traceable to a
   calibration source; otherwise say they are seeds.
#. Run the loader tests after editing:
   ``build/bin/LLKMachineTests`` covers schema, required fields, kinds,
   containment, equivalence and the shipped profiles.

.. seealso::

   :doc:`/models/mapping-policy` for the rules that consume these facts,
   :doc:`/tools/performance` for how predicted costs are reported, and
   :source:`ARCHITECTURE.md <ARCHITECTURE.md>` for component boundaries.
