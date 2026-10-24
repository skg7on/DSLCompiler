.. _models-schedules:

Schedules and the schedule database (``schedules/``)
====================================================

"Schedule" names two different things in this project, and they should not be
confused:

.. list-table::
   :header-rows: 1
   :widths: 26 40 34
   :class: wide-table

   * - Meaning
     - Representation
     - Where
   * - **Schedule data**: a chosen program shape
     - Tile sizes, vector width, pipeline depth, owner mapping, memory path
     - ``schedules/schedule_db.json``, Micro search spaces, YAML schedule records
   * - **The scheduler**: the analysis pass that orders events on resources
     - A deterministic list scheduler over normalized cost events
     - ``lib/Mapping/EventSchedule.cpp``, used by the search and the analyzer

This page documents the data. The event scheduler is described in
:doc:`/tools/performance`; it is the mechanism that turns a selected schedule
and a machine model into a predicted timeline.

.. _models-schedules-files:

What lives in ``schedules/``
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 26 40
   :class: wide-table

   * - Path
     - Role
     - Consumed by
   * - :source:`schedule_db.json <schedules/schedule_db.json>`
     - The legacy schedule database
     - ``--select-schedule``, ``--llk-to-micro`` and
       ``--llk-to-micro-search-space``. The ``llk-tune`` legacy grid *writes*
       this format; it never reads the database.
   * - :source:`x86_avx2/schedule_bf16.mlir <schedules/x86_avx2/schedule_bf16.mlir>`
     - A Transform-dialect reference script for the AVX2 BF16 schedule
     - Parse-checked only; **not applied by the compiler today**
   * - ``aarch64_sve/``
     - Placeholder for a future Arm schedule
     - Nothing yet

The database is the active input. The Transform script describes the loop
structure the legacy pass builds, but the pass does not read it; see
:ref:`models-schedules-transform`.

.. _models-schedules-db-schema:

Schedule database schema
------------------------

.. code-block:: json

   {
     "version": 1,
     "entries": [
       {
         "operation": "fused_swiglu",
         "target": "x86-avx2",
         "shape": { "M_bucket": 2, "N": 4096, "K": 4096 },
         "dtype": "bf16",
         "math_mode": "bounded_fast",
         "schedule": {
           "BM": 8, "BN": 64, "BK": 64,
           "VM": 2, "VN": 8,
           "vector_width": 8,
           "num_threads": 8,
           "parallel_axis": "n",
           "grain_size": 2,
           "pipeline_stages": 1,
           "memory_path": "dram:sram:acc",
           "mma_shape": "16x16x32",
           "tile_layout": "row_major",
           "owner_mapping": "worker/lane",
           "fragment_shape": "8x8x32",
           "tail_policy": "none"
         }
       }
     ]
   }

Entry keys
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 12 30 36
   :class: wide-table

   * - Key
     - Required
     - Meaning
     - Used in selection?
   * - ``operation``
     - yes
     - Workload name, e.g. ``fused_swiglu``, ``rope``, ``attention``
     - Yes, exact match
   * - ``shape.M_bucket``
     - yes
     - M bucket 0–4 (see below)
     - Yes, exact match
   * - ``shape.N``, ``shape.K``
     - yes
     - Contraction extents
     - Read, but **not** used for filtering or ranking today
   * - ``target``
     - no
     - Target label
     - No
   * - ``dtype``
     - no
     - Element type
     - No
   * - ``math_mode``
     - no
     - Numerical mode, e.g. ``bounded_fast``
     - No
   * - ``source``
     - no
     - Provenance, e.g. ``triton``
     - No
   * - ``schedule``
     - yes
     - The schedule object
     - Yes; the fields the loader reads are listed below

The top-level ``version`` key is present in the shipped file but the current
loader does not read it; only ``entries`` is consumed.

The declared key in :source:`CLAUDE.md <CLAUDE.md>` is
``(operation, M_bucket, N, K, dtype, ISA, math_mode)``. The current loader
requires ``operation``, ``shape.M_bucket``, ``shape.N`` and ``shape.K`` to be
present, but only ``operation`` and ``shape.M_bucket`` affect which entry wins;
``N`` and ``K`` are read and then ignored. Entries that differ only in ``N``,
``K``, ``target``, ``dtype`` or ``math_mode`` are indistinguishable to it, and
the first matching entry in file order wins. Order entries deliberately, or use
separate databases.

Schedule fields
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 14 34 28
   :class: wide-table

   * - Field
     - Default
     - Meaning
     - Used by the Micro export?
   * - ``BM``, ``BN``, ``BK``
     - 0 (the export rejects 0)
     - Output and reduction tile sizes
     - Yes
   * - ``VM``, ``VN``
     - 0
     - Legacy vector tile factors
     - Loaded and reported; not applied by the export or the fixed tiling pass
   * - ``vector_width``
     - 0
     - Explicit SIMD width
     - Yes, recorded in the tile layout attribute
   * - ``num_threads``
     - 1
     - Worker count
     - Loaded and reported; not applied by the export
   * - ``parallel_axis``
     - ``""``
     - Axis to parallelize (``m``, ``n``, ``bh``)
     - Loaded and reported; not applied by the export
   * - ``grain_size``
     - 1
     - Loop grain
     - Loaded and reported; not applied by the export
   * - ``softmax_block``
     - —
     - Attention softmax block
     - **Not read**: no field exists in the loader
   * - ``pipeline_stages``
     - 1
     - Stages emitted as ``micro.pipeline``
     - Yes
   * - ``prefetch_distance``
     - 0
     - Tiles prefetched ahead; negatives clamped to 0
     - Loaded; not consumed by the export
   * - ``memory_path``
     - ``dram:sram:acc``
     - Source and staging levels, ``:``-separated
     - Yes
   * - ``mma_shape``
     - ``16x16x32``
     - Declared instruction fragment
     - Recorded; default for ``fragment_shape``
   * - ``accumulator_space``
     - ``acc``
     - Memory space of the accumulator
     - Yes
   * - ``tile_layout``
     - ``row_major``
     - Canonical layout kind
     - Yes
   * - ``owner_mapping``
     - ``worker``
     - ``/``-separated owner chain, at most two levels
     - Yes
   * - ``fragment_owner``
     - ``vector_engine``
     - Owner of the instruction fragment
     - Yes
   * - ``fragment_shape``
     - follows ``mma_shape``
     - Concrete fragment triple
     - Yes
   * - ``enable_tile_masks``
     - true
     - Whether tails use masks
     - Loaded; not consumed by the export
   * - ``tail_policy``
     - ``none``
     - ``none`` rejects non-dividing tiles; ``pad`` zero-pads inputs and crops
       stores
     - Yes

Every schedule field is optional in the JSON and keeps the default above when
absent, so entries written before the Micro export existed still load unchanged.
A missing tile size becomes 0, which the export rejects with an explicit
diagnostic rather than silently choosing a tile.

.. _models-schedules-selection:

How an entry is selected
------------------------

M is classified into five buckets:

.. list-table::
   :header-rows: 1
   :widths: 18 20 62
   :class: wide-table

   * - Bucket
     - M range
     - Note
   * - 0
     - 1
     - Single row
   * - 1
     - 2–4
     - Small M
   * - 2
     - 5–16
     - Medium M
   * - 3
     - 17–64
     - Large M
   * - 4
     - ≥65
     - Very large M

The bucket is an adapter heuristic for the current contraction family. It
carries no inherent cost meaning and is not compared against a measured
threshold.

The shared selector then:

#. Reads ``entries`` from the file.
#. Keeps entries whose ``operation`` equals the workload and whose
   ``shape.M_bucket`` equals the classified bucket.
#. Returns the **first** such entry.

The loader prints a message for a missing file and for syntactically invalid
JSON. A structurally unexpected document (not an object, or no ``entries``
array) yields no matches without a loader message. In every empty-result case
the consuming pass warns that no entry matched and uses the built-in
conservative schedule:

.. code-block:: text

   BM=8 BN=32 BK=32 VM=1 VN=4
   vector_width=8 num_threads=4 grain_size=1 parallel_axis=n
   pipeline_stages=1 memory_path=dram:sram:acc accumulator_space=acc
   tile_layout=row_major owner_mapping=worker fragment_owner=vector_engine
   fragment_shape=16x16x32 enable_tile_masks=true tail_policy=none

The same fallback applies when no entry matches. The exporter then resolves tile
sizes against the actual extents: with ``tail_policy: none`` a tile that does not
divide its extent is an error; with ``pad`` the extent is padded and stores are
cropped.

.. _models-schedules-consumers:

Who reads the database
----------------------

.. list-table::
   :header-rows: 1
   :widths: 30 34 36
   :class: wide-table

   * - Consumer
     - What it does
     - Effect on generated code
   * - ``llk-opt --select-schedule``
     - For each supported LLK root, classifies M, loads a match, emits a remark
     - None. The remark is diagnostic.
   * - ``llk-opt --llk-to-micro``
     - Selects an entry and exports a concrete ``micro.kernel`` from it
     - Yes: the exported tile program is built from the entry
   * - ``llk-opt --llk-to-micro-search-space``
     - Builds a ``micro.search_space`` around the same entry
     - Defines choices; does not execute
   * - ``llk-compile`` legacy pipeline
     - Stage 4 runs ``select-schedule``
     - Remark only; stages 8+ use fixed tile constants
   * - ``llk-tune`` legacy grid
     - Writes the top three measured configurations in the database's
       ``entries`` JSON shape
     - Nothing directly; it proposes new entries

Two consequences are worth stating plainly:

* ``--select-schedule`` does **not** configure ``--tile-and-vectorize``. The
  legacy pass always uses ``BM=32``, ``BN=64``, ``BK=64``, ``VM=4``, ``VN=8``.
* Changing ``VM``, ``VN``, ``num_threads``, ``grain_size`` or
  ``parallel_axis`` in the database changes only the selection remark. It does
  not change the concrete exported kernel, and it does not seed the exported
  search space: that space takes its numeric starting points from ``BM``,
  ``BN``, ``BK``, ``vector_width`` and ``pipeline_stages``, while
  ``num_threads`` and ``grain_size`` are fixed choice sets. The concrete export
  uses ``owner_mapping`` for placement and ``BM``/``BN``/``BK`` for tiling.

.. _models-schedules-to-micro:

From an entry to MicroIR
------------------------

The concrete export translates schedule fields into IR and kernel attributes:

.. list-table::
   :header-rows: 1
   :widths: 26 74
   :class: wide-table

   * - Schedule field
     - MicroIR effect
   * - ``BM``, ``BN``, ``BK``
     - Two-level spatial nest over the output tile, and a K reduction loop.
       Tile sizes are resolved against M/N/K and checked for divisibility.
   * - ``pipeline_stages``
     - A ``micro.pipeline`` inside the K loop with that stage count.
   * - ``memory_path``
     - Source and staging memory spaces of the tile views and staged copies. The
       path must name at least two known spaces and stage through on-chip memory.
   * - ``accumulator_space``
     - Memory of the accumulator tile; must not be ``dram``.
   * - ``tile_layout``
     - ``#micro.layout`` kind on the tiles. A bare ``blocked`` or ``swizzled``
       is rejected because the export carries no block/swizzle parameter.
   * - ``owner_mapping``
     - Owner symbols on the spatial loops and tiles. At most two levels; a
       single level leaves the inner axis with the legacy ``lane`` scope.
   * - ``fragment_owner``, ``fragment_shape``, ``mma_shape``
     - Owner and instruction fragment of ``micro.mma``. The concrete fragment is
       clipped to the tile sizes.
   * - ``vector_width``
     - Width recorded in the ``#micro.layout`` vector dimension on the tiles.
   * - ``tail_policy``
     - ``none`` rejects a non-dividing tile; ``pad`` records padded extents and
       crops the final store.
   * - ``operation``, ``M_bucket``, ``dtype``
     - Workload name and M bucket become kernel attributes; element types are
       carried by the tile and tensor types; the source ``math_mode`` is copied
       when the root has one.

The exporter also records ``memory_path``, ``owner_mapping``, ``tile_layout``,
``mma_shape``, ``fragment_shape``, ``tail_policy`` and valid/padded extents as
kernel attributes, so a later stage can inspect the chosen shape without
replaying the database lookup.

.. _models-schedules-event-scheduler:

The scheduler that consumes a schedule
--------------------------------------

Once a schedule is materialized as a concrete tile program, the performance
side turns it into **normalized cost events** — one compute, transfer,
transform, wait or barrier event per operation and hop — and runs the shared
list scheduler in :source:`lib/Mapping/EventSchedule.cpp <lib/Mapping/EventSchedule.cpp>`.

.. list-table::
   :header-rows: 1
   :widths: 28 72
   :class: wide-table

   * - Input
     - Meaning
   * - Cost events
     - Duration, resource pool, dependencies and work volume, derived from the
       program and the machine model
   * - Machine model
     - Resource identities, ``concurrency``/``count`` slot counts, sync costs
   * - Optional owner list
     - Per-tile owner occupancy, bounded by the machine's owner count

The scheduler places each event on the resource pool it names, respecting
dependencies and resource multiplicity. Among ready events, the lowest event ID
goes first; an event starts when both its dependencies and its resource slots
allow. The result carries:

.. list-table::
   :header-rows: 1
   :widths: 30 70
   :class: wide-table

   * - Result
     - Use
   * - ``predictedCycles``
     - Candidate score in the mapping search and level-1 performance prediction
   * - ``criticalId``
     - The event on the critical path, for bottleneck reporting
   * - ``sequentialCycles``
     - Sum before overlap; the ratio to ``predictedCycles`` is overlap efficiency
   * - ``entries``
     - Per-event start/finish, for utilization figures

Two properties matter for interpreting a result:

* The mapping search and the performance analyzer call the same scheduler, so a
  candidate is ranked by the model the report later displays.
* A resource the machine does not model is scheduled on one slot rather than
  rejected. Scheduling produces a cost; cost never decides legality. A caller
  that needs a fully modeled stream validates resources first.

The scheduler does not read ``schedule_db.json``. By the time it runs, the
schedule choice is already expressed as operations and attributes in the
program; the database's only role was to help produce that program.

.. _models-schedules-transform:

The Transform-dialect script
----------------------------

:source:`schedules/x86_avx2/schedule_bf16.mlir <schedules/x86_avx2/schedule_bf16.mlir>`
is a `Transform dialect <https://mlir.llvm.org/docs/Dialects/Transform/>`__
named sequence for the AVX2 BF16 fused SwiGLU schedule: tile K with
``scf.for``, tile M/N with ``scf.forall``, and vectorize with sizes ``[4, 8]``.
It corresponds to ``BM=32, BN=64, BK=64, VM=4, VN=8``.

It is currently a **reference artifact and parse test**, not an input to
compilation:

* ``--tile-and-vectorize`` accepts ``schedule-file`` and ``target-isa`` options,
  but the implementation does not read either. It always applies its fixed
  constants.
* ``llk-opt`` registers the Transform dialect and its Linalg extensions, so the
  emulator could run such a script; no pipeline invokes it today.
* :source:`test/Transforms/schedule_bf16_check.mlir <test/Transforms/schedule_bf16_check.mlir>`
  checks that the script parses and has the expected structure. It does not
  check that it produces code.

Do not cite the file's presence as evidence that a Transform schedule runs. For
schedule experiments that affect generated MicroIR, use the database path
(``--llk-to-micro``) or the target-aware tuner.

.. _models-schedules-records:

Schedule records (the successor format)
---------------------------------------

``llk-tune`` writes a YAML **schedule record** for each selected candidate. It
is the tile-centric successor to a JSON database entry: where a database entry
stores loop tile sizes, a record also carries the tile hierarchy, layout, memory
placement, owner mapping, pipeline decisions, fragment choices, the machine
identity and the predicted or measured cost.

.. code-block:: yaml

   schema_version: 1
   workload: matmul
   target: x86-avx2
   machine: machines/x86-avx2-v2.yaml
   shape:
     M_bucket: 1
     M: 4
     N: 16
     K: 32
   dtype:
     input: f32
     weight: f32
     output: f32
     accumulator: f32
   candidate:
     id: 0
     values:
       BM: 4
       BN: 16
       BK: 32
       vector_width: 8
       num_threads: 4
     tile:
       hierarchy:
         worker: [4, 16]
         fragment: [4, 8]
       layout:
         input: row_major
       memory_path: [dram, sram, acc]
       owner_mapping:
         outer: worker
         fragment: vector_engine
       tail_policy: none
       pipeline_stages: 1
       vector_width: 8
   metrics:
     perf_level: 1
     predicted_cycles: 12345
     predicted_ns: 4115.0
     dram_bytes: 1048576
   measurement:
     measured: false

The example is abridged; the writer also emits every candidate value, all
layout roles, the remaining metrics and, when present, source and mapping
provenance.

The format is defined by
:source:`ScheduleRecord.h <include/LLK/Perf/ScheduleRecord.h>`. A record is
**not** a drop-in ``schedule_db.json`` entry:

* it stores Micro schedule axes the JSON schema has no field for;
* it separates predicted from measured metrics and records
  ``measurement.measured: false`` explicitly;
* target-aware records add mapping provenance, a frozen plan report and a replay
  identity;
* it is versioned independently (``schema_version``).

Converting a record into a database entry is a deliberate, lossy edit, not a
format conversion. See :doc:`/tools/tuning` and
:doc:`/tutorials/04-tuning`.

.. _models-schedules-experiment:

Inspecting and changing a schedule
----------------------------------

See the selected entry for a supported source:

.. code-block:: bash

   mkdir -p build/tutorial
   build/bin/llk-opt --select-schedule docs/manual/source/examples/matmul.mlir \
     -o build/tutorial/matmul.selected.mlir

The pass emits a remark naming the operation, bucket and selected tile sizes.
The output IR is unchanged.

Export a concrete kernel from the database:

.. code-block:: bash

   build/bin/llk-opt --llk-to-micro="schedule-db=schedules/schedule_db.json target=x86-avx2-cpu" \
     docs/manual/source/examples/matmul.mlir -o build/tutorial/matmul.micro.mlir

Force the fallback to make an experiment independent of the shipped database:

.. code-block:: bash

   build/bin/llk-opt --llk-to-micro="schedule-db=build/tutorial/absent.json" \
     docs/manual/source/examples/matmul.mlir -o build/tutorial/matmul.fallback.mlir

To change a schedule, copy the database, edit the entry, and pass the copy. Keep
the original file for comparisons. Because selection ignores ``N``, ``K``,
``dtype`` and ``math_mode``, verify that your edit is the entry the selector
actually returns: either keep one matching entry per ``(operation, M_bucket)``
or place the intended entry first.

.. _models-schedules-limits:

Limits and guidance
-------------------

* The schema is contraction-shaped. ``BM``/``BN``/``BK``, the five M buckets and
  ``VM``/``VN`` are adapter fields for the current contraction family, not
  generic schedule-language requirements. A new workload family needs its own
  parameter schema.
* Selection is deterministic but coarse: first match in file order, no cost
  comparison, no ranking among entries.
* The legacy ``llk-tune`` grid output is fixed to ``operation: fused_swiglu``,
  ``target: x86-avx2``, ``dtype: bf16`` and ``math_mode: bounded_fast``. It
  proposes entries for that one workload, not for an arbitrary operation.
* The legacy database does not affect the fixed tiling in
  ``--tile-and-vectorize``. Use the Micro export path when a schedule choice must
  change generated IR.
* A predicted cost from a schedule record is not a measurement. Missing
  measurements are recorded as absent, never as zero.
* ``schedules/aarch64_sve/`` is empty on purpose. Arm SVE has mapping policy, a
  machine profile and an executable f32 add/multiply subset with an emulated
  (not native) acceptance path, but no schedule database.

.. seealso::

   :doc:`/models/mapping-policy` for how a concrete Micro kernel is mapped,
   :doc:`/models/machine-model` for the resources a cost is computed against,
   :doc:`/tools/optimizer` for pass options and current inert flags, and
   :doc:`/tools/tuning` for candidate search and schedule records.
