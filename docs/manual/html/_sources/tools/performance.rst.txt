.. _tools-performance-micro-perf-estimate-a-concrete-micro-kernel:

``micro-perf``: estimate a concrete Micro kernel
================================================

``micro-perf`` reads a ``micro.kernel`` and a machine profile, builds a cost-event
graph, and reports modeled work, memory traffic, cycles, utilization, and
diagnostics. It does not execute the kernel, produce tensors, check numerical
accuracy, or measure the host CPU. Use it to compare explicit Micro schedules
under the same model and to inspect assumptions before attempting execution.

This manual describes main revision ``bb82038`` (2026-10-07). The current
:source:`gap assessment <docs/reviews/2026-10-07-issue67-current-gap-assessment.md>`
records remaining mapping and performance-model limitations. Measurement and
calibration work tracked by #51/#52 is separate from the simulator.

.. _tools-performance-first-report:

First report
------------

Run these commands from the repository root after building the tools. Paths
such as ``build/bin/micro-perf`` refer to your local build; profile and fixture paths
refer to the source checkout.

.. code-block:: sh

   build/bin/micro-perf --help
   build/bin/micro-perf --machine=machines/x86-avx2-v2.yaml --level=0 \
     test/Perf/micro_perf_cli.mlir
   build/bin/micro-perf --machine=machines/x86-avx2-v2.yaml --level=1 \
     test/Perf/micro_perf_cli.mlir
   build/bin/micro-perf --machine=machines/x86-avx2-v2.yaml --format=text \
     --kernel=gemm_tile test/Perf/micro_perf_cli.mlir

The fixture models a ``16×16×32`` BF16 MMA, two copies into SRAM, a wait,
and a store from an F32 accumulator. At this revision with the shipped x86
profile, both reports contain 16,384 MMA FLOPs, 3,072 DRAM bytes, 2,048 SRAM
bytes, and 1,024 accumulator bytes. L0 reports 316 predicted cycles; L1 reports
1,748 cycles and 582.7 predicted ns. These are model outputs, not benchmark
results or a claim about a particular processor.

Reports go to standard output. Redirect stdout to keep YAML for comparison;
diagnostic errors go to standard error.

.. code-block:: sh

   build/bin/micro-perf --machine=machines/x86-avx2-v2.yaml --level=1 \
     test/Perf/micro_perf_cli.mlir > build/gemm-perf.yaml

.. _tools-performance-options-and-defaults:

Options and defaults
--------------------

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Option
     - Default
     - Accepted values and effect
   * - Positional input
     - Required
     - MLIR file containing a concrete ``micro.kernel``; ``-`` reads stdin.
   * - ``--machine=<path>``
     - Required
     - MachineModel YAML, for example ``machines/x86-avx2-v2.yaml``.
   * - ``--level=<uint>``
     - ``1``
     - ``0``: static bound; ``1``: static bound plus resource scheduling. Spellings such as ``l1`` are not accepted.
   * - ``--format=<string>``
     - ``yaml``
     - ``yaml`` or ``text``. There is no JSON formatter.
   * - ``--kernel=<string>``
     - Empty
     - Symbol name, without ``@``; required if the file contains more than one kernel.
   * - ``--fail-on-capacity-violation``
     - Disabled
     - Print the report, then return status 2 if modeled capacity violations exist.
   * - ``--help``, ``--version``
     - —
     - LLVM command-line help/version information.

The tool registers standard MLIR dialects, ``micro``, and ``llk``, so compiler
output containing retained source operations can be parsed. It analyzes the
selected kernel only. A ``micro.search_space`` alone is not a concrete kernel;
use :doc:`llk-tune </tools/tuning>` for search-space analysis or the lowering/mapping
workflow to produce concrete IR.

If there is exactly one kernel, it is selected automatically. Kernel selection
does not filter individual operations, events, memory spaces, or report fields.
There are no CLI selectors for those finer filters.

.. _tools-performance-l0-and-l1:

L0 and L1
---------

Both levels extract the same Micro event graph and work counters. Supported
loops expand into events with their modeled execution multiplicity. A logical
view is normally free; a modeled layout conversion can require vector work.
Copies charge bytes to both memory endpoints.

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Level
     - Computation
     - Interpretation
   * - L0 (``--level=0``)
     - Aggregate matrix/vector event work divided by available engine slots; synchronization cost; the largest memory-level cost using its latency and bytes/bandwidth. ``predicted_cycles`` is the maximum of compute and memory bounds.
     - An optimistic static model that omits event dependencies and resource scheduling. It is a model bound, not a proven bound on measured hardware time.
   * - L1 (``--level=1``)
     - Deterministic scheduling of the extracted events against dependencies, engine slots, and modeled owner occupancy.
     - A resource-schedule prediction, with time derived from the profile clock and additional utilization/overlap fields.

The current event scheduler shares infrastructure with mapping cost analysis.
That sharing does not establish agreement between a mapping planner's score and
the performance report on the bound kernel: the two paths still describe some
work, multiplicity, and resource choices differently. Compare both outputs when
reviewing a selected plan; do not substitute one for the other.

.. _tools-performance-reading-the-report:

Reading the report
------------------

YAML uses ``schema_version: 1``. Collections have deterministic key order, making
reports useful for diffs. The text format presents the same analysis for people.

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Field
     - Unit / meaning
   * - ``machine``, ``kernel``, ``level``
     - Profile target name, selected kernel symbol, and analysis level. These fields do not include complete source/profile hashes.
   * - ``totals.flops``
     - Two FLOPs per MMA multiply-accumulate. Vector and reduction work is not included in this FLOP counter.
   * - ``totals.bytes``
     - Bytes attributed to each memory space. Movement can contribute to both source and destination.
   * - ``operations``
     - Event counts by kind after expansion, rather than a count of source operations.
   * - ``bounds.compute_cycles``
     - Aggregate modeled compute lower bound.
   * - ``bounds.memory_cycles``
     - Dominant modeled memory-level lower bound.
   * - ``predicted_cycles``
     - L0 bound or L1 makespan, in profile cycles.
   * - ``predicted_ns``
     - L1 only: cycles × ``1e9 / clock_hz``. A missing profile clock produces zero, not a measurement.
   * - ``utilization.matrix``, ``.vector``, ``.dma``
     - L1 busy cycles divided by modeled pool slots × makespan; fractions, not percentages.
   * - ``bandwidth.dram``, ``.sram``
     - L1 traffic divided by modeled bandwidth × makespan; fractions that can exceed 1 if the modeled schedule outruns that memory level.
   * - ``overlap_efficiency``
     - L1 ``1 - makespan / sum(event cycles)``, clamped to ``[0,1]``. Zero means no overlap under this definition.
   * - ``tiles.live_bytes``
     - Modeled simultaneously materialized storage by memory space. It is not process RSS or a measured allocation peak.
   * - ``tiles.layouts``, ``.owners``
     - Histograms of layout/owner metadata seen by analysis. Empty maps mean no such metadata was counted.
   * - ``bottleneck``
     - A model classification, for example ``matrix_engine``, ``vector_engine``, ``dma``, ``owner_occupancy``, or a memory name suffixed by ``_bandwidth``.
   * - ``capacity_violations``
     - Modeled live storage exceeding a corresponding machine memory capacity.
   * - ``layout_warnings``, ``owner_warnings``, ``warnings``
     - Assumptions, unsupported combinations, and incomplete modeling information.

For the first fixture the YAML contains this useful subset:

.. code-block:: yaml

   bounds:
     compute_cycles: 128
     memory_cycles: 316
   predicted_cycles: 1748
   predicted_ns: 582.7
   bottleneck: dma
   capacity_violations: []

The fixture also emits a warning that no ``acc -> dram`` copy path is declared and
the copy cost is composed from both endpoints. A successful exit does not mean
the report is warning-free. Read these lists before relying on a comparison.

.. _tools-performance-profiles-and-target-interpretation:

Profiles and target interpretation
----------------------------------

The shipped profiles are:

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Profile
     - Role
   * - ``machines/x86-avx2-v2.yaml``
     - CPU validation model with a 3 GHz modeled clock, worker/vector/matrix resources, and DRAM/L2/SRAM/accumulator memories. Its numbers are calibration seeds, not a measured CPU SKU.
   * - ``machines/generic-ai-accel-v2.yaml``
     - Generic accelerator topology and resource model. An estimate against this profile does not execute accelerator hardware.

The machine argument selects the model used for costs. It does not require the
host to have AVX2 and does not prove that the kernel's ``target`` attribute matches
the selected model. For a comparison, keep IR, machine profile, compiler revision,
analysis level, and warnings together.

The same small fixture can be analyzed against the accelerator model:

.. code-block:: sh

   build/bin/micro-perf --machine=machines/generic-ai-accel-v2.yaml --level=1 \
     test/Perf/micro_perf_cli.mlir

This is useful for checking sensitivity to modeled hardware. It does not
establish numerical correctness or executable target support.

.. _tools-performance-exit-status-and-troubleshooting:

Exit status and troubleshooting
-------------------------------

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Status / symptom
     - Meaning and next step
   * - ``0``
     - Analysis completed. Capacity and other warnings may still be present.
   * - ``1``
     - Invalid option value, unreadable/invalid profile, parse failure, kernel-selection failure, or analysis failure. Read stderr.
   * - ``2``
     - Analysis completed with capacity violations and ``--fail-on-capacity-violation`` enabled. The report has already been printed.
   * - “input contains no micro.kernel”
     - A source-only or search-space-only file was supplied. Lower or bind it first.
   * - “select one with --kernel”
     - There are multiple kernels. Use the exact symbol name without ``@``.
   * - “unsupported --level” / invalid integer
     - Use numeric ``0`` or ``1``.
   * - “unsupported --format”
     - Use ``yaml`` or ``text``.
   * - Model file cannot be opened
     - Run from the source root or use an absolute profile path. Relative paths follow the current working directory.
   * - “bytes are charged as zero”
     - A dynamic extent or unsupported element type prevented size accounting. Those zero counters are incomplete information.
   * - Kernel expands beyond the event limit
     - Start with a smaller concrete fixture; the simulator explicitly expands bounded loops and is not an unbounded symbolic analyzer.

To make modeled capacity a shell gate:

.. code-block:: sh

   build/bin/micro-perf --machine=machines/x86-avx2-v2.yaml \
     --fail-on-capacity-violation test/Perf/micro_perf_cli.mlir

This checked fixture fits the model and returns zero. A capacity violation
reports the memory, required bytes, and modeled available bytes. The model's
live-storage accounting can be conservative or incomplete; it is not a complete
physical-storage proof for a mapped executable. Layout and owner warnings do not
become fatal merely because the capacity flag is enabled.

.. _tools-performance-current-limits-and-contributor-entry-points:

Current limits and contributor entry points
-------------------------------------------

Do not use these predictions as calibrated hardware measurements. The current
mapping audit records selected-resource identity gaps, incomplete physical
storage, per-engine DMA occupancy issues, and planner/perf discrepancies. Those
limitations also constrain the interpretation of a bound-kernel report.
Selected AVX2 width and layout metadata is not by itself evidence of selected
AVX2 instruction execution.

The reporting API is ``analyzeKernel`` in
:source:`include/LLK/Perf/MicroPerfReport.h <include/LLK/Perf/MicroPerfReport.h>`.
Its implementation and serializers are in
:source:`lib/Perf/MicroPerfReport.cpp <lib/Perf/MicroPerfReport.cpp>`, DAG
extraction is in :source:`lib/Perf/MicroDAG.cpp <lib/Perf/MicroDAG.cpp>`, and the
L0/L1 calculations are in
:source:`lib/Perf/MicroCostModel.cpp <lib/Perf/MicroCostModel.cpp>`.
The CLI is :source:`tools/micro-perf/micro-perf.cpp <tools/micro-perf/micro-perf.cpp>`.
Start with ``MicroPerfCLI``, ``L0StaticBound``, and ``L1ResourceDAG`` tests when changing
model behavior; check registered CTest names in your build.

For timed host work, read :doc:`llk-bench </tools/benchmark>` before interpreting its
output. Its present synthetic workload is separate from Micro predictions and
mapped-kernel measurement.
