.. _tools-tuning-llk-tune-enumerate-and-rank-schedules:

``llk-tune``: enumerate and rank schedules
==========================================

``llk-tune`` has two workflows. Without ``--input``, it generates the legacy
SwiGLU shape grid and writes JSON schedule entries. With ``--input``, it loads one
``micro.search_space``, checks generated candidates, constructs compatibility
Micro kernels, predicts their cost, and writes ranked schedule YAML. The shipped
CLI does not execute candidates or tune the selected mapping of the original
source program.

This manual describes main revision ``bb82038`` (2026-10-07). The current
:source:`gap assessment <docs/reviews/2026-10-07-issue67-current-gap-assessment.md>`
explains the mapping-driven tuning gap. #129 and measurement/calibration work
under #51/#52 are not delivered by the workflows documented here.

.. _tools-tuning-choose-a-workflow:

Choose a workflow
-----------------

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Workflow
     - Selector
     - Input
     - Output
     - What determines ranking
   * - Legacy grid
     - Omit ``--input``
     - CLI shape and fixed built-in grid
     - Version-1 JSON with up to three entries
     - All ``measured_gflops`` values currently remain zero; there is no meaningful measured winner.
   * - Micro compatibility tuning
     - ``--input=<file.mlir>``
     - One ``micro.search_space``, CLI shape/dtypes, machine YAML
     - Schema-1 schedule YAML, one document per retained candidate
     - Search-space objective applied to L0/L1 predicted metrics.

The accepted search-space file option is **``--input``**. ``--search-space`` is not an
alias in this revision. Some older prose used that spelling; check
``build/llk-tune --help`` when adapting an example. This is distinct from the ``llk-opt`` pass named
``--llk-to-micro-search-space``.

.. _tools-tuning-a-small-micro-run:

A small Micro run
-----------------

Run from the repository root with a built ``build/llk-tune``:

.. code-block:: sh

   build/llk-tune --help
   build/llk-tune --input=test/Perf/llk_tune_search_space.mlir \
     --machine=machines/x86-avx2-v2.yaml -M=8 -N=64 -K=64 \
     --search=grid --max-candidates=4 --top-k=2 \
     --output=build/tune-grid.yaml

This fixture declares one candidate. The cap is deliberately small, so copying
the command to a larger space does not silently start exhaustive analysis. The
summary reports one generated candidate, one ranked candidate, zero rejected,
5,208 predicted cycles at perf level 1, the objective and report schema, and the
output path. Inspect the file:

.. code-block:: sh

   cat build/tune-grid.yaml

The record includes ``workload: fused_swiglu``, ``M_bucket: 2``, ``M: 8``, the
candidate's tile decisions and predicted cost, and ``measurement: {measured: false}`` in block form. ``--top-k=2`` is a maximum, not a promise of two records.

Use seeded random generation with the same supported options:

.. code-block:: sh

   build/llk-tune --input=test/Perf/llk_tune_search_space.mlir \
     --machine=machines/x86-avx2-v2.yaml -M=8 -N=64 -K=64 \
     --search=random --seed=42 --max-candidates=4 --top-k=2 \
     --output=build/tune-random.yaml

Because this space contains only one point, the grid and random commands cover
the same candidate. Use a search space with several choices when studying
candidate sampling.

.. _tools-tuning-micro-options-and-defaults:

Micro options and defaults
--------------------------

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Option
     - Default
     - Meaning
   * - ``--input=<path>``
     - Empty
     - Nonempty selects Micro mode. The file must contain exactly one search space; there is no CLI symbol selector for multiple spaces.
   * - ``--machine=<path>``
     - ``machines/x86-avx2-v2.yaml``
     - MachineModel used for legality and cost. Paths are relative to the working directory.
   * - ``-M``, ``-N``, ``-K``
     - ``128``, ``4096``, ``4096``
     - Workload rows, output columns, and contraction extent. Supply small explicit positive dimensions for a first run.
   * - ``--input-dtype``
     - ``bf16``
     - Input element dtype.
   * - ``--weight-dtype``
     - ``bf16``
     - Weight dtype supplied to legality checks; see compatibility binding limits below.
   * - ``--accumulator-dtype``
     - ``f32``
     - Accumulator element dtype.
   * - ``--output-dtype``
     - ``bf16``
     - Output element dtype.
   * - ``--workload=<string>``
     - Search-space workload
     - Override the workload label used by compatibility binding. It does not load or change source operations.
   * - ``--search=<string>``
     - ``grid``
     - ``grid`` or ``random``. ``staged`` is explicitly unimplemented.
   * - ``--seed=<uint64>``
     - ``0``
     - Seed used only by random search.
   * - ``--max-candidates=<uint64>``
     - ``0``
     - Cap generated candidates; zero means the whole finite space. Rejected candidates count against the cap.
   * - ``--perf-level=<uint>``
     - ``1``
     - ``0`` static bound or ``1`` resource schedule, as in :doc:`micro-perf </tools/performance>`.
   * - ``--top-k=<uint64>``
     - ``10``
     - Number of ranked candidates retained; zero retains all accepted candidates.
   * - ``--output=<path>``
     - ``tune_result.yaml``
     - Micro schedule YAML file; overwritten, not appended or merged.

Micro dtype vocabulary is ``bf16``, ``f16``, ``f32``, ``i8``, and ``i32``; declaring a dtype does
not guarantee the selected machine/constraints support it. A constraint or
binding may reject a candidate. Use the fixture's matching default dtypes for
the first run.

``-o`` and ``--dry-run`` belong to legacy mode. They do not configure Micro output
or a Micro measurement step. Conversely, Micro search/performance/output options
do not bound or alter the legacy grid. There is no CLI measurement, provider,
mapping-target, rule-file, layout-file, or mapped-compilation option here.

.. _tools-tuning-search-space-contract-and-supported-search:

Search-space contract and supported search
------------------------------------------

The input dialect represents finite domains with ``micro.param``, legality rules
with ``micro.constraint``, and ranking with ``micro.objective``. It can also contain
candidate declarations, but the tuner generates its own candidates from the
parameter product.

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Parameter kind
     - Choice representation
     - Example
   * - ``integer``
     - Integers
     - ``BM``, ``BN``, ``BK``, pipeline stages, vector width
   * - ``layout``
     - Strings
     - ``row_major``, ``vectorized``, ``blocked``
   * - ``memory_path``
     - Strings
     - ``dram:sram:acc``
   * - ``owner_mapping``
     - Strings
     - ``worker/vector_engine``
   * - ``fragment_shape``
     - Strings
     - ``16x16x32``
   * - ``tail_policy``
     - Strings
     - ``mask``

These domain examples illustrate syntax. The binder, constraints, and machine
still decide whether a combination is supported; a symbolic string is not a
declaration of executable target support.

The supported constraint kinds are ``sram_capacity``, ``acc_capacity``,
``mma_compatible``, ``mapping_extent``, ``tail_supported``, ``vector_width_supported``,
``tile_hierarchy_compatible``, ``layout_supported``, ``owner_supported``,
``fragment_compatible``, and ``pipeline_live_tiles``. Only constraints declared by
the space are checked as constraints. Binding and performance extraction can
still fail afterward. This pipeline does not establish complete physical
resource legality for a selected mapping plan.

Grid generation follows parameter declaration order and choice order, with the
last parameter changing fastest. A cap retains the prefix; it does not choose
the best subset before costing. Random generation uses ``mt19937_64``, the supplied
seed, and duplicate suppression. It uses bounded rejection sampling and modulo
selection, rather than an optimizer or a guaranteed uniform complete sampler;
large spaces near exhaustion can return fewer samples than requested. Always
read the generated count.

There is no staged, Bayesian, evolutionary, or mapping-covering search in this
CLI. Mapping search modes such as ``exact`` or ``beam`` belong to the mapping tools,
not ``--search`` here.

.. _tools-tuning-objective-and-ranking:

Objective and ranking
---------------------

The objective comes from ``micro.objective``, not a CLI objective flag. The primary
metric uses its declared ``minimize`` or ``maximize`` direction. Secondary metrics
are compared in their listed order with their intrinsic direction; ties use
candidate ID ascending for determinism.

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Metric name
     - Value
     - Secondary direction
   * - ``latency_cycles``
     - Predicted cycles at the chosen perf level
     - Lower
   * - ``dram_bytes``
     - Modeled DRAM traffic
     - Lower
   * - ``sram_bytes``
     - Modeled SRAM traffic
     - Lower
   * - ``matrix_utilization``
     - Modeled utilization fraction; zero at L0
     - Higher
   * - ``dma_utilization``
     - Modeled utilization fraction; zero at L0
     - Higher
   * - ``measured_ns``
     - Optional measured ns; otherwise predicted ns
     - Lower
   * - ``measured_gflops``
     - Optional measured GFLOPS; otherwise zero
     - Higher

Unknown metric names are errors. In particular, ``capacity_spill_bytes`` is not
produced by this tuner. The CLI installs no measurement provider, so measured
metric names do not cause measurements. Prefer a predicted metric such as
``latency_cycles`` for the shipped CLI.

After ranking, ``top-k`` truncates the retained vector. Consequently, the stdout
“ranked” count is the retained count; generated minus ranked is not necessarily
the number rejected. Rejection reasons are available in the C++ session report
but the CLI prints only their count, and its schedule output contains winners
only. An empty output can be a successful run with no accepted candidates.

.. _tools-tuning-persistence-yaml-versus-legacy-json:

Persistence: YAML versus legacy JSON
------------------------------------

Micro output is a multi-document YAML stream with ``schema_version: 1``. It
records workload, target name, machine path, shape and bucket, dtypes, candidate
ID, integer bindings, tile hierarchy/layout/memory path/owners/pipeline/vector
width, predicted metrics, and a measurement block. Symbolic bindings are not
serialized as a complete bindings map; selected symbolic decisions are
summarized under ``candidate.tile``.

The current writer does not serialize the ranking objective, search method,
seed, compiler revision, complete source hash, machine-content hash, mapping
plan, or measurement ABI identity. Keep those alongside the YAML yourself if
you need to reproduce or compare a study. The CLI prints the objective and
report schema to stdout; save that log too.

This YAML is not a replacement file for ``schedules/schedule_db.json``. The
schedule-consumption pipeline's existing database uses JSON ``version: 1`` and an
``entries`` array. There is no CLI round-trip that loads these Micro YAML records
as frozen mappings or installs them into the legacy JSON database.

Legacy output defaults to ``tune_result.json``. It contains up to three
``fused_swiglu``, ``x86-avx2``, ``bf16``, ``bounded_fast`` entries, keyed by M bucket, N,
and K, with schedule knobs and ``measured_gflops: 0``. It overwrites the requested
file rather than extending the shipped database.

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Bucket
     - M range
   * - ``0``
     - ``1``
   * - ``1``
     - ``2–4``
   * - ``2``
     - ``5–16``
   * - ``3``
     - ``17–64``
   * - ``4``
     - ``65`` and above

.. _tools-tuning-legacy-grid-behavior:

Legacy grid behavior
--------------------

.. code-block:: sh

   build/llk-tune -M=8 -N=64 -K=64 --dry-run \
     -o=build/tune-legacy.json

``--dry-run`` adds a preview of the first five generated configurations. It still
writes JSON. Omitting ``--dry-run`` also writes JSON and still performs no
measurement; the current legacy path has no measurement loop.

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Knob
     - Built-in choices
   * - ``BM``
     - ``1, 4, 8, 16, 32, 64``; M=1 restricts BM to 1; buckets 3–4 skip BM ≤ 4.
   * - ``BN``
     - ``16, 32, 64, 128, 256``
   * - ``BK``
     - ``32, 64, 128, 256``
   * - ``VM``
     - ``1, 2, 4``
   * - ``VN``
     - ``4, 8``
   * - ``vector_width``
     - Fixed at ``8``
   * - ``num_threads``
     - ``1, 2, 4, 8``
   * - ``grain_size``
     - ``1, 2, 4``, subject to the built-in tile-count filter
   * - ``parallel_axis``
     - ``n`` for M=1; ``m`` otherwise

The grid uses a fixed BF16 working-set estimate
``2 × (BM×BK + 2×BK×BN)`` bytes and rejects configurations above 80% of a fixed
32 KiB L1. It does not read ``--machine``, measure packing/dispatch cost, require
exact shape divisibility, or verify a mapping plan. Since every GFLOPS score is
zero, the saved top three are tied configurations, not validated fastest kernels.

.. _tools-tuning-mapping-and-measurement-api-boundaries:

Mapping and measurement API boundaries
--------------------------------------

The C++ tuning core in
:source:`include/LLK/Perf/TuningSession.h <include/LLK/Perf/TuningSession.h>`
offers an optional ``MeasurementProvider``. The session hands it a compatibility
bound module and candidate; the caller must provide compilation, invocation
inputs, and timing. A provider error rejects the candidate, while ``nullopt``
means a measurement miss and preserves the static result. Optional target,
machine, and ABI identity strings accompany observations in memory.

This callback is not a source-candidate or selected-mapping provider. The
production session calls ``bindCandidateToMicroKernel``; it does not instantiate
a mapping problem, select rules/layouts/routes, bind that covering, or compile
the original source through ``compileMappedKernel``. No source/mapped candidate
provider is wired into the shipped CLI.

Compatibility binding synthesizes two contractions plus SiLU/multiply when the
workload label is exactly ``fused_swiglu``; other labels use a single-contraction
template. Renaming the workload does not preserve arbitrary source semantics.
The compatibility emitter uses the input dtype for weight tensors even though
``--weight-dtype`` is supplied separately to validation. Do not treat mixed input
and weight dtype predictions as an implemented mixed-dtype executable contract.

Measurement occurs after static ranking and top-K selection. It can annotate
or reject those candidates, but it does not rerank or backfill from candidates
that were truncated. ``buildScheduleRecords`` currently does not transfer measured
observations or their identities to schedule records, so its output remains
``measured: false``. The presence of measurement fields in the record type does
not establish a production measured-record database.

The mapping subsystem also has a separate optional ``LatencyProvider``, with
operation/connection signatures and target content identity, in
:source:`include/LLK/Mapping/LatencyProvider.h <include/LLK/Mapping/LatencyProvider.h>`.
It is an integration API for static-cost replacement on matching observations;
it is not a persistent calibration store supplied by ``llk-tune``. Production
measurement storage, calibration fitting, and prediction validation remain
#51/#52 work.

.. _tools-tuning-reproduce-and-troubleshoot-a-run:

Reproduce and troubleshoot a run
--------------------------------

For reproducibility, record the compiler commit and build, exact input and
profile contents, M/N/K and dtypes, workload override, objective, perf level,
search mode, seed, cap, top-K, stdout, and resulting YAML. Keep the machine path
spelling fixed when checking byte-identical YAML, because the writer records
that path. Candidate IDs hash the complete integer/symbolic bindings; they are
not hashes of the source program, target package, machine, or executable ABI.

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Symptom
     - Interpretation and next step
   * - “Unknown command line argument '--search-space'”
     - Use ``--input``.
   * - “module has no micro.search_space”
     - Supply a space rather than only concrete kernel/source IR.
   * - “module has more than one micro.search_space”
     - Put the space you want in a separate file; the CLI has no symbol-selection option.
   * - “unsupported --search”
     - Use ``grid`` or ``random``.
   * - “unsupported --perf-level”
     - Use numeric ``0`` or ``1``.
   * - Zero ranked, nonzero rejected
     - Check declared constraints, mandatory tile bindings, shapes/dtypes, and machine support. Use a known-good fixture first; inspect the C++ report for individual reasons.
   * - Run takes too long
     - Set a nonzero candidate cap and small explicit shape. The cap limits candidates, not the number of expanded events per candidate.
   * - Output cannot be opened/written
     - Create the parent directory and use a writable file path; the tool does not create directories.
   * - Unexpected JSON
     - ``--input`` was absent or empty. ``--output`` alone does not select Micro mode.
   * - Empty YAML with status 0
     - No candidate was accepted; success means the session/output operation completed.

Usage, input, model, parse, objective, or output errors return status 1. Candidate
rejections alone do not make the process fail. Review counts and records before
using a result as a schedule recommendation.

Contributor sources are
:source:`tools/llk-tune/llk-tune.cpp <tools/llk-tune/llk-tune.cpp>`,
:source:`lib/Perf/TuningSession.cpp <lib/Perf/TuningSession.cpp>`,
:source:`lib/Perf/CandidateGenerator.cpp <lib/Perf/CandidateGenerator.cpp>`,
:source:`lib/Perf/CandidateBinding.cpp <lib/Perf/CandidateBinding.cpp>`, and
:source:`lib/Perf/ScheduleRecord.cpp <lib/Perf/ScheduleRecord.cpp>`.
The CLI fixture is
:source:`test/Perf/llk_tune_search_space.mlir <test/Perf/llk_tune_search_space.mlir>`.
