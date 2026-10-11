.. _tools-tuning-llk-tune-enumerate-and-rank-schedules:

``llk-tune``: enumerate and rank schedules
==========================================

PR #133 adds target-aware tuning from original source alongside the existing
compatibility and legacy workflows. Use the mode that matches the result you
need; static predictions and measured observations are separate outputs.

Choose a workflow
-----------------

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Workflow
     - Selector
     - Result
   * - Target-aware source tuning
     - ``--input`` plus ``--mapping-target`` and explicit ``--candidate-source``
     - Instantiated original source, verified mapped plan, static ranking, schedule YAML and optional report/replay artifacts
   * - Compatibility tuning
     - ``--input`` without ``--mapping-target``
     - Synthetic workload binding and predicted schedule YAML; does not preserve arbitrary original-source semantics
   * - Legacy shape grid
     - No ``--input``
     - Existing JSON schedule entries; no measured winner is established by the zero GFLOPS scores

``--input`` is the search-space file option; ``--search-space`` is not an alias.
Keep exactly one search space in the input. ``--search=grid`` or ``random`` chooses
parameter generation, while ``--mapping-mode`` chooses covering search.

Original-source tuning and replay
---------------------------------

Run from the worktree root with built tools:

.. code-block:: bash

   mkdir -p build/tutorial
   build/bin/llk-tune --input=test/Tuning/Inputs/issue129/mapped_tune.mlir \
     --machine=machines/x86-avx2-v2.yaml -M=8 -N=16 -K=32 \
     --mapping-target=x86-avx2 --mapping-root=. --mapping-mode=exact \
     --mapping-backend=reference --candidate-source=semantic --source-symbol=matmul \
     --top-k=1 --mapping-report=build/tutorial/tune.json \
     --candidate-artifacts=build/tutorial/candidates \
     --output=build/tutorial/matmul.schedule.yaml

The fixture retains the semantic function and its search space. Each binding
instantiates that source, selects a mapping and evaluates the materialized plan.
The JSON report includes complete bindings, plan identity/report, metric origins,
rejection counts and search truncation. Candidate directories contain
``source.mlir`` and ``plan.json``. The current fixture's deterministic candidate
ID is ``candidate_e4c6c3f6abcfbc65``; derive IDs from the report for other spaces.

Replay against the exact instantiated source and machine files:

.. code-block:: bash

   build/bin/llk-compile --mapping-target=x86-avx2 --mapping-root=. \
     --machine=machines/x86-avx2-v2.yaml --mapping-backend=reference \
     --mapping-stop=lowered \
     --plan-report=build/tutorial/candidates/candidate_e4c6c3f6abcfbc65/plan.json \
     build/tutorial/candidates/candidate_e4c6c3f6abcfbc65/source.mlir --emit=mlir

Frozen replay checks content/version/workload identity; replaying a different
source or already-materialized input is not equivalent. ``--candidate-source``
also accepts ``concrete-micro`` and explicit ``synthetic`` compatibility mode.
``--source-symbol`` identifies the semantic source root.

Backend and measurement
-----------------------

The example selects portable ``reference`` execution. For actual selected AVX2,
use ``--mapping-backend=selected-target`` on a compatible x86_64 AVX2 host.
An incompatible host or unsupported realization is rejected rather than counted
as successful selected execution.

To compile, check outputs and measure a top candidate, append:

.. code-block:: bash

   # Append these options to the target-aware tuning command above:
   --measure-top=1 \
   --measurement-inputs=test/Tuning/Inputs/issue129/measurement_inputs.json

The version-1 fixture declares ordered inputs/outputs with dtype, rank-two shape
and data (scalar fill or element array), output tolerances, warmups and repeats.
Ports are validated against the compiled ``KernelAbi``. Output correctness is
checked before accepting the median timed result. A mismatched fixture rejects
the candidate; its shape/ports must match that particular instantiated source.

Observations are written to ``<mapping-report>.measurements.json`` (or beside
schedule output when no report is requested). They carry source/binding/plan,
target/machine, ABI and execution identities. The static JSON report remains
byte-identical to the same static-only run. This verified timing path is not a
production measurement store, calibration fitter or prediction validator;
those remain #51/#52.

Options and ranking
-------------------

* ``--mapping-target`` selects a registered target package; ``--mapping-root``
  contains its ``machines/`` and ``mapping/`` files.
* ``--mapping-mode=deterministic|beam|exact`` selects covering search. Exact mode
  is bounded and reports caps; beam is heuristic.
* ``--search=grid|random``, ``--seed`` and ``--max-candidates`` control generation.
  Zero candidate cap requests the finite domain; start with a small cap.
* ``--top-k`` limits retained records; ``--perf-level=0|1`` selects static analysis
  in compatibility mode. Target-aware mode evaluates the complete selected plan.
* Explicit ``-M``, ``-N``, ``-K`` and dtype options must agree with the chosen source
  and binding contract; the semantic source is not replaced by the workload label.

The objective comes from ``micro.objective``. Unknown metrics are diagnosed;
missing measurement metrics are unavailable, not zero-cost winners. Reports
retain metric origins and rejection reasons. Static cycles/traffic parity is
verified separately from calibrated prediction. An empty accepted set can still
produce a successful session/output operation; inspect rejection counts.

Persistence and compatibility
-----------------------------

Schedule YAML remains schema 1 and is distinct from the legacy JSON
``schedules/schedule_db.json``. Use target-aware JSON reports and candidate
artifacts for complete bindings/selected plans and frozen replay; do not infer
complete mapped provenance from compatibility YAML alone.

Without ``--mapping-target``, Micro mode synthesizes compatibility workloads.
Without ``--input``, the legacy fixed shape grid writes up to three JSON entries
with zero measured GFLOPS. ``--dry-run`` previews the grid and still writes it.
Neither mode establishes the fastest executable for arbitrary original source.

.. code-block:: bash

   build/bin/llk-tune -M=8 -N=64 -K=64 --dry-run -o=build/tutorial/legacy.json

Implementation references: :source:`mapped tuning session <include/LLK/Conversion/MicroMapping/MappedTuningSession.h>`,
:source:`candidate instantiation <include/LLK/Conversion/MicroMapping/CandidateInstantiation.h>`,
:source:`CLI <tools/llk-tune/llk-tune.cpp>` and
:source:`CLI integration tests <test/Tuning/mapped_tune_cli.py>`.
