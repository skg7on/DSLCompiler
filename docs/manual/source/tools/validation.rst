.. _tools-validation-prediction-validation:

Prediction validation
======================

``llk-validate`` compares static whole-kernel predictions with verified measurement
records from :doc:`llk-tune <tuning>` or the mapped measurement API. It reads records
and writes a report; it fits no model and changes no machine profile or search
score. Build it with ``cmake --build build --target llk-validate``.

.. code-block:: bash

   build/bin/llk-validate measurements/ --format=json --output=validation.json
   build/bin/llk-validate record-a.json record-b.json --format=text

Inputs are explicit record files or directories. Directory discovery is
nonrecursive and reads regular ``*.json`` files in sorted order, ignoring store
temporaries. Keep reports and manifests outside record directories. Every record
is strictly decoded and validated, including records outside a selected holdout.
Corrupt records, duplicate input paths/record identities and nonfinite error
arithmetic are errors. Reports cannot overwrite input records or manifests,
including aliases to the same file.

Comparable groups and statistics
--------------------------------

Each group has the complete target, backend, compiler, environment, protocol,
verification and workload/decision metadata. Full CPU/ISA/thread/affinity/clock
identity matters; selected and reference backends never share a statistic. The
adapter, workload contract, candidate bindings and bottleneck also distinguish
groups. Dtype and shape are retained in the full opaque workload contract rather
than guessed from an operator name. Full fixture content and verifier/numerical
identity are compared conservatively, so changing input data or tolerances also
creates a separate group.

For prediction ``p`` and verified median measured latency ``m``:

.. code-block:: text

   absolute_error_ns       = abs(p - m)
   signed_relative_error   = (p - m) / m
   absolute_relative_error = abs((p - m) / m)

Rows preserve input order within each group. Median absolute relative error uses
the middle value, or the mean of the two middle values. The p90 uses nearest rank:
``ceil(0.9 * n) - 1`` in the sorted error array. A 100 ns prediction against 80 ns
and 200 ns measurements yields errors 0.25 and 0.5, median 0.375 and p90 0.5.

Missing/zero ``prediction.predicted_ns`` means ``no_prediction_time_basis``.
The tool never converts cycles without an available clock or treats unavailable
predictions as zero. Excluded-only groups have ``status=no_comparable_samples``
and null summary values. An empty input directory has no groups and the same
status. There is no pooled cross-group accuracy number. ``sample_count`` counts
raw measured repeats, including excluded records; ``compared_record_count``
counts the records that contribute an error. Raw repeats are not independent
prediction observations. Prediction origin is ``predicted_static`` and measurement
origin is ``verified_measurement_record``; kernel latency never becomes an
event-level measurement.

Held-out evaluation
--------------------

Without a manifest the report says ``held_out=false``: it is an observational
comparison, not evidence of held-out accuracy. Supply a version-1 JSON manifest
containing complete executable identity strings copied from the records:

.. code-block:: json

   {"version": 1, "identities": ["complete executable identity from a record"]}

.. code-block:: bash

   build/bin/llk-validate measurements/ --holdout-manifest=holdout.json \
     --training-manifest=training.json --calibration-manifest=calibration.json \
     --output=heldout-validation.json

Training and calibration options are repeatable and require an explicit holdout.
Duplicate manifest identities and any holdout/training/calibration overlap are
errors, even if that identity has no supplied record. Every holdout identity must
have a supplied record. Non-holdout records are validated but do not contribute
statistics; ``not_in_holdout_count`` makes this selection visible. Short content
hashes cannot stand in for full identities. No training or calibration occurs.

Baseline and optional device corpus
-----------------------------------

The :source:`uncalibrated arithmetic baseline <docs/reviews/issue164-prediction-baseline.json>`
is generated from checked-in synthetic records: it verifies report arithmetic,
provenance, counts and exclusions, and makes no hardware-accuracy claim.
The :source:`Phase D acceptance review <docs/reviews/issue164-prediction-validation.md>`
records the implementation and tested boundary.

For a device baseline, collect fresh verified records on a compatible host using
``--measurement-store`` with reuse off. Use the original contraction/fusion sources,
the bounded elementwise adapter, and a direct MicroIR movement fixture through the
contract-based mapped measurement API. The latter can describe movement without
inventing contraction M/N/K; the current CLI's concrete-Micro legacy path still
requires contraction provenance. Keep independently measured movement records in
their own store and retain the full source/plan identity in each record.

With those four explicit stores and a disjoint full-identity holdout manifest, run:

.. code-block:: bash

   build/bin/llk-validate corpus/contraction/ corpus/fusion/ \
     corpus/elementwise/ corpus/movement/ --holdout-manifest=corpus/holdout.json \
     --training-manifest=corpus/training.json --output=uncalibrated-device-baseline.json

Keep selected-target and reference records labeled by their actual backend. The
report partitions them even when supplied together. This optional command reads
an existing corpus; it does not collect timings or create missing movement
measurements. Record actual clock/affinity policies and sample counts. No error
threshold is justified until enough independent device observations establish a
baseline. Event attribution, fitting, confidence estimates and calibration
acceptance thresholds require a separate design.

The ``PredictionValidationMappedCLI`` E2E test also creates a small real baseline
under ``build/prediction-validation-evidence/``. CI publishes its records, fixture,
manifest and report in the ``selected-avx2-execution-evidence`` artifact. This
single elementwise observation validates the end-to-end reporting path; it is
insufficient to establish general accuracy or an error threshold.
