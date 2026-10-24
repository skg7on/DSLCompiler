.. _tools-compiler-llk-compile-export-and-compile-kernels:

``llk-compile``: export and compile kernels
===========================================

``llk-compile`` reads an MLIR module and either exports MicroIR, runs the legacy CPU lowering pipeline, or maps a concrete Micro kernel through a registered target package. It is a compilation and inspection driver: successful JIT compilation does **not** invoke the kernel or check its numerical result.

Start with the :doc:`first-kernel tutorial </tutorials/01-first-kernel>` and :doc:`mapping tutorial </tutorials/02-mapping>`. All commands below run from the repository root after the :doc:`build instructions </getting-started>`; ``build/`` is the chosen build directory. See the :doc:`documentation index </index>` for the other tools.

.. _tools-compiler-invocation-and-outputs:

Invocation and outputs
----------------------

.. code-block:: sh

   build/bin/llk-compile [options] input.mlir
   build/bin/llk-compile --help
   mkdir -p build/tutorial

The input filename is required. Textual output goes to stdout and diagnostics go to stderr. Use shell redirection to save an artifact; this driver has no ``-o`` output option. Errors return a nonzero exit status.

Mapped compilation also writes a ``capabilities:`` JSON receipt to stderr.
Receipt version 2 validates source, target and plan provenance before crediting
analysis or compilation evidence. Bound kernels use the original-source
projection, so inserted movement does not change that source identity.
Selected lowering establishes ``selected_lowering`` only; a separate reference
compilation establishes ``reference_lowering``.

The runtime fields distinguish four facts: ``abi_eligibility`` checks a supplied
workload contract against ABI v1 (an absent contract stays ``not_checked``),
``abi_preparation`` records successful callable preparation,
``host_compatibility`` records successful executable creation on the current
host, and ``invocation`` requires a call that returned successfully. This driver
does not invoke kernels, so successful executable compilation leaves
``invocation`` as ``not_checked``. A successful call alone does not establish
numerical correctness.
Eligibility requires positive static extents. A supplied contract that fits the
ABI envelope must also match the source signature and the executable's ordered
input/output shapes and dtypes before preparation or invocation is credited.

Without ``--mapping-target``, ``--emit`` chooses the path:

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Value
     - Default?
     - Work performed
     - Successful stdout
   * - ``llvm``
     - Yes
     - Full legacy lowering, LLVM IR translation, native ORC JIT compilation, entry-symbol lookup
     - ``Compilation successful``
   * - ``mlir``
     - No
     - Full legacy lowering, stopping before JIT compilation
     - A module in the **MLIR LLVM dialect**, with operations such as ``llvm.func``
   * - ``micro``
     - No
     - Concrete LLK-to-Micro export only
     - Original source functions plus generated ``micro.kernel`` operations
   * - ``micro-search``
     - No
     - LLK-to-Micro search-space export only
     - Original source functions plus ``micro.search_space``, parameters, constraints, and objectives

``--emit=llvm`` does not print LLVM assembly, emit an object file, or save a standalone executable. ``--emit=mlir`` means the fully lowered MLIR module, not an intermediate Linalg or Vector stage. Use :doc:`llk-opt </tools/optimizer>` to inspect those intermediate stages.

.. _tools-compiler-driver-options:

Driver options
--------------

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Option
     - Default
     - Meaning
   * - ``--emit=llvm\|mlir\|micro\|micro-search``
     - ``llvm``
     - Select the non-mapped path described above.
   * - ``--M=<integer>``
     - ``0``
     - Append an M value to the legacy JIT cache key. Does not resize or specialize input IR.
   * - ``--N=<integer>``
     - ``0``
     - Append an N value to the legacy JIT cache key. Does not change the tensor shapes.
   * - ``--K=<integer>``
     - ``0``
     - Append a K value to the legacy JIT cache key. Does not change the contraction dimension.
   * - ``--mapping-target=<name>``
     - Empty
     - Use a registered mapping package instead of the legacy/export path. Registered names are ``x86-avx2``, ``generic-ai-accel``, ``arm-sve`` and ``arm-sme``. The Arm packages are static mapping/analysis targets: they map, verify and analyze, but selected-target compilation is rejected.
   * - ``--mapping-root=<directory>``
     - ``.``
     - Root containing the package's ``machines/`` and ``mapping/`` directories.
   * - ``--mapping-mode=deterministic\|beam\|exact``
     - ``beam``
     - Choose the mapping search mode when searching a new plan.
   * - ``--mapping-stop=mapped-micro\|target-lowered\|lowered\|executable``
     - ``executable``
     - Choose the mapped compilation stopping point.
   * - ``--mapping-entry=<symbol>``
     - Empty
     - Entry symbol to compile, without ``@``. Normally omit it for a module with one kernel.
   * - ``--plan-report=<path>``
     - Empty
     - Read a frozen JSON selection instead of searching. Requires the mapping path.
   * - ``--source-adapter=<key>``
     - Empty
     - Assert the source adapter that produced the artifact. Empty accepts the adapter recorded in the frozen source. Requires the mapping path.

The ``--M``, ``--N``, and ``--K`` options affect only the legacy JIT cache key. Shapes in the MLIR types drive export and compilation. They do not affect Micro export, mapped search, or mapped compilation. The corresponding shape options on :doc:`llk-tune </tools/tuning>` and :doc:`llk-bench </tools/benchmark>` have different contracts.

When a nonempty ``--mapping-target`` is supplied, the mapping path takes precedence over every **valid** ``--emit`` value. Use ``--mapping-stop`` to select its output. Invalid ``--emit`` values still fail option validation before mapping starts.

.. _tools-compiler-export-a-concrete-kernel-and-a-search-space:

Export a concrete kernel and a search space
-------------------------------------------

The tutorial sources use static BF16 matrices and FP32 accumulation. The matmul source exports one kernel; SwiGLU exports two projection MMAs followed by the SiLU/multiply/conversion epilogue.

.. code-block:: sh

   build/bin/llk-compile --emit=micro docs/manual/source/examples/matmul.mlir \
     > build/tutorial/matmul.micro.mlir

   build/bin/llk-compile --emit=micro-search docs/manual/source/examples/swiglu.mlir \
     > build/tutorial/swiglu.search.mlir

   build/bin/llk-opt build/tutorial/matmul.micro.mlir -o build/tutorial/matmul.roundtrip.mlir

The first artifact contains ``micro.kernel @matmul_M16_N64_K64``. The second contains a search space around the schedule selected for the SwiGLU workload. Neither command runs a mapping search, lowers to machine code, or invokes a kernel. The original ``func.func`` remains beside the export, so seeing ``llk.*`` in the module is expected; the generated concrete kernel itself contains Micro execution operations.

Both exports read ``schedules/schedule_db.json`` relative to the current directory. An unreadable database or a missing matching entry selects the built-in conservative schedule; the optimizer export also surfaces a warning. That fallback is useful for experiments, but record whether it was used when comparing results. To choose a different database explicitly, use ``llk-opt --llk-to-micro="schedule-db=<path>"`` or ``--llk-to-micro-search-space="schedule-db=<path>"``; ``llk-compile`` does not expose a driver-level schedule-database option.

The current shared selector filters entries by operation and M bucket, then chooses the first match. It does not select by exact N/K, target, dtype, or math mode even though the database records those fields. The M buckets are ``{1}``, ``[2,4]``, ``[5,16]``, ``[17,64]``, and ``≥65``. The no-match fallback uses ``BM=8``, ``BN=32``, ``BK=32``, ``VM=1``, ``VN=4``, vector width 8, four threads, and grain size 1; resolved tiles can be clamped to smaller problem extents.

Current export support is narrower than the set of operations the parser accepts:

* Roots are ``llk.matmul`` and ``llk.fused_swiglu``; SwiGLU activation must be ``silu``.

* Operands and result must have static rank-two shapes. Matrix dimensions must agree: ``[M,K] × [K,N] → [M,N]``; SwiGLU's second weight must also be ``[K,N]``.

* Supported Micro tile element types are ``f32``, ``f16``, ``bf16``, ``i32``, and ``i8``; the accumulator attribute must name a Micro tile dtype. Parsing/export support does not imply every target has an executable rule for that dtype.

* The selected tile schedule must be realizable by the exporter. A non-divisible tile needs a supported tail policy; layout kinds requiring unresolved block/swizzle parameters are rejected.

* Each supported root adds its own export. The mapping front doors expect one concrete kernel, so use a module with one supported root for the workflows below.

* A file with no supported root can pass the export without adding a kernel or search space. Check the expected operation in the output before moving on.

``--emit=micro-search`` describes choices; it does not automatically choose a ``micro.candidate``, instantiate every candidate, or map candidates to target rules. See the :doc:`tuning manual </tools/tuning>` and the optimizer's :ref:`candidate= workflow <tools-optimizer-map-at-an-explicit-candidate>`.

.. _tools-compiler-legacy-cpu-compilation:

Legacy CPU compilation
----------------------

Use the SwiGLU source for this path:

.. code-block:: sh

   build/bin/llk-compile --emit=mlir docs/manual/source/examples/swiglu.mlir \
     > build/tutorial/swiglu.llvm.mlir

   build/bin/llk-compile docs/manual/source/examples/swiglu.mlir

The first command saves MLIR LLVM-dialect IR. The second prints ``Compilation successful`` after the native JIT has compiled an exported function. The JIT and its cache live only for this process; the command does not leave a callable executable file behind.

The fixed legacy pipeline is:

#. For inputs that reach it with ``tt.*`` operations, Triton structured-compute and grid lowering.

#. LLK-to-Linalg lowering and canonicalization.

#. Shape classification and schedule-selection pass, double-contraction fusion, canonicalization, weight-packing annotation.

#. Tiling/vectorization, canonicalization, forall linearization, and serial/parallel dispatch selection.

#. One-Shot Bufferize, scratch-allocation audit, remaining Linalg-to-loop conversion, and forall-to-LLRT runtime lowering.

#. Vector, SCF, Arith, Math, UB, Func, and MemRef conversion to LLVM dialect; unrealized-cast reconciliation.

#. For ``--emit=llvm``, LLVM IR translation and native ORC JIT compilation.

The legacy LLK-to-Linalg pass currently lowers SwiGLU, RoPE, and Attention. It **does not lower ``llk.matmul``**. The tutorial matmul therefore works for Micro export and mapped compilation, while a legacy ``llk-compile docs/manual/source/examples/matmul.mlir`` currently fails with ``op was not bufferized``. Start from a ``linalg.matmul`` program or use the mapped path for matmul.

There is no ``--scalar``, ``--vector``, tile-size, thread-count, or driver-level ISA selector for this fixed pipeline. ``TileAndVectorize`` currently uses fixed values ``BM=32``, ``BN=64``, ``BK=64``, ``VM=4``, ``VN=8``. Its ``schedule-file`` and ``target-isa`` pass options are exposed by ``llk-opt``, but the implementation does not consume them. The legacy ``select-schedule`` pass reports a selected entry; it does not feed these fixed tile constants. The Micro export path does consume its selected schedule. Use the optimizer's :ref:`scalar and vector inspection examples <tools-optimizer-inspect-scalar-and-vector-lowering>` to compare IR stages.

The parser registry is also intentionally narrower than ``llk-opt``'s all-dialects registry. The compiler's Triton detection is a lowering hook, not a general promise to parse arbitrary Triton dumps or GPU programs. Use the explicit staged Triton passes in ``llk-opt`` for that work.

.. _tools-compiler-map-through-a-registered-package:

Map through a registered package
--------------------------------

.. code-block:: sh

   build/bin/llk-compile --mapping-target=x86-avx2 --mapping-root=. \
     --mapping-mode=exact --mapping-stop=mapped-micro \
     docs/manual/source/examples/matmul.mlir > build/tutorial/matmul.mapped.log

   sed '1d' build/tutorial/matmul.mapped.log > build/tutorial/matmul.mapped.mlir
   build/bin/llk-opt build/tutorial/matmul.mapped.mlir -o build/tutorial/matmul.mapped.roundtrip.mlir

The compiler exports a concrete kernel if the input does not already contain one, extracts its workload graph, selects a covering plan, binds and verifies it, and stops at the requested stage. When it exports, it removes the LLK source function from the mapped compilation module. It uses the first selected plan returned by the search.

Every successful mapped invocation prints a status line before any IR:

.. code-block:: text

   mapping: target=x86-avx2 target-lowered-ops=0 reference-lowered-ops=0

The status line makes raw compiler stdout unsuitable as an MLIR file until it is removed. The ``sed`` command above removes that first line. For a pipeline that should produce directly parseable IR and a separate report, use ``llk-opt --micro-map`` instead.

The mapped stages are:

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - ``--mapping-stop``
     - Result after binding and target verification
   * - ``mapped-micro``
     - Prints the bound ``micro.kernel`` and selected mapping metadata. No emitters have run, so both counts are zero.
   * - ``target-lowered``
     - Runs selected target emitter hooks and prints their rewritten Micro form. Verification-only emitters leave operations for the reference bridge.
   * - ``lowered``
     - Applies Micro-to-Linalg, One-Shot Bufferize with identity function-boundary layouts, and Linalg-to-loops. Prints SCF/MemRef/backend input IR.
   * - ``executable``
     - Applies the descriptor calling convention and native JIT compilation; prints the counts and ``Compilation successful``.

For example:

.. code-block:: sh

   build/bin/llk-compile --mapping-target=x86-avx2 --mapping-root=. \
     --mapping-mode=exact --mapping-stop=lowered \
     docs/manual/source/examples/matmul.mlir > build/tutorial/matmul.lowered.log

   build/bin/llk-compile --mapping-target=x86-avx2 --mapping-root=. \
     --mapping-mode=exact docs/manual/source/examples/matmul.mlir

Mapped lowering reports verified selected groups, backend realization and reference
lowering separately. ``--mapping-backend=reference`` provides portable correctness;
``--mapping-backend=selected-target`` preserves selected vector widths/fused groups,
checks x86_64 AVX2 compatibility and rejects reference fallback. Dispatch counts
alone do not prove numerical execution. PR #133's selected numerical gate and
exact LLVM/plan artifacts are recorded in the
:source:`acceptance matrix <docs/reviews/issue67-final-acceptance.md>`.

``--mapping-root`` makes the package load these conventional paths:

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Package
     - Machine profile
     - Mapping policy
   * - ``x86-avx2``
     - ``machines/x86-avx2-v2.yaml``
     - :source:`layouts.llkmap <mapping/x86-avx2/layouts.llkmap>`, :source:`rules.llkmap <mapping/x86-avx2/rules.llkmap>`
   * - ``generic-ai-accel``
     - ``machines/generic-ai-accel-v2.yaml``
     - :source:`layouts.llkmap <mapping/generic-ai-accel/layouts.llkmap>`, :source:`rules.llkmap <mapping/generic-ai-accel/rules.llkmap>`
   * - ``arm-sve``
     - ``machines/arm-sve-v2.yaml``
     - :source:`layouts.llkmap <mapping/arm-sve/layouts.llkmap>`, :source:`rules.llkmap <mapping/arm-sve/rules.llkmap>`
   * - ``arm-sme``
     - ``machines/arm-sme-v2.yaml``
     - :source:`layouts.llkmap <mapping/arm-sme/layouts.llkmap>`, :source:`rules.llkmap <mapping/arm-sme/rules.llkmap>`

The package supplies its emitter keys in code. The compiler does not accept the optimizer's individual ``machine=``, ``layouts=``, ``rules=``, or ``emitters=`` pass options. Use :doc:`llk-opt </tools/optimizer>` when experimenting with policy files. The accelerator's shipped rules cover a smaller operation set; do not assume it can map the tile-level matmul/SwiGLU export or execute accelerator hardware. The Arm packages are static mapping and analysis targets: they accept direct-MicroIR fixtures, verify and report complete plans at modeled VL=SVL=512, but their emitters refuse lowering and selected-target compilation fails. See the :source:`Arm acceptance ledger <docs/reviews/arm-mapping-targets-acceptance.md>` for the exact domain.

Mapped compilation uses an executable binding contract by default. It rejects decisions the binder cannot materialize; the driver has no partial-binding toggle. It preserves concrete selected resources and rejects unresolved storage, scheduling or backend decisions. Use the optimizer's report-only mode for analysis without binding.

.. _tools-compiler-replay-a-saved-selection:

Replay a saved selection
------------------------

The compiler reads a report; it does not write one. Produce a versioned JSON report with :ref:`llk-opt --micro-map report=... <tools-optimizer-save-an-analysis-report-without-binding>`, then compile the **original concrete Micro input** it searched:

.. code-block:: sh

   # Files created by the optimizer manual's report-only example:
   build/bin/llk-compile --mapping-target=x86-avx2 --mapping-root=. \
     --plan-report=build/tutorial/matmul.plan.json \
     --mapping-stop=mapped-micro build/tutorial/matmul.micro.mlir \
     > build/tutorial/matmul.replayed.log

This reuses the report's frozen ``selectedState`` instead of re-running search; ``--mapping-mode`` is not used. The report version, target-content hash, and original workload-graph hash must match. Changing the source workload or target files can invalidate a report. Replaying against already-bound/materialized IR can change the graph, so retain the pre-mapping input along with the report.

``llk-opt --micro-bind-plan`` is a different interface: it reads a plan **id**, re-runs the same search, and finds that id among the retained plans. It does not load a report file.

Generic source replay
---------------------

The compiler does not instantiate a semantic source by itself. Produce the
instantiated source with ``llk-tune --candidate-artifacts`` (see the
:doc:`tuner manual </tools/tuning>`), then replay it exactly like a contraction
candidate: the frozen ``source.mlir`` already holds the realized ``micro.kernel``
and its retained ``micro.workload_contract`` and ``source_adapter`` attributes,
and ``plan.json`` holds the selected mapping state.

.. code-block:: sh

   build/bin/llk-compile --mapping-target=x86-avx2 --mapping-root=. \
     --machine=machines/x86-avx2-v2.yaml --mapping-backend=reference \
     --mapping-stop=lowered --source-adapter=linalg-elementwise-v1 \
     --plan-report=build/tutorial/elementwise-candidates/<candidate>/plan.json \
     build/tutorial/elementwise-candidates/<candidate>/source.mlir --emit=mlir

The status line reports ``source-adapter=``, ``source-contract-version=`` and
``replay-identity=`` next to the backend identity. The replay identity is the
same ``graph=<hash>|binding=<hash>|plan=<id>`` value the tuner retained, so a
matching value is evidence that source, binding and plan are the frozen ones.
A changed source attribute or port extent changes the workload graph and is
rejected by the plan report's graph hash; changed layout/rule content changes the
target hash; changed machine content changes the machine/target hash; and a
different execution mode (``reference`` versus ``selected-target``) produces a
different ``identity=``. None of those cases silently reuse the previous
artifact.

A ``capabilities:`` JSON object is printed on stderr after the status line. It
reports structure, interpretation, mapping, physical plan, reference lowering,
selected lowering and invocation as ``supported``, ``unsupported`` or
``not_checked`` for this exact run, keyed by source/contract/target/plan
identity. Diagnostics go to stderr so stdout remains "mapping line + MLIR" and
can be re-parsed as an artifact. A reference-backend run leaves selected
lowering ``not_checked``: the absence of a codegen contract means the stage was
not attempted, not that it failed.

The mapped ABI v1 is a rank-two, compact, offset-zero descriptor envelope with
at most 12 ports and read-only inputs. Analysis, reporting and lowering are
rank-neutral; only executable preparation applies that envelope. A rank-three
contract or a rank-one reduction can therefore reach lowered IR, while
``--mapping-stop=executable`` fails explicitly with a ``runtime_capability:``
diagnostic naming the rank, dtype, descriptor count or port effect it cannot
express. Ports whose contract declares unknown or dynamic extents are rejected
earlier as contract errors, not as execution-capability errors.

.. _tools-compiler-runtime-boundary:

Runtime boundary
----------------

The CLI provides neither input buffers nor an invocation command. To run a mapped kernel from C++, use the executable returned by ``compileMappedKernel`` and ``MappedExecutable::invoke``; see :source:`MappedExecutable.h <include/LLK/Runtime/MappedExecutable.h>` and the numerical examples in :source:`mapped_acceptance.cpp <test/Execution/mapped_acceptance.cpp>`.

The mapped API records declared ports before type erasure and invokes them in
port order. Prefer ``InvocationBuffer2D``: it carries a ``MemRef2D`` descriptor,
dtype and backing allocation bytes. Typed invocation checks arity, rank-two
shape, dtype, supported strides, offset/bounds and unsafe aliasing before calling
machine code. Caller buffers remain caller-owned; compiler scratch is released
per invocation. Unsupported ranks/dtypes are rejected before publishing an
executable. ``Tensor2D``, ``MemRef2D`` and ``KernelContext`` are distinct structures.

The deprecated untagged ``MemRef2D*`` overload checks arity/shape/stride but cannot
validate runtime dtype or backing allocation size. Do not treat it as equivalent
to typed invocation. The Micro matmul exporter has A/B inputs and one result;
its original ``%init`` is not an input in that exported ABI. Candidate semantic
instantiation can have a different ABI; read the executable's ``KernelAbi``
rather than inferring its ports from the original function.

.. _tools-compiler-troubleshooting:

Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Symptom
     - What to check
   * - ``Failed to parse ...`` or unknown operation/dialect
     - Validate syntax with ``llk-opt``; the compiler registers a narrower set of dialects. Confirm the file is textual MLIR accepted by the intended path.
   * - Micro output contains no kernel
     - The exporter found no ``llk.matmul``/``llk.fused_swiglu`` root. Export success can be a no-op.
   * - Dynamic-shape or rank diagnostic
     - Edit tensor types to supported static rank-two shapes. ``--M/--N/--K`` cannot repair the IR.
   * - Conservative-schedule warning
     - Check the current directory and ``schedules/schedule_db.json``. Use the optimizer export option for an explicit database.
   * - ``op was not bufferized`` for ``llk.matmul``
     - Use Micro export plus mapping, or provide an already-structured Linalg matmul for the legacy path.
   * - Mapping cannot load files
     - Run from the repository root or set ``--mapping-root`` to the directory containing both ``machines/`` and ``mapping/``.
   * - No complete plan
     - Check operation/dtype support and inspect an optimizer mapping report. Beam/candidate caps can limit the search; exact mode still has finite internal caps and reports truncation.
   * - No selected vector width / fused emitter failure
     - Inspect the selected rule and bundle. The shipped fused ``convert → silu → mul`` chain is verified. Inspect unsupported emitter/effect/region diagnostics and use the intended backend policy.
   * - ``runtime_capability:`` rank/dtype/effect diagnostic
     - The selected workload or reference program is valid, but the mapped ABI v1 cannot call it: it requires rank-two static ports, a supported dtype, at most 12 descriptors, read-only inputs and write-only outputs. Use a supported shape for execution, or stop at ``--mapping-stop=lowered`` for analysis/lowering evidence.
   * - Report hash/version mismatch
     - Use the original pre-mapping Micro input and the exact target configuration that produced the report; regenerate after intentional changes.
   * - Saved mapped output does not parse
     - Remove the leading ``mapping:`` line, or obtain pure MLIR output through ``llk-opt``.
   * - JIT symbol/translation error
     - First inspect ``--emit=mlir`` or ``--mapping-stop=lowered``; compilation requires a valid exported entry and all operations to reach the backend.

For cost estimates use :doc:`micro-perf </tools/performance>`. For candidate ranking use :doc:`llk-tune </tools/tuning>`. :doc:`llk-bench </tools/benchmark>` runs the current synthetic host harness; it does not time this compiled kernel. For verified candidate timing, use target-aware ``llk-tune --measure-top`` with a typed ``--measurement-inputs`` fixture, or a C++ caller that checks and times invocations.
