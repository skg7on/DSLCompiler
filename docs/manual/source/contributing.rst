.. _contributing-contributing:

Contributing
============

You can contribute through small kernels and reproducers, verifier tests,
mapping policies, machine profiles, performance analysis, backend work, or clearer
documentation. Start with a narrow change whose behavior can be inspected at one
IR boundary. The :doc:`feature guide </features>` lists useful extension areas and
current limitations.

.. _contributing-read-build-then-trace-one-workflow:

Read, build, then trace one workflow
------------------------------------

#. Read :doc:`concepts </concepts>`, :doc:`architecture </architecture>`, and
   :source:`CLAUDE.md <CLAUDE.md>`. For current status, consult the
   :source:`gap assessment <docs/reviews/2026-10-07-issue67-current-gap-assessment.md>` alongside
   historical completion claims.

#. Follow :doc:`getting started </getting-started>` to obtain a working toolchain.

#. Run :doc:`the first tutorial </tutorials/01-first-kernel>`, then inspect a nearby
   source file and its tests before changing behavior.

#. Pick a small problem and capture the input, command, expected result and
   observed result. A concrete fixture is easier to review than a broad request
   to improve scheduling or performance.

Use the codebase knowledge graph when available: search for a symbol, trace its
callers/callees, then read its source snippet. Fall back to text search for
non-code files, literals, or files absent from the index.

.. _contributing-work-in-an-isolated-checkout:

Work in an isolated checkout
----------------------------

The repository requires every write to use an isolated Git worktree under
``.claude/worktrees/``. The root checkout is for read-only inspection. Branches
use a category and a short kebab-case description, for example
``fix/tile-partition-verifier``. Use the docs category for documentation work.

From the repository root, an example setup is:

.. code-block:: sh

   git worktree add .claude/worktrees/tile-partition-verifier \
     -b fix/tile-partition-verifier main
   cd .claude/worktrees/tile-partition-verifier

Use an existing suitable worktree if one is already attached to your task. The
full policy is :source:`worktree-isolation.md <.claude/rules/worktree-isolation.md>`.

.. _contributing-build-and-run-the-registered-tests:

Build and run the registered tests
----------------------------------

The project uses C++20, CMake 3.20+, Ninja, LLVM/MLIR, GTest and FileCheck.
LLVM/MLIR must come from the same build. The current Linux CI pins LLVM 22.1.8;
the local reviewed build uses LLVM 24.0.0git. Those are observed validation
baselines rather than a promise that every newer or older MLIR revision works.
Build LLVM with its utilities enabled so the matching FileCheck is available.

After setting ``LLVM_BUILD`` to your existing LLVM/MLIR build directory, run from
the worktree:

.. code-block:: sh

   cmake -S . -B build -G Ninja \
     -DLLVM_PROJECT_BUILD_DIR="$LLVM_BUILD" \
     -DCMAKE_BUILD_TYPE=Release \
     -DLLK_BUILD_TOOLS=ON -DLLK_BUILD_E2E_TESTS=ON
   cmake --build build -j6
   cmake --build build --target check-llk

``check-llk`` builds test binaries and invokes the registered CTest suite. To
rerun tests without building, or select one test:

.. code-block:: sh

   ctest --test-dir build --output-on-failure
   ctest --test-dir build -R '^MicroDialectTileOps$' --output-on-failure

Inspect skipped tests and their reasons. A passing portable mapped test does not
prove selected AVX2 execution, and an x86 host alone does not prove a skipped
AVX2 test ran. For failures, preserve the exact toolchain revision, host, command,
input and output. See :source:`CI <.github/workflows/ci.yml>` and
:source:`CMakeLists.txt <CMakeLists.txt>` for the actual configuration and registrations.

.. _contributing-find-the-right-extension-point:

Find the right extension point
------------------------------

.. list-table::
   :header-rows: 1
   :class: wide-table

   * - Change
     - Main locations
     - Useful test area
   * - Canonical tile/search operation
     - :source:`Micro definitions <include/LLK/Dialect/Micro>`, :source:`dialect implementation <lib/Dialect/Micro>`
     - :source:`Micro dialect tests <test/Dialect/Micro>`
   * - Domain semantics or export
     - :source:`LLK definitions <include/LLK/Dialect>`, :source:`conversions <lib/Conversion>`
     - :source:`conversion tests <test/Conversion>`
   * - Schedule/CPU transform
     - :source:`transforms <lib/Transforms>`, :source:`transform interfaces <include/LLK/Transforms>`
     - :source:`transform tests <test/Transforms>`
   * - Generic mapping algorithm
     - :source:`mapping interfaces <include/LLK/Mapping>`, :source:`mapping implementation <lib/Mapping>`
     - :source:`mapping tests <test/Mapping>`
   * - Machine profile/schema
     - :source:`machine interfaces <include/LLK/Machine>`, :source:`loader <lib/Machine/MachineModelLoader.cpp>`, :source:`profiles <machines>`
     - :source:`machine tests <test/Machine>`
   * - Target mapping/lowering
     - :source:`target packages <lib/Target>`, :source:`target interfaces <include/LLK/Target>`, :source:`policy files <mapping>`
     - :source:`mapping conversion tests <test/Conversion/MicroMapping>`, :source:`execution tests <test/Execution>`
   * - Cost model or candidate tuning
     - :source:`performance interfaces <include/LLK/Perf>`, :source:`implementation <lib/Perf>`
     - :source:`performance tests <test/Perf>`, :source:`tuning session tests <test/Perf/tuning_session.cpp>`
   * - Runtime/ABI
     - :source:`runtime <runtime>`, :source:`runtime interfaces <include/LLK/Runtime>`
     - :source:`execution tests <test/Execution>`

.. _contributing-add-a-dialect-operation-or-attribute:

Add a dialect operation or attribute
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First establish why existing tensor/Linalg/Vector/memref operations cannot express
the required information. Keep domain semantics in LLK and scheduled execution
facts in Micro. Add the ODS definition, parser/printer or verifier work as needed,
then test round-tripping and invalid inputs. Adding a Micro enum also requires
its hand-written stringify/symbolize support in
:source:`MicroEnums.h <include/LLK/Dialect/Micro/MicroEnums.h>`; enum code generation is
not enabled by the current CMake setup.

.. _contributing-add-a-pass:

Add a pass
~~~~~~~~~~

Define a small input/output contract and identify which facts it preserves.
Put semantic conversions in the conversion layer and schedule transformations
in the transform layer. Register public passes/options with the relevant tool
and update its guide. Do not hard-code tile sizes in semantic lowering: consume
schedule data. Keep tensor tiling/fusion/vectorization before bufferization and
emit explicit Vector IR when the CPU backend promises SIMD behavior.

FileCheck fixtures are registered by root CMake, not a lit runner. A ``// RUN:``
line alone does not execute a test. Add the appropriate
``add_llk_filecheck_test`` entry or an explicit ``add_test`` for diagnostics/options.
Reconfigure after adding registrations.

.. _contributing-add-or-extend-a-target:

Add or extend a target
~~~~~~~~~~~~~~~~~~~~~~

Start with a topology profile and LLKMap layouts/rules, then implement only the
target-owned emitter behavior that is required. Use the shipped
:source:`AVX2 <mapping/x86-avx2>` and :source:`generic accelerator <mapping/generic-ai-accel>`
packages as examples, and consult :source:`formal syntax <docs/design/llkmap-syntax.md>`.
Owner refinements belong in target data; layout IDs and bundle names remain
opaque to canonical IR and generic algorithms.

Register a new package's factory through
:source:`MappingTargets.cpp <lib/Target/MappingTargets.cpp>` and add its sources and
tests in :source:`CMakeLists.txt <CMakeLists.txt>`. Test policy parsing, a supported
mapping, rejected capabilities, and any required movement before extending the
execution surface.

Do not add target-name branches to generic ``lib/Mapping`` or ``lib/Machine``.
Preserve ``LLKPerf → LLKMapping`` and keep JIT/runtime integration above the
generic libraries. A profile or matching rule establishes mapping support;
an executable support claim additionally needs target lowering and numerical
evidence on a compatible host.

.. _contributing-prepare-a-reviewable-change:

Prepare a reviewable change
---------------------------

For behavior changes, reproduce the problem in a focused failing test, implement
the smallest fix, and run the affected tests before the full required suite.
Test a meaningful boundary: malformed IR, rejected capability, required route,
layout conversion, replay consistency, or numerical result. Avoid tests that
only mirror a helper's implementation.

For documentation-only changes, verify relative links, referenced paths, tool
flags and runnable commands appropriate to the edit. ``DocReferences`` checks the
listed community and historical documents; add a new guide to its document list.
It checks help names and paths, while ``CommunityTutorials`` executes the example
workflows and checks their artifacts. Run both with
``ctest --test-dir build -R 'DocReferences|CommunityTutorials' --output-on-failure``.
A full LLVM rebuild is unnecessary for a prose edit.

Build the :doc:`Sphinx HTML manual </building-docs>` with warnings treated as
errors, then inspect each changed page in a browser. Add new pages to the
appropriate ``toctree`` so they appear in the sidebar and search. The manual's
build is independent of the LLVM toolchain; command checks use the actual tools.

Keep commits focused. In the PR, describe the trigger and resulting behavior,
include the verification actually run, and state any relevant limits. Record
skips separately from passes. Distinguish static predictions, portable numerical
correctness, selected-target execution and calibrated measurements.

Before proposing a #129 repair, read its
:source:`resource <docs/superpowers/plans/2026-10-07-issue129-resources-and-costs.md>`,
:source:`runtime <docs/superpowers/plans/2026-10-07-issue129-execution-and-runtime.md>`, or
:source:`tuner/acceptance <docs/superpowers/plans/2026-10-07-issue129-tuning-and-acceptance.md>`
companion plan. Those documents coordinate shared interfaces and dependencies;
their proposed APIs should not be treated as existing code.
