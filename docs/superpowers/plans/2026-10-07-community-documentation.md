# Community documentation implementation plan

**Goal:** help a developer understand MicroIR, run the current tools, and find a concrete contribution without reading the full design history first.

**Approach:** extend the existing Markdown documentation with a community entry point, concept/architecture/feature guides, five CLI manuals, four sequential tutorials, and small real MLIR inputs. Preserve deeper design documents and link them from the new guides. Document the implementation after #128/#130 rather than presenting #129 improvements as delivered.

**Source baseline:** `bb8203881c5b6de6eb5a1ddc8b4e05a11815380e` on main. Work in the isolated `docs/community-documentation` worktree. Production compiler behavior is outside this documentation change.

## Task 1: establish the reader path and examples

Create `docs/README.md`, `docs/getting-started.md`, and `docs/examples/`.

1. Check repository build settings against CMake and pinned CI; explain toolchain discovery, matching MLIR/FileCheck, configuration switches, and host/target differences.
2. Supply static BF16 matmul/SwiGLU source, a single explicit GEMM tile, and a one-candidate search-space file.
3. Parse every example with current tools; confirm exporter support and preserve the distinction between compilation artifacts and numerical invocation.

## Task 2: explain concepts, architecture, capabilities, and contributions

Create `docs/{concepts,architecture,features,contributing}.md`.

1. Explain semantics versus tile schedule, the five tile dimensions, logical/physical tile roles, and the conceptual uTile/uMap/uHW layers.
2. Diagram the two lowering paths and mapping phases; map components to current source directories and target-owned policy files.
3. Build a capability table using implementation evidence, with current limits and #129/#51/#52 handoffs.
4. Explain how a contributor can build, select tests, extend a dialect/pass/target, and provide reproducible review evidence.

## Task 3: write manuals and hands-on tutorials

Create `docs/tools/{compiler,optimizer,performance,tuning,benchmark}.md` and four numbered pages under `docs/tutorials/`.

1. Check each option's actual spelling, default, and implementation; do not rely on historical README examples alone.
2. Document output artifacts and failure modes, including legacy versus mapped compilation, prediction versus measurement, and synthetic harnesses.
3. Walk through source inspection/export, mapping/report/replay, performance interpretation, and bounded tuning.
4. Add links to deeper grammar/design references and known integration gaps.

## Task 4: keep the guides verifiable and discoverable

Modify README navigation, annotate the legacy architecture document, extend `test/Docs/check_doc_references.py`, and register `CommunityTutorials` in CMake.

1. Implement `docs/tutorials/run_tutorials.py` with only the Python standard library. Check real command success and artifact semantics: source export, search roundtrip, mapping metadata, byte-identical plan-ID replay, explicit-tile work/traffic, and predicted schedule output.
2. Include all community pages in path/help-name checks. Make `check-llk` depend on tools required by the tutorial runner.
3. Run the fresh tool build, tutorial runner, documentation reference check, registered CTest suite, and `git diff --check`.
4. Review guides as a new developer: no invented flags, no broken links, no planned capability described as shipped, and no synthetic timing described as compiled-kernel latency.
5. Commit documentation and verification changes, push the documentation branch, and open a ready-for-review PR against main.

The executable checks establish runnable documentation workflows and their stated artifacts. They do not replace numerical invocation, hardware ISA evidence, physical capacity proof, or calibrated prediction acceptance.
