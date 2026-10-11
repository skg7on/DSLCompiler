# Community documentation implementation plan

**Goal:** help a developer understand MicroIR, run the current tools, and find a concrete contribution without reading the full design history first.

**Approach:** create a native reStructuredText community manual built with Sphinx: an entry point, concept/architecture/feature guides, five CLI manuals, four sequential tutorials, and small real MLIR inputs. Preserve deeper design documents as Markdown references linked from the manual. Document the implementation after #128/#130 rather than presenting #129 improvements as delivered.

**Source baseline:** `bb8203881c5b6de6eb5a1ddc8b4e05a11815380e` on main. Work in the isolated `docs/community-documentation` worktree. Production compiler behavior is outside this documentation change.

## Task 1: establish the reader path and examples

Create `docs/index.rst`, `docs/getting-started.rst`, and `docs/examples/`.

1. Check repository build settings against CMake and pinned CI; explain toolchain discovery, matching MLIR/FileCheck, configuration switches, and host/target differences.
2. Supply static BF16 matmul/SwiGLU source, a single explicit GEMM tile, and a one-candidate search-space file.
3. Parse every example with current tools; confirm exporter support and preserve the distinction between compilation artifacts and numerical invocation.

## Task 2: explain concepts, architecture, capabilities, and contributions

Create `docs/{concepts,architecture,features,contributing}.rst`.

1. Explain semantics versus tile schedule, the five tile dimensions, logical/physical tile roles, and the conceptual uTile/uMap/uHW layers.
2. Diagram the two lowering paths and mapping phases; map components to current source directories and target-owned policy files.
3. Build a capability table using implementation evidence, with current limits and #129/#51/#52 handoffs.
4. Explain how a contributor can build, select tests, extend a dialect/pass/target, and provide reproducible review evidence.

## Task 3: write manuals and hands-on tutorials

Create `docs/tools/{compiler,optimizer,performance,tuning,benchmark}.rst` and four numbered pages under `docs/tutorials/`.

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

## Task 5: refine and build the Sphinx developer manual

User follow-up: convert every new community guide/manual/tutorial to native RST and build HTML on the existing ready PR #131.

1. Convert the 16 reader-facing Markdown pages, preserve every command and implementation limit, and delete their superseded Markdown copies. Use native `doc`, `ref`, and `download` roles; check repository links with a small `source` role.
2. Add `docs/conf.py`, pinned `docs/requirements.txt`, section indexes, `docs/building-docs.rst`, local Furo styling, and an offline SVG architecture diagram. Include fixture source with `literalinclude`.
3. Extend the existing checker to RST code blocks and links, retain legacy Markdown checking, and add focused regressions for broken links and invalid flags. Keep the tutorial runner's artifact checks.
4. Add an optional `docs-html` CMake target and a documentation CI job that uploads HTML without requiring LLVM. Normal compiler builds must not depend on Sphinx.
5. Run a strict `python -m sphinx -b html -n -W --keep-going docs build/docs/html` build, inspect desktop/mobile navigation and tables, run checker regressions and compiler documentation tests, and run the required suite after CMake changes.
6. Commit/push on `docs/community-documentation` and update PR #131's title/body around the final RST/Sphinx implementation.
