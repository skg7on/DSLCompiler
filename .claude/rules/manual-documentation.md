# Developer Manual Documentation

**Category:** documentation
**Scope:** `docs/manual/` and related developer-facing documentation
**Enforcement:** mandatory when reviewing, refining or fixing the manual

## Describe the current contract

- Organize documentation around developer goals, setup, APIs, supported inputs,
  outputs, ownership, diagnostics and limitations.
- Describe the final implemented behavior directly. Do not narrate which PR,
  issue or milestone added a feature, or append delivery notes to an overview.
  Keep implementation history in review records, plans and Git history.
- State unsupported behavior as a support boundary rather than a delivery
  promise. Design proposals and planned work are not implemented capabilities.
- Link implementation and runnable fixtures for API contracts. Link historical
  evidence only when it helps explain an exact validation domain or limitation.

For example, prefer “The strict Torch backend compiles supported ATen graphs
through the native frontend” over “The frontend milestone adds graph compilation.”

## Verify claims against the implementation

- Read `CLAUDE.md` and follow the isolated-worktree policy before editing.
  Use recent changes to identify affected topics, then verify their final state
  against current code, public interfaces, configuration and tests.
- Prefer the codebase knowledge graph for code discovery, as required by the
  repository instructions. Documentation, configuration and literal searches
  may use text search.
- Distinguish capture, export, import, compilation and invocation. Success at
  one stage does not establish the next stage or numerical correctness.
- Distinguish static Python specialization from symbolic native invocation,
  reference execution from selected-target execution, and modeled performance
  from measured results. Document host, ABI and resource limits explicitly.
- State support per entry point. Functional operators, strict Dynamo and
  Inductor composition can have different training and layout contracts.
- Document packaged configuration separately from the native shared library.
  Conservative package capability metadata is not a runtime support probe.
- For upstream Triton, document exact toolchain qualification, export and
  launch separately, including native-library coexistence requirements.
- Reconcile the guide, architecture page, feature matrix, tutorials and entry
  points together. Remove stale claims instead of leaving contradictory text
  beside a new section. Historical test counts are not a support inventory.

## Make examples reproducible

- State the working directory and keep commands there. Prefer
  `cmake -S . -B build` when subsequent paths are repository-relative.
- Use the project Python environment consistently through `LLK_PYTHON`;
  explain optional dependencies and version-specific test environments.
- State native-library selection and platform-specific filenames. Keep
  generated examples, evidence and build output under ignored `build/`.
- Save source-captured Python kernels in real files. Check API spelling,
  method calls, imports, launch coverage and cleanup against executable fixtures.
- Separate a minimal reference example from full acceptance prerequisites.
  Show every required lane before invoking a checker that requires all lanes;
  use fresh evidence directories and never count missing or skipped execution
  as successful acceptance.

## Validate and report accurately

- Keep manual pages in reStructuredText. Use `:doc:` / `:ref:` for manual
  navigation, `:source:` for repository paths, and `:download:` /
  `literalinclude` for checked-in examples. Register new pages in a toctree
  and the `DOCS` list in `test/Docs/check_doc_references.py`.
- Build with the pinned dependencies in `docs/manual/requirements.txt` and
  strict Sphinx checks: `-b html -n -W --keep-going`. Keep doctrees and HTML
  under `build/docs/`; do not commit generated HTML.
- Follow `docs/manual/source/building-docs.rst` for reference checks,
  documentation regressions and relevant tutorial/workflow checks. Verify
  edited pages even if they are absent from the checker's document list.
- Check Python examples for syntax and execute documented workflows when the
  required environment is available. A successful Sphinx build checks structure
  and references; it does not execute examples or prove backend support.
- Run `git diff --check` and review the complete diff. Report the checks
  actually run, skips and unexecuted native examples separately; do not turn a
  successful export or reference result into a selected-target execution claim.
