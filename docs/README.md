# Documentation

The [developer manual](manual/README.md) is the community entry point, published at <https://skg7on.github.io/DSLCompiler/>. The repository holds its sources; CI renders and deploys the HTML, so nothing requires Sphinx or LLVM to read.

| Directory | Contents |
|---|---|
| [`manual/`](manual/README.md) | RST manual sources, runnable examples, and Sphinx configuration |
| [`design/`](design/) | Detailed architecture and milestone designs, including LLKMap grammar |
| [`reviews/`](reviews/) | Revision-specific acceptance evidence and gap assessments |
| [`superpowers/specs/`](superpowers/specs/) | Detailed design specifications |
| [`superpowers/plans/`](superpowers/plans/) | Implementation plans and task history |

The project goal is a **generic MicroIR-based, mapping- and search-based DNN compiler framework**. Matmul and fused SwiGLU are validation examples. Start with [ARCHITECTURE.md](../ARCHITECTURE.md) and the [framework goal and design spec](superpowers/specs/2026-10-12-generic-microir-dnn-compiler-framework-design.md) for project direction, and the manual for current usage.

The [framework implementation plan](superpowers/plans/2026-10-12-generic-microir-framework-implementation.md) audits current code at `fb99595` and sequences 19 tasks across analysis coverage, generic workload/source search, primitive/target conformance and verified measurement persistence/validation. Its linked increment plans distinguish proposed APIs from merged foundations and pending PRs.

The proposed [dynamic-shape design](superpowers/specs/2026-10-17-dynamic-shape-llk-microir-design.md) reviews baseline `45560cb` and defines runtime M/N/K with fixed-capacity MicroIR tiles, checked shape relationships, bounded movement, explicit output destinations and shape-aware analysis/replay. Its [implementation plan](superpowers/plans/2026-10-17-dynamic-shape-llk-microir.md) makes fully dynamic contractions the first execution milestone. These documents are proposals, not delivered dynamic-shape support.

The framework specification governs project-level goal/scope statements in earlier specs. Detailed subsystem contracts remain applicable where consistent with it. Milestone plans and revision-pinned reviews preserve history and evidence; the [historical CPU architecture](design/legacy-cpu-architecture.md) records the original validation design.

The [Python frontend guide](manual/source/python-frontend.rst) records PF01's
installable package, safe Dynamo registration, structured errors, project-level
`.venv` setup and current capability limits. The
[PF01 environment evidence](reviews/2026-10-19-pf01-python-environment.md) records
21 passing tests with Python 3.14.8 / Torch 2.14.1 and the separately tested
Python 3.11/3.12 + Torch 2.9.0 reference matrix.

PF03 adds the [typed frontend JSON schema](design/python-frontend-v1.schema.json),
immutable program records, deterministic metadata binding and canonical request
bytes. Its [verification record](reviews/2026-10-23-pf03-typed-frontend.md) covers
compiler-independent schema/binding tests and the boundary for later AST capture,
native import and checked invocation tasks.

The [Python frontend refinement design](superpowers/specs/2026-10-18-python-frontend-refinement-design.md)
and [20-task implementation plan](superpowers/plans/2026-10-18-python-frontend-refinement.md)
address the [revision-pinned frontend review](reviews/2026-10-18-python-triton-inductor-gap-assessment.md).
PF01 is implemented in PR #233; the remaining executable decorator, strict Dynamo,
Inductor external-kernel and genuine upstream Triton milestones remain proposals
under #203. Their design text is not execution evidence.
