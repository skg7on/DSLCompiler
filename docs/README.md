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

The framework specification governs project-level goal/scope statements in earlier specs. Detailed subsystem contracts remain applicable where consistent with it. Milestone plans and revision-pinned reviews preserve history and evidence; the [historical CPU architecture](design/legacy-cpu-architecture.md) records the original validation design.
