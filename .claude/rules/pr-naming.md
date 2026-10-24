# PR Title Naming

**Category:** process
**Enforcement:** mandatory — every pull request title MUST follow this convention

## Rule

A pull request title has exactly this shape:

```text
<type>(<scope>): [<task-tag>] <imperative subject>
```

- **`<type>`** is one of `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`,
  `ci`, `chore`, `dev`, followed by a colon.
- **`<scope>`** is optional, lowercase, in parentheses, and names the subsystem the
  change belongs to: `mapping`, `machine`, `perf`, `target`, `micro`, `compile`,
  `tune`, `frontend`.
- **`<task-tag>`** is optional and goes *immediately after the prefix* — never at the
  start of the title.
- **`<imperative subject>`** is lowercase, imperative, has no trailing period, and
  keeps the whole title within roughly 72 characters.
- **Issue references** go at the end in parentheses: `(#123)`, `(D2, #162)`,
  `(issue #67 Stage C7)`.
- The Claude Code attribution line belongs to the **body**, never the title.

## Type Selection

| Primary change | Type | Example |
|---|---|---|
| New capability or behavior | `feat:` / `feat(scope):` | `feat(target): package and register static Arm SVE/SME targets` |
| Bug fix or correctness repair | `fix:` | `fix: require complete modeled analysis before ranking (#168)` |
| Tests and fixtures only | `test:` | `test(mapping): run the compiler-generated kernels and check their answers` |
| Documentation, designs, plans | `docs:` | `docs(target): design and plan the ARM SVE and SME mapping targets` |
| Restructuring without behavior change | `refactor:` | `refactor(micro): normalize abstract ownership through target data` |
| Performance only | `perf:` | — |
| Build, CI or tooling configuration | `build:` / `ci:` / `chore:` / `dev:` | `dev: add Zed debugging for LLK tools` |

The type reflects the **primary** change, not the largest diff. A slice that adds
only fixtures and assertions is `test:` even when it is the acceptance evidence for
a feature delivered by earlier PRs. The title should echo the primary commit's type
and scope; when a single PR carries one task, reuse that commit's prefix verbatim.

## Ordering

The prefix is always the first token. Task and sprint tags follow it; issue and
stage references close the title.

```text
✅ feat: [PF05] import checked frontend programs into structured workload IR
✅ test(mapping): verify typed ARM contraction placement and storage
✅ fix: report held-out prediction error by comparable regime (#164)
❌ [PF05] feat: import checked frontend programs ...
❌ [Framework E5] Verify typed ARM contraction placement ...
❌ [PF01] Package Python frontend and repair Dynamo registration
❌ Refactor LLK kernel lowerings into an explicit registry
```

## Process

1. Compose the title before `gh pr create`; check that the first token is one of the
   listed types.
2. Match the type to the primary change (see the table), not to the branch category.
3. If an existing PR violates the rule, fix it with
   `gh pr edit <number> --title "<corrected title>"` rather than leaving it.
4. Keep the body's ending attribution line intact when editing; this rule governs the
   title only.

## Evidence

As of 2026-10-24, 123 of 155 merged PRs use a type prefix, including every recent
one except #233. The convention started at the beginning of the repository history
(`chore:` on PR #3) and is the dominant form in the current PR list.
