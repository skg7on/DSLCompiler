# Worktree Isolation Policy

**Category:** process
**Enforcement:** hard gate — all writes MUST follow this policy

## Rule

Every design or implementation write MUST happen in an isolated git worktree. The main checkout at the repository root (`/Users/skg7on/Workspace/Projects/DSLCompiler`) is for read-only inspection only.

## Worktree Location

All worktrees MUST be created under `.claude/worktrees/` in the project root directory.

## Branch Naming Convention

All worktree branches MUST follow the pattern `category/short-description` using kebab-case.

**Categories:**
- `docs` — documentation, design specs, plans
- `feat` — feature implementation
- `fix` — bug fixes
- `refactor` — code restructuring without functional change
- `exp` — experiments, spikes, prototypes

**Examples:**
- `feat/llk-dialect-verifier`
- `docs/architecture-overview`
- `fix/scratch-analysis-false-positive`
- `refactor/extract-shared-tiling-utils`
- `exp/amx-tile-config-benchmark`

## Lifecycle

1. Create worktree + branch before making any changes
2. Commit work incrementally within the worktree
3. Push branch and open PR for review
4. After merge, delete the worktree and branch
5. If abandoned, delete the worktree and branch immediately
