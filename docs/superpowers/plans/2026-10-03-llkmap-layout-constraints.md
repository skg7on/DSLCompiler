# LLKMap Layout Constraints Implementation Plan (D3 / epic #67)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A small declarative language (**LLKMap**) for target layout legality: parse `layout` declarations, evaluate their typed constraints against a `MachineModel`, and solve them by bounded finite-domain enumeration — no SMT dependency.

**Architecture:** `include/LLK/Mapping/LayoutConstraints.h` + `lib/Mapping/LayoutConstraints.cpp` hold the lexer, parser, AST, evaluator, and solver. Target layout declarations live in target-owned files (`mapping/<target>/layouts.llkmap`). The solver reads machine facts only through `MachineModel` queries, so a layout file never embeds a concrete executor id (design §13.1, §13.4).

**Tech Stack:** C++20, MLIR `AffineMap`/`AffineExpr` for the logical→physical map, LLVM ADTs, CMake/Ninja, GoogleTest.

**Spec:** design §13.1–§13.4; epic #67 workstream 4; plan §4 D3.

## Global Constraints

- Layout IDs are target-owned strings (`avx2.blocked_2d`), never new enumerants in the `micro` dialect or `#micro.layout` (design §13.4).
- Deterministic and bounded: domain enumeration and solution order are fixed; the solver reports truncation rather than silently stopping.
- No SMT/Z3 dependency.
- Target vocabulary stays in the target's `layouts.llkmap`; generic code queries only through `MachineModel`.
- Diagnostics name the file, line, and column.

## Grammar (fixed here, per design §13.2)

```text
file       ::= layout*
layout     ::= "layout" id "(" params ")" "{" stmt* "}"
params     ::= [ param ("," param)* ]
param      ::= [ "int" | "sym" ] ident          // bare means int
stmt       ::= domain | require | map
domain     ::= "param" ident "in" ( "[" int ".." int "]" | "{" literal ("," literal)* "}" ) ";"
require    ::= "require" expr ";"
map        ::= "map" "(" ident ("," ident)* ")" "->" "(" expr ("," expr)* ")" ";"
expr       ::= or
or         ::= and ("||" and)*
and        ::= equality ("&&" equality)*
equality   ::= relational (("==" | "!=") relational)*
relational ::= additive (("<" | "<=" | ">" | ">=") additive)*
additive   ::= multiplicative (("+" | "-") multiplicative)*
multiplicative ::= unary (("*" | "/" | "%") unary)*
unary      ::= ("!" | "-") unary | postfix
postfix    ::= primary ("." ident "(" args ")")*
primary    ::= int | string | ident | "(" expr ")" | call
call       ::= ident "(" args ")"
id         ::= (letter | "_") (alnum | "_" | ".")*
```

Comments are `//` to end of line and `/* ... */`. Builtins available in expressions: `rank` (context), `element_type` (context), and machine queries:
`machine.compute(<kind>).lanes(<dtype>)`, `machine.compute(<kind>).count`,
`machine.memory(<id>).capacity_bytes`, `machine.memory(<id>).alignment_bytes`.

## MachineModel extension required

Design §13.2 queries `machine.compute("vector_engine").lanes(element_type)`, a per-dtype vector width. `MachineModel::ComputeNode` has shapes and element types but no dtype→width association, so Task 1 adds an optional `lanes` map to it (YAML `lanes: {f32: 8, bf16: 16}`) — additive and backward compatible.

## File Structure

| File | Responsibility |
|---|---|
| `include/LLK/Machine/MachineModel.h` (+ loader, profiles) | add `ComputeNode::lanes` |
| `include/LLK/Mapping/LayoutConstraints.h` | AST, `LayoutDef`, `LayoutRegistry`, `LayoutContext`, `LayoutSolution`, solver API, parse/load |
| `lib/Mapping/LayoutConstraints.cpp` | lexer, parser, evaluator, solver, AffineMap construction |
| `mapping/x86-avx2/layouts.llkmap` | AVX2 declarations |
| `test/Mapping/layout_constraints.cpp` | parser, evaluator, solver, positive/negative |
| `test/Mapping/Inputs/invalid-layouts.llkmap` | load-time rejection fixture |
| `docs/design/llkmap-layout-grammar.md` | the grammar above, documented for target authors |

---

### Task 1: `ComputeNode::lanes` and profile declarations

**Files:** modify `include/LLK/Machine/MachineModel.h`, `lib/Machine/MachineModel.cpp`, `lib/Machine/MachineModelLoader.cpp`, `machines/x86-avx2-v2.yaml`, `machines/generic-ai-accel-v2.yaml`; extend `test/Machine/machine_model.cpp` and `test/Machine/machine_model_loader.cpp`.

**Interfaces:** `std::map<std::string, int64_t> ComputeNode::lanes;` plus
`std::optional<int64_t> MachineModel::lanesFor(llvm::StringRef computeKind, llvm::StringRef elementType) const;`

- [ ] **Step 1: Write failing tests** — a model with `lanes = {{"f32", 8}}` returns 8 for `lanesFor("vector_engine", "f32")` and nullopt for an unknown dtype/kind; the AVX2 profile's vector engine exposes `lanesFor("vector_engine", "f32")`.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** the field, the query, loader key `lanes` (mapping of string→int64, duplicate-key rejected), canonical-string rendering (sorted), and add `lanes: {f32: 8, f16: 16, bf16: 16}` to the AVX2 vector engines and `{f32: 64, f16: 128}` to the generic VPU.
- [ ] **Step 4: Run to verify pass.** `./build/MachineModelTest && ./build/MachineModelV2LoaderTest`
- [ ] **Step 5: Commit** `feat(machine): add per-dtype compute lanes (D3 prerequisite)`

---

### Task 2: LLKMap lexer, parser, AST, registry

**Files:** create `include/LLK/Mapping/LayoutConstraints.h`, `lib/Mapping/LayoutConstraints.cpp`, `docs/design/llkmap-layout-grammar.md`; create `test/Mapping/layout_constraints.cpp`; modify `CMakeLists.txt`.

**Interfaces:**
- `using LayoutValue = std::variant<int64_t, std::string>;`
- `enum class ExprKind { IntLit, StringLit, ParamRef, Builtin, MachineQuery, Unary, Binary };`
- `struct Expr { ExprKind kind; int64_t intValue; std::string text, auxText, op; std::vector<std::shared_ptr<const Expr>> operands; };`
- `struct LayoutParam { std::string name; bool symbolic; };`
- `struct ParamDomain { std::vector<LayoutValue> values; };`
- `struct AffineMapSpec { std::vector<std::string> dims; std::vector<ExprPtr> results; };`
- `struct LayoutDef { std::string id; std::vector<LayoutParam> params; std::map<std::string, ParamDomain> domains; std::vector<ExprPtr> requires; std::optional<AffineMapSpec> map; };`
- `class LayoutRegistry { bool add(LayoutDef, std::string &error); const LayoutDef *find(llvm::StringRef) const; llvm::ArrayRef<LayoutDef> all() const; };`
- `llvm::Expected<LayoutRegistry> parseLayoutText(llvm::StringRef, llvm::StringRef sourceName);`
- `llvm::Expected<LayoutRegistry> loadLayoutFile(llvm::StringRef path);`

- [ ] **Step 1: Write failing tests** — parse a two-layout file; ids, params, domains, require count, and map dims are as declared; duplicate layout id rejected; a missing `;` rejected; an unknown top-level keyword rejected; a `map` with mismatched arity rejected.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement the lexer and recursive-descent parser** per the grammar, with `line:column` diagnostics; validate: duplicate layout ids, duplicate param names, a `map` referencing an undeclared dim, a `require` naming an undeclared parameter.
- [ ] **Step 4: Write the grammar doc** and register the library/test in CMake.
- [ ] **Step 5: Run to verify pass.**
- [ ] **Step 6: Commit** `feat(mapping): parse LLKMap layout declarations (D3)`

---

### Task 3: Expression evaluator and machine queries

**Interfaces:**
- `struct LayoutContext { int64_t rank = 0; std::string elementType; };`
- `llvm::Expected<LayoutValue> evaluateExpr(const Expr &, const llvm::StringMap<LayoutValue> &bindings, const machine::MachineModel &, const LayoutContext &);`

- [ ] **Step 1: Write failing tests** — integer arithmetic and precedence; `%` divisibility; comparisons and boolean ops; string equality; `rank`/`element_type` builtins from context; `machine.compute(kind).lanes(dtype)`, `.count`, `machine.memory(id).capacity_bytes`; unknown param, unknown query, unknown compute kind, and int/string type mismatch each error.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** the evaluator with strict typing (integers never compare to strings) and deterministic query resolution (first matching compute node in declaration order).
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** `feat(mapping): evaluate LLKMap expressions and machine queries (D3)`

---

### Task 4: Finite-domain solver and affine maps

**Interfaces:**
- `struct LayoutSolution { std::map<std::string, LayoutValue> values; mlir::AffineMap map; };`
- `struct SolverLimits { uint64_t maxAssignments = 100000; uint64_t maxSolutions = 8; };`
- `llvm::Expected<std::vector<LayoutSolution>> solveLayout(const LayoutDef &, const machine::MachineModel &, mlir::MLIRContext &, const LayoutContext &, const SolverLimits & = {});`
- `struct LayoutSolveResult { std::vector<LayoutSolution> solutions; bool truncated = false; };` (returned through `Expected`)

- [ ] **Step 1: Write failing tests** — a layout with `rank == 2` and `N % VW == 0` solves to the expected parameter tuples in deterministic order; a layout whose constraint can never hold returns no solutions; a parameter without a declared domain errors as unbound; exceeding `maxAssignments` reports truncation rather than a wrong answer; the solved `map` prints as the expected affine map.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** bounded cartesian enumeration in declared param order, early stop at `maxSolutions`, `truncated` when `maxAssignments` is hit; build the `AffineMap` from the `map` clause with the clause's idents as dims and referenced params as symbols, rejecting non-affine expressions.
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** `feat(mapping): solve LLKMap layouts by bounded enumeration (D3)`

---

### Task 5: AVX2 layouts, load-time validation, verification

**Files:** create `mapping/x86-avx2/layouts.llkmap`; extend `test/Mapping/layout_constraints.cpp`; add `LLK_MAPPING_DIR` compile definition; modify `CMakeLists.txt`.

- [ ] **Step 1: Write failing tests** — the shipped AVX2 file loads; `avx2.blocked_2d` solves for `f32` at rank 2 with `VW == 8` and `N % 8 == 0`; it does **not** solve for `bf16` when the profile's lanes for bf16 fall outside the declared domain; the invalid fixture is rejected at load with a diagnostic naming the problem.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Author `mapping/x86-avx2/layouts.llkmap`** with `avx2.row_major` and `avx2.blocked_2d` (domains narrow enough to enumerate quickly) and the invalid fixture.
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Full build and suite** — record registered/passed/skipped/failed.
- [ ] **Step 6: Commit** `feat(mapping): ship AVX2 LLKMap layouts (D3)`

---

## Self-Review

**Epic workstream 4 coverage:** grammar documented (Task 2 + doc); typed finite-domain params, int/bool expressions, divisibility, affine maps, machine queries (Tasks 2–4); deterministic bounded solver without SMT (Task 4); registry + load-time validation (Tasks 2, 5); target layout ids outside Micro enumerations (no Micro ODS touched); AVX2 declarations + positive/negative solver tests (Task 5). ✅

**Non-scope respected:** mapping rules (D4) reuse this language later; no placement/covering; no Micro dialect changes.

**Type consistency:** `LayoutValue` is defined once and reused by the evaluator and solutions; `LayoutDef`/`LayoutRegistry`/`LayoutSolution` appear in both the header and the tests with matching field names; `MachineModel` is used only through queries.
