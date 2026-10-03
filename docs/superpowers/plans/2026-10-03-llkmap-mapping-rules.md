# LLKMap Mapping Rules and Target Registry Implementation Plan (D4 / epic #67)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The LLKMap **rule** subset: a target declares how Micro operations map to its bundles, load-time validation rejects a broken rule file with a stable diagnostic, and a `MappingTarget` joins machine, layouts, rules, and emitters into one validated object.

**Architecture:** LLKMap's expression language is shared, so its lexer/expression parser/evaluator moves to `LLK/Mapping/LlkMap.h` + `lib/Mapping/LlkMap.cpp`, used by both layout and rule declarations. Rules live in `LLK/Mapping/MappingRules.h`; the target interface in `LLK/Mapping/MappingTarget.h`. Target rule files are `mapping/<target>/rules.llkmap` (design §13.1). Generic code never interprets bundle or emitter names — it only validates that they are declared.

**Tech Stack:** C++20, LLVM ADTs, MLIR `DictionaryAttr` for bundle parameters, CMake/Ninja, GoogleTest.

**Spec:** design §14 (rules and target bundles), §14.1 contents, §14.2 matching, §14.3 bundle contract, §14.4 load-time validation; epic #67 workstream 5; plan §4 D4.

## Global Constraints

- One-op rule matching only; no fusion across an unknown-effect, synchronization, or region boundary (design §14.2).
- Rule ids and versions are stable target-owned strings/integers; layouts are referenced by id, never redefined.
- A target bundle is opaque: name + typed parameters + emitter key. Generic code compares, hashes, reports, and hands it over; it never reads AVX2/TTGIR/NPU fields (design §14.3).
- Load-time validation, not late search failure (design §14.4).
- Diagnostics carry file, line, and column.

## Rule grammar (fixed here, extending design §14.1's illustration)

```text
file      ::= rule*
rule      ::= "rule" id [ "v" int ] "{" stmt* "}"
stmt      ::= match | domain | require | port | bundle | emit | cost
match     ::= "match" micro-op "(" [ predicate ("," predicate)* ] ")" ";"
predicate ::= ident "=" literal
domain    ::= "param" ident "in" ( "[" int ".." int "]" | "{" literal ("," literal)* "}" ) ";"
require   ::= "require" expr ";"
            | "require" executor "kind" ident ";"
            | "require" compute  "kind" ident ";"
            | "require" memory   "kind" ident ";"
            | "require" "layout" ident "satisfies" id ";"
port      ::= ( "input" | "output" ) string ";"
bundle    ::= "bundle" string ";"
emit      ::= "emit" string ";"
cost      ::= "cost" int ";"
micro-op  ::= "micro." ident
literal   ::= int | string | ident      // `f32` reads as the name "f32"
```

`match` names come from the workload node vocabulary (D1 `isWorkloadNodeOp`), so an unknown operation is a load-time error. `require ... satisfies <layout-id>` resolves against the layout registry (D3). `emit` resolves against the target's declared emitter keys, `bundle` against nothing — it is opaque.

## Load-time rejection (design §14.4)

duplicate rule ids; unknown Micro operation names; duplicate `match`, `bundle`, or `emit`; missing `match`/`bundle`/`emit`; duplicate port names; a `match` predicate naming an operation with no such attribute is *not* checked here (attributes are an IR fact, not a file fact); unbound parameters; unknown layout ids; unknown executor/compute/memory capability kinds; non-affine or unevaluable constraints; unknown emitter keys. Each with a stable `file:line:column` message.

## File Structure

| File | Responsibility |
|---|---|
| `include/LLK/Mapping/LlkMap.h` + `lib/Mapping/LlkMap.cpp` | shared AST, lexer, expression parser, `LayoutContext`, `EvalValue`, `evaluateExpr` (moved out of D3) |
| `include/LLK/Mapping/LayoutConstraints.h` + `lib/Mapping/LayoutConstraints.cpp` | layout declarations, registry, solver (unchanged behaviour; includes LlkMap.h) |
| `include/LLK/Mapping/MappingRules.h` + `lib/Mapping/MappingRules.cpp` | `RuleDef`, `RuleRegistry`, parse/load |
| `include/LLK/Mapping/MappingTarget.h` | `MappingTarget` interface + `TargetBundle` |
| `mapping/x86-avx2/rules.llkmap` | AVX2 rules |
| `test/Mapping/mapping_rules.cpp`, `test/Mapping/Inputs/invalid-rules.llkmap` | tests |
| `docs/design/llkmap-rule-grammar.md` | the grammar above |

---

### Task 1: Extract the shared LLKMap core (refactor)

**Files:** create `include/LLK/Mapping/LlkMap.h`, `lib/Mapping/LlkMap.cpp`; modify `include/LLK/Mapping/LayoutConstraints.h`, `lib/Mapping/LayoutConstraints.cpp`, `CMakeLists.txt`.

Move `LayoutValue`, `ExprKind`, `Expr`, `ExprPtr`, `LayoutContext`, `EvalValue`, `evaluateExpr`, and the lexer/expression-parser machinery out of the layout files into the shared unit. `LayoutConstraints.h` includes `LlkMap.h`, so existing includes keep working. Expose a parser base a declaration parser can derive from:

- `struct LlkMapToken { enum class Kind { Identifier, Int, String, Punct, End }; Kind kind; std::string text; int64_t intValue; unsigned line, column; };`
- `llvm::Expected<std::vector<LlkMapToken>> lexLlkMap(llvm::StringRef, llvm::StringRef sourceName);`
- `class LlkMapParser` — token cursor with `current()/advance()/atEnd()/isPunct()`, `fail()/failAt()/takeError()`, `expectPunct()/expectIdentifier()`, `parseExpression()`, `validateExpr()`, `numeric-limits helpers`.

- [ ] **Step 1:** Run the existing suite first to record the baseline: `ctest --test-dir build -R Mapping` (D3's 30 tests must pass before and after).
- [ ] **Step 2:** Move the code; `LayoutConstraints.cpp`'s `Parser` now derives from `LlkMapParser`.
- [ ] **Step 3:** `ninja -C build && ./build/MappingLayoutTest` — unchanged, 30/30. No behaviour change is intended; the test suite is the guard.
- [ ] **Step 4: Commit** `refactor(mapping): extract the shared LLKMap parser core`

---

### Task 2: Rule declarations and registry

**Files:** create `include/LLK/Mapping/MappingRules.h`, `lib/Mapping/MappingRules.cpp`; create `test/Mapping/mapping_rules.cpp`; modify `CMakeLists.txt`.

**Interfaces:**
- `struct RulePredicate { std::string attribute; LayoutValue value; };`
- `struct RulePort { std::string name; bool isInput; };`
- `struct KindRequirement { std::string role; std::string kind; };`  (`role` is `executor`|`compute`|`memory`)
- `struct LayoutRequirement { std::string port; std::string layoutId; };`
- `struct RuleDef { std::string id; uint64_t version; std::string matchOp; std::vector<RulePredicate> predicates; std::vector<LayoutParam> params; std::map<std::string, ParamDomain> domains; std::vector<ExprPtr> constraints; std::vector<KindRequirement> kindRequirements; std::vector<LayoutRequirement> layoutRequirements; std::vector<RulePort> ports; std::string bundle; std::string emitter; std::optional<uint64_t> costLowerBound; };`
- `class RuleRegistry { bool add(RuleDef, std::string &error); const RuleDef *find(llvm::StringRef) const; llvm::ArrayRef<RuleDef> all() const; };`
- `llvm::Expected<RuleRegistry> parseRuleText(llvm::StringRef, llvm::StringRef sourceName);`
- `llvm::Expected<RuleRegistry> loadRuleFile(llvm::StringRef path);`

- [ ] **Step 1: Write failing tests** — a valid two-rule file parses with the expected match op, predicates, ports, bundle, emitter, kind and layout requirements; a duplicate rule id is rejected; a missing `bundle` is rejected; a duplicate `emit` is rejected; an unknown Micro operation is rejected; an unknown `match` predicate form is rejected; a `require` naming an undeclared parameter is rejected.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** the rule parser on `LlkMapParser`, reusing `validateExpr`. Validate the Micro operation name against D1's workload vocabulary — add a `bool isWorkloadNodeOp(llvm::StringRef)` overload to `WorkloadGraph.h` for this (the existing overload needs an `OperationName`).
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** `feat(mapping): parse LLKMap mapping rules (D4)`

---

### Task 3: `MappingTarget` and cross-registry validation

**Files:** create `include/LLK/Mapping/MappingTarget.h`, `lib/Mapping/MappingTarget.cpp`; extend `test/Mapping/mapping_rules.cpp`; modify `CMakeLists.txt`.

**Interfaces:**
```cpp
struct TargetBundle { std::string name; mlir::DictionaryAttr parameters; std::string emitterKey; };

class MappingTarget {
public:
  virtual ~MappingTarget() = default;
  virtual llvm::StringRef name() const = 0;
  virtual const machine::MachineModel &machine() const = 0;
  virtual const LayoutRegistry &layouts() const = 0;
  virtual const RuleRegistry &rules() const = 0;
  virtual bool isKnownEmitter(llvm::StringRef key) const = 0;
};

/// Loads a target from `machines/<machine>.yaml`, `mapping/<target>/layouts.llkmap`,
/// and `mapping/<target>/rules.llkmap`, then validates the rules against the other two.
llvm::Expected<std::unique_ptr<MappingTarget>>
loadMappingTarget(llvm::StringRef machinePath, llvm::StringRef layoutPath,
                  llvm::StringRef rulePath, std::vector<std::string> emitterKeys);

/// Re-validates a target's rules against its layouts and machine. Fails on the
/// first unknown layout id, capability kind, or emitter key.
llvm::Error verifyMappingTarget(const MappingTarget &target);
```

- [ ] **Step 1: Write failing tests** — rules referencing a known layout and known kinds verify; an unknown layout id fails; an unknown capability kind fails; an unknown emitter fails; a rule whose `match` names an operation outside the vocabulary fails even when the file parses.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement** `FileMappingTarget` holding the three registries plus the emitter key set, and `verifyMappingTarget`.
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Commit** `feat(mapping): add MappingTarget and cross-registry validation (D4)`

---

### Task 4: Shipped AVX2 rules and verification

**Files:** create `mapping/x86-avx2/rules.llkmap`, `test/Mapping/Inputs/invalid-rules.llkmap`; extend `test/Mapping/mapping_rules.cpp`.

- [ ] **Step 1: Write failing tests** — the shipped rules load; they reference `avx2.blocked_2d`, the `worker` executor kind, and the `vector_engine`/`matrix_engine` compute kinds that `machines/x86-avx2-v2.yaml` declares; every emitted key is in the target's declared set; the invalid fixture is rejected.
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Author `mapping/x86-avx2/rules.llkmap`** with rules for `micro.vector`, `micro.mma`, and `micro.reduce`, plus the invalid fixture.
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5: Full build and suite** — record registered/passed/skipped/failed.
- [ ] **Step 6: Commit** `feat(mapping): ship AVX2 LLKMap rules (D4)`

---

## Self-Review

**Design §14 coverage:** rule contents (§14.1) → Task 2; one-op matching and side-effect discipline (§14.2) → Task 2 restricts `match` to the workload vocabulary and covers exactly one op; bundle contract (§14.3) → Task 3 `TargetBundle` carries name/params/emitter and is never interpreted; load-time validation (§14.4) → Tasks 2–3. ✅

**Non-scope respected:** placement (D5) and covering (D6) consume `MappingCandidate` from D1; no target-specific fields are added to generic ODS; the target plugin that *creates* emitters is D7 — D4 only validates keys.

**Type consistency:** `Expr`/`ExprPtr`/`LayoutValue`/`LayoutParam`/`ParamDomain` are defined once in `LlkMap.h` (Task 1) and reused by both declaration kinds; `RuleDef` field names match between header, parser, and tests.

## Verification Results (2026-10-03)

- **Build:** `ninja -C build` — clean.
- **New tests:** `MappingRulesTest` **19/19** (rule parsing + target validation); D3's `MappingLayoutTest` 30/30 unchanged after the parser extraction.
- **Full suite:** `ctest --test-dir build --output-on-failure` — 97 registered, **95 passed, 2 skipped, 0 failed**.
- **Deprecated-API audit:** clean across `LlkMap.h`, `LayoutConstraints.*`, `MappingRules.*`, `MappingTarget.*`, `WorkloadGraph.*`, `mapping_rules.cpp`.

### Decisions taken while implementing

1. **The shared core was extracted rather than duplicated.** `LlkMap.h` now holds the AST, lexer, `LlkMapParser` (cursor + expression grammar + `validateExpr`), and the evaluator; layouts and rules derive from it. D3's 30 tests were the guard for a behaviour-preserving move.
2. **A rule's `param` statement declares and bounds the parameter**, so it must appear before its first use — unlike a layout, whose parameters are listed in its header.
3. **Capability requirements are checked against the machine**, not against a static vocabulary: a rule requiring a `compute kind tensor_core` fails because `x86-avx2-v2.yaml` declares no such capability. Executor matching uses the same owner-kind refinement rule as placement (`worker` is satisfied by a refining `core`).
4. **`match` predicates are structural.** The attribute names are not checked against each operation's definition; validating them against the dialect is a later refinement.
5. **One lexer fix surfaced:** a single `=` was not in the punctuation set, so `kind = "add"` failed to lex. `->` had the same problem in D3.
