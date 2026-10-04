# Micro-IR Phase 4 — Binding-Driven Mapping (gap #4)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the persistent search binding actually **drive** mapping decisions. Today a `micro.candidate`'s values reach the plan only as provenance (a hash and a parameter map on the finished plan); they never constrain rule parameters, layout selection, or global legality, and the public mapping pass supplies no binding at all. Close the design's central bridge: `persistent micro.candidate → binding → mapping problem → selected plan`.

**Architecture:** A binding loader in the pass library (which already depends on both the Micro dialect and `LLKMapping`), a *constraining* binding threaded through `CoveringSearch` into rule resolution and layout selection, and pass-level plumbing so a named candidate drives a real search. No change to the target-ownership boundary and no `LLKMapping → LLKPerf` dependency.

**Tech Stack:** C++20, CMake + Ninja, LLVM/MLIR 24 local / 22 CI, GoogleTest, FileCheck.

**Spec:** `docs/superpowers/specs/2026-09-18-microir-inspired-dslcompiler-enhancement-design.md` — §5.2 (durable intent), §8.3 (SearchSpaceLoader handoff: every later plan records the source binding hash), §9.2/§9.3 (requirements are abstract; a binding constrains resolution), §13 (declarative layouts), §14.1 (parameter domains / derived expressions), §16.5 / §22.3 (`global_constraint_failed`).

**Gap source:** the PR-#98 review, item **#4** ("persistent search bindings still do not drive mapping decisions"), and the phase-1 ledger's note that `SearchBinding` is "used only by its own unit test".

## Global Constraints

- Branch `feat/micro-ir-binding`, **stacked on the phase-1..3 stack tip** (`origin/feat/micro-ir-phase2`). CI will not run (stacked base); report the locally verified subset.
- `.claude/rules/preferred-mlir-api.md` and `llvm-version-compatibility.md` apply.
- Tests hand-registered in `CMakeLists.txt`. `LLKMapping` is `-fno-rtti -fno-exceptions`; generic code stays target-neutral.
- **Layering:** `LLKPerf` already depends on `LLKMapping`. **`LLKMapping` must not depend on `LLKPerf`.** The MLIR-facing binding loader therefore lives in the pass library (`LLKMicroMapping`), which already links the dialect and `LLKMapping`. Where `LLKPerf` already has a typed parse (`loadSearchSpace`) that could be reused, reuse it *from the pass layer* — do not copy it into `lib/Mapping`, and do not add a Mapping→Perf edge.
- Determinism: parameter/value maps must be rendered sorted and type-tagged before reaching a hash or a report (reuse `canonicalSearchValueString`).
- Every task ends with `ninja` clean and the affected tests green before committing.

### Deferred (not in this plan)

**#5** full materialization, **#7** cost-contract parity, the remaining **#8** chains, and the `warp` owner-vocabulary decision. `llk-tune`'s separate Perf tuning path and `llk-compile`'s emit-only integration are **also out of scope here** (they are the "existing tuning code" migration in §26.3, not the mapping bridge); record them as follow-ups.

---

## Task 1: Load a `SearchBinding` from a `micro.candidate`

`mapping::SearchBinding` (phase-1 D1) has a hash, immutability, and canonical rendering — and no producer. `lib/Perf/SearchSpace.h` already loads a `micro.search_space` into a typed `SearchSpace` (params, constraints, objective); `micro.candidate` names a complete assignment.

**Files:**
- Create: `include/LLK/Conversion/MicroMapping/SearchBindingLoader.h` + `lib/Conversion/MicroMapping/SearchBindingLoader.cpp` (register the source once)
- Test: `test/Conversion/MicroMapping/` (a `.mlir` fixture + a GTest driving the loader, registered)

**Interfaces:**
- Produces: `llvm::Expected<mapping::SearchBinding> loadSearchBinding(mlir::ModuleOp module, llvm::StringRef candidateSymbol);` and a variant that picks the module's only candidate when the symbol is empty.

- [ ] **Step 1: Write the failing tests**

```cpp
TEST(SearchBindingLoader, LoadsEveryBoundParameterOfANamedCandidate) {
  // micro.search_space with BM{32,64} and tile_layout{"blocked","row_major"},
  // micro.candidate @candidate_17 binding both -> SearchBinding.values has both,
  // candidateId == "candidate_17", stableHash matches makeSearchBinding of the same map.
}
TEST(SearchBindingLoader, RejectsAnUnknownCandidateSymbol) { /* Expected error naming the symbol */ }
TEST(SearchBindingLoader, RejectsACandidateThatDoesNotBindEveryParameter) { /* incomplete binding */ }
TEST(SearchBindingLoader, RejectsAValueOutsideTheDeclaredDomain) { /* global constraint, §16.5 */ }
TEST(SearchBindingLoader, AnAmbiguousModuleRequiresASymbol) { /* >1 candidate, empty symbol -> error */ }
```

- [ ] **Step 2: Run to verify failure** — `ctest -R MicroMapping`.

- [ ] **Step 3: Implement** — read the `micro.search_space`, find the `micro.candidate` by symbol (or the sole one), and build `makeSearchBinding(symbol, values)`. Validate: every declared `micro.param` is bound; every value is one of that param's declared `micro.param` choices (this is the **global domain check** — a candidate outside its domain is a load error, not a late search failure); no bound name is undeclared. Reuse `llk::perf::loadSearchSpace` from this layer rather than re-parsing the ops — `LLKMicroMapping` may link `LLKPerf`; state in your report whether you linked it or read the ops directly, and why.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `feat(mapping): load a SearchBinding from a micro.candidate`.

---

## Task 2: The binding constrains rule parameter resolution

Today `CoveringSearch` records the binding's hash and values onto the finished plan (phase-1 T4) and nothing more. Rule parameters are resolved by bounded enumeration over declared domains (`resolveRuleConstraints`, phase-2 T15) with no reference to the binding.

**Files:** Modify: `include/LLK/Mapping/MappingRules.h` + `lib/Mapping/MappingRules.cpp` (the resolution entry point takes the binding or a pinned-value view), `lib/Mapping/CoveringSearch.cpp` (pass `binding_` in); Test: `test/Mapping/mapping_rules.cpp`, `test/Mapping/covering_search.cpp`.

**Interfaces:** e.g. `toMappingCandidate(node, def, machine, layoutContext, const llvm::StringMap<SearchValue> *pinned)` — a rule parameter named in `pinned` may only take that value; the agent must choose the exact shape and justify it.

- [ ] **Step 1: Failing tests** — (a) a rule with `param VW in [4..8]` and a binding `VW=8` resolves to `VW=8` even when `VW=4` would also satisfy its constraint; (b) a binding whose pinned value makes the rule's `require` unsatisfiable yields **no candidate** for that node (a non-match, not an error); (c) a param the binding does not name is still enumerated freely.
- [ ] **Step 2: Verify red** — today the binding is ignored, so (a) can resolve `VW=4`.
- [ ] **Step 3: Implement** — restrict the enumeration for pinned parameters to the bound value; keep the existing admissibility/truncation behaviour for unpinned ones.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `feat(mapping): constrain rule parameter resolution by the binding`.

---

## Task 3: The binding constrains layout selection

A binding may name a `layout`-kind parameter (e.g. `tile_layout = "blocked"`). Today the rule's `require layout <port> satisfies <id>` chooses the layout id with no reference to it.

**Files:** Modify: `lib/Mapping/MappingRules.cpp` (layout requirement resolution), `lib/Mapping/Placement.cpp` (the solve still decides the concrete parameters); Test: `test/Mapping/placement.cpp`, `test/Mapping/covering_search.cpp`.

- [ ] **Step 1: Failing test** — a search space whose `tile_layout` is bound to one value, and a rule offering a matching layout id, selects that layout; a binding naming a layout the rule does not offer yields no candidate.
- [ ] **Step 2: Verify red**.
- [ ] **Step 3: Implement** — map the binding's layout-kind value onto the rule's layout requirement (the dialect's `kind` distinguishes a layout parameter, so do not guess from the parameter's *name*).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `feat(mapping): constrain layout selection by the binding`.

---

## Task 4: `--micro-map` supplies the binding, end to end

The pass takes no binding today, so the persistent→mapping bridge is not wired at all.

**Files:** Modify: `lib/Conversion/MicroMapping/MicroMapPass.cpp` (+ `MicroMappingCommon.h` to construct `CoveringSearch` with the binding); Test: `test/Conversion/MicroMapping/` (a FileCheck or script test).

- [ ] **Step 1: Failing test** — `--micro-map candidate="candidate_17" …` on a kernel with a `micro.search_space` binds the resulting plan whose `sourceBindingHash` equals that candidate's `stableHash`, and the bound IR records it; a missing/unknown candidate symbol is a pass error naming it; `candidate=` absent behaves as today (no binding).
- [ ] **Step 2: Verify red** — the option does not exist.
- [ ] **Step 3: Implement** — add a `candidate` option; load via Task 1; pass it to `CoveringSearch`; keep the no-candidate path unchanged. Document that a binding changes which plans are legal, so a candidate that cannot be satisfied is a *search* failure with the frontier's diagnostics.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** — `feat(mapping): drive --micro-map from a persistent candidate`.

---

## Self-review

- **Coverage:** the review's #4 lists "rule parameters, layouts, placement, routes, or global legality". T2 = rule parameters; T3 = layouts; T1 = the domain/global-legality check at load; placement/routes remain binding-independent *by design* (a binding names search choices, not machine resources) — say so explicitly in the report rather than leaving it implicit.
- **Sequencing:** T1 before T2/T3 (they consume the binding); T2 and T3 both touch `MappingRules.cpp` — sequential; T4 last (it wires the others).
- **Layering risk:** T1 must not create an `LLKMapping → LLKPerf` edge; the loader lives in `LLKMicroMapping`.
- **No placeholders** beyond the interface shape T2 leaves to the implementer, flagged there.
