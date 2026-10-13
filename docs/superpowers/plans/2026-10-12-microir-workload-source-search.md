# Generic Workload Contracts and Source Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Support original-source mapped compilation/search for a non-contraction program through a generic workload contract and an explicit source adapter boundary.

**Architecture:** Add a small versioned port/axis/numerical contract, keep existing contraction facts as specialized compatibility data, and dispatch source realization through injected adapters. Preserve the existing finite candidate generator, workload graph, complete-plan evaluator and frozen mapping plan. First prove unchanged direct MicroIR, then a bounded static rank-two `linalg.generic` elementwise adapter.

**Tech Stack:** C++20, MLIR Linalg/Tensor/Arith/Micro, LLVM Expected/Error/JSON, GTest, FileCheck and Python CLI tests.

**Spec:** [Framework design §§3–5 and 7](../specs/2026-10-12-generic-microir-dnn-compiler-framework-design.md); [index/baseline](2026-10-12-generic-microir-framework-implementation.md).

## Global Constraints

- “The generic workload contract must accommodate elementwise, reduction, broadcast, contraction and composed dataflow as capabilities are added.”
- “It must not require every workload to have exactly M/N/K, one shared left-hand operand, one or two RHS arms, or a contraction epilogue.”
- “Binding must not substitute a synthetic GEMM for a different original workload or alter semantics through metadata alone.”
- “Each supported choice has a finite domain, legality rule and concrete realization.”
- “Search dimensions derive from workload/target capabilities.”
- “Frozen replay preserves full source/candidate content, bindings, plan, machine/layout/rule content, compiler version, ABI, numerical contract and execution mode.”
- A's audited analysis and complete-ranking gates are prerequisites. ABI v1 stays rank-two, compact, offset zero, with at most 12 descriptors and disjoint outputs.
- Preserve current source-mode and WorkloadShape entry points; synthetic mode remains explicitly requested legacy behavior.

## Review Focus

1. Rank-three/multi-port contracts must work without invented K dimensions, but unsupported execution must fail: B1/B7.
2. Duplicate roots, changed numerical attributes or unused output-init tensors must not change source selection or disappear from identity: B2/B4/B5.
3. A generic integer parameter cannot be consumed by name guessing; unknown/ambiguous/unused choices must reject: B3/B6.
4. A structured body with reduction, nonidentity indexing, dynamic shape or side effects must not pass the bounded adapter: B5.
5. Candidate cloning, mapping and frozen replay must preserve the source and all semantics across both new and legacy paths: B2/B6/B7.

---

## New file map and library boundaries

| New file | Responsibility |
|---|---|
| `include/LLK/Conversion/Workload/WorkloadContract.h` | Plain versioned workload/port/axis data; no operator enum or M/N/K requirement. |
| `lib/Conversion/Workload/WorkloadContract.cpp` | Validation, canonical serialization, MLIR attribute read/write. |
| `include/LLK/Conversion/Workload/SourceAdapter.h` | Adapter registration, schema ownership, selection and realization interfaces. |
| `lib/Conversion/Workload/SourceAdapter.cpp` | Immutable registry, exact root selection, source clone, contract verification. |
| `lib/Conversion/Workload/ContractionSourceAdapter.cpp` | Compatibility wrapper around the current LLK-to-Micro exporter. |
| `lib/Conversion/Workload/ElementwiseSourceAdapter.cpp` | Bounded Linalg body validation and tile program construction. |
| `lib/Conversion/Workload/BindingSchema.cpp` | Parameter ownership validation and binding receipts; no target-name policy. |
| `test/Tuning/FrameworkTestSupport.h` | Test-only dialect/source helpers shared by the new tests. |

Root `CMakeLists.txt` registers two small libraries: `LLKWorkloadContracts` for contract data/serialization (LLVM Support + MLIR IR) and `LLKSourceCandidates` for adapters (Micro/Linalg/Tensor/Arith, LLKToMicro, LLKPerf). `LLKPerf` may consume the contract header/library; it must not depend on `LLKSourceCandidates`. `LLKMicroMapping`/`LLKMappedTuning` consume source candidates from above the mapping core. No new runtime/performance dependencies enter `LLKMapping` or `LLKMachine`. If #142 is merged, the contraction wrapper calls its exporter/registry instead of importing operator-specific files.

## Shared proposed APIs

These are **new**, not symbols present at the audited revision:

```cpp
// WorkloadContract.h, namespace mlir::llk::workload
enum class PortDirection { Input, Output };
enum class PortEffect { None, Read, Write, ReadWrite };
struct WorkloadPort {
  std::string name;
  PortDirection direction = PortDirection::Input;
  PortEffect effect = PortEffect::Read;
  std::vector<int64_t> shape;
  std::string elementType;
};
struct WorkloadContract {
  uint32_t version = 1;
  std::vector<WorkloadPort> ports; // declared order, never sort ports
  std::map<std::string, int64_t> axes;
  std::string numericalContract;
  std::vector<std::string> requiredPrimitives;
};
llvm::Error validateWorkloadContract(const WorkloadContract &);
std::string canonicalWorkloadContract(const WorkloadContract &);
mlir::DictionaryAttr encodeWorkloadContract(mlir::MLIRContext *,
                                          const WorkloadContract &);
llvm::Expected<WorkloadContract> decodeWorkloadContract(mlir::DictionaryAttr);
```

Use forward declarations for MLIR types where practical; include standard headers explicitly. Static positive shapes are the first supported contract. Unknown dtype, duplicate port/axis names, unsupported versions and negative/dynamic/zero dimensions are errors. Empty axes are valid for direct MicroIR. Scalar rank-zero contracts may be represented only after explicit support is added; this increment rejects them. `numericalContract` is canonical semantic metadata, not a performance flag; the source's full attribute content also remains in identity.

```cpp
// SourceAdapter.h, namespace mlir::llk::workload
enum class BindingOwner { Source, Mapping };
enum class ReceiptStage { Source, Complete };
struct ParameterBinding {
  std::string parameter;
  BindingOwner owner = BindingOwner::Source;
  std::string role;
};
struct BindingReceipt {
  std::vector<std::string> source;
  std::vector<std::string> mapping;
};
struct SourceSelection {
  std::string symbol;
  uint64_t rootOrdinal = 0;
};
struct SourceRealization {
  mlir::Operation *kernel = nullptr; // owned by the caller's clone
  BindingReceipt receipt;
};
class SourceAdapter {
public:
  virtual ~SourceAdapter() = default;
  virtual llvm::StringRef key() const = 0;
  virtual bool accepts(mlir::Operation *) const = 0;
  virtual llvm::Expected<WorkloadContract>
  describe(mlir::Operation *) const = 0;
  virtual llvm::Expected<std::vector<ParameterBinding>>
  bindings(mlir::Operation *, const perf::SearchSpace &) const = 0;
  virtual llvm::Expected<SourceRealization> realize(
      mlir::ModuleOp clone, mlir::Operation *selectedRoot,
      const WorkloadContract &, const perf::SearchSpace &,
      const perf::Candidate &) const = 0;
};
class SourceAdapterRegistry {
public:
  llvm::Error add(std::unique_ptr<SourceAdapter>);
  llvm::Expected<const SourceAdapter *> resolve(mlir::Operation *,
                                               llvm::StringRef key = {}) const;
private:
  std::vector<std::unique_ptr<SourceAdapter>> adapters;
};
SourceAdapterRegistry makeDefaultSourceAdapters();
llvm::Expected<mlir::Operation *> selectSourceRoot(
    mlir::ModuleOp, const SourceSelection &, const SourceAdapterRegistry &,
    llvm::StringRef adapterKey = {});
llvm::Error validateBindingReceipt(const perf::SearchSpace &,
    const perf::Candidate &, llvm::ArrayRef<ParameterBinding>,
    const BindingReceipt &, ReceiptStage stage = ReceiptStage::Complete);
```

Source-stage validation requires all source-owned bindings realized and mapping-owned bindings declared but pending. Complete-stage validation requires both owners realized; the generic tuner invokes it after selection, before retention/reporting. No silent first-match dispatch: zero matches is unsupported, two matches require an explicit key. `accepts` performs structural family recognition; `describe` performs detailed validation. Symbol/ordinal selection counts accepted top-level roots within the selected function, excluding nested scalar bodies and unrelated functions. Concrete MicroIR selection resolves kernel symbols separately.

## Test helper used in B2–B7

Create this complete fixture in B2's test-support header. New tests include the proposed headers, existing CandidateGenerator/MappingTarget/Target factory headers, and standard/GTest headers explicitly.

```cpp
class FrameworkTest : public ::testing::Test {
protected:
  mlir::DialectRegistry dialects;
  std::unique_ptr<mlir::MLIRContext> context;
  void SetUp() override {
    dialects.insert<mlir::micro::MicroDialect, mlir::llk::LLKDialect,
                    mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                    mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
    context = std::make_unique<mlir::MLIRContext>(dialects);
    context->loadAllAvailableDialects();
  }
  mlir::OwningOpRef<mlir::ModuleOp> parse(llvm::StringRef text) {
    return mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
  }
  std::string print(mlir::Operation *op) {
    std::string text;
    llvm::raw_string_ostream out(text);
    op->print(out);
    return text;
  }
};
```

All new GTests are registered in an **always-built** `FrameworkSourceTest` target in `test/Tuning/CMakeLists.txt`; link contracts/source candidates/MicroMapping/target libraries and GTest, without an E2E guard. Runtime execution additions are separately guarded by `LLK_BUILD_E2E_TESTS`. Current candidate-instantiation/tuning GTests are guarded; generic static contracts must also be tested in non-JIT builds.

## B1: Add a rank-neutral versioned workload contract

**Files:** Create `include/LLK/Conversion/Workload/WorkloadContract.h`, `lib/Conversion/Workload/WorkloadContract.cpp`, `test/Tuning/workload_contract.cpp`; Modify root `CMakeLists.txt`, `test/Tuning/CMakeLists.txt`.

**Interfaces:** Produces `WorkloadContract` and serialization APIs above; no contraction adapter or runtime change.

- [ ] **Write failing tests**:

```cpp
TEST(WorkloadContract, ThreeInputsAndTwoResultsNeedNoContractionAxes) {
  workload::WorkloadContract contract;
  contract.numericalContract = "strict-f32";
  for (unsigned i = 0; i != 3; ++i)
    contract.ports.push_back({"input" + std::to_string(i),
        workload::PortDirection::Input, workload::PortEffect::Read,
        {2, 3, 4}, "f32"});
  contract.ports.push_back({"sum", workload::PortDirection::Output,
      workload::PortEffect::Write, {2, 3, 4}, "f32"});
  contract.ports.push_back({"product", workload::PortDirection::Output,
      workload::PortEffect::Write, {2, 3, 4}, "f32"});
  EXPECT_FALSE(workload::validateWorkloadContract(contract));
  mlir::MLIRContext context;
  auto restored = workload::decodeWorkloadContract(
      workload::encodeWorkloadContract(&context, contract));
  ASSERT_TRUE(bool(restored)) << llvm::toString(restored.takeError());
  EXPECT_EQ(workload::canonicalWorkloadContract(*restored),
            workload::canonicalWorkloadContract(contract));
  auto changed = contract;
  std::swap(changed.ports[0], changed.ports[1]);
  EXPECT_NE(workload::canonicalWorkloadContract(changed),
            workload::canonicalWorkloadContract(contract));
}
```

Add tests for dynamic/zero extents, duplicate names, unknown versions, dtype/effect/numerical changes, malformed attributes and delimiter characters in names. Assert errors name the field. Do not validate a contract by inventing M=N=K=1.
- [ ] **Run red:** configure/build `FrameworkSourceTest`; expected compile failure on the new interface, then failing validation/round-trip assertions until implemented.
- [ ] **Implement the schema** with explicit version/direction/effect fields and ordered ports. Encode axes in lexical order, preserve port order, sort/deduplicate primitive requirements, and length-delimit all canonical strings. Validate decoded input before returning it. Use `llvm::StringError` with stable `workload_contract:` prefixes. Add root CMake registrations; keep contract library below source-candidate/perf integration.

```cpp
// Canonical field encoding used for strings, not separator concatenation.
static void appendField(std::string &out, llvm::StringRef value) {
  out += std::to_string(value.size());
  out += ':';
  out.append(value.data(), value.size());
}
```

Include field tags, version, counts and each shape extent in the canonical rendering. A hash is an index into this full identity, never a substitute for it.
- [ ] **Run green:** `cmake --build build --target FrameworkSourceTest -j 6`; `build/bin/FrameworkSourceTest '--gtest_filter=WorkloadContract.*'`. Expected: all proposed contract cases run and pass.
- [ ] **Commit:** stage the named files; `git commit -m 'feat: describe generic workload ports and semantics'`.

## B2: Tune exact direct MicroIR without contraction provenance

**Files:** Create `test/Tuning/FrameworkTestSupport.h`, `test/Tuning/generic_source_candidates.cpp`; Modify `include/LLK/Conversion/MicroMapping/CandidateInstantiation.h`, `lib/Conversion/MicroMapping/CandidateInstantiation.cpp`, root `CMakeLists.txt`, `test/Tuning/CMakeLists.txt`. Link `LLKMicroMapping` to `LLKWorkloadContracts` for the descriptor serialization symbols.

**Interfaces:** Add a descriptor overload to the current API:

```cpp
llvm::Expected<tuning::CandidateInstance> tuning::instantiateCandidate(
    ModuleOp source, const perf::SearchSpace &, const perf::Candidate &,
    const workload::WorkloadContract &,
    const tuning::CandidateInstantiationOptions &);
```

Initially this overload supports `SourceMode::ConcreteMicro` with no source parameters. Add `workloadContract` to `CandidateInstance`. The old WorkloadShape overload keeps its checks until B4 wraps it.

- [ ] **Write failing test** with no `original_workload` attribute:

```cpp
TEST_F(FrameworkTest, DirectElementwiseNeedsNoMmaFacts) {
  auto source = parse(R"mlir(
module {
  micro.kernel @add(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>)
      -> tensor<8x8xf32> {
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>}
      : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b {shape = array<i64: 8, 8>}
      : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %s = micro.vector "add" %ta, %tb
      : !micro.tile<8x8xf32, memory = #micro.memory<sram>>,
        !micro.tile<8x8xf32, memory = #micro.memory<sram>>
        -> !micro.tile<8x8xf32, memory = #micro.memory<acc>>
    micro.yield %s : !micro.tile<8x8xf32, memory = #micro.memory<acc>>
  }
})mlir");
  ASSERT_TRUE(source);
  const std::string before = print(source->getOperation());
  workload::WorkloadContract contract;
  contract.numericalContract = "strict-f32";
  contract.ports = {
    {"input0", workload::PortDirection::Input, workload::PortEffect::Read,
      {8, 8}, "f32"},
    {"input1", workload::PortDirection::Input, workload::PortEffect::Read,
      {8, 8}, "f32"},
    {"output0", workload::PortDirection::Output, workload::PortEffect::Write,
      {8, 8}, "f32"}};
  contract.requiredPrimitives = {"micro.vector"};
  perf::SearchSpace space;
  space.name = "direct";
  space.workload = "diagnostic-only-label";
  perf::Candidate candidate;
  candidate.id = perf::computeCandidateId({}, {});
  tuning::CandidateInstantiationOptions options;
  options.sourceMode = tuning::SourceMode::ConcreteMicro;
  options.sourceSymbol = "add";
  auto instance = tuning::instantiateCandidate(*source, space, candidate,
                                               contract, options);
  ASSERT_TRUE(bool(instance)) << llvm::toString(instance.takeError());
  EXPECT_TRUE(instance->bindingFacts.contractions.empty());
  EXPECT_EQ(print(source->getOperation()), before);
  EXPECT_NE(instance->sourceGraphHash, 0u);
  EXPECT_FALSE(instance->workloadGraph.getNodes().empty());
}
```

The no-parameter in-memory singleton is intentional; parsed search IR still uses the verifier's existing nonempty-param requirement until B6 defines singleton serialization.
- [ ] **Run red:** build/run `FrameworkSourceTest` filter `FrameworkTest.DirectElementwiseNeedsNoMmaFacts`. Expected: the overload is missing, then old concrete checks reject the non-contraction source.
- [ ] **Implement cloning and signature-derived contract checks**. Resolve the selected kernel by `sourceSymbol`, or require exactly one. Check every ordered input/result shape/dtype against `function_type`, derive required primitive names from its actual body, and record `micro.workload_contract = encodeWorkloadContract(...)` on the private clone. Use the original canonical module text plus canonical contract for source identity; extract the existing workload graph. Permit empty contraction facts. Reject multiple/unknown symbols, a mismatched port, absent numerical metadata when the contract requires it, and any offered source parameter not yet realized. Keep the caller source unchanged.
- [ ] **Run green:** the new direct test plus mismatch/ambiguity/changed-label/changed-semantics tests; run existing `CandidateInstantiationTest` to preserve legacy behavior. Expected: a workload label does not dispatch or alter semantics.
- [ ] **Commit:** stage the named files; `git commit -m 'feat: instantiate non-contraction MicroIR sources'`.

## B3: Define adapter dispatch and parameter realization receipts

**Files:** Create `include/LLK/Conversion/Workload/SourceAdapter.h`, `lib/Conversion/Workload/SourceAdapter.cpp`, `lib/Conversion/Workload/BindingSchema.cpp`, `test/Tuning/source_adapter_registry.cpp`; Modify root `CMakeLists.txt`, `include/LLK/Conversion/MicroMapping/CandidateInstantiation.h`, `lib/Conversion/MicroMapping/CandidateInstantiation.cpp`, `test/Tuning/CMakeLists.txt`.

**Interfaces:** Produces the adapter/registry/schema APIs above. Add `std::string adapterKey` to `CandidateInstantiationOptions` and a further overload taking `const SourceAdapterRegistry &`; the descriptor overload without a registry constructs defaults. Add `BindingReceipt receipt` to `CandidateInstance`.

- [ ] **Write failing schema test**:

```cpp
TEST(BindingSchema, AnOfferedChoiceCannotDisappear) {
  perf::SearchSpace space;
  perf::SearchParam p;
  p.name = "T0"; p.kind = "integer"; p.role = "axis:d0";
  p.choices.emplace_back(int64_t{4});
  space.params.push_back(p);
  perf::Candidate candidate;
  candidate.values["T0"] = 4;
  candidate.id = perf::computeCandidateId(candidate.values, {});
  std::vector<workload::ParameterBinding> schema = {
      {"T0", workload::BindingOwner::Source, "axis:d0"}};
  workload::BindingReceipt receipt;
  auto missing = workload::validateBindingReceipt(space, candidate,
                                                   schema, receipt);
  ASSERT_TRUE(bool(missing));
  EXPECT_NE(llvm::toString(std::move(missing)).find("axis_not_realized: T0"),
            std::string::npos);
  receipt.source = {"T0"};
  EXPECT_FALSE(workload::validateBindingReceipt(space, candidate,
                                                schema, receipt));
  receipt.mapping = {"T0"};
  auto duplicate = workload::validateBindingReceipt(space, candidate,
                                                     schema, receipt);
  ASSERT_TRUE(bool(duplicate));
  llvm::consumeError(std::move(duplicate));
}
```

`SearchChoice` uses its existing `int64_t` constructor. Add a small test-only `SourceAdapter` class implementing all virtual methods, with two adapters accepting the same `linalg.generic`; assert default resolution errors and explicit key succeeds. Duplicate keys, null adapters and unknown explicit keys error.
- [ ] **Run red:** build/run `BindingSchema.*` and `SourceAdapterRegistry.*`; expected missing APIs or absent realization/ambiguity rejection.
- [ ] **Implement exact receipt validation**: candidate keys equal parameter names; each candidate value is in its declared domain; each parameter has exactly one schema owner; exactly one matching receipt entry appears in its owner list, and none in the other list. Reject duplicate roles where a role identifies one source axis or mapping port. For source axes accept explicit `role="axis:d0"`/`axis:d1`; never infer by spelling `T0`. For mapping choices reuse existing layout/owner_mapping/memory_path kinds and role-aware projection. Contract/shape-specific legality remains inside the adapter; machine constraints remain in existing legality/mapping.

```cpp
// Immutable after configuration; resolve performs a complete scan.
// Exact outcomes: 0 matches -> unsupported_source_adapter;
// 1 match -> selected adapter; >1 -> ambiguous_source_adapter.
```

Return stable diagnostics naming the parameter/role/adapter. Receipt entries are appended only after the corresponding IR or selected mapping choice is actually realized. Exposing a schema alone is not consumption evidence.
- [ ] **Run green:** registry/schema tests, existing generator/domain tests, and B2 direct singleton. Expected: no silent first-match or unused integer bindings.
- [ ] **Commit:** stage named files; `git commit -m 'feat: declare source adapters and binding ownership'`.

## B4: Wrap current contraction export behind the generic boundary

**Files:** Create `lib/Conversion/Workload/ContractionSourceAdapter.cpp`; Modify `lib/Conversion/Workload/SourceAdapter.cpp`, `lib/Conversion/MicroMapping/CandidateInstantiation.cpp`, root `CMakeLists.txt`; Test `test/Tuning/candidate_instantiation.cpp`, `test/Tuning/generic_source_candidates.cpp`.

**Interfaces:** Default adapter key `llk-contraction-v1`; accepts the currently supported LLK/Linalg contraction roots. Uses existing `exportMicroKernelFromSchedule(ModuleOp, StringRef, uint64_t, const ScheduleEntry &)` and returns `SourceRealization`. The existing WorkloadShape overload translates supplied M/N/K/dtypes into a contract and delegates, retaining strict mismatch checks.

- [ ] **Write failing compatibility test** in the existing candidate-instantiation test file, reusing its defined `testSpace`, `candidate`, `printModule` and `two_roots.mlir` setup. Instantiate the same selected root through both overloads, and compare primitive sequences, ordered signature types, original/bound contraction facts, init reads and source numerical attributes. The decisive assertions are:

```cpp
EXPECT_EQ(oldInstance->bindingFacts.contractions.size(),
          newInstance->bindingFacts.contractions.size());
EXPECT_EQ(oldInstance->kernel->getAttr("function_type"),
          newInstance->kernel->getAttr("function_type"));
EXPECT_EQ(oldInstance->kernel->getAttr("source_math_mode"),
          newInstance->kernel->getAttr("source_math_mode"));
EXPECT_EQ(printModule(*source), originalSource);
```

Add SwiGLU two-arm tests asserting two MMAs followed by silu/mul and output conversion, preserving multi-branch dataflow. The new generic provenance attribute may differ; compare semantics, not full text including the new metadata.
- [ ] **Run red:** build/run both candidate-instantiation targets; expected adapter missing or descriptor path unsupported for contraction source.
- [ ] **Implement the adapter** by moving contraction-specific shape/schedule realization out of generic `instantiateCandidate`. Keep BM/BN/BK/fragment/tail legality and typed `BindingFacts` in this adapter. Translate legacy axis names explicitly in its `bindings` method. Attach the new contract alongside `original_workload`; keep old attributes for existing consumers/replay readers. If #142 is merged, reuse its `LLKKernelRegistry` at this boundary; its M/N/K/arm APIs remain specialized. Generic dispatch must contain no checks for `fused_swiglu` or `matmul` names.
- [ ] **Run green:** existing semantic-root, wrong-axis, init, numerical-identity, SwiGLU and Linalg-matmul tests, plus old/new equivalence. Expected: source acceptance and rejection stay compatible; identities include new versioned metadata and do not silently reuse pre-contract records.
- [ ] **Commit:** stage the named files; `git commit -m 'refactor: isolate contraction source candidate realization'`.

## B5: Import a bounded structured elementwise workload

**Files:** Create `lib/Conversion/Workload/ElementwiseSourceAdapter.cpp`, `test/Tuning/Inputs/framework/elementwise.mlir`, `test/Tuning/elementwise_source_adapter.cpp`; Modify `lib/Conversion/Workload/SourceAdapter.cpp`, root `CMakeLists.txt`, `test/Tuning/CMakeLists.txt`.

**Interfaces:** Default adapter key `linalg-elementwise-v1`; accepts `linalg.generic` roots, then validates this increment's support: static positive rank-two f32 tensors, two inputs, one destination/result, identity indexing maps, two parallel iterators, one `arith.addf` or `arith.mulf` yielding its result, no destination-init reads, no other effects. `describe` produces named ports/axes `d0`, `d1`; `bindings` accepts source tile roles `axis:d0`, `axis:d1` and declared mapping choices.

- [ ] **Write this source fixture and failing importer test**:

```mlir
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @elementwise(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>)
      -> tensor<8x16xf32> {
    %init = tensor.empty() : tensor<8x16xf32>
    %out = linalg.generic {
      indexing_maps = [#id, #id, #id],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<8x16xf32>, tensor<8x16xf32>)
      outs(%init : tensor<8x16xf32>) {
      ^bb0(%x: f32, %y: f32, %unused: f32):
        %sum = arith.addf %x, %y : f32
        linalg.yield %sum : f32
      } -> tensor<8x16xf32>
    return %out : tensor<8x16xf32>
  }
}
```

Load it with `llvm::MemoryBuffer::getFile` and `FrameworkTest::parse`; build the default registry and call `selectSourceRoot(...,{"elementwise",0},registry,"linalg-elementwise-v1")`. Assert describe yields two read inputs, a destination input with `PortEffect::None`, one write result, `d0=8/d1=16`, no K, and required `micro.vector`. Instantiate a singleton full-tile candidate and assert `mlir::verify` succeeds, exactly one vector add, no MMA, no remaining Linalg/LLK semantic op inside the kernel and unchanged source module.

```cpp
auto root = workload::selectSourceRoot(*source, {"elementwise", 0},
                                       registry, "linalg-elementwise-v1");
ASSERT_TRUE(bool(root)) << llvm::toString(root.takeError());
auto adapter = registry.resolve(*root, "linalg-elementwise-v1");
ASSERT_TRUE(bool(adapter)) << llvm::toString(adapter.takeError());
auto contract = (*adapter)->describe(*root);
ASSERT_TRUE(bool(contract)) << llvm::toString(contract.takeError());
EXPECT_EQ(contract->axes.at("d0"), 8);
EXPECT_EQ(contract->axes.count("K"), 0u);
```

Add exact negative variants: change an iterator to reduction, swap one indexing map, dynamic extent, f64, scalar constant added to the body, init operand read, second result, or nonempty fastmath flags. Each rejects with a reason naming the unsupported feature; a rejected operation must not reach a contraction adapter. The generic descriptor still supports multiple result ports; this particular structured adapter declares its single-result envelope.
- [ ] **Run red:** `FrameworkSourceTest` filter `ElementwiseSourceAdapter.*`; expected unsupported adapter/structure.
- [ ] **Implement the validator and full-tile emitter**. Preserve operand order and every source numerical/fastmath attribute in canonical provenance. The first supported numerical mode is strict f32; reject approximation flags. Construct the `micro.kernel` signature from all root operands/results, using the existing destination-passing export pattern. Emit views, staged copies/waits and one vector operation, then yield the tile result. The scalar operation determines a primitive key inside this adapter only:

```cpp
StringRef primitive = isa<arith::AddFOp>(bodyOp) ? "add" : "mul";
Value result = micro::VectorOp::create(
    builder, loc, resultTileType, primitive, ValueRange{left, right},
    /*math_mode=*/StringAttr());
```

`bodyOp`, `builder`, `loc`, `left`, `right`, `resultTileType` are local variables in the adapter's `realize` method: validated scalar op, local OpBuilder/location, staged operand values and the f32 result TileType. Define memory/layout/owner from the declared schedule/target-independent defaults, never from a target-name test. Start with full extents when no tile parameters are offered; B6 adds smaller tiled realization. Source function-local values become kernel block arguments, preserving `IsolatedFromAbove`. A local `tensor.empty` destination has no incoming reads or materialized movement until used.
- [ ] **Run green:** importer positive/negative tests; lower the full tile through existing MicroToLinalg/reference path and verify its IR. Expected: correct port/dataflow and explicit unsupported semantics.
- [ ] **Commit:** stage named files; `git commit -m 'feat: import bounded Linalg elementwise programs to MicroIR'`.

## B6: Realize generic tiled candidates and integrate mapped tuning

**Files:** Modify `lib/Conversion/Workload/ElementwiseSourceAdapter.cpp`, `lib/Conversion/MicroMapping/CandidateInstantiation.cpp`, `include/LLK/Conversion/MicroMapping/MappedTuningSession.h`, `lib/Conversion/MicroMapping/MappedTuningSession.cpp`, `include/LLK/Perf/SearchSpace.h`, `lib/Perf/SearchSpace.cpp`, `include/LLK/Dialect/Micro/MicroOps.td`, `lib/Dialect/Micro/MicroOps.cpp`, `test/FileCheck.cmake`; Create `test/Dialect/Micro/search_space_singleton.mlir`, `test/Dialect/Micro/search_space_singleton_invalid.mlir`; Test `test/Tuning/elementwise_source_adapter.cpp`, `test/Tuning/generic_mapped_tuning.cpp`; Modify `test/Tuning/CMakeLists.txt`.

**Interfaces:** Add `runMappedTuningSession(ModuleOp, const SearchSpace &, const workload::WorkloadContract &, const MappingTarget &, const MappedTuningOptions &)` while preserving the WorkloadShape overload. Mapping receives the existing primitive WorkloadGraph and SearchBinding; source choices remain adapter-owned. No second generator or covering solver.

- [ ] **Write failing finite-domain test**. Declare `T0` choices `{4,8}` with `role="axis:d0"`, `T1` `{8,16}` with `role="axis:d1"`; use `perf::generateCandidates(space,{})`. For each of the four bindings, instantiate the same source and extract `micro.spatial_for` steps/tile-view shapes. Require four distinct concrete schedule identities, identical source identity and unchanged input source. Add a nondividing T0=3 choice which rejects as `source_legality: tile_divisibility`, and a `vector_width` integer with no realization which rejects as `axis_not_realized`.

```cpp
EXPECT_EQ(generated.size(), 4u);
EXPECT_EQ(sourceIdentities.size(), 1u);
EXPECT_EQ(instantiatedScheduleIdentities.size(), 4u);
EXPECT_EQ(print(source->getOperation()), before);
```

The sets above are `std::set<std::string>` filled with canonical source/realized-kernel text from each successful candidate; strip only candidate-ID attributes when comparing the actual schedule, not operation semantics. Add AVX2 modeled tuning and selected compilation at `MappedStop::Lowered`, with stable ordering and literal domain count. The generic accelerator's audited package lacks rules for several explicitly emitted movement primitives; record that unsupported envelope until C adds the selected primitive policy fixtures. Do not assume all source adapters map to every profile.
- [ ] **Run red:** build/run generic mapped/source tests; expected smaller tiles unrealized and contract tuner missing.
- [ ] **Implement nested spatial tiling**. Reuse the current MicroIR SSA-carried output pattern: allocate output, open d0/d1 spatial loops at declared T0/T1, create two views at loop coordinates, stage operands, compute add/mul, tile-store into the carried destination, yield through both loops, and yield the final output. The source handler records receipt entries only after loops/views are emitted. For this increment reject nondividing tile sizes; do not insert implicit padding, masking or reassociation.

```cpp
if (t0 <= 0 || t1 <= 0 || extent0 % t0 || extent1 % t1)
  return llvm::createStringError(
      "source_legality: tile_divisibility: T0/T1 must divide d0/d1");
```

Factor the mapped tuner so the descriptor overload instantiates via adapters and then uses the existing complete evaluator/search/compiler/measurement sequence. Existing `projectAxes` stays role-aware; reject unconsumed source choices and verify mapping receipt against the selected plan's resolved layout/owner/path. Contraction-only constraints reject missing contraction facts, while generic shape-independent constraints continue to work. For serialized no-param singleton sources, allow zero params only when `micro.search_space` explicitly carries `singleton = true`; verify that no candidate has bindings and there is at most one explicit candidate. Add `bool singleton = false` to `perf::SearchSpace`, a defaulted BoolAttr to SearchSpaceOp, loader and verifier, with round-trip/invalid FileCheck registrations in `test/FileCheck.cmake`. Do not change regular empty spaces silently. After source realization validate ReceiptStage::Source. Inside the injected complete-plan evaluator, build plan-specific mapping receipts and validate ReceiptStage::Complete before the proposal can be retained. A failed mapping receipt is a candidate rejection and search continues; it is not a post-topK filter. Persist the accepted receipt with the selected candidate.
- [ ] **Run green:** source/mapped/loader/verifier tests, capacity-before-topK tests and old mapped tuning. Expected: bounded domain and truncation metadata remain truthful; infeasible/unused choices reject before retention.
- [ ] **Commit:** stage named files plus the singleton IR/verifier/FileCheck changes; `git commit -m 'feat: search realized non-contraction source schedules'`.

## B7: Expose generic CLI, replay and numerical acceptance

**Files:** Modify `tools/llk-tune/llk-tune.cpp`, `tools/llk-compile/llk-compile.cpp`, `lib/Conversion/MicroMapping/MappedTuningSession.cpp`, `lib/Perf/ScheduleRecord.cpp`; Create `test/Tuning/generic_tune_cli.py`, `test/Execution/generic_elementwise.cpp`; Modify Tuning/Execution CMake, `docs/manual/source/{features.rst,architecture.rst}`, tool reference pages and mapping workflow.

**Interfaces:** New optional CLI `--source-adapter=<key>`; the selected source supplies its contract. Existing explicit M/N/K options remain for legacy/synthetic modes, and supplying them for a non-contraction adapter is an error. Generic frozen records add contract version/canonical content, adapter key/version and realization receipt, preserving the existing source and full mapping plan.

- [ ] **Write failing CLI acceptance** in a new Python test. Run `llk-tune` on the structured fixture with `--source-adapter=linalg-elementwise-v1`, target-aware options and four candidate tile bindings; emit the selected source/plan using the existing replay flags. Recompile that frozen artifact with `llk-compile`, and compare contract/source/binding/plan identity. Change one source attribute, port extent, target rule or execution mode and assert reuse refuses or yields a different identity. Required assertions:

```python
assert report["generated"] == 4
assert report["source_contract"]["version"] == 1
assert report["source_adapter"] == "linalg-elementwise-v1"
assert replay_identity == selected_identity
assert changed_identity != selected_identity
```

These new JSON fields are produced by this task; retain existing report fields. Use a `subprocess.run(..., check=True, capture_output=True, text=True)` helper and pass tool paths/source/build-artifact directories as CTest arguments. Add a runtime test with literal inputs `a[i]=i/8`, `b[i]=(127-i)/16`, result `a[i]+b[i]` (and multiply variant); execute full and tiled selected/reference versions and compare every f32 output to the scalar expression with strict f32 tolerance. Test changed input-init buffers cannot affect the elementwise result.
- [ ] **Run red:** build tools/tests, run `GenericTuneCLI` and new execution GTests. Expected: generic CLI/record fields missing; no adapter-based numerical fixture.
- [ ] **Implement contract-based CLI flow**: infer selected root contract before candidate generation; call the descriptor overload; keep contraction-only shape flags in its adapter path. Persist the exact instantiated module and existing frozen decision plan; replay binds it without regenerating source choices. Reject ABI-v1 incompatible rank/dtype/descriptor count/effects at executable preparation with `runtime_capability:` diagnostics. Add static rank-three direct-MicroIR contract/report test proving analysis remains possible while execution fails explicitly. Do not flatten shapes or reinterpret rank-two descriptors.

```cpp
if (requestExecutable && llvm::any_of(contract.ports, [](const auto &port) {
      return port.shape.size() != 2;
    }))
  return llvm::createStringError(
      "runtime_capability: mapped ABI v1 requires rank-two ports");
```

Use the existing full identity construction and typed numerical-verifier hooks. Include adapter/contract content in identity, not only a workload label. Apply invocation-rank gating only at `MappedStop::Executable` or a measured invocation request, not merely because binding uses `requireExecutable=true`; rank-one reduction may still reach verified selected Lowered IR. Selected tests must assert `referenceGroupsLowered == 0`, positive backend realization and the chosen vector width in emitted evidence. Reference tests remain separately named.
- [ ] **Run green:** `GenericTuneCLI`, contract/source/tuning tests, existing frozen-replay/typed invocation tests and the full suite. Require selected AVX2 execution on supported x86 CI; report host limitations on other machines. Strict Sphinx and documentation-reference checks must pass after adding the supported contract and limits to the manual.
- [ ] **Commit:** stage the named files; `git commit -m 'feat: expose generic source search and frozen replay'`.

## Increment acceptance

- [ ] A direct non-MMA kernel can enter mapped tuning without fake contraction facts.
- [ ] The bounded structured elementwise adapter produces multiple genuinely different source schedules with correct output and preserved numerical contract.
- [ ] Generic planner/analysis contains no matmul/SwiGLU dispatch; legacy adapters retain init/tail/dtype/fusion behavior.
- [ ] Every offered binding has exactly one realized owner; truncation/domain/replay information is reproducible.
- [ ] Rank-neutral descriptions are distinct from rank-two execution support; unsupported structured/effect/runtime cases have explicit diagnostics.
- [ ] Capability docs and PR evidence describe exactly this increment's support.
