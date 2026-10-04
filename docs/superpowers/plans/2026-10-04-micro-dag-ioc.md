# MicroDAG Operation-Builder IoC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make leaf-operation DAG extraction injectable while preserving the default simulator, then independently fix missing movement data dependencies.

**Architecture:** An owning operation-name registry supplies typed handlers to DAGBuilder. An operation-scoped context delegates graph mutations to the core; traversal and State remain private. The existing two-argument buildMicroDAG delegates to built-in registrations.

**Tech Stack:** C++20, LLVM/MLIR, GoogleTest, CMake/CTest; -fno-rtti/-fno-exceptions.

**Spec:** `docs/superpowers/specs/2026-10-04-micro-dag-ioc-design.md` (approved).

## Global Constraints

- Work only in an isolated worktree under `.claude/worktrees/`; main is read-only.
- Start from the current integrated main plus the approved spec/plan. Preserve concurrent mapping changes, especially concrete route identity. The design baseline is 31fd9eb, not a requirement to revert newer implementation.
- Preserve costs, resources, event order, warnings, storage factors, and unknown-operation fallback during mechanical extraction.
- Keep ForOp, SpatialForOp, PipelineOp, and YieldOp traversal core-owned; reject registration for those names.
- No mutable global registration, RTTI, exceptions, dynamic plugin loader, new Micro ODS, or scheduler changes.
- Stored callbacks own their captures; never store llvm::function_ref. Contexts are non-copyable, scoped to one handler invocation, and never retained.
- Registration errors and handler errors use llvm::Error. Explicit replacement must find an existing registration.
- Register every new test with add_llk_perf_test in root CMakeLists.txt.
- Use current MLIR APIs and repository LLVM-version rules; do not assume local LLVM matches CI.
- Commit the extraction and dependency correction separately.

## Review Focus

1. Captured callback configuration outlives registration scope and remains isolated between registries — Task 2 ownership test and Task 3 independent registries.
2. A handler failure cannot fall through as free work or leave a reusable partially successful graph — Task 3 propagation test.
3. A structural registration must fail explicitly rather than appear to override a bypassed handler — Task 2 rejection test.
4. Zero-event aliases and multi-event movement must bind producers correctly across loop/pipeline state — Task 3 graph tests, Task 4 dependent-copy test.
5. Default callers preserve output on both targets, including mapped route identity introduced concurrently — Task 5 comparison and route suites.

## File map

- Create `include/LLK/Perf/TileInfo.h`: public tile description without generated Micro operation classes.
- Create `lib/Perf/TileInfo.cpp`: type decoding using the existing implementations.
- Modify `lib/Perf/MicroTileInfo.h`: private compatibility wrapper retaining generated classes needed by existing consumers.
- Create `include/LLK/Perf/OpDAGBuilder.h`: owning registry and restricted abstract context.
- Create `lib/Perf/OpDAGBuilder.cpp`: registry validation and lookup.
- Create `lib/Perf/MicroOpDAGBuilders.cpp`: built-in typed operation handlers and default factory.
- Modify `include/LLK/Perf/MicroDAG.h`: injected overload and API documentation.
- Modify `lib/Perf/MicroDAG.cpp`: registry injection, private context adapter, central emission and traversal.
- Create `test/Perf/op_dag_builders.cpp`: extension, ownership, error, and equivalence tests.
- Modify `test/Perf/l1_resource_dag.cpp`: dependent movement regression and overlap assertions.
- Modify `CMakeLists.txt`: source/test registration.
- Modify `docs/design/micro-ir-mapping-workflow.md`: API extension example and compatibility boundary.

---

### Task 1: Expose tile descriptions without generated operation headers

**Files:** TileInfo.h/.cpp, MicroTileInfo.h, CMakeLists.txt, test/Perf/op_dag_builders.cpp.

**Interfaces:** Produces the unchanged `TileInfo` fields/methods, `TileInfo describeType(mlir::Type)`, and `std::string shapeString(llvm::ArrayRef<int64_t>)` in mlir::llk::perf.

- [ ] Add a test file including only public perf headers plus ordinary MLIR types. Add it through `add_llk_perf_test(OpDAGBuildersTest test/Perf/op_dag_builders.cpp)`.

```cpp
TEST(OpDAGBuilders, PublicTileDescriptionRetainsTensorSize) {
  mlir::MLIRContext context;
  auto type = mlir::RankedTensorType::get(
      {8, 16}, mlir::Float32Type::get(&context));
  TileInfo info = describeType(type);
  EXPECT_FALSE(info.isTile);
  EXPECT_EQ(info.dtype, "f32");
  EXPECT_EQ(info.elements(), 128u);
  EXPECT_EQ(info.bytes(), 512u);
  EXPECT_EQ(shapeString(info.shape), "8x16");
}
```

- [ ] Run `cmake --build build --target OpDAGBuildersTest`; expect the missing public header/API to fail before implementation.
- [ ] Move the existing TileInfo definition into the public header, including only MLIR types and required LLVM/STL containers. Move describeType/shapeString bodies unchanged to TileInfo.cpp. Keep the private MicroTileInfo.h wrapper's generated attribute/type/op includes, replacing its old definitions with `#include "LLK/Perf/TileInfo.h"`. Add TileInfo.cpp to LLKPerf.
- [ ] Build and run OpDAGBuildersTest, L0StaticBoundTest, and L1ResourceDAGTest. Include dynamic-shape and unsupported-dtype assertions preserving zero-byte behavior.
- [ ] Commit: `refactor(perf): expose reusable tile descriptions`.

### Task 2: Add the owning registry and context contract

**Files:** OpDAGBuilder.h/.cpp, CMakeLists.txt, op_dag_builders.cpp.

**Interfaces:** Consumes TileInfo and MicroEvent. Produces the registry contract and context signatures below; implementation of the context is Task 3.

```cpp
class OpDAGBuildContext {
public:
  virtual ~OpDAGBuildContext() = default;
  OpDAGBuildContext(const OpDAGBuildContext &) = delete;
  OpDAGBuildContext &operator=(const OpDAGBuildContext &) = delete;
  virtual mlir::Operation &operation() const = 0;
  virtual const machine::MachineModel &machineModel() const = 0;
  virtual llvm::StringRef kernelName() const = 0;
  virtual uint32_t emitEvent(MicroEvent event, mlir::ValueRange inputs,
                            mlir::ValueRange outputs,
                            llvm::ArrayRef<uint32_t> extraDeps = {}) = 0;
  virtual void aliasValue(mlir::Value result, mlir::Value source) = 0;
  virtual std::string memoryOf(mlir::Value value) const = 0;
  virtual void accountStorage(llvm::StringRef memory, uint64_t bytes) = 0;
  virtual void noteUnsizable(const TileInfo &info) = 0;
  virtual void noteWarning(std::string message) = 0;
  virtual void noteLayoutUsage(llvm::StringRef engine,
      const std::vector<std::string> &supported, llvm::StringRef layout) = 0;
  virtual llvm::Error buildLogicalTile(mlir::Value source, mlir::Value result,
      const std::optional<std::string> &owner, EventKind kind) = 0;
  virtual llvm::Error buildMovement(llvm::StringRef src, llvm::StringRef dst,
      const TileInfo &source, const TileInfo &result, EventKind kind,
      llvm::ArrayRef<mlir::Value> outputs) = 0;
  virtual llvm::Error noteAllocation(const TileInfo &info,
                                    llvm::StringRef memory) = 0;
  virtual const machine::ComputeNode *pickMatrixEngine(
      llvm::StringRef requested, std::string &reason) const = 0;
  virtual const machine::ComputeNode *pickVectorEngine(
      std::string &reason) const = 0;
  virtual uint64_t vectorCycles(const machine::ComputeNode *engine,
      llvm::StringRef dtype, uint64_t elements) const = 0;
  virtual uint64_t mmaCycles(const machine::ComputeNode &engine,
      llvm::ArrayRef<int64_t> shape, uint64_t flops) const = 0;
protected:
  OpDAGBuildContext() = default;
};

class OpDAGBuilderRegistry {
public:
  using Callback = std::function<llvm::Error(
      mlir::Operation &, OpDAGBuildContext &)>;
  llvm::Error registerBuilder(llvm::StringRef name, Callback callback);
  llvm::Error replaceBuilder(llvm::StringRef name, Callback callback);
  const Callback *lookup(llvm::StringRef name) const;
  template <typename OpT, typename Fn>
  llvm::Error registerBuilder(Fn fn);
  template <typename OpT, typename Fn>
  llvm::Error replaceBuilder(Fn fn);
private:
  llvm::StringMap<Callback> builders;
};
OpDAGBuilderRegistry makeDefaultOpDAGBuilders();
```

The registry permits raw operation names for externally defined operations as well as typed adapters. Reject empty names, empty callbacks, duplicates, absent replacements, and the four structural names. Typed adapters use MLIR dyn_cast and return an error on a type mismatch instead of aborting. Include mlir/IR/ValueRange.h and Operation.h where required; verify the installed MLIR header location rather than assuming a transitive include. Do not modify canonical ODS.

- [ ] Add failing tests for duplicate registration, absent replacement, empty callback/name, and each structural name.

```cpp
TEST(OpDAGBuilders, DuplicateRegistrationFails) {
  OpDAGBuilderRegistry registry;
  auto fn = [](mlir::Operation &, OpDAGBuildContext &) {
    return llvm::Error::success();
  };
  ASSERT_FALSE(registry.registerBuilder("test.probe", fn));
  llvm::Error error = registry.registerBuilder("test.probe", fn);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("test.probe"),
            std::string::npos);
}
```

Use `llvm::toString` to consume every expected error. Test that a callback capturing `std::string("owned")` by value remains callable after its defining scope ends; invoke it through a test-only minimal context implementing the declared facade.

- [ ] Build/run OpDAGBuildersTest and observe failure before adding registry implementation.
- [ ] Implement lookup/registration/replacement. Keep lookup read-only and avoid iteration-based dispatch. Add OpDAGBuilder.cpp to LLKPerf. Fill the test-only context with deterministic no-op services and simple event-ID recording; its engine queries return null with a reason.
- [ ] Run OpDAGBuildersTest. Confirm public headers compile without generated operation classes and callbacks require no C++ RTTI.
- [ ] Commit: `feat(perf): add injectable operation-builder registry`.

### Task 3: Inject the registry and extract built-in handlers

**Files:** MicroDAG.h/.cpp, MicroOpDAGBuilders.cpp, CMakeLists.txt, op_dag_builders.cpp.

**Interfaces:** Consumes Task 2 registry/context. Produces the three-argument buildMicroDAG overload and default factory; the existing entry point delegates.

- [ ] Add a parsed fixture helper local to op_dag_builders.cpp using DialectRegistry, registerAllDialects, MicroDialect, parseSourceString<ModuleOp>, verify, and findMicroKernel. Keep MLIRContext alive through the graph build.
- [ ] Add a failing injection test using a verified kernel containing an arith.constant. Register the typed arith::ConstantOp handler and emit one synthetic event.

```cpp
// Kernel fixture:
// module { micro.kernel @probe { %c = arith.constant 0 : index
//                                micro.yield } }
auto builders = makeDefaultOpDAGBuilders();
llvm::Error error = builders.registerBuilder<mlir::arith::ConstantOp>(
    [](mlir::arith::ConstantOp op, OpDAGBuildContext &context) {
      MicroEvent event;
      event.kind = EventKind::Vector;
      event.resource = ResourceKind::VectorEngine;
      event.resourceName = "probe";
      event.sourceOpName = op->getName().getStringRef().str();
      event.minCycles = 7;
      context.emitEvent(std::move(event), {}, op->getResults());
      return llvm::Error::success();
    });
ASSERT_FALSE(error);
auto dag = buildMicroDAG(parsed->kernel, model, builders);
ASSERT_TRUE(static_cast<bool>(dag)) << llvm::toString(dag.takeError());
ASSERT_EQ(dag->events.size(), 1u);
EXPECT_EQ(dag->events[0].minCycles, 7u);
```

For this synthetic graph use `machine::MachineModel model; model.target = "probe";` without scheduling; no engine lookup occurs in the custom handler. Add an error-producing handler and assert buildMicroDAG returns that error, not a successful graph. Build with two independently configured registries and assert their outputs remain independent.

- [ ] Run the new tests; expect missing overload/factory behavior to fail.
- [ ] Add a `const OpDAGBuilderRegistry &` to DAGBuilder. Implement a private nested context adapter holding DAGBuilder&, Operation&, and State&. Each context method delegates to the existing helper; emitEvent resolves explicit input producers, inserts through addEvent, and binds explicitly listed outputs to the returned ID. accountStorage always applies the current operation/storageFactor.
- [ ] Replace buildOp with exact-name lookup and invocation. Preserve the zero-event fallback. Leave walkBlock/walkLoop structural cases unchanged.

```cpp
llvm::Error DAGBuilder::buildOp(mlir::Operation &op, State &state) {
  const auto *handler = builders.lookup(op.getName().getStringRef());
  if (!handler)
    return llvm::Error::success();
  Context context(*this, op, state);
  return (*handler)(op, context);
}
```

- [ ] Move every original branch into typed built-in handlers: TileViewOp, TilePartitionOp, TileAllocOp, AllocOp, TileAsyncCopyOp, AsyncCopyOp, TileStoreOp, StoreOp, MmaOp, VectorOp, ReduceOp, WaitOp. Preserve engine selection, validation messages, event fields, and zero-cost behavior. Group functions by family in MicroOpDAGBuilders.cpp. The default factory registers only these built-ins; treat any factory registration failure as an internal invariant failure through LLVM error utilities, never exceptions.
- [ ] Keep movement and logical helpers behind the context for this commit; do not yet change their missing dependency behavior. Preserve any newer mapped-route APIs from the integrated base.
- [ ] Add explicit override tests for a built-in leaf, handler failure propagation, and unregistered arith.constant remaining zero-event by default. Compare ordered event fields/deps/diagnostics/storage between the two-argument call and explicit default registry on copy/wait/MMA and loop/pipeline fixtures. Test aliases through a zero-cost view and multi-hop result/token production with the existing route fixtures.
- [ ] Run OpDAGBuildersTest, L0StaticBoundTest, L1ResourceDAGTest, and any RouteIdentityTest registered in the integrated base; confirm unchanged baseline expectations.
- [ ] Commit: `refactor(perf): dispatch MicroDAG extraction through injected builders`.

### Task 4: Fix movement dependencies independently of extraction

**Files:** MicroDAG.cpp, l1_resource_dag.cpp; op_dag_builders.cpp if extension coverage is needed.

**Interfaces:** Uses existing producerDeps/addEvent and Task 3 context emission. Public APIs remain unchanged.

- [ ] Add a failing verified kernel whose second copy consumes the first copy's result, with no wait between them:

```mlir
module {
  micro.kernel @dependent_copies {
    %ext = tensor.empty() : tensor<8x8xf32>
    %a, %ta = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    %b, %tb = micro.async_copy %a {src_memory = #micro.memory<sram>, dst_memory = #micro.memory<acc>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    micro.wait %tb
    micro.yield
  }
}
```

Use parseKernel and parseMachine(testMachine(2, 2, 4)) from l1_resource_dag.cpp; the machine declares dram→sram and sram→acc connectivity. Assert the second copy depends on event 0, its scheduled start is no earlier than event 0's finish, and the wait depends on the final copy.

```cpp
EXPECT_TRUE(llvm::is_contained(dag->events[1].deps, 0u));
L1Report report = scheduleL1(*dag, model);
EXPECT_LE(report.schedule[0].finish, report.schedule[1].start);
```

- [ ] Run only the new regression and confirm the missing edge causes failure. Keep the existing IndependentCopiesOverlapWithMoreDmaEngines test green.
- [ ] At the first movement event, pass `producerDeps(op.getOperands())` to central insertion; subsequent hops depend on the previous hop. Store operands also participate. Do not introduce unconditional ordering between independent copies.

```cpp
llvm::SmallVector<uint32_t, 4> deps = producerDeps(op.getOperands());
// No route: addEvent(event, state, deps).
// Route hop 0: addEvent(event, state, deps).
// Route hop n > 0: addEvent(event, state, {previousHop}).
```

- [ ] Add a logical-transform dependency regression using `micro.tile_partition` from a row_major tile to a vectorized tile, matching the accepted fixture in l0_static_bound.cpp:257–258. Assert its source producer dependency survives even when preceding work has the same vector resource kind. If reproduced, insert `producerDeps({source})` for that physical-transform event; zero-cost aliases remain aliases. Use accepted tile layout spellings already present in the L0 fixtures.
- [ ] Run L0StaticBoundTest, L1ResourceDAGTest, OpDAGBuildersTest and mapped route tests. Review numerical changes only for the newly corrected dependent work; do not update unrelated expectations to hide failures.
- [ ] Commit: `fix(perf): retain source dependencies for movement events`.

### Task 5: Complete integration, documentation, and verification

**Files:** op_dag_builders.cpp, workflow guide, CMakeLists.txt as needed.

**Interfaces:** Document makeDefaultOpDAGBuilders, typed registration/replacement, context lifetime, and both buildMicroDAG entry points.

- [ ] Add registration-order invariance tests: register two distinct leaf handlers in opposite orders, build the same kernel, and compare ordered events/dependencies. Order follows IR traversal, not registry construction.
- [ ] Compare default and explicitly injected default-registry graphs against both shipped machine profiles using a copy/wait fixture. Compare all MicroEvent fields, all warning lists, and liveTileBytesByMemory; use a local comparison helper rather than adding production equality solely for tests.
- [ ] Document a real typed handler example using arith::ConstantOp and emitEvent, plus explicit built-in replacement. State that structural operations cannot be overridden and custom registries must start from the default factory when built-ins are desired.
- [ ] Configure/build in the isolated implementation worktree using CLAUDE.md's LLVM path and matching FileCheck. Run `cmake --build build --target check-llk`, then `git diff --check`. Do not install a separate LLVM or silently skip registered tests.
- [ ] Inspect changed public headers for generated Micro includes, inspect buildOp for remaining typed opcode branches, and inspect callbacks for borrowed stored captures. Verify no changes to Micro ODS, machine seeds, scheduler, or generic target policy were introduced.
- [ ] Record exact test counts/skips, LLVM version, and any intentional prediction changes from Task 4. No runtime equivalence claim may rely solely on a successful build.
- [ ] Commit: `docs(perf): document extensible DAG operation builders`.

## Spec coverage and handoff

Task 1 covers public tile descriptions; Task 2 covers owned typed registration, deterministic validation, and restricted context signatures; Task 3 covers injection, built-in extraction, default compatibility, aliases, traversal ownership, and route preservation; Task 4 covers separately reviewed producer dependencies; Task 5 covers determinism, two-target equivalence, documentation, and the full registered suite.

All tasks depend on the preceding interface decisions. Native execution is recommended to keep those interfaces consistent, with independent review of the completed branch. Implementation begins after the user reviews this plan and selects an execution method.
