# Operator-Kernel Lowering Framework Design

**Project scope alignment (2026-10-12):** Design a generic MicroIR-based, mapping- and search-based DNN compiler framework. Matmul and fused SwiGLU are example operator kernels used to test it. The [framework goal and design spec](2026-10-12-generic-microir-dnn-compiler-framework-design.md) governs project-level goals. This is an incremental semantic-adapter refactor, not the final generic DNN workload interface. Its M/N/K contract and shared contraction driver serve the current example kernels; broader rank, iteration and dataflow contracts remain future capability work. Original draft implementation/status statements retain their revision scope.

**Status:** Draft for review
**Date:** 2026-10-11
**Scope:** `lib/Conversion/` (LLK→Linalg, LLK→Micro) and the two Transforms passes that consume the same per-op facts
**Parent roadmap:** [Canonical DNN Micro-IR roadmap](https://github.com/skg7on/DSLCompiler/issues/41)

**Related design documents:**

- [Canonical Micro-IR architecture](../../design/m9-canonical-micro-ir-architecture.md)
- [Micro-IR core concepts](../../design/m9-micro-ir-core-concepts.md)
- [LLK → Micro lowering implementation spec](2026-08-11-llk-to-micro-lowering-implementation-spec.md)
- [Micro-IR tile programming model](2026-08-13-micro-ir-tile-programming-model-spec.md)

---

## 1. Purpose

Every LLK operator kernel's lowering currently lives inside a hand-written dispatch
inside the pass that performs it. This document specifies a **registration framework**
so that an operator kernel's lowering is a self-contained unit that registers itself,
and the passes become generic drivers that never name an operator.

The change is structural, not behavioural: the IR produced by every pass must be
byte-for-byte what it is today. The framework exists so that the *next* operator
kernel — the fifth, tenth, fiftieth — is added by writing one file, not by editing
four places inside a pass.

---

## 2. Problem Statement

### 2.1 The two converters are closed sets

`LLKToLinalg` lowers `{llk.fused_swiglu, llk.rope, llk.attention}`. It expresses that
set as three `SmallVector`s, three `module.walk` calls, three dispatch loops, and
three methods declared on the pass:

```cpp
// lib/Conversion/LLKToLinalg/LLKToLinalg.cpp:60-62
void lowerSwiGLU(OpBuilder &builder, mlir::llk::FusedSwiGLUOp op);
void lowerRoPE(OpBuilder &builder, mlir::llk::RoPEOp op);
void lowerAttention(OpBuilder &builder, mlir::llk::AttentionOp op);

// :626-645
SmallVector<mlir::llk::FusedSwiGLUOp> swigluOps;
SmallVector<mlir::llk::RoPEOp> ropeOps;
SmallVector<mlir::llk::AttentionOp> attentionOps;
module.walk([&](mlir::llk::FusedSwiGLUOp op) { swigluOps.push_back(op); });
module.walk([&](mlir::llk::RoPEOp op) { ropeOps.push_back(op); });
module.walk([&](mlir::llk::AttentionOp op) { attentionOps.push_back(op); });
for (auto op : swigluOps) { builder.setInsertionPoint(op); lowerSwiGLU(builder, op); }
// ... two more loops
```

Adding an operator is a **four-site edit inside the pass**: a `SmallVector`, a
`module.walk`, a dispatch loop, and a method declaration plus definition. The pass is
the only place that knows the operator exists.

`LLKToMicro` lowers `{llk.fused_swiglu, llk.matmul}` and hardcodes the set twice, then
threads a `dyn_cast` through its contract inference and kernel assembly:

```cpp
// :1093 and :1149
if (isa<llk::FusedSwiGLUOp, llk::MatmulOp>(op)) roots.push_back(op);

// :425
auto fused = dyn_cast<llk::FusedSwiGLUOp>(root);
StringRef workload = fused ? "fused_swiglu" : "matmul";

// :807-813
auto fused = dyn_cast<llk::FusedSwiGLUOp>(root);
auto matmul = dyn_cast<llk::MatmulOp>(root);
Value lhs = fused ? fused.getX() : matmul.getA();
Value rhs = fused ? fused.getWg() : matmul.getB();

// :448, :533 — arm count by ternary
for (size_t arm = 0; arm < (fused ? 2u : 1u); ++arm)

// :724 — epilogue by branch
if (fused) { /* silu, then mul */ }
```

Every new operator adds another arm to each ternary and another branch to the
epilogue. The shapes compose multiplicatively: two sinks, two op sets, four
independent places to keep in sync.

### 2.2 The op sets are undiscoverable

The two sinks support **different** operator sets — `{swiglu, rope, attention}` versus
`{swiglu, matmul}`. Neither set is declared anywhere. It exists only as `isa<>` and
`walk<>` literals inside pass bodies, so there is no way to ask the compiler "which
operators can be lowered to Micro-IR?" without reading two `.cpp` files.

### 2.3 The same dispatch is repeated in Transforms

`ScheduleSelection` needs a workload name and an M/N shape per operator, and rebuilds
that mapping with its own chain of `isa`/`dyn_cast` — including a second, slightly
different N derivation per operator:

```cpp
// lib/Transforms/ScheduleSelection.cpp:60-102
if (isa<llk::FusedSwiGLUOp>(op))       opName = "fused_swiglu";
else if (isa<llk::RoPEOp>(op))         opName = "rope";
else if (isa<llk::AttentionOp>(op))    opName = "attention";
// ...
N = swiglu.getWg().getType().getDimSize(1);   // or
N = rope.getX().getType().getDimSize(3);      // or
N = attn.getQ().getType().getDimSize(3);
```

`ShapeSpecialization` walks `llk.FusedSwiGLUOp` by name, and carries a **private static
copy** of `classifyM` that duplicates `llk::classifyM` in `ScheduleLoader.h`.

### 2.4 Two defects the audit surfaced

1. **A stale contract.** `llk.matmul`'s ODS description states it is *"Lowered
   identically to `linalg.matmul` by LLKToLinalg"* (`include/LLK/Dialect/LLKOps.td:217`).
   `LLKToLinalg` does not implement that lowering. The op is Micro-only today, and the
   documentation asserts a capability the compiler does not have.
2. **An unused matching key.** `loadScheduleDB` and `selectBestSchedule` accept `N`
   and `K` but never filter on them; the code says so in a comment. `ScheduleSelection`
   passes `K = 0` while `LLKToMicro` passes the real K. The divergence is currently
   invisible because nothing reads either argument.

Both are recorded here because the framework makes the first cheap to fix and the
second relevant: once operators declare a real contract, both consumers can read the
same numbers.

---

## 3. Goals and Non-Goals

### Goals

1. One registration point per operator kernel, in one file, covering both sinks.
2. Passes that never name an operator: they walk, look up, and dispatch.
3. One discoverable answer to "what does this operator declare?" — its workload name,
   contraction shape, element types, and accumulator type — consumed by the Micro
   export, schedule selection, and shape specialization alike.
4. No change to emitted IR. The existing FileCheck and execution suites are the
   acceptance test.
5. No RTTI, no exceptions — the registry keys on strings, not `dynamic_cast`.

### Non-Goals

- **Not** changing `MicroToLinalg`. It already uses `OpConversionPattern` +
  `ConversionTarget` + `RewritePatternSet` — the idiomatic MLIR mechanism, and the
  in-repo precedent this design follows where it fits. It is out of scope.
- **Not** changing the `micro` dialect ODS, the LLK dialect ODS (except the §2.4
  docstring correction), or the `schedule_db.json` schema.
- **Not** adding operators. The framework lands with the four existing kernels
  registered through it.
- **Not** introducing autotuning or search-space changes.
- **Not** a generic plugin/ABI boundary. Registration is compile-time and static.

---

### Framework scope boundary

Registration is an adapter extension seam. The generic project is not a catalog of registered contraction kernels. `KernelContract::M/N/K` and `MicroKernelFragment`'s shared lhs/RHS arms and epilogue describe the current matmul/SwiGLU export family. A future generic workload interface must express non-contraction iteration/dataflow, arbitrary applicable port/rank structure, effects and capability-specific search parameters. Do not force new operator families into this template or add workload-name dispatch to mapping/analysis. This proposal's implementation remains bounded; that generalization needs a separate design and plan.

## 4. Design

### 4.1 The unit of extension is the operator kernel

A new library, `LLKKernel`, owns the registry and one translation unit per operator.
Each operator file provides everything any consumer needs about that operator — its
contract and, optionally, one lowering per sink:

```
include/LLK/Conversion/LLKKernel/LLKKernelLowering.h   ← Contract, fragment, lowering, registry
lib/Conversion/LLKKernel/
    Registry.cpp        ← LLKKernelRegistry + populateLLKKernelLowerings()
    SwiGLU.cpp          ← contract + Linalg hook + Micro hook
    RoPE.cpp
    Attention.cpp
    Matmul.cpp
lib/Conversion/LLKToLinalg/LLKToLinalg.cpp   ← thin pass: walk, look up, dispatch
lib/Conversion/LLKToMicro/LLKToMicro.cpp     ← thin pass: walk, look up, dispatch
                 /MicroKernelDriver.{h,cpp}  ← shared kernel assembly, moved out of the pass
```

The dependency graph is acyclic and points one way:

```
LLKToLinalg ─┐
LLKToMicro   ─┼──► LLKKernel ──► LLKDialect, MicroDialect
ScheduleSel  ─┤
ShapeSpec    ─┘
```

Critically, `LLKKernel` does **not** depend on `LLKToMicro`: the per-operator Micro
hook only *fills a struct*, and the driver that *consumes* that struct lives in
`LLKToMicro`. This is what keeps the cycle out while letting one file serve both sinks.

### 4.2 Core interfaces

```cpp
// include/LLK/Conversion/LLKKernel/LLKKernelLowering.h
namespace mlir::llk {

/// The per-operator facts every consumer needs: the Micro export, the schedule
/// selector, and shape specialization. Filled by LLKKernelLowering::inferContract.
struct KernelContract {
  std::string workload;                     // "fused_swiglu", "matmul", …
  int64_t M = 0, N = 0, K = 0;
  Type inputElemType, accumulatorElemType, outputElemType;
  int64_t mBucket = 0;                      // llk::classifyM(M)
};

/// The per-operator part of a concrete micro.kernel. Everything else -- the
/// spatial nest, the K loop, the pipeline, the staged copies, the MMA and the
/// store -- is supplied by the shared driver in LLKToMicro.
struct MicroKernelFragment {
  /// MMA/accumulator arms. Each arm after the first is a separate tile the
  /// epilogue combines; llk.fused_swiglu has 2, llk.matmul has 1.
  unsigned armCount = 1;
  /// External operand types in signature order, `armCount` entries, normally
  /// [K, N]. The shared lhs ([M, K]) is implied and supplied by the driver.
  SmallVector<Type> rhsOperandTypes;
  /// Attributes added to the micro.kernel beyond the driver's default set.
  SmallVector<NamedAttribute> extraAttrs;
  /// Builds the elementwise epilogue from the accumulated arms, in accumulator
  /// element type. The driver emits the accumulator→output convert afterwards,
  /// uniformly for every operator.
  function_ref<Value(OpBuilder &, Location, ArrayRef<Value> accumulators)>
      buildEpilogue;
};

class LLKKernelLowering {
public:
  virtual ~LLKKernelLowering() = default;

  /// The op this lowering registers for, e.g. "llk.fused_swiglu".
  virtual StringRef getOpName() const = 0;

  /// Validates the operation and reports its contract. Every registered operator
  /// must implement this; it is what the Transforms consumers read.
  virtual LogicalResult inferContract(Operation *op,
                                      KernelContract &contract) const = 0;

  /// LLK → Linalg. Absent for operators with no Linalg lowering.
  virtual bool hasLinalgLowering() const { return false; }
  virtual LogicalResult lowerToLinalg(Operation *op,
                                      OpBuilder &builder) const {
    return failure();
  }

  /// LLK → Micro. Absent for operators with no Micro export.
  virtual bool hasMicroLowering() const { return false; }
  virtual LogicalResult describeMicroKernel(Operation *op,
                                            const KernelContract &contract,
                                            MicroKernelFragment &fragment) const {
    return failure();
  }
};

class LLKKernelRegistry {
public:
  template <typename T, typename... Args> void add(Args &&...args);
  /// nullptr when no operator with this name is registered.
  const LLKKernelLowering *lookup(StringRef opName) const;
  /// Registration order, which is also lowering order.
  ArrayRef<const LLKKernelLowering *> entries() const;
};

/// The single ordering point. One line per operator kernel.
void populateLLKKernelLowerings(LLKKernelRegistry &registry);

} // namespace mlir::llk
```

**Why strings and not RTTI.** The registry is keyed on `getOpName()` returning a
`StringRef`, and `lookup` is called with `op->getName().getStringRef()`. No
`dynamic_cast`, no `typeid`, so the library links cleanly under the project's
`-fno-rtti -fno-exceptions`. `OperationName` is deliberately **not** used as the
stored key: in LLVM 24 it has no default constructor, so a bare `OperationName`
member would delete the registry entry's default constructor. Names are stored as
`std::string` inside a `llvm::StringMap`.

### 4.3 Registration

Registration is an explicit populate anchor, not a static initializer:

```cpp
// lib/Conversion/LLKKernel/Registry.cpp
void populateLLKKernelLowerings(LLKKernelRegistry &registry) {
  registerSwiGLULowering(registry);
  registerRoPELowering(registry);
  registerAttentionLowering(registry);
  registerMatmulLowering(registry);
}
```

Each operator file exposes exactly one free function:

```cpp
// lib/Conversion/LLKKernel/SwiGLU.cpp
void registerSwiGLULowering(LLKKernelRegistry &registry) {
  registry.add<SwiGLULowering>();
}
```

This was chosen over file-scope static self-registration deliberately. These targets
are built with `add_mlir_library`, whose output is an archive whenever the shared-library
option is off — which is how CI builds. An object file in a static archive that nothing
references is dropped by the linker, so a self-registering translation unit can compile,
pass locally against a dylib, and silently vanish in a static link. This repository has
already been bitten by exactly that class of local-green/CI-red defect. The explicit
anchor cannot be dropped, and it matches MLIR's own `populate*Patterns` convention.

The cost — one shared file that names every operator — is one line per operator in one
place, against four edit sites per operator in two places today.

### 4.4 What each consumer becomes

| Consumer | Today | After |
|---|---|---|
| `LLKToLinalg` | 3 vectors, 3 walks, 3 loops, 3 methods | one `module.walk`; `lookup(op->getName())` → `lowerToLinalg` |
| `LLKToMicro` concrete | `isa<...>` ×2, `dyn_cast` ×2, ternaries | `inferContract` → `selectSchedule` → driver + `describeMicroKernel` |
| `LLKToMicro` search space | as above | `inferContract` → `selectSchedule` → generic `buildSearchSpace` |
| `ScheduleSelection` | `isa`/`dyn_cast` chain + per-op N derivation | `inferContract` → `contract.workload/M/N` |
| `ShapeSpecialization` | `walk<FusedSwiGLUOp>` + private `classifyM` | `inferContract` → `contract.M`; the duplicate `classifyM` is deleted |

Two invariants the refactor must preserve:

1. **Collect before lowering.** Both passes currently gather their roots into a
   `SmallVector` before mutating, because lowering an operator invalidates the walk.
   The registry-driven versions keep the collect-then-dispatch shape; only the
   *predicate* changes from `isa<A,B>` to `registry.lookup(...) != nullptr`.
2. **Sink capability gates the walk.** `llk.matmul` is registered but has no Linalg
   lowering, and `llk.rope` is registered but has no Micro export. Each pass skips
   entries whose sink hook is absent (`hasLinalgLowering()` / `hasMicroLowering()`),
   so registering an operator for one sink never makes it a root in the other.

### 4.5 Factoring the Micro driver

`buildKernel` currently interleaves operator-agnostic kernel assembly with the three
`fused`-dependent sites listed in §2.1. Its operator-agnostic 200-odd lines —
`TilePlan`, `resolvePlan`, `parseMemoryPath`, `parseOwnerHierarchy`, `checkTensor`,
the symbol-naming loop, the spatial nest, the K loop, the pipeline, the logical views,
the staged copies, the partitions, the MMA, the store, and the yield plumbing — move
unchanged into `lib/Conversion/LLKToMicro/MicroKernelDriver.{h,cpp}`.

The three `fused`-dependent sites collapse to:

| Today | After |
|---|---|
| `StringRef workload = fused ? "fused_swiglu" : "matmul"` | `contract.workload` |
| `for (arm = 0; arm < (fused ? 2u : 1u); ++arm)` ×2 | `fragment.armCount` / `fragment.rhsOperandTypes` |
| `if (fused) { silu; mul; }` | `fragment.buildEpilogue(builder, loc, accumulators)` |

`buildSearchSpace` is already operator-agnostic — it reads `info.workload`,
`info.M/N/K`, and `info.mBucket` — so it needs nothing beyond the contract. Per-operator
search-space extensions are deliberately **not** introduced; there is no operator that
needs one today.

The driver move is the highest-risk step in the migration and is a pure refactor:
`LLKToMicroSwiGLU` is a byte-for-byte FileCheck over the emitted kernel and is the
proof that the parameterization is faithful.

### 4.6 Shared contract inference

`inferRootInfo` already separates cleanly along the operator boundary. The generic half
becomes a helper that both operators call:

```cpp
/// Validates that `lhs`, `rhs` and `out` form a static rank-2 contraction and
/// fills shape, element types, accumulator type, and M bucket. Operator-specific
/// checks (extra operands, attribute legality) stay in the operator's own hook.
LogicalResult inferContractionContract(Operation *root, Value lhs, Value rhs,
                                       Value out, Attribute accumulatorType,
                                       KernelContract &contract);
```

`SwiGLULowering::inferContract` calls it, then adds its own two checks — that the
up-projection weight is `[K, N]`, and that the activation is `silu`. `MatmulLowering`
calls it and adds nothing. `checkTensor` and the contraction-dimension check become
shared, which is a strict deduplication of the current code.

### 4.7 Build changes

```cmake
# New library holding the registry and one file per operator kernel.
add_mlir_library(LLKKernel
    lib/Conversion/LLKKernel/Registry.cpp
    lib/Conversion/LLKKernel/SwiGLU.cpp
    lib/Conversion/LLKKernel/RoPE.cpp
    lib/Conversion/LLKKernel/Attention.cpp
    lib/Conversion/LLKKernel/Matmul.cpp
    DEPENDS LLKDialect MicroDialect LLKCommon
    LINK_LIBS PUBLIC LLKDialect MicroDialect LLKCommon
)
mlir_target_link_libraries(LLKKernel PUBLIC
    MLIRLinalgDialect MLIRArithDialect MLIRMathDialect MLIRTensorDialect MLIRPass)
```

`LLKToLinalg`, `LLKToMicro`, `LLKScheduleSelection`, and `LLKShapeSpecialization` each
add `LLKKernel` to their `DEPENDS`/`LINK_LIBS`.

**Link-closure risk to verify in CI.** `LLKKernel` depends on `MLIRLinalgDialect`, so
`LLKToMicro` — which does not depend on Linalg today — acquires it transitively.
A local dylib build can hide a missing static archive component here. The full
`check-llk` run on CI is the gate, and the CMake dependency must be written so the
dylib/archive split resolves, per the project's existing `mlir_target_link_libraries`
convention.

---

## 5. Behaviour Changes

Three deliberate, reviewable departures from today's behaviour. Each is small, and
each is reversible independently.

### 5.1 `ShapeSpecialization` generalizes from swiglu-only to all registered operators

Today the pass walks `llk.FusedSwiGLUOp` only and emits a remark per swiglu op.
Registry-driven, it walks every operator whose contract declares an M and emits the
same remark shape for each. Every existing test uses swiglu inputs, so the suite stays
green — but rope, attention, and matmul inputs now produce remarks where they previously
produced none. This is the intended direction: the pass's own description is "classify M
dimension into buckets", which was never swiglu-specific.

### 5.2 `ScheduleSelection` passes the contract's real `K`

`ScheduleSelection` passes `K = 0` and `LLKToMicro` passes the real K. Because
`loadScheduleDB` and `selectBestSchedule` never filter or select on `N` or `K`, both
are behaviour-preserving today. The framework gives one honest answer, so the call site
passes `contract.K` and the loader's "a fuller implementation would compare N and K"
comment is updated to record that the arguments are accepted but unused. If the
arguments are ever given meaning, both consumers will already be passing correct values.

### 5.3 `llk.matmul` gains the Linalg lowering its documentation already claims

§2.4(1) records a documented capability that does not exist. The framework makes the
fix a ~10-line registration — `llk.matmul` → `linalg.matmul` — and a new FileCheck
test. The alternative is to delete the ODS sentence and leave `llk.matmul` Micro-only.
Implementing it is recommended: it is the cheapest available proof that the framework
absorbs a new operator end-to-end, and it makes the documentation true rather than
merely consistent.

**Note:** with §5.3, `llk.matmul` is the first operator registered in both sinks. It
also becomes visible to `ShapeSpecialization` and `ScheduleSelection` under §5.1, which
is consistent — those passes already tolerate any static-shape operator.

---

## 6. Migration Plan

Ordered TDD tasks. Each is a separate commit, each begins with a failing test, per the
project's stated convention. Every step except T8 must leave `check-llk` fully green.

| Task | Work | Test that proves it |
|---|---|---|
| **T1** | `LLKKernelLowering.h` + `Registry.cpp`: registry, `add<T>()`, `lookup`, `entries`, empty `populateLLKKernelLowerings` | New `LLKKernelRegistryTest` GTest — unknown name → `nullptr`, registration order preserved, `hasLinalgLowering`/`hasMicroLowering` default false |
| **T2** | `KernelContract` + `inferContractionContract` helper | GTest over a synthetic op: static shape, contraction mismatch, non-tile dtype, accumulator attr |
| **T3** | Move SwiGLU's Linalg lowering to `LLKKernel/SwiGLU.cpp`; wire the populate anchor; make `LLKToLinalg` registry-driven | `LLKToLinalgConversion` + all `Execution/swiglu_*` stay green |
| **T4** | Move RoPE, then Attention, the same way | `LLKToLinalgConversion`, Triton/rope/attention FileCheck stay green |
| **T5** | Extract `MicroKernelDriver`; add `describeMicroKernel` to SwiGLU and `Matmul.cpp`; make `LLKToMicro` registry-driven | `LLKToMicroSwiGLU`, `LLKToMicroMatmul`, `LLKToMicroPerfRoundTrip`, `LLKToMicroLoweringInvalid`, `Execution/mapped_acceptance` stay green |
| **T6** | Make `LLKToMicroSearchSpacePass` registry-driven | `LLKToMicroSearchSpace`, `LLKToMicroSearchSpaceMBucket`, micro-search compile test stay green |
| **T7** | Make `ShapeSpecialization` and `ScheduleSelection` registry-driven; delete the duplicated static `classifyM` | `shape_specialization.mlir`, `schedule_selection.mlir`, `schedule_fallback.mlir` stay green |
| **T8** | `llk.matmul` → `linalg.matmul` lowering; correct the ODS description | New FileCheck: `matmul_to_linalg.mlir` |
| **T9** | CMake cleanup, `-fno-rtti`/`-fno-exceptions` audit, deprecated-API audit, full `check-llk` | Whole suite; CI static link |

T5 is the riskiest and should be reviewed on its own. T7 is where the duplicate
`classifyM` disappears.

---

## 7. Testing

**New tests.**

- `LLKKernelRegistryTest` — the registry's own contract: lookup hit and miss, ordering,
  capability flags. This is the test that fails before the framework exists (T1).
- `KernelContractTest` — contract inference over valid and invalid shapes (T2).
- `matmul_to_linalg.mlir` — FileCheck over `llk.matmul → linalg.matmul` (T8).

**Regression net — these already exercise the code being moved and must stay
byte-for-byte green.**

- `LLKToLinalgConversion`, `LLKToLinalgTritonFast`
- `LLKToMicroSwiGLU`, `LLKToMicroMatmul`, `LLKToMicroSearchSpace`,
  `LLKToMicroSearchSpaceMBucket`, `LLKToMicroPerfRoundTrip`, `LLKToMicroLoweringInvalid`
- `test/Transforms/shape_specialization.mlir`, `schedule_selection.mlir`,
  `schedule_fallback.mlir`
- `test/Execution/`: `swiglu_scalar`, `swiglu_vector_avx2`, `swiglu_fused_memory`,
  `mapped_acceptance` — these execute the lowered IR, so they catch a structurally
  valid but semantically wrong kernel that FileCheck would pass.

**Process gates.** `ninja` clean under `-Werror=deprecated-declarations`; the three
deprecated-API audit greps from `.claude/rules/preferred-mlir-api.md` clean; CI green
including the static link, per §4.7.

---

## 8. Risks

| Risk | Mitigation |
|---|---|
| Static-archive registration silently dropping operators in CI | Registration is an explicit populate anchor, not a static initializer (§4.3) |
| `LLKKernel` pulling Linalg into `LLKToMicro`'s link closure breaks a CI static link | Dependency written through `mlir_target_link_libraries` per project convention; §4.7 calls it out explicitly; CI `check-llk` is the gate |
| The `MicroKernelDriver` extraction changes emitted IR subtly | It is a move plus three parameterizations (§4.5); `LLKToMicroSwiGLU` FileCheck is byte-for-byte, and `mapped_acceptance` executes the result |
| `ShapeSpecialization` generalization surprises a downstream consumer | All existing tests are swiglu-only; the change is additive remarks (§5.1) |
| Registry ordering becomes a hidden dependency | Order is registration order, and `populateLLKKernelLowerings` is the single, reviewable ordering point (§4.3) |

---

## 9. Alternatives Considered

| Alternative | Why not |
|---|---|
| **File-scope static self-registration** (each operator file constructs its own registrar) | Zero central edits, but unreferenced objects in a static archive are dropped by the linker — a known local-green/CI-red failure mode in this repository. Needs `LLVM_ATTRIBUTE_USED` anchors or whole-archive per library. The explicit anchor is one line per operator and cannot break. |
| **MLIR `DialectConversion` for both sinks** (`OpConversionPattern` + `ConversionTarget`) | Correct and idiomatic for LLK→Linalg, and already used by `MicroToLinalg`. But `LLKToMicro` does not rewrite its root op — it *appends* a new `micro.kernel` symbol and leaves the original untouched. That is not a conversion, so DialectConversion does not apply to the Micro sink, and adopting it would leave the two sinks on different mechanisms. |
| **A separate registry per consumer** | Maximum decoupling, but each operator's shape-inference logic becomes registrable in two or three places, and there is no single answer to "what does this operator declare?" — which is precisely the question §2.3 shows the consumers failing to share. |
| **An ODS interface on the LLK ops** (`LLK_KernelOpInterface`) | Would move the contract into tablegen, giving stronger typing than a virtual `inferContract`. But the per-operator *lowering code* still needs a home, so a registry is still required; the interface would replace only part of the design. Worth revisiting as a later layer once the contract's shape has settled in practice. |
| **Do nothing; accept the dispatch** | The operator set is expected to grow well beyond four. The cost is paid once per operator, forever, at four sites per sink — and the divergence between the two sinks' op sets (§2.2) is the first symptom of the sets drifting apart with nothing to detect it. |

---

## 10. Open Decisions

Both are reversible, and both are recorded here because they were recommended rather
than requested.

- **D1 — Does `llk.matmul` get its Linalg lowering?** Recommended: yes (§5.3), because
  the documentation already claims it and the framework makes it a ~10-line proof that
  a new operator registers end-to-end. The alternative is to delete the ODS sentence
  and leave `llk.matmul` Micro-only.
- **D2 — Does `ScheduleSelection` pass the real `K`?** Recommended: yes (§5.2), since
  the value is presently ignored and passing the honest one removes a latent
  divergence between the two consumers. The alternative is to freeze `K = 0` at the
  call site and record why.

---

## 11. Acceptance Criteria

1. Adding an operator kernel requires exactly one new file plus one line in
   `populateLLKKernelLowerings` — no edit to either pass, and no edit to any Transforms
   pass.
2. No pass in `lib/Conversion/` or `lib/Transforms/` names an LLK operator type.
3. The two sinks' operator sets are readable from `populateLLKKernelLowerings` alone.
4. Every existing FileCheck and execution test passes unchanged.
5. `LLKKernelRegistryTest`, `KernelContractTest`, and (if D1 is accepted)
   `matmul_to_linalg.mlir` pass.
6. CI is green, including the static link.
7. No deprecated MLIR/LLVM API is introduced; the libraries still build under
   `-fno-rtti -fno-exceptions`.
