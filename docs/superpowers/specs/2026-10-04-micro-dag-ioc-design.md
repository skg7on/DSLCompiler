# MicroDAG operation-builder inversion of control

Date: 2026-10-04
Status: Proposed written specification; registry/context direction approved, written-spec review pending.
Baseline: main at 31fd9eb. Concurrent mapping work must be integrated without losing its route-identity fixes.

## Intent and scope

Refactor DAGBuilder::buildOp so callers can supply operation event builders without editing the central dispatch chain. Preserve the default performance-model behavior and existing buildMicroDAG callers. This is an architectural extension seam for operation interpretation, not a new IR, scheduler, or dynamic plugin loader.

The generic DAG builder retains graph invariants. Handlers interpret operation operands and attributes, choose machine resources through shared services, and request events or aliases. Target-specific handlers can be provided by target packages without adding target policy to canonical Micro ODS.

## Selected approach

Use an explicitly injected, immutable operation-builder registry, keyed by the registered operation's full name. Typed registration adapters perform MLIR casts and pass the typed operation to an owning callback. Neither registration nor dispatch requires C++ RTTI.

A TypeSwitch plus helper extraction would shorten buildOp but would still require central edits for every extension. MLIR operation interfaces or external models would provide an alternative extension mechanism, but they add dialect/context registration machinery unnecessary for this performance-only seam.

## Public entry points and ownership

Add include/LLK/Perf/OpDAGBuilder.h for OpDAGBuilderRegistry and OpDAGBuildContext. Keep MicroDAG.h focused on events and graph construction, forward-declaring the registry where possible.

Retain the existing entry point:

```cpp
llvm::Expected<MicroDAG>
buildMicroDAG(Operation *kernel, const machine::MachineModel &machine);
```

Add an overload:

```cpp
llvm::Expected<MicroDAG>
buildMicroDAG(Operation *kernel, const machine::MachineModel &machine,
              const OpDAGBuilderRegistry &builders);
```

Provide makeDefaultOpDAGBuilders(), returning an owned registry containing the built-in handlers. The convenience entry point creates or obtains an immutable default registry and delegates to the injected overload. The injected overload uses precisely the registry supplied by its caller; it does not silently merge global registrations.

Registration accepts owning callbacks, such as std::function<llvm::Error(Operation &, OpDAGBuildContext &)>. Do not store llvm::function_ref: its borrowed lifetime is inappropriate for a registry. A registerBuilder<OpT> adapter keys registration with OpT::getOperationName() and wraps a typed callback. Callbacks may capture immutable configuration. Capture lifetimes are the caller's responsibility; handlers must not retain the per-operation context.

Registration returns llvm::Error on a duplicate name. replaceBuilder<OpT> is an explicit replacement operation and fails if the name is absent. Neither API silently applies last-registration-wins precedence. Registry mutation is permitted before construction, never during a build; dispatch accepts a const registry. There is no mutable global registry or static-constructor registration.

## Build context and invariants

OpDAGBuildContext is a non-copyable, operation-scoped facade implemented by the core. It exposes no mutable MicroDAG, producer map, counted-storage set, or traversal State. Core-owned services include:

- Read-only MachineModel access and shared tile description, memory lookup, engine selection, and cost queries.
- emitEvent(event, explicit input values, explicit produced values, extra event dependencies), returning the inserted event ID.
- aliasValue(result, source), preserving the source producer through a logical operation.
- accountStorage(memory, bytes), automatically using the current operation and traversal storage factor for once-per-allocation-site accounting.
- Diagnostics for unsupported layouts/owners and unsizable values, retaining current deduplication and ordering.
- Shared movement construction, including selected route resolution and hop chaining.

Every emitted event goes through the existing central insertion policy: assign sequential IDs; set cost category and inherited owner; combine explicit producer dependencies, pending region dependencies, and program-order dependencies; sort/deduplicate dependency IDs; update traversal state; record diagnostics.

Input and output lists are explicit because logical aliases, async tokens, stores, and multi-hop movements do not all have the same production semantics. Multi-hop movement binds its results and token to the final hop. The context does not mutate MLIR operations.

The context initially adapts existing helper behavior. It must not turn registration into unrestricted access to private DAGBuilder state. Shared TileInfo/value-description types needed by external handlers move to a public perf header without exporting generated Micro operation headers transitively to all clients.

## Operation handlers

Move the operation branches into typed handlers organized by family:

- Logical tile view/partition handlers preserve aliases and current physical-transform modeling.
- Allocation handlers record storage without manufacturing execution events.
- Copy/store handlers normalize endpoints and delegate route/copy construction to common services.
- MMA, vector, and reduce handlers interpret work volume, choose resources, request costs, and emit events.
- Wait handlers emit synchronization events with explicit token dependencies.

A suitable repository organization is lib/Perf/OpDAGBuilder.cpp for registration/context support and lib/Perf/MicroOpDAGBuilders.cpp for built-ins. Keep implementations grouped by family; do not require one class or file per operation.

DAGBuilder::buildOp becomes lookup plus invocation. It does not contain the built-in opcode chain. An unregistered operation follows the existing zero-event fallback in this refactor, preserving MVP behavior. Tests document that fallback. Changing unsupported-operation policy is separate work, so registration failure must not masquerade as successful zero-cost extraction.

## Structural traversal

walkBlock and walkLoop remain core-owned. ForOp, SpatialForOp, PipelineOp, and YieldOp are handled before leaf dispatch. They control scope, iteration expansion, storage factors, pending dependencies, and created-event collection. Structural-operation registration/replacement is rejected with a diagnostic; it must not imply an override that traversal would never invoke.

This first interface extends leaf extraction only. A future region-builder protocol requires a separate design for ownership of recursion and scope transitions.

## Compatibility and the dependency finding

Preserve existing resource names, event order, cycle equations, route interpretation, storage estimates, warnings, zero-cost aliases, default fallback, and public callers during the mechanical refactor. In particular, do not replace route-identity improvements from concurrent mapping changes with the older endpoint-space implementation.

Current buildCopyOp omits explicit source-producer dependencies, while addEvent drops program-order edges between same-resource-kind events. Logical transform emission similarly requires explicit dependency review. Resolve these as a separate regression-driven correctness change after extraction: dependent copies must retain data edges even when they use the same resource kind, whereas independent copies must remain eligible to overlap. The context API supports explicit dependencies from the outset; the refactor commit must not obscure an intentional behavior change among moved code.

## Validation and acceptance

1. Existing default callers compile without changes; L0/L1 and route-identity suites retain their prior outputs after mechanical extraction.
2. A test supplies a builder for a distinct operation and produces a graph without modifying DAGBuilder::buildOp.
3. An explicit replacement changes only that operation's extraction; unrelated built-ins remain available through the caller-created default registry.
4. Duplicate registration, replacement of an absent operation, and structural registration return deterministic errors.
5. Handler errors propagate as llvm::Error rather than falling through to the zero-event fallback.
6. Tests cover logical producer aliasing, allocations counted once across unrolling, async tokens, multi-hop chains, and loop/pipeline ordering.
7. The separate dependency fix tests dependent same-kind copies and independent overlapping copies; add logical-transform coverage if its dependency omission is confirmed.
8. Registry dispatch never depends on registration iteration order. Two builds with the same immutable registry/configuration produce equal ordered events and diagnostics.
9. The injected API works with the project's -fno-rtti/-fno-exceptions configuration and supported LLVM versions.
10. Register tests in CMakeLists.txt and run check-llk before claiming implementation complete.

## Non-goals

No dynamic shared-library loading, target backend lowering, solver changes, cost-model recalibration, new Micro operations, handler-managed region traversal, mutable global registration, or scheduler changes.

## Delivery sequence

After written-spec approval, write a task-level implementation plan covering the public API, context facade, built-in handler extraction, injection tests, behavior-equivalence checks, and the separate dependency regression/fix. Implementation must happen in an isolated worktree under .claude/worktrees/ and preserve changes already merged from concurrent Micro-IR work.
