# micro-perf / MachineModel v2 Unification

**Goal:** One machine model. `LLK/Perf`'s simulator, cost model, legality, tuning, and report move onto `LLKMachine`'s v2 topology; the v1 model, its loader, and its two profiles are retired. This closes four open items on #67:

- micro-perf observes every route hop;
- normalized compute, transfer-hop, transform, synchronization, and capacity events are shared with #46;
- mapping search and micro-perf share MachineModel and cost-event definitions;
- an optional `LatencyProvider` with static-cost fallback exists.

## What the split is today

| Layer | Model | Consumers |
|---|---|---|
| `LLK/Machine` (v2) | executor / memory / compute / transfer-engine graph | `LLK/Mapping`, target packages |
| `LLK/Perf` (v1) | flat owners / engines / memory levels / dma / sync | `micro-perf`, tuning, legality, candidate binding |

They are independent: neither `Legality` nor `MicroCostModel` can see the topology the mapping layers use, which is exactly why the four items above are open.

## The finding that shapes the order

**v2 is missing three facts the simulator needs**, so the migration cannot lead:

- `clock_hz` — the report prints `predicted_ns`, and every cycle estimate is meaningless without it;
- `sync` barrier and wait costs — `micro.wait` is charged from them;
- a worker-thread count — `micro.sparallel` dispatch is bounded by it.

Retiring v1 without them would silently drop information rather than migrate it.

## Slices

1. **Extend v2 with the perf facts** (this PR): `clockHz`, `workerThreads`, and a `SyncModel` on `MachineModel`, loaded from YAML, covered by the canonical string and content hash, with both shipped v2 profiles declaring them. Backward compatible: each has a default, so existing v2 files still load.
2. **Migrate `LLKPerf` onto v2**: `MachineModelLoader` deleted in favour of `LLK/Machine`'s; `MicroDAG`, `MicroCostModel`, `Legality`, `TuningSession`, and `MicroPerfReport` rewritten against v2 queries (`isVisible`, `computesFor`, `transferEnginesFor`, `ownerMatches`). `LLKPerf` gains a dependency on `LLKMachine` (and `LLKMapping` for the shared `Cost`).
3. **Route-hop accounting and shared events**: perf reads a mapped kernel's `micro.routes` metadata and charges **each hop**, and the compute/transfer-hop/transform/synchronization/capacity event vocabulary moves to one shared definition both layers use.
4. **`LatencyProvider`**: an optional interface with static-cost fallback, plus the `MappingTarget` accessor that exposes it.
5. **Retire v1**: delete the v1 model, loader, and `machines/x86-avx2-cpu.yaml` / `machines/generic-ai-accel-v1.yaml`; repoint every test; verify the full suite.

Slices 2–5 are each their own PR. Slice 1 is a prerequisite for all of them, and for the routing item's layout/type validation, which needs the same kind of additive field.

## Slice 1 detail

**Files:** `include/LLK/Machine/MachineModel.h`, `lib/Machine/MachineModel.cpp`, `lib/Machine/MachineModelLoader.cpp`, `machines/x86-avx2-v2.yaml`, `machines/generic-ai-accel-v2.yaml`, `test/Machine/machine_model.cpp`, `test/Machine/machine_model_loader.cpp`.

**Grammar:** a top-level `clock_hz: <int>`, `worker_threads: <int>`, and

```yaml
sync:
  barrier_cycles: 64
  wait_cycles: 4
```

each optional, defaulting to `0`, `1`, and zero costs respectively.

**Validation:** a declared `clock_hz` of 0 is rejected (it would make every nanosecond estimate infinite); `worker_threads` must be at least 1.

**Determinism:** all three participate in the canonical string and therefore the content hash, so two models differing only in clock do not share a latency-cache key.

## Verification (slice 1)

- `ninja -C build` clean; `MachineModelTest` and `MachineModelV2LoaderTest` extended; full suite recorded.
- Both shipped v2 profiles declare the new facts and still verify.

## Slice 2a: the cost facts v2 was still missing

Migrating perf onto v2 turned out not to be a rename: the two models disagree
about **where cost lives**. v1 charges memory *access* from the memory level
(`bandwidth_bytes_per_cycle`, `latency_cycles`); v2 puts transfer cost on
*links*. A real hierarchy has both facts, so v2 gains the access side rather
than losing it:

- `MemoryNode::bandwidthBytesPerCycle`, `MemoryNode::latencyCycles` — access
  cost, distinct from a link's transfer cost; zero means unmodelled;
- `TransferEngineNode::setupCycles` — the fixed cost of starting a transfer;
- `ComputeNode::accumulatorDTypes` — what a capability accumulates into, when
  that differs from its inputs;
- `MachineModel::findMemoryOfKind` — profiles describe memory *spaces*
  (`sram`), nodes are instances (`sram.0`), and the simulator asks by space;
- `MachineModel::ownerCount` — "8 workers" becomes "the executors that match
  `worker`, counting their concurrency", which is what the count means once
  executors are concrete nodes.

Both shipped profiles carry the v1 numbers verbatim (DRAM 32 B/cycle at 220
cycles, SRAM 64 at 4, ACC 256 at 1; the accelerator's 1024/300, 4096/20,
8192/1; DMA setup 0 and 16; FMA/MXU accumulators), so the migration that
follows preserves the existing L0/L1 cycle estimates instead of re-baselining
them.

Every field is appended to its struct, so existing aggregate initialisers and
v2 profiles keep working unchanged.

## Slice 2b: LLKPerf runs on v2, and v1 is retired (2026-10-03)

`LLKPerf`'s five consumers -- `MicroDAG`, `MicroCostModel`, `Legality`,
`MicroPerfReport`, `TuningSession` -- now read `LLKMachine`'s v2 model. The v1
model, its loader, its test, and both v1 profiles are deleted; `LLKPerf` links
`LLKMachine`.

### Semantic mappings

- **resource kinds -> nodes.** `machine.dma.engines` becomes
  `transferEngineCount()`; an engine's multiplicity is a node's `concurrency`;
  `findMemory("sram")` becomes `findMemoryOfKind("sram")`.
- **a kind hierarchy -> executor containment.** v1 nested owner *kinds*
  (`vector_engine` under `lane` under `worker`); v2 nests concrete executors, so
  `nestedUnder` now looks for an executor of the inner kind beneath one of the
  outer kind.
- **fixed triples -> variable rank.** `tile_shapes` were `[m, n, k]`; v2
  `shapes` are arbitrary-rank, so fragment arithmetic filters by rank.
- **a value clock -> an optional one.** `predictedNs` is zero without a clock.

### The profile is a translation, not a redesign

`machines/x86-avx2-v2.yaml` was re-authored to carry v1's numbers, because the
performance suite's baselines are those numbers. Three translations were needed:

1. v1's owner kinds with multiplicities became one executor per kind carrying
   the count as `concurrency`, chained by v1's parent relation
   (`worker.0 -> lane.0 -> veng.0`). This is also what keeps
   `owner_mapping = "worker/vector_engine"` legal.
2. v1's engine classes (`matrix_engines: count 8`) became **one** capability
   node with `concurrency: 8`, not eight nodes of one. A schedule that names an
   engine must share its slots; eight nodes would give it one slot each, and the
   spatial-overlap tests caught exactly that.
3. Each link carries what v1's copy formula produced for that hop -- the slower
   endpoint's latency, the destination's bandwidth -- since v1 charged copies
   from their endpoints and v2 charges them from the link.

### Verification

- `ninja -C build` clean; `ctest` **102 registered, 100 passed, 2 skipped, 0 failed**.
- The only expectations that changed are *identities*, not numbers: the machine
  name in reports (`x86-avx2-cpu` -> `x86-avx2`, the v2 profile's `target`), and
  the engine named in a layout warning (`avx2-vector` -> `vpu`). Every cycle
  estimate, bottleneck, capacity check, and overlap number is unchanged.
