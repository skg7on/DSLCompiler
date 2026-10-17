# ARM SVE and SME Mapping Targets — Design

**Date:** 2026-10-17
**Status:** Revised after the review of [PR #174](https://github.com/skg7on/DSLCompiler/pull/174); design intent, not delivered backend support.
**Scope:** Two static-conformance targets and the shared contraction-analysis repair needed to map them honestly.
**Implementation plan:** [ARM mapping targets](../plans/2026-10-17-arm-sve-sme-mapping-targets.md).

## 1. Intent and boundaries

Add `arm-sve` and `arm-sme` as mapping/analysis targets. Both load profiles and LLKMap policy, extract typed MicroIR, match, place, connect, cover, materialize, verify and report complete static plans. SVE assigns contraction to a vector engine; SME assigns the same contraction to a matrix engine with ZA-modeled storage. Target emitters verify opaque bundles and explicitly refuse lowering.

This extends the [generic framework](2026-10-12-generic-microir-dnn-compiler-framework-design.md) through its actual seams. It requires **shared, target-neutral analysis changes**, profiles, policy, packages and acceptance tests. It is not a data-only addition: the current performance interpreter assumes every `micro.mma` uses a matrix engine. Search must continue to reject plans with incomplete analysis before retention.

Canonical tile and contraction semantics remain unchanged. `lib/Mapping` and `lib/Machine` gain no target-name dispatch; `LLKMapping` gains no dependency on performance/runtime/JIT. The required analysis changes live in `LLKPerf` and its existing MLIR integration. No machine-schema or Micro dialect change is required for this bounded, fixed-length modeled increment.

Non-goals are executable SVE/SME code, real ZA allocation/instruction semantics, native or emulated ARM execution, length-agnostic programs, arbitrary CPU-exported schedules, calibrated performance, and cleanup of legacy SVE feature-detection stubs. These are separate capabilities. Static completion is not executable completion.

## 2. Audited current behavior

At PR head `359109d`, only `x86-avx2` and `generic-ai-accel` are registered mapping targets. Legacy `TargetSVE`, `CpuIsa::SVE`, the unread `target-isa` option and `SVE_F32_MIN` do not establish an ARM mapping backend. The Linux-only nested function in `TargetSVE.cpp` deserves a separate repair.

The review built that head and extracted the original proposed configuration. Both profiles mapped the f32 vector fixture. BF16/i8 vector variants failed their width domains. A minimal SVE contraction failed selected analysis even with RF/SRAM accumulation because `MicroOpDAGBuilders::buildMma` and `MicroDAG::Context::pickMatrixEngine` require a matrix engine. Compiler-exported contractions additionally failed because SVE had no `acc` memory. SME rejected the current CPU schedules on capacity/materialization; observed ZA peaks were 6144 bytes for matmul and 8192 for SwiGLU against a 4096-byte capacity.

Those observations determine the implementation order: fix shared analysis, define storage and finite policy domains, then prove the public mapping path. They are historical probe results, not claims that the revised target packages exist.

## 3. Chosen scope and alternatives

| Choice | Decision and reason |
|---|---|
| Target pair | Keep `arm-sve` / `arm-sme`; a real selected compute/resource difference tests the extension boundary. |
| Vector lengths | Model non-streaming VL=512 and streaming SVL=512 independently. Equality is an experimental control. |
| Shared interpretation | Repair contraction interpretation to honor selected compute identity; do not fake a matrix engine on SVE. |
| Logical accumulation | Keep `#micro.memory<acc>` on both targets; it does not imply a dedicated hardware matrix array. |
| Accepted input | Typed, directly authored MicroIR fixtures with explicit movement and bounded storage; no automatic retargeting of CPU schedules. |
| ZA representation | `acc`-kind memory plus a compute node, without inventing physical ZA semantics or a redundant ZA layout. |
| Backend capability | File-backed declared emitters only; no ISA requirements or lowering pipeline. |

Dropping contraction from SVE would avoid the analysis repair but lose the intended vector-versus-matrix conformance test. Calling SVE's vector compute node a matrix engine would hide the extension defect. Requiring scalable IR and physical ZA semantics first would enlarge this static increment unnecessarily. The chosen scope preserves the useful target comparison while addressing the demonstrated shared assumption.

## 4. Architectural facts and modeling assumptions

SVE non-streaming VL is a multiple of 128 bits in the architectural range 128–2048. SME streaming SVL is a power of two in that range, including 1024 and 2048. VL and SVL can differ. Neither is inferred from the other. These profiles deliberately choose 512 for both; different lengths require separately validated profiles and policy domains, not just replacing one number.

At SVL=512, ZA holds `(SVL/8) × (SVL/8) = 4096` bytes. ZA.S has four logical 16×16 f32 tiles with interleaved slices. These are not physical quadrants, and their count does not specify hardware memory banks. Arranging them as a 32×32 accumulator is a software choice. `banks`, bandwidth, latency, concurrency and throughput in these profiles are explicit modeling seeds. A modeled 16×16×16 matrix fragment is a macro-work unit, not one FMOPA instruction.

Sources: [Arm SME supplement, DDI0616 A.a, §B2.1–B2.3](https://documentation-service.arm.com/static/62015c6c965f7d118e3f5f4c), [Arm SME introduction](https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/arm-scalable-matrix-extension-introduction).

Apple Silicon must not be described as universally NEON-only: [Apple documents SME support](https://github.com/apple-oss-distributions/xnu/blob/main/doc/arm/sme.md) on supported systems, including SVE in streaming mode. This work has not established the local machine's model/OS/toolchain capabilities. Static-only scope is a project decision and does not depend on assuming no suitable hardware exists.

## 5. Shared contraction interpretation contract

### 5.1 Selection and compatibility

Extend `OpDAGBuildContext` with:

```cpp
virtual const machine::ComputeNode *
pickContractionEngine(llvm::StringRef requested, std::string &reason) const;
```

Its default delegates to the existing `pickMatrixEngine`, preserving source compatibility for injected contexts. The core `DAGBuilder::Context` overrides it. On a mapped contraction it requires exactly one selected compute attachment of kind `vector_engine` or `matrix_engine`, resolves the concrete node, validates its kind and executor attachment, and rejects missing, conflicting or unknown identity. An explicit `micro.mma` engine must agree with that selection. It must never silently switch to the machine's first matrix engine. Unmapped/reference analysis retains the existing matrix default; vector-only execution is evidenced through selected mappings, not inferred.

Add `supportsContractionTypes(const machine::ComputeNode &, llvm::StringRef inputDtype, llvm::StringRef accumulatorDtype)` in `LLKPerf`; it accepts only vector/matrix compute kinds whose `elementTypes` and `accumulatorDTypes` contain the pair. Reuse it in contraction event interpretation and `mma_compatible` legality. Mapped unsupported pairs are errors, not warnings accepted as complete. Preserve documented legacy/reference behavior outside this selected contract.

### 5.2 Events, costs and legality

`buildMma` keeps `EventKind::Mma`, MAC/work and data dependencies. It sets `resourceName` to the selected concrete engine, `ownerPool` to the selected executor, and `ResourceKind` from that engine's actual kind. A vector contraction and vector epilogue using `svp` share one resource pool. Matrix events use `za`. Normalized plan/materialized events must agree on compute identity, owner, cycles, traffic and storage.

Add a second non-pure context method, with a default that returns the existing `mmaCycles` value for injected-context compatibility:

```cpp
virtual llvm::Expected<uint64_t>
contractionCycles(const machine::ComputeNode &engine,
                  llvm::ArrayRef<int64_t> shape, uint64_t flops,
                  llvm::StringRef accumulatorDtype) const;
```

The core override validates the cost facts and returns errors; `buildMma` propagates them before emitting an event. It checks positive extents and multiplication of MACs/FLOPs before either cost callback. Existing `mmaCycles` remains available; do not change a `uint64_t` callback to encode errors as sentinel cycle counts.

For a positive `throughputPerCycle`, retain `ceil(2*M*N*K / throughputPerCycle)`. For a vector engine without throughput, use `M*K*ceil(N/lanes(accumulator_dtype))*max(1,issueCycles)` logical vector MAC issues. Missing/nonpositive lanes are incomplete analysis, never a one-cycle fallback. Use checked arithmetic. Existing matrix fragment costing remains unchanged. This is an uncalibrated model, not a claim about FMLA/BFDOT instruction counts.

`mma_compatible` examines compatible engines without assuming a matrix engine exists. A vector alternative needs a valid accumulator lane count and, for unmasked choices, N divisible by that width; M and K remain positive semantic extents. Matrix alternatives retain fragment constraints. Accept when at least one compatible alternative meets its constraints, then let mapping select the concrete implementation. Final selected checks remain authoritative. No target or emitter string is interpreted as policy.

Tests cover vector-only, matrix-only and mixed machines; same-kind multiple nodes; missing/conflicting selected metadata; incompatible dtypes; absent lanes; shared vector resource contention; and selected-versus-materialized analysis parity. Existing AVX2/generic-accelerator tests remain regression controls.

## 6. Machine profiles and storage

Both profiles use schema `llk.machine.v2`. Common facts follow the existing v2 profile conventions:

| Fact | Value |
|---|---|
| Clock / workers / sync | 3 GHz, 8 workers, barrier 100 cycles, wait 0 |
| Executor chain | `cluster.0` (`cluster`, refines `group`) → `worker.0` (`worker`, concurrency 8) → `sve_eng.0` (`sve_engine`, refines `vector`, concurrency 8) |
| DRAM / L2 / SRAM | `dram.0`: 64 GiB; `l2.0`: 1 MiB; `sram.0`: 64 KiB; alignment 64 bytes |
| Visibility | DRAM/L2 at `cluster.0`; SRAM and accumulator at `worker.0` |
| Vector compute | `svp`, kind `vector_engine`, attached to `worker.0`; element types f32/bf16/i8; accumulator dtype f32; lanes 16/32/64; shapes `[16]`, `[32]`, `[64]`; throughput 32 FLOPs/cycle; issue 1, latency 4, concurrency 8 |
| Transfer resource | `dma.0`, kind `dma`, refines `transfer`, attached to worker; count 1, max outstanding 8, setup 0 |

`arm-sve` has no matrix engine. Its single register-storage node is `acc.0`, kind `acc`, 2048 bytes (32 × 64-byte Z-register storage), visible from worker, alignment 64, supported layouts row_major/blocked. It is the modeled accumulator budget; there is **no additional independent RF node** that would double-count those registers. SRAM/acc transfers model register movement abstractly. Operand register allocation, predicates and real instruction realizability are not established by this budget.

`arm-sme` has `za_eng.0` (`za_engine`, refines `matrix`, parent worker) and compute `za` of kind `matrix_engine`, attached to worker. It supports f32/bf16 input, f32 accumulation, row_major/blocked layouts, modeled shape `[16,16,16]`, throughput 512 FLOPs/cycle, issue 1, latency 5, concurrency 8. Its accumulator memory is `za.0`, kind `acc`, **4096 bytes**, alignment 64, supported layouts row_major/blocked. `banks: 4` is a declared seed, not a derivation from ZA tile count. Vector compute supports row_major/vectorized/blocked layouts on both profiles. DRAM supports row_major, L2 row_major/vectorized, and SRAM row_major/vectorized/blocked. Ordinary vector-register allocation is not separately modeled. This does not establish legal direct vector access to physical ZA.

Reuse the original common DRAM/L2/SRAM and link calibration seeds from the AVX2 profile: memory bandwidth/latency 32/220, 64/12, 64/4; banks 1 for SRAM. Accumulator bandwidth/latency is 256/1, banks 32 on SVE and modeled 4 on SME. Links in each profile are `dram_to_l2.0`, `l2_to_sram.0`, `dram_to_sram.0`, `sram_to_acc.0`, `acc_to_sram.0` (SVE), or `sram_to_za.0`, `za_to_sram.0` (SME), and `sram_to_dram.0`; all use transaction_bytes 64 and `dma.0`. Bandwidth/latency respectively: 64/220, 64/12, 64/220, 256/4, 64/4, 32/220. The accumulator endpoints are `acc.0` or `za.0`. Do not retain orphan RF links.

Actual live-storage analysis, including initialized input and result versions, determines feasibility before retention. Logical `acc` is not remapped to SRAM by changing only report metadata. Profiles keep architectural capacities; unsupported schedules must be adapted or rejected.

## 7. Layouts, rules and supported domain

Each target declares exactly one layout implementing `row_major` and one implementing `blocked`. Use prefixes `sve` / `sme`. Row-major is the identity map with rank 2 and no unused size parameter. The complete blocked declaration is:

```
layout sve.blocked_2d(int VW) {
  param VW in {16, 32, 64};
  require rank == 2;
  require VW == machine.compute("vector_engine").lanes(element_type);
  implements blocked;
  map (m, n) -> (m, floordiv(n, VW), mod(n, VW));
}
```

SME uses the same declaration with `sme.`. The three assignments support the advertised dtypes without exhausting the solver. VW describes physical blocking; it is not an assertion that every logical tile dimension equals a lane count. Small/remainder logical tiles can have padding under this static map; no executable masking behavior is claimed. There is no `za_blocked_2d` layout: this increment does not model physical ZA addressing.

Both packages expose ten keys: `<prefix>_vector_add`, `_vector_convert`, `_vector_silu`, `_vector_mul`, `_reduce`, `_copy`, `_tile_copy`, `_tile_store`, `_fused_convert_silu_mul`, plus `sve_mma` or `sme_mma_za`.

| Rules | Supported static signature and requirements |
|---|---|
| vector add/mul | Separate typed rules for f32, bf16 and i8; both operands and result same dtype; vector compute; blocked operand layout |
| vector convert | bf16→f32 and f32→bf16, with typed input/output predicates; blocked variant and row-major initializer variant |
| vector silu / reduce sum | f32 only, typed predicates; vector compute; explicit numerical attributes preserved |
| contraction | Separate f32×f32→f32 and bf16×bf16→f32 rules, checking both inputs and accumulator/result; row-major operand/acc layouts; input `acc` and output `result` require memory kind acc; SVE compute vector_engine, SME matrix_engine |
| async copy | Declared DRAM source and SRAM result ports, as in existing policy |
| tile copy / tile store | Preserve typed memory occurrences and use shared routing; no fabricated missing endpoint |
| fused convert→silu→mul | bf16→f32 conversion, f32 activation and multiply, typed graph-node predicates; vector compute; external port layouts solved per operand dtype |

Use stable ids `<prefix>.vector_add_<dtype>`, `.vector_mul_<dtype>`, `.vector_convert_<src>_<dst>[_row_major]`, `.vector_silu_f32`, `.reduce_sum_f32`, `.mma_f32`, `.mma_bf16`, `.async_copy`, `.tile_async_copy`, `.tile_store`, `.fused_convert_silu_mul`. Bundles include the dtype in their opaque names. No unused rule parameter or global `element_type` equality may accidentally pin a mixed-dtype operand to the first graph value. Layout requirements provide per-port VW; contraction bundle parameters need no duplicate width.

Rule cost lower bounds remain explicit seeds: add/mul 4, convert 2, silu 4, reduce 3, copy/tile-copy/tile-store 2, fused 3, SVE contraction 16, SME contraction 8. Complete-plan shared analysis, not these constants or emitter spelling, determines reported ranking. Do not assert SME is universally faster.

## 8. Packaging and extension files

Headers `include/LLK/Target/AArch64/Mapping/{SVEMappingTarget,SMEMappingTarget}.h` declare existing package-style APIs in `mlir::llk::target::{sve,sme}`:

```cpp
llvm::ArrayRef<llvm::StringRef> emitterKeys();
llvm::Expected<std::unique_ptr<mapping::MappingTarget>>
createMappingTarget(llvm::StringRef configurationRoot,
                    llvm::StringRef machinePath = {});
```

Sources in `lib/Target/AArch64/Mapping/` load `machines/arm-*-v2.yaml` and `mapping/arm-*/{layouts,rules}.llkmap` using `loadMappingTarget`, returning its file-backed target unchanged. Add sources to `LLKTargetMapping` and both factories to `registerAllMappingTargets`. No target subclass adds codegen capability. Machine override follows the existing loader validation contract.

Shared changes: `include/LLK/Perf/OpDAGBuilder.h`, new `include/LLK/Perf/ContractionSupport.h` and `lib/Perf/ContractionSupport.cpp`, `lib/Perf/{MicroDAG,MicroOpDAGBuilders,Legality}.cpp`, and `lib/Perf/CMakeLists.txt`. Existing normalized event/scheduling integration remains shared; change other consumers only where new tests demonstrate an actual vector-resource assumption.

Tests live in existing Machine, Perf, Mapping and MicroMapping categories and are explicitly registered. Use fully qualified namespaces in global test helpers. Helpers returning `llvm::Expected` propagate errors before dereference; nonfatal `EXPECT_TRUE` does not make dereference safe.

## 9. Acceptance fixtures and evidence

### 9.1 Required positive domain

1. Per-target configuration tests load profiles and policies, check every declared/rule emitter, and verify well-formed and malformed bundles. Loader assertions require SVE `acc.0` plus no matrix node, and SME `za.0` plus matrix compute; no assertion prohibits SVE accumulation.
2. Vector add/mul fixtures in f32, bf16 and i8 use typed rank-two tiles and exercise widths 16/32/64. Include bf16→f32 and reverse conversion, f32 silu/reduction and the stated fused graph. Floating and integer operations outside that table are unsupported.
3. One **identical source** contraction fixture per input dtype uses M=N=K=16, f32 initialization and output, rank-two row-major tiles, generic worker ownership, no spatial replication and no overlapping pipeline. It stages A/B/init DRAM→SRAM, copies init SRAM→acc, computes one MMA, copies its result acc→SRAM, and yields the result. Copying out precedes any vector epilogue. This is a bounded conformance program, not a complete tuned GEMM.
4. For each target/dtype, run extraction → search with canonical complete-plan evaluation → materialize/bind → verify → report → `micro-perf`. Assert selected SVE `vector_engine=svp` or SME `matrix_engine=za`, the actual accumulator memory id, live peak at or below 2048/4096, realized movement/storage identities and no incomplete-analysis reason. Retain source/input hashes; compare planner and materialized cycles/traffic/peak. Do not use engine/emitter strings alone as the placement test.
5. Freeze a reported plan id and replay it with identical target/search inputs in a fresh process; compare normalized IR/report. Test deterministic mode and a bounded exact control with truthful truncation.
6. Exercise both names through `registerAllMappingTargets` → `createRegisteredMappingTarget`, including machine override. Separately exercise both explicit `--micro-map` pass specifications. `--help` is not registration evidence.

The staged M=N=K=16 fixture is fixed by the revision probe: both f32 and bf16 variants complete SME mapping with a 1024-byte accumulator peak. An SVE diagnostic variant naming `engine="svp"` also yields a 1024-byte peak with the 2048-byte budget, but uses the old matrix-labeled interpreter path and is **storage evidence only**. It is not the acceptance source and does not validate the shared repair. Keep the canonical acceptance fixtures free of concrete engine ids, and rerun them after that repair. Their expected live accumulator peak is 1024 bytes on each target; a smaller-capacity negative control uses 512 bytes. Do not increase capacities or substitute an uninitialized allocation for the initializer to make tests pass.

### 9.2 Required negative controls

Reject missing/ambiguous/invalid compute identity, incompatible type pairs, unknown vector width, an unknown route endpoint, a removed required link, missing accumulator memory and a capacity reduced below the measured live peak. Verify rejection before top-K retention. A supported vector op sharing `svp` with contraction must contend for the same compute pool.

Every emitter reports `hasLowering()==false`; a valid bundle's `lower()` returns `target_lowering_unsupported`. Both targets have no codegen requirements and reject selected-target compilation. Static bind/report tests use the appropriate partial/static contract and assert physical completeness separately; do not require executable lowering from a static target or count reference execution as selected ARM evidence.

### 9.3 Claims and limits

Current CPU adapter schedules contain different tile sizes, owner symbols, storage and overlap assumptions. Their matmul/SwiGLU exports are **not accepted ARM workloads by this design**. A future adapter increment must generate feasible candidates; it must not replace the original workload or silently rewrite capacities. Small direct fixtures do not establish arbitrary DNN support.

Run the registered full suite on the local LLVM/MLIR build and CI LLVM 22.1.8; report exact source/toolchain/counts/skips. Required ARM static tests are host-independent. Preserve expected AVX2 host-capability failures/skips with their existing documented contract rather than calling them ARM execution. No test count is predicted in this design.

## 10. Implementation sequence

1. Shared contraction capability, selected engine/resource interpretation, costs and legality, with synthetic target-neutral tests.
2. Profiles plus finite layouts/rules, and an early full-pipeline typed contraction probe that fixes the accepted fixture/lifetime budget.
3. Target packages and registration, including static-only and malformed-configuration controls.
4. Typed positive/negative public pipelines, per-dtype coverage, parity, replay and registry/pass-spec tests for both targets.
5. Full regression evidence and current support documentation.

Each increment has its own failing-test/implementation/verification cycle. The paired target delivery depends on the shared repair; it must not be described as self-contained configuration-only work.

## 11. Executable-backend follow-up (#175)

Issue #175 is a separate workstream. Its design must compare at least a fixed-VL executable vector backend, true length-agnostic semantics/modeling, and SME stateful lowering. [Clang supports vector-length-specific SVE](https://clang.llvm.org/docs/LanguageExtensions.html#vectors-and-extended-vectors); [MLIR lowers linalg.matmul to ArmSME/FMOPA and supports emulator integration](https://mlir.llvm.org/docs/Dialects/ArmSME/). These are implementation options, not proof of a working LLK backend. The absence of scalable `LayoutAttr` or a canonical ZA op alone does not prove that every executable backend requires those dialect additions.

The follow-up must decide which width, accumulator lifetime and streaming facts must remain visible to canonical schedule/analysis and which can live in target lowering. Keep target-specific instructions in backend dialects. Cover selected bundle realization, feature and separate VL/SVL checks, tails/predication, dtype/numerical contracts, streaming entry/exit, calls and ZA ownership/preservation/spills/extraction; see [LLVM's SME ABI model](https://llvm.org/docs/AArch64SME.html).

Use the current [PR #133 selected-AVX2 acceptance](../../reviews/issue67-final-acceptance.md#verified-pr-133-implementation-evidence--2026-10-11), which supersedes historical host-portable-only sections. Selected-target and reference execution have distinct identities; current evidence includes 13 required selected AVX2 passes, target/feature checks and JIT artifacts. Do not lower the baseline by quoting the superseded reference-mode account. Object/disassembly checks are an additional evidence decision, not interchangeable with numeric invocation.

Define evidence tiers explicitly: modeled static plan; cross-lowered artifact; emulated selected-target numerical execution; native selected-target numerical execution. Artifacts carry source, plan, machine/policy, compiler/LLVM, ABI, target features, VL/SVL and runner identity, with no silent reference fallback. Emulation can supply functional evidence but not native performance measurements. Any deliberately weaker acceptance must enumerate omitted checks. This revision changes the spec/plan, not the remote issue or delivered support matrix.
