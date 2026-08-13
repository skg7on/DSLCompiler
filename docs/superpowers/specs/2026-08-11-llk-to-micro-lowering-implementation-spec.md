# LLK/Linalg to Micro-IR Lowering Implementation Spec

**Status:** Draft
**Date:** 2026-08-11
**Parent Spec:** [Canonical Micro-IR Redesign Spec](2026-08-11-canonical-micro-ir-redesign.md)

---

## 1. Goal

Implement an export path from the current LLK/Linalg scheduling pipeline into concrete `micro.kernel` and optional `micro.search_space` IR. The lowering must preserve the current AVX2/JIT pipeline while producing a canonical execution artifact for performance modeling and future backends.

The exported Micro-IR is tile-centric. Lowering should produce:

```text
Linalg/Tensor semantics
    -> logical tile dataflow
    -> materialized memory tiles
    -> owner-mapped execution tiles
    -> pipeline/sync structure
    -> MMA/vector/reduction instruction fragments
```

The MVP may keep existing op names such as `micro.async_copy`, `micro.mma`, and `micro.vector`, but the values and attributes must preserve tile shape, dtype, layout, memory space, and owner information.

---

## 2. Integration Point

Current `llk-compile` pipeline:

```text
Triton staged lowering
LLKToLinalg
canonicalize
ShapeSpecialization
ScheduleSelection
FuseDoubleContraction
canonicalize
PackWeights
TileAndVectorize
canonicalize
LinearizeForall
SerialParallelDispatch
OneShotBufferize
ScratchAnalysis
LinalgToLoops
ForallToLLRT
VectorToLLVM
SCF/Arith/Math/Func/MemRef to LLVM
```

Initial `LLKToMicro` export path should run after:

```text
LLKToLinalg
canonicalize
ShapeSpecialization
ScheduleSelection
FuseDoubleContraction
canonicalize
PackWeights
```

and before:

```text
TileAndVectorize
OneShotBufferize
LLVM lowering
```

Reason: the export needs semantic structure, selected schedule data, and fusion/packing decisions, but should still see tensor-level shapes and avoid post-bufferization noise.

---

## 3. Repository Files

Create:

```text
include/LLK/Conversion/LLKToMicro/LLKToMicro.h
lib/Conversion/LLKToMicro/LLKToMicro.cpp

test/Conversion/LLKToMicro/swiglu_to_micro.mlir
test/Conversion/LLKToMicro/matmul_to_micro.mlir
test/Conversion/LLKToMicro/search_space_from_schedule.mlir
test/Conversion/LLKToMicro/lowering_invalid.mlir
```

Modify:

```text
CMakeLists.txt
tools/llk-opt/llk-opt.cpp
tools/llk-compile/llk-compile.cpp
include/LLK/Transforms/Common/ScheduleLoader.h
lib/Transforms/Common/ScheduleLoader.cpp
```

Optional later files:

```text
include/LLK/Conversion/LLKToMicro/ScheduleToMicro.h
lib/Conversion/LLKToMicro/ScheduleToMicro.cpp
```

Split helper files only when `LLKToMicro.cpp` becomes too large to review comfortably.

---

## 4. Public Interfaces

Header:

```cpp
#ifndef LLK_CONVERSION_LLKTOMICRO_LLKTOMICRO_H
#define LLK_CONVERSION_LLKTOMICRO_LLKTOMICRO_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir::llk {

std::unique_ptr<mlir::Pass> createLLKToMicroPass();
std::unique_ptr<mlir::Pass> createLLKToMicroSearchSpacePass();

} // namespace mlir::llk

#endif // LLK_CONVERSION_LLKTOMICRO_LLKTOMICRO_H
```

Pass arguments:

```text
--llk-to-micro
--llk-to-micro-search-space
```

The concrete pass emits `micro.kernel`.

The search-space pass emits `micro.search_space` from the selected schedule and legal nearby choices.

---

## 5. Schedule Data Required by Lowering

Existing `ScheduleEntry`:

```cpp
struct ScheduleEntry {
  int64_t BM{0}, BN{0}, BK{0};
  int64_t VM{0}, VN{0};
  int64_t vector_width{0};
  int64_t num_threads{1};
  int64_t grain_size{1};
  std::string parallel_axis;
};
```

Extend it for micro lowering:

```cpp
struct ScheduleEntry {
  int64_t BM{0}, BN{0}, BK{0};
  int64_t VM{0}, VN{0};
  int64_t vector_width{0};
  int64_t num_threads{1};
  int64_t grain_size{1};
  std::string parallel_axis;

  int64_t pipeline_stages{1};
  int64_t prefetch_distance{0};
  std::string memory_path{"dram:sram:acc"};
  std::string mma_shape{"16x16x32"};
  std::string accumulator_space{"acc"};
  std::string tile_layout{"row_major"};
  std::string owner_mapping{"worker"};
  std::string fragment_owner{"vector_engine"};
  std::string fragment_shape{"16x16x32"};
  bool enable_tile_masks{true};
};
```

Backward compatibility:

- missing `pipeline_stages` defaults to 1
- missing `memory_path` defaults to `dram:sram:acc`
- missing `mma_shape` defaults to `16x16x32`
- missing `fragment_shape` defaults to `mma_shape`
- missing `owner_mapping` defaults to `worker`
- missing `fragment_owner` defaults to `vector_engine`
- existing JSON schedule entries remain loadable

The first implementation should not migrate `schedules/schedule_db.json`; it only adds optional fields.

---

## 6. Concrete Lowering: Fused SwiGLU

### 6.1 Input Pattern

The first concrete lowering supports `llk.fused_swiglu` before `LLKToLinalg` or its fused Linalg equivalent after `FuseDoubleContraction`.

Preferred MVP:

Lower directly from `llk.fused_swiglu` when `--llk-to-micro` is run before `LLKToLinalg`.

Follow-up:

Lower from scheduled/fused Linalg once pass placement is stable.

Reason: direct LLK lowering produces useful micro examples sooner and avoids fragile pattern matching in the first milestone.

### 6.2 Output Shape

For `M x K` input, `K x N` weights, output `M x N`, schedule `BM/BN/BK`:

```text
spatial_for bm over M step BM
spatial_for bn over N step BN
logical tile_view X[BM x BK]
logical tile_view Wg[BK x BN]
logical tile_view Wu[BK x BN]
alloc gate accumulator BM x BN in acc
alloc up accumulator BM x BN in acc
pipeline stages = schedule.pipeline_stages
  temporal for bk over K step BK
    async_copy materializes X tile DRAM -> SRAM
    async_copy materializes Wg tile DRAM -> SRAM
    async_copy materializes Wu tile DRAM -> SRAM
    wait copies
    partition materialized tiles into instruction fragments
    mma X, Wg -> gate accumulator
    mma X, Wu -> up accumulator
vector silu gate
vector mul gate, up
store output ACC -> DRAM
```

### 6.3 Example Output

```mlir
micro.kernel @fused_swiglu_M4_N4096_K4096
    attributes {
      workload = "fused_swiglu",
      target = "x86-avx2-cpu",
      schedule_id = "schedule_db_v1_bucket4"
    } {
  micro.spatial_for %bm = 0 to %M step 32 map = #micro.map<worker> {
    micro.spatial_for %bn = 0 to %N step 64 map = #micro.map<lane> {
      %gate_acc = micro.alloc tensor<32x64xf32> memory = #micro.memory<acc>
      %up_acc = micro.alloc tensor<32x64xf32> memory = #micro.memory<acc>

      micro.pipeline stages = 1 {
        micro.for %bk = 0 to %K step 64 {
          %x_view = micro.tile_view %arg0[%bm, %bk]
              {shape = [32, 64], layout = #micro.layout<row_major>}
              : tensor<?x?xbf16> -> !micro.tile<32x64xbf16>
          %wg_view = micro.tile_view %arg1[%bk, %bn]
              {shape = [64, 64], layout = #micro.layout<row_major>}
              : tensor<?x?xbf16> -> !micro.tile<64x64xbf16>
          %wu_view = micro.tile_view %arg2[%bk, %bn]
              {shape = [64, 64], layout = #micro.layout<row_major>}
              : tensor<?x?xbf16> -> !micro.tile<64x64xbf16>
          %x, %x_tok = micro.async_copy %x_view
              {src_memory = #micro.memory<dram>,
               dst_memory = #micro.memory<sram>,
               owner = #micro.owner<worker>}
              : tensor<32x64xbf16>
          %wg, %wg_tok = micro.async_copy %wg_view
              {src_memory = #micro.memory<dram>,
               dst_memory = #micro.memory<sram>,
               owner = #micro.owner<worker>}
              : tensor<64x64xbf16>
          %wu, %wu_tok = micro.async_copy %wu_view
              {src_memory = #micro.memory<dram>,
               dst_memory = #micro.memory<sram>,
               owner = #micro.owner<worker>}
              : tensor<64x64xbf16>
          micro.wait %x_tok, %wg_tok, %wu_tok
          %x_frag = micro.tile_partition %x
              {shape = [16, 32], owner = #micro.owner<vector_engine>}
              : tensor<32x64xbf16> -> !micro.tile<16x32xbf16>
          %wg_frag = micro.tile_partition %wg
              {shape = [32, 16], owner = #micro.owner<vector_engine>}
              : tensor<64x64xbf16> -> !micro.tile<32x16xbf16>
          %wu_frag = micro.tile_partition %wu
              {shape = [32, 16], owner = #micro.owner<vector_engine>}
              : tensor<64x64xbf16> -> !micro.tile<32x16xbf16>
          micro.mma %x_frag, %wg_frag, %gate_acc
              {shape = [16, 16, 32], input = #micro.dtype<bf16>,
               accumulator = #micro.dtype<f32>,
               owner = #micro.owner<vector_engine>}
          micro.mma %x_frag, %wu_frag, %up_acc
              {shape = [16, 16, 32], input = #micro.dtype<bf16>,
               accumulator = #micro.dtype<f32>,
               owner = #micro.owner<vector_engine>}
        }
      }

      %gate = micro.vector "silu" %gate_acc {math_mode = "bounded_fast"}
      %out = micro.vector "mul" %gate, %up_acc
      micro.store %out, %arg3
          {src_memory = #micro.memory<acc>,
           dst_memory = #micro.memory<dram>}
          : tensor<32x64xbf16>
    }
  }
  micro.yield
}
```

Indexing details can be represented as attributes in the MVP if full subview syntax is too large for the first pass. The performance model needs tile shapes and memory spaces first; exact SSA index arithmetic can follow.

The example uses both `micro.tile_view` and legacy tensor-typed compute spellings to show the migration path. Once `!micro.tile` is implemented, `micro.tile_async_copy`, `micro.tile_mma`, `micro.tile_vector`, and `micro.tile_store` should be the preferred spellings.

---

## 7. Concrete Lowering: Matmul

Matmul lowering emits:

```text
spatial_for bm
spatial_for bn
alloc acc
pipeline
  for bk
    async_copy A
    async_copy B
    wait
    mma
store C
```

This should share helper functions with SwiGLU:

```cpp
struct MicroTileConfig {
  int64_t BM;
  int64_t BN;
  int64_t BK;
  int64_t stages;
  llvm::SmallVector<int64_t, 3> mmaShape;
  llvm::SmallVector<int64_t, 3> fragmentShape;
  mlir::micro::MemorySpace accumulatorSpace;
  mlir::micro::LayoutAttr layout;
  mlir::micro::OwnerAttr owner;
  mlir::micro::OwnerAttr fragmentOwner;
};

MicroTileConfig getTileConfig(Operation *op, const ScheduleEntry &schedule);
Value createAsyncTileCopy(...);
void createMmaLoopNest(...);
```

---

## 8. Search-Space Lowering

`--llk-to-micro-search-space` emits legal choices rather than one candidate.

Initial choices derive from existing `llk-tune` grid:

```text
BM: [1, 4, 8, 16, 32, 64]
BN: [16, 32, 64, 128, 256]
BK: [32, 64, 128, 256]
tile_hierarchy: ["worker", "worker/vector_engine"]
tile_layout: ["row_major", "blocked_16x16"]
owner_mapping: ["worker", "worker_lane"]
fragment_shape: ["16x16x32", "8x8x32"]
VM: [1, 2, 4]
VN: [4, 8]
vector_width: [8]
num_threads: [1, 2, 4, 8]
grain_size: [1, 2, 4]
pipeline_stages: [1, 2]
```

Constraints:

```text
sram_capacity
acc_capacity
mma_compatible
tile_hierarchy_compatible
layout_supported
owner_supported
fragment_compatible
tail_supported
vector_width_supported
mapping_extent
pipeline_live_tiles
```

M-bucket rules from existing `llk-tune`:

- M bucket 0 only allows `BM = 1`
- M bucket >= 3 rejects `BM <= 4`

These rules become constraints rather than hard-coded generator branches.

## 8.1 Tile-Centric Lowering Rules

Lowering must follow these rules:

- create logical tile views before materializing data movement
- represent packing as a layout attribute when it is a reinterpretation
- represent packing as a physical layout transform when bytes are rearranged
- materialize DRAM to SRAM/RF/ACC movement with tile copy/load/store ops
- attach owner mapping to spatial loops and execution tiles
- partition execution tiles into instruction fragments before MMA/vector/reduce
- emit waits and barriers at the scope where tile producers and consumers meet
- preserve enough tile metadata for `micro-perf` to compute bytes, capacity, and utilization without re-inferring schedule intent from names

---

## 9. Pass Behavior

### 9.1 `LLKToMicroPass`

Algorithm:

```text
for each func.func:
  create one micro.kernel per supported LLK root op
  infer operation kind
  infer M/N/K and dtype
  classify M bucket
  load matching ScheduleEntry
  select best schedule using existing selectBest
  lower supported op to concrete micro.kernel
  erase or leave original op based on mode
```

MVP mode:

- produce a new `micro.kernel` in the module
- leave original functions untouched

Rationale: this makes export non-destructive and keeps FileCheck simpler.

### 9.2 `LLKToMicroSearchSpacePass`

Algorithm:

```text
for each supported LLK root op:
  infer workload and shape
  emit micro.search_space with param choices
  emit constraints based on machine-independent legality
  emit objective latency with secondary metrics
```

This pass does not require a machine YAML. Machine-dependent pruning happens in the tuner or candidate verifier.

---

## 10. `llk-compile --emit=micro`

Add CLI enum:

```text
--emit=llvm
--emit=mlir
--emit=micro
--emit=micro-search
```

MVP:

- default remains existing behavior
- `--emit=micro` runs pipeline through `LLKToMicroPass`, prints module, exits
- `--emit=micro-search` runs `LLKToMicroSearchSpacePass`, prints module, exits

No JIT execution occurs in micro emit modes.

---

## 11. CMake Integration

Add library:

```cmake
add_mlir_library(LLKToMicro
    lib/Conversion/LLKToMicro/LLKToMicro.cpp
    DEPENDS LLKDialect MicroDialect
    LINK_LIBS PUBLIC LLKDialect MicroDialect LLKCommon MLIRFuncDialect
                     MLIRTensorDialect MLIRPass MLIRTransforms
)
```

Link into:

```text
llk-opt
llk-compile
```

---

## 12. Tests

### 12.1 Direct LLK to Micro

`test/Conversion/LLKToMicro/swiglu_to_micro.mlir`

Checks:

- `micro.kernel`
- two nested `micro.spatial_for`
- logical `micro.tile_view` ops for X, Wg, and Wu
- two accumulator allocations
- three async copies
- wait
- tile partitions or fragment metadata
- two MMA ops
- silu and mul vector ops
- store

### 12.2 Matmul to Micro

`test/Conversion/LLKToMicro/matmul_to_micro.mlir`

Checks:

- logical tile views for A and B
- one accumulator allocation
- two async copies
- one MMA op inside K loop
- store

### 12.3 Search Space

`test/Conversion/LLKToMicro/search_space_from_schedule.mlir`

Checks:

- `micro.search_space`
- params for BM/BN/BK/vector_width/num_threads/grain_size/pipeline_stages
- params or constraints for tile layout, owner mapping, and fragment shape
- constraints
- objective

### 12.4 Invalid Cases

`test/Conversion/LLKToMicro/lowering_invalid.mlir`

Cases:

- dynamic rank unsupported
- unsupported dtype
- missing schedule entry with no fallback
- malformed schedule optional fields
- unsupported owner mapping
- unsupported tile layout
- fragment shape that cannot divide the selected tile shape without masks

---

## 13. Verification Commands

```bash
cmake --build build --target llk-opt llk-compile

build/llk-opt --llk-to-micro \
  test/Conversion/LLKToMicro/swiglu_to_micro.mlir \
  | FileCheck test/Conversion/LLKToMicro/swiglu_to_micro.mlir

build/llk-opt --llk-to-micro \
  test/Conversion/LLKToMicro/matmul_to_micro.mlir \
  | FileCheck test/Conversion/LLKToMicro/matmul_to_micro.mlir

build/llk-opt --llk-to-micro-search-space \
  test/Conversion/LLKToMicro/search_space_from_schedule.mlir \
  | FileCheck test/Conversion/LLKToMicro/search_space_from_schedule.mlir

build/llk-opt --verify-diagnostics --split-input-file \
  --llk-to-micro test/Conversion/LLKToMicro/lowering_invalid.mlir
```

---

## 14. Acceptance Criteria

- `LLKToMicro` builds and registers in `llk-opt`.
- `llk-compile --emit=micro` prints concrete micro IR and exits without JIT.
- `llk-compile --emit=micro-search` prints search-space micro IR and exits without JIT.
- current `llk-compile` default behavior is unchanged.
- SwiGLU and matmul lowering tests pass.
- search-space lowering test passes.
- generated Micro-IR includes tile shape, dtype, layout, memory, owner, and fragment metadata.
- invalid lowering diagnostics are actionable.
