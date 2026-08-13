# Micro-IR Dialect Implementation Spec

**Status:** Draft
**Date:** 2026-08-11
**Parent Spec:** [Canonical Micro-IR Redesign Spec](2026-08-11-canonical-micro-ir-redesign.md)

---

## 1. Goal

Implement the `micro` MLIR dialect as the canonical execution IR for DNN kernels. The dialect must represent both concrete hardware-near execution schedules and search-space metadata for auto-scheduling.

The dialect is tile-centric. The primary execution value is:

```text
Tile = (shape, dtype, layout, memory_space, owner)
```

The implementation should prefer first-class `!micro.tile` values. If custom type parsing is staged, tensor values with required tile attributes may be used only as a temporary bridge.

The MVP is parser, printer, verifier, and FileCheck coverage. It does not require a backend lowering in the first task batch.

---

## 2. Scope

### 2.1 MVP Concrete Execution Ops

```text
micro.kernel
micro.for
micro.spatial_for
micro.pipeline
micro.tile_view
micro.tile_partition
micro.tile_alloc
micro.alloc
micro.tile_async_copy
micro.async_copy
micro.wait
micro.tile_mma
micro.mma
micro.tile_vector
micro.vector
micro.tile_reduce
micro.reduce
micro.tile_store
micro.store
micro.yield
```

### 2.2 MVP Search Ops

```text
micro.search_space
micro.param
micro.constraint
micro.objective
micro.candidate
```

### 2.3 MVP Attributes

```text
#micro.memory<dram>
#micro.memory<l2>
#micro.memory<sram>
#micro.memory<rf>
#micro.memory<acc>
#micro.memory<scratch>

#micro.map<cluster.x>
#micro.map<cluster.y>
#micro.map<core.x>
#micro.map<core.y>
#micro.map<pe.x>
#micro.map<pe.y>
#micro.map<lane>
#micro.map<worker>
#micro.map<matrix_engine>
#micro.map<vector_engine>
#micro.map<dma>

#micro.owner<cluster>
#micro.owner<core>
#micro.owner<warp>
#micro.owner<wave>
#micro.owner<subgroup>
#micro.owner<pe_group>
#micro.owner<pe>
#micro.owner<lane>
#micro.owner<worker>
#micro.owner<matrix_engine>
#micro.owner<vector_engine>
#micro.owner<dma>

#micro.layout<row_major>
#micro.layout<col_major>
#micro.layout<blocked>
#micro.layout<vectorized>
#micro.layout<swizzled>

#micro.dtype<f32>
#micro.dtype<f16>
#micro.dtype<bf16>
#micro.dtype<i32>
#micro.dtype<i8>
```

Composite dtypes, sparse dtypes, communication ops, and target lowerings are not required for the first implementation batch.

---

## 3. Repository Files

Create:

```text
include/LLK/Dialect/Micro/MicroDialect.h
include/LLK/Dialect/Micro/MicroDialect.td
include/LLK/Dialect/Micro/MicroOps.td
include/LLK/Dialect/Micro/MicroTypes.td

lib/Dialect/Micro/MicroDialect.cpp
lib/Dialect/Micro/MicroOps.cpp

test/Dialect/Micro/ops.mlir
test/Dialect/Micro/ops_invalid.mlir
test/Dialect/Micro/tile_ops.mlir
test/Dialect/Micro/tile_ops_invalid.mlir
test/Dialect/Micro/search_space.mlir
test/Dialect/Micro/search_space_invalid.mlir
```

Modify:

```text
CMakeLists.txt
tools/llk-opt/llk-opt.cpp
```

Optional tool added in a later task:

```text
tools/micro-opt/micro-opt.cpp
```

Optional later split:

```text
include/LLK/Dialect/Micro/MicroTileOps.td
lib/Dialect/Micro/MicroTileOps.cpp
```

Do not split tile ops in the first patch unless `MicroOps.td` becomes difficult to review.

The first implementation can register `micro` in `llk-opt` to avoid a new tool target until dialect parse/print is stable.

---

## 4. Namespace and Library Naming

MLIR dialect prefix:

```text
micro
```

C++ namespace:

```cpp
namespace mlir::micro
```

CMake target:

```text
MicroDialect
```

Generated TableGen targets:

```text
MicroDialectGen
MicroOpsGen
```

Header include convention:

```cpp
#include "LLK/Dialect/Micro/MicroDialect.h"
```

The files live under `include/LLK` because this repository already keeps project dialects there, but the dialect itself is not an `llk` semantic dialect.

---

## 5. Dialect Definition

`include/LLK/Dialect/Micro/MicroDialect.td` defines:

```tablegen
def Micro_Dialect : Dialect {
  let name = "micro";
  let cppNamespace = "::mlir::micro";
  let summary = "Canonical DNN execution micro-IR";
  let description = [{
    The micro dialect represents scheduled tensor execution in terms of
    compute engines, memory hierarchy, data movement, mapping, pipeline,
    synchronization, and search-space metadata.
  }];
}
```

`MicroTypes.td` should define enum-style attributes rather than custom storage classes in the MVP. That keeps parse/print and verifier implementation small.

Exception: `!micro.tile` and `!micro.async_token` should be real types once the dialect skeleton is stable. The temporary tensor-plus-attributes form is allowed only if it reduces risk in the first TableGen patch.

---

## 6. Attribute Definitions

### 6.1 Memory Space Attribute

Use an enum attribute:

```tablegen
def Micro_DRAM : I32EnumAttrCase<"dram", 0, "dram">;
def Micro_L2 : I32EnumAttrCase<"l2", 1, "l2">;
def Micro_SRAM : I32EnumAttrCase<"sram", 2, "sram">;
def Micro_RF : I32EnumAttrCase<"rf", 3, "rf">;
def Micro_ACC : I32EnumAttrCase<"acc", 4, "acc">;
def Micro_Scratch : I32EnumAttrCase<"scratch", 5, "scratch">;

def Micro_MemorySpace : I32EnumAttr<"MemorySpace", "Memory space", [
  Micro_DRAM,
  Micro_L2,
  Micro_SRAM,
  Micro_RF,
  Micro_ACC,
  Micro_Scratch
]> {
  let cppNamespace = "::mlir::micro";
  let genSpecializedAttr = 0;
}

def Micro_MemorySpaceAttr :
    EnumAttr<Micro_Dialect, Micro_MemorySpace, "memory"> {
  let assemblyFormat = "`<` $value `>`";
}
```

### 6.2 Mapping Target Attribute

Use enum values for:

```text
cluster_x
cluster_y
core_x
core_y
pe_x
pe_y
lane
worker
matrix_engine
vector_engine
dma
```

The printed form should use dots for hierarchy:

```text
#micro.map<cluster.x>
#micro.map<cluster.y>
#micro.map<worker>
```

If TableGen enum cases cannot directly print dotted values cleanly, keep enum symbols as `cluster_x` and provide stringify/symbolize helpers that parse/print dotted spellings in `MicroOps.cpp`.

### 6.3 DType Attribute

The MVP dtype attribute supports:

```text
f32
f16
bf16
i32
i8
```

Use this dtype attribute for machine modeling metadata. MLIR tensor element types still carry actual tensor element types. Verifiers must ensure explicit dtype attributes agree with tensor element types when both are present.

### 6.4 Layout Attribute

Use an enum-style layout kind plus optional dictionary parameters.

Required kind values:

```text
row_major
col_major
blocked
vectorized
swizzled
```

Assembly examples:

```mlir
#micro.layout<row_major>
#micro.layout<blocked, block = [16, 16]>
#micro.layout<vectorized, vector = 8, align = 32>
#micro.layout<swizzled, swizzle = "xor", banks = 32>
```

Verifier:

- `block` extents must be positive.
- `vector` width must be positive.
- `align` must be positive and a power of two.
- `banks` must be positive when present.
- `swizzle` requires `swizzled` layout kind.

### 6.5 Owner Attribute

Use enum values for:

```text
cluster
core
warp
wave
subgroup
pe_group
pe
lane
worker
matrix_engine
vector_engine
dma
```

Assembly examples:

```mlir
#micro.owner<worker>
#micro.owner<matrix_engine>
#micro.owner<pe_group>
```

Verifier:

- owner values must be known.
- owner compatibility with MachineModel is checked by machine-aware verification and the performance layer, not by the dialect parser alone.

### 6.6 Tile Type

Preferred assembly:

```mlir
!micro.tile<32x64xbf16,
             layout = #micro.layout<row_major, vector = 8>,
             memory = #micro.memory<sram>,
             owner = #micro.owner<worker>>
```

Type parameters:

- ranked shape, with static dimensions for the MVP
- MLIR element type or `#micro.dtype`
- optional layout, default `#micro.layout<row_major>`
- optional memory space, required after materialization
- optional owner, required for execution tiles and fragments

Verifier:

- dimensions must be positive or explicitly dynamic.
- dtype must map to a supported Micro dtype.
- materialized memory tiles require a memory space.
- execution tiles and instruction fragments require an owner.
- instruction fragment shape must fit inside parent tile shape when statically known.

Temporary bridge:

If implementing custom tile type parsing blocks initial progress, allow tensor result types plus attributes named `micro.layout`, `micro.memory`, and `micro.owner`. This bridge must be removed once tile op tests pass with `!micro.tile`.

---

## 7. Concrete Execution Ops

### 7.1 `micro.kernel`

Purpose: top-level container for a concrete execution schedule.

Shape:

```mlir
micro.kernel @name
    attributes {workload = "fused_swiglu", target = "x86-avx2-cpu"} {
  ...
}
```

TableGen requirements:

- symbol name
- one region
- isolated from above
- optional attributes:
  - `workload`
  - `target`
  - `candidate`
  - `schedule_id`

Verifier:

- body region must not be empty
- search-only ops are forbidden inside `micro.kernel`
- terminator is `micro.yield`
- `workload` is optional but, if present, must be a non-empty string

### 7.2 `micro.for`

Purpose: temporal loop.

Shape:

```mlir
micro.for %i = 0 to %K step 64 {
  ...
}
```

MVP representation:

- model it after `scf.for` in syntax and operands
- use `index` typed lower bound, upper bound, and step
- no iter_args in the first implementation

Verifier:

- step must be positive if statically known
- body block has one induction variable
- search ops are not allowed inside a concrete `micro.for`

### 7.3 `micro.spatial_for`

Purpose: spatial mapping over machine resources.

Shape:

```mlir
micro.spatial_for %bm = 0 to %M step 32 map = #micro.map<worker> {
  ...
}
```

Operands:

- lower bound
- upper bound
- step

Attributes:

- required `map`
- optional `parallel_axis`

Verifier:

- `map` must be a known `#micro.map`
- step must be positive if statically known
- nested `micro.spatial_for` mappings may repeat in MVP, but the verifier should emit a warning later if the same finite mapping dimension is oversubscribed

### 7.4 `micro.pipeline`

Purpose: mark pipeline overlap between movement and compute.

Shape:

```mlir
micro.pipeline stages = 2 {
  ...
}
```

Attributes:

- required integer `stages`

Verifier:

- `stages >= 1`
- nested `micro.pipeline` is rejected in MVP
- `stages > 1` requires at least one async op in the body

### 7.5 `micro.alloc`

Purpose: allocate logical tile storage in a memory space.

Shape:

```mlir
%acc = micro.alloc tensor<32x64xf32> memory = #micro.memory<acc>
```

Result:

- tensor type in MVP

Attributes:

- required memory space

Verifier:

- result must be shaped type
- memory space must not be `dram`; DRAM operands represent external buffers, not local allocations
- static footprint should be computable when all dimensions are static

### 7.5a Tile Construction and Materialization Ops

The tile MVP adds these ops. They may live in `MicroOps.td` initially.

#### `micro.tile_view`

Purpose: create a logical zero-cost tile view from a tensor or larger tile.

Shape:

```mlir
%t = micro.tile_view %A[%i, %j]
    {shape = [32, 64], layout = #micro.layout<row_major>}
    : tensor<?x?xbf16> -> !micro.tile<32x64xbf16>
```

Verifier:

- result is `!micro.tile` or a staged tensor-plus-tile-attributes value.
- view shape has positive static dimensions where known.
- no memory allocation or copy token is produced.

#### `micro.tile_partition`

Purpose: partition a tile into smaller logical or execution fragments.

Shape:

```mlir
%frag = micro.tile_partition %tile
    {shape = [16, 16], owner = #micro.owner<vector_engine>}
    : !micro.tile<32x64xbf16> -> !micro.tile<16x16xbf16>
```

Verifier:

- fragment shape divides the parent shape or carries explicit mask/tail metadata.
- owner, when present, is a known owner attribute.
- partition is logical unless followed by materialization or compute consumption.

#### `micro.tile_alloc`

Purpose: allocate a materialized tile in a memory space.

Shape:

```mlir
%acc = micro.tile_alloc
    {shape = [32, 64], dtype = #micro.dtype<f32>,
     memory = #micro.memory<acc>, owner = #micro.owner<worker>}
    : !micro.tile<32x64xf32>
```

Verifier:

- memory must not be DRAM for local allocations.
- dtype, memory, layout, and owner must agree with the result tile type.
- static footprint must be computable when dimensions are static.

### 7.6 `micro.async_copy`

Purpose: asynchronous movement between memory spaces.

Shape:

```mlir
%token, %tile = micro.async_copy %src[%i, %j]
    : #micro.memory<dram> -> #micro.memory<sram>
    tensor<32x64xbf16>
```

MVP simplification:

- return a tile value and an async token if multi-result printing is practical
- otherwise return only a token and use a named destination allocation

Recommended MVP interface:

```mlir
%tile, %tok = micro.async_copy %src
    {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>}
    : tensor<32x64xbf16>
```

Verifier:

- source and destination memory spaces differ
- `src_memory` and `dst_memory` are known
- result tile type is shaped
- token result has a dedicated token type or `!micro.async_token`

`micro.tile_async_copy` is the tile-typed spelling:

```mlir
%tile, %tok = micro.tile_async_copy %logical
    to #micro.memory<sram> owner = #micro.owner<worker>
    : !micro.tile<32x64xbf16> -> !micro.tile<32x64xbf16>, !micro.async_token
```

Verifier:

- source must be a tile or staged tile-compatible tensor.
- destination memory differs from source materialized memory when the source has one.
- destination memory and owner are present in the result tile metadata.
- byte count must be derivable from shape, dtype, layout, and padding.

### 7.7 `micro.wait`

Purpose: wait on async movement or async compute tokens.

Shape:

```mlir
micro.wait %tok0, %tok1
```

Verifier:

- all operands have `!micro.async_token`
- at least one token operand is required

### 7.8 `micro.mma`

Purpose: matrix/tensor engine compute one level above vendor ISA.

Shape:

```mlir
micro.mma %a, %b, %acc
    {shape = [16, 16, 32], input = #micro.dtype<bf16>,
     accumulator = #micro.dtype<f32>}
```

Operands:

- lhs tile
- rhs tile
- accumulator tile

Attributes:

- `shape`: dense i64 array of length 3, interpreted as M, N, K
- `input`: dtype attr
- `accumulator`: dtype attr
- optional `engine`: string

Verifier:

- shape has exactly three positive dimensions
- lhs/rhs/acc result shapes are compatible when static
- accumulator dtype matches accumulator tensor element type when expressible
- tile operands, when present, have compatible layout and owner metadata
- fragment shape divides or masks the input execution tiles

`micro.tile_mma` is the explicit tile spelling. It has the same semantic contract but requires tile operands and returns or mutates a tile accumulator according to the selected MLIR op style.

### 7.9 `micro.vector`

Purpose: vector elementwise operation.

Shape:

```mlir
%gate = micro.vector "silu" %acc_g {math_mode = "bounded_fast"}
%out = micro.vector "mul" %gate, %acc_u
```

MVP:

- generic op string attribute `op`
- variadic operands
- one result

Verifier:

- op must be one of `add`, `sub`, `mul`, `div`, `exp`, `silu`, `sigmoid`, `max`, `min`, `select`, `convert`
- all shaped operands must have identical static shapes unless op is `select`
- result type must be shaped
- tile operands, when present, must have compatible owner and layout metadata

`micro.tile_vector` is the explicit tile spelling and should be used once `!micro.tile` is available.

### 7.10 `micro.reduce`

Purpose: reduce along one or more axes.

Shape:

```mlir
%m = micro.reduce "max" %scores {axis = 1}
```

Verifier:

- op is `sum`, `max`, `min`, or `prod`
- `axis` is in range for static ranked operands
- result rank is operand rank minus one in MVP
- tile reduction outputs must define accumulator dtype, layout, and owner

`micro.tile_reduce` is the explicit tile spelling and should be used once `!micro.tile` is available.

### 7.11 `micro.store`

Purpose: store a tile to a destination memory space.

Shape:

```mlir
micro.store %out, %Y
    {src_memory = #micro.memory<acc>, dst_memory = #micro.memory<dram>}
    : tensor<32x64xbf16>
```

Verifier:

- source and destination memory spaces differ
- stored value is shaped
- destination is external or previously allocated

`micro.tile_store` is the tile-typed spelling:

```mlir
micro.tile_store %tile, %Y[%i, %j]
    to #micro.memory<dram>
    : !micro.tile<32x64xbf16>
```

Verifier:

- stored value is a tile or staged tile-compatible tensor.
- destination memory differs from source memory unless this is an explicit writeback/update.
- byte count is derivable from tile metadata.

### 7.12 `micro.yield`

Terminator for `micro.kernel`, `micro.for`, `micro.spatial_for`, and `micro.pipeline`.

MVP:

- no operands
- add operands only when concrete loop-carried state is required later

---

## 8. Search Ops

### 8.1 `micro.search_space`

Purpose: top-level container for legal schedule choices.

Shape:

```mlir
micro.search_space @swiglu_space attributes {workload = "fused_swiglu"} {
  ...
}
```

Verifier:

- body must contain at least one `micro.param`
- `workload` must be present and non-empty
- concrete execution ops are not allowed inside `micro.search_space`

### 8.2 `micro.param`

Purpose: declare one tunable parameter.

Shape:

```mlir
%BM = micro.param "BM" : i64 {choices = [8, 16, 32, 64]}
```

Verifier:

- parameter name is non-empty
- choices array is non-empty
- all choices are positive integers for MVP
- duplicate parameter names in the same search space are rejected

### 8.3 `micro.constraint`

Purpose: declare typed legality constraints.

MVP design:

Do not implement a free-form expression language. Represent constraints as named constraint records:

```mlir
micro.constraint "sram_capacity"
    {params = ["BM", "BN", "BK"], memory = #micro.memory<sram>}

micro.constraint "mma_compatible"
    {params = ["BM", "BN", "BK"], input = #micro.dtype<bf16>,
     accumulator = #micro.dtype<f32>}
```

Supported MVP constraints:

```text
sram_capacity
acc_capacity
mma_compatible
mapping_extent
tail_supported
vector_width_supported
tile_hierarchy_compatible
layout_supported
owner_supported
fragment_compatible
pipeline_live_tiles
```

Verifier:

- constraint name is supported
- referenced params exist in the nearest `micro.search_space`
- required attributes for that constraint are present

### 8.4 `micro.objective`

Purpose: ranking objective for candidate search.

Shape:

```mlir
micro.objective minimize "latency"
    secondary ["matrix_utilization", "dram_bytes"]
```

Verifier:

- direction is `minimize` or `maximize`
- primary metric is supported
- secondary metrics are supported and unique

### 8.5 `micro.candidate`

Purpose: bind params to one selected or generated assignment.

Shape:

```mlir
micro.candidate @candidate_17
    {BM = 32, BN = 64, BK = 64, num_stages = 2}
```

Verifier:

- every param in the referenced search space is bound exactly once
- bound values are members of the corresponding choices list
- candidate name is unique

---

## 9. Token Type

Add a `!micro.async_token` type.

MVP implementation choices:

1. Define it as a custom type in `MicroTypes.td` if the codebase pattern is straightforward.
2. If custom type setup is too much for the first patch, use MLIR `NoneType` only for parse tests and immediately follow with a custom token type task.

Preferred implementation: add the custom token type in M9 because `micro.wait` verifier quality depends on it.

The token type is used by both `micro.async_copy` and `micro.tile_async_copy`.

---

## 10. CMake Integration

Add TableGen setup:

```cmake
set(LLVM_TARGET_DEFINITIONS include/LLK/Dialect/Micro/MicroDialect.td)
mlir_tablegen(include/LLK/Dialect/Micro/MicroDialect.h.inc -gen-dialect-decls -dialect=micro)
mlir_tablegen(include/LLK/Dialect/Micro/MicroDialect.cpp.inc -gen-dialect-defs -dialect=micro)
add_public_tablegen_target(MicroDialectGen)

set(LLVM_TARGET_DEFINITIONS include/LLK/Dialect/Micro/MicroOps.td)
mlir_tablegen(include/LLK/Dialect/Micro/MicroOps.h.inc -gen-op-decls
  EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include/LLK/Dialect/Micro)
mlir_tablegen(include/LLK/Dialect/Micro/MicroOps.cpp.inc -gen-op-defs
  EXTRA_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include/LLK/Dialect/Micro)
add_public_tablegen_target(MicroOpsGen)
```

Add dialect library:

```cmake
add_mlir_dialect_library(MicroDialect
    lib/Dialect/Micro/MicroDialect.cpp
    lib/Dialect/Micro/MicroOps.cpp
    DEPENDS MicroDialectGen MicroOpsGen
    LINK_LIBS PUBLIC MLIRIR MLIRDialect MLIRSupport MLIRFuncDialect
)
```

Link `MicroDialect` into `llk-opt`.

---

## 11. `llk-opt` Registration

Modify `tools/llk-opt/llk-opt.cpp`:

```cpp
#include "LLK/Dialect/Micro/MicroDialect.h"

registry.insert<mlir::micro::MicroDialect>();
```

No pass registration is required for initial parse/print tests.

---

## 12. FileCheck Tests

### 12.1 `test/Dialect/Micro/ops.mlir`

Must include:

- `micro.kernel`
- nested `micro.spatial_for`
- `micro.pipeline`
- `micro.tile_view`
- `micro.tile_partition`
- `micro.alloc`
- `micro.async_copy`
- `micro.wait`
- `micro.mma`
- `micro.vector`
- `micro.reduce`
- `micro.store`

Run:

```bash
build/llk-opt test/Dialect/Micro/ops.mlir | FileCheck test/Dialect/Micro/ops.mlir
```

Expected:

- IR round-trips
- key attributes are preserved

### 12.2 `test/Dialect/Micro/ops_invalid.mlir`

Use `--verify-diagnostics`.

Cases:

- `micro.wait` with non-token operand
- `micro.pipeline stages = 0`
- `micro.mma` shape with two dimensions
- `micro.alloc` in `dram`
- search op inside `micro.kernel`

### 12.2a `test/Dialect/Micro/tile_ops.mlir`

Must include:

- `!micro.tile` type examples
- `#micro.layout` attributes
- `#micro.owner` attributes
- logical tile view
- tile allocation
- tile async copy returning a token
- tile partition into an instruction fragment
- tile MMA/vector/reduce/store examples

### 12.2b `test/Dialect/Micro/tile_ops_invalid.mlir`

Use `--verify-diagnostics`.

Cases:

- tile with invalid layout parameter
- tile allocation missing memory space
- execution tile missing owner
- tile partition with incompatible fragment shape and no mask metadata
- tile async copy with identical source/destination memory space
- tile MMA with unsupported fragment metadata

### 12.3 `test/Dialect/Micro/search_space.mlir`

Must include:

- params
- supported constraints
- objective
- candidate

### 12.4 `test/Dialect/Micro/search_space_invalid.mlir`

Cases:

- duplicate param names
- empty choices
- candidate value outside choices
- unsupported objective metric
- unknown constraint name

---

## 13. Verification Strategy

MVP verification commands:

```bash
cmake --build build --target llk-opt
build/llk-opt test/Dialect/Micro/ops.mlir | FileCheck test/Dialect/Micro/ops.mlir
build/llk-opt --verify-diagnostics --split-input-file test/Dialect/Micro/ops_invalid.mlir
build/llk-opt test/Dialect/Micro/tile_ops.mlir | FileCheck test/Dialect/Micro/tile_ops.mlir
build/llk-opt --verify-diagnostics --split-input-file test/Dialect/Micro/tile_ops_invalid.mlir
build/llk-opt test/Dialect/Micro/search_space.mlir | FileCheck test/Dialect/Micro/search_space.mlir
build/llk-opt --verify-diagnostics --split-input-file test/Dialect/Micro/search_space_invalid.mlir
```

The first implementation plan should add these tests before implementation code.

---

## 14. Implementation Order

1. Add dialect skeleton and CMake integration.
2. Add memory, dtype, layout, and owner attributes.
3. Add `!micro.async_token`.
4. Add `!micro.tile` or the explicitly scoped tensor-plus-tile-attributes bridge.
5. Add tile construction and materialization ops with minimal verifiers.
6. Add concrete execution ops with tile-compatible verifiers.
7. Add concrete and tile FileCheck tests.
8. Add search-space ops and verifiers.
9. Add search-space FileCheck tests.
10. Register dialect in `llk-opt`.
11. Add docs links from architecture/spec docs if needed.

---

## 15. Acceptance Criteria

- `micro` dialect builds as `MicroDialect`.
- `!micro.tile`, `#micro.layout`, and `#micro.owner` parse, print, and verify, or the temporary tensor-plus-attributes bridge is documented in tests.
- `llk-opt` can parse and print concrete `micro.kernel` examples.
- `llk-opt` can parse and print tile-centric `micro.kernel` examples.
- `llk-opt` can parse and print `micro.search_space` examples.
- verifiers reject the invalid cases listed above.
- all new tests are registered in CMake.
- existing LLK dialect tests continue to pass.
