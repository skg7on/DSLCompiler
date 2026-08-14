# Micro-IR Tile Type and Attributes (M9a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the tile-centric type and attributes to the `micro` dialect: `#micro.owner` (owner enum), `#micro.layout` (layout attribute with parameters), and the first-class `!micro.tile` type, with parse/print/verify and FileCheck coverage.

**Architecture:** The dialect already has the skeleton, `#micro.memory`, `#micro.map`, `#micro.dtype`, and `!micro.async_token` (PR #55). This plan adds the two remaining attributes and the canonical tile type. `#micro.owner` follows the existing `I32EnumAttr` pattern; `#micro.layout` and `!micro.tile` need custom `parse`/`print` because their assembly syntax (`#micro.layout<blocked, block = [16,16]>`, `!micro.tile<32x64xbf16, layout = …, memory = …, owner = …>`) is not expressible declaratively.

**Tech Stack:** C++20, MLIR TableGen (`-gen-attrdef-*`, `-gen-typedef-*`), FileCheck, Ninja.

## Global Constraints

- **Worktree isolation (HARD GATE):** all writes happen in `.claude/worktrees/docs-micro-ir-concrete-execution-ops/` (branch `docs/micro-ir-concrete-execution-ops`).
- **LLVM version compat:** all APIs used (`I32EnumAttr`, `AttrDef`, `TypeDef`, `AsmParser::parseKeyword`/`parseInteger`/`parseType`/`parseAttribute`, `ArrayRefParameter`) are stable LLVM 22–24. No `#if LLVM_VERSION_MAJOR` guards needed.
- **Preferred API (mandatory, `-Werror=deprecated-declarations`):** `mlir::cast<T>(x)`/`isa<T>(x)`/`dyn_cast<T>(x)`; never `x.cast<T>()`/`x.isa<T>()`.
- **CMake:** no tablegen CMake changes needed — `MicroDialect.td` already feeds `-gen-attrdef-*` and `MicroTypes.td` feeds `-gen-typedef-*`. Only the new FileCheck tests need `add_llk_filecheck_test`/`add_test` entries (Task 4).
- **Build:** `ninja -C build llk-opt`, then the FileCheck commands below.

## File Structure

| File | Responsibility |
|------|----------------|
| `include/LLK/Dialect/Micro/MicroEnums.h` | add `Owner` and `LayoutKind` enums + stringify/symbolize |
| `include/LLK/Dialect/Micro/MicroDialect.td` | add `#micro.owner` EnumAttr + `#micro.layout` AttrDef |
| `include/LLK/Dialect/Micro/MicroTypes.td` | add `!micro.tile` TypeDef |
| `lib/Dialect/Micro/MicroDialect.cpp` | `LayoutAttr` parse/print/verify + `TileType` parse/print/verify |
| `test/Dialect/Micro/tile_ops.mlir` | new round-trip tests for attrs + type |
| `test/Dialect/Micro/tile_ops_invalid.mlir` | new verify-diagnostics tests |
| `CMakeLists.txt` | register the two new FileCheck tests |

> Note: `LayoutAttr`/`TileType` parse/print/verify live in `MicroDialect.cpp` (not `MicroOps.cpp`) because they are attributes/types, not operations. That file already includes the generated attribute/type `.inc` files.

## Design Decisions

1. **`#micro.owner`** is a plain 12-value `I32EnumAttr` (`cluster, core, warp, wave, subgroup, pe_group, pe, lane, worker, matrix_engine, vector_engine, dma`), identical in structure to `#micro.memory`. It is distinct from `#micro.map` (which is the dotted `spatial_for` mapping hierarchy).
2. **`#micro.layout`** is a custom `AttrDef` holding a `LayoutKind` enum plus optional params (`block`, `vector`, `align`, `banks`, `swizzle`). Absent params are represented as null/zero and simply not printed.
3. **`!micro.tile`** is a first-class `TypeDef` with parameters `(shape, elementType, layout, memory, owner)`. `layout`/`memory`/`owner` are nullable parameters — null means "absent" (layout defaults to `row_major` at access time; memory/owner are required only after materialization / for execution tiles). Custom `parse`/`print` produces the `!micro.tile<32x64xbf16, layout = …, memory = …, owner = …>` spelling.

---

### Task 1: `#micro.owner` enum attribute

**Files:**
- Modify: `include/LLK/Dialect/Micro/MicroEnums.h`
- Modify: `include/LLK/Dialect/Micro/MicroDialect.td`
- Modify: `test/Dialect/Micro/tile_ops.mlir` (create)

**Interfaces:**
- Produces: `micro::Owner` enum, `stringifyOwner`/`symbolizeOwner`, and `Micro_OwnerAttr` (`getValue()` → `Owner`).

- [ ] **Step 1: Add the `Owner` enum + stringify/symbolize to `MicroEnums.h`**

Append before the closing `} // namespace mlir::micro`:

```cpp
//===----------------------------------------------------------------------===//
// Owner enum
//===----------------------------------------------------------------------===//

enum Owner : uint32_t {
  cluster = 0,
  core = 1,
  warp = 2,
  wave = 3,
  subgroup = 4,
  pe_group = 5,
  pe = 6,
  lane = 7,
  worker = 8,
  matrix_engine = 9,
  vector_engine = 10,
  dma = 11,
};

inline llvm::StringRef stringifyOwner(Owner val) {
  switch (val) {
  case cluster: return "cluster";
  case core: return "core";
  case warp: return "warp";
  case wave: return "wave";
  case subgroup: return "subgroup";
  case pe_group: return "pe_group";
  case pe: return "pe";
  case lane: return "lane";
  case worker: return "worker";
  case matrix_engine: return "matrix_engine";
  case vector_engine: return "vector_engine";
  case dma: return "dma";
  }
  return "";
}

inline std::optional<Owner> symbolizeOwner(llvm::StringRef str) {
  return llvm::StringSwitch<std::optional<Owner>>(str)
      .Case("cluster", cluster)
      .Case("core", core)
      .Case("warp", warp)
      .Case("wave", wave)
      .Case("subgroup", subgroup)
      .Case("pe_group", pe_group)
      .Case("pe", pe)
      .Case("lane", lane)
      .Case("worker", worker)
      .Case("matrix_engine", matrix_engine)
      .Case("vector_engine", vector_engine)
      .Case("dma", dma)
      .Default(std::nullopt);
}
```

- [ ] **Step 2: Add the `#micro.owner` EnumAttr to `MicroDialect.td`**

Append after the existing `Micro_DTypeAttr` block (before `#endif`):

```tablegen
//===----------------------------------------------------------------------===//
// Owner enum attribute
//===----------------------------------------------------------------------===//

def Micro_OwnerCluster : I32EnumAttrCase<"cluster", 0, "cluster">;
def Micro_OwnerCore : I32EnumAttrCase<"core", 1, "core">;
def Micro_OwnerWarp : I32EnumAttrCase<"warp", 2, "warp">;
def Micro_OwnerWave : I32EnumAttrCase<"wave", 3, "wave">;
def Micro_OwnerSubgroup : I32EnumAttrCase<"subgroup", 4, "subgroup">;
def Micro_OwnerPEGroup : I32EnumAttrCase<"pe_group", 5, "pe_group">;
def Micro_OwnerPE : I32EnumAttrCase<"pe", 6, "pe">;
def Micro_OwnerLane : I32EnumAttrCase<"lane", 7, "lane">;
def Micro_OwnerWorker : I32EnumAttrCase<"worker", 8, "worker">;
def Micro_OwnerMatrixEngine : I32EnumAttrCase<"matrix_engine", 9, "matrix_engine">;
def Micro_OwnerVectorEngine : I32EnumAttrCase<"vector_engine", 10, "vector_engine">;
def Micro_OwnerDMA : I32EnumAttrCase<"dma", 11, "dma">;

def Micro_Owner : I32EnumAttr<"Owner", "Owner resource scope", [
    Micro_OwnerCluster, Micro_OwnerCore, Micro_OwnerWarp, Micro_OwnerWave,
    Micro_OwnerSubgroup, Micro_OwnerPEGroup, Micro_OwnerPE, Micro_OwnerLane,
    Micro_OwnerWorker, Micro_OwnerMatrixEngine, Micro_OwnerVectorEngine,
    Micro_OwnerDMA
]> {
    let cppNamespace = "::mlir::micro";
    let genSpecializedAttr = 0;
}

def Micro_OwnerAttr :
    EnumAttr<Micro_Dialect, Micro_Owner, "owner"> {
    let assemblyFormat = "`<` $value `>`";
}
```

- [ ] **Step 3: Add the failing round-trip test**

Create `test/Dialect/Micro/tile_ops.mlir`:

```mlir
// RUN: llk-opt %s | llk-opt | FileCheck %s

// Test that #micro.owner parses and round-trips.
// CHECK-LABEL: func.func @test_owner_attr
func.func @test_owner_attr() attributes {
  owners = [
    #micro.owner<cluster>, #micro.owner<core>, #micro.owner<warp>,
    #micro.owner<wave>, #micro.owner<subgroup>, #micro.owner<pe_group>,
    #micro.owner<pe>, #micro.owner<lane>, #micro.owner<worker>,
    #micro.owner<matrix_engine>, #micro.owner<vector_engine>, #micro.owner<dma>
  ]
} {
  // CHECK: owners = [#micro.owner<cluster>, #micro.owner<core>, #micro.owner<warp>, #micro.owner<wave>, #micro.owner<subgroup>, #micro.owner<pe_group>, #micro.owner<pe>, #micro.owner<lane>, #micro.owner<worker>, #micro.owner<matrix_engine>, #micro.owner<vector_engine>, #micro.owner<dma>]
  return
}
```

- [ ] **Step 4: Run the test to verify it fails** (expect `#micro.owner` unknown).

Run: `ninja -C build llk-opt && build/llk-opt test/Dialect/Micro/tile_ops.mlir 2>&1 | head`

- [ ] **Step 5: Build and run to verify it passes**

Run: `build/llk-opt test/Dialect/Micro/tile_ops.mlir | build/llk-opt | FileCheck test/Dialect/Micro/tile_ops.mlir`

- [ ] **Step 6: Commit**

```bash
git add include/LLK/Dialect/Micro/MicroEnums.h include/LLK/Dialect/Micro/MicroDialect.td test/Dialect/Micro/tile_ops.mlir
git commit -m "feat(micro): add #micro.owner enum attribute"
```

---

### Task 2: `#micro.layout` attribute with parameters

**Files:**
- Modify: `include/LLK/Dialect/Micro/MicroEnums.h` (add `LayoutKind`)
- Modify: `include/LLK/Dialect/Micro/MicroDialect.td` (add `Micro_LayoutAttr` AttrDef)
- Modify: `lib/Dialect/Micro/MicroDialect.cpp` (parse/print/verify)
- Modify: `test/Dialect/Micro/tile_ops.mlir`

**Interfaces:**
- Produces: `micro::LayoutKind` enum, `stringifyLayoutKind`/`symbolizeLayoutKind`, and `micro::LayoutAttr` with accessors `getKind()` → `LayoutKind`, `getBlock()` → `DenseI64ArrayAttr`, `getVector()`/`getAlign()`/`getBanks()` → `int64_t`, `getSwizzle()` → `StringAttr`.

- [ ] **Step 1: Add the `LayoutKind` enum + stringify/symbolize to `MicroEnums.h`**

```cpp
//===----------------------------------------------------------------------===//
// Layout kind enum
//===----------------------------------------------------------------------===//

enum LayoutKind : uint32_t {
  row_major = 0,
  col_major = 1,
  blocked = 2,
  vectorized = 3,
  swizzled = 4,
};

inline llvm::StringRef stringifyLayoutKind(LayoutKind val) {
  switch (val) {
  case row_major: return "row_major";
  case col_major: return "col_major";
  case blocked: return "blocked";
  case vectorized: return "vectorized";
  case swizzled: return "swizzled";
  }
  return "";
}

inline std::optional<LayoutKind> symbolizeLayoutKind(llvm::StringRef str) {
  return llvm::StringSwitch<std::optional<LayoutKind>>(str)
      .Case("row_major", row_major)
      .Case("col_major", col_major)
      .Case("blocked", blocked)
      .Case("vectorized", vectorized)
      .Case("swizzled", swizzled)
      .Default(std::nullopt);
}
```

- [ ] **Step 2: Add the `Micro_LayoutAttr` AttrDef to `MicroDialect.td`**

Append after the `Micro_OwnerAttr` block:

```tablegen
//===----------------------------------------------------------------------===//
// Layout attribute
//===----------------------------------------------------------------------===//

def Micro_LayoutAttr : AttrDef<Micro_Dialect, "Layout"> {
    let mnemonic = "layout";
    let summary = "Memory layout with optional vectorization/blocking params";

    let parameters = (ins
        "LayoutKind":$kind,
        "::mlir::DenseI64ArrayAttr":$block,
        "int64_t":$vector,
        "int64_t":$align,
        "int64_t":$banks,
        "::mlir::StringAttr":$swizzle
    );

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
      static ::mlir::Attribute parse(::mlir::AsmParser &parser,
                                     ::mlir::Type odsType);
      void print(::mlir::AsmPrinter &p) const;
    }];
}
```

- [ ] **Step 3: Add the failing round-trip test**

Append to `test/Dialect/Micro/tile_ops.mlir`:

```mlir
// CHECK-LABEL: func.func @test_layout_attr
func.func @test_layout_attr() attributes {
  layouts = [
    #micro.layout<row_major>,
    #micro.layout<col_major>,
    #micro.layout<blocked, block = [16, 16]>,
    #micro.layout<vectorized, vector = 8, align = 32>,
    #micro.layout<swizzled, swizzle = "xor", banks = 32>
  ]
} {
  // CHECK: layouts = [#micro.layout<row_major>, #micro.layout<col_major>, #micro.layout<blocked, block = [16, 16]>, #micro.layout<vectorized, vector = 8, align = 32>, #micro.layout<swizzled, swizzle = "xor", banks = 32>]
  return
}
```

- [ ] **Step 4: Run the test to verify it fails** (expect `#micro.layout` unknown).

- [ ] **Step 5: Implement parse/print/verify in `MicroDialect.cpp`**

Add these includes (after the existing generated `.inc` includes):

```cpp
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/StringExtras.h"
```

Append at the end of `MicroDialect.cpp` (before `mlir::micro::MicroDialect::initialize()`), and add a `using namespace mlir;` if not already present:

```cpp
//===----------------------------------------------------------------------===//
// LayoutAttr: parse/print/verify
//===----------------------------------------------------------------------===//

Attribute LayoutAttr::parse(AsmParser &parser, Type odsType) {
  // Parse the required kind keyword.
  StringRef kindStr;
  if (parser.parseKeyword(&kindStr))
    return {};
  auto kind = symbolizeLayoutKind(kindStr);
  if (!kind) {
    parser.emitError(parser.getCurrentLocation(), "unknown layout kind '")
        << kindStr << "'";
    return {};
  }

  DenseI64ArrayAttr block;
  int64_t vector = 0, align = 0, banks = 0;
  StringAttr swizzle;

  // Parse optional ", key = value" pairs.
  while (succeeded(parser.parseOptionalComma())) {
    StringRef key;
    if (parser.parseKeyword(&key))
      return {};
    if (key == "block") {
      if (parser.parseEqual() || parser.parseAttribute(block))
        return {};
    } else if (key == "vector") {
      if (parser.parseEqual() || parser.parseInteger(vector))
        return {};
    } else if (key == "align") {
      if (parser.parseEqual() || parser.parseInteger(align))
        return {};
    } else if (key == "banks") {
      if (parser.parseEqual() || parser.parseInteger(banks))
        return {};
    } else if (key == "swizzle") {
      if (parser.parseEqual() || parser.parseAttribute(swizzle))
        return {};
    } else {
      parser.emitError(parser.getCurrentLocation(), "unknown layout parameter '")
          << key << "'";
      return {};
    }
  }

  return LayoutAttr::get(parser.getContext(), *kind, block, vector, align,
                         banks, swizzle);
}

void LayoutAttr::print(AsmPrinter &p) const {
  p << "<" << stringifyLayoutKind(getKind());
  if (getBlock())
    p << ", block = " << getBlock();
  if (getVector() > 0)
    p << ", vector = " << getVector();
  if (getAlign() > 0)
    p << ", align = " << getAlign();
  if (getBanks() > 0)
    p << ", banks = " << getBanks();
  if (getSwizzle())
    p << ", swizzle = " << getSwizzle();
  p << ">";
}

LogicalResult LayoutAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    LayoutKind kind, DenseI64ArrayAttr block, int64_t vector, int64_t align,
    int64_t banks, StringAttr swizzle) {
  if (block && block.getSize() > 0) {
    for (int64_t d : block.asArrayRef())
      if (d <= 0)
        return emitError() << "layout block extents must be positive";
  }
  if (vector < 0)
    return emitError() << "layout vector width must be positive";
  if (align < 0 || (align != 0 && (align & (align - 1)) != 0))
    return emitError() << "layout align must be a positive power of two";
  if (banks < 0)
    return emitError() << "layout banks must be positive";
  if (swizzle && kind != LayoutKind::swizzled)
    return emitError() << "swizzle parameter requires 'swizzled' layout kind";
  return success();
}
```

- [ ] **Step 6: Build and run to verify it passes**

Run: `ninja -C build llk-opt && build/llk-opt test/Dialect/Micro/tile_ops.mlir | build/llk-opt | FileCheck test/Dialect/Micro/tile_ops.mlir`

> If `LayoutAttr::get(...)`'s parameter order differs from the tablegen `parameters` declaration, the compiler will flag it — align the call order with the `parameters` list. If `parser.parseAttribute(block)` cannot bind to `DenseI64ArrayAttr`, bind to a generic `Attribute` and use `dyn_cast<DenseI64ArrayAttr>`.

- [ ] **Step 7: Add the invalid layout test** (append to `tile_ops_invalid.mlir`, created here)

```mlir
// RUN: llk-opt --verify-diagnostics --split-input-file %s

// -----

func.func @layout_bad_align() attributes {
  bad = #micro.layout<vectorized, vector = 8, align = 24>
} {
  return
}
// expected-error @-3 {{layout align must be a positive power of two}}

// -----

func.func @layout_swizzle_requires_swizzled() attributes {
  bad = #micro.layout<row_major, swizzle = "xor">
} {
  return
}
// expected-error @-3 {{swizzle parameter requires 'swizzled' layout kind}}
```

- [ ] **Step 8: Commit**

```bash
git add include/LLK/Dialect/Micro/MicroEnums.h include/LLK/Dialect/Micro/MicroDialect.td lib/Dialect/Micro/MicroDialect.cpp test/Dialect/Micro/tile_ops.mlir test/Dialect/Micro/tile_ops_invalid.mlir
git commit -m "feat(micro): add #micro.layout attribute with parameters"
```

---

### Task 3: `!micro.tile` type

**Files:**
- Modify: `include/LLK/Dialect/Micro/MicroTypes.td`
- Modify: `lib/Dialect/Micro/MicroDialect.cpp` (parse/print/verify)
- Modify: `test/Dialect/Micro/tile_ops.mlir`

**Interfaces:**
- Produces: `micro::TileType` with accessors `getShape()` → `ArrayRef<int64_t>`, `getElementType()` → `Type`, `getLayout()` → `LayoutAttr` (nullable), `getMemory()` → `MemorySpaceAttr` (nullable), `getOwner()` → `OwnerAttr` (nullable).

- [ ] **Step 1: Add the `TileType` TypeDef to `MicroTypes.td`**

Append after `Micro_AsyncToken`:

```tablegen
//===----------------------------------------------------------------------===//
// Tile type
//===----------------------------------------------------------------------===//

def Micro_Tile : TypeDef<Micro_Dialect, "Tile"> {
    let mnemonic = "tile";
    let summary = "A canonical tile: shape, dtype, layout, memory, owner";
    let description = [{
        The primary execution value of the micro dialect. A tile carries its
        shape, element type, layout, memory space, and owner resource scope.
        layout, memory, and owner are nullable (absent until materialization
        or mapping).
    }];

    let parameters = (ins
        ArrayRefParameter<"int64_t">:$shape,
        "Type":$elementType,
        "LayoutAttr":$layout,
        "MemorySpaceAttr":$memory,
        "OwnerAttr":$owner
    );

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
      static ::mlir::Type parse(::mlir::AsmParser &parser);
      void print(::mlir::AsmPrinter &p) const;
    }];
}
```

- [ ] **Step 2: Add the failing round-trip test**

Append to `test/Dialect/Micro/tile_ops.mlir`:

```mlir
// CHECK-LABEL: func.func @test_tile_type
func.func @test_tile_type(
    // CHECK: !micro.tile<32x64xbf16>
    %t0 : !micro.tile<32x64xbf16>,
    // CHECK: !micro.tile<32x64xbf16, layout = #micro.layout<row_major, vector = 8>, memory = #micro.memory<sram>, owner = #micro.owner<worker>>
    %t1 : !micro.tile<32x64xbf16, layout = #micro.layout<row_major, vector = 8>, memory = #micro.memory<sram>, owner = #micro.owner<worker>>) {
  return
}
```

- [ ] **Step 3: Run the test to verify it fails** (expect `!micro.tile` unknown).

- [ ] **Step 4: Implement parse/print/verify in `MicroDialect.cpp`**

Append after the `LayoutAttr` code:

```cpp
//===----------------------------------------------------------------------===//
// TileType: parse/print/verify
//===----------------------------------------------------------------------===//

Type TileType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return {};

  // Parse the shape + element type: "32x64xbf16" (or just "bf16").
  SmallVector<int64_t> shape;
  Type elementType;
  while (true) {
    int64_t dim;
    auto dimResult = parser.parseOptionalInteger(dim);
    if (dimResult.has_value() && succeeded(dimResult.value())) {
      shape.push_back(dim);
      if (succeeded(parser.parseXInDimensionList()))
        continue; // another dimension follows
      // No more dims: parse the element type.
      if (parser.parseType(elementType))
        return {};
      break;
    }
    // No dimension token: element type only.
    if (parser.parseType(elementType))
      return {};
    break;
  }

  LayoutAttr layout;
  MemorySpaceAttr memory;
  OwnerAttr owner;

  // Parse optional ", key = value" pairs.
  while (succeeded(parser.parseOptionalComma())) {
    StringRef key;
    if (parser.parseKeyword(&key))
      return {};
    if (key == "layout") {
      if (parser.parseEqual() || parser.parseAttribute(layout))
        return {};
    } else if (key == "memory") {
      if (parser.parseEqual() || parser.parseAttribute(memory))
        return {};
    } else if (key == "owner") {
      if (parser.parseEqual() || parser.parseAttribute(owner))
        return {};
    } else {
      parser.emitError(parser.getCurrentLocation(), "unknown tile parameter '")
          << key << "'";
      return {};
    }
  }

  if (parser.parseGreater())
    return {};

  return TileType::get(parser.getContext(), shape, elementType, layout,
                       memory, owner);
}

void TileType::print(AsmPrinter &p) const {
  p << "<";
  llvm::interleaveComma(getShape(), p, [&](int64_t d) { p << d; });
  p << "x" << getElementType();
  if (getLayout())
    p << ", layout = " << getLayout();
  if (getMemory())
    p << ", memory = " << getMemory();
  if (getOwner())
    p << ", owner = " << getOwner();
  p << ">";
}

LogicalResult TileType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ArrayRef<int64_t> shape, Type elementType, LayoutAttr layout,
    MemorySpaceAttr memory, OwnerAttr owner) {
  for (int64_t d : shape)
    if (d <= 0)
      return emitError() << "tile dimensions must be positive";
  if (!elementType)
    return emitError() << "tile requires an element type";
  return success();
}
```

> `llvm::interleaveComma` requires `#include "llvm/ADT/STLExtras.h"`. If the `MemorySpaceAttr`/`OwnerAttr`/`LayoutAttr` parameters cannot be bound to `parseAttribute`, bind to generic `Attribute` and `dyn_cast` them; if the null-default for those params is rejected by the generated builder, switch them to `"Attribute"`-typed parameters and expose typed accessors in `extraClassDeclaration`.

- [ ] **Step 5: Build and run to verify it passes**

Run: `ninja -C build llk-opt && build/llk-opt test/Dialect/Micro/tile_ops.mlir | build/llk-opt | FileCheck test/Dialect/Micro/tile_ops.mlir`

- [ ] **Step 6: Add the invalid tile tests** (append to `tile_ops_invalid.mlir`)

```mlir
// -----

func.func @tile_zero_dim() {
  "test.op"() : () -> !micro.tile<0x64xbf16>
  return
}
// expected-error @-2 {{tile dimensions must be positive}}
```

- [ ] **Step 7: Commit**

```bash
git add include/LLK/Dialect/Micro/MicroTypes.td lib/Dialect/Micro/MicroDialect.cpp test/Dialect/Micro/tile_ops.mlir test/Dialect/Micro/tile_ops_invalid.mlir
git commit -m "feat(micro): add first-class !micro.tile type"
```

---

### Task 4: Register the tile FileCheck tests in CMake

**Files:**
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add the two test registrations**

In the `if(FILECHECK_BIN)` block, after the existing `add_llk_filecheck_test(MicroDialectOps ...)` line, add:

```cmake
  add_llk_filecheck_test(MicroTileOps test/Dialect/Micro/tile_ops.mlir)
```

In the `if(LLK_BUILD_TOOLS)` block, after the existing `MicroDialectOpsInvalid` `add_test`, add:

```cmake
add_test(NAME MicroTileOpsInvalid
    COMMAND $<TARGET_FILE:llk-opt> --verify-diagnostics --split-input-file
            ${CMAKE_SOURCE_DIR}/test/Dialect/Micro/tile_ops_invalid.mlir
)
```

- [ ] **Step 2: Build and run the full suite**

Run:
```bash
ninja -C build llk-opt
ninja -C build check-llk
```

Expected: `MicroTileOps` and `MicroTileOpsInvalid` pass, plus all pre-existing tests.

- [ ] **Step 3: Commit**

```bash
git add CMakeLists.txt
git commit -m "test(micro): register tile type and attribute FileCheck tests"
```

---

## Self-Review

**Spec coverage** (dialect spec §6.4–6.6, tile spec §2, §6–7):

| Requirement | Task |
|-------------|------|
| `#micro.owner` 12 values | Task 1 |
| `#micro.layout` 5 kinds + params + verifier (block positive, vector positive, align power-of-2, banks positive, swizzle⇒swizzled) | Task 2 |
| `!micro.tile` shape×dtype + optional layout/memory/owner | Task 3 |
| `tile_ops.mlir` + `tile_ops_invalid.mlir` | Tasks 1–3 (content), Task 4 (registration) |

**Placeholder scan:** none — all steps have concrete code or an exact command.

**Type consistency:** `Owner`/`LayoutKind` enums match the `MicroEnums.h` pattern (plain `enum : uint32_t` + `stringify`/`symbolize`). `LayoutAttr`/`TileType` accessors match the TableGen `parameters` names. `TileType::get(...)` argument order is `(context, shape, elementType, layout, memory, owner)` per the `parameters` declaration.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-13-micro-ir-tile-type-and-attributes.md`. M9b (tile ops + concrete ops) is a separate plan. Two execution options: **1. Subagent-Driven** (recommended) or **2. Inline Execution**. Which approach?
