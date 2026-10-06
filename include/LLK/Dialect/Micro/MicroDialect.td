#ifndef MICRO_DIALECT_TD
#define MICRO_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def Micro_Dialect : Dialect {
    let name = "micro";
    let cppNamespace = "::mlir::micro";
    let summary = "Canonical DNN execution micro-IR";
    let description = [{
        The micro dialect represents scheduled tensor execution in terms of
        compute engines, memory hierarchy, data movement, mapping, pipeline,
        synchronization, and search-space metadata.
    }];
    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Memory space enum attribute
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Mapping target attribute
//===----------------------------------------------------------------------===//

// Where a spatial loop's iterations *may* be placed is, like a tile's owner, a
// **symbol** and not an enumeration. The canonical dialect knows that a spatial
// axis has a name the owning target resolves -- a thread-pool worker, a SIMD
// unit, a systolic array, an x/y coordinate axis -- and nothing about which
// names any particular machine uses. A profile that distributes its iterations
// over its own executor names says so in its own machine model, which maps those
// labels onto the abstract owner classes; naming them here would put one
// backend's spatial vocabulary in the canonical IR, the same boundary §13.4
// draws for layouts and §5.4 for policy.
//
// Keeping the attribute open also keeps legacy text parseable: a spatial loop
// written with a target's spelling still reads as *unresolved analysis data*,
// and only target execution requires it to be normalized.
def Micro_MappingTargetAttr : AttrDef<Micro_Dialect, "MappingTarget"> {
    let mnemonic = "map";
    let summary = "The resource a spatial loop is placed on, as a target-neutral symbol";
    let description = [{
        Names the spatial axis a loop's iterations may be distributed over. The
        spelling is a symbol the owning target resolves: the dialect verifies
        that it is present and well-formed, never that it is one of a fixed set.
    }];

    let parameters = (ins "::mlir::StringAttr":$symbol);

    // A bare identifier (`#micro.owner<worker>`), not a quoted string: the
    // spelling is a symbol the target resolves, and legacy text already writes
    // it unquoted. A quoted string is also accepted, so a spelling that is not
    // a bare identifier (a coordinate axis such as `cluster.x`) stays writable.
    let hasCustomAssemblyFormat = 1;

    let genVerifyDecl = 1;

    let builders = [
        AttrBuilder<(ins "::llvm::StringRef":$symbol), [{
          return $_get($_ctxt, ::mlir::StringAttr::get($_ctxt, symbol));
        }]>
    ];
}

//===----------------------------------------------------------------------===//
// DType enum attribute
//===----------------------------------------------------------------------===//

def Micro_F32 : I32EnumAttrCase<"f32", 0, "f32">;
def Micro_F16 : I32EnumAttrCase<"f16", 1, "f16">;
def Micro_BF16 : I32EnumAttrCase<"bf16", 2, "bf16">;
def Micro_I32 : I32EnumAttrCase<"i32", 3, "i32">;
def Micro_I8 : I32EnumAttrCase<"i8", 4, "i8">;

def Micro_DType : I32EnumAttr<"DType", "Data type for machine modeling", [
    Micro_F32,
    Micro_F16,
    Micro_BF16,
    Micro_I32,
    Micro_I8
]> {
    let cppNamespace = "::mlir::micro";
    let genSpecializedAttr = 0;
}

def Micro_DTypeAttr :
    EnumAttr<Micro_Dialect, Micro_DType, "dtype"> {
    let assemblyFormat = "`<` $value `>`";
}

//===----------------------------------------------------------------------===//
// Owner attribute
//===----------------------------------------------------------------------===//

// A tile's owner is a **symbol**, not an enumeration. The canonical dialect
// knows the abstract classes a tile can belong to -- a group of workers, one
// worker, the vector engine, the matrix engine, the movement engine -- and
// nothing about how any particular machine spells them. A profile that calls
// its finest execution unit and its collaborating set whatever words it likes
// says so in its own machine model, which maps those labels onto the abstract
// classes; naming them here would put one backend's hierarchy in the canonical
// IR, which is the same boundary §13.4 draws for layouts and §5.4 for target
// policy.
//
// Keeping the attribute open also keeps legacy text parseable: a kernel written
// with a target's spelling still reads as *unresolved analysis data*, and only
// target execution requires it to be normalized.
def Micro_OwnerAttr : AttrDef<Micro_Dialect, "Owner"> {
    let mnemonic = "owner";
    let summary = "The owner a tile belongs to, as a target-neutral symbol";
    let description = [{
        Names the resource scope a tile is owned by. The spelling is a symbol
        the owning target resolves: the dialect verifies that it is present and
        well-formed, never that it is one of a fixed set.
    }];

    let parameters = (ins "::mlir::StringAttr":$symbol);

    // A bare identifier (`#micro.owner<worker>`), not a quoted string: the
    // spelling is a symbol the target resolves, and legacy text already writes
    // it unquoted. A quoted string is also accepted, so a spelling that is not
    // a bare identifier (a coordinate axis such as `cluster.x`) stays writable.
    let hasCustomAssemblyFormat = 1;

    let genVerifyDecl = 1;

    let builders = [
        AttrBuilder<(ins "::llvm::StringRef":$symbol), [{
          return $_get($_ctxt, ::mlir::StringAttr::get($_ctxt, symbol));
        }]>
    ];
}

//===----------------------------------------------------------------------===//
// Layout attribute (kind + optional parameters)
//===----------------------------------------------------------------------===//

def Micro_LayoutAttr : AttrDef<Micro_Dialect, "Layout"> {
    let mnemonic = "layout";
    let summary = "Tile memory layout kind with optional parameters";
    let description = [{
        Describes the memory layout of a tile: order (row/column/blocked/
        vectorized/swizzled) plus optional block shape, vector width,
        alignment, banking, and swizzle parameters.
    }];

    let parameters = (ins
        "uint32_t":$kind,
        "DenseI64ArrayAttr":$block,
        "int64_t":$vector,
        "int64_t":$align,
        "int64_t":$banks,
        "StringAttr":$swizzle
    );

    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;
}

#endif
