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
// Mapping target enum attribute
//===----------------------------------------------------------------------===//

def Micro_ClusterX : I32EnumAttrCase<"cluster_x", 0, "cluster_x">;
def Micro_ClusterY : I32EnumAttrCase<"cluster_y", 1, "cluster_y">;
def Micro_CoreX : I32EnumAttrCase<"core_x", 2, "core_x">;
def Micro_CoreY : I32EnumAttrCase<"core_y", 3, "core_y">;
def Micro_PEX : I32EnumAttrCase<"pe_x", 4, "pe_x">;
def Micro_PEY : I32EnumAttrCase<"pe_y", 5, "pe_y">;
def Micro_Lane : I32EnumAttrCase<"lane", 6, "lane">;
def Micro_Worker : I32EnumAttrCase<"worker", 7, "worker">;
def Micro_MatrixEngine : I32EnumAttrCase<"matrix_engine", 8, "matrix_engine">;
def Micro_VectorEngine : I32EnumAttrCase<"vector_engine", 9, "vector_engine">;
def Micro_DMA : I32EnumAttrCase<"dma", 10, "dma">;

def Micro_MappingTarget : I32EnumAttr<"MappingTarget", "Mapping target", [
    Micro_ClusterX,
    Micro_ClusterY,
    Micro_CoreX,
    Micro_CoreY,
    Micro_PEX,
    Micro_PEY,
    Micro_Lane,
    Micro_Worker,
    Micro_MatrixEngine,
    Micro_VectorEngine,
    Micro_DMA
]> {
    let cppNamespace = "::mlir::micro";
    let genSpecializedAttr = 0;
}

def Micro_MappingTargetAttr :
    EnumAttr<Micro_Dialect, Micro_MappingTarget, "map"> {
    let assemblyFormat = "`<` $value `>`";
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

#endif
