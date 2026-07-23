#ifndef LLK_DIALECT_TD
#define LLK_DIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def LLK_Dialect : Dialect {
    let name = "llk";
    let cppNamespace = "::mlir::llk";
    let summary = "Low-Level Kernel dialect for LLM operations";
    let description = [{
        The LLK dialect represents domain-specific LLM kernel operations
        (SwiGLU, RoPE, Attention) before they are lowered to Linalg.
        It retains semantic information unavailable in generic MLIR.
    }];
    let useDefaultAttributePrinterParser = 1;
    let useDefaultTypePrinterParser = 1;
    let hasConstantMaterializer = 1;
}

// Activation enum
def LLK_SiLU : I32EnumAttrCase<"silu", 0, "silu">;
def LLK_Activation : I32EnumAttr<"Activation", "LLM activation function", [
    LLK_SiLU
]> {
    let cppNamespace = "::mlir::llk";
    let genSpecializedAttr = 0;
}
def LLK_ActivationAttr : EnumAttr<LLK_Dialect, LLK_Activation, "activation"> {
    let assemblyFormat = "`<` $value `>`";
}

// Math mode enum
def LLK_Strict      : I32EnumAttrCase<"strict", 0, "strict">;
def LLK_BoundedFast : I32EnumAttrCase<"bounded_fast", 1, "bounded_fast">;
def LLK_UnsafeFast  : I32EnumAttrCase<"unsafe_fast", 2, "unsafe_fast">;
def LLK_MathMode : I32EnumAttr<"MathMode", "Numerical approximation policy", [
    LLK_Strict, LLK_BoundedFast, LLK_UnsafeFast
]> {
    let cppNamespace = "::mlir::llk";
    let genSpecializedAttr = 0;
}
def LLK_MathModeAttr : EnumAttr<LLK_Dialect, LLK_MathMode, "math_mode"> {
    let assemblyFormat = "`<` $value `>`";
}

// Softmax mode enum
def LLK_OnlineSoftmax : I32EnumAttrCase<"online", 0, "online">;
def LLK_SoftmaxMode : I32EnumAttr<"SoftmaxMode", "Softmax algorithm", [
    LLK_OnlineSoftmax
]> {
    let cppNamespace = "::mlir::llk";
    let genSpecializedAttr = 0;
}
def LLK_SoftmaxModeAttr : EnumAttr<LLK_Dialect, LLK_SoftmaxMode, "softmax_mode"> {
    let assemblyFormat = "`<` $value `>`";
}

#endif
