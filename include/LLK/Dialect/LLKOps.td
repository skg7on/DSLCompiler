#ifndef LLK_OPS_TD
#define LLK_OPS_TD

include "LLKDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/TilingInterface.td"
include "mlir/IR/SymbolInterfaces.td"

class LLK_Op<string mnemonic, list<Trait> traits = []>
    : Op<LLK_Dialect, mnemonic, traits>;

def LLK_FusedSwiGLUOp : LLK_Op<"fused_swiglu", [
    DeclareOpInterfaceMethods<DestinationStyleOpInterface>,
    DeclareOpInterfaceMethods<TilingInterface>
]> {
    let summary = "Fused SiLU-gated double linear projection";
    let description = [{
        Computes: Y = SiLU(X @ Wg^T) * (X @ Wu^T)

        X:   [M, K] bf16 input
        Wg:  [K, N] bf16 gate projection weight
        Wu:  [K, N] bf16 up projection weight
        Y:   [M, N] bf16 output

        Uses FP32 accumulation internally.
    }];

    let arguments = (ins
        TensorOf<[BF16]>:$x,
        TensorOf<[BF16]>:$wg,
        TensorOf<[BF16]>:$wu,
        TensorOf<[BF16]>:$init,
        AnyAttr:$accumulator_type,
        LLK_ActivationAttr:$activation,
        LLK_MathModeAttr:$math_mode
    );

    let results = (outs
        TensorOf<[BF16]>:$result
    );

    let assemblyFormat = [{
        `ins` `(` $x `,` $wg `,` $wu `:` type($x) `,` type($wg) `,` type($wu) `)`
        `outs` `(` $init `:` type($init) `)`
        attr-dict
        `->` type($result)
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
      // TilingInterface methods (overrides of default implementations).
      ::mlir::SmallVector<::mlir::utils::IteratorType> getLoopIteratorTypes();
      ::mlir::SmallVector<::mlir::Range> getIterationDomain(::mlir::OpBuilder &b);
      ::mlir::FailureOr<::mlir::TilingResult> getTiledImplementation(
          ::mlir::OpBuilder &b,
          ::mlir::ArrayRef<::mlir::OpFoldResult> offsets,
          ::mlir::ArrayRef<::mlir::OpFoldResult> sizes);
      ::llvm::LogicalResult getResultTilePosition(
          ::mlir::OpBuilder &b, unsigned resultNumber,
          ::mlir::ArrayRef<::mlir::OpFoldResult> offsets,
          ::mlir::ArrayRef<::mlir::OpFoldResult> sizes,
          ::mlir::SmallVector<::mlir::OpFoldResult> &resultOffsets,
          ::mlir::SmallVector<::mlir::OpFoldResult> &resultSizes);
    }];
}

def LLK_RoPEOp : LLK_Op<"rope", [
    DeclareOpInterfaceMethods<DestinationStyleOpInterface>
]> {
    let summary = "Rotary Position Embedding";
    let description = [{
        Applies RoPE: rotates pairs of dimensions by position-dependent angles.
        X: [B, H, L, D] f32 input
        position_ids: [L] i64 positions
        theta: base frequency (default 10000.0)
    }];

    let arguments = (ins
        TensorOf<[F32]>:$x,
        TensorOf<[I64]>:$position_ids,
        TensorOf<[F32]>:$init,
        F64Attr:$theta,
        LLK_MathModeAttr:$math_mode
    );
    let results = (outs TensorOf<[F32]>:$result);

    let assemblyFormat = [{
        `ins` `(` $x `,` $position_ids `:` type($x) `,` type($position_ids) `)`
        `outs` `(` $init `:` type($init) `)`
        attr-dict
        `->` type($result)
    }];

    let hasVerifier = 1;
}

def LLK_AttentionOp : LLK_Op<"attention", [
    DeclareOpInterfaceMethods<DestinationStyleOpInterface>
]> {
    let summary = "Simplified scaled dot-product attention with online softmax";
    let description = [{
        Computes: O = softmax(Q @ K^T / sqrt(D) + mask) @ V
        Q: [B, H, Lq, D] f32
        K: [B, H, Lk, D] f32
        V: [B, H, Lk, D] f32
        Returns O: [B, H, Lq, D] f32
        Uses online softmax for numerical stability.
    }];

    let arguments = (ins
        TensorOf<[F32]>:$q,
        TensorOf<[F32]>:$k,
        TensorOf<[F32]>:$v,
        TensorOf<[F32]>:$init,
        F32Attr:$scale,
        DefaultValuedAttr<BoolAttr, "false">:$causal_mask,
        LLK_SoftmaxModeAttr:$softmax_mode,
        LLK_MathModeAttr:$math_mode
    );
    let results = (outs TensorOf<[F32]>:$result);

    let assemblyFormat = [{
        `ins` `(` $q `,` $k `,` $v `:` type($q) `,` type($k) `,` type($v) `)`
        `outs` `(` $init `:` type($init) `)`
        attr-dict
        `->` type($result)
    }];

    let hasVerifier = 1;
}

def LLK_ReturnOp : LLK_Op<"return", [Terminator]> {
    let summary = "Return from a dispatch variant region";
    let arguments = (ins Variadic<AnyType>:$operands);
}

def LLK_AssumeOp : LLK_Op<"assume", [
    DeclareOpInterfaceMethods<DestinationStyleOpInterface>
]> {
    let summary = "Divisibility and alignment hints for downstream passes";
    let description = [{
        Annotates tensors with constraints that the compiler can exploit.
        Does NOT change program semantics — a verifier check that would
        abort if violated.

        Constraints:
        - divisible(dim, factor): dimension is divisible by factor
        - aligned(ptr_bytes): underlying buffer is aligned to ptr_bytes boundary

        Example:
          %x_assumed = llk.assume ins(%x : tensor<MxKxbf16>)
              outs(%init : tensor<MxKxbf16>)
              {divisible_dims = [0, 1], divisible_factors = [8, 64],
               alignment_bytes = 64}
              -> tensor<MxKxbf16>
    }];

    let arguments = (ins
        AnyTensor:$input,
        AnyTensor:$init,
        DenseI64ArrayAttr:$divisible_dims,
        DenseI64ArrayAttr:$divisible_factors,
        DefaultValuedAttr<I64Attr, "0">:$alignment_bytes
    );
    let results = (outs AnyTensor:$result);

    let assemblyFormat = [{
        `ins` `(` $input `:` type($input) `)`
        `outs` `(` $init `:` type($init) `)`
        attr-dict
        `->` type($result)
    }];

    let hasVerifier = 1;
}

def LLK_DispatchOp : LLK_Op<"dispatch", [
    DeclareOpInterfaceMethods<SymbolUserOpInterface>
]> {
    let summary = "Multi-variant dispatch guard for shape specialization";
    let description = [{
        Dispatches to one of several variant regions based on runtime
        shape checks. The last region is the fallback (otherwise clause).

        Each variant has a condition written as an arith dialect predicate
        over symbolic dims M, N, K.

        Example:
          llk.dispatch { M = %m, N = %n, K = %k }
          variants {
              @variant1 when(%m == 1),
              @variant2 when(%m <= 4),
              @generic otherwise
          }
    }];

    let arguments = (ins
        Variadic<AnyType>:$inputs,
        I64Attr:$M,
        I64Attr:$N,
        I64Attr:$K
    );
    let results = (outs Variadic<AnyType>:$results);
    let regions = (region VariadicRegion<SizedRegion<1>>:$variants);
    let hasVerifier = 1;
}

def LLK_MatmulOp : LLK_Op<"matmul", [
    DeclareOpInterfaceMethods<DestinationStyleOpInterface>,
    DeclareOpInterfaceMethods<TilingInterface>
]> {
    let summary = "Generic batched matmul fallback";
    let description = [{
        Generic matrix multiplication — fallback when no fusion pattern
        matches. Lowered identically to linalg.matmul by LLKToLinalg.
    }];

    let arguments = (ins
        TensorOf<[BF16, F16, F32]>:$a,
        TensorOf<[BF16, F16, F32]>:$b,
        TensorOf<[BF16, F16, F32]>:$init,
        AnyAttr:$accumulator_type,
        LLK_MathModeAttr:$math_mode
    );

    let results = (outs
        TensorOf<[BF16, F16, F32]>:$result
    );

    let assemblyFormat = [{
        `ins` `(` $a `,` $b `:` type($a) `,` type($b) `)`
        `outs` `(` $init `:` type($init) `)`
        attr-dict
        `->` type($result)
    }];

    let hasVerifier = 1;

    let extraClassDeclaration = [{
      ::mlir::SmallVector<::mlir::utils::IteratorType> getLoopIteratorTypes();
      ::mlir::SmallVector<::mlir::Range> getIterationDomain(::mlir::OpBuilder &b);
      ::mlir::FailureOr<::mlir::TilingResult> getTiledImplementation(
          ::mlir::OpBuilder &b,
          ::mlir::ArrayRef<::mlir::OpFoldResult> offsets,
          ::mlir::ArrayRef<::mlir::OpFoldResult> sizes);
      ::llvm::LogicalResult getResultTilePosition(
          ::mlir::OpBuilder &b, unsigned resultNumber,
          ::mlir::ArrayRef<::mlir::OpFoldResult> offsets,
          ::mlir::ArrayRef<::mlir::OpFoldResult> sizes,
          ::mlir::SmallVector<::mlir::OpFoldResult> &resultOffsets,
          ::mlir::SmallVector<::mlir::OpFoldResult> &resultSizes);
    }];
}

def LLK_MakeTensorOp : LLK_Op<"make_tensor", [
    DeclareOpInterfaceMethods<DestinationStyleOpInterface>
]> {
    let summary = "Wrap raw pointer as tensor for pipeline entry";
    let description = [{
        Bridges Triton's pointer-based ABI to MLIR's tensor pipeline.
        Eliminated during bufferization — the tensor becomes a memref
        backed by the original raw pointer.
    }];

    let arguments = (ins
        AnyType:$base_ptr,
        I64Attr:$dim0,
        I64Attr:$dim1,
        TypeAttr:$dtype
    );

    let results = (outs AnyTensor:$result);

    let assemblyFormat = [{
        `from` `(` $base_ptr `:` type($base_ptr) `)`
        `shape` `(` $dim0 `,` $dim1 `)`
        `dtype` `(` $dtype `)`
        attr-dict
        `->` type($result)
    }];

    let hasVerifier = 1;
}

#endif
