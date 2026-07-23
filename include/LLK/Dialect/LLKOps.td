#ifndef LLK_OPS_TD
#define LLK_OPS_TD

include "LLKDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/DestinationStyleOpInterface.td"
include "mlir/Interfaces/TilingInterface.td"

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

#endif
