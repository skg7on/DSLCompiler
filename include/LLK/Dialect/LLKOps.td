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
    let hasCustomAssemblyFormat = 1;

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

#endif
