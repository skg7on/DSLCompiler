//===- PackWeights.h - Weight packing annotation pass -----------*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Annotates weight operands in Linalg matmul/generic ops with packed layout
// attributes. The downstream TileAndVectorize pass reads these attributes to
// choose the correct interpretation when tiling/vectorizing the K dimension.
//
// The pass walks the module looking for:
//   - linalg::MatmulOp: annotates the RHS (weight) operand
//   - linalg::GenericOp: annotates operands whose indexing maps carry a
//     reduction dimension (the K dimension of the contraction)
//
// Weights identified as function arguments or tensor::EmptyOp producers are
// annotated with "llk.packed_layout" = block_size.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_TRANSFORMS_PACKWEIGHTS_H
#define LLK_TRANSFORMS_PACKWEIGHTS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace llk {
std::unique_ptr<Pass> createPackWeightsPass();
} // namespace llk
} // namespace mlir

#endif
