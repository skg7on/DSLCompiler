//===- LegacyValidation.h - legacy contracts ------------------------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Experimental generic-op schemas, shared by every legacy pass entry point.
#ifndef LLK_CONVERSION_TRITONTOLLK_LEGACYVALIDATION_H
#define LLK_CONVERSION_TRITONTOLLK_LEGACYVALIDATION_H
#include "mlir/IR/BuiltinOps.h"
namespace mlir::llk {
LogicalResult validateLegacyTriton(ModuleOp module,
                                   bool rejectResidual = false);
}
#endif
