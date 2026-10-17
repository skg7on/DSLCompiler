//===- ElementwiseSourceAdapter.h - Bounded elementwise source family -----===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The first structured non-contraction source family (framework B5). It
// recognizes `linalg.generic` structurally and then admits exactly one bounded
// shape of program: static rank-two f32 tensors, two read inputs, a destination
// that carries no incoming data, one `arith.addf`/`arith.mulf` yielding its
// result, identity indexing maps and two parallel iterators.
//
// The bound is deliberate. Arbitrary Linalg import, broadcasting, dynamic
// shapes, reductions and approximation flags are separate capabilities; this
// adapter rejects them with the feature that made them unsupported instead of
// widening its own envelope. The family proves that a non-contraction source
// enters the same adapter, contract and realization boundary the contraction
// examples use, without a new opcode or an operator-name switch in the generic
// driver.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CONVERSION_WORKLOAD_ELEMENTWISESOURCEADAPTER_H
#define LLK_CONVERSION_WORKLOAD_ELEMENTWISESOURCEADAPTER_H

#include "LLK/Conversion/Workload/SourceAdapter.h"

#include <memory>

namespace mlir::llk::workload {

/// The bounded elementwise family's registration key. It is part of source
/// identity.
llvm::StringRef elementwiseSourceAdapterKey();

/// Builds the bounded elementwise family. `makeDefaultSourceAdapters()` adds
/// it, so an unconfigured tool resolves the supported structured roots.
std::unique_ptr<SourceAdapter> makeElementwiseSourceAdapter();

} // namespace mlir::llk::workload

#endif // LLK_CONVERSION_WORKLOAD_ELEMENTWISESOURCEADAPTER_H
