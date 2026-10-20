//===- ContractionSupport.h - Contraction dtype support ---------*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The one place that answers "can this capability execute a contraction with
// this input/accumulator pair?". A contraction is not tied to a matrix engine:
// a vector capability that declares accumulator dtypes and lanes can run one
// too, and a machine with no matrix engine is not a machine without
// accumulation. Keeping the predicate shared means the DAG interpreter and the
// `mma_compatible` legality rule cannot drift on what "supported" means.

#ifndef LLK_PERF_CONTRACTIONSUPPORT_H
#define LLK_PERF_CONTRACTIONSUPPORT_H

#include "LLK/Machine/MachineModel.h"

#include "llvm/ADT/StringRef.h"

namespace mlir::llk::perf {

/// True when `engine` is a vector or matrix capability that declares both the
/// input and the accumulator dtype of the pair. Any other capability kind --
/// a movement engine, an executor-only node -- cannot run a contraction, so it
/// never matches.
bool supportsContractionTypes(const machine::ComputeNode &engine,
                              llvm::StringRef inputDtype,
                              llvm::StringRef accumulatorDtype);

} // namespace mlir::llk::perf

#endif // LLK_PERF_CONTRACTIONSUPPORT_H
