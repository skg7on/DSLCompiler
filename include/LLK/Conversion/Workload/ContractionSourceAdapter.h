//===- ContractionSourceAdapter.h - Contraction source family ---*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The built-in contraction source family (framework B4). It is a compatibility
// wrapper, not a new implementation: the family it recognizes is exactly the
// set of roots the operator-kernel registry gives a Micro-IR lowering, and the
// work it does is the work the contraction export already did, moved behind the
// generic adapter boundary so that the driver, the tuner and the mapping stage
// no longer carry contraction-specific code.
//
// Everything that is contraction-specific lives on this side of the boundary --
// BM/BN/BK realization, the arm count, the initialized destination, the
// padding tail policy and the emitted contraction facts. Generic dispatch can
// therefore stay structural: it asks whether an adapter accepts a root, never
// whether that root is spelled `matmul` or `fused_swiglu`.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CONVERSION_WORKLOAD_CONTRACTIONSOURCEADAPTER_H
#define LLK_CONVERSION_WORKLOAD_CONTRACTIONSOURCEADAPTER_H

#include "LLK/Conversion/Workload/SourceAdapter.h"
#include "LLK/Perf/SearchSpace.h"

#include <memory>

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir::llk::workload {

/// The contraction family's registration key. It is part of source identity.
llvm::StringRef contractionSourceAdapterKey();

/// Builds the built-in contraction family. `makeDefaultSourceAdapters()` adds
/// it, so an unconfigured tool resolves the supported contraction roots.
std::unique_ptr<SourceAdapter> makeContractionSourceAdapter();

/// The M/N/K and dtype facts `root` declares, read through the operator-kernel
/// registry that owns the per-operator knowledge rather than re-derived from
/// operand positions here. The legacy `WorkloadShape` entry point checks a
/// caller's workload against this before translating it into a generic
/// contract, which keeps that entry point's strict mismatch check exactly where
/// it was. Fails for a root no registered lowering handles.
llvm::Expected<perf::WorkloadShape>
describeContractionShape(mlir::Operation *root);

} // namespace mlir::llk::workload

#endif // LLK_CONVERSION_WORKLOAD_CONTRACTIONSOURCEADAPTER_H
