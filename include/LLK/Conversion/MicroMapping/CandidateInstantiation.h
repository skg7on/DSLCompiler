//===- CandidateInstantiation.h - Instantiate a candidate -------*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CONVERSION_MICROMAPPING_CANDIDATEINSTANTIATION_H
#define LLK_CONVERSION_MICROMAPPING_CANDIDATEINSTANTIATION_H

#include "LLK/Conversion/Workload/SourceAdapter.h"
#include "LLK/Conversion/Workload/WorkloadContract.h"
#include "LLK/Machine/MachineModel.h"
#include "LLK/Mapping/WorkloadGraph.h"
#include "LLK/Perf/Candidate.h"
#include "LLK/Perf/Legality.h"
#include "LLK/Perf/SearchSpace.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace mlir::llk::tuning {

enum class SourceMode { SemanticSource, ConcreteMicro, Synthetic };

struct CandidateInstantiationOptions {
  SourceMode sourceMode = SourceMode::SemanticSource;
  std::string sourceSymbol;
  uint64_t sourceRootOrdinal = 0;
  /// Names the source adapter to dispatch to. Empty means the source root must
  /// be accepted by exactly one registered adapter.
  std::string adapterKey;
  const machine::MachineModel *machine = nullptr;
};

struct CandidateInstance {
  /// Owns both the source clone and the emitted kernel.
  OwningOpRef<ModuleOp> module;
  Operation *kernel = nullptr;
  uint64_t sourceGraphHash = 0;
  uint64_t bindingHash = 0;
  SourceMode sourceMode = SourceMode::SemanticSource;
  perf::BindingFacts bindingFacts;
  /// Populated by the contract-based source path; no contraction facts needed.
  workload::WorkloadContract workloadContract;
  /// What the source stage realized. Mapping-owned choices stay pending here;
  /// the mapping stage completes the receipt before retention and reporting.
  workload::BindingReceipt receipt;
  mapping::WorkloadGraph workloadGraph;
};

/// Validates and realizes a complete candidate. SemanticSource clones the
/// exact LLK root and exports it with the selected schedule. ConcreteMicro
/// accepts an unchanged concrete kernel only. Synthetic uses the legacy
/// source-independent binder explicitly.
llvm::Expected<CandidateInstance>
instantiateCandidate(ModuleOp source, const perf::SearchSpace &space,
                     const perf::Candidate &candidate,
                     const perf::WorkloadShape &shape,
                     const CandidateInstantiationOptions &options);

/// Instantiates an exact, parameter-free ConcreteMicro kernel. Tensor inputs
/// and results match the ordered contract ports (Read and Write respectively).
/// A nonempty numerical contract requires matching micro.numerical_contract
/// source metadata. Required primitives must occur in the selected body and
/// cover every execution/movement primitive derived from its workload graph.
/// sourceSymbol selects an exact kernel; empty requires a unique kernel.
/// The caller's module is never modified.
llvm::Expected<CandidateInstance>
instantiateCandidate(ModuleOp source, const perf::SearchSpace &space,
                     const perf::Candidate &candidate,
                     const workload::WorkloadContract &contract,
                     const CandidateInstantiationOptions &options);

/// The same, with the source adapters injected. SemanticSource dispatches to
/// the adapter that accepts the selected root: exactly one must, unless
/// options.adapterKey names it. The adapter's described contract must match
/// `contract`, every source-owned parameter must be realized by the returned
/// receipt, and mapping-owned parameters must still be pending -- the receipt
/// is source-stage evidence, not a claim about work that has not happened.
/// The emitted kernel must belong to the private clone, match the concrete
/// tensor/numerical contract and pass IR verification. An adapter may erase
/// the original root/function while replacing it; identity is retained first.
///
/// The overload without a registry constructs makeDefaultSourceAdapters().
llvm::Expected<CandidateInstance>
instantiateCandidate(ModuleOp source, const perf::SearchSpace &space,
                     const perf::Candidate &candidate,
                     const workload::WorkloadContract &contract,
                     const CandidateInstantiationOptions &options,
                     const workload::SourceAdapterRegistry &registry);

} // namespace mlir::llk::tuning

#endif
