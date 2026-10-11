#ifndef LLK_CONVERSION_MICROMAPPING_CANDIDATEINSTANTIATION_H
#define LLK_CONVERSION_MICROMAPPING_CANDIDATEINSTANTIATION_H

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

} // namespace mlir::llk::tuning

#endif
