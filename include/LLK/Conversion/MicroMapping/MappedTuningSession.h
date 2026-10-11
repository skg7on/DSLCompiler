#ifndef LLK_CONVERSION_MICROMAPPING_MAPPEDTUNINGSESSION_H
#define LLK_CONVERSION_MICROMAPPING_MAPPEDTUNINGSESSION_H

#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Conversion/MicroMapping/CandidateInstantiation.h"
#include "LLK/Mapping/PlanReport.h"
#include "LLK/Perf/CandidateGenerator.h"
#include "LLK/Perf/TuningSession.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::llk::tuning {

struct MappedTuningOptions {
  perf::CandidateGeneratorOptions generator;
  mapping::MappingSearchOptions mapping;
  CandidateInstantiationOptions source;
  ::llk::MappedBackend backend = ::llk::MappedBackend::SelectedTarget;
  bool executable = true;
  uint64_t topK = 10;
};

struct MappedTuningCandidate {
  perf::TuningResult ranking;
  mapping::CoveringPlan plan;
  uint64_t sourceGraphHash = 0;
  uint64_t bindingHash = 0;
  bool materializationReady = false;
  std::string planReport;
  bool executable = false;
};

struct MappedTuningReport {
  uint64_t generated = 0;
  uint64_t rejectedBySource = 0;
  uint64_t rejectedByLegality = 0;
  uint64_t rejectedByMapping = 0;
  uint64_t rejectedByCompile = 0;
  uint64_t rejectedByCapacity = 0;
  uint64_t rejectedByRoute = 0;
  uint64_t failedCompileAttempts = 0;
  uint64_t mappingCandidateCount = 0;
  uint64_t mappingRouteCount = 0;
  uint64_t mappingPlanCount = 0;
  bool generatorTruncated = false;
  bool mappingTruncated = false;
  std::vector<MappedTuningCandidate> ranked;
  std::vector<perf::TuningResult> rejected;
};

llvm::Expected<MappedTuningReport>
runMappedTuningSession(ModuleOp source, const perf::SearchSpace &space,
                       const perf::WorkloadShape &shape,
                       const mapping::MappingTarget &target,
                       const MappedTuningOptions &options = {});

} // namespace mlir::llk::tuning

#endif
