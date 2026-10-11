#ifndef LLK_CONVERSION_MICROMAPPING_MAPPEDTUNINGSESSION_H
#define LLK_CONVERSION_MICROMAPPING_MAPPEDTUNINGSESSION_H

#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Conversion/MicroMapping/CandidateInstantiation.h"
#include "LLK/Mapping/PlanReport.h"
#include "LLK/Perf/CandidateGenerator.h"
#include "LLK/Perf/TuningSession.h"
#include "LLK/Runtime/MappedExecutable.h"
#include "LLK/Runtime/MappedInvocation.h"
#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace mlir::llk::tuning {

/// The complete, length-delimited identity of one mapped executable. The
/// canonical string is suitable for external measurement records; no such
/// records are stored by the tuning session.
struct MappedMeasurementIdentity {
  std::string canonical;
  std::string contentHash;
  uint64_t planId = 0;
  uint64_t originalGraphHash = 0;
  uint64_t instantiatedGraphHash = 0;
  uint64_t bindingHash = 0;
  uint64_t abiHash = 0;
  std::vector<std::string> operationKeys;
  std::vector<std::string> connectionKeys;
  std::string targetIdentity;
  std::string backendIdentity;
};

/// Buffers remain owned for the full provider callback, including its warmup
/// and timed invocations. The descriptors point into `storage` allocations.
struct OwnedInvocationBuffers {
  std::vector<std::vector<uint8_t>> storage;
  std::vector<::llk::InvocationBuffer2D> inputs;
  std::vector<::llk::InvocationBuffer2D> outputs;
};

struct MappedTuningCandidate;

struct MappedMeasurementRequest {
  const MappedTuningCandidate &candidate;
  ::llk::MappedExecutable &executable;
  const MappedMeasurementIdentity &identity;
  llvm::ArrayRef<::llk::InvocationBuffer2D> inputs;
  llvm::ArrayRef<::llk::InvocationBuffer2D> outputs;
};

using MappedInputProvider =
    std::function<llvm::Expected<OwnedInvocationBuffers>(
        const ::llk::KernelAbi &)>;
using MappedMeasurementProvider =
    std::function<llvm::Expected<std::optional<perf::CandidateMetrics>>(
        const MappedMeasurementRequest &)>;
using MappedMeasurementVerifier = std::function<llvm::Error(
    const MappedTuningCandidate &, llvm::ArrayRef<::llk::InvocationBuffer2D>,
    llvm::ArrayRef<::llk::InvocationBuffer2D>)>;

struct MappedMeasurementOptions {
  MappedInputProvider inputs;
  MappedMeasurementProvider measure;
  MappedMeasurementVerifier verify;
};

struct MappedTuningOptions {
  perf::CandidateGeneratorOptions generator;
  mapping::MappingSearchOptions mapping;
  CandidateInstantiationOptions source;
  ::llk::MappedBackend backend = ::llk::MappedBackend::SelectedTarget;
  bool executable = true;
  uint64_t topK = 10;
  uint64_t measureTop = 1;
  uint64_t maxMeasurementAttempts = 0;
  MappedMeasurementOptions measurement;
};

struct MappedTuningCandidate {
  perf::TuningResult ranking;
  mapping::CoveringPlan plan;
  uint64_t sourceGraphHash = 0;
  uint64_t bindingHash = 0;
  bool materializationReady = false;
  std::string planReport;
  bool executable = false;
  std::optional<MappedMeasurementIdentity> measurementIdentity;
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
  bool measuredCohortOnly = false;
  bool hasMeasuredResult = false;
  uint64_t measuredCohortSize = 0;
  uint64_t measurementAttempts = 0;
  std::vector<MappedTuningCandidate> ranked;
  std::vector<MappedTuningCandidate> unrankable;
  std::vector<perf::TuningResult> rejected;
};

llvm::Expected<MappedTuningReport>
runMappedTuningSession(ModuleOp source, const perf::SearchSpace &space,
                       const perf::WorkloadShape &shape,
                       const mapping::MappingTarget &target,
                       const MappedTuningOptions &options = {});

} // namespace mlir::llk::tuning

#endif
