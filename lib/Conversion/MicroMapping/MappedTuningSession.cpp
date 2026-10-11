#include "LLK/Conversion/MicroMapping/MappedTuningSession.h"

#include "CompletePlanEvaluation.h"
#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Mapping/MappingMetadata.h"
#include "LLK/Mapping/WorkloadGraph.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <limits>
#include <map>

using namespace mlir;

namespace mlir::llk::tuning {
namespace {

llvm::Error fail(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

mapping::LayoutContext layoutContext(const mapping::WorkloadGraph &graph) {
  mapping::LayoutContext result;
  result.elementType = "f32";
  for (const mapping::WorkloadValue &value : graph.getValues()) {
    auto shaped = dyn_cast<ShapedType>(value.type);
    if (!shaped || !shaped.hasStaticShape())
      continue;
    result.rank = shaped.getRank();
    Type type = shaped.getElementType();
    if (isa<BFloat16Type>(type))
      result.elementType = "bf16";
    else if (isa<Float16Type>(type))
      result.elementType = "f16";
    else if (isa<Float32Type>(type))
      result.elementType = "f32";
    else if (auto integer = dyn_cast<IntegerType>(type))
      result.elementType = ("i" + llvm::Twine(integer.getWidth())).str();
    break;
  }
  return result;
}

llvm::Expected<std::pair<llvm::StringMap<std::string>, mapping::BoundAxes>>
projectAxes(const perf::SearchSpace &space, const perf::Candidate &candidate,
            const mapping::MappingTarget &target) {
  llvm::StringMap<std::string> layouts;
  llvm::StringMap<std::string> layoutNames;
  mapping::BoundAxes axes;
  llvm::StringSet<> kinds;
  for (const perf::SearchParam &param : space.params) {
    auto value = candidate.symbol(param.name);
    if (param.kind == "layout") {
      if (!value)
        return fail("candidate layout parameter '" + param.name +
                    "' is not symbolic");
      if (layoutNames.count(param.role))
        return fail(
            "more than one layout parameter selects role '" +
            (param.role.empty() ? StringRef("<none>") : StringRef(param.role)) +
            "'");
      layoutNames[param.role] = param.name;
      std::vector<const mapping::LayoutDef *> implementing =
          target.layouts().implementing(*value);
      if (implementing.size() > 1)
        return fail("bound layout kind '" + *value +
                    "' is ambiguous on target '" + target.name() + "'");
      layouts[param.role] =
          implementing.empty() ? value->str() : implementing.front()->id;
    } else if (param.kind == "owner_mapping" || param.kind == "memory_path") {
      if (!kinds.insert(param.kind).second)
        return fail("multiple parameters declare mapping axis '" + param.kind +
                    "'");
      if (!value)
        return fail("candidate mapping axis '" + param.name +
                    "' is not symbolic");
      if (param.kind == "owner_mapping")
        axes.ownerMapping = value->str();
      else
        axes.memoryPath = value->str();
    }
  }
  return std::make_pair(std::move(layouts), std::move(axes));
}

uint64_t candidateDomainSize(const perf::SearchSpace &space) {
  uint64_t result = 1;
  for (const perf::SearchParam &param : space.params) {
    if (param.choices.empty())
      return 0;
    if (result > std::numeric_limits<uint64_t>::max() / param.choices.size())
      return std::numeric_limits<uint64_t>::max();
    result *= param.choices.size();
  }
  return result;
}

mapping::SearchBinding toBinding(const perf::Candidate &candidate) {
  llvm::StringMap<mapping::SearchValue> values;
  for (const auto &entry : candidate.values)
    values[entry.first] = entry.second;
  for (const auto &entry : candidate.symbolicValues)
    values[entry.first] = entry.second;
  return mapping::makeSearchBinding(candidate.id, std::move(values));
}

bool ranksBefore(const MappedTuningCandidate &lhs,
                 const MappedTuningCandidate &rhs,
                 const perf::SearchObjective &objective) {
  if (perf::ranksBefore(lhs.ranking, rhs.ranking, objective))
    return true;
  if (perf::ranksBefore(rhs.ranking, lhs.ranking, objective))
    return false;
  return lhs.plan.id < rhs.plan.id;
}

} // namespace

llvm::Expected<MappedTuningReport>
runMappedTuningSession(ModuleOp source, const perf::SearchSpace &space,
                       const perf::WorkloadShape &shape,
                       const mapping::MappingTarget &target,
                       const MappedTuningOptions &options) {
  if (!source)
    return fail("mapped tuning requires a source module");
  if (llvm::Error error = perf::validateObjective(space.objective))
    return std::move(error);
  MappedTuningReport report;
  std::vector<perf::Candidate> candidates =
      perf::generateCandidates(space, options.generator);
  report.generated = candidates.size();
  report.generatorTruncated = options.generator.maxCandidates &&
                              candidates.size() < candidateDomainSize(space);

  for (const perf::Candidate &candidate : candidates) {
    if (options.executable && !target.codegenRequirements()) {
      report.rejectedByCompile++;
      report.rejected.push_back(
          {candidate,
           {},
           false,
           "backend_unavailable: target '" + target.name().str() +
               "' has no selected backend code-generation contract"});
      continue;
    }
    CandidateInstantiationOptions sourceOptions = options.source;
    sourceOptions.machine = &target.machine();
    auto instance =
        instantiateCandidate(source, space, candidate, shape, sourceOptions);
    if (!instance) {
      std::string reason = llvm::toString(instance.takeError());
      if (StringRef(reason).starts_with("candidate is illegal:")) {
        report.rejectedByLegality++;
        report.rejected.push_back(
            {candidate, {}, false, "source_legality: " + reason});
      } else {
        report.rejectedBySource++;
        report.rejected.push_back(
            {candidate, {}, false, "source_binding: " + reason});
      }
      continue;
    }

    auto axes = projectAxes(space, candidate, target);
    if (!axes) {
      report.rejectedBySource++;
      report.rejected.push_back(
          {candidate,
           {},
           false,
           "source_binding: " + llvm::toString(axes.takeError())});
      continue;
    }
    auto binding = toBinding(candidate);

    mapping::MappingSearchOptions searchOptions = options.mapping;
    const unsigned requestedPlans = searchOptions.topK;
    const mapping::BindContract contract =
        options.executable ? mapping::BindContract::Executable
                           : mapping::BindContract::Partial;
    const uint64_t memoryBudget = searchOptions.memoryBudgetBytes;
    searchOptions.evaluateCompletePlan =
        [module = instance->module.get(), graph = &instance->workloadGraph,
         &target, contract, memoryBudget](const mapping::CoveringPlan &proposal)
        -> llvm::Expected<mapping::CompletePlanEvaluation> {
      return mapping::evaluateCompletePlan(module, *graph, target, proposal,
                                           contract, memoryBudget);
    };
    searchOptions.requireCompleteEvaluation = true;
    mapping::CoveringSearch search(
        instance->workloadGraph, target, *instance->module->getContext(),
        layoutContext(instance->workloadGraph), searchOptions, binding,
        axes->first, axes->second);
    auto found = search.search();
    if (!found)
      return found.takeError();
    report.mappingCandidateCount += found->candidateCount;
    report.mappingRouteCount += found->routeCount;
    report.mappingPlanCount += found->planCount;
    report.mappingTruncated |=
        found->searchTruncated || found->plans.size() < found->planCount;
    if (found->plans.empty()) {
      report.rejectedByMapping++;
      const bool capacity =
          llvm::any_of(found->frontier.diagnostics,
                       [](const mapping::Diagnostic &diagnostic) {
                         return diagnostic.code ==
                                mapping::DiagnosticCode::MemoryCapacityExceeded;
                       });
      const bool route = llvm::any_of(
          found->frontier.diagnostics,
          [](const mapping::Diagnostic &diagnostic) {
            return diagnostic.code == mapping::DiagnosticCode::NoMemoryRoute;
          });
      report.rejectedByCapacity += capacity;
      report.rejectedByRoute += route;
      std::string reason =
          capacity ? "capacity_failure: no feasible covering"
                   : (route ? "no_route: no feasible covering"
                            : "mapping_failure: no feasible covering");
      if (!found->frontier.diagnostics.empty())
        reason += ": " + found->frontier.diagnostics.front().message;
      report.rejected.push_back({candidate, {}, false, std::move(reason)});
      continue;
    }

    MappedTuningCandidate selected;
    bool compiled = false;
    std::string compileFailure;
    const size_t tryCount =
        requestedPlans == 0
            ? found->plans.size()
            : std::min<size_t>(requestedPlans, found->plans.size());
    for (size_t i = 0; i < tryCount; ++i) {
      ::llk::MappedCompileOptions compileOptions;
      compileOptions.requireExecutable = options.executable;
      compileOptions.backend = options.executable
                                   ? options.backend
                                   : ::llk::MappedBackend::Reference;
      compileOptions.stop = options.executable ? ::llk::MappedStop::Lowered
                                               : ::llk::MappedStop::MappedMicro;
      auto compilation = ::llk::compileMappedKernel(
          instance->module.get(), target, found->plans[i], compileOptions);
      if (!compilation) {
        compileFailure = llvm::toString(compilation.takeError());
        report.failedCompileAttempts++;
        continue;
      }
      selected.plan = found->plans[i];
      selected.sourceGraphHash = instance->sourceGraphHash;
      selected.bindingHash = instance->bindingHash;
      selected.materializationReady = selected.plan.materialized;
      selected.ranking = {candidate, {}, true, {}};
      selected.ranking.metrics.predictedCycles = static_cast<uint64_t>(
          std::max(0.0, selected.plan.totalCost.latencyCycles));
      selected.ranking.metrics.predictedNs =
          target.machine().clockHz
              ? selected.plan.totalCost.latencyCycles * 1.0e9 /
                    static_cast<double>(*target.machine().clockHz)
              : 0.0;
      selected.ranking.metrics.dramBytes = selected.plan.totalCost.dramBytes;
      selected.ranking.metrics.sramBytes = selected.plan.totalCost.localBytes;
      selected.ranking.metrics.matrixUtilization =
          selected.plan.totalCost.computeUtilization;
      selected.ranking.metrics.dmaUtilization =
          selected.plan.totalCost.transferUtilization;
      selected.executable = options.executable;
      auto onePlan = *found;
      auto selectedPlan =
          std::find_if(onePlan.plans.begin(), onePlan.plans.end(),
                       [&](const mapping::CoveringPlan &plan) {
                         return plan.id == selected.plan.id;
                       });
      if (selectedPlan != onePlan.plans.begin())
        std::iter_swap(onePlan.plans.begin(), selectedPlan);
      selected.planReport = mapping::writePlanReport(
          onePlan, target.machine(), target, searchOptions,
          mapping::computeSourceGraphHash(instance->workloadGraph));
      compiled = true;
      break;
    }
    if (!compiled) {
      report.rejectedByCompile++;
      report.rejected.push_back(
          {candidate,
           {},
           false,
           "backend_lowering: " + (compileFailure.empty()
                                       ? std::string("no retained plan")
                                       : compileFailure)});
      if (tryCount < found->plans.size())
        report.mappingTruncated = true;
      continue;
    }
    if (tryCount < found->plans.size())
      report.mappingTruncated = true;
    report.ranked.push_back(std::move(selected));
  }

  std::stable_sort(
      report.ranked.begin(), report.ranked.end(),
      [&](const MappedTuningCandidate &a, const MappedTuningCandidate &b) {
        return ranksBefore(a, b, space.objective);
      });
  if (options.topK && report.ranked.size() > options.topK)
    report.ranked.resize(options.topK);
  return report;
}

} // namespace mlir::llk::tuning
