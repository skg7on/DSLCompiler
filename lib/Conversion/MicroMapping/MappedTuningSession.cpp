//===- MappedTuningSession.cpp - Mapped tuning session --------------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/MicroMapping/MappedTuningSession.h"

#include "CompletePlanEvaluation.h"
#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Mapping/CostEvent.h"
#include "LLK/Mapping/LatencyProvider.h"
#include "LLK/Mapping/MappingMetadata.h"
#include "LLK/Mapping/WorkloadGraph.h"
#include "LLK/Version.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cmath>
#include <functional>
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
  // An explicitly declared singleton has one empty binding; an ordinary empty
  // space has none.
  if (space.params.empty())
    return space.singleton ? 1 : 0;
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

bool objectiveUsesMeasuredMetric(const perf::SearchObjective &objective) {
  auto measured = [](llvm::StringRef name) {
    return name == "measured_ns" || name == "measured_gflops";
  };
  return measured(objective.primaryMetric) ||
         llvm::any_of(objective.secondaryMetrics, measured);
}

void appendField(std::string &out, llvm::StringRef value) {
  out += std::to_string(value.size());
  out += ':';
  out += value.str();
}

std::string stableContentHash(llvm::StringRef value) {
  uint64_t hash = 14695981039346656037ull;
  for (unsigned char byte : value.bytes()) {
    hash ^= byte;
    hash *= 1099511628211ull;
  }
  return std::to_string(hash);
}

std::string renderSearchValue(const mapping::SearchValue &value) {
  if (const auto *integer = std::get_if<int64_t>(&value))
    return "i" + std::to_string(*integer);
  return "s" + std::get<std::string>(value);
}

bool buffersAreOwned(const OwnedInvocationBuffers &buffers) {
  auto owned = [&](const ::llk::InvocationBuffer2D &buffer) {
    return llvm::any_of(buffers.storage,
                        [&](const std::vector<uint8_t> &bytes) {
                          return !bytes.empty() &&
                                 buffer.descriptor.allocated == bytes.data() &&
                                 buffer.descriptor.aligned == bytes.data() &&
                                 buffer.allocationBytes <= bytes.size();
                        });
  };
  return llvm::all_of(buffers.inputs, owned) &&
         llvm::all_of(buffers.outputs, owned);
}

MappedMeasurementIdentity
makeMeasurementIdentity(const MappedTuningCandidate &candidate,
                        const CandidateInstance &instance,
                        const ::llk::MappedCompilation &compiled,
                        const mapping::MappingTarget &target) {
  MappedMeasurementIdentity identity;
  identity.planId = candidate.plan.id;
  identity.originalGraphHash = instance.sourceGraphHash;
  identity.instantiatedGraphHash =
      mapping::computeSourceGraphHash(instance.workloadGraph);
  identity.bindingHash = instance.bindingHash;
  identity.abiHash = compiled.executable->abiHash();
  identity.targetIdentity = std::to_string(candidate.plan.targetHash) + ":" +
                            std::to_string(candidate.plan.machineHash) + ":" +
                            std::to_string(candidate.plan.layoutHash) + ":" +
                            std::to_string(candidate.plan.ruleHash);
  identity.backendIdentity = compiled.executable->executionIdentity().str();
  auto render = [](auto value) {
    std::string text;
    llvm::raw_string_ostream stream(text);
    value.print(stream);
    return stream.str();
  };
  for (const mapping::PlanPlacement &placement : candidate.plan.placements) {
    mapping::OperationSignature signature;
    if (const mapping::WorkloadNode *node =
            instance.workloadGraph.findNode(placement.node)) {
      signature.operation = node->opName;
      std::vector<std::string> operands, results;
      for (const mapping::WorkloadPort &port : node->inputs)
        operands.push_back(render(port.type));
      for (const mapping::WorkloadPort &port : node->outputs)
        results.push_back(render(port.type));
      signature.operandTypes = llvm::join(operands, ",");
      signature.resultTypes = llvm::join(results, ",");
      signature.attributes = render(node->attributes);
    }
    signature.rule = placement.rule;
    if (const mapping::RuleDef *rule = target.rules().find(placement.rule))
      signature.ruleVersion = rule->version;
    signature.bundle = placement.bundle.emitterKey;
    std::vector<std::string> parameters;
    for (const auto &entry : placement.resolvedParameters)
      parameters.push_back(entry.getKey().str() + "=" +
                           renderSearchValue(entry.getValue()));
    llvm::sort(parameters);
    signature.bundleParameters = llvm::join(parameters, ";");
    std::vector<std::string> layouts;
    for (const auto &entry : placement.layoutSolutions) {
      std::vector<std::string> values;
      for (const auto &parameter : entry.second.parameters)
        values.push_back(parameter.getKey().str() + "=" +
                         renderSearchValue(parameter.getValue()));
      llvm::sort(values);
      layouts.push_back(entry.second.layoutClass + "{" +
                        llvm::join(values, ",") + "}");
    }
    llvm::sort(layouts);
    signature.layout = llvm::join(layouts, ";");
    signature.placementClass = placement.executor;
    signature.placement = placement.executor;
    std::vector<std::string> memories;
    for (const auto &entry : placement.memories)
      memories.push_back(entry.getKey().str() + "=" + entry.second);
    llvm::sort(memories);
    signature.placement += "|" + llvm::join(memories, ",");
    std::vector<std::string> computes;
    for (const auto &entry : placement.computeBindings)
      computes.push_back(entry.getKey().str() + "=" + entry.second);
    llvm::sort(computes);
    signature.compute = llvm::join(computes, ",");
    identity.operationKeys.push_back(signature.canonicalString());
  }
  for (const mapping::PlanConnection &planned :
       candidate.plan.connectionPlans) {
    mapping::ConnectionPlan connection;
    connection.id = planned.id;
    connection.value = planned.value;
    connection.kind = planned.kind;
    connection.memoryRoute = planned.route;
    connection.transferEngines = planned.engines;
    connection.transform = planned.transform;
    connection.producerPort = planned.producerPort;
    connection.consumerPorts = planned.consumerPorts;
    for (const mapping::PortRef &producer : planned.producerPorts)
      connection.producers.push_back(producer.node);
    connection.gatherSemantics = planned.gatherSemantics;
    connection.concatAxis = planned.concatAxis;
    auto portMap =
        [&](const mapping::PortRef &port) -> std::optional<mlir::AffineMap> {
      const mapping::WorkloadNode *node =
          instance.workloadGraph.findNode(port.node);
      if (!node)
        return std::nullopt;
      const auto &ports = port.direction == mapping::PortDirection::Input
                              ? node->inputs
                              : node->outputs;
      if (port.index >= ports.size())
        return std::nullopt;
      return ports[port.index].accessMap;
    };
    if (planned.producerPort)
      connection.producerMap = portMap(*planned.producerPort);
    for (const mapping::PortRef &consumer : planned.consumerPorts)
      if (auto map = portMap(consumer))
        connection.consumerMaps.push_back(*map);
    if (!planned.producerPorts.empty())
      connection.producer = planned.producerPorts.front().node;
    else if (planned.producerPort)
      connection.producer = planned.producerPort->node;
    for (const mapping::PortRef &consumer : planned.consumerPorts)
      connection.consumers.push_back(consumer.node);
    connection.cost = planned.cost;
    identity.connectionKeys.push_back(
        mapping::connectionSignatureFor(connection, instance.workloadGraph,
                                        target.machine())
            .canonicalString());
  }
  if (identity.operationKeys.empty())
    identity.operationKeys.push_back(candidate.planReport);
  if (identity.connectionKeys.empty())
    identity.connectionKeys.push_back(candidate.planReport);
  std::string canonical;
  appendField(canonical, "mapped-measurement-v1");
  appendField(canonical, std::to_string(identity.planId));
  appendField(canonical, std::to_string(identity.originalGraphHash));
  appendField(canonical, std::to_string(identity.instantiatedGraphHash));
  appendField(canonical, std::to_string(identity.bindingHash));
  appendField(canonical, std::to_string(identity.abiHash));
  appendField(canonical, identity.targetIdentity);
  appendField(canonical, identity.backendIdentity);
  appendField(canonical, LLK_COMPILER_VERSION);
  appendField(canonical, std::to_string(mapping::kCostModelVersion));
  appendField(canonical, std::to_string(mapping::kConnectionKeyVersion));
  appendField(canonical, candidate.planReport);
  for (const std::string &key : identity.operationKeys)
    appendField(canonical, key);
  for (const std::string &key : identity.connectionKeys)
    appendField(canonical, key);
  identity.canonical = std::move(canonical);
  identity.contentHash = stableContentHash(identity.canonical);
  return identity;
}

/// How one candidate is realized: through the legacy shape translation or
/// through a source contract. The rest of the session is shared, so the two
/// paths cannot drift in legality, search, retention or measurement.
using InstantiateCandidateFn = std::function<llvm::Expected<CandidateInstance>(
    const perf::Candidate &, const CandidateInstantiationOptions &)>;

/// The target layout id a bound symbolic value resolves to. `projectAxes` uses
/// the same resolution, so the receipt is checked against the value the search
/// actually consumed.
std::string resolvedLayoutId(const mapping::MappingTarget &target,
                             llvm::StringRef value) {
  std::vector<const mapping::LayoutDef *> implementing =
      target.layouts().implementing(value);
  return implementing.empty() ? value.str() : implementing.front()->id;
}

/// True when the plan's own solved layouts show the bound value in use. A
/// role-less parameter governs the axis as a whole, so any solved layout
/// counts; a parameter with a role is realized only by a solution on that
/// occurrence.
bool layoutConsumed(const perf::SearchSpace &space, llvm::StringRef parameter,
                    const perf::Candidate &candidate,
                    const mapping::CoveringPlan &plan,
                    const mapping::MappingTarget &target) {
  std::optional<llvm::StringRef> value = candidate.symbol(parameter);
  if (!value)
    return false;
  const perf::SearchParam *param = space.findParam(parameter);
  if (!param)
    return false;
  const std::string resolved = resolvedLayoutId(target, *value);
  // The selected plan shows the value in use when a placement's rule declares
  // the requirement for this role (or, role-less, anywhere) and the plan
  // carries the solved layout for it. A value no selected rule consumes stays
  // out of the receipt.
  for (const mapping::PlanPlacement &placement : plan.placements) {
    const mapping::RuleDef *rule = target.rules().find(placement.rule);
    if (!rule)
      continue;
    const bool declared = llvm::any_of(
        rule->layoutRequirements,
        [&](const mapping::RuleLayoutRequirement &requirement) {
          if (!param->role.empty() && requirement.port != param->role)
            return false;
          return requirement.layoutId == resolved;
        });
    if (!declared)
      continue;
    const bool solved =
        llvm::any_of(placement.layoutSolutions, [&](const auto &entry) {
          return entry.second.layoutClass == resolved;
        });
    if (solved)
      return true;
  }
  return false;
}

bool executorNestedUnder(const machine::MachineModel &machine,
                         llvm::StringRef inner, llvm::StringRef outer) {
  const machine::ExecutorNode *current = machine.findExecutor(inner);
  llvm::StringSet<> visited;
  while (current && current->parent && visited.insert(current->id).second) {
    if (*current->parent == outer)
      return true;
    current = machine.findExecutor(*current->parent);
  }
  return false;
}

/// Resolve the path under one selected executor. Intermediate owners belong to
/// machine containment; they need not each run a separate selected operation.
/// A final compute-only kind is realized by its selected attached capability.
bool ownerMappingConsumed(llvm::StringRef value,
                          const mapping::CoveringPlan &plan,
                          const machine::MachineModel &machine) {
  llvm::SmallVector<llvm::StringRef, 4> owners;
  value.split(owners, '/');
  if (owners.empty() ||
      llvm::any_of(owners, [](llvm::StringRef owner) { return owner.empty(); }))
    return false;
  for (const mapping::PlanPlacement &placement : plan.placements) {
    if (!machine.ownerMatches(owners.front(), placement.executor))
      continue;
    llvm::SmallVector<const machine::ExecutorNode *, 4> scopes{
        machine.findExecutor(placement.executor)};
    for (size_t index = 1; index < owners.size(); ++index) {
      llvm::SmallVector<const machine::ExecutorNode *, 4> nested;
      for (const machine::ExecutorNode &executor : machine.executors)
        if (machine.ownerMatches(owners[index], executor.id) &&
            llvm::any_of(scopes, [&](const machine::ExecutorNode *scope) {
              return executorNestedUnder(machine, executor.id, scope->id);
            }))
          nested.push_back(&executor);

      if (index + 1 == owners.size())
        for (const auto &binding : placement.computeBindings) {
          const machine::ComputeNode *compute =
              machine.findCompute(binding.second);
          if (compute &&
              (compute->kind == owners[index] ||
               llvm::is_contained(compute->refines, owners[index])) &&
              llvm::any_of(scopes, [&](const machine::ExecutorNode *scope) {
                return compute->attachedTo == scope->id ||
                       executorNestedUnder(machine, compute->attachedTo,
                                           scope->id);
              }))
            return true;
        }
      scopes = std::move(nested);
      if (scopes.empty())
        break;
    }
    if (!scopes.empty())
      return true;
  }
  return false;
}

/// True when the selected plan's resolved memories are constrained by the bound
/// path. The path *restricts* which memory kinds a rule may require, so a plan
/// consumes it when it made at least one memory decision and every kind it used
/// is on the path. Requiring every path level to appear would reject a legal
/// plan that stays in a subset of the allowed levels.
bool memoryPathConsumed(llvm::StringRef value,
                        const mapping::CoveringPlan &plan,
                        const machine::MachineModel &machine) {
  llvm::StringSet<> used;
  auto add = [&](llvm::StringRef id) {
    if (const machine::MemoryNode *memory = machine.findMemory(id))
      used.insert(memory->kind);
  };
  for (const mapping::PlanPlacement &placement : plan.placements)
    for (const auto &entry : placement.memories)
      add(entry.second);
  for (const mapping::PlanConnection &connection : plan.connectionPlans)
    for (llvm::StringRef id : connection.route)
      add(id);
  if (used.empty())
    return false;

  llvm::SmallVector<llvm::StringRef, 4> levels;
  value.split(levels, ':');
  if (levels.empty())
    return false;
  for (llvm::StringRef level : levels)
    if (!machine.findMemoryOfKind(level))
      return false;
  for (const auto &entry : used)
    if (!llvm::is_contained(levels, llvm::StringRef(entry.getKey())))
      return false;
  return true;
}

/// Completes `receipt` with the mapping-owned choices the selected plan
/// realizes and validates the result at the complete stage. A choice no
/// resolved decision consumes stays out of the receipt, which the shared
/// validator reports as `axis_not_realized`.
llvm::Error completeMappingReceipt(
    const perf::SearchSpace &space, const perf::Candidate &candidate,
    llvm::ArrayRef<workload::ParameterBinding> schema,
    const mapping::CoveringPlan &plan, const mapping::MappingTarget &target,
    workload::BindingReceipt &receipt) {
  for (const workload::ParameterBinding &binding : schema) {
    if (binding.owner != workload::BindingOwner::Mapping)
      continue;
    const perf::SearchParam *param = space.findParam(binding.parameter);
    if (!param)
      continue; // The shared validator names the unknown parameter.
    bool consumed = false;
    if (param->kind == "layout")
      consumed =
          layoutConsumed(space, binding.parameter, candidate, plan, target);
    else if (param->kind == "owner_mapping")
      consumed = ownerMappingConsumed(
          candidate.symbol(binding.parameter).value_or(llvm::StringRef()), plan,
          target.machine());
    else if (param->kind == "memory_path")
      consumed = memoryPathConsumed(
          candidate.symbol(binding.parameter).value_or(llvm::StringRef()), plan,
          target.machine());
    if (consumed)
      receipt.mapping.push_back(binding.parameter);
  }
  return workload::validateBindingReceipt(space, candidate, schema, receipt,
                                          workload::ReceiptStage::Complete);
}

llvm::Error measureSelectedCandidates(ModuleOp source,
                                      const perf::SearchSpace &space,
                                      const mapping::MappingTarget &target,
                                      const MappedTuningOptions &options,
                                      const InstantiateCandidateFn &instantiate,
                                      MappedTuningReport &report) {
  const auto &measurement = options.measurement;
  if (!measurement.measure)
    return llvm::Error::success();
  if (!measurement.inputs || !measurement.verify)
    return fail("mapped measurement requires input-buffer and output-verifier "
                "providers");

  std::vector<size_t> rejected;
  const bool measuredObjective = objectiveUsesMeasuredMetric(space.objective);
  const size_t attemptLimit =
      options.maxMeasurementAttempts
          ? std::min<size_t>(options.maxMeasurementAttempts,
                             report.ranked.size())
          : report.ranked.size();
  size_t measuredCount = 0;
  size_t completedCount = 0;
  size_t index = 0;
  for (; index < attemptLimit &&
         (measuredObjective ? measuredCount < options.measureTop
                            : completedCount < options.measureTop);
       ++index) {
    report.measurementAttempts++;
    MappedTuningCandidate &candidate = report.ranked[index];
    CandidateInstantiationOptions sourceOptions = options.source;
    sourceOptions.machine = &target.machine();
    auto instance = instantiate(candidate.ranking.candidate, sourceOptions);
    if (!instance) {
      candidate.ranking.rejectionReason =
          "measurement source: " + llvm::toString(instance.takeError());
      report.rejected.push_back(candidate.ranking);
      rejected.push_back(index);
      continue;
    }
    if (instance->sourceGraphHash != candidate.sourceGraphHash ||
        instance->bindingHash != candidate.bindingHash ||
        candidate.plan.sourceBindingHash != candidate.bindingHash) {
      candidate.ranking.rejectionReason =
          "measurement identity: source or binding hash changed";
      report.rejected.push_back(candidate.ranking);
      rejected.push_back(index);
      continue;
    }

    // Re-run the same complete-plan evaluator against the re-instantiated
    // graph. A measurement is only attached if this exact choice still wins
    // the same durable content identity under the current libraries.
    auto axes = projectAxes(space, candidate.ranking.candidate, target);
    if (!axes) {
      candidate.ranking.rejectionReason =
          "measurement mapping: " + llvm::toString(axes.takeError());
      report.rejected.push_back(candidate.ranking);
      rejected.push_back(index);
      continue;
    }
    auto binding = toBinding(candidate.ranking.candidate);
    mapping::MappingSearchOptions searchOptions = options.mapping;
    searchOptions.requireCompleteEvaluation = true;
    searchOptions.evaluateCompletePlan =
        [module = instance->module.get(), graph = &instance->workloadGraph,
         &target, memoryBudget = searchOptions.memoryBudgetBytes](
            const mapping::CoveringPlan &proposal)
        -> llvm::Expected<mapping::CompletePlanEvaluation> {
      return mapping::evaluateCompletePlan(
          module, *graph, target, proposal, mapping::BindContract::Executable,
          memoryBudget, mapping::AnalysisRequirement::Complete);
    };
    mapping::CoveringSearch search(
        instance->workloadGraph, target, *instance->module->getContext(),
        layoutContext(instance->workloadGraph), searchOptions, binding,
        axes->first, axes->second);
    auto replay = search.search();
    if (!replay) {
      candidate.ranking.rejectionReason =
          "measurement mapping: " + llvm::toString(replay.takeError());
      report.rejected.push_back(candidate.ranking);
      rejected.push_back(index);
      continue;
    }
    auto exactPlan = std::find_if(replay->plans.begin(), replay->plans.end(),
                                  [&](const mapping::CoveringPlan &plan) {
                                    return plan.id == candidate.plan.id;
                                  });
    if (exactPlan == replay->plans.end() ||
        exactPlan->graphHash != candidate.plan.graphHash ||
        exactPlan->targetHash != candidate.plan.targetHash) {
      candidate.ranking.rejectionReason =
          "measurement mapping: the selected plan identity changed on replay";
      report.rejected.push_back(candidate.ranking);
      rejected.push_back(index);
      continue;
    }

    ::llk::MappedCompileOptions compileOptions;
    compileOptions.requireExecutable = true;
    compileOptions.stop = ::llk::MappedStop::Executable;
    compileOptions.backend = options.backend;
    if (auto symbol = instance->kernel->getAttrOfType<StringAttr>("sym_name"))
      compileOptions.entrySymbol = symbol.getValue().str();
    auto compiled = ::llk::compileMappedKernel(instance->module.get(), target,
                                               *exactPlan, compileOptions);
    if (!compiled) {
      candidate.ranking.rejectionReason =
          "measurement compile: " + llvm::toString(compiled.takeError());
      report.rejected.push_back(candidate.ranking);
      report.rejectedByCompile++;
      rejected.push_back(index);
      continue;
    }
    if (!compiled->executable) {
      candidate.ranking.rejectionReason =
          "measurement compile: compiler did not produce an executable";
      report.rejected.push_back(candidate.ranking);
      report.rejectedByCompile++;
      rejected.push_back(index);
      continue;
    }
    auto buffers = measurement.inputs(compiled->executable->abi());
    if (!buffers) {
      candidate.ranking.rejectionReason =
          "measurement buffers: " + llvm::toString(buffers.takeError());
      report.rejected.push_back(candidate.ranking);
      rejected.push_back(index);
      continue;
    }
    if (!buffersAreOwned(*buffers)) {
      candidate.ranking.rejectionReason =
          "measurement buffers: descriptors must refer to caller-owned storage";
      report.rejected.push_back(candidate.ranking);
      rejected.push_back(index);
      continue;
    }
    if (llvm::Error error = ::llk::validateMappedInvocation(
            compiled->executable->abi(), buffers->inputs, buffers->outputs)) {
      candidate.ranking.rejectionReason =
          "measurement ABI: " + llvm::toString(std::move(error));
      report.rejected.push_back(candidate.ranking);
      rejected.push_back(index);
      continue;
    }
    MappedMeasurementIdentity identity =
        makeMeasurementIdentity(candidate, *instance, *compiled, target);
    MappedMeasurementRequest request{candidate, *compiled->executable, identity,
                                     buffers->inputs, buffers->outputs};
    auto observed = measurement.measure(request);
    if (!observed) {
      candidate.ranking.rejectionReason =
          "measurement provider: " + llvm::toString(observed.takeError());
      report.rejected.push_back(candidate.ranking);
      rejected.push_back(index);
      continue;
    }
    if (!*observed) {
      if (!measuredObjective)
        completedCount++;
      continue; // A miss leaves the legal static candidate untouched.
    }
    if (llvm::Error error =
            measurement.verify(candidate, buffers->inputs, buffers->outputs)) {
      candidate.ranking.rejectionReason =
          "measurement verification: " + llvm::toString(std::move(error));
      report.rejected.push_back(candidate.ranking);
      rejected.push_back(index);
      continue;
    }
    perf::CandidateMetrics metrics = **observed;
    if (!metrics.measuredNs || !std::isfinite(*metrics.measuredNs) ||
        *metrics.measuredNs <= 0.0 ||
        (metrics.measuredGflops && (!std::isfinite(*metrics.measuredGflops) ||
                                    *metrics.measuredGflops < 0.0))) {
      candidate.ranking.rejectionReason =
          "measurement metric: duration must be finite and positive; "
          "throughput must be finite and nonnegative";
      report.rejected.push_back(candidate.ranking);
      rejected.push_back(index);
      continue;
    }
    candidate.ranking.metrics.measuredNs = metrics.measuredNs;
    candidate.ranking.metrics.measuredGflops = metrics.measuredGflops;
    candidate.measurementIdentity = std::move(identity);
    bool hasObjectiveMetrics =
        bool(perf::resolveMetric(candidate.ranking.metrics,
                                 space.objective.primaryMetric)
                 .value);
    for (const std::string &secondary : space.objective.secondaryMetrics)
      hasObjectiveMetrics &=
          bool(perf::resolveMetric(candidate.ranking.metrics, secondary).value);
    if (hasObjectiveMetrics)
      measuredCount++;
    completedCount++;
  }
  for (auto it = rejected.rbegin(); it != rejected.rend(); ++it)
    report.ranked.erase(report.ranked.begin() + *it);
  return llvm::Error::success();
}

/// The shared search sequence. The only path-specific step is candidate
/// realization, so the legacy shape translation and the contract path cannot
/// diverge in legality, complete-plan evaluation, retention or measurement.
llvm::Expected<MappedTuningReport>
runMappedTuningSessionImpl(ModuleOp source, const perf::SearchSpace &space,
                           const mapping::MappingTarget &target,
                           const MappedTuningOptions &options,
                           const InstantiateCandidateFn &instantiate) {
  if (!source)
    return fail("mapped tuning requires a source module");
  if (llvm::Error error = perf::validateObjective(space.objective))
    return std::move(error);
  const bool measuredObjective = objectiveUsesMeasuredMetric(space.objective);
  if (measuredObjective && !options.measurement.measure)
    return fail("a measured objective requires a mapped measurement provider");
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
    auto instance = instantiate(candidate, sourceOptions);
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
    // The completion evaluator is the last point before a proposal can be
    // retained, so the plan-specific mapping receipt is completed and checked
    // here. A failed receipt rejects that proposal and the search keeps
    // enumerating; it is never a post-top-K filter.
    std::map<uint64_t, workload::BindingReceipt> acceptedReceipts;
    const CandidateInstance *instancePtr = &*instance;
    searchOptions.evaluateCompletePlan =
        [module = instance->module.get(), graph = &instance->workloadGraph,
         &target, contract, memoryBudget, &space, &candidate, instancePtr,
         &acceptedReceipts](const mapping::CoveringPlan &proposal)
        -> llvm::Expected<mapping::CompletePlanEvaluation> {
      auto evaluation = mapping::evaluateCompletePlan(
          module, *graph, target, proposal, contract, memoryBudget,
          mapping::AnalysisRequirement::Complete);
      if (!evaluation)
        return evaluation.takeError();
      if (!evaluation->plan)
        return evaluation;
      if (instancePtr->schema.empty())
        return evaluation;
      workload::BindingReceipt receipt = instancePtr->receipt;
      if (llvm::Error invalid =
              completeMappingReceipt(space, candidate, instancePtr->schema,
                                     *evaluation->plan, target, receipt)) {
        evaluation->rejection = mapping::Diagnostic{
            mapping::DiagnosticCode::GlobalConstraintFailed,
            "mapping_receipt: " + llvm::toString(std::move(invalid))};
        evaluation->plan.reset();
        return evaluation;
      }
      acceptedReceipts[evaluation->plan->id] = std::move(receipt);
      return evaluation;
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
      if (!found->frontier.diagnostics.empty()) {
        const mapping::Diagnostic &primary =
            found->frontier.diagnostics.front();
        reason += ": " + mapping::stringifyDiagnosticCode(primary.code).str() +
                  ": " + primary.message;
      }
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
      // Persist the receipt the completion evaluator accepted for this exact
      // plan alongside the selected candidate.
      if (auto accepted = acceptedReceipts.find(selected.plan.id);
          accepted != acceptedReceipts.end())
        selected.receipt = std::move(accepted->second);
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
  if (llvm::Error error = measureSelectedCandidates(
          source, space, target, options, instantiate, report))
    return std::move(error);
  if (measuredObjective) {
    report.measuredCohortOnly = true;
    for (auto it = report.ranked.begin(); it != report.ranked.end();) {
      bool available = bool(perf::resolveMetric(it->ranking.metrics,
                                                space.objective.primaryMetric)
                                .value);
      for (const std::string &secondary : space.objective.secondaryMetrics)
        available &=
            bool(perf::resolveMetric(it->ranking.metrics, secondary).value);
      if (!available) {
        it->ranking.rejectionReason =
            "measurement_unavailable: candidate has no value for every "
            "objective metric";
        report.unrankable.push_back(std::move(*it));
        it = report.ranked.erase(it);
      } else {
        report.measuredCohortSize++;
        ++it;
      }
    }
    report.hasMeasuredResult = !report.ranked.empty();
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

} // namespace

llvm::Expected<MappedTuningReport>
runMappedTuningSession(ModuleOp source, const perf::SearchSpace &space,
                       const perf::WorkloadShape &shape,
                       const mapping::MappingTarget &target,
                       const MappedTuningOptions &options) {
  // The legacy entry point is a translation: its contraction facts become the
  // source selection the adapter path realizes, and everything after candidate
  // realization is the shared sequence.
  InstantiateCandidateFn instantiate =
      [source, space, shape](const perf::Candidate &candidate,
                             const CandidateInstantiationOptions &sourceOptions)
      -> llvm::Expected<CandidateInstance> {
    return instantiateCandidate(source, space, candidate, shape, sourceOptions);
  };
  return runMappedTuningSessionImpl(source, space, target, options,
                                    instantiate);
}

llvm::Expected<MappedTuningReport>
runMappedTuningSession(ModuleOp source, const perf::SearchSpace &space,
                       const workload::WorkloadContract &contract,
                       const mapping::MappingTarget &target,
                       const MappedTuningOptions &options) {
  InstantiateCandidateFn instantiate =
      [source, space,
       contract](const perf::Candidate &candidate,
                 const CandidateInstantiationOptions &sourceOptions)
      -> llvm::Expected<CandidateInstance> {
    return instantiateCandidate(source, space, candidate, contract,
                                sourceOptions);
  };
  return runMappedTuningSessionImpl(source, space, target, options,
                                    instantiate);
}

} // namespace mlir::llk::tuning
