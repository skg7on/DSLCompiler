//===- MappingTarget.cpp - A validated target configuration (D4) ----------===//

#include "LLK/Mapping/MappingTarget.h"

#include "LLK/Machine/MachineModelLoader.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>
#include <utility>

namespace mlir::llk::mapping {

namespace {

using machine::MachineModel;

llvm::Error targetError(const std::string &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

/// True when the machine offers a capability of `kind` for a requirement's
/// role. Executor matching uses the owner-kind rule (design §11.5), so a PE
/// that refines `worker` satisfies a `worker` requirement.
bool machineOffers(const MachineModel &machine, llvm::StringRef role,
                   llvm::StringRef kind) {
  if (role == "executor") {
    return llvm::any_of(machine.executors, [&](const machine::ExecutorNode &e) {
      return machine.ownerMatches(kind, e.id);
    });
  }
  if (role == "compute") {
    return llvm::any_of(machine.computes, [&](const machine::ComputeNode &c) {
      return c.kind == kind;
    });
  }
  if (role == "memory") {
    return llvm::any_of(machine.memories, [&](const machine::MemoryNode &m) {
      return m.kind == kind;
    });
  }
  return false;
}

bool isKnownRole(llvm::StringRef role) {
  return role == "executor" || role == "compute" || role == "memory";
}

/// Resolves a `machine.compute`/`machine.memory` query subject against the
/// machine, for `resolveMachineQueries`. A compute subject is a capability
/// *kind* (`vector_engine`); a memory subject is a node *id* (`sram.0`), which
/// is what the evaluator looks memory facts up by. Returns the diagnostic when
/// the subject is unknown, or nullopt when it resolves.
std::optional<std::string> resolveMachineQuery(const MachineModel &machine,
                                               llvm::StringRef callee,
                                               llvm::StringRef subject) {
  if (callee == "machine.compute") {
    if (llvm::any_of(machine.computes, [&](const machine::ComputeNode &node) {
          return node.kind == subject;
        }))
      return std::nullopt;
    return "unknown compute kind '" + subject.str() + "'";
  }
  if (callee == "machine.memory") {
    if (machine.findMemory(subject))
      return std::nullopt;
    return "unknown memory '" + subject.str() + "'";
  }
  // The grammar knows no other machine query; `isKnownCall` already rejected
  // an unknown callee at parse time.
  return std::nullopt;
}

} // namespace

FileMappingTarget::FileMappingTarget(std::string name, MachineModel machine,
                                     LayoutRegistry layouts, RuleRegistry rules,
                                     std::vector<std::string> emitterKeys,
                                     const LatencyProvider *provider)
    : name_(std::move(name)), machine_(std::move(machine)),
      layouts_(std::move(layouts)), rules_(std::move(rules)),
      emitterKeys_(std::move(emitterKeys)), provider_(provider) {}

bool FileMappingTarget::isKnownEmitter(llvm::StringRef key) const {
  return llvm::is_contained(emitterKeys_, key);
}

llvm::Error verifyMappingTarget(const MappingTarget &target) {
  const MachineModel &machine = target.machine();
  auto resolver = [&](llvm::StringRef callee, llvm::StringRef subject) {
    return resolveMachineQuery(machine, callee, subject);
  };
  for (const RuleDef &rule : target.rules().all()) {
    // Port adequacy: a rule's declared ports must cover every port it
    // references. A port predicate reads the matched node by index, so the
    // rule must declare an input/output port for each index it names; a layout
    // requirement names a port by name (design §14.4, "missing ports"). A rule
    // that declares no port it references would match a boundary it can never
    // wire, so it is rejected here rather than silently producing a candidate
    // with no ports.
    //
    // This is reference-based on purpose. The mapping core has no Micro
    // op-to-arity table (several workload ops are variadic), so a declared
    // port cannot be checked against the op's true arity here; and a blanket
    // "must declare at least one port" would reject live rules that carry an
    // implicit boundary (`avx2.async_copy`). The residual hole is a rule that
    // references no port at all: it loads with an empty boundary.
    size_t declaredInputs = 0;
    for (const RulePort &port : rule.ports)
      if (port.isInput)
        ++declaredInputs;
    size_t declaredOutputs = rule.ports.size() - declaredInputs;
    for (const RulePredicate &predicate : rule.predicates) {
      if (!predicate.directionSet)
        continue;
      size_t declared = predicate.isInput ? declaredInputs : declaredOutputs;
      if (predicate.portIndex >= 0 &&
          static_cast<size_t>(predicate.portIndex) >= declared)
        return targetError("rule '" + rule.id + "': port predicate " +
                           (predicate.isInput ? "input[" : "output[") +
                           std::to_string(predicate.portIndex) +
                           "] names a port the rule does not declare");
    }
    for (const RuleLayoutRequirement &requirement : rule.layoutRequirements) {
      if (!llvm::any_of(rule.ports, [&](const RulePort &port) {
            return port.name == requirement.port;
          }))
        return targetError("rule '" + rule.id +
                           "': layout requirement names undeclared port '" +
                           requirement.port + "'");
      if (!target.layouts().find(requirement.layoutId))
        return targetError("rule '" + rule.id + "': unknown layout '" +
                           requirement.layoutId + "'");
    }
    // A `require` expression may query the machine; an unknown capability
    // subject (a compute kind or memory id) is a load-time error, not a
    // failure deferred to the first match.
    for (const ExprPtr &constraint : rule.constraints) {
      std::string message = resolveMachineQueries(*constraint, resolver);
      if (!message.empty())
        return targetError("rule '" + rule.id + "': " + message);
    }
    for (const KindRequirement &requirement : rule.kindRequirements) {
      if (!isKnownRole(requirement.role))
        return targetError("rule '" + rule.id +
                           "': unknown requirement role '" + requirement.role +
                           "'");
      if (!machineOffers(machine, requirement.role, requirement.kind))
        return targetError("rule '" + rule.id + "': machine offers no " +
                           requirement.role + " of kind '" + requirement.kind +
                           "'");
    }
    if (!target.isKnownEmitter(rule.emitter))
      return targetError("rule '" + rule.id + "': unknown emitter '" +
                         rule.emitter + "'");
  }
  // Layout declarations carry machine queries too; resolve them against the
  // same machine so a bad subject is a load-time diagnostic there as well.
  for (const LayoutDef &layout : target.layouts().all()) {
    for (const ExprPtr &constraint : layout.constraints) {
      std::string message = resolveMachineQueries(*constraint, resolver);
      if (!message.empty())
        return targetError("layout '" + layout.id + "': " + message);
    }
  }
  return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<MappingTarget>>
loadMappingTarget(llvm::StringRef name, llvm::StringRef machinePath,
                  llvm::StringRef layoutPath, llvm::StringRef rulePath,
                  std::vector<std::string> emitterKeys) {
  llvm::Expected<MachineModel> machine = machine::loadMachineModel(machinePath);
  if (!machine)
    return machine.takeError();
  llvm::Expected<LayoutRegistry> layouts = loadLayoutFile(layoutPath);
  if (!layouts)
    return layouts.takeError();
  llvm::Expected<RuleRegistry> rules = loadRuleFile(rulePath);
  if (!rules)
    return rules.takeError();

  auto target = std::make_unique<FileMappingTarget>(
      name.str(), std::move(*machine), std::move(*layouts), std::move(*rules),
      std::move(emitterKeys));
  if (llvm::Error error = verifyMappingTarget(*target))
    return std::move(error);
  return std::unique_ptr<MappingTarget>(std::move(target));
}

} // namespace mlir::llk::mapping
