//===- MappingTarget.cpp - A validated target configuration (D4) ----------===//

#include "LLK/Mapping/MappingTarget.h"

#include "LLK/Machine/MachineModelLoader.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"

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

} // namespace

FileMappingTarget::FileMappingTarget(std::string name, MachineModel machine,
                                     LayoutRegistry layouts, RuleRegistry rules,
                                     std::vector<std::string> emitterKeys)
    : name_(std::move(name)), machine_(std::move(machine)),
      layouts_(std::move(layouts)), rules_(std::move(rules)),
      emitterKeys_(std::move(emitterKeys)) {}

bool FileMappingTarget::isKnownEmitter(llvm::StringRef key) const {
  return llvm::is_contained(emitterKeys_, key);
}

llvm::Error verifyMappingTarget(const MappingTarget &target) {
  const MachineModel &machine = target.machine();
  for (const RuleDef &rule : target.rules().all()) {
    for (const RuleLayoutRequirement &requirement : rule.layoutRequirements) {
      if (!target.layouts().find(requirement.layoutId))
        return targetError("rule '" + rule.id + "': unknown layout '" +
                           requirement.layoutId + "'");
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
