//===- Placement.cpp - Placement and connection synthesis (D5) -----------===//

#include "LLK/Mapping/Placement.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"

#include <string>
#include <utility>
#include <vector>

namespace mlir::llk::mapping {

namespace {

using machine::ComputeNode;
using machine::ExecutorNode;
using machine::MachineModel;
using machine::MemoryNode;

llvm::Error placementError(llvm::StringRef message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

/// Compute node ids attached to `executor`, sorted for comparison.
std::vector<std::string> attachedComputes(const MachineModel &machine,
                                          const ExecutorNode &executor) {
  std::vector<std::string> ids;
  for (const ComputeNode *node : machine.computesFor(executor.id))
    ids.push_back(node->id);
  llvm::sort(ids);
  return ids;
}

/// Memory node ids visible from `executor`, sorted for comparison.
std::vector<std::string> visibleMemories(const MachineModel &machine,
                                         const ExecutorNode &executor) {
  std::vector<std::string> ids;
  for (const MemoryNode &node : machine.memories)
    if (machine.isVisible(node.id, executor.id))
      ids.push_back(node.id);
  llvm::sort(ids);
  return ids;
}

/// Two executors are interchangeable when swapping them cannot change any
/// binding: same kind, same parent, and exactly the same attached compute and
/// visible memory nodes. Symmetry reduction is only sound under this rule.
bool interchangeable(const MachineModel &machine, const ExecutorNode &lhs,
                     const ExecutorNode &rhs) {
  return lhs.kind == rhs.kind && lhs.parent == rhs.parent &&
         attachedComputes(machine, lhs) == attachedComputes(machine, rhs) &&
         visibleMemories(machine, lhs) == visibleMemories(machine, rhs);
}

std::vector<const ExecutorNode *>
reduceSymmetric(const MachineModel &machine,
                const std::vector<const ExecutorNode *> &executors) {
  std::vector<const ExecutorNode *> representatives;
  for (const ExecutorNode *candidate : executors) {
    bool duplicate = false;
    for (const ExecutorNode *kept : representatives) {
      if (interchangeable(machine, *kept, *candidate)) {
        duplicate = true;
        break;
      }
    }
    if (!duplicate)
      representatives.push_back(candidate);
  }
  return representatives;
}

} // namespace

llvm::Expected<std::vector<CandidateInstance>>
enumeratePlacements(const MappingCandidate &candidate,
                    const MappingTarget &target, mlir::MLIRContext &context,
                    const LayoutContext &layoutContext,
                    const PlacementOptions &options) {
  const MachineModel &machine = target.machine();

  // A layout requirement that cannot solve makes the candidate unplaceable,
  // independently of which executor would run it.
  for (const LayoutRequirement &requirement : candidate.layoutRequirements) {
    const LayoutDef *def = target.layouts().find(requirement.layoutClass);
    if (!def)
      return placementError("placement: unknown layout '" +
                            requirement.layoutClass + "'");
    llvm::Expected<LayoutSolveResult> solved =
        solveLayout(*def, machine, context, layoutContext);
    if (!solved)
      return solved.takeError();
    if (solved->solutions.empty())
      return std::vector<CandidateInstance>{};
  }

  std::vector<const ExecutorNode *> executors;
  for (const ExecutorNode &executor : machine.executors) {
    bool matches = true;
    for (const ExecutorRequirement &requirement :
         candidate.executorRequirements) {
      if (!machine.ownerMatches(requirement.capability, executor.id)) {
        matches = false;
        break;
      }
    }
    if (matches)
      executors.push_back(&executor);
  }
  if (options.reduceSymmetry)
    executors = reduceSymmetric(machine, executors);

  std::vector<CandidateInstance> instances;
  for (const ExecutorNode *executor : executors) {
    CandidateInstance instance;
    instance.candidate = candidate.id;
    instance.executorBindings["executor"] = executor->id;

    // Compute attachments: the first attached capability of each required kind.
    bool computesOk = true;
    for (const ComputeRequirement &requirement :
         candidate.computeRequirements) {
      const ComputeNode *match = nullptr;
      for (const ComputeNode *node : machine.computesFor(executor->id)) {
        if (node->kind == requirement.kind) {
          match = node;
          break;
        }
      }
      if (!match) {
        computesOk = false;
        break;
      }
      instance.computeBindings[requirement.kind] = match->id;
    }
    if (!computesOk)
      continue;

    // Memory attachments: the first visible memory of each required kind.
    bool memoriesOk = true;
    for (const MemoryRequirement &requirement : candidate.memoryRequirements) {
      const MemoryNode *match = nullptr;
      for (const MemoryNode &node : machine.memories) {
        if (node.kind == requirement.kind &&
            machine.isVisible(node.id, executor->id)) {
          match = &node;
          break;
        }
      }
      if (!match) {
        memoriesOk = false;
        break;
      }
      instance.memoryBindings[requirement.kind] = match->id;
    }
    if (!memoriesOk)
      continue;

    for (const LayoutRequirement &requirement : candidate.layoutRequirements)
      instance.layoutBindings[requirement.layoutClass] =
          requirement.layoutClass;

    instance.resourceUsage.executorSlots = 1;
    for (const MemoryRequirement &requirement : candidate.memoryRequirements)
      instance.resourceUsage.memoryBytes[requirement.kind] =
          requirement.minBytes;

    instance.localCost = candidate.lowerBound;
    instance.id = computeInstanceId(instance);
    instances.push_back(std::move(instance));
    if (instances.size() >= options.maxInstances)
      break;
  }
  return instances;
}

} // namespace mlir::llk::mapping
