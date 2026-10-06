//===- MappingHelpers.h - Shared, mechanical mapping helpers --------------===//
//
// Small, target-neutral helpers used by more than one component of the mapping
// subsystem. They live here rather than duplicated per translation unit so the
// copies cannot drift before later work adds callers: the rule-port resolver
// must agree with the positional wiring generation uses, the memory-space
// attribute spelling must match the Micro dialect's `#micro.memory<...>` form,
// and the transparent-logical-op walk must resolve the same indirection the
// verifier and the materializer see.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_MAPPINGHELPERS_H
#define LLK_MAPPING_MAPPINGHELPERS_H

#include "LLK/Mapping/MappingRules.h"
#include "LLK/Mapping/WorkloadGraph.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir::llk::mapping {

/// The `#micro.memory<kind>` attribute a machine memory kind names, or a null
/// attribute when the kind is not a Micro memory space.
inline mlir::Attribute memoryAttrFor(mlir::MLIRContext *context,
                                     llvm::StringRef kind) {
  return mlir::parseAttribute(("#micro.memory<" + kind + ">").str(), context);
}

/// Resolves `value` through the transparent logical ops (`micro.tile_view`,
/// `micro.tile_partition`) to the operation-produced value it ultimately reads.
/// A consumer reading a movement through a view still reads the movement.
inline mlir::Value resolveThroughTransparentOps(mlir::Value value) {
  while (mlir::Operation *defining = value.getDefiningOp()) {
    llvm::StringRef name = defining->getName().getStringRef();
    if ((name == "micro.tile_view" || name == "micro.tile_partition") &&
        defining->getNumOperands() > 0) {
      value = defining->getOperand(0);
      continue;
    }
    break;
  }
  return value;
}

/// The occurrence a rule port resolves to on `node`, mirroring the positional
/// wiring generation uses (inputs then outputs, in declaration order).
inline std::optional<PortRef> portRefForRulePort(const RuleDef &rule,
                                                 const WorkloadNode &node,
                                                 const RulePort &subject) {
  size_t inputIndex = 0;
  size_t outputIndex = 0;
  for (const RulePort &port : rule.ports) {
    size_t &index = port.isInput ? inputIndex : outputIndex;
    if (port.name == subject.name && port.isInput == subject.isInput) {
      llvm::ArrayRef<WorkloadPort> ports =
          port.isInput ? node.inputs : node.outputs;
      if (index < ports.size())
        return PortRef{node.id,
                       port.isInput ? PortDirection::Input
                                    : PortDirection::Output,
                       static_cast<uint32_t>(index)};
      return std::nullopt;
    }
    ++index;
  }
  return std::nullopt;
}

/// The rule's port occurrence for `portName`, mirroring the positional wiring
/// generation uses (inputs then outputs). The first declared port with that
/// name wins, whichever direction it is.
inline std::optional<PortRef> portRefForRulePort(const RuleDef &rule,
                                                 const WorkloadNode &node,
                                                 llvm::StringRef portName) {
  size_t inputIndex = 0;
  size_t outputIndex = 0;
  for (const RulePort &port : rule.ports) {
    if (port.isInput) {
      if (port.name == portName && inputIndex < node.inputs.size())
        return PortRef{node.id, PortDirection::Input,
                       static_cast<uint32_t>(inputIndex)};
      ++inputIndex;
    } else {
      if (port.name == portName && outputIndex < node.outputs.size())
        return PortRef{node.id, PortDirection::Output,
                       static_cast<uint32_t>(outputIndex)};
      ++outputIndex;
    }
  }
  return std::nullopt;
}

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_MAPPINGHELPERS_H
