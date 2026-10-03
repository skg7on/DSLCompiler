//===- MappingTarget.h - A validated target configuration (D4) ------------===//
//
// Part of the target-independent mapping core (epic #67, workstream D4).
//
// A `MappingTarget` is one validated bundle of target facts: the machine
// topology, the layouts the target supports, the rules that map Micro
// operations onto it, and the emitter keys its plugin understands. Generic
// mapping code talks to this interface and never to AVX2 (or Ampere, or NPU)
// specifics.
//
// A target bundle is opaque: a name, typed parameters, and an emitter key.
// Generic code may compare, hash, report, and hand a bundle to the plugin; it
// must not read any field as target semantics (design §14.3).
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_MAPPINGTARGET_H
#define LLK_MAPPING_MAPPINGTARGET_H

#include "LLK/Machine/MachineModel.h"
#include "LLK/Mapping/LayoutConstraints.h"
#include "LLK/Mapping/MappingRules.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>
#include <vector>

namespace mlir::llk::mapping {

/// An opaque target-owned implementation name plus typed parameters. Only the
/// target's emitter interprets `name` and `parameters`.
struct TargetBundle {
  std::string name;
  mlir::DictionaryAttr parameters;
  std::string emitterKey;
};

/// The interface target-independent mapping code uses.
class MappingTarget {
public:
  virtual ~MappingTarget() = default;

  virtual llvm::StringRef name() const = 0;
  virtual const machine::MachineModel &machine() const = 0;
  virtual const LayoutRegistry &layouts() const = 0;
  virtual const RuleRegistry &rules() const = 0;
  virtual bool isKnownEmitter(llvm::StringRef key) const = 0;
};

/// A target built from already-loaded registries. `loadMappingTarget` is the
/// file-backed constructor; this one exists so tests and future plugins can
/// assemble a target in memory.
class FileMappingTarget : public MappingTarget {
public:
  FileMappingTarget(std::string name, machine::MachineModel machine,
                    LayoutRegistry layouts, RuleRegistry rules,
                    std::vector<std::string> emitterKeys);

  llvm::StringRef name() const override { return name_; }
  const machine::MachineModel &machine() const override { return machine_; }
  const LayoutRegistry &layouts() const override { return layouts_; }
  const RuleRegistry &rules() const override { return rules_; }
  bool isKnownEmitter(llvm::StringRef key) const override;

private:
  std::string name_;
  machine::MachineModel machine_;
  LayoutRegistry layouts_;
  RuleRegistry rules_;
  std::vector<std::string> emitterKeys_;
};

/// Loads a target from a machine profile, a layout file, a rule file, and the
/// emitter keys the target plugin understands, then verifies it.
llvm::Expected<std::unique_ptr<MappingTarget>>
loadMappingTarget(llvm::StringRef name, llvm::StringRef machinePath,
                  llvm::StringRef layoutPath, llvm::StringRef rulePath,
                  std::vector<std::string> emitterKeys);

/// Checks a target's rules against its layouts and machine: every referenced
/// layout id exists, every required capability kind is one the machine offers,
/// and every emitter key is declared. Returns the first violation, in rule
/// order, so diagnostics are deterministic.
llvm::Error verifyMappingTarget(const MappingTarget &target);

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_MAPPINGTARGET_H
