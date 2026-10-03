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
#include "LLK/Mapping/LatencyProvider.h"
#include "LLK/Mapping/LayoutConstraints.h"
#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/MappingRules.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>
#include <vector>

namespace mlir::llk::mapping {

/// A target plugin's code emitter. It is opaque to the mapping core: generic
/// code selects a bundle and hands it to the plugin emitter, which decides
/// whether the bundle is complete enough to lower (design §18.3, phase 3). The
/// interface deliberately stops at that boundary -- it exposes no lowering or
/// LIR API, because emission to machine code is a plugin concern.
///
/// An emitter instance handles exactly one emitter key, the one it was created
/// for. `verify` is pure and may be called repeatedly.
class TargetEmitter {
public:
  virtual ~TargetEmitter() = default;

  /// The emitter key this emitter handles. Always one the owning target
  /// declares (`MappingTarget::isKnownEmitter`).
  virtual llvm::StringRef key() const = 0;

  /// Verifies `bundle` is complete for this emitter before lowering. Returns
  /// an error when the bundle names an emitter this target does not declare,
  /// when the bundle targets a different emitter key than the one this emitter
  /// handles, or when a bundle parameter is not the integer/string shape the
  /// plugin contract permits.
  virtual llvm::Error verify(const TargetBundle &bundle) const = 0;
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

  /// The emitter for `key`, or null when the target does not declare that key.
  /// A target that declares several emitter keys exposes each one here; the
  /// caller passes the key a bundle's `emitterKey` names.
  virtual std::unique_ptr<TargetEmitter>
  createEmitter(llvm::StringRef key) const = 0;

  /// The target's default emitter -- the one for its first declared key -- or
  /// null for a target that declares no emitter keys (design §19). Kept for
  /// the common single-entry case; a multi-key target is reached through
  /// `createEmitter(key)`.
  virtual std::unique_ptr<TargetEmitter> createEmitter() const = 0;

  /// Optional measured or calibrated latencies. Null means the target has
  /// none -- and a provider with no entry for a signature is the same as null
  /// *for that lookup*: the static estimate stands and legality is unaffected
  /// (design §17.3). Defaulting to null keeps a target that predates
  /// measurement working unchanged.
  virtual const LatencyProvider *latencyProvider() const { return nullptr; }
};

/// A target built from already-loaded registries. `loadMappingTarget` is the
/// file-backed constructor; this one exists so tests and future plugins can
/// assemble a target in memory.
class FileMappingTarget : public MappingTarget {
public:
  /// `provider` is borrowed, not owned: the caller keeps it alive for as long
  /// as the target is used.
  FileMappingTarget(std::string name, machine::MachineModel machine,
                    LayoutRegistry layouts, RuleRegistry rules,
                    std::vector<std::string> emitterKeys,
                    const LatencyProvider *provider = nullptr);

  llvm::StringRef name() const override { return name_; }
  const machine::MachineModel &machine() const override { return machine_; }
  const LayoutRegistry &layouts() const override { return layouts_; }
  const RuleRegistry &rules() const override { return rules_; }
  bool isKnownEmitter(llvm::StringRef key) const override;
  std::unique_ptr<TargetEmitter>
  createEmitter(llvm::StringRef key) const override;
  std::unique_ptr<TargetEmitter> createEmitter() const override;
  const LatencyProvider *latencyProvider() const override { return provider_; }

private:
  std::string name_;
  machine::MachineModel machine_;
  LayoutRegistry layouts_;
  RuleRegistry rules_;
  std::vector<std::string> emitterKeys_;
  const LatencyProvider *provider_ = nullptr;
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
