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
#include "LLK/Mapping/MappingLowering.h"
#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/MappingRules.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>
#include <vector>

namespace mlir {
class Operation;
class RewriterBase;
} // namespace mlir

namespace mlir::llk::mapping {

/// A target plugin's code emitter. It is opaque to the mapping core: generic
/// code selects a bundle and hands it to the plugin emitter, which decides
/// whether the bundle is complete enough to lower and then lowers it (design
/// §18.3, phase 3).
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

  /// Lowers the covered operation group according to `bundle`.
  ///
  /// `coveredOps` are the operations the selected candidate covers, and the
  /// emitter replaces them with the implementation the bundle names. It must
  /// validate the whole bundle and the selected port/resource/layout contract
  /// *before* rewriting anything, so a rejected bundle leaves `coveredOps`
  /// untouched rather than half-rewritten.
  ///
  /// The default rejects. Declaring an emitter key is not implementing one,
  /// and a lowering hook that quietly did nothing would let a rule file
  /// impersonate a backend: a target opts in by overriding this. That is what
  /// keeps "the bundle is well-formed" and "this target can execute it"
  /// separate questions (§18.3, §22.3).
  virtual llvm::Error lower(llvm::ArrayRef<mlir::Operation *> coveredOps,
                            const TargetBundle &bundle,
                            const TargetLoweringContext &context,
                            mlir::RewriterBase &rewriter) const {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "target_lowering_unsupported: emitter '" +
                                       key().str() +
                                       "' has no lowering implementation");
  }
};

/// The part of an emitter that is the same for every target: it handles one
/// key out of the set its target declared, and it checks that a bundle is
/// well-formed *as an opaque value* -- the key is declared and matches, and
/// every parameter is the integer/string shape the rule parser types bundle
/// parameters with, never a meaning.
///
/// A target's real emitters derive from this and override `lower`; the
/// default emitter a `FileMappingTarget` hands out is this class unchanged, so
/// a configuration-only target keeps rejecting lowering explicitly.
class DeclaredTargetEmitter : public TargetEmitter {
public:
  DeclaredTargetEmitter(std::string key, std::vector<std::string> declaredKeys);

  llvm::StringRef key() const override { return key_; }

  llvm::Error verify(const TargetBundle &bundle) const override;

private:
  std::string key_;
  std::vector<std::string> declaredKeys_;
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
  /// caller passes the key a bundle's `emitterKey` names. This is the one
  /// factory a target must implement.
  virtual std::unique_ptr<TargetEmitter>
  createEmitter(llvm::StringRef key) const = 0;

  /// A target's single default emitter, if it has one (design §19). The base
  /// returns null: a target that declares several keys has no single default,
  /// so callers reach each key through `createEmitter(key)` and a target opts
  /// in to a default only when it genuinely has a principal emitter. Because
  /// this is non-pure, an out-of-tree target need implement only the keyed
  /// factory. The default `FileMappingTarget` supplies is declaration-order
  /// dependent (its first declared key), so it is meaningful for a
  /// single-key target and a convenience otherwise -- prefer the keyed form.
  virtual std::unique_ptr<TargetEmitter> createEmitter() const {
    return nullptr;
  }

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
