//===- MappedCompilation.h - Compile a selected mapping plan (C5) ---------===//
//
// Part of the mapping-integration layer (epic #67, stage C5).
//
// This is the one path from "a plan the search selected" to "code that runs".
// The compiler and the tuner both take it, so a schedule's provenance does not
// depend on which tool asked for it:
//
//   bind the plan -> verify the mapped IR -> let the target's emitters rewrite
//   what they implement -> lower to Linalg, bufferize, roll into loops ->
//   compile an executable
//
// The two claims the result keeps apart are the ones that are easy to confuse:
// an operation a *selected emitter* rewrote is selected-bundle execution, and
// an operation the reference bridge carried is reference execution. They are
// reported separately because only the first is evidence about a target.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CONVERSION_MAPPEDCOMPILATION_H
#define LLK_CONVERSION_MAPPEDCOMPILATION_H

#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Runtime/MappedExecutable.h"
#include "LLK/Runtime/MappedJitOptions.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace llk {

/// How far a mapped compilation is taken.
///
/// The earlier stops are the inspection points: what the plan bound, and what
/// the target made of it. They are not a debugging convenience -- a caller that
/// wants the mapped Micro-IR has no business running a JIT to get it.
enum class MappedStop {
  /// The selected plan is bound and verified; the IR is still Micro.
  MappedMicro,
  /// The target's own emitters have rewritten the operations they implement.
  TargetLowered,
  /// Bufferized and rolled into loops -- Linalg, SCF and memref, ready for a
  /// backend but with no calling convention yet.
  Lowered,
  /// Compiled: the calling convention is applied and the code is in memory.
  Executable,
};

enum class MappedBackend { Reference, SelectedTarget };

struct MappedCompileOptions {
  /// The kernel symbol to compile. Empty means the module's single
  /// `micro.kernel`, which is what the binder already requires.
  std::string entrySymbol;
  /// Whether a decision the binder cannot materialize is an error. A caller
  /// that will run the result needs this on: a partial plan is honest about
  /// what it left out, and must not be mistaken for executable code.
  bool requireExecutable = true;
  MappedStop stop = MappedStop::Executable;
  MappedBackend backend = MappedBackend::Reference;
  /// Optional observer for the exact LLVM input handed to the mapped JIT.
  MappedJitEvidenceSink evidenceSink;
};

/// What a mapped compilation produced.
struct MappedCompilation {
  /// The compiled module. Present for every stop before `Executable`; a
  /// successful `Executable` compilation hands the module to the JIT, so this
  /// is empty there.
  mlir::OwningOpRef<mlir::ModuleOp> module;
  MappedStop stopped = MappedStop::MappedMicro;
  /// Set only when the compilation reached `Executable`.
  std::unique_ptr<MappedExecutable> executable;

  /// Selected operations a target emitter rewrote.
  unsigned targetLowered = 0;
  /// Selected operations whose emitter only verifies them, so the reference
  /// bridge carried them. Keeping this count next to the other is the point:
  /// a run that is entirely reference execution has said nothing about the
  /// target's code generation.
  unsigned referenceLowered = 0;
  /// Selected instance groups that passed bundle and plan-context validation.
  unsigned selectedGroupsVerified = 0;
  /// Selected groups lowered by a target-owned backend emitter.
  unsigned backendGroupsRealized = 0;
  /// Selected groups left to the reference bridge.
  unsigned referenceGroupsLowered = 0;
  /// Declared ISA for a selected backend, retained for every reached stop.
  std::optional<mlir::llk::mapping::TargetCodegenRequirements>
      codegenRequirements;
  uint64_t planId = 0;
  uint64_t machineHash = 0;
  std::string targetName;
  /// Compiler, target, math mode, and host/ISA contract identity.
  std::string executionIdentity;
};

/// Runs the shared mapped-compilation sequence over `source`.
///
/// `source` is not modified: binding clones it, exactly as `bindPlan` does.
/// `plan` is a plan the caller's search produced -- this does not search, so
/// the caller owns the schedule-instantiation step and this owns everything
/// after it.
///
/// Fails when the plan cannot be bound under `requireExecutable`, when the
/// mapped IR does not verify, when the selected emitters reject their own
/// bundles, or when the module cannot be lowered and compiled.
llvm::Expected<MappedCompilation>
compileMappedKernel(mlir::ModuleOp source,
                    const mlir::llk::mapping::MappingTarget &target,
                    const mlir::llk::mapping::CoveringPlan &plan,
                    const MappedCompileOptions &options);

/// Compiles concrete Micro-IR that carries no mapping plan.
///
/// A bound tuning candidate is exactly that: the search space already selected
/// its schedule, so there is no plan to bind and no target to verify against.
/// The sequence from the lowering on is the same one -- bufferize, roll into
/// loops, apply the calling convention, compile -- and sharing it is what keeps
/// a measured candidate and a compiled mapped kernel the same kind of artifact.
/// The first two stops are unreachable here and are refused rather than
/// reported as reached.
llvm::Expected<MappedCompilation>
compileConcreteMicroKernel(mlir::ModuleOp source,
                           const MappedCompileOptions &options);

} // namespace llk

#endif // LLK_CONVERSION_MAPPEDCOMPILATION_H
