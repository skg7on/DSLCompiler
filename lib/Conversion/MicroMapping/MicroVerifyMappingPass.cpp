//===- MicroVerifyMappingPass.cpp - `micro-verify-mapping` (design §21)
//----===//
//
// Phase 2 of the layered verification (design §18.3): an already-mapped
// Micro-IR module is checked against a target, so the ids the binder recorded
// are *resolved* rather than trusted. Phase 1 -- the dialect verifier -- knows
// only that `micro.plan` and `micro.mapping` are well-formed attributes; it
// cannot know whether the rule id `avx2.vector_add` names a rule the target
// declares, whether `worker.0` exists, whether a memory is visible from the
// executor bound to it, whether the route is joined by links, or whether an
// emitter key is one the target understands. Phase 3 -- the target emitter --
// is where a resolved bundle is finally interpreted.
//
// The pass loads the target from disk exactly as `micro-map` does (same five
// keys) and hands the module to `verifyMappedMicroIR`, which returns the first
// violation as a §22.3-coded diagnostic. It never mutates the module: a
// successful verification leaves the input IR byte-identical, so it can be
// dropped into a pipeline or run over a checked-in mapped kernel either way.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/MicroMapping/MicroMappingPasses.h"
#include "MicroMappingCommon.h"

#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/PlanBinder.h"

#include "mlir/Pass/Pass.h"

#include "llvm/Support/Error.h"

#include <memory>
#include <string>

namespace mlir {
namespace llk {

namespace {

struct MicroVerifyMappingPass
    : public PassWrapper<MicroVerifyMappingPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MicroVerifyMappingPass)

  MicroVerifyMappingPass() = default;
  MicroVerifyMappingPass(const MicroVerifyMappingPass &other)
      : PassWrapper(other) {}

  Option<std::string> target{
      *this, "target",
      llvm::cl::desc("Opaque target label (required, as for micro-map)")};
  Option<std::string> machine{*this, "machine",
                              llvm::cl::desc("Machine profile YAML path")};
  Option<std::string> layouts{
      *this, "layouts", llvm::cl::desc("Layout declaration (.llkmap) path")};
  Option<std::string> rules{*this, "rules",
                            llvm::cl::desc("Rule declaration (.llkmap) path")};
  Option<std::string> emitters{
      *this, "emitters",
      llvm::cl::desc("Comma-separated emitter keys the target understands")};

  StringRef getArgument() const override { return "micro-verify-mapping"; }

  StringRef getDescription() const override {
    return "Verify already-mapped Micro-IR against a target: resolve every "
           "rule, "
           "executor, memory, layout, route, and emitter id recorded on the "
           "kernel, and fail with a stable diagnostic on the first violation "
           "(design §18.3, phase 2). The module is never modified, e.g. "
           "--micro-verify-mapping=\"target=x86-avx2 "
           "machine=machines/x86-avx2-v2.yaml "
           "layouts=mapping/x86-avx2/layouts.llkmap "
           "rules=mapping/x86-avx2/rules.llkmap "
           "emitters=avx2_vector_add,avx2_mma\"";
  }

  MicroVerifyMappingOptions currentOptions() const {
    MicroVerifyMappingOptions options;
    options.target = target.getValue();
    options.machinePath = machine.getValue();
    options.layoutPath = layouts.getValue();
    options.rulePath = rules.getValue();
    options.emitterKeys =
        micro_mapping_detail::parseEmitterKeys(emitters.getValue());
    return options;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    constexpr llvm::StringLiteral passName = "micro-verify-mapping";
    MicroVerifyMappingOptions options = currentOptions();

    // The same five required keys `micro-map` takes: resolving rule, layout,
    // and emitter ids needs their registries, and the machine needs a profile.
    // A missing key is a usage error, reported rather than defaulted.
    if (llvm::Error error = micro_mapping_detail::requireKey(passName, "target",
                                                             options.target))
      return fail(module, std::move(error));
    if (llvm::Error error = micro_mapping_detail::requireKey(
            passName, "machine", options.machinePath))
      return fail(module, std::move(error));
    if (llvm::Error error = micro_mapping_detail::requireKey(
            passName, "layouts", options.layoutPath))
      return fail(module, std::move(error));
    if (llvm::Error error = micro_mapping_detail::requireKey(passName, "rules",
                                                             options.rulePath))
      return fail(module, std::move(error));
    if (options.emitterKeys.empty())
      return fail(module,
                  llvm::createStringError(llvm::inconvertibleErrorCode(),
                                          (passName + ": missing required key "
                                                      "'emitters'")
                                              .str()));

    // Loading the target validates it (`verifyMappingTarget`), so a broken
    // target file fails here, before any kernel is inspected.
    llvm::Expected<std::unique_ptr<mapping::MappingTarget>> target =
        mapping::loadMappingTarget(options.target, options.machinePath,
                                   options.layoutPath, options.rulePath,
                                   options.emitterKeys);
    if (!target)
      return fail(module, target.takeError());

    // The pass verifies *mapped* Micro-IR, so a module with no `micro.kernel`
    // has nothing to verify. That is a failure, not a vacuous success: a
    // silent pass on the wrong file would look exactly like a verified one
    // (the same reasoning that makes `micro-map` require a kernel).
    if (!micro_mapping_detail::findMicroKernel(module))
      return fail(module,
                  llvm::createStringError(
                      llvm::inconvertibleErrorCode(),
                      (passName + ": the module has no micro.kernel").str()));

    if (llvm::Error error = mapping::verifyMappedMicroIR(module, **target))
      return fail(module, std::move(error));
    // Nothing to do on success: phase 2 reports, it does not rewrite.
  }

private:
  void fail(ModuleOp module, llvm::Error error) {
    module.emitError() << llvm::toString(std::move(error));
    signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createMicroVerifyMappingPass() {
  return std::make_unique<MicroVerifyMappingPass>();
}

std::unique_ptr<Pass>
createMicroVerifyMappingPass(const MicroVerifyMappingOptions &options) {
  auto pass = std::make_unique<MicroVerifyMappingPass>();
  pass->target = options.target;
  pass->machine = options.machinePath;
  pass->layouts = options.layoutPath;
  pass->rules = options.rulePath;
  std::string emitters;
  for (const std::string &key : options.emitterKeys) {
    if (!emitters.empty())
      emitters += ",";
    emitters += key;
  }
  pass->emitters = emitters;
  return pass;
}

} // namespace llk
} // namespace mlir
