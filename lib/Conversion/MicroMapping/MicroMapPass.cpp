//===- MicroMapPass.cpp - `micro-map`: bind the best plan (design §21)
//-----===//
//
// A compiler front door for the mapping engine: load a target by path, extract
// the module's workload graph, search it, and bind the best covering plan onto
// the kernel. The pass is target-neutral -- it never names a backend.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/MicroMapping/MicroMappingPasses.h"
#include "MicroMappingCommon.h"

#include "mlir/Pass/Pass.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <string>

namespace mlir {
namespace llk {

namespace {

struct MicroMapPass
    : public PassWrapper<MicroMapPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MicroMapPass)

  MicroMapPass() = default;
  MicroMapPass(const MicroMapPass &other) : PassWrapper(other) {}

  Option<std::string> target{
      *this, "target",
      llvm::cl::desc("Opaque target label recorded on the selected plan")};
  Option<std::string> machine{*this, "machine",
                              llvm::cl::desc("Machine profile YAML path")};
  Option<std::string> layouts{
      *this, "layouts", llvm::cl::desc("Layout declaration (.llkmap) path")};
  Option<std::string> rules{*this, "rules",
                            llvm::cl::desc("Rule declaration (.llkmap) path")};
  Option<std::string> emitters{
      *this, "emitters",
      llvm::cl::desc("Comma-separated emitter keys the target understands")};
  Option<std::string> mode{
      *this, "mode",
      llvm::cl::desc("Search mode: deterministic, beam, or exact"),
      llvm::cl::init("beam")};
  Option<unsigned> topK{
      *this, "top-k",
      llvm::cl::desc("Maximum number of complete plans to keep"),
      llvm::cl::init(8)};
  Option<unsigned> beamWidth{*this, "beam-width",
                             llvm::cl::desc("Beam mode's frontier width"),
                             llvm::cl::init(64)};
  Option<std::string> report{
      *this, "report",
      llvm::cl::desc("Write the versioned JSON plan report (design §22.2) to "
                     "this path; the report never changes the IR")};

  StringRef getArgument() const override { return "micro-map"; }

  StringRef getDescription() const override {
    return "Search a micro.kernel for a covering plan and bind the best one to "
           "it (e.g. --micro-map=\"target=x86-avx2 "
           "machine=machines/x86-avx2-v2.yaml "
           "layouts=mapping/x86-avx2/layouts.llkmap "
           "rules=mapping/x86-avx2/rules.llkmap "
           "emitters=avx2_vector_add mode=beam top-k=8\")";
  }

  MicroMapOptions currentOptions() const {
    MicroMapOptions options;
    options.target = target.getValue();
    options.machinePath = machine.getValue();
    options.layoutPath = layouts.getValue();
    options.rulePath = rules.getValue();
    options.emitterKeys =
        micro_mapping_detail::parseEmitterKeys(emitters.getValue());
    options.mode = mode.getValue();
    options.topK = topK.getValue();
    options.beamWidth = beamWidth.getValue();
    options.reportPath = report.getValue();
    return options;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MicroMapOptions options = currentOptions();
    llvm::Expected<micro_mapping_detail::MappingRun> run =
        micro_mapping_detail::runMappingSearch(module, "micro-map", options);
    if (!run) {
      module.emitError() << llvm::toString(run.takeError());
      signalPassFailure();
      return;
    }
    // The report describes the input module the search ran over, so it is
    // written before binding, while the module is still unmodified. It is
    // metadata only -- the IR below is bound exactly as it would be without a
    // report.
    if (!options.reportPath.empty()) {
      uint64_t moduleHash = micro_mapping_detail::computeModuleHash(module);
      if (llvm::Error error = mapping::writePlanReportFile(
              options.reportPath, run->result, run->target->machine(),
              *run->target, run->searchOptions, moduleHash)) {
        module.emitError() << llvm::toString(std::move(error));
        signalPassFailure();
        return;
      }
    }
    if (llvm::Error error = micro_mapping_detail::bindPlanOntoModule(
            module, run->result.plans.front(), *run->target)) {
      module.emitError() << llvm::toString(std::move(error));
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<Pass> createMicroMapPass() {
  return std::make_unique<MicroMapPass>();
}

std::unique_ptr<Pass> createMicroMapPass(const MicroMapOptions &options) {
  auto pass = std::make_unique<MicroMapPass>();
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
  pass->mode = options.mode;
  pass->topK = options.topK;
  pass->beamWidth = options.beamWidth;
  pass->report = options.reportPath;
  return pass;
}

} // namespace llk
} // namespace mlir
