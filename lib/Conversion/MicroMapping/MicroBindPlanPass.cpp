//===- MicroBindPlanPass.cpp - `micro-bind-plan` (design §21) -------------===//
//
// Binds one specific plan, named by its stable content-derived id, onto the
// module's kernel. A plan id is a hash of the plan's content, so the only way
// to reproduce it is to re-run the search -- plans are never persisted between
// passes. The search runs deterministically, which is what makes an id from a
// deterministic `--micro-map` reproducible here.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/MicroMapping/MicroMappingPasses.h"
#include "MicroMappingCommon.h"

#include "mlir/Pass/Pass.h"

#include "llvm/Support/Error.h"

#include <memory>
#include <optional>
#include <string>

namespace mlir {
namespace llk {

namespace {

struct MicroBindPlanPass
    : public PassWrapper<MicroBindPlanPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MicroBindPlanPass)

  MicroBindPlanPass() = default;
  MicroBindPlanPass(const MicroBindPlanPass &other) : PassWrapper(other) {}

  Option<std::string> planId{
      *this, "plan-id",
      llvm::cl::desc("Stable id of the plan to bind (required)")};
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
  Option<unsigned> topK{
      *this, "top-k",
      llvm::cl::desc("Maximum number of complete plans to keep"),
      llvm::cl::init(8)};

  StringRef getArgument() const override { return "micro-bind-plan"; }

  StringRef getDescription() const override {
    return "Bind the covering plan with the requested stable id onto a "
           "micro.kernel, e.g. --micro-bind-plan=\"plan-id=12345 "
           "target=x86-avx2 machine=machines/x86-avx2-v2.yaml "
           "layouts=mapping/x86-avx2/layouts.llkmap "
           "rules=mapping/x86-avx2/rules.llkmap "
           "emitters=avx2_vector_add\"";
  }

  MicroMapOptions currentOptions() const {
    MicroMapOptions options;
    options.target = target.getValue();
    options.machinePath = machine.getValue();
    options.layoutPath = layouts.getValue();
    options.rulePath = rules.getValue();
    options.emitterKeys =
        micro_mapping_detail::parseEmitterKeys(emitters.getValue());
    options.topK = topK.getValue();
    return options;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // A plan id is content-derived, so parse it before doing any work: a bad id
    // is a usage error, not a failed search. `parsePlanId` accepts the hex
    // spelling `PlanReport` prints, `0x`-hex, and decimal signed or unsigned --
    // all spellings of the same 64-bit hash (see the rule in
    // MicroMappingCommon.h).
    if (planId.getValue().empty()) {
      module.emitError() << "micro-bind-plan: missing required key 'plan-id'";
      signalPassFailure();
      return;
    }
    std::optional<uint64_t> parsedId =
        micro_mapping_detail::parsePlanId(planId.getValue());
    if (!parsedId) {
      module.emitError()
          << "micro-bind-plan: 'plan-id' must be a 64-bit plan id in bare hex "
             "(the report's spelling, e.g. 0081ef1286442d39), 0x-prefixed hex, "
             "or decimal, got '"
          << planId.getValue() << "'";
      signalPassFailure();
      return;
    }
    uint64_t requestedId = *parsedId;

    llvm::Expected<micro_mapping_detail::MappingRun> run =
        micro_mapping_detail::runMappingSearch(module, "micro-bind-plan",
                                               currentOptions(),
                                               /*forceDeterministic=*/true);
    if (!run) {
      module.emitError() << llvm::toString(run.takeError());
      signalPassFailure();
      return;
    }

    const mapping::CoveringPlan *selected = nullptr;
    for (const mapping::CoveringPlan &plan : run->result.plans) {
      if (plan.id == requestedId) {
        selected = &plan;
        break;
      }
    }
    if (!selected) {
      module.emitError()
          << "micro-bind-plan: no plan in the deterministic search has id "
          << requestedId << " (the search produced " << run->result.plans.size()
          << " plan(s))";
      signalPassFailure();
      return;
    }

    if (llvm::Error error = micro_mapping_detail::bindPlanOntoModule(
            module, *selected, *run->target)) {
      module.emitError() << llvm::toString(std::move(error));
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<Pass> createMicroBindPlanPass() {
  return std::make_unique<MicroBindPlanPass>();
}

std::unique_ptr<Pass>
createMicroBindPlanPass(const MicroBindPlanOptions &options) {
  auto pass = std::make_unique<MicroBindPlanPass>();
  pass->target = options.search.target;
  pass->machine = options.search.machinePath;
  pass->layouts = options.search.layoutPath;
  pass->rules = options.search.rulePath;
  std::string emitters;
  for (const std::string &key : options.search.emitterKeys) {
    if (!emitters.empty())
      emitters += ",";
    emitters += key;
  }
  pass->emitters = emitters;
  pass->topK = options.search.topK;
  pass->planId = std::to_string(options.planId);
  return pass;
}

} // namespace llk
} // namespace mlir
