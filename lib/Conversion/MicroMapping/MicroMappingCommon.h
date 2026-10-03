//===- MicroMappingCommon.h - Shared mapping-pass plumbing ------*- C++ -*-===//
//
// The two passes in this directory differ only in which plan they select --
// `micro-map` takes the best one, `micro-bind-plan` takes the one a stable id
// names -- so the chain that turns a module plus a target specification into a
// bound plan lives here once.
//
// Everything here is target-neutral: a target is loaded from files and the
// kernel is found by dialect op name, never by a backend branch.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CONVERSION_MICROMAPPING_MICROMAPPINGCOMMON_H
#define LLK_CONVERSION_MICROMAPPING_MICROMAPPINGCOMMON_H

#include "LLK/Conversion/MicroMapping/MicroMappingPasses.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/PlanBinder.h"
#include "LLK/Mapping/WorkloadGraph.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace mlir::llk::micro_mapping_detail {

/// The 2-D rank and element-type string the layout stage instantiates its
/// declarations against, read off the workload graph rather than assumed: the
/// first statically shaped value's rank and element type are what the rules and
/// layouts are solved for. Falling back to `f32` keeps a graph with no shaped
/// value (which cannot produce a layout anyway) from crashing the pass.
inline mapping::LayoutContext
deriveLayoutContext(const mapping::WorkloadGraph &graph) {
  mapping::LayoutContext context;
  context.elementType = "f32";
  for (const mapping::WorkloadValue &value : graph.getValues()) {
    auto shaped = llvm::dyn_cast<mlir::ShapedType>(value.type);
    if (!shaped || !shaped.hasStaticShape())
      continue;
    context.rank = shaped.getRank();
    Type element = shaped.getElementType();
    if (llvm::isa<mlir::Float32Type>(element))
      context.elementType = "f32";
    else if (llvm::isa<mlir::BFloat16Type>(element))
      context.elementType = "bf16";
    else if (llvm::isa<mlir::Float16Type>(element))
      context.elementType = "f16";
    else if (auto integer = llvm::dyn_cast<mlir::IntegerType>(element))
      context.elementType = ("i" + llvm::Twine(integer.getWidth())).str();
    break;
  }
  return context;
}

/// Maps the `mode=` spelling onto the search mode. Unknown spellings are an
/// error, never a silent default.
inline std::optional<mapping::SearchMode>
parseSearchMode(llvm::StringRef text) {
  return llvm::StringSwitch<std::optional<mapping::SearchMode>>(text)
      .Case("deterministic", mapping::SearchMode::Deterministic)
      .Case("beam", mapping::SearchMode::Beam)
      .Case("exact", mapping::SearchMode::Exact)
      .Default(std::nullopt);
}

/// Splits the comma-separated emitter keys, dropping empty entries so a stray
/// trailing comma is not a key.
inline std::vector<std::string> parseEmitterKeys(llvm::StringRef text) {
  std::vector<std::string> keys;
  llvm::SmallVector<llvm::StringRef, 8> parts;
  text.split(parts, ',');
  for (llvm::StringRef part : parts) {
    part = part.trim();
    if (!part.empty())
      keys.push_back(part.str());
  }
  return keys;
}

/// The first `micro.kernel` in the module, or null. Matched by dialect op name
/// so this library does not depend on the Micro dialect's generated classes.
inline Operation *findMicroKernel(ModuleOp module) {
  Operation *kernel = nullptr;
  module.walk([&](Operation *op) {
    if (!kernel && op->getName().getStringRef() == "micro.kernel")
      kernel = op;
  });
  return kernel;
}

/// A target loaded from disk plus the search it produced over one kernel.
struct MappingRun {
  std::unique_ptr<mapping::MappingTarget> target;
  Operation *kernel = nullptr;
  mapping::MappingSearchResult result;
};

/// The five file/label keys every mapping pass must be given. An empty value is
/// a missing key, reported rather than defaulted.
inline llvm::Error requireKey(llvm::StringRef passName, llvm::StringRef key,
                              llvm::StringRef value) {
  if (!value.empty())
    return llvm::Error::success();
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      (passName + ": missing required key '" + key + "'").str());
}

/// The whole chain up to (but not including) plan selection: load the target,
/// find the kernel, extract its workload graph, and search it. `mode` is forced
/// to `deterministic` when `forceDeterministic` is set, which is what makes a
/// content-derived plan id reproducible.
inline llvm::Expected<MappingRun>
runMappingSearch(ModuleOp module, llvm::StringRef passName,
                 const MicroMapOptions &options, bool forceDeterministic) {
  if (llvm::Error error = requireKey(passName, "target", options.target))
    return std::move(error);
  if (llvm::Error error = requireKey(passName, "machine", options.machinePath))
    return std::move(error);
  if (llvm::Error error = requireKey(passName, "layouts", options.layoutPath))
    return std::move(error);
  if (llvm::Error error = requireKey(passName, "rules", options.rulePath))
    return std::move(error);
  if (options.emitterKeys.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        (passName + ": missing required key 'emitters'").str());

  mapping::SearchMode mode = mapping::SearchMode::Beam;
  if (!forceDeterministic) {
    std::optional<mapping::SearchMode> parsed = parseSearchMode(options.mode);
    if (!parsed)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          (passName + ": unknown mode '" + options.mode +
           "' (expected deterministic, beam, or exact)")
              .str());
    mode = *parsed;
  }

  MappingRun run;
  llvm::Expected<std::unique_ptr<mapping::MappingTarget>> target =
      mapping::loadMappingTarget(options.target, options.machinePath,
                                 options.layoutPath, options.rulePath,
                                 options.emitterKeys);
  if (!target)
    return target.takeError();
  run.target = std::move(*target);

  run.kernel = findMicroKernel(module);
  if (!run.kernel)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        (passName + ": the module has no micro.kernel").str());

  llvm::Expected<mapping::WorkloadGraph> graph =
      mapping::extractWorkloadGraph(run.kernel);
  if (!graph)
    return graph.takeError();

  mapping::MappingSearchOptions searchOptions;
  searchOptions.mode = mode;
  searchOptions.topK = options.topK;
  searchOptions.beamWidth = options.beamWidth;
  mapping::CoveringSearch search(*graph, *run.target, *module.getContext(),
                                 deriveLayoutContext(*graph), searchOptions);
  llvm::Expected<mapping::MappingSearchResult> result = search.search();
  if (!result)
    return result.takeError();
  run.result = std::move(*result);
  if (run.result.plans.empty())
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        (passName + ": the search produced no complete plan").str());
  return std::move(run);
}

/// Binds `plan` onto the module. `bindPlan` writes onto a private clone, so the
/// clone's body replaces the module's own to make the mapped IR the pass's
/// output; the module op itself is preserved so the pass manager stays valid.
inline llvm::Error bindPlanOntoModule(ModuleOp module,
                                      const mapping::CoveringPlan &plan,
                                      const mapping::MappingTarget &target) {
  llvm::Expected<mapping::BoundPlan> bound =
      mapping::bindPlan(module, plan, target);
  if (!bound)
    return bound.takeError();
  module.getBodyRegion().takeBody(bound->module->getBodyRegion());
  return llvm::Error::success();
}

} // namespace mlir::llk::micro_mapping_detail

#endif // LLK_CONVERSION_MICROMAPPING_MICROMAPPINGCOMMON_H
