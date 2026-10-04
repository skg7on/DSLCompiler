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
#include "LLK/Conversion/MicroMapping/SearchBindingLoader.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/PlanBinder.h"
#include "LLK/Mapping/PlanReport.h"
#include "LLK/Mapping/StableHash.h"
#include "LLK/Mapping/WorkloadGraph.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

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

/// Parses a 64-bit plan id in every spelling the project emits or accepts.
///
/// A plan id is an unsigned 64-bit content hash. `PlanReport` prints it as bare
/// lowercase hex (16 digits, from `hexId`), so that spelling -- the one the
/// documented report -> bind workflow copies -- must be accepted verbatim. The
/// other accepted spellings denote the same hash: `0x`/`0X`-prefixed hex, and
/// decimal read as either unsigned or signed (a hash with the high bit set is
/// negative when read as a signed i64, and both spellings must round-trip).
///
/// Spelling rule, deterministic and tested. In order:
///   1. a `0x`/`0X` prefix forces hex;
///   2. otherwise exactly 16 characters, all hex digits, is hex -- this is
///      `hexId`'s `%016llx` output, so the report's token round-trips even when
///      all 16 digits happen to be decimal;
///   3. otherwise a string made only of hex digits that contains at least one
///      hex letter (`a`-`f`, either case) is hex;
///   4. everything else is decimal, read as unsigned then (for a leading `-`)
///      signed.
/// So the digit-only `12345` is decimal 12345 and not 0x12345, while `12ab` is
/// hex 4779. Numbers too large for the chosen base, or containing any other
/// character, are rejected. Returns nullopt on rejection.
///
/// Trade-off, stated plainly: at exactly 16 bare hex digits the parser prefers
/// hex, so a 16-digit *decimal* value cannot be written bare -- it would be
/// read as hex. Write it in `0x` hex instead, or in any other form that is not
/// 16 bare hex digits. The round-trip of the report's token is the primary
/// contract, and it is exactly this width, so it wins at this width.
inline std::optional<uint64_t> parsePlanId(llvm::StringRef text) {
  llvm::StringRef body = text;
  if (body.consume_front("0x") || body.consume_front("0X")) {
    uint64_t hex = 0;
    if (body.empty() || body.getAsInteger(16, hex))
      return std::nullopt;
    return hex;
  }

  if (body.empty())
    return std::nullopt;

  // The report always prints 16 hex digits; at that exact width hex wins, so an
  // id like 0x1234567890123456 (all-decimal digits) round-trips. Off that
  // width, a hex letter is what separates the two ambiguous spellings.
  bool allHexDigits =
      llvm::all_of(body, [](char c) { return llvm::isHexDigit(c); });
  bool reportWidth = body.size() == 16;
  bool hasHexLetter = llvm::any_of(
      body, [](char c) { return llvm::isHexDigit(c) && !llvm::isDigit(c); });
  if (allHexDigits && (reportWidth || hasHexLetter)) {
    uint64_t hex = 0;
    if (body.getAsInteger(16, hex))
      return std::nullopt;
    return hex;
  }

  uint64_t unsignedValue = 0;
  if (!body.getAsInteger(10, unsignedValue))
    return unsignedValue;
  int64_t signedValue = 0;
  if (!body.getAsInteger(10, signedValue))
    return static_cast<uint64_t>(signedValue);
  return std::nullopt;
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

/// The comparison order the module's `micro.objective` declares, if it has one.
/// Absence is not an error: the caller keeps its default (latency minimized).
/// A *declared* objective the cost model cannot honor is an error, never a
/// silent downgrade to latency (§17.1: a target may not replace the declared
/// objective).
inline llvm::Expected<std::optional<mapping::ObjectiveOrder>>
objectiveOrderFromModule(ModuleOp module) {
  Operation *objectiveOp = nullptr;
  module.walk([&](Operation *op) {
    if (!objectiveOp && op->getName().getStringRef() == "micro.objective")
      objectiveOp = op;
  });
  if (!objectiveOp)
    return std::optional<mapping::ObjectiveOrder>{};

  auto metric = objectiveOp->getAttrOfType<StringAttr>("metric");
  auto direction = objectiveOp->getAttrOfType<StringAttr>("direction");
  if (!metric || !direction)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "micro.objective is missing its 'metric' or 'direction' attribute");

  llvm::SmallVector<llvm::StringRef, 4> secondary;
  if (auto metrics = objectiveOp->getAttrOfType<ArrayAttr>("secondary"))
    for (Attribute entry : metrics) {
      auto spelling = dyn_cast<StringAttr>(entry);
      if (!spelling)
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "micro.objective has a non-string secondary metric");
      secondary.push_back(spelling.getValue());
    }

  llvm::Expected<mapping::ObjectiveOrder> order =
      mapping::objectiveOrderFromMicro(
          metric.getValue(), direction.getValue() != "maximize", secondary);
  if (!order)
    return order.takeError();
  return std::optional<mapping::ObjectiveOrder>(std::move(*order));
}

/// A target loaded from disk plus the search it produced over one kernel.
struct MappingRun {
  std::unique_ptr<mapping::MappingTarget> target;
  Operation *kernel = nullptr;
  /// The options the search actually ran with -- the requested mode and the
  /// module's declared objective included -- so a report can state them without
  /// reconstructing them from the CLI.
  mapping::MappingSearchOptions searchOptions;
  mapping::MappingSearchResult result;
};

/// Content hash of the module's printed form. Printing is deterministic for a
/// given IR, so two runs over identical input agree byte for byte -- which is
/// what lets the plan report name the exact input it was produced from. Call it
/// before binding, while the module is still the unmodified input.
inline uint64_t computeModuleHash(ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module->print(stream);
  return mapping::stableHash(stream.str());
}

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

/// `report-only` fixes what the pass does with a search result, not how it
/// searches, so it is only meaningful together with a report path. A
/// report-only run with nothing to report into would search and then throw the
/// result away -- a success that produced nothing and could be mistaken for a
/// successful mapping -- so the pair is required rather than defaulted.
inline llvm::Error requireReportPathForReportOnly(llvm::StringRef passName,
                                                  bool reportOnly,
                                                  llvm::StringRef reportPath) {
  if (!reportOnly || !reportPath.empty())
    return llvm::Error::success();
  return llvm::createStringError(
      llvm::inconvertibleErrorCode(),
      (passName + ": report-only requires report=<path>").str());
}

/// The whole chain up to (but not including) plan selection: load the target,
/// find the kernel, extract its workload graph, and search it in the mode
/// `options.mode` names. Both entry points share this so a plan id is
/// reproducible from either: re-running with the same options replays the same
/// search and so exposes the same content-derived ids.
inline llvm::Expected<MappingRun>
runMappingSearch(ModuleOp module, llvm::StringRef passName,
                 const MicroMapOptions &options) {
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

  std::optional<mapping::SearchMode> parsed = parseSearchMode(options.mode);
  if (!parsed)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        (passName + ": unknown mode '" + options.mode +
         "' (expected deterministic, beam, or exact)")
            .str());
  mapping::SearchMode mode = *parsed;

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
  // §17.1: the declared `micro.objective` supplies the comparison order. A
  // module without one leaves the default (latency minimized) in place; one
  // the cost model cannot honor fails the pass rather than being downgraded.
  llvm::Expected<std::optional<mapping::ObjectiveOrder>> objective =
      objectiveOrderFromModule(module);
  if (!objective)
    return objective.takeError();
  if (*objective)
    searchOptions.objective = **objective;
  run.searchOptions = searchOptions;

  // §8.3/§9.5: `candidate=` binds the search to one point of the module's own
  // search space. The candidate is loaded from the module (phase-4 task 1) --
  // its values pin rule parameters of the same name -- and its layout-kind
  // parameter, resolved *by kind* rather than by name, becomes the bound layout
  // the rules must offer (carried item A). An absent candidate loads nothing,
  // so the search is byte-identical to the binding-free one.
  //
  // A binding changes which plans are *legal*, not just which is cheapest: a
  // candidate that pins a parameter no `require` accepts, or a layout no rule
  // offers, leaves nodes without a rule, so the search finds no complete plan
  // and the pass fails with the frontier's diagnostics (below) instead of
  // binding a plan the binding does not describe.
  std::optional<mapping::SearchBinding> binding;
  std::optional<std::string> boundLayout;
  if (!options.candidate.empty()) {
    llvm::Expected<mapping::SearchBinding> loaded =
        mapping::loadSearchBinding(module, options.candidate);
    if (!loaded)
      return loaded.takeError();
    binding = std::move(*loaded);
    llvm::Expected<std::optional<std::string>> layout =
        mapping::loadBoundLayout(module, *binding);
    if (!layout)
      return layout.takeError();
    boundLayout = std::move(*layout);
  }

  mapping::CoveringSearch search(*graph, *run.target, *module.getContext(),
                                 deriveLayoutContext(*graph), searchOptions,
                                 std::move(binding), std::move(boundLayout));
  llvm::Expected<mapping::MappingSearchResult> result = search.search();
  if (!result)
    return result.takeError();
  run.result = std::move(*result);
  if (run.result.plans.empty()) {
    // The frontier explains why no plan was found -- unmatched nodes, rejected
    // rules, and their reasons -- so the failure is diagnosable rather than a
    // bare "no plan". A truncated search is called out separately: no plan
    // under a cap does not mean the graph is unmappable (design §16.2).
    std::string message =
        (passName + ": the search produced no complete plan").str();
    if (run.result.searchTruncated)
      message += " (a search cap was hit)";
    for (const mapping::Diagnostic &detail : run.result.frontier.diagnostics)
      message += "\n  " + mapping::stringifyDiagnosticCode(detail.code).str() +
                 ": " + detail.message;
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   std::move(message));
  }
  return std::move(run);
}

/// Binds `plan` onto the module. `bindPlan` writes onto a private clone, so the
/// clone's body replaces the module's own to make the mapped IR the pass's
/// output; the module op itself is preserved so the pass manager stays valid.
///
/// A connection the binder cannot materialize is *reported*, not silently
/// dropped (design §18.2): the mapped IR is still valid and useful, so each
/// report is surfaced as a warning and the pass still succeeds. The reports are
/// captured before `takeBody`, because taking the body is what lets the
/// `BoundPlan` fall out of scope.
inline llvm::Error bindPlanOntoModule(ModuleOp module,
                                      const mapping::CoveringPlan &plan,
                                      const mapping::MappingTarget &target) {
  llvm::Expected<mapping::BoundPlan> bound =
      mapping::bindPlan(module, plan, target);
  if (!bound)
    return bound.takeError();
  for (const std::string &unmaterialized : bound->unmaterialized)
    module.emitWarning() << "mapping: connection not materialized: "
                         << unmaterialized;
  module.getBodyRegion().takeBody(bound->module->getBodyRegion());
  return llvm::Error::success();
}

} // namespace mlir::llk::micro_mapping_detail

#endif // LLK_CONVERSION_MICROMAPPING_MICROMAPPINGCOMMON_H
