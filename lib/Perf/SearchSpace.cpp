//===- SearchSpace.cpp - micro.search_space loader ------------------------===//
//
// Part of the M12 tuning core (issue #49). See SearchSpace.h.
//
// The dialect verifier already accepts or rejects the IR; the loader's job is
// to preserve what it finds in the typed structs without losing information.
// The few checks here are the ones that matter once the records have left
// MLIR, where no verifier will run again: a parameter with no choices, a
// duplicated choice, or a constraint kind with no C++ evaluator.
//
//===----------------------------------------------------------------------===//

#include "LLK/Perf/SearchSpace.h"

// MicroDialect/MicroEnums/MicroHelpers are prerequisites for the generated
// attribute, type, and op declarations below, not direct uses here.
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Dialect/Micro/MicroEnums.h"
#include "LLK/Dialect/Micro/MicroHelpers.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <set>

// Micro attribute, type, and op declarations.
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

namespace mlir::llk::perf {

using llvm::StringRef;

//===----------------------------------------------------------------------===//
// Vocabulary
//===----------------------------------------------------------------------===//

llvm::StringRef stringifyConstraintKind(ConstraintKind kind) {
  switch (kind) {
  case ConstraintKind::SramCapacity:
    return "sram_capacity";
  case ConstraintKind::AccCapacity:
    return "acc_capacity";
  case ConstraintKind::MmaCompatible:
    return "mma_compatible";
  case ConstraintKind::MappingExtent:
    return "mapping_extent";
  case ConstraintKind::TailSupported:
    return "tail_supported";
  case ConstraintKind::VectorWidthSupported:
    return "vector_width_supported";
  case ConstraintKind::TileHierarchyCompatible:
    return "tile_hierarchy_compatible";
  case ConstraintKind::LayoutSupported:
    return "layout_supported";
  case ConstraintKind::OwnerSupported:
    return "owner_supported";
  case ConstraintKind::FragmentCompatible:
    return "fragment_compatible";
  case ConstraintKind::PipelineLiveTiles:
    return "pipeline_live_tiles";
  }
  return "";
}

std::optional<ConstraintKind> symbolizeConstraintKind(llvm::StringRef text) {
  return llvm::StringSwitch<std::optional<ConstraintKind>>(text)
      .Case("sram_capacity", ConstraintKind::SramCapacity)
      .Case("acc_capacity", ConstraintKind::AccCapacity)
      .Case("mma_compatible", ConstraintKind::MmaCompatible)
      .Case("mapping_extent", ConstraintKind::MappingExtent)
      .Case("tail_supported", ConstraintKind::TailSupported)
      .Case("vector_width_supported", ConstraintKind::VectorWidthSupported)
      .Case("tile_hierarchy_compatible",
            ConstraintKind::TileHierarchyCompatible)
      .Case("layout_supported", ConstraintKind::LayoutSupported)
      .Case("owner_supported", ConstraintKind::OwnerSupported)
      .Case("fragment_compatible", ConstraintKind::FragmentCompatible)
      .Case("pipeline_live_tiles", ConstraintKind::PipelineLiveTiles)
      .Default(std::nullopt);
}

int64_t classifyMBucket(int64_t M) {
  if (M == 1)
    return 0;
  if (M <= 4)
    return 1;
  if (M <= 16)
    return 2;
  if (M <= 64)
    return 3;
  return 4;
}

llvm::StringRef stringifyObjectiveDirection(ObjectiveDirection direction) {
  switch (direction) {
  case ObjectiveDirection::Minimize:
    return "minimize";
  case ObjectiveDirection::Maximize:
    return "maximize";
  }
  return "";
}

std::optional<ObjectiveDirection>
symbolizeObjectiveDirection(llvm::StringRef text) {
  return llvm::StringSwitch<std::optional<ObjectiveDirection>>(text)
      .Case("minimize", ObjectiveDirection::Minimize)
      .Case("maximize", ObjectiveDirection::Maximize)
      .Default(std::nullopt);
}

//===----------------------------------------------------------------------===//
// SearchSpace lookups
//===----------------------------------------------------------------------===//

const SearchParam *SearchSpace::findParam(llvm::StringRef name) const {
  for (const SearchParam &param : params)
    if (param.name == name)
      return &param;
  return nullptr;
}

const SearchParam *SearchSpace::findParamOfKind(llvm::StringRef kind) const {
  const SearchParam *found = nullptr;
  for (const SearchParam &param : params) {
    if (param.kind != kind)
      continue;
    // Ambiguity is not resolvable: "integer" names many parameters, none of
    // which is *the* integer parameter. Only a unique match is useful.
    if (found)
      return nullptr;
    found = &param;
  }
  return found;
}

//===----------------------------------------------------------------------===//
// Loading
//===----------------------------------------------------------------------===//

namespace {

llvm::Error invalid(micro::SearchSpaceOp op, const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(
      (op.getSymName() + ": " + message).str(), llvm::inconvertibleErrorCode());
}

/// Converts one `micro.param` choice, preserving its domain type.
llvm::Expected<SearchChoice> loadChoice(micro::SearchSpaceOp op,
                                        StringRef param, Attribute choice) {
  if (auto integer = dyn_cast<IntegerAttr>(choice))
    return SearchChoice(integer.getInt());
  if (auto text = dyn_cast<StringAttr>(choice))
    return SearchChoice(text.getValue().str());
  return invalid(op, "parameter '" + param +
                         "' has a choice that is neither an integer nor a "
                         "string");
}

llvm::Expected<SearchParam> loadParam(micro::SearchSpaceOp op,
                                      micro::ParamOp param) {
  SearchParam loaded;
  loaded.name = param.getName().str();
  loaded.kind = param.getKind().str();
  if (auto role = param.getRole())
    loaded.role = role->str();

  if (param.getChoices().empty())
    return invalid(op, "parameter '" + param.getName() + "' has no choices");

  std::set<std::string> seen;
  for (Attribute choice : param.getChoices()) {
    auto loadedChoice = loadChoice(op, param.getName(), choice);
    if (!loadedChoice)
      return loadedChoice.takeError();

    // The dialect orders integer choices, but symbolic ones are not required
    // to be unique, so the loader is where a repeated string choice is caught.
    std::string key = loadedChoice->isInteger()
                          ? "i:" + std::to_string(loadedChoice->integer())
                          : "s:" + loadedChoice->symbol().str();
    if (!seen.insert(std::move(key)).second)
      return invalid(op,
                     "parameter '" + param.getName() + "' repeats a choice");
    loaded.choices.push_back(std::move(*loadedChoice));
  }
  return loaded;
}

llvm::Expected<SearchConstraint>
loadConstraint(micro::SearchSpaceOp op, micro::ConstraintOp constraint) {
  SearchConstraint loaded;
  std::optional<ConstraintKind> kind =
      symbolizeConstraintKind(constraint.getKind());
  if (!kind)
    return invalid(op, "constraint kind '" + constraint.getKind() +
                           "' has no legality evaluator");
  loaded.kind = *kind;

  for (Attribute param : constraint.getParams()) {
    auto name = dyn_cast<StringAttr>(param);
    if (!name)
      return invalid(op, "constraint parameters must be strings");
    loaded.params.push_back(name.getValue().str());
  }
  return loaded;
}

void loadObjective(SearchObjective &objective, micro::ObjectiveOp op) {
  if (auto direction = symbolizeObjectiveDirection(op.getDirection()))
    objective.direction = *direction;
  objective.primaryMetric = op.getMetric().str();
  objective.secondaryMetrics.clear();
  if (std::optional<ArrayAttr> secondary = op.getSecondary())
    for (Attribute metric : *secondary)
      objective.secondaryMetrics.push_back(
          cast<StringAttr>(metric).getValue().str());
}

} // namespace

llvm::Expected<SearchSpace> loadSearchSpace(micro::SearchSpaceOp op) {
  SearchSpace space;
  space.name = op.getSymName().str();
  space.workload = op.getWorkload().str();

  for (Operation &nested : op.getBody().front().without_terminator()) {
    if (auto param = dyn_cast<micro::ParamOp>(nested)) {
      auto loaded = loadParam(op, param);
      if (!loaded)
        return loaded.takeError();
      space.params.push_back(std::move(*loaded));
      continue;
    }
    if (auto constraint = dyn_cast<micro::ConstraintOp>(nested)) {
      auto loaded = loadConstraint(op, constraint);
      if (!loaded)
        return loaded.takeError();
      space.constraints.push_back(std::move(*loaded));
      continue;
    }
    if (auto objective = dyn_cast<micro::ObjectiveOp>(nested)) {
      loadObjective(space.objective, objective);
      continue;
    }
  }
  return space;
}

llvm::Expected<SearchSpace> loadSearchSpace(mlir::ModuleOp module,
                                            llvm::StringRef symbol) {
  llvm::SmallVector<micro::SearchSpaceOp, 2> spaces;
  module.walk([&](micro::SearchSpaceOp space) { spaces.push_back(space); });

  if (symbol.empty()) {
    if (spaces.empty())
      return llvm::make_error<llvm::StringError>(
          "module has no micro.search_space", llvm::inconvertibleErrorCode());
    if (spaces.size() > 1)
      return llvm::make_error<llvm::StringError>(
          "module has more than one micro.search_space; name the one to load",
          llvm::inconvertibleErrorCode());
    return loadSearchSpace(spaces.front());
  }

  for (micro::SearchSpaceOp space : spaces)
    if (space.getSymName() == symbol)
      return loadSearchSpace(space);

  return llvm::make_error<llvm::StringError>(
      ("no micro.search_space named '" + symbol + "'").str(),
      llvm::inconvertibleErrorCode());
}

} // namespace mlir::llk::perf
