//===- WorkloadContract.cpp - Rank-neutral workload contract --------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/Workload/WorkloadContract.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <limits>
#include <set>
#include <utility>

namespace mlir::llk::workload {

/// The newest schema this increment accepts. Version 1 keeps its original
/// integer encoding; version 2 adds symbolic dimensions and symbol bounds.
static constexpr uint32_t kSupportedVersion = 2;

llvm::StringRef stringifyPortDirection(PortDirection direction) {
  switch (direction) {
  case PortDirection::Input:
    return "input";
  case PortDirection::Output:
    return "output";
  }
  llvm_unreachable("unknown port direction");
}

std::optional<PortDirection> symbolizePortDirection(llvm::StringRef text) {
  if (text == "input")
    return PortDirection::Input;
  if (text == "output")
    return PortDirection::Output;
  return std::nullopt;
}

llvm::StringRef stringifyPortEffect(PortEffect effect) {
  switch (effect) {
  case PortEffect::None:
    return "none";
  case PortEffect::Read:
    return "read";
  case PortEffect::Write:
    return "write";
  case PortEffect::ReadWrite:
    return "read_write";
  }
  llvm_unreachable("unknown port effect");
}

std::optional<PortEffect> symbolizePortEffect(llvm::StringRef text) {
  if (text == "none")
    return PortEffect::None;
  if (text == "read")
    return PortEffect::Read;
  if (text == "write")
    return PortEffect::Write;
  if (text == "read_write")
    return PortEffect::ReadWrite;
  return std::nullopt;
}

namespace {

/// The scalar element types the contract accepts. Kept in step with the
/// dtypes the machine/mapping layers model.
bool isKnownDtype(llvm::StringRef dtype) {
  static const std::set<std::string> known = {
      "f16", "bf16", "f32", "f64", "i8", "i16", "i32", "i64", "index"};
  return known.count(dtype.str()) != 0;
}

/// A `workload_contract:`-prefixed StringError.
llvm::Error contractError(const std::string &message) {
  return llvm::make_error<llvm::StringError>(message,
                                             llvm::inconvertibleErrorCode());
}

/// A `workload_contract: malformed:` StringError for attribute-shape problems.
llvm::Error malformed(const std::string &message) {
  return contractError("workload_contract: malformed: " + message);
}

/// Length-delimited field: `tag=size:value;`. The explicit length means a
/// delimiter inside `value` cannot be confused with framing.
void appendField(std::string &out, llvm::StringRef tag, llvm::StringRef value) {
  out.append(tag.data(), tag.size());
  out += '=';
  out += std::to_string(value.size());
  out += ':';
  out.append(value.data(), value.size());
  out += ';';
}

/// Tagged integer field.
void appendInt(std::string &out, llvm::StringRef tag, int64_t value) {
  out.append(tag.data(), tag.size());
  out += '=';
  out += std::to_string(value);
  out += ';';
}

/// Version 1 renders a constant extent exactly as before. A symbolic extent
/// cannot appear in a valid version 1 contract, but the renderer stays
/// deterministic for callers that hash before validating.
void appendV1Extent(std::string &out, llvm::StringRef tag,
                    const shape::Dimension &dimension) {
  if (dimension.isConstant()) {
    appendInt(out, tag, dimension.getConstant());
    return;
  }
  out.append(tag.data(), tag.size());
  out += "=symbol:";
  out += std::to_string(dimension.getSymbol().size());
  out += ':';
  out.append(dimension.getSymbol().data(), dimension.getSymbol().size());
  out += ';';
}

/// Version 2 tags constants and symbols distinctly, so a symbol named "5"
/// never collides with the constant five.
void appendV2Dimension(std::string &out, const shape::Dimension &dimension) {
  if (dimension.isConstant()) {
    out += "dim=constant:";
    out += std::to_string(dimension.getConstant());
    out += ';';
    return;
  }
  out += "dim=symbol:";
  out += std::to_string(dimension.getSymbol().size());
  out += ':';
  out.append(dimension.getSymbol().data(), dimension.getSymbol().size());
  out += ';';
}

/// True when a multiple of `divisor` lies in the inclusive range.
bool hasMultipleInRange(int64_t minimum, std::optional<int64_t> maximum,
                        int64_t divisor) {
  if (divisor <= 0)
    return false;
  int64_t remainder = minimum % divisor;
  if (remainder == 0)
    return true;
  int64_t delta = divisor - remainder;
  if (minimum > std::numeric_limits<int64_t>::max() - delta)
    return false;
  int64_t first = minimum + delta;
  return !maximum || first <= *maximum;
}

/// Sorts attributes by name; `DictionaryAttr::get` requires sorted, unique
/// keys.
void sortNamedAttrs(llvm::SmallVectorImpl<mlir::NamedAttribute> &attrs) {
  std::sort(
      attrs.begin(), attrs.end(),
      [](const mlir::NamedAttribute &lhs, const mlir::NamedAttribute &rhs) {
        return lhs.getName().getValue() < rhs.getName().getValue();
      });
}

/// Builds a DictionaryAttr from name/value pairs, sorting by name. Building
/// the sorted vector directly (rather than through NamedAttrList) keeps this
/// independent of a header location that moved across LLVM releases.
mlir::DictionaryAttr makeDictionary(
    mlir::MLIRContext *context,
    std::initializer_list<std::pair<llvm::StringRef, mlir::Attribute>>
        entries) {
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  attrs.reserve(entries.size());
  for (const auto &[name, value] : entries)
    attrs.push_back(
        mlir::NamedAttribute(mlir::StringAttr::get(context, name), value));
  sortNamedAttrs(attrs);
  return mlir::DictionaryAttr::get(context, attrs);
}

/// The declared constraint for one symbol, or nullopt when the attribute is
/// not a dictionary of integers.
std::optional<shape::SymbolConstraint>
decodeConstraint(mlir::DictionaryAttr attr) {
  shape::SymbolConstraint constraint;
  if (mlir::IntegerAttr min = attr.getAs<mlir::IntegerAttr>("min"))
    constraint.min = min.getValue().getSExtValue();
  else
    return std::nullopt;
  if (mlir::IntegerAttr max = attr.getAs<mlir::IntegerAttr>("max"))
    constraint.max = max.getValue().getSExtValue();
  if (mlir::IntegerAttr multiple = attr.getAs<mlir::IntegerAttr>("multiple_of"))
    constraint.multipleOf = multiple.getValue().getSExtValue();
  else
    return std::nullopt;
  return constraint;
}

/// A shape entry is an integer constant or a symbol name.
std::optional<shape::Dimension> decodeDimension(mlir::Attribute attr,
                                                bool &wellFormed) {
  wellFormed = true;
  if (auto constant = mlir::dyn_cast<mlir::IntegerAttr>(attr))
    return shape::Dimension(constant.getValue().getSExtValue());
  if (auto symbol = mlir::dyn_cast<mlir::StringAttr>(attr))
    return shape::Dimension::symbol(symbol.getValue().str());
  wellFormed = false;
  return std::nullopt;
}

} // namespace

llvm::Error validateWorkloadContract(const WorkloadContract &contract) {
  if (contract.version < 1 || contract.version > kSupportedVersion)
    return contractError("workload_contract: unknown_version: " +
                         std::to_string(contract.version));

  const bool version1 = contract.version == 1;

  std::set<std::string> seen;
  for (const WorkloadPort &port : contract.ports)
    if (!seen.insert(port.name).second)
      return contractError("workload_contract: duplicate_port: '" + port.name +
                           "'");

  llvm::StringSet<> portSymbols;
  for (const WorkloadPort &port : contract.ports) {
    if (port.shape.empty())
      return contractError("workload_contract: rank_zero: port '" + port.name +
                           "'");
    for (size_t dim = 0; dim < port.shape.size(); ++dim) {
      const shape::Dimension &extent = port.shape[dim];
      if (extent.isSymbol()) {
        // Version 1 has no symbolic encoding at all.
        if (version1)
          return contractError("workload_contract: dynamic_extent: port '" +
                               port.name + "' dim " + std::to_string(dim));
        if (extent.getSymbol().empty())
          return contractError("workload_contract: empty_symbol: port '" +
                               port.name + "' dim " + std::to_string(dim));
        portSymbols.insert(extent.getSymbol());
        continue;
      }
      int64_t value = extent.getConstant();
      if (value == mlir::ShapedType::kDynamic)
        return contractError("workload_contract: dynamic_extent: port '" +
                             port.name + "' dim " + std::to_string(dim));
      if (value == 0) {
        if (version1)
          return contractError("workload_contract: zero_extent: port '" +
                               port.name + "' dim " + std::to_string(dim));
        continue;
      }
      if (value < 0)
        return contractError("workload_contract: negative_extent: port '" +
                             port.name + "' dim " + std::to_string(dim));
    }
    if (!isKnownDtype(port.elementType))
      return contractError("workload_contract: unknown_dtype: '" +
                           port.elementType + "'");
  }

  llvm::StringSet<> axisSymbols;
  for (const auto &[name, extent] : contract.axes) {
    if (extent.isSymbol()) {
      if (version1)
        return contractError("workload_contract: dynamic_extent: axis '" +
                             name + "'");
      if (extent.getSymbol().empty())
        return contractError("workload_contract: empty_symbol: axis '" + name +
                             "'");
      axisSymbols.insert(extent.getSymbol());
      continue;
    }
    if (!version1 && extent.getConstant() < 0)
      return contractError("workload_contract: negative_extent: axis '" + name +
                           "'");
  }

  if (version1) {
    if (!contract.symbols.empty())
      return contractError(
          "workload_contract: unexpected_symbols: version 1 has no symbol "
          "table");
    return llvm::Error::success();
  }

  for (const auto &entry : portSymbols)
    if (!contract.symbols.count(entry.getKey().str()))
      return contractError("workload_contract: undeclared_symbol: '" +
                           entry.getKey().str() + "'");
  for (const auto &entry : axisSymbols) {
    std::string name = entry.getKey().str();
    if (!contract.symbols.count(name))
      return contractError("workload_contract: undeclared_symbol: '" + name +
                           "'");
    if (!portSymbols.count(name))
      return contractError("workload_contract: axis_symbol_not_in_port: '" +
                           name + "'");
  }

  for (const auto &[name, constraint] : contract.symbols) {
    if (name.empty())
      return contractError(
          "workload_contract: empty_symbol: a declared symbol has an empty "
          "name");
    if (constraint.min < 0)
      return contractError("workload_contract: negative_minimum: '" + name +
                           "'");
    if (constraint.max && *constraint.max < constraint.min)
      return contractError("workload_contract: unsatisfiable_bounds: '" + name +
                           "' has max below min");
    if (constraint.multipleOf <= 0)
      return contractError("workload_contract: invalid_divisor: '" + name +
                           "' has multiple_of " +
                           std::to_string(constraint.multipleOf));
    if (!hasMultipleInRange(constraint.min, constraint.max,
                            constraint.multipleOf))
      return contractError("workload_contract: unsatisfiable_bounds: '" + name +
                           "' has no multiple of " +
                           std::to_string(constraint.multipleOf) + " in range");
    if (!portSymbols.count(name))
      return contractError("workload_contract: unused_symbol: '" + name +
                           "' is declared but no port uses it");
  }

  return llvm::Error::success();
}

std::string canonicalWorkloadContract(const WorkloadContract &contract) {
  const bool version1 = contract.version == 1;
  std::string out;
  appendInt(out, "version", static_cast<int64_t>(contract.version));
  appendInt(out, "ports", static_cast<int64_t>(contract.ports.size()));
  for (const WorkloadPort &port : contract.ports) {
    appendField(out, "name", port.name);
    appendField(out, "direction", stringifyPortDirection(port.direction));
    appendField(out, "effect", stringifyPortEffect(port.effect));
    appendInt(out, "rank", static_cast<int64_t>(port.shape.size()));
    if (version1) {
      for (const shape::Dimension &extent : port.shape)
        appendV1Extent(out, "extent", extent);
    } else {
      for (const shape::Dimension &extent : port.shape)
        appendV2Dimension(out, extent);
    }
    appendField(out, "element_type", port.elementType);
  }

  // `axes` is a std::map, so iteration is already lexical.
  appendInt(out, "axes", static_cast<int64_t>(contract.axes.size()));
  for (const auto &[name, extent] : contract.axes) {
    appendField(out, "axis", name);
    if (version1)
      appendV1Extent(out, "extent", extent);
    else
      appendV2Dimension(out, extent);
  }

  if (!version1) {
    appendInt(out, "symbols", static_cast<int64_t>(contract.symbols.size()));
    for (const auto &[name, constraint] : contract.symbols) {
      appendField(out, "symbol", name);
      appendInt(out, "min", constraint.min);
      if (constraint.max)
        appendInt(out, "max", *constraint.max);
      else
        out += "max=none;";
      appendInt(out, "multiple_of", constraint.multipleOf);
    }
  }

  appendField(out, "numerical_contract", contract.numericalContract);

  // Unordered field: sort and de-duplicate.
  std::vector<std::string> primitives = contract.requiredPrimitives;
  std::sort(primitives.begin(), primitives.end());
  primitives.erase(std::unique(primitives.begin(), primitives.end()),
                   primitives.end());
  appendInt(out, "primitives", static_cast<int64_t>(primitives.size()));
  for (const std::string &primitive : primitives)
    appendField(out, "primitive", primitive);

  return out;
}

llvm::Expected<mlir::DictionaryAttr>
encodeWorkloadContract(mlir::MLIRContext *context,
                       const WorkloadContract &contract) {
  if (llvm::Error error = validateWorkloadContract(contract))
    return std::move(error);

  const bool version1 = contract.version == 1;
  mlir::IntegerType i64 = mlir::IntegerType::get(context, 64);

  llvm::SmallVector<mlir::Attribute> portAttrs;
  for (const WorkloadPort &port : contract.ports) {
    llvm::SmallVector<mlir::Attribute> shape;
    for (const shape::Dimension &extent : port.shape) {
      if (extent.isConstant()) {
        shape.push_back(mlir::IntegerAttr::get(i64, extent.getConstant()));
        continue;
      }
      // Version 1 only carries positive constants; validation rejected any
      // other value above, so this branch is unreachable for version 1.
      shape.push_back(mlir::StringAttr::get(context, extent.getSymbol()));
    }

    portAttrs.push_back(makeDictionary(
        context,
        {{"name", mlir::StringAttr::get(context, port.name)},
         {"direction", mlir::StringAttr::get(
                           context, stringifyPortDirection(port.direction))},
         {"effect",
          mlir::StringAttr::get(context, stringifyPortEffect(port.effect))},
         {"shape", mlir::ArrayAttr::get(context, shape)},
         {"element_type", mlir::StringAttr::get(context, port.elementType)}}));
  }

  llvm::SmallVector<mlir::NamedAttribute> axisAttrs;
  for (const auto &[name, extent] : contract.axes) {
    mlir::Attribute value =
        extent.isConstant()
            ? mlir::Attribute(mlir::IntegerAttr::get(i64, extent.getConstant()))
            : mlir::Attribute(
                  mlir::StringAttr::get(context, extent.getSymbol()));
    axisAttrs.push_back(
        mlir::NamedAttribute(mlir::StringAttr::get(context, name), value));
  }
  sortNamedAttrs(axisAttrs);

  llvm::SmallVector<mlir::Attribute> primitives;
  for (const std::string &primitive : contract.requiredPrimitives)
    primitives.push_back(mlir::StringAttr::get(context, primitive));

  llvm::SmallVector<mlir::NamedAttribute> entries = {
      mlir::NamedAttribute(mlir::StringAttr::get(context, "version"),
                           mlir::IntegerAttr::get(i64, contract.version)),
      mlir::NamedAttribute(mlir::StringAttr::get(context, "ports"),
                           mlir::ArrayAttr::get(context, portAttrs)),
      mlir::NamedAttribute(mlir::StringAttr::get(context, "axes"),
                           mlir::DictionaryAttr::get(context, axisAttrs)),
      mlir::NamedAttribute(
          mlir::StringAttr::get(context, "numerical_contract"),
          mlir::StringAttr::get(context, contract.numericalContract)),
      mlir::NamedAttribute(
          mlir::StringAttr::get(context, "required_primitives"),
          mlir::ArrayAttr::get(context, primitives))};

  if (!version1) {
    llvm::SmallVector<mlir::NamedAttribute> symbolAttrs;
    for (const auto &[name, constraint] : contract.symbols) {
      llvm::SmallVector<mlir::NamedAttribute> bounds = {
          mlir::NamedAttribute(mlir::StringAttr::get(context, "min"),
                               mlir::IntegerAttr::get(i64, constraint.min)),
          mlir::NamedAttribute(
              mlir::StringAttr::get(context, "multiple_of"),
              mlir::IntegerAttr::get(i64, constraint.multipleOf))};
      if (constraint.max)
        bounds.push_back(
            mlir::NamedAttribute(mlir::StringAttr::get(context, "max"),
                                 mlir::IntegerAttr::get(i64, *constraint.max)));
      symbolAttrs.push_back(
          mlir::NamedAttribute(mlir::StringAttr::get(context, name),
                               mlir::DictionaryAttr::get(context, bounds)));
    }
    sortNamedAttrs(symbolAttrs);
    entries.push_back(
        mlir::NamedAttribute(mlir::StringAttr::get(context, "symbols"),
                             mlir::DictionaryAttr::get(context, symbolAttrs)));
  }

  sortNamedAttrs(entries);
  return mlir::DictionaryAttr::get(context, entries);
}

llvm::Expected<WorkloadContract>
decodeWorkloadContract(mlir::DictionaryAttr attr) {
  if (!attr)
    return malformed("contract attribute is null");

  WorkloadContract contract;

  mlir::IntegerAttr version = attr.getAs<mlir::IntegerAttr>("version");
  if (!version)
    return malformed("'version' must be an integer");
  contract.version = static_cast<uint32_t>(version.getValue().getSExtValue());

  mlir::ArrayAttr ports = attr.getAs<mlir::ArrayAttr>("ports");
  if (!ports)
    return malformed("'ports' must be an array");
  for (mlir::Attribute portAttr : ports) {
    auto portDict = mlir::dyn_cast<mlir::DictionaryAttr>(portAttr);
    if (!portDict)
      return malformed("each port must be a dictionary");
    WorkloadPort port;

    mlir::StringAttr name = portDict.getAs<mlir::StringAttr>("name");
    if (!name)
      return malformed("port 'name' must be a string");
    port.name = name.getValue().str();

    mlir::StringAttr direction = portDict.getAs<mlir::StringAttr>("direction");
    if (!direction)
      return malformed("port 'direction' must be a string");
    auto dir = symbolizePortDirection(direction.getValue());
    if (!dir)
      return contractError("workload_contract: unknown_direction: '" +
                           direction.getValue().str() + "'");
    port.direction = *dir;

    mlir::StringAttr effect = portDict.getAs<mlir::StringAttr>("effect");
    if (!effect)
      return malformed("port 'effect' must be a string");
    auto eff = symbolizePortEffect(effect.getValue());
    if (!eff)
      return contractError("workload_contract: unknown_effect: '" +
                           effect.getValue().str() + "'");
    port.effect = *eff;

    mlir::ArrayAttr shape = portDict.getAs<mlir::ArrayAttr>("shape");
    if (!shape)
      return malformed("port 'shape' must be an array");
    for (mlir::Attribute extentAttr : shape) {
      bool wellFormed = false;
      auto extent = decodeDimension(extentAttr, wellFormed);
      if (!wellFormed)
        return malformed("port 'shape' extents must be integers or symbols");
      port.shape.push_back(*extent);
    }

    mlir::StringAttr elementType =
        portDict.getAs<mlir::StringAttr>("element_type");
    if (!elementType)
      return malformed("port 'element_type' must be a string");
    port.elementType = elementType.getValue().str();

    contract.ports.push_back(std::move(port));
  }

  mlir::DictionaryAttr axes = attr.getAs<mlir::DictionaryAttr>("axes");
  if (!axes)
    return malformed("'axes' must be a dictionary");
  for (const mlir::NamedAttribute &entry : axes) {
    bool wellFormed = false;
    auto extent = decodeDimension(entry.getValue(), wellFormed);
    if (!wellFormed)
      return malformed("axis extents must be integers or symbols");
    contract.axes[entry.getName().getValue().str()] = *extent;
  }

  if (mlir::DictionaryAttr symbols =
          attr.getAs<mlir::DictionaryAttr>("symbols")) {
    for (const mlir::NamedAttribute &entry : symbols) {
      auto bounds = mlir::dyn_cast<mlir::DictionaryAttr>(entry.getValue());
      if (!bounds)
        return malformed("symbol constraints must be dictionaries");
      auto constraint = decodeConstraint(bounds);
      if (!constraint)
        return malformed("symbol constraints need integer 'min' and "
                         "'multiple_of'");
      contract.symbols[entry.getName().getValue().str()] = *constraint;
    }
  }

  mlir::StringAttr numerical =
      attr.getAs<mlir::StringAttr>("numerical_contract");
  if (!numerical)
    return malformed("'numerical_contract' must be a string");
  contract.numericalContract = numerical.getValue().str();

  mlir::ArrayAttr primitives =
      attr.getAs<mlir::ArrayAttr>("required_primitives");
  if (!primitives)
    return malformed("'required_primitives' must be an array");
  for (mlir::Attribute primitive : primitives) {
    auto text = mlir::dyn_cast<mlir::StringAttr>(primitive);
    if (!text)
      return malformed("'required_primitives' entries must be strings");
    contract.requiredPrimitives.push_back(text.getValue().str());
  }

  if (llvm::Error error = validateWorkloadContract(contract))
    return std::move(error);

  return contract;
}

} // namespace mlir::llk::workload
