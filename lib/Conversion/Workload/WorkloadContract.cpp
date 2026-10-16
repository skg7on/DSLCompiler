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

#include <algorithm>
#include <set>
#include <utility>

namespace mlir::llk::workload {

/// The only contract schema this increment accepts.
static constexpr uint32_t kSupportedVersion = 1;

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

} // namespace

llvm::Error validateWorkloadContract(const WorkloadContract &contract) {
  if (contract.version != kSupportedVersion)
    return contractError("workload_contract: unknown_version: " +
                         std::to_string(contract.version));

  std::set<std::string> seen;
  for (const WorkloadPort &port : contract.ports)
    if (!seen.insert(port.name).second)
      return contractError("workload_contract: duplicate_port: '" + port.name +
                           "'");

  for (const WorkloadPort &port : contract.ports) {
    if (port.shape.empty())
      return contractError("workload_contract: rank_zero: port '" + port.name +
                           "'");
    for (size_t dim = 0; dim < port.shape.size(); ++dim) {
      int64_t extent = port.shape[dim];
      if (extent == mlir::ShapedType::kDynamic)
        return contractError("workload_contract: dynamic_extent: port '" +
                             port.name + "' dim " + std::to_string(dim));
      if (extent == 0)
        return contractError("workload_contract: zero_extent: port '" +
                             port.name + "' dim " + std::to_string(dim));
      if (extent < 0)
        return contractError("workload_contract: negative_extent: port '" +
                             port.name + "' dim " + std::to_string(dim));
    }
    if (!isKnownDtype(port.elementType))
      return contractError("workload_contract: unknown_dtype: '" +
                           port.elementType + "'");
  }

  return llvm::Error::success();
}

std::string canonicalWorkloadContract(const WorkloadContract &contract) {
  std::string out;
  appendInt(out, "version", static_cast<int64_t>(contract.version));
  appendInt(out, "ports", static_cast<int64_t>(contract.ports.size()));
  for (const WorkloadPort &port : contract.ports) {
    appendField(out, "name", port.name);
    appendField(out, "direction", stringifyPortDirection(port.direction));
    appendField(out, "effect", stringifyPortEffect(port.effect));
    appendInt(out, "rank", static_cast<int64_t>(port.shape.size()));
    for (int64_t extent : port.shape)
      appendInt(out, "extent", extent);
    appendField(out, "element_type", port.elementType);
  }

  // `axes` is a std::map, so iteration is already lexical.
  appendInt(out, "axes", static_cast<int64_t>(contract.axes.size()));
  for (const auto &[name, extent] : contract.axes) {
    appendField(out, "axis", name);
    appendInt(out, "extent", extent);
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

mlir::DictionaryAttr encodeWorkloadContract(mlir::MLIRContext *context,
                                            const WorkloadContract &contract) {
  mlir::IntegerType i64 = mlir::IntegerType::get(context, 64);

  llvm::SmallVector<mlir::Attribute> portAttrs;
  for (const WorkloadPort &port : contract.ports) {
    llvm::SmallVector<mlir::Attribute> shape;
    for (int64_t extent : port.shape)
      shape.push_back(mlir::IntegerAttr::get(i64, extent));

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
  for (const auto &[name, extent] : contract.axes)
    axisAttrs.push_back(
        mlir::NamedAttribute(mlir::StringAttr::get(context, name),
                             mlir::IntegerAttr::get(i64, extent)));
  sortNamedAttrs(axisAttrs);

  llvm::SmallVector<mlir::Attribute> primitives;
  for (const std::string &primitive : contract.requiredPrimitives)
    primitives.push_back(mlir::StringAttr::get(context, primitive));

  return makeDictionary(
      context,
      {{"version", mlir::IntegerAttr::get(i64, contract.version)},
       {"ports", mlir::ArrayAttr::get(context, portAttrs)},
       {"axes", mlir::DictionaryAttr::get(context, axisAttrs)},
       {"numerical_contract",
        mlir::StringAttr::get(context, contract.numericalContract)},
       {"required_primitives", mlir::ArrayAttr::get(context, primitives)}});
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
      auto extent = mlir::dyn_cast<mlir::IntegerAttr>(extentAttr);
      if (!extent)
        return malformed("port 'shape' extents must be integers");
      port.shape.push_back(extent.getValue().getSExtValue());
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
    auto extent = mlir::dyn_cast<mlir::IntegerAttr>(entry.getValue());
    if (!extent)
      return malformed("axis extents must be integers");
    contract.axes[entry.getName().getValue().str()] =
        extent.getValue().getSExtValue();
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
