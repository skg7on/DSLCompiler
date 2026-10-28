//===- ShapeContract.cpp - Plain symbolic shape data and binding ----------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Shape/ShapeContract.h"

#include "llvm/ADT/StringSet.h"

#include <array>
#include <limits>
#include <set>
#include <utility>

namespace mlir::llk::shape {

namespace {

llvm::Error shapeError(const std::string &message) {
  return llvm::make_error<llvm::StringError>("shape_contract: " + message,
                                             llvm::inconvertibleErrorCode());
}

llvm::StringRef directionName(PortDirection direction) {
  return stringifyDirection(direction);
}

/// Length-delimited field: `tag=size:value;`.
void appendField(std::string &out, llvm::StringRef tag, llvm::StringRef value) {
  out.append(tag.data(), tag.size());
  out += '=';
  out += std::to_string(value.size());
  out += ':';
  out.append(value.data(), value.size());
  out += ';';
}

void appendInt(std::string &out, llvm::StringRef tag, int64_t value) {
  out.append(tag.data(), tag.size());
  out += '=';
  out += std::to_string(value);
  out += ';';
}

/// Constants and symbols use distinct tags so equal-looking values never
/// collide, and a symbol name is length-delimited like every other name.
void appendDimension(std::string &out, const Dimension &dimension) {
  if (dimension.isSymbol()) {
    out += "dim=symbol:";
    out += std::to_string(dimension.getSymbol().size());
    out += ':';
    out.append(dimension.getSymbol().data(), dimension.getSymbol().size());
    out += ';';
    return;
  }
  out += "dim=constant:";
  out += std::to_string(dimension.getConstant());
  out += ';';
}

/// True when a multiple of `divisor` lies in the inclusive range. Nonpositive
/// divisors are rejected before this runs.
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

/// Validates the symbol table against the symbols that occur in a port.
llvm::Error
validateSymbolTable(const std::map<std::string, SymbolConstraint> &symbols,
                    const llvm::StringSet<> &used) {
  for (const auto &[name, constraint] : symbols) {
    if (name.empty())
      return shapeError("empty_symbol: a declared symbol has an empty name");
    if (constraint.min < 0)
      return shapeError("negative_minimum: '" + name + "'");
    if (constraint.max && *constraint.max < constraint.min)
      return shapeError("unsatisfiable_bounds: '" + name +
                        "' has max below min");
    if (constraint.multipleOf <= 0)
      return shapeError("invalid_divisor: '" + name + "' has multiple_of " +
                        std::to_string(constraint.multipleOf));
    if (!hasMultipleInRange(constraint.min, constraint.max,
                            constraint.multipleOf))
      return shapeError("unsatisfiable_bounds: '" + name +
                        "' has no multiple of " +
                        std::to_string(constraint.multipleOf) + " in range");
    if (!used.count(name))
      return shapeError("unused_symbol: '" + name +
                        "' is declared but no port uses it");
  }
  for (const auto &entry : used)
    if (!symbols.count(entry.getKey().str()))
      return shapeError("undeclared_symbol: '" + entry.getKey().str() + "'");
  return llvm::Error::success();
}

} // namespace

llvm::StringRef stringifyDirection(PortDirection direction) {
  switch (direction) {
  case PortDirection::Input:
    return "input";
  case PortDirection::Output:
    return "output";
  }
  llvm_unreachable("unknown port direction");
}

std::optional<PortDirection> symbolizeDirection(llvm::StringRef text) {
  if (text == "input")
    return PortDirection::Input;
  if (text == "output")
    return PortDirection::Output;
  return std::nullopt;
}

Dimension Dimension::symbol(std::string name) {
  Dimension dimension;
  dimension.symbolic_ = true;
  dimension.symbol_ = std::move(name);
  return dimension;
}

llvm::Error validateShapeContract(const ShapeContract &contract) {
  std::set<std::string> names;
  for (const ShapePort &port : contract.ports)
    if (!names.insert(port.name).second)
      return shapeError("duplicate_port: '" + port.name + "'");

  llvm::StringSet<> used;
  for (const ShapePort &port : contract.ports) {
    if (port.shape.empty())
      return shapeError("rank_zero: port '" + port.name + "'");
    for (size_t dim = 0; dim < port.shape.size(); ++dim) {
      const Dimension &extent = port.shape[dim];
      if (extent.isConstant()) {
        if (extent.getConstant() < 0)
          return shapeError("negative_extent: port '" + port.name + "' dim " +
                            std::to_string(dim));
        continue;
      }
      if (extent.getSymbol().empty())
        return shapeError("empty_symbol: port '" + port.name + "' dim " +
                          std::to_string(dim));
      used.insert(extent.getSymbol());
    }
  }

  return validateSymbolTable(contract.symbols, used);
}

llvm::Expected<ShapeValues>
bindShapeValues(const ShapeContract &contract,
                llvm::ArrayRef<PortShapeValues> values) {
  if (llvm::Error error = validateShapeContract(contract))
    return std::move(error);

  // Position of each contract port among the ports of its own direction.
  std::vector<std::array<uint32_t, 2>> positions;
  positions.reserve(contract.ports.size());
  std::array<uint32_t, 2> next = {0, 0};
  for (const ShapePort &port : contract.ports) {
    uint32_t side = port.direction == PortDirection::Input ? 0 : 1;
    positions.push_back({side, next[side]++});
  }

  std::vector<const PortShapeValues *> supplied(contract.ports.size(), nullptr);
  for (const PortShapeValues &entry : values) {
    uint32_t side = entry.direction == PortDirection::Input ? 0 : 1;
    bool found = false;
    for (size_t port = 0; port < contract.ports.size(); ++port) {
      if (positions[port][0] != side || positions[port][1] != entry.index)
        continue;
      if (supplied[port])
        return shapeError("duplicate_port_values: port '" +
                          contract.ports[port].name + "'");
      supplied[port] = &entry;
      found = true;
      break;
    }
    if (!found)
      return shapeError(
          "unknown_port: " + directionName(entry.direction).str() + "[" +
          std::to_string(entry.index) + "]");
  }

  for (size_t port = 0; port < contract.ports.size(); ++port)
    if (!supplied[port])
      return shapeError("missing_port: '" + contract.ports[port].name + "'");

  ShapeValues bound;
  for (size_t port = 0; port < contract.ports.size(); ++port) {
    const ShapePort &declared = contract.ports[port];
    const PortShapeValues &actual = *supplied[port];
    if (actual.extents.size() != declared.shape.size())
      return shapeError("rank_mismatch: port '" + declared.name +
                        "' expects rank " +
                        std::to_string(declared.shape.size()) + " but got " +
                        std::to_string(actual.extents.size()));
    for (size_t dim = 0; dim < declared.shape.size(); ++dim) {
      int64_t extent = actual.extents[dim];
      if (extent < 0)
        return shapeError("negative_value: port '" + declared.name + "' dim " +
                          std::to_string(dim));
      const Dimension &expected = declared.shape[dim];
      if (expected.isConstant()) {
        if (extent != expected.getConstant())
          return shapeError("constant_mismatch: port '" + declared.name +
                            "' dim " + std::to_string(dim) + " expects " +
                            std::to_string(expected.getConstant()) +
                            " but got " + std::to_string(extent));
        continue;
      }
      auto [entry, inserted] =
          bound.insert({expected.getSymbol().str(), extent});
      if (!inserted && entry->second != extent)
        return shapeError("symbol_conflict: '" + entry->first + "' binds " +
                          std::to_string(entry->second) + " and " +
                          std::to_string(extent));
    }
  }

  for (const auto &[name, value] : bound) {
    const SymbolConstraint &constraint = contract.symbols.at(name);
    if (value < constraint.min)
      return shapeError("constraint_violation: '" + name + "' bound " +
                        std::to_string(value) + " is below its minimum " +
                        std::to_string(constraint.min));
    if (constraint.max && value > *constraint.max)
      return shapeError("constraint_violation: '" + name + "' bound " +
                        std::to_string(value) + " is above its maximum " +
                        std::to_string(*constraint.max));
    if (value % constraint.multipleOf != 0)
      return shapeError("constraint_violation: '" + name + "' bound " +
                        std::to_string(value) + " is not a multiple of " +
                        std::to_string(constraint.multipleOf));
  }

  return bound;
}

std::string canonicalShapeContract(const ShapeContract &contract) {
  std::string out;
  appendInt(out, "ports", static_cast<int64_t>(contract.ports.size()));
  for (const ShapePort &port : contract.ports) {
    appendField(out, "name", port.name);
    appendField(out, "direction", directionName(port.direction));
    appendInt(out, "rank", static_cast<int64_t>(port.shape.size()));
    for (const Dimension &dimension : port.shape)
      appendDimension(out, dimension);
  }

  // A std::map iterates lexically, so the symbol table is already canonical.
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
  return out;
}

llvm::StringRef stringifyPredicateVerdict(PredicateVerdict verdict) {
  switch (verdict) {
  case PredicateVerdict::Proven:
    return "proven";
  case PredicateVerdict::Disproven:
    return "disproven";
  case PredicateVerdict::RuntimeGuard:
    return "runtime_guard";
  }
  return "";
}

PredicateVerdict classifyPredicate(const ShapePredicate &predicate,
                                   const ShapeContract &contract) {
  auto declared = contract.symbols.find(predicate.symbol);
  // A symbol the contract does not declare has no domain to prove anything
  // over. Validation rejects the contract itself; classification stays
  // conservative rather than inventing a domain from the value.
  if (declared == contract.symbols.end())
    return PredicateVerdict::RuntimeGuard;
  const SymbolConstraint &constraint = declared->second;
  // A contract that has not been validated may carry a nonpositive divisor.
  // Classification never divides by it: an unvalidated domain proves nothing.
  if (constraint.multipleOf <= 0)
    return PredicateVerdict::RuntimeGuard;

  switch (predicate.kind) {
  case ShapePredicate::Kind::DivisibleBy: {
    if (predicate.value <= 0)
      return PredicateVerdict::Disproven;
    // The declared congruence implies the predicate whenever it is a multiple
    // of the queried divisor.
    if (constraint.multipleOf % predicate.value == 0)
      return PredicateVerdict::Proven;
    // A single-point domain decides immediately.
    if (constraint.max && constraint.min == *constraint.max)
      return constraint.min % predicate.value == 0
                 ? PredicateVerdict::Proven
                 : PredicateVerdict::Disproven;
    // No admissible extent can satisfy it.
    if (!hasMultipleInRange(constraint.min, constraint.max, predicate.value))
      return PredicateVerdict::Disproven;
    return PredicateVerdict::RuntimeGuard;
  }
  case ShapePredicate::Kind::EqualTo: {
    if (predicate.value < constraint.min ||
        (constraint.max && predicate.value > *constraint.max))
      return PredicateVerdict::Disproven;
    // A declared congruence that the value violates rules it out even inside
    // the range.
    if (predicate.value % constraint.multipleOf != 0)
      return PredicateVerdict::Disproven;
    if (constraint.max && constraint.min == *constraint.max)
      return PredicateVerdict::Proven;
    return PredicateVerdict::RuntimeGuard;
  }
  case ShapePredicate::Kind::AtLeast: {
    if (constraint.min >= predicate.value)
      return PredicateVerdict::Proven;
    if (constraint.max && *constraint.max < predicate.value)
      return PredicateVerdict::Disproven;
    return PredicateVerdict::RuntimeGuard;
  }
  }
  return PredicateVerdict::RuntimeGuard;
}

} // namespace mlir::llk::shape
