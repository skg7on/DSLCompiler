//===- BindingSchema.cpp - Parameter ownership and receipts ---------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework B3: which stage owns each search parameter, and what has actually
// been realized. Ownership is declared; realization is evidenced. Exposing a
// schema is not consumption evidence, so a parameter that appears in no receipt
// list at its stage is an unrealized axis, not a default.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/Workload/SourceAdapter.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"

#include <optional>
#include <string>

namespace mlir::llk::workload {
namespace {

llvm::Error error(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

/// The source-axis prefix. A source role states the axis it realizes; it is
/// never derived from the parameter's spelling.
constexpr llvm::StringLiteral axisRolePrefix = "axis:";

} // namespace

llvm::Error validateCandidateBindings(const perf::SearchSpace &space,
                                      const perf::Candidate &candidate) {
  for (const auto &entry : candidate.values)
    if (!space.findParam(entry.first))
      return error("unknown_binding: " + entry.first);
  for (const auto &entry : candidate.symbolicValues)
    if (!space.findParam(entry.first))
      return error("unknown_binding: " + entry.first);

  for (const perf::SearchParam &param : space.params) {
    bool hasInteger = candidate.values.count(param.name) != 0;
    bool hasSymbol = candidate.symbolicValues.count(param.name) != 0;
    if (!hasInteger && !hasSymbol)
      return error("missing_binding: " + param.name);
    if (hasInteger && hasSymbol)
      return error("out_of_domain: " + param.name);
    bool inDomain = false;
    for (const perf::SearchChoice &choice : param.choices) {
      if (hasInteger && choice.isInteger() &&
          choice.integer() == candidate.values.at(param.name))
        inDomain = true;
      if (hasSymbol && choice.isSymbolic() &&
          choice.symbol() == candidate.symbolicValues.at(param.name))
        inDomain = true;
    }
    if (!inDomain)
      return error("out_of_domain: " + param.name);
  }
  return llvm::Error::success();
}

llvm::Error validateBindingReceipt(const perf::SearchSpace &space,
                                   const perf::Candidate &candidate,
                                   llvm::ArrayRef<ParameterBinding> schema,
                                   const BindingReceipt &receipt,
                                   ReceiptStage stage) {
  if (llvm::Error invalid = validateCandidateBindings(space, candidate))
    return invalid;

  // Exactly one owner per declared parameter, and a role that matches what the
  // space declares rather than what the binding would prefer it to be.
  llvm::StringMap<const ParameterBinding *> owners;
  llvm::StringMap<const ParameterBinding *> roles;
  for (const ParameterBinding &binding : schema) {
    const perf::SearchParam *param = space.findParam(binding.parameter);
    if (!param)
      return error("unknown_parameter: " + binding.parameter);
    if (binding.role.empty())
      return error("role_mismatch: " + binding.parameter);
    if (!param->role.empty()) {
      if (binding.role != param->role)
        return error("role_mismatch: " + binding.parameter);
    } else if (binding.owner == BindingOwner::Mapping) {
      // A mapping choice's role is its declared kind unless the space names a
      // port for it.
      if (binding.role != param->kind)
        return error("role_mismatch: " + binding.parameter);
    } else if (!llvm::StringRef(binding.role).starts_with(axisRolePrefix)) {
      return error("role_mismatch: " + binding.parameter);
    }
    if (owners.count(binding.parameter))
      return error("duplicate_owner: " + binding.parameter);
    owners[binding.parameter] = &binding;
    if (roles.count(binding.role))
      return error("duplicate_role: " + binding.role);
    roles[binding.role] = &binding;
  }
  for (const perf::SearchParam &param : space.params)
    if (!owners.count(param.name))
      return error("unowned_parameter: " + param.name);

  // Every receipt entry names a declared parameter, exactly once per list.
  auto collect = [&](llvm::ArrayRef<std::string> names,
                     llvm::StringSet<> &realized) -> llvm::Error {
    for (const std::string &name : names) {
      if (!owners.count(name))
        return error("unknown_receipt: " + name);
      if (!realized.insert(name).second)
        return error("duplicate_receipt: " + name);
    }
    return llvm::Error::success();
  };
  llvm::StringSet<> realizedSource;
  llvm::StringSet<> realizedMapping;
  if (llvm::Error invalid = collect(receipt.source, realizedSource))
    return invalid;
  if (llvm::Error invalid = collect(receipt.mapping, realizedMapping))
    return invalid;

  // At `stage`, each parameter must be realized by its owner and by no one
  // else. A mapping choice is expected to be absent at the source stage, so
  // claiming it there is an unexpected receipt rather than early credit.
  for (const ParameterBinding &binding : schema) {
    bool inSource = realizedSource.count(binding.parameter) != 0;
    bool inMapping = realizedMapping.count(binding.parameter) != 0;
    std::optional<BindingOwner> expected;
    if (binding.owner == BindingOwner::Source)
      expected = BindingOwner::Source;
    else if (stage == ReceiptStage::Complete)
      expected = BindingOwner::Mapping;
    if (!expected) {
      if (inSource || inMapping)
        return error("unexpected_receipt: " + binding.parameter);
      continue;
    }
    if (*expected == BindingOwner::Source) {
      if (inMapping)
        return error("unexpected_receipt: " + binding.parameter);
      if (!inSource)
        return error("axis_not_realized: " + binding.parameter);
      continue;
    }
    if (inSource)
      return error("unexpected_receipt: " + binding.parameter);
    if (!inMapping)
      return error("axis_not_realized: " + binding.parameter);
  }
  return llvm::Error::success();
}

} // namespace mlir::llk::workload
