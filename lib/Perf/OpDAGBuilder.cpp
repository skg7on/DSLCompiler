//===- OpDAGBuilder.cpp - Injectable leaf-op DAG extraction ---------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Perf/OpDAGBuilder.h"

namespace mlir::llk::perf {
namespace {
llvm::Error invalid(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}
llvm::Error validate(llvm::StringRef name,
                     const OpDAGBuilderRegistry::Callback &callback) {
  if (name.empty())
    return invalid("operation builder name is empty");
  if (name == "micro.for" || name == "micro.spatial_for" ||
      name == "micro.pipeline" || name == "micro.yield")
    return invalid("cannot register structural operation '" + name + "'");
  if (!callback)
    return invalid("operation builder callback is empty for '" + name + "'");
  return llvm::Error::success();
}
} // namespace
llvm::Error OpDAGBuilderRegistry::registerBuilder(llvm::StringRef name,
                                                  Callback callback) {
  if (auto error = validate(name, callback))
    return error;
  if (builders.count(name))
    return invalid("duplicate operation builder for '" + name + "'");
  builders.try_emplace(name, Entry{std::move(callback), DAGLeafKind::Modeled});
  return llvm::Error::success();
}
llvm::Error OpDAGBuilderRegistry::replaceBuilder(llvm::StringRef name,
                                                 Callback callback) {
  if (auto error = validate(name, callback))
    return error;
  auto found = builders.find(name);
  if (found == builders.end())
    return invalid("no operation builder to replace for '" + name + "'");
  found->second.callback = std::move(callback);
  return llvm::Error::success();
}
const OpDAGBuilderRegistry::Callback *
OpDAGBuilderRegistry::lookup(llvm::StringRef name) const {
  auto found = builders.find(name);
  return found == builders.end() ? nullptr : &found->second.callback;
}
llvm::Error OpDAGBuilderRegistry::registerZeroCost(llvm::StringRef name,
                                                   Callback callback) {
  if (auto error = registerBuilder(name, std::move(callback)))
    return error;
  builders.find(name)->second.kind = DAGLeafKind::ZeroCost;
  return llvm::Error::success();
}
std::optional<DAGLeafKind>
OpDAGBuilderRegistry::classification(llvm::StringRef name) const {
  auto found = builders.find(name);
  if (found == builders.end())
    return std::nullopt;
  return found->second.kind;
}
} // namespace mlir::llk::perf
