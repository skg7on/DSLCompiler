//===- SourceAdapter.cpp - Source adapter registry and root selection -----===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework B3: the immutable adapter registry and exact root selection.
// Resolution is deliberately total -- zero matches is unsupported and several
// matches demand an explicit key -- so a new adapter can never silently take
// over a source another adapter used to own.
//
// Root selection resolves the function by symbol and then counts only the
// top-level roots inside it. A nested scalar body belongs to the op that
// contains it, and a second function is a different source, so neither can
// shift the ordinal.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/Workload/SourceAdapter.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"

#include <string>
#include <utility>

using namespace mlir;

namespace mlir::llk::workload {
namespace {

llvm::Error error(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

/// True for a function definition. Only a semantic source function is a root
/// owner; a concrete `micro.kernel` is selected by symbol instead.
bool isFunction(Operation *op) {
  return op->getName().getStringRef() == "func.func";
}

llvm::Expected<Operation *> resolveFunction(ModuleOp module,
                                            const SourceSelection &selection) {
  if (!selection.symbol.empty()) {
    Operation *function = SymbolTable::lookupSymbolIn(module, selection.symbol);
    if (!function)
      return error("unknown source function '@" + selection.symbol + "'");
    if (!isFunction(function))
      return error("source symbol '@" + selection.symbol +
                   "' is not a function");
    return function;
  }
  SmallVector<Operation *, 2> functions;
  for (Operation &op : module.getBody()->getOperations())
    if (isFunction(&op))
      functions.push_back(&op);
  if (functions.size() != 1)
    return error(
        "source function symbol is required when the module declares " +
        Twine(functions.size()) + " functions");
  return functions.front();
}

} // namespace

llvm::Error SourceAdapterRegistry::add(std::unique_ptr<SourceAdapter> adapter) {
  if (!adapter)
    return error("null_source_adapter");
  if (adapter->key().empty())
    return error("empty_source_adapter_key");
  if (findByKey(adapter->key()))
    return error("duplicate_source_adapter: " + adapter->key());
  adapters.push_back(std::move(adapter));
  return llvm::Error::success();
}

llvm::SmallVector<const SourceAdapter *, 2>
SourceAdapterRegistry::matches(Operation *root) const {
  llvm::SmallVector<const SourceAdapter *, 2> accepted;
  if (!root)
    return accepted;
  for (const std::unique_ptr<SourceAdapter> &adapter : adapters)
    if (adapter->accepts(root))
      accepted.push_back(adapter.get());
  return accepted;
}

const SourceAdapter *
SourceAdapterRegistry::findByKey(llvm::StringRef key) const {
  if (key.empty())
    return nullptr;
  for (const std::unique_ptr<SourceAdapter> &adapter : adapters)
    if (adapter->key() == key)
      return adapter.get();
  return nullptr;
}

llvm::Expected<const SourceAdapter *>
SourceAdapterRegistry::resolve(Operation *root, llvm::StringRef key) const {
  if (!key.empty()) {
    const SourceAdapter *adapter = findByKey(key);
    if (!adapter)
      return error("unknown_source_adapter: " + key);
    if (!adapter->accepts(root))
      return error("adapter_rejects_source: " + key);
    return adapter;
  }

  llvm::SmallVector<const SourceAdapter *, 2> accepted = matches(root);
  if (accepted.empty())
    return error("unsupported_source_adapter");
  if (accepted.size() > 1)
    return error("ambiguous_source_adapter");
  return accepted.front();
}

bool SourceAdapterRegistry::empty() const { return adapters.empty(); }

SourceAdapterRegistry makeDefaultSourceAdapters() {
  // The families are registered by the increments that implement them. An
  // unregistered family is unsupported rather than guessed at, so this stays
  // empty until a real adapter exists to add.
  return SourceAdapterRegistry();
}

llvm::Expected<Operation *>
selectSourceRoot(ModuleOp module, const SourceSelection &selection,
                 const SourceAdapterRegistry &registry,
                 llvm::StringRef adapterKey) {
  if (!module)
    return error("source selection requires a module");
  llvm::Expected<Operation *> function = resolveFunction(module, selection);
  if (!function)
    return function.takeError();
  const std::string symbol =
      SymbolTable::getSymbolName(*function).getValue().str();
  if ((*function)->getNumRegions() == 0 || (*function)->getRegion(0).empty())
    return error("source function '@" + symbol + "' has no body");

  SmallVector<Operation *, 4> roots;
  for (Operation &op : (*function)->getRegion(0).front().getOperations())
    if (!op.hasTrait<OpTrait::IsTerminator>())
      roots.push_back(&op);

  SmallVector<Operation *, 4> accepted;
  if (!adapterKey.empty()) {
    const SourceAdapter *adapter = registry.findByKey(adapterKey);
    if (!adapter)
      return error("unknown_source_adapter: " + adapterKey);
    for (Operation *root : roots)
      if (adapter->accepts(root))
        accepted.push_back(root);
    if (accepted.empty())
      return error("adapter_rejects_source: " + adapterKey);
  } else {
    for (Operation *root : roots)
      if (!registry.matches(root).empty())
        accepted.push_back(root);
    if (accepted.empty())
      return error("unsupported_source_adapter: @" + symbol);
  }

  if (selection.rootOrdinal >= accepted.size())
    return error("source_root_ordinal " + Twine(selection.rootOrdinal) +
                 " is out of range for @" + symbol);
  Operation *selected = accepted[selection.rootOrdinal];
  auto adapter = registry.resolve(selected, adapterKey);
  if (!adapter)
    return adapter.takeError();
  return selected;
}

} // namespace mlir::llk::workload
