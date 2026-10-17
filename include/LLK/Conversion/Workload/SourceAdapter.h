//===- SourceAdapter.h - Source adapter registration and receipts ---------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The explicit source-adapter boundary (framework B3). An *adapter* recognizes
// one structural family of source roots, describes its workload contract,
// declares which search parameters it owns, and realizes a candidate as a
// `micro.kernel`. The *registry* holds adapters immutably and resolves a root
// to exactly one of them -- zero matches is unsupported, and more than one
// requires the caller to name a key. There is no silent first-match dispatch,
// and no operator-name switch lives in the generic driver.
//
// Every declared choice is accounted for by a *receipt*: a parameter may only
// appear in the receipt once the IR or the selected mapping choice that
// realizes it exists. `Source` ownership is realized by the adapter; `Mapping`
// ownership stays pending until the mapping stage binds it, so the same receipt
// is checked at two stages rather than being taken on trust.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CONVERSION_WORKLOAD_SOURCEADAPTER_H
#define LLK_CONVERSION_WORKLOAD_SOURCEADAPTER_H

#include "LLK/Conversion/Workload/WorkloadContract.h"
#include "LLK/Perf/Candidate.h"
#include "LLK/Perf/SearchSpace.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mlir {
class ModuleOp;
class Operation;
} // namespace mlir

namespace mlir::llk::perf {
struct BindingFacts;
} // namespace mlir::llk::perf

namespace mlir::llk::workload {

/// Whether a search parameter is realized by the source adapter or by the later
/// mapping stage.
enum class BindingOwner { Source, Mapping };

/// The point at which a receipt is checked. The source stage runs before
/// mapping, so mapping-owned choices are declared but still pending there.
enum class ReceiptStage { Source, Complete };

/// One search parameter's declared owner and the thing it governs. A `role`
/// names a source axis (`axis:d0`) or a mapping choice (a layout, memory path,
/// owner mapping, fragment shape or tail policy); it is never inferred from the
/// parameter's spelling, and one role identifies exactly one axis or port.
struct ParameterBinding {
  std::string parameter;
  BindingOwner owner = BindingOwner::Source;
  std::string role;
};

/// Which parameters have actually been realized, per owner. An entry appears
/// only after the corresponding IR or selected mapping choice exists.
struct BindingReceipt {
  std::vector<std::string> source;
  std::vector<std::string> mapping;
};

/// Names an exact source root: the function symbol plus the ordinal of the
/// accepted top-level root inside it. Nested bodies and unrelated functions
/// never contribute an ordinal.
struct SourceSelection {
  std::string symbol;
  uint64_t rootOrdinal = 0;
};

/// The outcome of realizing one candidate: the emitted kernel, which stays
/// owned by the caller's clone, plus the receipt of what was realized.
struct SourceRealization {
  mlir::Operation *kernel = nullptr;
  BindingReceipt receipt;
};

/// One structural family of source roots. `accepts` is cheap structural family
/// recognition; `describe` performs the detailed validation and returns the
/// contract the root actually implements -- comparing it against the caller's
/// contract is what stops a diagnostic label from standing in for semantics.
class SourceAdapter {
public:
  virtual ~SourceAdapter() = default;

  /// The stable registration key. It is part of source identity.
  virtual llvm::StringRef key() const = 0;

  /// True when `root` belongs to this adapter's structural family.
  virtual bool accepts(mlir::Operation *root) const = 0;

  /// The contract `root` implements.
  virtual llvm::Expected<WorkloadContract>
  describe(mlir::Operation *root) const = 0;

  /// The parameter ownership this adapter declares for `root` in `space`.
  /// Declaring a binding is not evidence of realizing it.
  virtual llvm::Expected<std::vector<ParameterBinding>>
  bindings(mlir::Operation *root, const perf::SearchSpace &space) const = 0;

  /// The workload facts `candidate` realizes, reported without emitting
  /// anything. Machine-aware legality is the cheap analytical half of the
  /// verdict and is evaluated before realization, so the facts it reads come
  /// from the family that knows them rather than from a half-built kernel.
  /// Reporting facts is not a claim that they were realized: the receipt and
  /// the emitted IR are that evidence. The default reports none, which is what
  /// a family with no contraction content has to say.
  virtual llvm::Expected<perf::BindingFacts>
  realizedFacts(mlir::Operation *selectedRoot, const WorkloadContract &contract,
                const perf::SearchSpace &space,
                const perf::Candidate &candidate) const;

  /// Realizes `candidate` inside `clone`, appending a receipt entry only for
  /// each binding it actually realized. The returned kernel is owned by
  /// `clone`; this call must not modify any other module.
  virtual llvm::Expected<SourceRealization>
  realize(mlir::ModuleOp clone, mlir::Operation *selectedRoot,
          const WorkloadContract &contract, const perf::SearchSpace &space,
          const perf::Candidate &candidate) const = 0;
};

/// An immutable set of adapters. Configure it through `add` before use; `add`
/// rejects null adapters and duplicate keys, and resolution always performs a
/// complete scan.
class SourceAdapterRegistry {
public:
  SourceAdapterRegistry() = default;
  SourceAdapterRegistry(const SourceAdapterRegistry &) = delete;
  SourceAdapterRegistry &operator=(const SourceAdapterRegistry &) = delete;
  SourceAdapterRegistry(SourceAdapterRegistry &&) = default;
  SourceAdapterRegistry &operator=(SourceAdapterRegistry &&) = default;

  /// Registers `adapter`. Fails for null adapters, empty keys or duplicate keys.
  llvm::Error add(std::unique_ptr<SourceAdapter> adapter);

  /// Resolves `root` to the one adapter that accepts it. With an empty key: no
  /// matching adapter is `unsupported_source_adapter` and two or more are
  /// `ambiguous_source_adapter`. With a key: an unregistered key is
  /// `unknown_source_adapter`, and a registered adapter that does not accept
  /// `root` is `adapter_rejects_source`.
  llvm::Expected<const SourceAdapter *> resolve(mlir::Operation *root,
                                                llvm::StringRef key = {}) const;

  /// Every adapter that accepts `root`, in registration order. An empty result
  /// means the root belongs to no registered family; more than one means the
  /// caller must name a key.
  llvm::SmallVector<const SourceAdapter *, 2>
  matches(mlir::Operation *root) const;

  /// The adapter registered under `key`, or null when no adapter claims it.
  const SourceAdapter *findByKey(llvm::StringRef key) const;

  /// True when no adapter is registered.
  bool empty() const;

private:
  std::vector<std::unique_ptr<SourceAdapter>> adapters;
};

/// The adapters an unconfigured tool gets. Each increment registers the family
/// it implements; a family that is not registered is unsupported rather than
/// guessed at.
SourceAdapterRegistry makeDefaultSourceAdapters();

/// Selects the exact source root `selection` names: resolve the function, count
/// only the top-level roots its registered adapter accepts, and take
/// `rootOrdinal`. `adapterKey` overrides the resolution when two families match.
llvm::Expected<mlir::Operation *>
selectSourceRoot(mlir::ModuleOp module, const SourceSelection &selection,
                 const SourceAdapterRegistry &registry,
                 llvm::StringRef adapterKey = {});

/// Validates that `candidate` binds exactly the declared parameters to values
/// inside their declared domains. Shared by every source path, so the
/// diagnostics are the same whichever one runs.
llvm::Error validateCandidateBindings(const perf::SearchSpace &space,
                                      const perf::Candidate &candidate);

/// Validates `receipt` against `schema` and `candidate`. Every declared
/// parameter must have exactly one owner, every role must match the parameter's
/// declaration, and at `stage` every parameter the stage requires must appear
/// exactly once in its owner's list -- and nowhere else. Diagnostics name the
/// parameter, role or adapter that failed.
/// Source axes cannot be deferred to mapping; layout/owner/path placement
/// kinds cannot be claimed by source realization through a port-role override.
llvm::Error validateBindingReceipt(const perf::SearchSpace &space,
                                   const perf::Candidate &candidate,
                                   llvm::ArrayRef<ParameterBinding> schema,
                                   const BindingReceipt &receipt,
                                   ReceiptStage stage = ReceiptStage::Complete);

} // namespace mlir::llk::workload

#endif // LLK_CONVERSION_WORKLOAD_SOURCEADAPTER_H
