//===- FrameworkCapabilities.h - Stage-specific capability receipts ------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework C1. A known emitter key, a complete static mapping and a
// well-formed bundle are three different facts, and none of them proves that a
// target lowers or runs this kernel. A capability receipt states, per stage,
// only what was actually established for one concrete source/contract/target/
// plan artifact.
//
// Four status classes are deliberately distinguished:
//
//   * Supported   -- the stage ran or was proven for this artifact;
//   * Unsupported -- a concrete blocker was found (no backend contract,
//                    unknown emitter, malformed bundle, incompatible ABI);
//   * NotChecked  -- the stage was never attempted, so no claim is made.
//
// The receipt lives above mapping, performance and runtime so it can read all
// three without putting capability policy into any of them. It never runs a
// backend lowering or JIT merely to print a report: optional analysis and
// compilation pointers carry the evidence a caller already has.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CONVERSION_FRAMEWORKCAPABILITIES_H
#define LLK_CONVERSION_FRAMEWORKCAPABILITIES_H

#include "LLK/Conversion/Workload/WorkloadContract.h"
#include "LLK/Mapping/MappingTarget.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
class Operation;
} // namespace mlir

namespace mlir::llk::perf {
struct SelectedKernelAnalysis;
} // namespace mlir::llk::perf

namespace llk {
struct MappedCompilation;
} // namespace llk

namespace mlir::llk::framework {

/// The verdict for one compiler stage.
enum class CapabilityStatus { Supported, Unsupported, NotChecked };

/// Stable lowercase rendering used by the JSON sidecar.
llvm::StringRef stringifyCapabilityStatus(CapabilityStatus status);

/// The report schema version. A reader keys its parser on it; a compatible
/// addition of a field does not change the version, but a change to an
/// existing field's meaning does.
inline constexpr uint32_t kCapabilityReceiptVersion = 1;

/// One stage-by-stage capability receipt for a concrete artifact.
///
/// `structural` covers the source module's dialect/metadata verification,
/// `interpretation` the audited selected-kernel analysis, `mapping` the
/// selected covering and its rule/machine resolution, `physicalPlan` the
/// finalized storage/occupancy decisions, `referenceLowering` the portable
/// bridge that reached backend form, `selectedLowering` the target's own
/// emitters, and `invocation` the mapped ABI plus an actual executable.
///
/// Identities reuse the existing complete renderings: the source graph content
/// hash, the canonical workload contract, the target content hash and the
/// decision-only plan id. A receipt is therefore keyed by what it describes,
/// not by a tool invocation.
struct CapabilityReceipt {
  uint32_t version = kCapabilityReceiptVersion;
  CapabilityStatus structural = CapabilityStatus::NotChecked;
  CapabilityStatus interpretation = CapabilityStatus::NotChecked;
  CapabilityStatus mapping = CapabilityStatus::NotChecked;
  CapabilityStatus physicalPlan = CapabilityStatus::NotChecked;
  CapabilityStatus referenceLowering = CapabilityStatus::NotChecked;
  CapabilityStatus selectedLowering = CapabilityStatus::NotChecked;
  CapabilityStatus invocation = CapabilityStatus::NotChecked;
  /// Ordered, stage-prefixed blockers. Only stages whose verdict is
  /// `Unsupported` contribute; an empty list therefore means no stage recorded
  /// a blocker, not that every stage succeeded.
  std::vector<std::string> reasons;
  /// `graph=<hex>` for the exact source graph (or the plan's recorded graph).
  std::string sourceIdentity;
  /// The canonical contract rendering, empty when no contract was supplied.
  std::string contractIdentity;
  /// `target=<name>|content=<hex>`.
  std::string targetIdentity;
  /// The selected plan's decision-only id, zero when no plan was supplied.
  uint64_t planId = 0;
};

/// Describes what the given artifacts establish for each compiler stage.
///
/// `kernel` is the source or bound `micro.kernel` the receipt describes.
/// `contract` is the versioned workload contract of that source; an empty
/// contract (no ports) means the caller had none, so ABI eligibility is left
/// `NotChecked` rather than guessed. `analysis`, `plan` and `compilation` are
/// evidence the caller already holds; a null one leaves its stages `NotChecked`
/// unless a concrete blocker can be established without it.
///
/// This performs structural verification and pure bundle validation only. It
/// never runs a target lowering, prepares a callable or starts a JIT.
CapabilityReceipt
describeCapabilities(mlir::Operation *kernel,
                     const workload::WorkloadContract &contract,
                     const mapping::MappingTarget &target,
                     const perf::SelectedKernelAnalysis *analysis = nullptr,
                     const mapping::CoveringPlan *plan = nullptr,
                     const ::llk::MappedCompilation *compilation = nullptr);

/// Serializes the receipt. Every stage status key is always present, so an
/// absent stage can never be read as support by omission.
llvm::json::Object capabilityReceiptToJson(const CapabilityReceipt &receipt);

} // namespace mlir::llk::framework

#endif // LLK_CONVERSION_FRAMEWORKCAPABILITIES_H
