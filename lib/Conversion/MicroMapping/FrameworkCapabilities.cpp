//===- FrameworkCapabilities.cpp - Stage-specific capability receipts ----===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/FrameworkCapabilities.h"

#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Mapping/MappingMetadata.h"
#include "LLK/Mapping/WorkloadGraph.h"
#include "LLK/Perf/SelectedKernelAnalysis.h"
#include "LLK/Runtime/KernelAbi.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace mlir::llk::framework {
namespace {

std::string hexId(uint64_t value) {
  return llvm::formatv("{0:x}", value).str();
}

/// The element types the delivered mapped ABI v1 can describe.
bool isSupportedAbiDtype(llvm::StringRef dtype) {
  return dtype == "f32" || dtype == "bf16" || dtype == "f16" ||
         dtype == "i32" || dtype == "i8";
}

/// True when `stop` reached at least `stage`.
bool reached(::llk::MappedStop stop, ::llk::MappedStop stage) {
  return static_cast<int>(stop) >= static_cast<int>(stage);
}

std::string describePort(const workload::WorkloadPort &port) {
  return "port '" + port.name + "'";
}

} // namespace

llvm::StringRef stringifyCapabilityStatus(CapabilityStatus status) {
  switch (status) {
  case CapabilityStatus::Supported:
    return "supported";
  case CapabilityStatus::Unsupported:
    return "unsupported";
  case CapabilityStatus::NotChecked:
    return "not_checked";
  }
  llvm_unreachable("unknown capability status");
}

CapabilityReceipt
describeCapabilities(mlir::Operation *kernel,
                     const workload::WorkloadContract &contract,
                     const mapping::MappingTarget &target,
                     const perf::SelectedKernelAnalysis *analysis,
                     const mapping::CoveringPlan *plan,
                     const ::llk::MappedCompilation *compilation) {
  CapabilityReceipt receipt;
  auto reject = [&](CapabilityStatus &stage, llvm::StringRef stageName,
                    const llvm::Twine &detail) {
    stage = CapabilityStatus::Unsupported;
    receipt.reasons.push_back((llvm::Twine(stageName) + ": " + detail).str());
  };

  // --- identity -----------------------------------------------------------
  //
  // Prefer the plan's recorded provenance: for an instantiated source that is
  // the exact graph the plan was selected over, which is not necessarily the
  // graph a fresh extraction of the original source would produce.
  if (plan && plan->graphHash) {
    receipt.sourceIdentity = "graph=" + hexId(plan->graphHash);
  } else if (kernel) {
    llvm::Expected<mapping::WorkloadGraph> graph =
        mapping::extractWorkloadGraph(kernel);
    if (graph)
      receipt.sourceIdentity =
          "graph=" + hexId(mapping::computeSourceGraphHash(*graph));
    else
      llvm::consumeError(graph.takeError());
  }

  if (!contract.ports.empty())
    receipt.contractIdentity = workload::canonicalWorkloadContract(contract);

  // The receipt describes the target it was handed, so its identity is that
  // target's content. A plan's recorded target hash is provenance (and may
  // name a differently named but policy-identical wrapper); it must not make
  // the receipt claim the wrong target identity.
  receipt.targetIdentity = "target=" + target.name().str() + "|content=" +
                           hexId(mapping::computeTargetContentHash(target));
  receipt.planId = plan ? plan->id : 0;

  // --- structural ---------------------------------------------------------
  if (!kernel) {
    reject(receipt.structural, "structure", "no kernel operation was provided");
  } else if (mlir::ModuleOp module =
                 kernel->getParentOfType<mlir::ModuleOp>()) {
    if (mlir::failed(mlir::verify(module)))
      reject(receipt.structural, "structure",
             "the module failed dialect/metadata verification");
    else
      receipt.structural = CapabilityStatus::Supported;
  } else {
    reject(receipt.structural, "structure",
           "the kernel is not nested in a module");
  }

  // --- interpretation -----------------------------------------------------
  if (!analysis) {
    receipt.interpretation = CapabilityStatus::NotChecked;
  } else if (analysis->complete) {
    receipt.interpretation = CapabilityStatus::Supported;
  } else {
    receipt.interpretation = CapabilityStatus::Unsupported;
    for (const std::string &reason : analysis->incompleteReasons)
      receipt.reasons.push_back("interpretation: " + reason);
    if (analysis->incompleteReasons.empty())
      receipt.reasons.push_back(
          "interpretation: the selected analysis is incomplete");
  }

  // --- mapping ------------------------------------------------------------
  if (!plan) {
    receipt.mapping = CapabilityStatus::NotChecked;
  } else {
    bool okay = true;
    auto rejectMapping = [&](const llvm::Twine &detail) {
      okay = false;
      receipt.reasons.push_back((llvm::Twine("mapping: ") + detail).str());
    };
    if (plan->placements.empty())
      rejectMapping("the selected plan has no placements");
    for (const mapping::PlanPlacement &placement : plan->placements) {
      if (!target.rules().find(placement.rule))
        rejectMapping(llvm::Twine("rule '") + placement.rule +
                      "' is not declared by target '" + target.name().str() +
                      "'");
      if (std::optional<std::string> missing =
              mapping::missingComputeBinding(placement))
        rejectMapping(
            llvm::Twine("placement for node ") + llvm::Twine(placement.node) +
            " records no compute node for required kind '" + *missing + "'");
      if (!target.machine().findExecutor(placement.executor))
        rejectMapping(llvm::Twine("placement for node ") +
                      llvm::Twine(placement.node) +
                      " names unknown executor '" + placement.executor + "'");
      for (const auto &entry : placement.memories)
        if (!target.machine().findMemory(entry.getValue()))
          rejectMapping(llvm::Twine("placement for node ") +
                        llvm::Twine(placement.node) + " port '" +
                        entry.getKey() + "' names unknown memory '" +
                        entry.getValue() + "'");
    }
    receipt.mapping =
        okay ? CapabilityStatus::Supported : CapabilityStatus::Unsupported;
  }

  // --- physical plan ------------------------------------------------------
  if (!plan) {
    receipt.physicalPlan = CapabilityStatus::NotChecked;
  } else if (!plan->materialized) {
    reject(receipt.physicalPlan, "physical_plan",
           "the selected plan is not materialized");
  } else if (!plan->diagnostics.physicalComplete) {
    receipt.physicalPlan = CapabilityStatus::Unsupported;
    for (const std::string &reason : plan->diagnostics.physicalReasons)
      receipt.reasons.push_back("physical_plan: " + reason);
    if (plan->diagnostics.physicalReasons.empty())
      receipt.reasons.push_back(
          "physical_plan: the plan records incomplete physical facts");
  } else if (plan->targetHash == 0 && !compilation) {
    reject(receipt.physicalPlan, "physical_plan",
           "the selected plan records no finalized physical provenance");
  } else {
    receipt.physicalPlan = CapabilityStatus::Supported;
  }

  // --- selected lowering --------------------------------------------------
  //
  // Three separate questions, in order: does the target declare a codegen
  // contract at all; does every selected bundle name an emitter that actually
  // lowers; and did a compilation realize the selected groups without falling
  // back to the reference bridge? A known key alone establishes none of them.
  {
    bool blocked = false;
    auto rejectSelected = [&](const llvm::Twine &detail) {
      blocked = true;
      receipt.reasons.push_back(
          (llvm::Twine("selected_lowering: ") + detail).str());
    };
    if (!target.codegenRequirements())
      rejectSelected("target '" + target.name().str() +
                     "' declares no codegen contract");
    if (plan)
      for (const mapping::PlanPlacement &placement : plan->placements) {
        const std::string &key = placement.bundle.emitterKey;
        if (!target.isKnownEmitter(key)) {
          rejectSelected("emitter '" + key + "' is not declared by target '" +
                         target.name().str() + "'");
          continue;
        }
        std::unique_ptr<mapping::TargetEmitter> emitter =
            target.createEmitter(key);
        if (!emitter) {
          rejectSelected("emitter '" + key +
                         "' is declared but cannot be "
                         "created");
          continue;
        }
        if (llvm::Error error = emitter->verify(placement.bundle))
          rejectSelected("emitter '" + key + "' rejected bundle '" +
                         placement.bundle.name +
                         "': " + llvm::toString(std::move(error)));
        else if (!emitter->hasLowering())
          rejectSelected("emitter '" + key +
                         "' only verifies its bundle, so the selected group "
                         "would fall back to the reference bridge");
      }
    // A reference-backend compilation carries no codegen contract even for a
    // target that has one: that run never attempted selected lowering, so the
    // absence of realized groups is not a blocker. Only a compilation that
    // actually asked for the selected backend can establish or refute the
    // stage.
    const bool selectedBackendAttempted =
        compilation && compilation->codegenRequirements.has_value();
    if (selectedBackendAttempted &&
        reached(compilation->stopped, ::llk::MappedStop::Lowered)) {
      if (compilation->backendGroupsRealized == 0)
        rejectSelected("the selected backend realized no groups");
      if (compilation->referenceGroupsLowered != 0)
        rejectSelected(llvm::Twine(compilation->referenceGroupsLowered) +
                       " selected group(s) fell back to the reference bridge");
    }
    if (blocked)
      receipt.selectedLowering = CapabilityStatus::Unsupported;
    else if (selectedBackendAttempted &&
             reached(compilation->stopped, ::llk::MappedStop::Lowered))
      receipt.selectedLowering = CapabilityStatus::Supported;
    else
      receipt.selectedLowering = CapabilityStatus::NotChecked;
  }

  // --- reference lowering -------------------------------------------------
  if (compilation && reached(compilation->stopped, ::llk::MappedStop::Lowered))
    receipt.referenceLowering = CapabilityStatus::Supported;
  else
    receipt.referenceLowering = CapabilityStatus::NotChecked;

  // --- invocation ---------------------------------------------------------
  //
  // Eligibility is the declared contract's fit with mapped ABI v1, which can
  // be decided statically. A successful invocation additionally needs an
  // actually compiled executable; preparation alone is not execution.
  {
    bool abiOkay = true;
    auto rejectAbi = [&](const llvm::Twine &detail) {
      abiOkay = false;
      receipt.reasons.push_back(
          (llvm::Twine("invocation: runtime_capability: ") + detail).str());
    };
    if (!contract.ports.empty()) {
      if (contract.ports.size() > ::llk::kMappedMaxDescriptors)
        rejectAbi(llvm::Twine("mapped ABI v1 supports at most ") +
                  llvm::Twine(::llk::kMappedMaxDescriptors) + " descriptors");
      for (const workload::WorkloadPort &port : contract.ports) {
        if (port.shape.size() != 2)
          rejectAbi(llvm::Twine("mapped ABI v1 requires rank-two ports; ") +
                    describePort(port) + " has rank " +
                    llvm::Twine(port.shape.size()));
        if (!isSupportedAbiDtype(port.elementType))
          rejectAbi("mapped ABI v1 does not support element type '" +
                    port.elementType + "' on " + describePort(port));
        using workload::PortDirection;
        using workload::PortEffect;
        if (port.direction == PortDirection::Input &&
            port.effect == PortEffect::ReadWrite)
          rejectAbi("mapped ABI v1 requires read-only inputs; " +
                    describePort(port) + " is read-write");
        if (port.direction == PortDirection::Output &&
            port.effect != PortEffect::Write)
          rejectAbi("mapped ABI v1 requires write-only outputs; " +
                    describePort(port) + " is not write-only");
      }
    }
    if (!abiOkay)
      receipt.invocation = CapabilityStatus::Unsupported;
    else if (compilation &&
             compilation->stopped == ::llk::MappedStop::Executable)
      receipt.invocation = CapabilityStatus::Supported;
    else
      receipt.invocation = CapabilityStatus::NotChecked;
  }

  return receipt;
}

llvm::json::Object capabilityReceiptToJson(const CapabilityReceipt &receipt) {
  llvm::json::Object object;
  object["version"] = receipt.version;
  object["structural"] = stringifyCapabilityStatus(receipt.structural).str();
  object["interpretation"] =
      stringifyCapabilityStatus(receipt.interpretation).str();
  object["mapping"] = stringifyCapabilityStatus(receipt.mapping).str();
  object["physical_plan"] =
      stringifyCapabilityStatus(receipt.physicalPlan).str();
  object["reference_lowering"] =
      stringifyCapabilityStatus(receipt.referenceLowering).str();
  object["selected_lowering"] =
      stringifyCapabilityStatus(receipt.selectedLowering).str();
  object["invocation"] = stringifyCapabilityStatus(receipt.invocation).str();

  llvm::json::Array reasons;
  for (const std::string &reason : receipt.reasons)
    reasons.push_back(reason);
  object["reasons"] = std::move(reasons);

  object["source_identity"] = receipt.sourceIdentity;
  object["contract_identity"] = receipt.contractIdentity;
  object["target_identity"] = receipt.targetIdentity;
  object["plan_id"] = hexId(receipt.planId);
  return object;
}

} // namespace mlir::llk::framework
