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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
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

/// Read the declared tensor/memref boundary without running ABI preparation.
/// The source and its bound kernel retain the same logical signature.
std::optional<::llk::KernelAbi> sourceSignature(mlir::Operation *kernel) {
  if (!kernel)
    return std::nullopt;
  auto attr = kernel->getAttrOfType<mlir::TypeAttr>("function_type");
  auto type = attr ? mlir::dyn_cast<mlir::FunctionType>(attr.getValue())
                   : mlir::FunctionType{};
  if (!type)
    return std::nullopt;
  ::llk::KernelAbi signature;
  auto append = [](mlir::TypeRange types,
                   std::vector<::llk::KernelAbi::Port> &ports) {
    for (mlir::Type type : types) {
      auto shaped = mlir::dyn_cast<mlir::ShapedType>(type);
      if (!shaped || !shaped.hasRank())
        return false;
      std::string element;
      llvm::raw_string_ostream stream(element);
      shaped.getElementType().print(stream);
      ports.push_back({std::vector<int64_t>(shaped.getShape().begin(),
                                            shaped.getShape().end()),
                       stream.str()});
    }
    return true;
  };
  if (!append(type.getInputs(), signature.inputs) ||
      !append(type.getResults(), signature.outputs))
    return std::nullopt;
  return signature;
}

/// Names/effects belong to the workload contract; the runtime ABI records the
/// ordered shapes and element types separately for input and output ports.
std::optional<std::string>
contractMismatch(const workload::WorkloadContract &contract,
                 const ::llk::KernelAbi &abi, llvm::StringRef boundary) {
  size_t input = 0, output = 0;
  for (const workload::WorkloadPort &port : contract.ports) {
    bool isInput = port.direction == workload::PortDirection::Input;
    const auto &actual = isInput ? abi.inputs : abi.outputs;
    size_t index = isInput ? input++ : output++;
    if (index >= actual.size() || port.shape != actual[index].shape ||
        port.elementType != actual[index].elementType)
      return "the declared contract " + describePort(port) +
             " does not match the " + boundary.str();
  }
  if (input != abi.inputs.size() || output != abi.outputs.size())
    return "the declared contract input/output counts do not match the " +
           boundary.str();
  return std::nullopt;
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
  // Resolve the actual artifact. The projection handles binder-inserted
  // movement and transforms; trusting the plan's recorded hash would instead
  // lend stale evidence to an unrelated source.
  uint64_t sourceHash = 0;
  if (kernel) {
    auto hash = mapping::computeModuleSourceGraphHash(kernel);
    if (hash) {
      sourceHash = *hash;
      receipt.sourceIdentity = "graph=" + hexId(sourceHash);
    } else
      llvm::consumeError(hash.takeError());
  }

  if (!contract.ports.empty())
    receipt.contractIdentity = workload::canonicalWorkloadContract(contract);

  const uint64_t targetHash = mapping::computeTargetContentHash(target);
  const uint64_t machineHash = machine::computeContentHash(target.machine());
  receipt.targetIdentity =
      "target=" + target.name().str() + "|content=" + hexId(targetHash);
  receipt.planId = plan ? plan->id : 0;

  std::vector<std::string> planErrors;
  if (plan) {
    if (!sourceHash || plan->graphHash != sourceHash)
      planErrors.push_back(
          "the plan source identity does not match the kernel");
    if (plan->targetHash != targetHash)
      planErrors.push_back(
          "the plan target identity does not match the target");
  }
  std::vector<std::string> compilationErrors;
  if (compilation) {
    if (!sourceHash || compilation->sourceGraphHash != sourceHash)
      compilationErrors.push_back(
          "the compilation source identity does not match the kernel");
    if (compilation->targetHash != targetHash ||
        compilation->machineHash != machineHash ||
        compilation->targetName != target.name())
      compilationErrors.push_back(
          "the compilation target identity does not match the target");
    if (plan && compilation->planId != plan->id)
      compilationErrors.push_back(
          "the compilation plan identity does not match the plan");
  }

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
    // Ask the emitter about the operation group the lowering dispatch would
    // hand it, not only about its key: one key can cover several dtype or
    // shape signatures and implement a subset of them. A bound kernel records
    // that group as `micro.mapping` instance metadata; the source kernel a
    // caller is about to map does not, so resolve its placement nodes through
    // the same workload graph the search used.
    llvm::DenseMap<uint64_t, llvm::SmallVector<mlir::Operation *>>
        opsByInstance;
    bool kernelCarriesMapping = false;
    if (kernel)
      kernel->walk([&](mlir::Operation *op) {
        auto mapping = op->getAttrOfType<mlir::DictionaryAttr>("micro.mapping");
        if (!mapping)
          return;
        auto instance = mapping.getAs<mlir::IntegerAttr>("instance");
        if (!instance)
          return;
        kernelCarriesMapping = true;
        opsByInstance[instance.getInt()].push_back(op);
      });
    if (kernel && !kernelCarriesMapping && plan) {
      mapping::WorkloadGraphBinding binding;
      llvm::Expected<mapping::WorkloadGraph> graph =
          mapping::extractWorkloadGraph(kernel, &binding);
      if (!graph) {
        llvm::consumeError(graph.takeError());
        rejectSelected("the selected plan's operation groups could not be "
                       "resolved from the kernel");
      } else {
        for (const mapping::PlanPlacement &placement : plan->placements)
          if (mlir::Operation *op = binding.opFor(placement.node))
            opsByInstance[placement.instance].push_back(op);
          else
            rejectSelected("the selected plan's node " +
                           llvm::Twine(placement.node) +
                           " has no operation in the kernel");
      }
    }
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
        else if (!emitter->hasLoweringFor(
                     opsByInstance.lookup(placement.instance)))
          rejectSelected("emitter '" + key +
                         "' cannot lower the selected group's operations, so "
                         "it would fall back to the reference bridge");
      }
    // A reference-backend compilation carries no codegen contract even for a
    // target that has one: that run never attempted selected lowering, so the
    // absence of realized groups is not a blocker. Only a compilation that
    // actually asked for the selected backend can establish or refute the
    // stage.
    const bool selectedBackendAttempted =
        compilation &&
        compilation->backend == ::llk::MappedBackend::SelectedTarget;
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
  if (compilation && compilation->backend == ::llk::MappedBackend::Reference &&
      reached(compilation->stopped, ::llk::MappedStop::Lowered))
    receipt.referenceLowering = CapabilityStatus::Supported;
  else
    receipt.referenceLowering = CapabilityStatus::NotChecked;

  // --- invocation ---------------------------------------------------------
  //
  // Eligibility is static. Descriptor preparation and host compatibility are
  // evidenced by successful executable creation; a returned invocation is
  // recorded by that executable itself, never inferred from a stopping point.
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
        for (int64_t extent : port.shape)
          if (extent <= 0) {
            rejectAbi("mapped ABI v1 requires positive static extents; " +
                      describePort(port) + " has a nonpositive/dynamic extent");
            break;
          }
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
      receipt.abiEligibility = receipt.abiPreparation = receipt.invocation =
          CapabilityStatus::Unsupported;
    else {
      if (!contract.ports.empty())
        receipt.abiEligibility = CapabilityStatus::Supported;
      if (compilation && compilation->executable &&
          compilation->stopped == ::llk::MappedStop::Executable) {
        receipt.abiPreparation = CapabilityStatus::Supported;
        receipt.hostCompatibility = CapabilityStatus::Supported;
        if (compilation->executable->successfulInvocations() > 0)
          receipt.invocation = CapabilityStatus::Supported;
      }
    }
  }

  // A declaration can fit the ABI envelope while describing a different
  // artifact. Never credit this artifact's preparation or calls to it. The
  // executable ABI is authoritative even if no source signature is available.
  if (!contract.ports.empty()) {
    auto checkAssociation = [&](const ::llk::KernelAbi &abi,
                                llvm::StringRef boundary) {
      if (auto reason = contractMismatch(contract, abi, boundary)) {
        reject(receipt.abiPreparation, "abi_preparation", *reason);
        reject(receipt.invocation, "invocation", *reason);
      }
    };
    if (auto signature = sourceSignature(kernel))
      checkAssociation(*signature, "source signature");
    if (compilation && compilation->executable)
      checkAssociation(compilation->executable->abi(), "executable ABI");
  }

  // Evidence is credited only to the artifact it actually validated. Keep a
  // structurally valid changed source visible while refusing the stale stages.
  for (const std::string &reason : planErrors) {
    reject(receipt.mapping, "mapping", reason);
    reject(receipt.physicalPlan, "physical_plan", reason);
    reject(receipt.selectedLowering, "selected_lowering", reason);
  }
  if (analysis && (!sourceHash || analysis->sourceGraphHash != sourceHash ||
                   analysis->machineHash != machineHash ||
                   (plan && analysis->planId && analysis->planId != plan->id)))
    reject(
        receipt.interpretation, "interpretation",
        "the analysis source/target/plan identity does not match the artifact");
  for (const std::string &reason : compilationErrors) {
    if (compilation->backend == ::llk::MappedBackend::SelectedTarget)
      reject(receipt.selectedLowering, "selected_lowering", reason);
    else if (reached(compilation->stopped, ::llk::MappedStop::Lowered))
      reject(receipt.referenceLowering, "reference_lowering", reason);
    if (compilation->stopped == ::llk::MappedStop::Executable) {
      reject(receipt.abiPreparation, "abi_preparation", reason);
      reject(receipt.hostCompatibility, "host_compatibility", reason);
      reject(receipt.invocation, "invocation", reason);
    }
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
  object["abi_eligibility"] =
      stringifyCapabilityStatus(receipt.abiEligibility).str();
  object["abi_preparation"] =
      stringifyCapabilityStatus(receipt.abiPreparation).str();
  object["host_compatibility"] =
      stringifyCapabilityStatus(receipt.hostCompatibility).str();
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
