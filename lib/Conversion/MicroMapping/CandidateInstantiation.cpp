//===- CandidateInstantiation.cpp - Instantiate a candidate ---------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/MicroMapping/CandidateInstantiation.h"

#include "LLK/Conversion/MicroMapping/SearchBindingLoader.h"
#include "LLK/Conversion/Workload/ContractionSourceAdapter.h"
#include "LLK/Mapping/MappingMetadata.h"
#include "LLK/Mapping/SearchBinding.h"
#include "LLK/Mapping/StableHash.h"
#include "LLK/Perf/CandidateBinding.h"
#include "LLK/Perf/Legality.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"

#include <algorithm>
#include <map> // IWYU pragma: keep
#include <optional>
#include <string>

using namespace mlir;

namespace mlir::llk::tuning {
namespace {

llvm::Error error(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

/// Drops the tuning directives from a private clone. A `micro.search_space` is
/// input metadata for this instantiation, not part of the executable candidate,
/// so the compiler never sees an unlowered directive when materializing or
/// replaying the selected kernel. The caller's module keeps them.
void eraseSearchSpaces(ModuleOp module) {
  SmallVector<Operation *> searchSpaces;
  module.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "micro.search_space")
      searchSpaces.push_back(op);
  });
  for (Operation *op : searchSpaces)
    op->erase();
}

/// The stable hash of every binding the candidate carries. A parameterized
/// source realizes its bindings in IR, so they belong in the identity rather
/// than being recoverable only from the candidate id.
uint64_t candidateBindingHash(const perf::Candidate &candidate) {
  llvm::StringMap<mapping::SearchValue> values;
  for (const auto &entry : candidate.values)
    values[entry.first] = entry.second;
  for (const auto &entry : candidate.symbolicValues)
    values[entry.first] = entry.second;
  return mapping::makeSearchBinding(candidate.id, std::move(values)).stableHash;
}

/// The symbol of the module-level operation that contains `nested`. Source
/// identity names the function a root belongs to, never the emitted kernel.
std::string enclosingSymbol(ModuleOp module, Operation *nested) {
  Operation *owner = nested;
  while (owner->getParentOp() && owner->getParentOp() != module.getOperation())
    owner = owner->getParentOp();
  StringAttr symbol = SymbolTable::getSymbolName(owner);
  return symbol ? symbol.getValue().str() : std::string();
}

std::string printOperation(Operation *op) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  op->print(stream);
  stream.flush();
  return text;
}

llvm::Expected<Operation *>
selectConcreteKernel(ModuleOp module, const CandidateInstantiationOptions &o) {
  if (o.sourceRootOrdinal != 0)
    return error("ConcreteMicro source root ordinal must be zero");
  SmallVector<Operation *> kernels;
  for (Operation &op : module.getBody()->getOperations()) {
    if (op.getName().getStringRef() != "micro.kernel")
      continue;
    auto symbol =
        op.getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    if (o.sourceSymbol.empty() ||
        (symbol && symbol.getValue() == o.sourceSymbol))
      kernels.push_back(&op);
  }
  if (kernels.empty())
    return error("unknown ConcreteMicro kernel symbol '" + o.sourceSymbol +
                 "'");
  if (kernels.size() != 1)
    return error("ambiguous ConcreteMicro kernel symbol '" + o.sourceSymbol +
                 "'");
  return kernels.front();
}

llvm::Error verifyConcreteContract(Operation *kernel,
                                   const workload::WorkloadContract &contract) {
  auto typeAttr = kernel->getAttrOfType<TypeAttr>("function_type");
  auto type =
      typeAttr ? dyn_cast<FunctionType>(typeAttr.getValue()) : FunctionType{};
  if (!type)
    return error(
        "ConcreteMicro kernel requires function_type for port validation");
  // Which contract port each signature entry realizes. An unbound (v1)
  // kernel keeps the positional convention -- inputs first, then results. A
  // bound kernel names the port explicitly, which is what lets an output
  // destination be an *argument* rather than a positional result.
  SmallVector<unsigned> argumentPort;
  SmallVector<bool> argumentIsOutputPort;
  SmallVector<unsigned> resultPort;
  const bool hasBindings =
      static_cast<bool>(kernel->getAttrOfType<ArrayAttr>("argument_bindings"));
  if (auto bindings = kernel->getAttrOfType<ArrayAttr>("argument_bindings")) {
    if (bindings.size() != type.getNumInputs())
      return error("workload_contract: argument bindings do not cover the "
                   "kernel signature");
    llvm::SmallDenseSet<unsigned, 8> bound;
    for (unsigned i = 0, e = bindings.size(); i < e; ++i) {
      auto entry = dyn_cast<DictionaryAttr>(bindings[i]);
      auto role = entry ? entry.getAs<StringAttr>("role") : StringAttr();
      auto port = entry ? entry.getAs<IntegerAttr>("port") : IntegerAttr();
      if (!role || !port || port.getInt() < 0 ||
          static_cast<size_t>(port.getInt()) >= contract.ports.size() ||
          (role.getValue() != "input" && role.getValue() != "output"))
        return error("workload_contract: malformed argument binding");
      unsigned index = static_cast<unsigned>(port.getInt());
      if (!bound.insert(index).second)
        return error(
            "workload_contract: contract port is bound more than once");
      argumentPort.push_back(index);
      argumentIsOutputPort.push_back(role.getValue() == "output");
    }
    if (bound.size() != contract.ports.size())
      return error(
          "workload_contract: kernel does not bind every contract port");
    auto results = kernel->getAttrOfType<ArrayAttr>("result_bindings");
    if (!results || results.size() != type.getNumResults())
      return error("workload_contract: result bindings do not cover the kernel "
                   "signature");
    for (Attribute attr : results) {
      auto port = dyn_cast<IntegerAttr>(attr);
      if (!port || port.getInt() < 0 ||
          static_cast<size_t>(port.getInt()) >= contract.ports.size())
        return error("workload_contract: malformed result binding");
      resultPort.push_back(static_cast<unsigned>(port.getInt()));
    }
  } else {
    if (contract.ports.size() != type.getNumInputs() + type.getNumResults())
      return error(
          "workload_contract: port count does not match kernel signature");
    for (unsigned i = 0; i < type.getNumInputs(); ++i) {
      argumentPort.push_back(i);
      argumentIsOutputPort.push_back(false);
    }
    for (unsigned i = 0; i < type.getNumResults(); ++i)
      resultPort.push_back(type.getNumInputs() + i);
  }

  // A port's declared direction/effect has to agree with how the signature
  // uses it, and a static extent has to match a constant port dimension. A
  // dynamic extent is bound by the invocation's symbol values.
  auto validatePort = [&](const workload::WorkloadPort &port, bool asOutput,
                          Type signatureType) -> llvm::Error {
    auto direction = asOutput ? workload::PortDirection::Output
                              : workload::PortDirection::Input;
    // An input port is read, except when it is a destination whose contents
    // are never consumed (`PortEffect::None`). Such a port is still part of
    // the ordered signature -- a local `tensor.empty` destination is passed,
    // not fabricated -- but the kernel must not read it.
    const bool effectMatches =
        asOutput ? port.effect == workload::PortEffect::Write
                 : port.effect == workload::PortEffect::Read ||
                       port.effect == workload::PortEffect::None;
    if (port.direction != direction || !effectMatches)
      return error("workload_contract: port '" + port.name +
                   "' direction/effect does not match tensor signature");
    auto tensor = dyn_cast<RankedTensorType>(signatureType);
    if (!tensor || tensor.getRank() != static_cast<int64_t>(port.shape.size()))
      return error("workload_contract: port '" + port.name +
                   "' shape does not match tensor signature");
    for (size_t dim = 0; dim < port.shape.size(); ++dim) {
      int64_t extent = tensor.getDimSize(dim);
      if (ShapedType::isDynamic(extent)) {
        if (port.shape[dim].isConstant())
          return error("workload_contract: port '" + port.name +
                       "' dynamic dimension does not match its static shape");
        continue;
      }
      if (!port.shape[dim].isConstant() ||
          port.shape[dim].getConstant() != extent)
        return error("workload_contract: port '" + port.name +
                     "' shape does not match tensor signature");
    }
    std::string dtype;
    llvm::raw_string_ostream out(dtype);
    tensor.getElementType().print(out);
    if (dtype != port.elementType)
      return error("workload_contract: port '" + port.name +
                   "' dtype does not match kernel signature");
    return llvm::Error::success();
  };

  for (unsigned i = 0, e = argumentPort.size(); i < e; ++i) {
    const workload::WorkloadPort &port = contract.ports[argumentPort[i]];
    if (llvm::Error invalid =
            validatePort(port, argumentIsOutputPort[i], type.getInput(i)))
      return invalid;
    if (!argumentIsOutputPort[i] && port.effect == workload::PortEffect::None) {
      // Only completely unused inputs have no incoming data effect. Keep this
      // conservative until logical-only uses have an explicit effect analysis.
      if (kernel->getNumRegions() != 1 || kernel->getRegion(0).empty() ||
          i >= kernel->getRegion(0).front().getNumArguments() ||
          !kernel->getRegion(0).front().getArgument(i).use_empty())
        return error("workload_contract: port '" + port.name +
                     "' with effect none requires an unused kernel input");
    }
  }
  for (unsigned i = 0, e = resultPort.size(); i < e; ++i) {
    unsigned portIndex = resultPort[i];
    // A bound kernel updates a caller-owned destination argument; the
    // unbound v1 convention has no such argument, so its results are simply
    // the trailing output ports.
    if (hasBindings) {
      bool realized = false;
      for (unsigned a = 0; a < argumentPort.size(); ++a)
        if (argumentPort[a] == portIndex && argumentIsOutputPort[a]) {
          realized = true;
          break;
        }
      if (!realized)
        return error("workload_contract: result binds port " +
                     std::to_string(portIndex) +
                     " that no output argument realizes");
    }
    if (llvm::Error invalid =
            validatePort(contract.ports[portIndex],
                         /*asOutput=*/true, type.getResult(i)))
      return invalid;
  }
  Attribute numericalAttr = kernel->getAttr("micro.numerical_contract");
  auto numerical = dyn_cast_or_null<StringAttr>(numericalAttr);
  if ((!contract.numericalContract.empty() || numericalAttr) &&
      (!numerical || numerical.getValue() != contract.numericalContract))
    return error(
        "workload_contract: numerical metadata is missing or mismatched");

  // Requirements describe actual operations, never the diagnostic workload
  // label. Include logical and structural ops when explicitly requested too.
  llvm::StringSet<> primitives;
  kernel->walk([&](Operation *op) {
    if (op != kernel)
      primitives.insert(op->getName().getStringRef());
  });
  for (const std::string &primitive : contract.requiredPrimitives)
    if (!primitives.contains(primitive))
      return error("workload_contract: required primitive '" + primitive +
                   "' is absent from selected kernel");
  if (Attribute retained = kernel->getAttr("micro.workload_contract")) {
    auto decoded =
        workload::decodeWorkloadContract(dyn_cast<DictionaryAttr>(retained));
    if (!decoded)
      return decoded.takeError();
    if (workload::canonicalWorkloadContract(*decoded) !=
        workload::canonicalWorkloadContract(contract))
      return error(
          "workload_contract: retained source contract does not match");
  }
  return llvm::Error::success();
}

/// The direct MicroIR path: an exact, parameter-free concrete kernel. Source
/// parameter realization belongs to an adapter, so an offered parameter here is
/// an unrealized axis rather than something to guess at.
llvm::Expected<CandidateInstance>
instantiateConcreteMicro(ModuleOp source, const perf::SearchSpace &space,
                         const perf::Candidate &candidate,
                         const workload::WorkloadContract &contract,
                         const CandidateInstantiationOptions &options) {
  if (!space.params.empty())
    return error("axis_not_realized: " + space.params.front().name);

  CandidateInstance instance;
  instance.module = OwningOpRef<ModuleOp>(cast<ModuleOp>(source->clone()));
  auto selected = selectConcreteKernel(*instance.module, options);
  if (!selected)
    return selected.takeError();
  instance.kernel = *selected;
  if (llvm::Error invalid = verifyConcreteContract(instance.kernel, contract))
    return std::move(invalid);
  eraseSearchSpaces(*instance.module);
  if (failed(verify(*instance.module)))
    return error("ConcreteMicro source module verification failed");

  instance.sourceMode = options.sourceMode;
  instance.workloadContract = contract;
  instance.bindingFacts = mapping::extractBindingFacts(instance.kernel);
  instance.bindingHash =
      mapping::makeSearchBinding(candidate.id, {}).stableHash;
  // Provenance names the source the kernel came from as the compiler sees it:
  // the tuning directives are input metadata that never reach lowering, so a
  // materialized or replayed artifact must not carry them back in.
  const std::string semanticText = printOperation(*instance.module);
  const std::string canonical = workload::canonicalWorkloadContract(contract);
  StringRef symbol = SymbolTable::getSymbolName(instance.kernel).getValue();
  // Keep each full component in provenance; the length-delimited hash input
  // includes the exact selection even when two roots share a signature/body.
  std::string identity = "concrete-micro-v1:";
  for (StringRef field :
       {StringRef(semanticText), StringRef(canonical), symbol}) {
    identity += std::to_string(field.size()) + ":";
    identity.append(field.data(), field.size());
  }
  instance.sourceGraphHash = mapping::stableHash(identity);
  auto *ctx = source.getContext();
  auto i64 = IntegerType::get(ctx, 64);
  auto encoded = workload::encodeWorkloadContract(ctx, contract);
  if (!encoded)
    return encoded.takeError();
  instance.kernel->setAttr("micro.workload_contract", *encoded);
  instance.kernel->setAttr("source_symbol", StringAttr::get(ctx, symbol));
  instance.kernel->setAttr("source_semantics",
                           StringAttr::get(ctx, semanticText));
  instance.kernel->setAttr(
      "source_graph_hash",
      IntegerAttr::get(i64, llvm::APInt(64, instance.sourceGraphHash)));
  instance.kernel->setAttr("candidate", StringAttr::get(ctx, candidate.id));
  instance.kernel->setAttr(
      "candidate_hash",
      IntegerAttr::get(i64, llvm::APInt(64, instance.bindingHash)));
  auto graph = mapping::extractWorkloadGraph(instance.kernel);
  if (!graph)
    return graph.takeError();
  // Derive execution/movement requirements from the existing graph, rather
  // than maintaining a second operation classifier at the source boundary.
  for (const mapping::WorkloadNode &node : graph->getNodes())
    if (std::find(contract.requiredPrimitives.begin(),
                  contract.requiredPrimitives.end(),
                  node.opName) == contract.requiredPrimitives.end())
      return error("workload_contract: required primitives omit actual '" +
                   node.opName + "'");
  instance.workloadGraph = std::move(*graph);
  return instance;
}

/// The adapter path: the semantic source is dispatched to the one adapter that
/// accepts the selected root, which declares the parameter ownership and
/// realizes it. The receipt it returns is checked at the source stage, so a
/// mapping-owned choice is still pending here rather than assumed.
llvm::Expected<CandidateInstance>
instantiateAdapterSource(ModuleOp source, const perf::SearchSpace &space,
                         const perf::Candidate &candidate,
                         const workload::WorkloadContract &contract,
                         const CandidateInstantiationOptions &options,
                         const workload::SourceAdapterRegistry &registry) {
  CandidateInstance instance;
  instance.module = OwningOpRef<ModuleOp>(cast<ModuleOp>(source->clone()));
  eraseSearchSpaces(*instance.module);
  // Provenance names the source the kernel came from as the compiler sees it:
  // the tuning directives are input metadata that never reach lowering, so a
  // materialized or replayed artifact must not carry them back in. Capture it
  // before realization, or the text would describe the kernel this candidate
  // produced rather than the source it came from -- and two schedules of one
  // source would no longer share a source identity.
  const std::string semanticText = printOperation(*instance.module);
  const workload::SourceSelection selection{options.sourceSymbol,
                                            options.sourceRootOrdinal};
  auto root = workload::selectSourceRoot(*instance.module, selection, registry,
                                         options.adapterKey);
  if (!root)
    return root.takeError();
  auto adapter = registry.resolve(*root, options.adapterKey);
  if (!adapter)
    return adapter.takeError();
  const std::string adapterKey = (*adapter)->key().str();

  // The adapter describes what the root actually implements. Comparing that
  // against the supplied contract is what stops a label from standing in for
  // semantics: the two must agree in full, not merely by name.
  auto described = (*adapter)->describe(*root);
  if (!described)
    return described.takeError();
  if (workload::canonicalWorkloadContract(*described) !=
      workload::canonicalWorkloadContract(contract))
    return error("source_contract_mismatch: adapter '" + adapterKey +
                 "' describes a different contract");
  auto schema = (*adapter)->bindings(*root, space);
  if (!schema)
    return schema.takeError();

  // Machine-resource legality is the cheap analytical half of the verdict and
  // runs before anything is emitted, so a candidate the machine rejects is
  // reported as illegal rather than as a realization failure. The facts come
  // from the adapter, because only the family knows what its candidate
  // realizes; declaring them is not evidence they were realized, which is what
  // the receipt and the emitted IR are for.
  auto realized = (*adapter)->realizedFacts(*root, contract, space, candidate);
  if (!realized)
    return realized.takeError();
  if (!options.machine && !space.constraints.empty())
    return error("candidate legality requires a machine model");
  machine::MachineModel emptyMachine;
  perf::LegalityResult legality =
      perf::checkLegality(space, candidate, *realized,
                          options.machine ? *options.machine : emptyMachine);
  if (!legality.legal)
    return error("candidate is illegal: " + legality.reason);
  // Realization may replace the selected root or its entire source function.
  // Capture identity while the selected operation is still alive.
  const std::string symbol = enclosingSymbol(*instance.module, *root);
  auto realization =
      (*adapter)->realize(*instance.module, *root, contract, space, candidate);
  if (!realization)
    return realization.takeError();
  if (!realization->kernel)
    return error("source_adapter_produced_no_kernel: " + adapterKey);
  if (realization->kernel->getParentOfType<ModuleOp>() != *instance.module)
    return error("source adapter returned kernel outside source clone: " +
                 adapterKey);
  if (realization->kernel->getName().getStringRef() != "micro.kernel")
    return error("source_adapter_produced_non_kernel: " + adapterKey);
  if (llvm::Error invalid =
          verifyConcreteContract(realization->kernel, contract))
    return std::move(invalid);
  if (failed(verify(*instance.module)))
    return error("source adapter module verification failed: " + adapterKey);
  if (llvm::Error invalid = workload::validateBindingReceipt(
          space, candidate, *schema, realization->receipt,
          workload::ReceiptStage::Source))
    return std::move(invalid);
  instance.kernel = realization->kernel;
  instance.sourceMode = options.sourceMode;
  instance.workloadContract = contract;
  instance.schema = *schema;
  instance.receipt = realization->receipt;
  instance.bindingFacts = mapping::extractBindingFacts(instance.kernel);
  instance.bindingHash = candidateBindingHash(candidate);

  const std::string canonical = workload::canonicalWorkloadContract(contract);
  const std::string ordinal = std::to_string(options.sourceRootOrdinal);
  // Source identity is the original source content, the contract it implements,
  // and the exact adapter and root that produced this kernel.
  std::string identity = "source-adapter-v1:";
  for (StringRef field :
       {StringRef(semanticText), StringRef(canonical), StringRef(adapterKey),
        StringRef(symbol), StringRef(ordinal)}) {
    identity += std::to_string(field.size()) + ":";
    identity.append(field.data(), field.size());
  }
  instance.sourceGraphHash = mapping::stableHash(identity);
  auto graph = mapping::extractWorkloadGraph(instance.kernel);
  if (!graph)
    return graph.takeError();
  for (const mapping::WorkloadNode &node : graph->getNodes())
    if (std::find(contract.requiredPrimitives.begin(),
                  contract.requiredPrimitives.end(),
                  node.opName) == contract.requiredPrimitives.end())
      return error("workload_contract: required primitives omit actual '" +
                   node.opName + "'");
  instance.workloadGraph = std::move(*graph);

  auto *ctx = source.getContext();
  auto i64 = IntegerType::get(ctx, 64);
  auto encoded = workload::encodeWorkloadContract(ctx, contract);
  if (!encoded)
    return encoded.takeError();
  instance.kernel->setAttr("micro.workload_contract", *encoded);
  instance.kernel->setAttr("source_adapter", StringAttr::get(ctx, adapterKey));
  instance.kernel->setAttr("source_symbol", StringAttr::get(ctx, symbol));
  instance.kernel->setAttr("source_root_ordinal",
                           IntegerAttr::get(i64, options.sourceRootOrdinal));
  instance.kernel->setAttr("source_semantics",
                           StringAttr::get(ctx, semanticText));
  instance.kernel->setAttr(
      "source_graph_hash",
      IntegerAttr::get(i64, llvm::APInt(64, instance.sourceGraphHash)));
  instance.kernel->setAttr("candidate", StringAttr::get(ctx, candidate.id));
  instance.kernel->setAttr(
      "candidate_hash",
      IntegerAttr::get(i64, llvm::APInt(64, instance.bindingHash)));
  return instance;
}

} // namespace

llvm::Expected<CandidateInstance>
instantiateCandidate(ModuleOp source, const perf::SearchSpace &space,
                     const perf::Candidate &candidate,
                     const workload::WorkloadContract &contract,
                     const CandidateInstantiationOptions &options,
                     const workload::SourceAdapterRegistry &registry) {
  if (!source)
    return error("candidate instantiation requires a source module");
  if (options.sourceMode == SourceMode::Synthetic)
    return error("the workload contract path does not realize the synthetic "
                 "binder; synthetic mode is explicitly requested legacy "
                 "behavior");
  if (llvm::Error invalid = workload::validateWorkloadContract(contract))
    return std::move(invalid);
  if (llvm::Error invalid =
          workload::validateCandidateBindings(space, candidate))
    return std::move(invalid);
  if (options.sourceMode == SourceMode::ConcreteMicro)
    return instantiateConcreteMicro(source, space, candidate, contract,
                                    options);
  return instantiateAdapterSource(source, space, candidate, contract, options,
                                  registry);
}

llvm::Expected<CandidateInstance>
instantiateCandidate(ModuleOp source, const perf::SearchSpace &space,
                     const perf::Candidate &candidate,
                     const workload::WorkloadContract &contract,
                     const CandidateInstantiationOptions &options) {
  workload::SourceAdapterRegistry registry =
      workload::makeDefaultSourceAdapters();
  return instantiateCandidate(source, space, candidate, contract, options,
                              registry);
}

llvm::Expected<CandidateInstance>
instantiateCandidate(ModuleOp source, const perf::SearchSpace &space,
                     const perf::Candidate &candidate,
                     const perf::WorkloadShape &shape,
                     const CandidateInstantiationOptions &options) {
  if (!source)
    return error("candidate instantiation requires a source module");
  if (llvm::Error invalid =
          workload::validateCandidateBindings(space, candidate))
    return std::move(invalid);

  // A semantic source is realized behind the generic adapter boundary. This
  // entry point is now only a translation: the selected root still has to
  // match the supplied workload exactly, but the adapter -- not this file --
  // owns what a contraction is, and the realization itself runs the same path
  // every other source family uses.
  if (options.sourceMode == SourceMode::SemanticSource) {
    workload::SourceAdapterRegistry registry =
        workload::makeDefaultSourceAdapters();
    workload::SourceSelection selection{options.sourceSymbol,
                                        options.sourceRootOrdinal};
    auto selected = workload::selectSourceRoot(source, selection, registry,
                                               options.adapterKey);
    if (!selected)
      return selected.takeError();
    auto sourceShape = workload::describeContractionShape(*selected);
    if (!sourceShape)
      return sourceShape.takeError();
    if (sourceShape->M != shape.M || sourceShape->N != shape.N ||
        sourceShape->K != shape.K ||
        sourceShape->inputDType != shape.inputDType ||
        sourceShape->weightDType != shape.weightDType ||
        sourceShape->accumulatorDType != shape.accumulatorDType ||
        sourceShape->outputDType != shape.outputDType)
      return error("original source contraction facts do not match the "
                   "supplied workload");
    auto adapter = registry.resolve(*selected, options.adapterKey);
    if (!adapter)
      return adapter.takeError();
    auto contract = (*adapter)->describe(*selected);
    if (!contract)
      return contract.takeError();
    return instantiateCandidate(source, space, candidate, *contract, options,
                                registry);
  }

  CandidateInstance instance;
  instance.module = OwningOpRef<ModuleOp>(cast<ModuleOp>(source->clone()));
  // The search-space op is input metadata for this instantiation, not part of
  // the executable candidate. Keep it in the caller's source module but drop
  // it from the private clone so the compiler never sees an unlowered tuning
  // directive when materializing or replaying the selected kernel.
  SmallVector<Operation *> searchSpaces;
  instance.module->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "micro.search_space")
      searchSpaces.push_back(op);
  });
  for (Operation *op : searchSpaces)
    op->erase();
  instance.sourceMode = options.sourceMode;
  instance.bindingHash = candidateBindingHash(candidate);

  if (options.sourceMode == SourceMode::Synthetic) {
    auto bound = perf::bindCandidateToMicroKernel(*instance.module, space,
                                                  candidate, shape);
    if (!bound)
      return bound.takeError();
    instance.kernel = bound->kernel;
  } else if (options.sourceMode == SourceMode::ConcreteMicro) {
    if (!space.params.empty())
      return error("axis_not_realized: " + space.params.front().name);
    instance.module->walk([&](Operation *op) {
      if (!instance.kernel && op->getName().getStringRef() == "micro.kernel")
        instance.kernel = op;
    });
    if (!instance.kernel)
      return error("ConcreteMicro source has no retained micro.kernel");
    instance.bindingFacts = mapping::extractBindingFacts(instance.kernel);
    if (!instance.bindingFacts.originalWorkload)
      return error("ConcreteMicro source has no original_workload provenance");
    const perf::WorkloadShape &retained =
        *instance.bindingFacts.originalWorkload;
    if (retained.M != shape.M || retained.N != shape.N ||
        retained.K != shape.K || retained.inputDType != shape.inputDType ||
        retained.weightDType != shape.weightDType ||
        retained.accumulatorDType != shape.accumulatorDType ||
        retained.outputDType != shape.outputDType)
      return error("ConcreteMicro original workload does not match the "
                   "supplied workload");
    if (instance.bindingFacts.contractions.empty())
      return error("ConcreteMicro source has no retained contraction facts");
  }

  instance.kernel->setAttr("candidate",
                           StringAttr::get(source.getContext(), candidate.id));
  instance.kernel->setAttr(
      "candidate_hash",
      IntegerAttr::get(IntegerType::get(source.getContext(), 64),
                       llvm::APInt(64, instance.bindingHash)));

  auto graph = mapping::extractWorkloadGraph(instance.kernel);
  if (!graph)
    return graph.takeError();
  instance.workloadGraph = std::move(*graph);
  if (!instance.sourceGraphHash)
    instance.sourceGraphHash =
        mapping::computeSourceGraphHash(instance.workloadGraph);
  return instance;
}

} // namespace mlir::llk::tuning
