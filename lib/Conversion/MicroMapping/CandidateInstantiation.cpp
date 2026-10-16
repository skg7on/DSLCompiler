//===- CandidateInstantiation.cpp - Instantiate a candidate ---------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/MicroMapping/CandidateInstantiation.h"

#include "LLK/Conversion/LLKToMicro/LLKToMicro.h"
#include "LLK/Conversion/MicroMapping/SearchBindingLoader.h"
#include "LLK/Conversion/MicroMapping/TileExtent.h"
#include "LLK/Dialect/Micro/MicroEnums.h"
#include "LLK/Dialect/Micro/MicroHelpers.h"
#include "LLK/Mapping/MappingMetadata.h"
#include "LLK/Mapping/SearchBinding.h"
#include "LLK/Mapping/StableHash.h"
#include "LLK/Perf/CandidateBinding.h"
#include "LLK/Perf/Legality.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Errc.h"

#include <algorithm>
#include <map>
#include <optional>
#include <string>

using namespace mlir;

namespace mlir::llk::tuning {
namespace {

llvm::Error error(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

llvm::Expected<perf::WorkloadShape> rootShape(Operation *root) {
  bool fused = root->getName().getStringRef() == "llk.fused_swiglu";
  if (!fused && root->getName().getStringRef() != "llk.matmul" &&
      root->getName().getStringRef() != "linalg.matmul")
    return error("unsupported semantic source root '" +
                 root->getName().getStringRef() + "'");
  if (root->getNumOperands() < (fused ? 4u : 3u))
    return error("source root does not have the expected operand list");
  auto lhs = dyn_cast<ShapedType>(root->getOperand(0).getType());
  auto rhs = dyn_cast<ShapedType>(root->getOperand(1).getType());
  auto out = dyn_cast<ShapedType>(root->getOperand(fused ? 3 : 2).getType());
  if (!lhs || !rhs || !out || !lhs.hasRank() || !rhs.hasRank() ||
      !out.hasRank() || lhs.getRank() != 2 || rhs.getRank() != 2 ||
      out.getRank() != 2 || !lhs.hasStaticShape() || !rhs.hasStaticShape() ||
      !out.hasStaticShape())
    return error("source root must have static rank-2 tensor operands");
  if (lhs.getDimSize(0) != out.getDimSize(0) ||
      lhs.getDimSize(1) != rhs.getDimSize(0) ||
      rhs.getDimSize(1) != out.getDimSize(1))
    return error("source root has mismatched contraction shapes");
  perf::WorkloadShape shape;
  shape.M = lhs.getDimSize(0);
  shape.N = rhs.getDimSize(1);
  shape.K = lhs.getDimSize(1);
  auto dtypeName = [](Type type) -> std::string {
    auto dtype = micro::dtypeOfElementType(type);
    return dtype ? micro::stringifyDType(*dtype).str() : "unsupported";
  };
  shape.inputDType = dtypeName(lhs.getElementType());
  shape.weightDType = dtypeName(rhs.getElementType());
  shape.outputDType = dtypeName(out.getElementType());
  TypeAttr accum;
  if (Attribute attr = root->getAttr("accumulator_type"))
    accum = dyn_cast<TypeAttr>(attr);
  if (!accum && root->getName().getStringRef() == "linalg.matmul")
    accum = TypeAttr::get(out.getElementType());
  if (!accum)
    return error("source root has no typed accumulator_type");
  shape.accumulatorDType = dtypeName(accum.getValue());
  if (shape.inputDType == "unsupported" || shape.weightDType == "unsupported" ||
      shape.outputDType == "unsupported" ||
      shape.accumulatorDType == "unsupported")
    return error("source root uses a dtype unsupported by Micro");
  return shape;
}

llvm::Expected<Operation *> selectRoot(ModuleOp module,
                                       const CandidateInstantiationOptions &o) {
  Operation *function = SymbolTable::lookupSymbolIn(module, o.sourceSymbol);
  if (!function || function->getName().getStringRef() != "func.func")
    return error("unknown source function '" + o.sourceSymbol + "'");
  SmallVector<Operation *> roots;
  function->walk([&](Operation *op) {
    StringRef name = op->getName().getStringRef();
    if (name == "llk.matmul" || name == "llk.fused_swiglu" ||
        name == "linalg.matmul")
      roots.push_back(op);
  });
  if (o.sourceRootOrdinal >= roots.size())
    return error("source root ordinal " + Twine(o.sourceRootOrdinal) +
                 " is out of range for @" + o.sourceSymbol);
  Operation *root = roots[o.sourceRootOrdinal];
  return root;
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

llvm::Expected<ScheduleEntry> toSchedule(const perf::SearchSpace &space,
                                         const perf::Candidate &candidate) {
  ScheduleEntry schedule;
  llvm::StringSet<> realizedKinds;
  for (const perf::SearchParam &param : space.params) {
    auto integer = candidate.integer(param.name);
    auto symbol = candidate.symbol(param.name);
    if (param.kind == "integer") {
      if (!integer || symbol)
        return error("candidate value for '" + param.name +
                     "' is not an integer");
      int64_t value = *integer;
      if ((param.name == "BM" || param.name == "BN" || param.name == "BK" ||
           param.name == "vector_width" || param.name == "pipeline_stages") &&
          value <= 0)
        return error("candidate parameter '" + param.name +
                     "' must be positive");
      if (param.name == "BM")
        schedule.BM = value;
      else if (param.name == "BN")
        schedule.BN = value;
      else if (param.name == "BK")
        schedule.BK = value;
      else if (param.name == "VM") {
        return error("axis_not_realized: VM");
      } else if (param.name == "VN") {
        return error("axis_not_realized: VN");
      } else if (param.name == "vector_width")
        schedule.vector_width = value;
      else if (param.name == "num_threads") {
        if (value != 1)
          return error("axis_not_realized: num_threads");
        schedule.num_threads = value;
      } else if (param.name == "grain_size") {
        if (value != 1)
          return error("axis_not_realized: grain_size");
        schedule.grain_size = value;
      } else if (param.name == "pipeline_stages")
        schedule.pipeline_stages = value;
      else if (param.name == "prefetch_distance") {
        return error("axis_not_realized: prefetch_distance");
      } else
        return error("axis_not_realized: " + param.name);
      continue;
    }
    if (!symbol || integer)
      return error("candidate value for '" + param.name + "' is not symbolic");
    if (param.kind == "layout" || param.kind == "memory_path" ||
        param.kind == "owner_mapping" || param.kind == "fragment_shape" ||
        param.kind == "tail_policy") {
      if (!realizedKinds.insert(param.kind).second)
        return error("axis_not_realized: " + param.name);
    }
    if (param.kind == "layout")
      schedule.tile_layout = symbol->str();
    else if (param.kind == "memory_path")
      schedule.memory_path = symbol->str();
    else if (param.kind == "owner_mapping")
      schedule.owner_mapping = symbol->str();
    else if (param.kind == "fragment_shape") {
      schedule.fragment_shape = symbol->str();
      schedule.mma_shape = symbol->str();
    } else if (param.kind == "tail_policy")
      schedule.tail_policy = symbol->str();
    else
      return error("axis_not_realized: " + param.name);
  }
  if (schedule.BM <= 0 || schedule.BN <= 0 || schedule.BK <= 0)
    return error("candidate must bind positive BM, BN, and BK tiles");
  return schedule;
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
  if (contract.ports.size() != type.getNumInputs() + type.getNumResults())
    return error(
        "workload_contract: port count does not match kernel signature");
  for (size_t i = 0; i < contract.ports.size(); ++i) {
    const auto &port = contract.ports[i];
    bool input = i < type.getNumInputs();
    auto direction = input ? workload::PortDirection::Input
                           : workload::PortDirection::Output;
    auto effect =
        input ? workload::PortEffect::Read : workload::PortEffect::Write;
    if (port.direction != direction || port.effect != effect)
      return error("workload_contract: port '" + port.name +
                   "' direction/effect does not match tensor signature");
    Type signatureType =
        input ? type.getInput(i) : type.getResult(i - type.getNumInputs());
    auto tensor = dyn_cast<RankedTensorType>(signatureType);
    if (!tensor || !tensor.hasStaticShape() ||
        tensor.getShape() != ArrayRef<int64_t>(port.shape))
      return error("workload_contract: port '" + port.name +
                   "' shape does not match static tensor signature");
    std::string dtype;
    llvm::raw_string_ostream out(dtype);
    tensor.getElementType().print(out);
    if (dtype != port.elementType)
      return error("workload_contract: port '" + port.name +
                   "' dtype does not match kernel signature");
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
  const std::string semanticText = printOperation(source);
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
  instance.kernel->setAttr("micro.workload_contract",
                           workload::encodeWorkloadContract(ctx, contract));
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
  auto realization =
      (*adapter)->realize(*instance.module, *root, contract, space, candidate);
  if (!realization)
    return realization.takeError();
  if (!realization->kernel)
    return error("source_adapter_produced_no_kernel: " + adapterKey);
  if (llvm::Error invalid = workload::validateBindingReceipt(
          space, candidate, *schema, realization->receipt,
          workload::ReceiptStage::Source))
    return std::move(invalid);
  instance.kernel = realization->kernel;
  instance.sourceMode = options.sourceMode;
  instance.workloadContract = contract;
  instance.receipt = realization->receipt;
  instance.bindingFacts = mapping::extractBindingFacts(instance.kernel);
  instance.bindingHash = candidateBindingHash(candidate);

  const std::string semanticText = printOperation(source);
  const std::string canonical = workload::canonicalWorkloadContract(contract);
  const std::string symbol = enclosingSymbol(*instance.module, *root);
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
  instance.kernel->setAttr("micro.workload_contract",
                           workload::encodeWorkloadContract(ctx, contract));
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
  } else {
    auto selected = selectRoot(*instance.module, options);
    if (!selected)
      return selected.takeError();
    auto sourceShape = rootShape(*selected);
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
    instance.bindingFacts.originalWorkload = *sourceShape;
    std::optional<int64_t> bm = candidate.integer("BM");
    std::optional<int64_t> bn = candidate.integer("BN");
    std::optional<int64_t> bk = candidate.integer("BK");
    std::optional<StringRef> tail = std::nullopt;
    if (const perf::SearchParam *tailParam =
            space.findParamOfKind("tail_policy"))
      tail = candidate.symbol(tailParam->name);
    if (!bm || !bn || !bk)
      return error(
          "candidate must bind BM, BN, and BK before legality checking");
    const bool padded = tail && *tail == "pad";
    perf::WorkloadShape contraction = *sourceShape;
    contraction.M = padded ? *bm : std::min(*bm, sourceShape->M);
    contraction.N = padded ? *bn : std::min(*bn, sourceShape->N);
    contraction.K = padded ? *bk : std::min(*bk, sourceShape->K);
    contraction.outputDType = sourceShape->accumulatorDType;
    instance.bindingFacts.contractions.push_back(contraction);
    if ((*selected)->getName().getStringRef() == "llk.fused_swiglu")
      instance.bindingFacts.contractions.push_back(contraction);

    if (!options.machine && !space.constraints.empty())
      return error("candidate legality requires a machine model");
    machine::MachineModel emptyMachine;
    perf::LegalityResult legality =
        perf::checkLegality(space, candidate, instance.bindingFacts,
                            options.machine ? *options.machine : emptyMachine);
    if (!legality.legal)
      return error("candidate is illegal: " + legality.reason);
    auto schedule = toSchedule(space, candidate);
    if (!schedule)
      return schedule.takeError();

    std::string semanticText = printOperation(*selected);
    instance.sourceGraphHash = mapping::stableHash(semanticText);
    auto exported =
        exportMicroKernelFromSchedule(*instance.module, options.sourceSymbol,
                                      options.sourceRootOrdinal, *schedule);
    if (!exported)
      return exported.takeError();
    instance.kernel = *exported;
    llvm::SmallVector<SmallVector<int64_t, 3>, 2> emittedContractions;
    instance.kernel->walk([&](Operation *op) {
      if (op->getName().getStringRef() != "micro.mma")
        return;
      auto dims = op->getAttrOfType<DenseI64ArrayAttr>("shape");
      if (dims && dims.size() == 3)
        emittedContractions.push_back({dims[0], dims[1], dims[2]});
    });
    if (emittedContractions.size() != instance.bindingFacts.contractions.size())
      return error(
          "bound Micro-IR contraction count does not match source semantics");
    for (size_t i = 0; i < emittedContractions.size(); ++i) {
      const perf::WorkloadShape &fact = instance.bindingFacts.contractions[i];
      if (emittedContractions[i][0] != fact.M ||
          emittedContractions[i][1] != fact.N ||
          emittedContractions[i][2] != fact.K)
        return error("bound Micro-IR contraction facts do not match the "
                     "candidate and original extents");
    }
    perf::BindingFacts emittedFacts =
        mapping::extractBindingFacts(instance.kernel);
    auto sameShape = [](const perf::WorkloadShape &lhs,
                        const perf::WorkloadShape &rhs) {
      return lhs.M == rhs.M && lhs.N == rhs.N && lhs.K == rhs.K &&
             lhs.inputDType == rhs.inputDType &&
             lhs.weightDType == rhs.weightDType &&
             lhs.accumulatorDType == rhs.accumulatorDType &&
             lhs.outputDType == rhs.outputDType;
    };
    if (!emittedFacts.originalWorkload ||
        !sameShape(*emittedFacts.originalWorkload, *sourceShape) ||
        emittedFacts.contractions.size() !=
            instance.bindingFacts.contractions.size())
      return error("exported original/bound contraction facts do not match the "
                   "selected source root");
    for (size_t i = 0; i < emittedFacts.contractions.size(); ++i)
      if (!sameShape(emittedFacts.contractions[i],
                     instance.bindingFacts.contractions[i]))
        return error("exported contraction dtype or extent does not match the "
                     "selected source root");
    instance.bindingFacts = std::move(emittedFacts);
    instance.kernel->setAttr(
        "source_symbol",
        StringAttr::get(source.getContext(), options.sourceSymbol));
    instance.kernel->setAttr(
        "source_root_ordinal",
        IntegerAttr::get(IntegerType::get(source.getContext(), 64),
                         options.sourceRootOrdinal));
    instance.kernel->setAttr(
        "source_graph_hash",
        IntegerAttr::get(IntegerType::get(source.getContext(), 64),
                         llvm::APInt(64, instance.sourceGraphHash)));
    instance.kernel->setAttr(
        "source_semantics", StringAttr::get(source.getContext(), semanticText));
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
