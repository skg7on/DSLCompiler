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

llvm::Error validateCandidate(const perf::SearchSpace &space,
                              const perf::Candidate &candidate) {
  if (candidate.values.size() + candidate.symbolicValues.size() !=
      space.params.size())
    return error("candidate binding is incomplete or contains undeclared axes");
  for (const perf::SearchParam &param : space.params) {
    bool hasInteger = candidate.values.count(param.name) != 0;
    bool hasSymbol = candidate.symbolicValues.count(param.name) != 0;
    if (hasInteger == hasSymbol)
      return error("candidate must bind parameter '" + param.name +
                   "' exactly once");
    for (const perf::SearchChoice &choice : param.choices) {
      if (hasInteger && choice.isInteger() &&
          choice.integer() == candidate.values.at(param.name))
        goto in_domain;
      if (hasSymbol && choice.isSymbolic() &&
          choice.symbol() == candidate.symbolicValues.at(param.name))
        goto in_domain;
    }
    return error("candidate value for '" + param.name +
                 "' is not one of the declared choices");
  in_domain:;
  }
  return llvm::Error::success();
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

} // namespace

llvm::Expected<CandidateInstance>
instantiateCandidate(ModuleOp source, const perf::SearchSpace &space,
                     const perf::Candidate &candidate,
                     const perf::WorkloadShape &shape,
                     const CandidateInstantiationOptions &options) {
  if (!source)
    return error("candidate instantiation requires a source module");
  if (llvm::Error invalid = validateCandidate(space, candidate))
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
  mapping::SearchBinding binding =
      mapping::makeSearchBinding(candidate.id, [&] {
        llvm::StringMap<mapping::SearchValue> values;
        for (const auto &entry : candidate.values)
          values[entry.first] = entry.second;
        for (const auto &entry : candidate.symbolicValues)
          values[entry.first] = entry.second;
        return values;
      }());
  instance.bindingHash = binding.stableHash;

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
