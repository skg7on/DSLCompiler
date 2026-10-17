//===- ContractionSourceAdapter.cpp - Contraction source family -----------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework B4: the current LLK/Linalg contraction export, wrapped behind the
// generic source-adapter boundary. The realization itself is unchanged -- the
// same operator-kernel registry describes the root, the same
// `exportMicroKernelFromSchedule` emits the kernel, and the same contraction
// facts are checked against it -- but no generic code has to know that any of
// it is contraction-specific any more.
//
// Two things are deliberately not duplicated from the old inline path:
//
//   * The recognized family is the set of roots the operator-kernel registry
//     gives a Micro-IR lowering, so `accepts` is a capability query rather than
//     a list of operation names.
//   * The described contract's required primitives are the primitives this
//     export actually emits. A required primitive the export never emits is a
//     false contract, and the adapter is where the export's behaviour is known.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/Workload/ContractionSourceAdapter.h"

#include "LLK/Conversion/LLKKernel/LLKKernelLowering.h"
#include "LLK/Conversion/LLKToMicro/LLKToMicro.h"
#include "LLK/Dialect/LLKEnums.h"
#include "LLK/Dialect/Micro/MicroEnums.h"
#include "LLK/Dialect/Micro/MicroHelpers.h"
#include "LLK/Perf/Legality.h"
#include "LLK/Transforms/Common/ScheduleLoader.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;

namespace mlir::llk::workload {
namespace {

llvm::Error error(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

constexpr llvm::StringLiteral contractionKey = "llk-contraction-v1";

/// The per-operator knowledge, taken from the registry that owns it rather than
/// re-derived from operand positions. `hasMicroLowering` is what makes a root
/// part of this family, so an operator that lowers only to Linalg is not
/// claimed here.
const LLKKernelLowering *microLowering(const LLKKernelRegistry &registry,
                                       llvm::StringRef name) {
  const LLKKernelLowering *lowering = registry.lookup(name);
  return lowering && lowering->hasMicroLowering() ? lowering : nullptr;
}

LLKKernelRegistry makeKernelRegistry() {
  LLKKernelRegistry registry;
  populateLLKKernelLowerings(registry);
  return registry;
}

bool acceptsContraction(Operation *root) {
  if (!root)
    return false;
  LLKKernelRegistry registry = makeKernelRegistry();
  return microLowering(registry, root->getName().getStringRef()) != nullptr;
}

struct ContractionFacts {
  KernelContract contract;
  MicroKernelFragment fragment;
};

/// The contract and fragment the source root declares. `inferContract` and
/// `describeMicroKernel` are the same calls the export makes, so a root this
/// adapter describes is a root the export can emit.
llvm::Expected<ContractionFacts> contractionFacts(Operation *root) {
  if (!root)
    return error("a contraction source root is required");
  LLKKernelRegistry registry = makeKernelRegistry();
  const LLKKernelLowering *lowering =
      microLowering(registry, root->getName().getStringRef());
  if (!lowering)
    return error("unsupported semantic source root '" +
                 root->getName().getStringRef() + "'");
  ContractionFacts facts;
  if (failed(lowering->inferContract(root, facts.contract)))
    return error("source root does not describe a supported contraction");
  if (failed(
          lowering->describeMicroKernel(root, facts.contract, facts.fragment)))
    return error("source root has no Micro-IR lowering");
  return facts;
}

/// The Micro dtype name for an element type. A type Micro does not model is
/// reported as `unsupported` so the contract validator, not this adapter,
/// produces the diagnostic.
std::string dtypeName(Type type) {
  auto dtype = micro::dtypeOfElementType(type);
  return dtype ? micro::stringifyDType(*dtype).str()
               : std::string("unsupported");
}

std::vector<int64_t> shapeOf(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped || !shaped.hasStaticShape())
    return {};
  return std::vector<int64_t>(shaped.getShape().begin(),
                              shaped.getShape().end());
}

/// The scalar element type of a shaped value. The registry reports the
/// right-hand arms as operand types, while the contract's own dtypes are
/// already element types, so both spellings have to reach the same answer.
Type elementTypeOf(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type))
    return shaped.getElementType();
  return type;
}

/// The canonical numerical policy the root declares, or empty when it declares
/// none. The Micro export already records this as `source_math_mode`; carrying
/// it as the contract's numerical semantics is what makes an approximation an
/// explicit IR fact rather than a compiler-flag side effect.
std::string numericalContractName(Operation *root) {
  auto mode = dyn_cast_or_null<IntegerAttr>(root->getAttr("math_mode"));
  if (!mode)
    return std::string();
  return stringifyMathMode(static_cast<MathMode>(mode.getInt())).str();
}

/// The primitives this export emits for `facts`. The movement and compute core
/// is unconditional; a vector operation appears only when the export material-
/// izes a conversion or an activation epilogue, which is exactly the condition
/// the exporter uses.
std::vector<std::string> emittedPrimitives(const ContractionFacts &facts) {
  std::vector<std::string> primitives{"micro.mma", "micro.tile_async_copy",
                                      "micro.tile_store"};
  const bool epilogue = facts.fragment.armCount > 1;
  const bool conversion =
      facts.contract.accumulatorElemType != facts.contract.outputElemType;
  if (epilogue || conversion)
    primitives.push_back("micro.vector");
  return primitives;
}

/// The parameter ownership this family declares. A source-realized parameter's
/// role names the contraction axis it tiles (`BM` tiles `M`, `BK` tiles `K`),
/// and for the execution choices that govern no tensor axis it names the
/// parameter itself. Declaring is what the boundary asks for; nothing here
/// infers a binding from a parameter's spelling at dispatch time. Layout,
/// memory path and owner mapping stay mapping-owned -- the source stage does not
/// claim placement it has not bound.
std::vector<ParameterBinding>
contractionSchema(const perf::SearchSpace &space) {
  std::vector<ParameterBinding> schema;
  schema.reserve(space.params.size());
  for (const perf::SearchParam &param : space.params) {
    const bool mappingKind = param.kind == "layout" ||
                             param.kind == "owner_mapping" ||
                             param.kind == "memory_path";
    ParameterBinding binding;
    binding.parameter = param.name;
    binding.owner = mappingKind ? BindingOwner::Mapping : BindingOwner::Source;
    if (!param.role.empty())
      binding.role = param.role;
    else if (mappingKind)
      binding.role = param.kind;
    else if (param.name == "BM")
      binding.role = "axis:M";
    else if (param.name == "BN")
      binding.role = "axis:N";
    else if (param.name == "BK")
      binding.role = "axis:K";
    else
      binding.role = "axis:" + param.name;
    schema.push_back(std::move(binding));
  }
  return schema;
}

/// The symbol of the module-level operation that contains `nested`. Source
/// identity names the function a root belongs to, never the emitted kernel.
std::string enclosingSymbol(Operation *nested) {
  Operation *owner = nested;
  while (owner->getParentOp() && !isa<ModuleOp>(owner->getParentOp()))
    owner = owner->getParentOp();
  StringAttr symbol = SymbolTable::getSymbolName(owner);
  return symbol ? symbol.getValue().str() : std::string();
}

/// The zero-based position of `root` among the accepted top-level roots of the
/// function that contains it. Realization is handed the operation, not the
/// selection that found it, so the export's own selection key is recomputed
/// with the same predicate `selectSourceRoot` counted with.
uint64_t rootOrdinalIn(Operation *root) {
  Operation *function = root->getParentOp();
  while (function && function->getParentOp() &&
         !isa<ModuleOp>(function->getParentOp()))
    function = function->getParentOp();
  if (!function || function->getNumRegions() == 0 ||
      function->getRegion(0).empty())
    return 0;
  LLKKernelRegistry registry = makeKernelRegistry();
  uint64_t ordinal = 0;
  for (Operation &op : function->getRegion(0).front().getOperations()) {
    if (&op == root)
      return ordinal;
    if (microLowering(registry, op.getName().getStringRef()))
      ++ordinal;
  }
  return ordinal;
}

/// Turns a candidate into the schedule the contraction export consumes. A
/// parameter this family cannot realize is an explicit `axis_not_realized`
/// rather than a value the export silently ignores.
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

/// The workload facts a candidate realizes. The source's own extents are the
/// authority -- the legacy WorkloadShape entry point compares them before it
/// gets here, and a direct contract caller gets the same source -- and every
/// emitted MMA performs the tile the candidate binds, narrowed to the source
/// extent unless the tail policy pads.
llvm::Expected<perf::BindingFacts>
realizedContractionFacts(Operation *root, const perf::SearchSpace &space,
                         const perf::Candidate &candidate) {
  auto facts = contractionFacts(root);
  if (!facts)
    return facts.takeError();

  perf::WorkloadShape sourceShape;
  sourceShape.M = facts->contract.M;
  sourceShape.N = facts->contract.N;
  sourceShape.K = facts->contract.K;
  sourceShape.inputDType = dtypeName(facts->contract.inputElemType);
  sourceShape.weightDType =
      facts->fragment.rhsOperandTypes.empty()
          ? std::string()
          : dtypeName(elementTypeOf(facts->fragment.rhsOperandTypes.front()));
  sourceShape.accumulatorDType = dtypeName(facts->contract.accumulatorElemType);
  sourceShape.outputDType = dtypeName(facts->contract.outputElemType);

  std::optional<int64_t> bm = candidate.integer("BM");
  std::optional<int64_t> bn = candidate.integer("BN");
  std::optional<int64_t> bk = candidate.integer("BK");
  std::optional<StringRef> tail = std::nullopt;
  if (const perf::SearchParam *tailParam = space.findParamOfKind("tail_policy"))
    tail = candidate.symbol(tailParam->name);
  if (!bm || !bn || !bk)
    return error("candidate must bind BM, BN, and BK before legality checking");
  const bool padded = tail && *tail == "pad";
  perf::WorkloadShape contraction = sourceShape;
  contraction.M = padded ? *bm : std::min(*bm, sourceShape.M);
  contraction.N = padded ? *bn : std::min(*bn, sourceShape.N);
  contraction.K = padded ? *bk : std::min(*bk, sourceShape.K);
  contraction.outputDType = sourceShape.accumulatorDType;

  perf::BindingFacts realized;
  realized.originalWorkload = sourceShape;
  // One emitted contraction per right-hand arm: the gated projection is two
  // MMAs over the same extents, never one widened one.
  for (unsigned arm = 0; arm < facts->fragment.armCount; ++arm)
    realized.contractions.push_back(contraction);
  return realized;
}

/// The one family the current exporter supports, behind the generic boundary.
class ContractionSourceAdapter final : public SourceAdapter {
public:
  llvm::StringRef key() const override { return contractionKey; }

  bool accepts(Operation *root) const override {
    return acceptsContraction(root);
  }

  llvm::Expected<WorkloadContract> describe(Operation *root) const override {
    auto facts = contractionFacts(root);
    if (!facts)
      return facts.takeError();
    const KernelContract &contract = facts->contract;
    if (contract.M <= 0 || contract.N <= 0 || contract.K <= 0)
      return error("source root has a non-positive contraction extent");

    WorkloadContract described;
    described.axes = {{"M", contract.M}, {"N", contract.N}, {"K", contract.K}};
    // Ports follow the emitted kernel's signature: the left operand, one port
    // per right-hand arm, the caller's initialized destination when the
    // lowering takes one, and the written result.
    described.ports.push_back({"lhs",
                               PortDirection::Input,
                               PortEffect::Read,
                               {contract.M, contract.K},
                               dtypeName(contract.inputElemType)});
    if (facts->fragment.rhsOperandTypes.size() < facts->fragment.armCount)
      return error("source root declares more contraction arms than operands");
    for (size_t arm = 0; arm < facts->fragment.armCount; ++arm) {
      Type armType = facts->fragment.rhsOperandTypes[arm];
      described.ports.push_back(
          {"rhs" + std::to_string(arm), PortDirection::Input, PortEffect::Read,
           shapeOf(armType), dtypeName(elementTypeOf(armType))});
    }
    if (facts->fragment.hasOutputInit)
      described.ports.push_back({"init",
                                 PortDirection::Input,
                                 PortEffect::Read,
                                 {contract.M, contract.N},
                                 dtypeName(contract.outputElemType)});
    described.ports.push_back({"output",
                               PortDirection::Output,
                               PortEffect::Write,
                               {contract.M, contract.N},
                               dtypeName(contract.outputElemType)});
    described.numericalContract = numericalContractName(root);
    described.requiredPrimitives = emittedPrimitives(*facts);
    return described;
  }

  llvm::Expected<std::vector<ParameterBinding>>
  bindings(Operation *, const perf::SearchSpace &space) const override {
    return contractionSchema(space);
  }

  llvm::Expected<perf::BindingFacts>
  realizedFacts(Operation *selectedRoot, const WorkloadContract &,
                const perf::SearchSpace &space,
                const perf::Candidate &candidate) const override {
    return realizedContractionFacts(selectedRoot, space, candidate);
  }

  llvm::Expected<SourceRealization>
  realize(ModuleOp clone, Operation *selectedRoot,
          const WorkloadContract &contract, const perf::SearchSpace &space,
          const perf::Candidate &candidate) const override {
    auto realized = realizedContractionFacts(selectedRoot, space, candidate);
    if (!realized)
      return realized.takeError();
    const std::vector<perf::WorkloadShape> &expected = realized->contractions;

    auto schedule = toSchedule(space, candidate);
    if (!schedule)
      return schedule.takeError();

    auto exported =
        exportMicroKernelFromSchedule(clone, enclosingSymbol(selectedRoot),
                                      rootOrdinalIn(selectedRoot), *schedule);
    if (!exported)
      return exported.takeError();
    Operation *kernel = *exported;

    llvm::SmallVector<llvm::SmallVector<int64_t, 3>, 2> emitted;
    kernel->walk([&](Operation *op) {
      if (op->getName().getStringRef() != "micro.mma")
        return;
      auto dims = op->getAttrOfType<DenseI64ArrayAttr>("shape");
      if (dims && dims.size() == 3)
        emitted.push_back({dims[0], dims[1], dims[2]});
    });
    if (emitted.size() != expected.size())
      return error(
          "bound Micro-IR contraction count does not match source semantics");
    for (size_t i = 0; i < emitted.size(); ++i) {
      if (emitted[i][0] != expected[i].M || emitted[i][1] != expected[i].N ||
          emitted[i][2] != expected[i].K)
        return error("bound Micro-IR contraction facts do not match the "
                     "candidate and original extents");
    }

    // The contract's numerical semantics have to be observable in the IR, or a
    // consumer cannot tell a bounded approximation from an exact result.
    if (!contract.numericalContract.empty())
      kernel->setAttr(
          "micro.numerical_contract",
          StringAttr::get(kernel->getContext(), contract.numericalContract));

    SourceRealization realization;
    realization.kernel = kernel;
    // The receipt names what realization actually produced. Every source-owned
    // parameter reached `toSchedule`, which either consumed it or rejected it.
    for (const ParameterBinding &binding : contractionSchema(space))
      if (binding.owner == BindingOwner::Source)
        realization.receipt.source.push_back(binding.parameter);
    return realization;
  }
};

} // namespace

llvm::StringRef contractionSourceAdapterKey() { return contractionKey; }

std::unique_ptr<SourceAdapter> makeContractionSourceAdapter() {
  return std::make_unique<ContractionSourceAdapter>();
}

llvm::Expected<perf::WorkloadShape>
describeContractionShape(mlir::Operation *root) {
  auto facts = contractionFacts(root);
  if (!facts)
    return facts.takeError();
  const KernelContract &contract = facts->contract;
  perf::WorkloadShape shape;
  shape.M = contract.M;
  shape.N = contract.N;
  shape.K = contract.K;
  shape.inputDType = dtypeName(contract.inputElemType);
  shape.weightDType =
      facts->fragment.rhsOperandTypes.empty()
          ? std::string()
          : dtypeName(elementTypeOf(facts->fragment.rhsOperandTypes.front()));
  shape.accumulatorDType = dtypeName(contract.accumulatorElemType);
  shape.outputDType = dtypeName(contract.outputElemType);
  if (shape.inputDType == "unsupported" || shape.weightDType == "unsupported" ||
      shape.outputDType == "unsupported" ||
      shape.accumulatorDType == "unsupported")
    return error("source root uses a dtype unsupported by Micro");
  return shape;
}

} // namespace mlir::llk::workload
