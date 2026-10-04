//===- CostModel.cpp - Multi-dimensional mapping cost ---------------------===//

#include "LLK/Mapping/CostModel.h"

#include "LLK/Machine/MachineModel.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>

namespace mlir::llk::mapping {

namespace {
struct MetricInfo {
  CostMetric metric;
  llvm::StringRef name;
};

/// Declaration order is the canonical order used by `canonicalCostString`.
constexpr std::array<MetricInfo, 6> kMetrics{{
    {CostMetric::LatencyCycles, "latency_cycles"},
    {CostMetric::DramBytes, "dram_bytes"},
    {CostMetric::LocalBytes, "local_bytes"},
    {CostMetric::SpillBytes, "spill_bytes"},
    {CostMetric::ComputeUtilization, "compute_utilization"},
    {CostMetric::TransferUtilization, "transfer_utilization"},
}};

/// Resolves a `micro.objective` metric spelling. The Micro dialect verifier
/// (`ObjectiveOp::verify`) accepts six names; only two coincide with a
/// `CostMetric` spelling, so the four that differ are aliased here. A
/// `CostMetric` spelling is also accepted, so the bridge is usable from either
/// vocabulary.
std::optional<CostMetric> symbolizeMicroMetric(llvm::StringRef text) {
  if (text == "latency_cycles")
    return CostMetric::LatencyCycles;
  if (text == "dram_bytes")
    return CostMetric::DramBytes;
  if (text == "sram_bytes")
    return CostMetric::LocalBytes;
  if (text == "matrix_utilization")
    return CostMetric::ComputeUtilization;
  if (text == "dma_utilization")
    return CostMetric::TransferUtilization;
  if (text == "capacity_spill_bytes")
    return CostMetric::SpillBytes;
  return symbolizeCostMetric(text);
}

/// The machine's synchronization period in cycles, or nullopt when it models no
/// barrier and no wait (so there is no window to measure utilization against).
std::optional<double>
machineSyncPeriodCycles(const machine::MachineModel &machine) {
  uint64_t cycles = machine.sync.barrierCycles + machine.sync.waitCycles;
  if (cycles == 0)
    return std::nullopt;
  return static_cast<double>(cycles);
}

/// The machine spelling of an element type (`f32`, `bf16`, `i8`), or empty when
/// the type has no width a transform could move. This mirrors the micro dtype
/// vocabulary without depending on the dialect: the cost model sits below it,
/// so it names element widths itself.
std::string elementTypeName(mlir::Type type) {
  if (auto floatType = llvm::dyn_cast<mlir::FloatType>(type)) {
    if (floatType.isF64())
      return "f64";
    if (floatType.isF32())
      return "f32";
    if (floatType.isF16())
      return "f16";
    if (floatType.isBF16())
      return "bf16";
    return "f" + std::to_string(floatType.getWidth());
  }
  if (auto intType = llvm::dyn_cast<mlir::IntegerType>(type))
    return "i" + std::to_string(intType.getWidth());
  return {};
}

/// Bytes of one element, or nullopt for an element type with no width.
std::optional<unsigned> elementByteWidth(mlir::Type type) {
  if (auto floatType = llvm::dyn_cast<mlir::FloatType>(type))
    return static_cast<unsigned>(llvm::divideCeil(floatType.getWidth(), 8u));
  if (auto intType = llvm::dyn_cast<mlir::IntegerType>(type))
    return static_cast<unsigned>(llvm::divideCeil(intType.getWidth(), 8u));
  return std::nullopt;
}

/// A transform footprint must be a shaped value with static extents so the
/// element count is a checked fact rather than a guess.
mlir::ShapedType transformFootprintType(const TransformCostInput &input) {
  for (mlir::Type candidate : {input.outputType, input.inputType})
    if (candidate)
      if (auto shaped = llvm::dyn_cast<mlir::ShapedType>(candidate))
        return shaped;
  return {};
}
} // namespace

llvm::StringRef stringifyCostMetric(CostMetric metric) {
  for (const MetricInfo &info : kMetrics)
    if (info.metric == metric)
      return info.name;
  return "";
}

std::optional<CostMetric> symbolizeCostMetric(llvm::StringRef text) {
  for (const MetricInfo &info : kMetrics)
    if (info.name == text)
      return info.metric;
  return std::nullopt;
}

double costMetric(const Cost &cost, CostMetric metric) {
  switch (metric) {
  case CostMetric::LatencyCycles:
    return cost.latencyCycles;
  case CostMetric::DramBytes:
    return static_cast<double>(cost.dramBytes);
  case CostMetric::LocalBytes:
    return static_cast<double>(cost.localBytes);
  case CostMetric::SpillBytes:
    return static_cast<double>(cost.spillBytes);
  case CostMetric::ComputeUtilization:
    return cost.computeUtilization;
  case CostMetric::TransferUtilization:
    return cost.transferUtilization;
  }
  return 0.0;
}

Cost addCost(const Cost &lhs, const Cost &rhs) {
  Cost sum;
  sum.latencyCycles = lhs.latencyCycles + rhs.latencyCycles;
  sum.dramBytes = lhs.dramBytes + rhs.dramBytes;
  sum.localBytes = lhs.localBytes + rhs.localBytes;
  sum.spillBytes = lhs.spillBytes + rhs.spillBytes;
  sum.computeUtilization = lhs.computeUtilization + rhs.computeUtilization;
  sum.transferUtilization = lhs.transferUtilization + rhs.transferUtilization;
  return sum;
}

std::optional<double> utilizationEstimate(double busyCycles,
                                          const machine::MachineModel &machine,
                                          uint32_t parallelUnits) {
  if (parallelUnits == 0)
    return std::nullopt;
  std::optional<double> period = machineSyncPeriodCycles(machine);
  if (!period)
    return std::nullopt;
  return busyCycles / (*period * static_cast<double>(parallelUnits));
}

bool costLess(const Cost &lhs, const Cost &rhs, const ObjectiveOrder &order) {
  auto compare = [&](CostMetric metric) -> int {
    double l = costMetric(lhs, metric);
    double r = costMetric(rhs, metric);
    if (l == r)
      return 0;
    bool lhsFirst = order.minimize ? (l < r) : (l > r);
    return lhsFirst ? -1 : 1;
  };

  if (int verdict = compare(order.primary); verdict != 0)
    return verdict < 0;
  for (CostMetric metric : order.secondary)
    if (int verdict = compare(metric); verdict != 0)
      return verdict < 0;
  return false;
}

bool ranksBefore(const Cost &lhs, uint64_t lhsId, const Cost &rhs,
                 uint64_t rhsId, const ObjectiveOrder &order) {
  if (costLess(lhs, rhs, order))
    return true;
  if (costLess(rhs, lhs, order))
    return false;
  return lhsId < rhsId;
}

Cost bestCostForObjective(const Cost &lhs, const Cost &rhs,
                          const ObjectiveOrder &order) {
  auto pick = [&](double l, double r) {
    return order.minimize ? std::min(l, r) : std::max(l, r);
  };
  auto pickUnsigned = [&](uint64_t l, uint64_t r) {
    return order.minimize ? std::min(l, r) : std::max(l, r);
  };
  Cost best;
  best.latencyCycles = pick(lhs.latencyCycles, rhs.latencyCycles);
  best.dramBytes = pickUnsigned(lhs.dramBytes, rhs.dramBytes);
  best.localBytes = pickUnsigned(lhs.localBytes, rhs.localBytes);
  best.spillBytes = pickUnsigned(lhs.spillBytes, rhs.spillBytes);
  best.computeUtilization =
      pick(lhs.computeUtilization, rhs.computeUtilization);
  best.transferUtilization =
      pick(lhs.transferUtilization, rhs.transferUtilization);
  return best;
}

Cost infiniteCost() {
  Cost cost;
  cost.latencyCycles = std::numeric_limits<double>::infinity();
  cost.dramBytes = std::numeric_limits<uint64_t>::max();
  cost.localBytes = std::numeric_limits<uint64_t>::max();
  cost.spillBytes = std::numeric_limits<uint64_t>::max();
  cost.computeUtilization = std::numeric_limits<double>::infinity();
  cost.transferUtilization = std::numeric_limits<double>::infinity();
  return cost;
}

bool boundIsDead(const Cost &cost) {
  // The `infiniteCost()` sentinel saturates *every* dimension. Keying only on
  // `latencyCycles` would be fragile: under a maximize objective an infinite
  // latency is the *best* value, so a single-dimension test could mistake a
  // live bound for a dead one. Requiring all six dimensions to be saturated
  // cannot collide with a real completion, whose costs are finite.
  return std::isinf(cost.latencyCycles) &&
         std::isinf(cost.computeUtilization) &&
         std::isinf(cost.transferUtilization) &&
         cost.dramBytes == std::numeric_limits<uint64_t>::max() &&
         cost.localBytes == std::numeric_limits<uint64_t>::max() &&
         cost.spillBytes == std::numeric_limits<uint64_t>::max();
}

bool boundIsBetterThan(const Cost &lhs, const Cost &rhs,
                       const ObjectiveOrder &order) {
  if (boundIsDead(lhs))
    return false;
  if (boundIsDead(rhs))
    return true;
  return costLess(lhs, rhs, order);
}

llvm::Expected<ObjectiveOrder>
objectiveOrderFromMicro(llvm::StringRef metric, bool minimize,
                        llvm::ArrayRef<llvm::StringRef> secondary) {
  std::optional<CostMetric> primary = symbolizeMicroMetric(metric);
  if (!primary)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        llvm::Twine("micro.objective names an unknown metric '") + metric +
            "'");

  ObjectiveOrder order;
  order.primary = *primary;
  order.minimize = minimize;
  order.secondary.reserve(secondary.size());
  for (llvm::StringRef spelling : secondary) {
    std::optional<CostMetric> resolved = symbolizeMicroMetric(spelling);
    if (!resolved)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          llvm::Twine("micro.objective names an unknown secondary metric '") +
              spelling + "'");
    order.secondary.push_back(*resolved);
  }
  return order;
}

std::string canonicalCostString(const Cost &cost) {
  std::string out;
  for (const MetricInfo &info : kMetrics) {
    if (!out.empty())
      out += ';';
    double value = costMetric(cost, info.metric);
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.6f", value);
    out += info.name.str();
    out += '=';
    out += buffer;
  }
  return out;
}

llvm::Expected<Cost>
estimateTransformCost(const TransformCostInput &input,
                      const machine::MachineModel &machine) {
  auto fail = [](const llvm::Twine &message) {
    return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
  };

  // Footprint first: a value whose size cannot be computed is reported, never
  // charged as zero.
  mlir::ShapedType footprint = transformFootprintType(input);
  if (!footprint)
    return fail("layout transform cost: neither the input nor the output type "
                "is a shaped type, so the transform footprint is unknown");
  if (!footprint.hasStaticShape())
    return fail("layout transform cost: a dynamic extent leaves the transform "
                "footprint unknown");
  std::string dtype = elementTypeName(footprint.getElementType());
  std::optional<unsigned> width = elementByteWidth(footprint.getElementType());
  if (dtype.empty() || !width)
    return fail("layout transform cost: the element type is not a sized float "
                "or integer, so the transform footprint is unknown");
  uint64_t elements = static_cast<uint64_t>(footprint.getNumElements());
  uint64_t bytes = elements * static_cast<uint64_t>(*width);

  // The selected capability. A name the machine does not model is a diagnostic;
  // an unnamed resource falls back to the machine's declared vector engine --
  // an explicit default policy, never a branch on a target name.
  const machine::ComputeNode *engine = nullptr;
  if (!input.computeResource.empty()) {
    engine = machine.findCompute(input.computeResource);
    if (!engine)
      return fail(llvm::Twine("layout transform cost: compute resource '") +
                  llvm::Twine(input.computeResource) +
                  "' is not declared by machine '" +
                  llvm::Twine(machine.target) + "'");
  } else {
    std::vector<const machine::ComputeNode *> engines =
        machine.computesOfKind("vector_engine");
    if (engines.empty())
      return fail(llvm::Twine("layout transform cost: machine '") +
                  llvm::Twine(machine.target) +
                  "' declares no compute resource to run a layout conversion");
    engine = engines.front();
  }

  // The memory the conversion runs in, when one was named. Both a node id
  // (`sram.0`) and a space kind (`sram`) are accepted: a hand-written kernel
  // names the space, a bound plan names the node.
  if (!input.memoryNode.empty() && !machine.findMemory(input.memoryNode) &&
      !machine.findMemoryOfKind(input.memoryNode))
    return fail(llvm::Twine("layout transform cost: memory node '") +
                llvm::Twine(input.memoryNode) +
                "' is not modeled by machine '" + llvm::Twine(machine.target) +
                "'");

  Cost cost;
  cost.localBytes = bytes;

  // An explicit identity re-representation does not reorder the tile, so it is
  // modeled as zero arithmetic. A conversion that changes the index relation is
  // charged what the selected capability takes to issue it -- the machine's
  // answer, with a one-element-per-issue fallback for a dtype it does not model
  // (slower than the hardware, never faster).
  const bool identity =
      input.srcMap && input.dstMap && input.srcMap == input.dstMap;
  if (!identity) {
    int64_t lanes = 1;
    auto it = engine->lanes.find(dtype);
    if (it != engine->lanes.end() && it->second > 0)
      lanes = it->second;
    uint64_t issues = llvm::divideCeil(elements, static_cast<uint64_t>(lanes));
    cost.latencyCycles = static_cast<double>(
        issues * std::max<uint64_t>(1, engine->issueCycles));
  }

  if (std::optional<double> utilization = utilizationEstimate(
          cost.latencyCycles, machine, machine.workerThreads))
    cost.computeUtilization = *utilization;
  return cost;
}

} // namespace mlir::llk::mapping
