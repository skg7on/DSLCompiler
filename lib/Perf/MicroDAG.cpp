//===- MicroDAG.cpp - Event DAG extraction from concrete Micro-IR ---------===//
//
// Implements buildMicroDAG(): the single traversal that turns a concrete
// `micro.kernel` into schedulable events plus the program properties the
// capacity check needs. See MicroDAG.h for the dependency rules.
//
// Costs come from the machine model, never from constants in this file: an
// event's `minCycles` is the machine's own answer for the work the tile
// metadata describes. That is what keeps a second target from needing a second
// extractor.
//
//===----------------------------------------------------------------------===//

#include "LLK/Perf/MicroDAG.h"

#include "MicroTileInfo.h"

#include "LLK/Dialect/Micro/MicroEnums.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mlir::llk::perf {
namespace {

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

/// Trip count of a `micro.for`/`micro.spatial_for` when both bounds and the
/// step are constants, which is what a concrete schedule is expected to have.
std::optional<uint64_t> staticTripCount(mlir::Value lower, mlir::Value upper,
                                        mlir::Value step) {
  auto constantOf = [](mlir::Value value) -> std::optional<int64_t> {
    auto constant = value.getDefiningOp<mlir::arith::ConstantOp>();
    if (!constant)
      return std::nullopt;
    if (auto integer = mlir::dyn_cast<mlir::IntegerAttr>(constant.getValue()))
      return integer.getInt();
    return std::nullopt;
  };

  std::optional<int64_t> begin = constantOf(lower);
  std::optional<int64_t> end = constantOf(upper);
  std::optional<int64_t> stride = constantOf(step);
  if (!begin || !end || !stride || *stride <= 0)
    return std::nullopt;
  if (*end <= *begin)
    return uint64_t{0};
  return static_cast<uint64_t>((*end - *begin + *stride - 1) / *stride);
}

} // namespace

//===----------------------------------------------------------------------===//
// Event DAG builder
//===----------------------------------------------------------------------===//

namespace {

class DAGBuilder {
public:
  DAGBuilder(const MachineModel &machine, llvm::StringRef kernelName)
      : machine(machine), kernelName(kernelName.str()) {}

  llvm::Expected<MicroDAG> run(mlir::Operation *kernel);

private:
  /// Walk state for one region. `pending` holds dependencies the next event
  /// must take whatever kind it is -- loop-carried ordering that the
  /// same-resource-kind shortcut must not drop.
  struct State {
    bool hasPrevious = false;
    uint32_t previous = 0;
    ResourceKind previousKind = ResourceKind::Dma;
    std::vector<uint32_t> pending;
    std::string owner;
    /// Buffers a body must hold at once: pipeline stages, times the owners a
    /// spatial loop spreads its iterations over.
    uint64_t storageFactor = 1;
  };

  llvm::Error walkBlock(mlir::Block &block, State state,
                        std::vector<uint32_t> &created);
  llvm::Error walkLoop(mlir::Block &body, mlir::Value lower, mlir::Value upper,
                       mlir::Value step, State &outer,
                       std::vector<uint32_t> &created, bool concurrent,
                       llvm::StringRef mapTarget);
  llvm::Error buildOp(mlir::Operation &op, State &state);

  llvm::Error buildLogicalTileOp(mlir::Operation &op, mlir::Value source,
                                 mlir::Value result,
                                 const std::optional<std::string> &owner,
                                 EventKind kind, State &state);
  llvm::Error buildCopyOp(mlir::Operation &op, llvm::StringRef srcMemory,
                          llvm::StringRef dstMemory, const TileInfo &sourceInfo,
                          const TileInfo &resultInfo, EventKind kind,
                          State &state, llvm::ArrayRef<mlir::Value> results);
  llvm::Error noteAllocation(mlir::Operation &op, const TileInfo &info,
                             llvm::StringRef space, const State &state);

  uint32_t addEvent(MicroEvent event, State &state,
                    llvm::ArrayRef<uint32_t> extraDeps = {});
  void noteDiagnostics(const MicroEvent &event);
  void noteWarning(std::string message);
  void noteLayoutUsage(llvm::StringRef engineName,
                       const std::vector<std::string> &supported,
                       llvm::StringRef layout);
  /// Counts a buffer once per allocation site: a loop body holds one copy of
  /// its tiles however many times the loop runs.
  void noteStorage(mlir::Operation &op, llvm::StringRef space, uint64_t bytes,
                   uint64_t storageFactor);

  /// Cost helpers, all answered by the machine model.
  uint64_t copyCycles(llvm::StringRef src, llvm::StringRef dst,
                      uint64_t bytes) const;
  uint64_t vectorCycles(const VectorEngineModel *engine, llvm::StringRef dtype,
                        uint64_t elements) const;
  uint64_t mmaCycles(const MatrixEngineModel &engine,
                     llvm::ArrayRef<int64_t> shape, uint64_t flops) const;

  const MatrixEngineModel *pickMatrixEngine(llvm::StringRef requested,
                                            std::string &reason) const;
  const VectorEngineModel *pickVectorEngine(std::string &reason) const;

  llvm::SmallVector<uint32_t, 4> producerDeps(mlir::ValueRange values) const;
  void inheritProducer(mlir::Value result, mlir::Value source);
  std::string memoryOf(mlir::Value value) const;

  /// Resolves a movement endpoint, rejecting a space the machine does not model
  /// because no bandwidth or latency would be available for it.
  llvm::Error requireMemory(mlir::Operation &op, llvm::StringRef space) const;
  llvm::Error requireReachable(mlir::Operation &op, llvm::StringRef src,
                               llvm::StringRef dst);

  const MachineModel &machine;
  std::string kernelName;
  MicroDAG dag;
  llvm::DenseMap<mlir::Value, uint32_t> producers;
  llvm::SmallPtrSet<mlir::Operation *, 32> countedStorage;
  llvm::StringSet<> seenWarnings;
  std::map<std::string, uint64_t> liveBytes;
};

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

void DAGBuilder::noteWarning(std::string message) {
  if (seenWarnings.insert(message).second)
    dag.warnings.push_back(std::move(message));
}

void DAGBuilder::noteDiagnostics(const MicroEvent &event) {
  if (!event.tileOwner.empty() && !machine.findOwner(event.tileOwner)) {
    std::string message = "owner '" + event.tileOwner +
                          "' is not modeled by machine '" + machine.name + "'";
    if (seenWarnings.insert(message).second)
      dag.ownerWarnings.push_back(std::move(message));
  }

  if (event.tileLayout.empty() || event.tileMemory.empty())
    return;
  const MemoryLevelModel *level = machine.findMemory(event.tileMemory);
  if (!level || llvm::is_contained(level->supportedLayouts, event.tileLayout))
    return;
  std::string message =
      "memory '" + event.tileMemory + "' does not support layout '" +
      event.tileLayout +
      "' (supported: " + llvm::join(level->supportedLayouts, ", ") + ")";
  if (seenWarnings.insert(message).second)
    dag.layoutWarnings.push_back(std::move(message));
}

void DAGBuilder::noteLayoutUsage(llvm::StringRef engineName,
                                 const std::vector<std::string> &supported,
                                 llvm::StringRef layout) {
  if (layout.empty() || llvm::is_contained(supported, layout))
    return;
  std::string message = "engine '" + engineName.str() +
                        "' does not support layout '" + layout.str() +
                        "' (supported: " + llvm::join(supported, ", ") + ")";
  if (seenWarnings.insert(message).second)
    dag.layoutWarnings.push_back(std::move(message));
}

void DAGBuilder::noteStorage(mlir::Operation &op, llvm::StringRef space,
                             uint64_t bytes, uint64_t storageFactor) {
  // DRAM holds the buffers the kernel reads from and writes back to, which the
  // caller owns -- `micro.alloc` refuses to target it for the same reason. Only
  // storage the kernel itself materializes can overflow a machine.
  if (space.empty() || space == "dram" || !countedStorage.insert(&op).second)
    return;
  liveBytes[space.str()] += bytes * storageFactor;
}

//===----------------------------------------------------------------------===//
// Costs
//===----------------------------------------------------------------------===//

uint64_t DAGBuilder::copyCycles(llvm::StringRef src, llvm::StringRef dst,
                                uint64_t bytes) const {
  const MemoryLevelModel *srcLevel = machine.findMemory(src);
  const MemoryLevelModel *dstLevel = machine.findMemory(dst);
  const CopyPathModel *path = machine.findCopyPath(src, dst);

  // A copy waits on the slower endpoint's latency; the destination's port sets
  // the rate unless the machine models the path separately.
  uint64_t latency = 0;
  if (srcLevel)
    latency = std::max(latency, srcLevel->latencyCycles);
  if (dstLevel)
    latency = std::max(latency, dstLevel->latencyCycles);
  double bandwidth = dstLevel ? dstLevel->bandwidthBytesPerCycle : 0.0;

  if (path) {
    if (path->latencyCycles)
      latency = *path->latencyCycles;
    if (path->bandwidthBytesPerCycle)
      bandwidth = *path->bandwidthBytesPerCycle;
  }

  uint64_t cycles = latency + machine.dma.setupCycles;
  if (bandwidth > 0)
    cycles += static_cast<uint64_t>(
        std::ceil(static_cast<double>(bytes) / bandwidth));
  return cycles;
}

uint64_t DAGBuilder::vectorCycles(const VectorEngineModel *engine,
                                  llvm::StringRef dtype,
                                  uint64_t elements) const {
  if (!engine)
    return elements;
  // A dtype the engine does not declare is issued one element at a time. That
  // is slower than the hardware, never faster, so it cannot hide a bottleneck.
  int64_t lanes = 1;
  auto it = engine->lanes.find(dtype.str());
  if (it != engine->lanes.end() && it->second > 0)
    lanes = it->second;
  uint64_t issues = llvm::divideCeil(elements, static_cast<uint64_t>(lanes));
  return issues * std::max<uint64_t>(1, engine->issueCycles);
}

uint64_t DAGBuilder::mmaCycles(const MatrixEngineModel &engine,
                               llvm::ArrayRef<int64_t> shape,
                               uint64_t flops) const {
  if (engine.flopsPerCycle && *engine.flopsPerCycle > 0)
    return static_cast<uint64_t>(
        std::ceil(static_cast<double>(flops) / *engine.flopsPerCycle));

  // No throughput figure: count how many declared fragments the shape needs.
  auto volume = [](const std::array<int64_t, 3> &fragment) {
    return fragment[0] * fragment[1] * fragment[2];
  };
  const std::array<int64_t, 3> *fragment = &engine.tileShapes.front();
  for (const std::array<int64_t, 3> &candidate : engine.tileShapes)
    if (volume(candidate) > volume(*fragment))
      fragment = &candidate;

  uint64_t issues = 1;
  for (size_t i = 0; i < 3; ++i)
    issues *= llvm::divideCeil(static_cast<uint64_t>(shape[i]),
                               static_cast<uint64_t>((*fragment)[i]));
  return issues * std::max<uint64_t>(1, engine.issueCycles);
}

const MatrixEngineModel *
DAGBuilder::pickMatrixEngine(llvm::StringRef requested,
                             std::string &reason) const {
  if (!requested.empty()) {
    if (const MatrixEngineModel *engine = machine.findMatrixEngine(requested))
      return engine;
    reason = "engine '" + requested.str() + "' is not declared by machine '" +
             machine.name + "'";
    return nullptr;
  }
  if (machine.matrixEngines.empty()) {
    reason = "machine '" + machine.name + "' declares no matrix engine";
    return nullptr;
  }
  return &machine.matrixEngines.front();
}

const VectorEngineModel *
DAGBuilder::pickVectorEngine(std::string &reason) const {
  if (machine.vectorEngines.empty()) {
    reason = "machine '" + machine.name + "' declares no vector engine";
    return nullptr;
  }
  return &machine.vectorEngines.front();
}

//===----------------------------------------------------------------------===//
// Producers
//===----------------------------------------------------------------------===//

llvm::SmallVector<uint32_t, 4>
DAGBuilder::producerDeps(mlir::ValueRange values) const {
  llvm::SmallVector<uint32_t, 4> deps;
  for (mlir::Value value : values) {
    auto it = producers.find(value);
    if (it != producers.end())
      deps.push_back(it->second);
  }
  return deps;
}

void DAGBuilder::inheritProducer(mlir::Value result, mlir::Value source) {
  auto it = producers.find(source);
  if (it != producers.end())
    producers[result] = it->second;
}

std::string DAGBuilder::memoryOf(mlir::Value value) const {
  TileInfo info = describeType(value.getType());
  if (!info.memory.empty())
    return info.memory;
  // A logical view carries no memory of its own; the event that produced or
  // last moved the underlying data does.
  auto it = producers.find(value);
  if (it != producers.end())
    return dag.events[it->second].tileMemory;
  return {};
}

//===----------------------------------------------------------------------===//
// Event insertion
//===----------------------------------------------------------------------===//

uint32_t DAGBuilder::addEvent(MicroEvent event, State &state,
                              llvm::ArrayRef<uint32_t> extraDeps) {
  if (event.tileOwner.empty())
    event.tileOwner = state.owner;

  event.id = static_cast<uint32_t>(dag.events.size());
  ResourceKind kind = event.resource;

  // Program order, except between two events that share a resource kind: those
  // are ordered by their pool, and dropping the edge is what lets independent
  // same-class work overlap.
  if (state.hasPrevious && state.previousKind != kind)
    event.deps.push_back(state.previous);
  llvm::append_range(event.deps, state.pending);
  llvm::append_range(event.deps, extraDeps);

  llvm::sort(event.deps);
  event.deps.erase(std::unique(event.deps.begin(), event.deps.end()),
                   event.deps.end());
  event.deps.erase(std::remove(event.deps.begin(), event.deps.end(), event.id),
                   event.deps.end());

  noteDiagnostics(event);
  dag.events.push_back(std::move(event));

  state.previous = dag.events.back().id;
  state.previousKind = kind;
  state.hasPrevious = true;
  state.pending.clear();
  return state.previous;
}

//===----------------------------------------------------------------------===//
// Region walk
//===----------------------------------------------------------------------===//

llvm::Error DAGBuilder::walkBlock(mlir::Block &block, State state,
                                  std::vector<uint32_t> &created) {
  for (mlir::Operation &op : block) {
    if (llvm::isa<micro::YieldOp>(op))
      continue;

    if (auto forOp = llvm::dyn_cast<micro::ForOp>(op)) {
      State outer = state;
      if (llvm::Error err =
              walkLoop(forOp.getBody().front(), forOp.getLowerBound(),
                       forOp.getUpperBound(), forOp.getStep(), outer, created,
                       /*concurrent=*/false, llvm::StringRef()))
        return err;
      state = std::move(outer);
      continue;
    }

    if (auto spatialOp = llvm::dyn_cast<micro::SpatialForOp>(op)) {
      std::string target =
          micro::stringifyMappingTarget(spatialOp.getMap()).str();
      State outer = state;
      if (llvm::Error err =
              walkLoop(spatialOp.getBody().front(), spatialOp.getLowerBound(),
                       spatialOp.getUpperBound(), spatialOp.getStep(), outer,
                       created, /*concurrent=*/true, target))
        return err;
      state = std::move(outer);
      continue;
    }

    if (auto pipelineOp = llvm::dyn_cast<micro::PipelineOp>(op)) {
      State inner = state;
      inner.storageFactor =
          state.storageFactor * std::max<uint64_t>(1, pipelineOp.getStages());
      std::vector<uint32_t> pipelineEvents;
      if (llvm::Error err =
              walkBlock(pipelineOp.getBody().front(), inner, pipelineEvents))
        return err;
      created.insert(created.end(), pipelineEvents.begin(),
                     pipelineEvents.end());
      // Everything after the pipeline waits for all of it.
      state.pending = std::move(pipelineEvents);
      state.hasPrevious = false;
      continue;
    }

    size_t before = dag.events.size();
    if (llvm::Error err = buildOp(op, state))
      return err;
    for (size_t id = before; id < dag.events.size(); ++id)
      created.push_back(static_cast<uint32_t>(id));
  }
  return llvm::Error::success();
}

llvm::Error DAGBuilder::walkLoop(mlir::Block &body, mlir::Value lower,
                                 mlir::Value upper, mlir::Value step,
                                 State &outer, std::vector<uint32_t> &created,
                                 bool concurrent, llvm::StringRef mapTarget) {
  uint64_t trips = 1;
  if (std::optional<uint64_t> count = staticTripCount(lower, upper, step))
    trips = *count;
  else
    noteWarning("a loop has non-static bounds; the simulator runs a single "
                "iteration for it");

  if (trips == 0)
    return llvm::Error::success();

  // A `micro.pipeline stages >= 2` directly inside the loop is what lets a copy
  // for the next iteration issue while this one still computes.
  bool pipelined = false;
  for (mlir::Operation &op : body) {
    auto pipelineOp = llvm::dyn_cast<micro::PipelineOp>(op);
    if (pipelineOp && pipelineOp.getStages() >= 2) {
      pipelined = true;
      break;
    }
  }
  bool independent = concurrent || pipelined;

  std::string owner = outer.owner;
  uint64_t storageFactor = outer.storageFactor;
  if (concurrent && !mapTarget.empty()) {
    if (micro::symbolizeOwner(mapTarget)) {
      owner = mapTarget.str();
      // Each owner needs its own copy of the tiles the body materializes.
      storageFactor *= std::max<uint64_t>(1, machine.getOwnerCount(owner));
    } else {
      noteWarning("micro.spatial_for maps to '" + mapTarget.str() +
                  "', which is not an owner scope; owner occupancy is not "
                  "modeled for it");
    }
  }

  std::vector<uint32_t> previousIteration;
  for (uint64_t iteration = 0; iteration < trips; ++iteration) {
    State inner;
    inner.owner = owner;
    inner.storageFactor = storageFactor;
    if (iteration == 0) {
      inner.hasPrevious = outer.hasPrevious;
      inner.previous = outer.previous;
      inner.previousKind = outer.previousKind;
      inner.pending = outer.pending;
    } else if (!independent) {
      // Serial iterations: the next one starts only after this one is done.
      inner.pending = previousIteration;
    }

    std::vector<uint32_t> iterationEvents;
    if (llvm::Error err = walkBlock(body, inner, iterationEvents))
      return err;
    previousIteration = std::move(iterationEvents);
    created.insert(created.end(), previousIteration.begin(),
                   previousIteration.end());

    if (dag.events.size() > kMaxEvents)
      return invalid("kernel '" + kernelName + "' unrolls to more than " +
                     llvm::Twine(kMaxEvents) +
                     " events; it is too large for the MVP simulator");
  }

  outer.pending = std::move(previousIteration);
  outer.hasPrevious = false;
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Per-op extraction
//===----------------------------------------------------------------------===//

llvm::Error DAGBuilder::buildOp(mlir::Operation &op, State &state) {
  if (auto view = llvm::dyn_cast<micro::TileViewOp>(op))
    return buildLogicalTileOp(op, view.getSource(), view.getResult(),
                              std::nullopt, EventKind::TileView, state);

  if (auto partition = llvm::dyn_cast<micro::TilePartitionOp>(op)) {
    std::optional<std::string> owner;
    if (std::optional<micro::Owner> attr = partition.getOwner())
      owner = micro::stringifyOwner(*attr).str();
    return buildLogicalTileOp(op, partition.getSource(), partition.getResult(),
                              owner, EventKind::TilePartition, state);
  }

  if (auto alloc = llvm::dyn_cast<micro::TileAllocOp>(op)) {
    TileInfo info = describeType(alloc.getResult().getType());
    return noteAllocation(op, info, info.memory, state);
  }

  if (auto alloc = llvm::dyn_cast<micro::AllocOp>(op)) {
    TileInfo info = describeType(alloc.getResult().getType());
    return noteAllocation(
        op, info, micro::stringifyMemorySpace(alloc.getMemory()), state);
  }

  if (auto copy = llvm::dyn_cast<micro::TileAsyncCopyOp>(op)) {
    TileInfo resultInfo = describeType(copy.getResult().getType());
    TileInfo sourceInfo = describeType(copy.getSource().getType());
    std::string src = memoryOf(copy.getSource());
    if (src.empty())
      src = "dram"; // an untagged operand is an external buffer
    return buildCopyOp(op, src,
                       micro::stringifyMemorySpace(copy.getDstMemory()),
                       sourceInfo, resultInfo, EventKind::AsyncCopy, state,
                       {copy.getResult(), copy.getToken()});
  }

  if (auto copy = llvm::dyn_cast<micro::AsyncCopyOp>(op)) {
    TileInfo resultInfo = describeType(copy.getResult().getType());
    TileInfo sourceInfo = describeType(copy.getSource().getType());
    return buildCopyOp(op, micro::stringifyMemorySpace(copy.getSrcMemory()),
                       micro::stringifyMemorySpace(copy.getDstMemory()),
                       sourceInfo, resultInfo, EventKind::AsyncCopy, state,
                       {copy.getResult(), copy.getToken()});
  }

  if (auto store = llvm::dyn_cast<micro::TileStoreOp>(op)) {
    TileInfo sourceInfo = describeType(store.getSource().getType());
    std::string src = memoryOf(store.getSource());
    if (src.empty())
      src = "sram"; // a tile without a tag came from on-chip storage
    return buildCopyOp(op, src,
                       micro::stringifyMemorySpace(store.getDstMemory()),
                       sourceInfo, TileInfo(), EventKind::Store, state, {});
  }

  if (auto store = llvm::dyn_cast<micro::StoreOp>(op)) {
    TileInfo sourceInfo = describeType(store.getSource().getType());
    return buildCopyOp(op, micro::stringifyMemorySpace(store.getSrcMemory()),
                       micro::stringifyMemorySpace(store.getDstMemory()),
                       sourceInfo, TileInfo(), EventKind::Store, state, {});
  }

  if (auto mma = llvm::dyn_cast<micro::MmaOp>(op)) {
    llvm::StringRef requested;
    if (auto engineAttr = mma.getEngine())
      requested = *engineAttr;
    std::string reason;
    const MatrixEngineModel *engine = pickMatrixEngine(requested, reason);
    if (!engine)
      return invalid("kernel '" + kernelName + "' uses micro.mma: " + reason);

    llvm::ArrayRef<int64_t> shape = mma.getShape();
    if (shape.size() != 3)
      return invalid("micro.mma shape must have three dimensions");

    uint64_t macs = static_cast<uint64_t>(shape[0]) *
                    static_cast<uint64_t>(shape[1]) *
                    static_cast<uint64_t>(shape[2]);
    std::string inputDtype = micro::stringifyDType(mma.getInput()).str();
    std::string accumulatorDtype =
        micro::stringifyDType(mma.getAccumulator()).str();
    if (!llvm::is_contained(engine->inputDTypes, inputDtype))
      noteWarning("micro.mma input dtype '" + inputDtype +
                  "' is not supported by engine '" + engine->name +
                  "' (supported: " + llvm::join(engine->inputDTypes, ", ") +
                  ")");
    if (!llvm::is_contained(engine->accumulatorDTypes, accumulatorDtype))
      noteWarning(
          "micro.mma accumulator dtype '" + accumulatorDtype +
          "' is not supported by engine '" + engine->name +
          "' (supported: " + llvm::join(engine->accumulatorDTypes, ", ") + ")");

    TileInfo lhsInfo = describeType(mma.getLhs().getType());
    TileInfo accInfo = describeType(mma.getAcc().getType());

    MicroEvent event;
    event.kind = EventKind::Mma;
    event.resource = ResourceKind::MatrixEngine;
    // An op that names no engine asks for the class as a whole, so the whole
    // pool is available to it; naming one confines it to that engine's slots.
    event.resourceName = requested.str();
    event.workItems = macs;
    event.minCycles = mmaCycles(*engine, shape, 2 * macs);
    event.sourceOpName = op.getName().getStringRef().str();
    event.tileShape = shapeString(shape);
    event.tileLayout = lhsInfo.layout.empty() ? accInfo.layout : lhsInfo.layout;
    event.tileMemory = accInfo.memory;
    event.tileOwner = accInfo.owner;
    noteLayoutUsage(engine->name, engine->supportedLayouts, event.tileLayout);

    uint32_t id =
        addEvent(std::move(event), state,
                 producerDeps({mma.getLhs(), mma.getRhs(), mma.getAcc()}));
    producers[mma.getResult()] = id;
    return llvm::Error::success();
  }

  if (auto vector = llvm::dyn_cast<micro::VectorOp>(op)) {
    std::string reason;
    const VectorEngineModel *engine = pickVectorEngine(reason);
    if (!engine)
      return invalid("kernel '" + kernelName +
                     "' uses micro.vector: " + reason);

    TileInfo resultInfo = describeType(vector.getResult().getType());
    MicroEvent event;
    event.kind = EventKind::Vector;
    event.resource = ResourceKind::VectorEngine;
    // No op names a vector engine, so these ask for the class as a whole.
    event.workItems = resultInfo.elements();
    event.minCycles =
        vectorCycles(engine, resultInfo.dtype, resultInfo.elements());
    event.sourceOpName = op.getName().getStringRef().str();
    event.tileShape = shapeString(resultInfo.shape);
    event.tileLayout = resultInfo.layout;
    event.tileMemory = resultInfo.memory;
    event.tileOwner = resultInfo.owner;
    noteLayoutUsage(engine->name, engine->supportedLayouts, event.tileLayout);

    uint32_t id =
        addEvent(std::move(event), state, producerDeps(vector.getInputs()));
    producers[vector.getResult()] = id;
    return llvm::Error::success();
  }

  if (auto reduce = llvm::dyn_cast<micro::ReduceOp>(op)) {
    std::string reason;
    const VectorEngineModel *engine = pickVectorEngine(reason);
    if (!engine)
      return invalid("kernel '" + kernelName +
                     "' uses micro.reduce: " + reason);

    TileInfo inputInfo = describeType(reduce.getInput().getType());
    MicroEvent event;
    event.kind = EventKind::Reduce;
    event.resource = ResourceKind::VectorEngine;
    event.workItems = inputInfo.elements();
    event.minCycles =
        vectorCycles(engine, inputInfo.dtype, inputInfo.elements());
    event.sourceOpName = op.getName().getStringRef().str();
    event.tileShape = shapeString(inputInfo.shape);
    event.tileLayout = inputInfo.layout;
    event.tileMemory = inputInfo.memory;
    event.tileOwner = inputInfo.owner;
    noteLayoutUsage(engine->name, engine->supportedLayouts, event.tileLayout);

    uint32_t id =
        addEvent(std::move(event), state, producerDeps({reduce.getInput()}));
    producers[reduce.getResult()] = id;
    return llvm::Error::success();
  }

  if (auto wait = llvm::dyn_cast<micro::WaitOp>(op)) {
    MicroEvent event;
    event.kind = EventKind::Wait;
    event.resource = ResourceKind::Sync;
    event.resourceName = "sync";
    event.minCycles = machine.sync.waitCycles;
    event.sourceOpName = op.getName().getStringRef().str();
    addEvent(std::move(event), state, producerDeps(wait.getTokens()));
    return llvm::Error::success();
  }

  // Everything else -- arithmetic feeding loop bounds, micro.yield -- has no
  // execution cost in the MVP model.
  return llvm::Error::success();
}

llvm::Error DAGBuilder::buildLogicalTileOp(
    mlir::Operation &op, mlir::Value source, mlir::Value result,
    const std::optional<std::string> &owner, EventKind kind, State &state) {
  TileInfo sourceInfo = describeType(source.getType());
  TileInfo resultInfo = describeType(result.getType());
  if (owner) {
    resultInfo.owner = *owner;
    resultInfo.isTile = true;
  }

  // Value flow is preserved across a logical op, so a consumer still depends on
  // whatever produced the underlying data.
  inheritProducer(result, source);

  // A logical view is zero-cost metadata: it changes how the same bytes are
  // read, not how many there are. Only two known layouts that differ name real
  // work -- a view of an untagged tensor says nothing about the memory order it
  // has, so reading one as row-major is not a transform.
  if (!resultInfo.isTile || resultInfo.layout.empty() ||
      sourceInfo.layout.empty() || resultInfo.layout == sourceInfo.layout)
    return llvm::Error::success();

  std::string reason;
  const VectorEngineModel *engine = pickVectorEngine(reason);
  if (!engine)
    return invalid("kernel '" + kernelName +
                   "' performs a layout transform but " + reason);

  MicroEvent event;
  event.kind = kind;
  event.resource = ResourceKind::VectorEngine;
  event.workItems = resultInfo.elements();
  event.bytes = resultInfo.bytes();
  event.minCycles =
      vectorCycles(engine, resultInfo.dtype, resultInfo.elements());
  event.sourceOpName = op.getName().getStringRef().str();
  event.tileShape = shapeString(resultInfo.shape);
  event.tileLayout = resultInfo.layout;
  event.tileMemory =
      resultInfo.memory.empty() ? sourceInfo.memory : resultInfo.memory;
  event.srcMemory = sourceInfo.memory;
  event.tileOwner = resultInfo.owner;
  noteLayoutUsage(engine->name, engine->supportedLayouts, event.tileLayout);

  std::string destination = event.tileMemory;
  uint32_t id = addEvent(std::move(event), state);
  producers[result] = id;
  noteStorage(op, destination, resultInfo.bytes(), state.storageFactor);
  return llvm::Error::success();
}

llvm::Error
DAGBuilder::buildCopyOp(mlir::Operation &op, llvm::StringRef srcMemory,
                        llvm::StringRef dstMemory, const TileInfo &sourceInfo,
                        const TileInfo &resultInfo, EventKind kind,
                        State &state, llvm::ArrayRef<mlir::Value> results) {
  if (llvm::Error err = requireMemory(op, dstMemory))
    return err;
  if (llvm::Error err = requireMemory(op, srcMemory))
    return err;
  if (llvm::Error err = requireReachable(op, srcMemory, dstMemory))
    return err;

  const TileInfo &shapeInfo = resultInfo.elements() ? resultInfo : sourceInfo;
  uint64_t bytes = shapeInfo.bytes();
  if (shapeInfo.hasDynamicShape())
    noteWarning(op.getName().getStringRef().str() +
                " moves a tile with a dynamic extent; its bytes are charged as "
                "zero");

  MicroEvent event;
  event.kind = kind;
  event.resource = ResourceKind::Dma;
  event.resourceName = "dma";
  event.workItems = shapeInfo.elements();
  event.bytes = bytes;
  event.minCycles = copyCycles(srcMemory, dstMemory, bytes);
  event.sourceOpName = op.getName().getStringRef().str();
  event.tileShape = shapeString(shapeInfo.shape);
  event.tileLayout =
      resultInfo.layout.empty() ? sourceInfo.layout : resultInfo.layout;
  event.tileMemory = dstMemory.str();
  event.srcMemory = srcMemory.str();
  event.tileOwner = resultInfo.owner;

  uint32_t id = addEvent(std::move(event), state);
  for (mlir::Value result : results)
    producers[result] = id;

  // The moved tile is live in its destination. A pipelined or spatially mapped
  // body needs one buffer per concurrent iteration.
  noteStorage(op, dstMemory, bytes, state.storageFactor);
  return llvm::Error::success();
}

llvm::Error DAGBuilder::noteAllocation(mlir::Operation &op,
                                       const TileInfo &info,
                                       llvm::StringRef space,
                                       const State &state) {
  if (info.hasDynamicShape()) {
    noteWarning(op.getName().getStringRef().str() +
                " allocates a tile with a dynamic extent; it is not counted "
                "against capacity");
    return llvm::Error::success();
  }
  if (space.empty()) {
    noteWarning(op.getName().getStringRef().str() +
                " allocates a tile that names no memory space; it is not "
                "counted against capacity");
    return llvm::Error::success();
  }
  if (!machine.findMemory(space)) {
    // An allocation needs no bandwidth or latency, so an unmodeled space only
    // costs us the capacity check.
    noteWarning("allocations target memory space '" + space.str() +
                "', which machine '" + machine.name +
                "' does not model; capacity is not checked for it");
    return llvm::Error::success();
  }
  noteStorage(op, space, info.bytes(), state.storageFactor);
  return llvm::Error::success();
}

llvm::Error DAGBuilder::requireMemory(mlir::Operation &op,
                                      llvm::StringRef space) const {
  if (space.empty())
    return invalid("kernel '" + kernelName +
                   "': " + op.getName().getStringRef() +
                   " moves data through a memory space that is neither named "
                   "on the value nor modeled by machine '" +
                   machine.name + "'");
  if (!machine.findMemory(space))
    return invalid("kernel '" + kernelName +
                   "': " + op.getName().getStringRef() +
                   " moves data through memory space '" + space +
                   "', which machine '" + machine.name + "' does not model");
  return llvm::Error::success();
}

llvm::Error DAGBuilder::requireReachable(mlir::Operation &op,
                                         llvm::StringRef src,
                                         llvm::StringRef dst) {
  if (src == dst)
    return invalid("kernel '" + kernelName +
                   "': " + op.getName().getStringRef() +
                   " copies from and to '" + src.str() + "'");

  // A machine declares the copy paths it can perform. A pair it does not list
  // is still charged through both endpoints' latencies, but the gap is worth
  // reporting: that cost comes from a composition the machine never described.
  if (!machine.findCopyPath(src, dst))
    noteWarning("machine '" + machine.name + "' declares no copy path '" +
                src.str() + " -> " + dst.str() +
                "'; its cost is composed from both endpoints");
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

llvm::Expected<MicroDAG> DAGBuilder::run(mlir::Operation *kernel) {
  State state;
  std::vector<uint32_t> created;
  if (llvm::Error err = walkBlock(kernel->getRegion(0).front(), state, created))
    return std::move(err);

  if (dag.events.empty())
    noteWarning("kernel '" + kernelName +
                "' has no schedulable events; every figure below is zero");

  dag.liveTileBytesByMemory = std::move(liveBytes);
  return std::move(dag);
}

} // namespace

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

llvm::Expected<MicroDAG> buildMicroDAG(mlir::Operation *kernel,
                                       const MachineModel &machine) {
  auto kernelOp = llvm::dyn_cast<micro::KernelOp>(kernel);
  if (!kernelOp)
    return invalid("expected a micro.kernel, got '" +
                   kernel->getName().getStringRef() + "'");

  DAGBuilder builder(machine, kernelOp.getSymName());
  return builder.run(kernel);
}

llvm::Expected<mlir::Operation *> findMicroKernel(mlir::Operation *root,
                                                  llvm::StringRef symbol) {
  llvm::SmallVector<micro::KernelOp, 2> kernels;
  root->walk([&](micro::KernelOp kernel) { kernels.push_back(kernel); });

  if (!symbol.empty()) {
    for (micro::KernelOp kernel : kernels)
      if (kernel.getSymName() == symbol)
        return kernel.getOperation();
    return invalid("no micro.kernel named '" + symbol + "' in the input");
  }

  if (kernels.empty())
    return invalid("the input contains no micro.kernel");
  if (kernels.size() > 1)
    return invalid("the input contains " + llvm::Twine(kernels.size()) +
                   " micro.kernel ops; select one with --kernel=<name>");
  return kernels.front().getOperation();
}

llvm::StringRef stringifyEventKind(EventKind kind) {
  switch (kind) {
  case EventKind::TileView:
    return "tile_view";
  case EventKind::TilePartition:
    return "tile_partition";
  case EventKind::AsyncCopy:
    return "async_copy";
  case EventKind::Load:
    return "load";
  case EventKind::Store:
    return "store";
  case EventKind::Mma:
    return "mma";
  case EventKind::Vector:
    return "vector";
  case EventKind::Reduce:
    return "reduce";
  case EventKind::Wait:
    return "wait";
  case EventKind::Barrier:
    return "barrier";
  }
  return "unknown";
}

llvm::StringRef stringifyResourceKind(ResourceKind kind) {
  switch (kind) {
  case ResourceKind::Dma:
    return "dma";
  case ResourceKind::MatrixEngine:
    return "matrix_engine";
  case ResourceKind::VectorEngine:
    return "vector_engine";
  case ResourceKind::MemoryRead:
    return "memory_read";
  case ResourceKind::MemoryWrite:
    return "memory_write";
  case ResourceKind::Sync:
    return "sync";
  }
  return "unknown";
}

} // namespace mlir::llk::perf
