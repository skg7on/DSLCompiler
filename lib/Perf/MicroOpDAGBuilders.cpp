//===- MicroOpDAGBuilders.cpp - Built-in Micro leaf event builders --------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Mapping/CostModel.h"
#include "LLK/Perf/ContractionSupport.h"
#include "LLK/Perf/OpDAGBuilder.h"
#include "MicroTileInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CheckedArithmetic.h"
#include <cmath>
#include <optional>
namespace mlir::llk::perf {
namespace {
llvm::Error invalid(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

/// The plain shaped type the shared transform-cost model reads. A `!micro.tile`
/// carries its shape and element type the same way a ranked tensor does, but
/// the cost model sits below the dialect, so the tile is presented to it as the
/// shaped type it describes.
mlir::Type shapedCostType(mlir::Type type) {
  if (llvm::isa<mlir::ShapedType>(type))
    return type;
  if (auto tile = llvm::dyn_cast<micro::TileType>(type))
    return mlir::RankedTensorType::get(tile.getShape(), tile.getElementType());
  return type;
}

llvm::StringRef mappedExecutor(mlir::Operation &op) {
  auto mapping = op.getAttrOfType<mlir::DictionaryAttr>("micro.mapping");
  if (!mapping)
    return {};
  if (auto executor = mapping.getAs<mlir::StringAttr>("executor"))
    return executor.getValue();
  return {};
}

llvm::StringRef transformExecutor(mlir::Operation &op) {
  if (auto engine = op.getAttrOfType<mlir::StringAttr>("micro.engine"))
    return engine.getValue();
  return mappedExecutor(op);
}

llvm::Error buildTileView(micro::TileViewOp view, OpDAGBuildContext &context) {

  return context.buildLogicalTile(view.getSource(), view.getResult(),
                                  std::nullopt, EventKind::TileView);
}

llvm::Error buildTilePartition(micro::TilePartitionOp partition,
                               OpDAGBuildContext &context) {
  std::optional<std::string> owner;
  if (std::optional<micro::OwnerAttr> attr = partition.getOwner())
    owner = attr->getSymbol().str();
  return context.buildLogicalTile(partition.getSource(), partition.getResult(),
                                  owner, EventKind::TilePartition);
}

llvm::Error buildTileAlloc(micro::TileAllocOp alloc,
                           OpDAGBuildContext &context) {
  TileInfo info = describeType(alloc.getResult().getType());
  return context.noteAllocation(info, info.memory);
}

llvm::Error buildAlloc(micro::AllocOp alloc, OpDAGBuildContext &context) {
  TileInfo info = describeType(alloc.getResult().getType());
  return context.noteAllocation(info,
                                micro::stringifyMemorySpace(alloc.getMemory()));
}

llvm::Error buildTileAsyncCopy(micro::TileAsyncCopyOp copy,
                               OpDAGBuildContext &context) {
  TileInfo resultInfo = describeType(copy.getResult().getType());
  TileInfo sourceInfo = describeType(copy.getSource().getType());
  std::string src = context.memoryOf(copy.getSource());
  if (src.empty())
    src = "dram"; // an untagged operand is an external buffer
  return context.buildMovement(
      src, micro::stringifyMemorySpace(copy.getDstMemory()), sourceInfo,
      resultInfo, EventKind::AsyncCopy, {copy.getResult(), copy.getToken()});
}

llvm::Error buildAsyncCopy(micro::AsyncCopyOp copy,
                           OpDAGBuildContext &context) {
  TileInfo resultInfo = describeType(copy.getResult().getType());
  TileInfo sourceInfo = describeType(copy.getSource().getType());
  return context.buildMovement(micro::stringifyMemorySpace(copy.getSrcMemory()),
                               micro::stringifyMemorySpace(copy.getDstMemory()),
                               sourceInfo, resultInfo, EventKind::AsyncCopy,
                               {copy.getResult(), copy.getToken()});
}

llvm::Error buildTileStore(micro::TileStoreOp store,
                           OpDAGBuildContext &context) {
  TileInfo sourceInfo = describeType(store.getSource().getType());
  std::string src = context.memoryOf(store.getSource());
  if (src.empty())
    src = "sram"; // a tile without a tag came from on-chip storage
  return context.buildMovement(
      src, micro::stringifyMemorySpace(store.getDstMemory()), sourceInfo,
      TileInfo(), EventKind::Store, {});
}

llvm::Error buildStore(micro::StoreOp store, OpDAGBuildContext &context) {
  TileInfo sourceInfo = describeType(store.getSource().getType());
  return context.buildMovement(
      micro::stringifyMemorySpace(store.getSrcMemory()),
      micro::stringifyMemorySpace(store.getDstMemory()), sourceInfo, TileInfo(),
      EventKind::Store, {});
}

llvm::Error buildMma(micro::MmaOp mma, OpDAGBuildContext &context) {
  mlir::Operation &op = *mma;
  llvm::StringRef requested;
  if (auto engineAttr = mma.getEngine())
    requested = *engineAttr;
  std::string reason;
  const machine::ComputeNode *engine =
      context.pickContractionEngine(requested, reason);
  if (!engine)
    return invalid("kernel '" + context.kernelName().str() +
                   "' uses micro.mma: " + reason);

  llvm::ArrayRef<int64_t> shape = mma.getShape();
  if (shape.size() != 3)
    return invalid("micro.mma shape must have three dimensions");

  uint64_t macs = 1;
  for (int64_t extent : shape) {
    std::optional<uint64_t> product =
        llvm::checkedMulUnsigned<uint64_t>(macs, static_cast<uint64_t>(extent));
    if (!product)
      return invalid("micro.mma MAC count overflows 64 bits");
    macs = *product;
  }
  std::optional<uint64_t> flops = llvm::checkedMulUnsigned<uint64_t>(macs, 2);
  if (!flops)
    return invalid("micro.mma FLOP count overflows 64 bits");

  std::string inputDtype = micro::stringifyDType(mma.getInput()).str();
  std::string accumulatorDtype =
      micro::stringifyDType(mma.getAccumulator()).str();
  const bool mapped =
      op.getAttrOfType<mlir::DictionaryAttr>("micro.mapping") != nullptr;
  if (!supportsContractionTypes(*engine, inputDtype, accumulatorDtype)) {
    // A selected plan may not be described as complete when its engine cannot
    // run the dtype pair; only hand-written/reference analysis keeps the
    // historical warning.
    if (mapped)
      return invalid("micro.mma dtype pair ('" + inputDtype + "', '" +
                     accumulatorDtype +
                     "') is not supported by selected "
                     "engine '" +
                     engine->id + "'");
    if (!llvm::is_contained(engine->elementTypes, inputDtype))
      context.noteWarning(
          "micro.mma input dtype '" + inputDtype +
          "' is not supported by engine '" + engine->id +
          "' (supported: " + llvm::join(engine->elementTypes, ", ") + ")");
    if (!llvm::is_contained(engine->accumulatorDTypes, accumulatorDtype))
      context.noteWarning(
          "micro.mma accumulator dtype '" + accumulatorDtype +
          "' is not supported by engine '" + engine->id +
          "' (supported: " + llvm::join(engine->accumulatorDTypes, ", ") + ")");
  }

  TileInfo lhsInfo = describeType(mma.getLhs().getType());
  TileInfo accInfo = describeType(mma.getAcc().getType());

  MicroEvent event;
  event.kind = EventKind::Mma;
  // The selected engine's actual kind is the resource the work occupies: a
  // vector contraction contends with vector epilogues on the same pool.
  event.resource = engine->kind == "matrix_engine" ? ResourceKind::MatrixEngine
                                                   : ResourceKind::VectorEngine;
  // The event names the one concrete pool the selected engine owns, so two
  // events selecting the same engine share its slots instead of each being
  // handed the whole machine.
  event.resourceName = engine->id;
  event.ownerPool = mappedExecutor(op).str();
  event.workItems = macs;
  llvm::Expected<uint64_t> cycles =
      context.contractionCycles(*engine, shape, *flops, accumulatorDtype);
  if (!cycles)
    return cycles.takeError();
  event.minCycles = *cycles;
  event.sourceOpName = op.getName().getStringRef().str();
  event.tileShape = shapeString(shape);
  event.tileLayout = lhsInfo.layout.empty() ? accInfo.layout : lhsInfo.layout;
  event.tileMemory = accInfo.memory;
  event.tileOwner = accInfo.owner;
  context.noteLayoutUsage(engine->id, engine->supportedLayouts,
                          event.tileLayout);

  context.emitEvent(std::move(event),
                    {mma.getLhs(), mma.getRhs(), mma.getAcc()},
                    {mma.getResult()});
  return llvm::Error::success();
}

llvm::Error buildVector(micro::VectorOp vector, OpDAGBuildContext &context) {
  mlir::Operation &op = *vector;
  std::string reason;
  const machine::ComputeNode *engine = context.pickVectorEngine(reason);
  if (!engine)
    return invalid("kernel '" + context.kernelName().str() +
                   "' uses micro.vector: " + reason);

  TileInfo resultInfo = describeType(vector.getResult().getType());
  MicroEvent event;
  event.kind = EventKind::Vector;
  event.resource = ResourceKind::VectorEngine;
  event.resourceName = engine->id;
  event.ownerPool = mappedExecutor(op).str();
  event.workItems = resultInfo.elements();
  event.minCycles =
      context.vectorCycles(engine, resultInfo.dtype, resultInfo.elements());
  event.sourceOpName = op.getName().getStringRef().str();
  event.tileShape = shapeString(resultInfo.shape);
  event.tileLayout = resultInfo.layout;
  event.tileMemory = resultInfo.memory;
  event.tileOwner = resultInfo.owner;
  context.noteLayoutUsage(engine->id, engine->supportedLayouts,
                          event.tileLayout);

  context.emitEvent(std::move(event), vector.getInputs(), {vector.getResult()});
  return llvm::Error::success();
}

llvm::Error buildReduce(micro::ReduceOp reduce, OpDAGBuildContext &context) {
  mlir::Operation &op = *reduce;
  std::string reason;
  const machine::ComputeNode *engine = context.pickVectorEngine(reason);
  if (!engine)
    return invalid("kernel '" + context.kernelName().str() +
                   "' uses micro.reduce: " + reason);

  TileInfo inputInfo = describeType(reduce.getInput().getType());
  MicroEvent event;
  event.kind = EventKind::Reduce;
  event.resource = ResourceKind::VectorEngine;
  event.resourceName = engine->id;
  event.ownerPool = mappedExecutor(op).str();
  event.workItems = inputInfo.elements();
  event.minCycles =
      context.vectorCycles(engine, inputInfo.dtype, inputInfo.elements());
  event.sourceOpName = op.getName().getStringRef().str();
  event.tileShape = shapeString(inputInfo.shape);
  event.tileLayout = inputInfo.layout;
  event.tileMemory = inputInfo.memory;
  event.tileOwner = inputInfo.owner;
  context.noteLayoutUsage(engine->id, engine->supportedLayouts,
                          event.tileLayout);

  context.emitEvent(std::move(event), {reduce.getInput()},
                    {reduce.getResult()});
  return llvm::Error::success();
}

llvm::Error buildGather(micro::GatherOp gather, OpDAGBuildContext &context) {
  mlir::Operation &op = *gather;
  // A gather combines several producer values, so it is real compute work on
  // the engine the plan selected -- charged like the reduction it performs,
  // never folded into the zero-cost path. Its semantics must be one the model
  // understands; an undeclared or unknown kind is rejected rather than
  // charged some invented cost.
  if (!micro::symbolizeGatherKind(gather.getKind()))
    return invalid("kernel '" + context.kernelName().str() +
                   "' uses micro.gather: unknown "
                   "gather kind '" +
                   gather.getKind().str() + "'");
  std::string reason;
  const machine::ComputeNode *engine = context.pickVectorEngine(reason);
  if (!engine)
    return invalid("kernel '" + context.kernelName().str() +
                   "' uses micro.gather: " + reason);

  TileInfo resultInfo = describeType(gather.getResult().getType());
  MicroEvent event;
  event.kind = EventKind::Reduce;
  event.resource = ResourceKind::VectorEngine;
  event.resourceName = engine->id;
  event.ownerPool = mappedExecutor(op).str();
  event.workItems = resultInfo.elements();
  event.minCycles =
      context.vectorCycles(engine, resultInfo.dtype, resultInfo.elements());
  event.sourceOpName = op.getName().getStringRef().str();
  event.tileShape = shapeString(resultInfo.shape);
  event.tileLayout = resultInfo.layout;
  event.tileMemory = resultInfo.memory;
  event.tileOwner = resultInfo.owner;
  context.noteLayoutUsage(engine->id, engine->supportedLayouts,
                          event.tileLayout);

  context.emitEvent(std::move(event), gather.getInputs(), {gather.getResult()});
  return llvm::Error::success();
}

llvm::Error buildTransform(micro::TransformOp transform,
                           OpDAGBuildContext &context) {
  mlir::Operation &op = *transform;
  // A conversion is real work: it runs on a compute capability like the other
  // mapped elementwise work, and its selected resource is the executor the
  // plan stamped on it (`micro.engine`, task B8) -- the same one the planner
  // charged -- falling back to `micro.mapping` and then the machine default.
  // A hand-written or pre-B8 kernel may instead name the resource directly
  // with `micro.compute_resource`; when present it wins over the stamped
  // executor, so a stale or misspelled resource is rejected rather than
  // silently charged against some other engine.
  std::string reason;
  const machine::ComputeNode *engine = nullptr;
  if (mlir::Attribute raw = op.getAttr("micro.compute_resource")) {
    auto resource = mlir::dyn_cast<mlir::StringAttr>(raw);
    engine = resource ? context.machineModel().findCompute(resource.getValue())
                      : nullptr;
    if (!engine || engine->kind != "vector_engine")
      return invalid("micro.transform selects an invalid vector resource");
  } else {
    engine = context.pickTransformEngine(reason);
  }
  if (!engine)
    return invalid("kernel '" + context.kernelName().str() +
                   "' uses micro.transform: " + reason);

  TileInfo sourceInfo = describeType(transform.getSource().getType());
  TileInfo resultInfo = describeType(transform.getResult().getType());
  // The conversion reads the memory its source landed in and writes its
  // output there: a logical view carries no memory of its own, so the
  // producing event is asked (as `memoryOf` does for a copy source).
  std::string sourceMemory = context.memoryOf(transform.getSource());
  std::string destination =
      resultInfo.memory.empty() ? sourceMemory : resultInfo.memory;

  // A hand-written or pre-B8 kernel may name the landing memory explicitly;
  // the stamp wins over the memory the result type implies, so the cost model
  // charges the placement the kernel actually declared.
  if (mlir::Attribute raw = op.getAttr("micro.memory_node")) {
    auto memory = mlir::dyn_cast<mlir::StringAttr>(raw);
    if (!memory)
      return invalid("micro.transform memory node is not a string");
    destination = memory.getValue().str();
  }

  // The one shared estimate the planner and the simulator both charge. An
  // unknown resource or footprint comes back as an error, never a silent
  // zero, so a mis-modeled conversion cannot vanish from the report.
  mapping::TransformCostInput costInput;
  costInput.inputType = shapedCostType(transform.getSource().getType());
  costInput.outputType = shapedCostType(transform.getResult().getType());
  if (std::optional<mlir::AffineMap> srcMap = transform.getSrcMap())
    costInput.srcMap = *srcMap;
  if (std::optional<mlir::AffineMap> dstMap = transform.getDstMap())
    costInput.dstMap = *dstMap;
  costInput.memoryNode = destination;
  costInput.computeResource = engine->id;

  llvm::Expected<mapping::Cost> cost =
      mapping::estimateTransformCost(costInput, context.machineModel());
  if (!cost)
    return invalid(
        "kernel '" + context.kernelName().str() +
        "' uses micro.transform: " + llvm::toString(cost.takeError()));

  MicroEvent event;
  event.kind = EventKind::Transform;
  event.resource = ResourceKind::VectorEngine;
  event.resourceName = engine->id;
  event.ownerPool = transformExecutor(op).str();
  event.workItems =
      resultInfo.elements() ? resultInfo.elements() : sourceInfo.elements();
  event.bytes = cost->localBytes;
  event.minCycles = static_cast<uint64_t>(std::ceil(cost->latencyCycles));
  event.sourceOpName = op.getName().getStringRef().str();
  event.tileShape = shapeString(resultInfo.shape.empty() ? sourceInfo.shape
                                                         : resultInfo.shape);
  event.tileLayout =
      resultInfo.layout.empty() ? sourceInfo.layout : resultInfo.layout;
  event.tileMemory = destination;
  event.srcMemory = sourceMemory;
  event.tileOwner =
      resultInfo.owner.empty() ? sourceInfo.owner : resultInfo.owner;
  context.noteLayoutUsage(engine->id, engine->supportedLayouts,
                          event.tileLayout);

  const uint64_t bytes = event.bytes;
  context.emitEvent(std::move(event), {transform.getSource()},
                    {transform.getResult()});
  // The conversion materializes a fresh output buffer, which the capacity
  // check must see.
  context.accountStorage(destination, bytes);
  return llvm::Error::success();
}

llvm::Error buildWait(micro::WaitOp wait, OpDAGBuildContext &context) {
  mlir::Operation &op = *wait;
  MicroEvent event;
  event.kind = EventKind::Wait;
  event.resource = ResourceKind::Sync;
  event.resourceName = "sync";
  event.minCycles = context.machineModel().sync.waitCycles;
  event.sourceOpName = op.getName().getStringRef().str();
  context.emitEvent(std::move(event), wait.getTokens(), {});
  return llvm::Error::success();
}

llvm::Error buildBarrier(micro::BarrierOp barrier, OpDAGBuildContext &context) {
  mlir::Operation &op = *barrier;
  // A barrier is charged as synchronization, not as free program order. Its
  // scope must be one the dialect and the machine can express; an unknown
  // scope is rejected rather than silently treated as a full-machine fence.
  if (!micro::symbolizeBarrierScope(barrier.getScope()))
    return invalid("kernel '" + context.kernelName().str() +
                   "' uses micro.barrier: unknown "
                   "barrier scope '" +
                   barrier.getScope().str() + "'");
  MicroEvent event;
  event.kind = EventKind::Barrier;
  event.resource = ResourceKind::Sync;
  event.resourceName = "sync";
  event.minCycles = context.machineModel().sync.barrierCycles;
  event.sourceOpName = op.getName().getStringRef().str();
  // A barrier depends on every token it covers, so the work it orders cannot
  // be scheduled as if the barrier were absent.
  context.emitEvent(std::move(event), barrier.getTokens(), {});
  return llvm::Error::success();
}
} // namespace
OpDAGBuilderRegistry makeDefaultOpDAGBuilders() {
  OpDAGBuilderRegistry builders;
  // Immutable compile-time values do not issue hardware work.
  auto constant = [](mlir::Operation &, OpDAGBuildContext &) {
    return llvm::Error::success();
  };
  // Only scalar index arithmetic is control scaffolding. Integer/vector data
  // arithmetic must still report missing modeling coverage.
  auto indexOnly = [](mlir::Operation &op, OpDAGBuildContext &context) {
    if (op.getNumResults() != 1 || !op.getResult(0).getType().isIndex() ||
        op.getNumOperands() != 2 ||
        !llvm::all_of(op.getOperandTypes(),
                      [](mlir::Type type) { return type.isIndex(); }))
      context.noteIncomplete("unmodeled_leaf: " +
                             op.getName().getStringRef().str());
    return llvm::Error::success();
  };
  // tensor.empty describes shape with unspecified contents, without storage.
  // Canonical materialized storage is accounted by micro.{tile_alloc,alloc};
  // copies and physical layout changes retain their existing charged handlers.
  auto emptyTensor = [](mlir::Operation &op, OpDAGBuildContext &context) {
    if (op.getNumResults() != 1 ||
        !llvm::isa<mlir::RankedTensorType>(op.getResult(0).getType()) ||
        !llvm::all_of(op.getOperandTypes(),
                      [](mlir::Type type) { return type.isIndex(); }))
      context.noteIncomplete("unmodeled_leaf: tensor.empty");
    return llvm::Error::success();
  };
  llvm::cantFail(builders.registerZeroCost("arith.constant", constant));
  llvm::cantFail(builders.registerZeroCost("arith.addi", indexOnly));
  llvm::cantFail(builders.registerZeroCost("arith.muli", indexOnly));
  llvm::cantFail(builders.registerZeroCost("tensor.empty", emptyTensor));
  llvm::cantFail(builders.registerBuilder<micro::TileViewOp>(buildTileView));
  llvm::cantFail(
      builders.registerBuilder<micro::TilePartitionOp>(buildTilePartition));
  llvm::cantFail(builders.registerBuilder<micro::TileAllocOp>(buildTileAlloc));
  llvm::cantFail(builders.registerBuilder<micro::AllocOp>(buildAlloc));
  llvm::cantFail(
      builders.registerBuilder<micro::TileAsyncCopyOp>(buildTileAsyncCopy));
  llvm::cantFail(builders.registerBuilder<micro::AsyncCopyOp>(buildAsyncCopy));
  llvm::cantFail(builders.registerBuilder<micro::TileStoreOp>(buildTileStore));
  llvm::cantFail(builders.registerBuilder<micro::StoreOp>(buildStore));
  llvm::cantFail(builders.registerBuilder<micro::MmaOp>(buildMma));
  llvm::cantFail(builders.registerBuilder<micro::VectorOp>(buildVector));
  llvm::cantFail(builders.registerBuilder<micro::ReduceOp>(buildReduce));
  llvm::cantFail(builders.registerBuilder<micro::GatherOp>(buildGather));
  llvm::cantFail(builders.registerBuilder<micro::TransformOp>(buildTransform));
  llvm::cantFail(builders.registerBuilder<micro::WaitOp>(buildWait));
  llvm::cantFail(builders.registerBuilder<micro::BarrierOp>(buildBarrier));
  return builders;
}
} // namespace mlir::llk::perf
