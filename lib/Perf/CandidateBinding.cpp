//===- CandidateBinding.cpp - Bind a candidate into a concrete kernel -----===//
//
// Part of the M12 tuning core (issue #50). See CandidateBinding.h.
//
// The binder is the tuning stage's counterpart to the LLKToMicro export: same
// tile program, same verifier constraints, different input. The export reads a
// ScheduleEntry selected from `schedule_db.json`; the binder reads a Candidate
// generated from a `micro.search_space`. Both end at a concrete `micro.kernel`,
// and the two have to agree on what a memory path, an owner hierarchy, or a
// fragment shape means, so the resolution rules here follow the export's:
//
//   * a tile is clamped to its problem extent, and a tile that still does not
//     divide its extent is rejected -- the dialect lowers no masks
//   * the memory path's first space is where external tensors live, the second
//     is the staging level (which must be on-chip), and a third, when present,
//     is the accumulator; without one the accumulator defaults to `acc`
//   * an owner mapping names the outer scope that maps the tiled axes and the
//     inner scope that owns instruction fragments; a single owner is completed
//     with `lane`, exactly as the export completes it
//   * the MMA runs over the whole staged tile and its `shape` is the worker
//     tile, because MmaOp's verifier requires operands of [M,K], [K,N], [M,N]
//     and the cost model charges `shape` as the work done; the instruction
//     fragment is metadata on `micro.tile_partition` and the kernel attributes
//
// Symbolic roles are found through the parameter's dialect `kind`, not its
// name, so a space that calls its layout parameter something else still binds.
// A role the space does not declare falls back to the export's default rather
// than failing: binding a space that declines to search a dimension should use
// the conservative value, not error out.
//
//===----------------------------------------------------------------------===//

#include "LLK/Perf/CandidateBinding.h"

#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Dialect/Micro/MicroEnums.h"
#include "LLK/Dialect/Micro/MicroHelpers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

// Micro attribute, type, and op declarations.
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

using namespace mlir;

namespace mlir::llk::perf {

using llvm::StringRef;

namespace {

//===----------------------------------------------------------------------===//
// Small helpers
//===----------------------------------------------------------------------===//

llvm::Error invalid(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

/// Largest divisor of `extent` that does not exceed `bound`, or 1 when there is
/// none. Both arguments must be positive.
int64_t largestDivisorAtMost(int64_t extent, int64_t bound) {
  if (bound >= extent)
    return extent;
  for (int64_t candidate = std::max<int64_t>(bound, 1); candidate > 1;
       --candidate)
    if (extent % candidate == 0)
      return candidate;
  return 1;
}

/// Parses a `MxNxK` triple such as `16x16x32`.
FailureOr<SmallVector<int64_t, 3>> parseShapeTriple(StringRef text) {
  SmallVector<StringRef, 3> parts;
  text.split(parts, 'x');
  if (parts.size() != 3)
    return failure();

  SmallVector<int64_t, 3> shape;
  for (StringRef part : parts) {
    int64_t value = 0;
    if (part.getAsInteger(10, value) || value <= 0)
      return failure();
    shape.push_back(value);
  }
  return shape;
}

/// The MLIR element type for a Micro dtype name, or nullopt when the name is
/// not a dtype the dialect models.
std::optional<Type> elementTypeFor(MLIRContext *context, StringRef name) {
  std::optional<micro::DType> dtype = micro::symbolizeDType(name);
  if (!dtype)
    return std::nullopt;
  switch (*dtype) {
  case micro::DType::f32:
    return Float32Type::get(context);
  case micro::DType::f16:
    return Float16Type::get(context);
  case micro::DType::bf16:
    return BFloat16Type::get(context);
  case micro::DType::i32:
    return IntegerType::get(context, 32);
  case micro::DType::i8:
    return IntegerType::get(context, 8);
  }
  return std::nullopt;
}

/// Returns the spatial mapping target of the same name as `owner`.
///
/// `MappingTarget` and `Owner` are independent vocabularies with different
/// enumerator values, so this is a name match, not a numeric cast. Returns
/// nullopt for the owner scopes that have no spatial axis.
std::optional<micro::MappingTarget> spatialTargetFor(micro::Owner owner) {
  switch (owner) {
  case micro::Owner::cluster:
    return micro::MappingTarget::cluster_x;
  case micro::Owner::core:
    return micro::MappingTarget::core_x;
  case micro::Owner::pe:
    return micro::MappingTarget::pe_x;
  case micro::Owner::lane:
    return micro::MappingTarget::lane;
  case micro::Owner::worker:
    return micro::MappingTarget::worker;
  case micro::Owner::matrix_engine:
    return micro::MappingTarget::matrix_engine;
  case micro::Owner::vector_engine:
    return micro::MappingTarget::vector_engine;
  case micro::Owner::dma:
    return micro::MappingTarget::dma;
  case micro::Owner::warp:
  case micro::Owner::wave:
  case micro::Owner::subgroup:
  case micro::Owner::pe_group:
    return std::nullopt;
  }
  return std::nullopt;
}

/// Builds a `#micro.layout` attribute. `vectorWidth` is 0 when the layout makes
/// no claim about vectorization.
micro::LayoutAttr makeLayoutAttr(MLIRContext *context, micro::LayoutKind kind,
                                 int64_t vectorWidth) {
  return micro::LayoutAttr::get(context, static_cast<uint32_t>(kind),
                                DenseI64ArrayAttr(), vectorWidth, 0, 0,
                                StringAttr());
}

/// Creates a fresh single-block region body, terminates it with `micro.yield`,
/// and leaves `builder` positioned immediately before that terminator so the
/// region's contents can be emitted in order.
Block *startRegionBody(OpBuilder &builder, Region &region, Location loc,
                       TypeRange argTypes = {}) {
  Block *block = builder.createBlock(
      &region, {}, argTypes, SmallVector<Location>(argTypes.size(), loc));
  builder.setInsertionPointToEnd(block);
  micro::YieldOp::create(builder, loc);
  builder.setInsertionPoint(block->getTerminator());
  return block;
}

//===----------------------------------------------------------------------===//
// Resolution
//===----------------------------------------------------------------------===//

/// The bound value of the parameter declared with `kind`, resolved through the
/// space so the binder can find "the layout" without knowing its name.
std::optional<StringRef> symbolOfKind(const SearchSpace &space,
                                      const Candidate &candidate,
                                      StringRef kind) {
  const SearchParam *param = space.findParamOfKind(kind);
  if (!param)
    return std::nullopt;
  return candidate.symbol(param->name);
}

/// Reads a required integer parameter, reporting which binding is missing
/// rather than defaulting to zero.
llvm::Expected<int64_t> requiredInteger(const Candidate &candidate,
                                        StringRef name) {
  std::optional<int64_t> value = candidate.integer(name);
  if (!value)
    return invalid("candidate does not bind parameter '" + name + "'");
  return *value;
}

/// Clamps `requested` to `extent` and requires the result to divide it. A
/// non-positive request is rejected rather than clamped to 1, because it is a
/// malformed binding, not a small tile.
llvm::Expected<int64_t> resolveTile(StringRef axis, int64_t extent,
                                    int64_t requested) {
  if (requested <= 0)
    return invalid("candidate has no " + axis + " tile size");
  int64_t tile = std::min(requested, extent);
  if (extent % tile != 0)
    return invalid(axis + " tile " + llvm::Twine(tile) + " does not divide " +
                   axis + " = " + llvm::Twine(extent) +
                   "; the binder emits no tile masks");
  return tile;
}

/// Resolves the candidate's memory path into its source, staging, and
/// accumulator spaces. The path must name at least two known spaces and stage
/// through on-chip memory.
llvm::Error resolveMemoryPath(StringRef path, std::vector<std::string> &spaces,
                              std::string &source, std::string &staging,
                              std::string &accumulator) {
  llvm::SmallVector<StringRef, 4> parts;
  path.split(parts, ':');
  if (parts.size() < 2)
    return invalid("memory_path '" + path +
                   "' must join at least two memory spaces with ':'");
  for (StringRef space : parts) {
    if (!micro::symbolizeMemorySpace(space))
      return invalid("memory_path '" + path + "' names unknown memory space '" +
                     space + "'");
    spaces.push_back(space.str());
  }

  source = parts[0].str();
  staging = parts[1].str();
  if (parts[1] == "dram")
    return invalid("memory_path '" + path +
                   "' stages tiles in dram; the staged level must be on-chip");
  // The path's last level is where compute accumulates; a two-level path says
  // nothing about the accumulator, so it defaults to `acc`.
  accumulator = (parts.size() >= 3 ? parts.back() : StringRef("acc")).str();
  return llvm::Error::success();
}

/// Resolves the owner mapping into an outer scope and a fragment owner. A
/// single owner names the tiled axes; the other keeps `lane`, which is what the
/// export uses for the AVX2 validation backend.
llvm::Error resolveOwnerMapping(StringRef mapping, std::string &outer,
                                std::string &fragment) {
  if (mapping.empty())
    return invalid("owner_mapping must name at least one owner");
  llvm::SmallVector<StringRef, 4> owners;
  mapping.split(owners, '/');
  if (owners.size() > 2)
    return invalid("owner_mapping '" + mapping +
                   "' names more than two owner levels; the binder emits one "
                   "spatial loop per tiled axis");
  for (StringRef owner : owners) {
    std::optional<micro::Owner> resolved = micro::symbolizeOwner(owner);
    if (!resolved)
      return invalid("owner_mapping names unknown owner '" + owner + "'");
    if (!spatialTargetFor(*resolved))
      return invalid("owner_mapping owner '" + owner +
                     "' has no spatial axis to map onto");
  }
  outer = owners[0].str();
  fragment = (owners.size() > 1 ? owners[1] : StringRef("lane")).str();
  return llvm::Error::success();
}

/// Everything the kernel emitter needs beyond the candidate's decisions.
struct KernelContext {
  bool fused = false;
  Type inputElemType;
  Type accumulatorElemType;
  Type outputElemType;
};

llvm::Expected<KernelContext> resolveContext(MLIRContext *context,
                                             const SearchSpace &space,
                                             const WorkloadShape &shape) {
  auto input = elementTypeFor(context, shape.inputDType);
  auto accumulator = elementTypeFor(context, shape.accumulatorDType);
  auto output = elementTypeFor(context, shape.outputDType);
  if (!input)
    return invalid("input dtype '" + shape.inputDType +
                   "' is not a tile dtype");
  if (!accumulator)
    return invalid("accumulator dtype '" + shape.accumulatorDType +
                   "' is not a tile dtype");
  if (!output)
    return invalid("output dtype '" + shape.outputDType +
                   "' is not a tile dtype");
  if (!elementTypeFor(context, shape.weightDType))
    return invalid("weight dtype '" + shape.weightDType +
                   "' is not a tile dtype");

  return KernelContext{space.workload == "fused_swiglu", *input, *accumulator,
                       *output};
}

llvm::Expected<BoundTileDecisions>
resolveDecisions(MLIRContext *context, const SearchSpace &space,
                 const Candidate &candidate, const WorkloadShape &shape) {
  if (shape.M <= 0 || shape.N <= 0 || shape.K <= 0)
    return invalid("workload shape must have positive M, N, and K");

  BoundTileDecisions decisions;
  decisions.mBucket = classifyMBucket(shape.M);

  llvm::Expected<int64_t> bm = requiredInteger(candidate, "BM");
  if (!bm)
    return bm.takeError();
  llvm::Expected<int64_t> bn = requiredInteger(candidate, "BN");
  if (!bn)
    return bn.takeError();
  llvm::Expected<int64_t> bk = requiredInteger(candidate, "BK");
  if (!bk)
    return bk.takeError();

  llvm::Expected<int64_t> m = resolveTile("BM", shape.M, *bm);
  if (!m)
    return m.takeError();
  llvm::Expected<int64_t> n = resolveTile("BN", shape.N, *bn);
  if (!n)
    return n.takeError();
  llvm::Expected<int64_t> k = resolveTile("BK", shape.K, *bk);
  if (!k)
    return k.takeError();
  decisions.workerTile = {*m, *n, *k};

  // --- layout ------------------------------------------------------------
  StringRef layoutName =
      symbolOfKind(space, candidate, "layout").value_or("row_major");
  std::optional<micro::LayoutKind> layout =
      micro::symbolizeLayoutKind(layoutName);
  if (!layout)
    return invalid("tile_layout '" + layoutName + "' is not a known layout");
  decisions.tileLayout = layoutName.str();

  // --- memory path -------------------------------------------------------
  StringRef memoryPath =
      symbolOfKind(space, candidate, "memory_path").value_or("dram:sram:acc");
  llvm::Error pathError =
      resolveMemoryPath(memoryPath, decisions.memoryPath, decisions.sourceSpace,
                        decisions.stagingSpace, decisions.accumulatorSpace);
  if (pathError)
    return std::move(pathError);

  // --- owner mapping -----------------------------------------------------
  StringRef ownerMapping =
      symbolOfKind(space, candidate, "owner_mapping").value_or("worker/lane");
  llvm::Error ownerError = resolveOwnerMapping(
      ownerMapping, decisions.outerOwner, decisions.fragmentOwner);
  if (ownerError)
    return std::move(ownerError);

  // --- instruction fragment ---------------------------------------------
  // A space that does not search the fragment binds the whole worker tile: the
  // partition would carry no information and is not emitted.
  SmallVector<int64_t, 3> declared(decisions.workerTile.begin(),
                                   decisions.workerTile.end());
  if (std::optional<StringRef> bound =
          symbolOfKind(space, candidate, "fragment_shape")) {
    FailureOr<SmallVector<int64_t, 3>> parsed = parseShapeTriple(*bound);
    if (failed(parsed))
      return invalid("fragment_shape '" + *bound +
                     "' must be a positive MxNxK triple");
    declared = *parsed;
  }
  decisions.declaredFragment.assign(declared.begin(), declared.end());
  decisions.fragmentShape = {
      largestDivisorAtMost(decisions.workerTile[0],
                           std::min(declared[0], decisions.workerTile[0])),
      largestDivisorAtMost(decisions.workerTile[1],
                           std::min(declared[1], decisions.workerTile[1])),
      largestDivisorAtMost(decisions.workerTile[2],
                           std::min(declared[2], decisions.workerTile[2])),
  };

  decisions.tailPolicy =
      symbolOfKind(space, candidate, "tail_policy").value_or("none").str();
  decisions.pipelineStages =
      std::max<int64_t>(1, candidate.integer("pipeline_stages").value_or(1));
  decisions.vectorWidth =
      std::max<int64_t>(0, candidate.integer("vector_width").value_or(0));
  return decisions;
}

//===----------------------------------------------------------------------===//
// Emission
//===----------------------------------------------------------------------===//

/// Appends one concrete `micro.kernel` for `decisions` to `module`.
micro::KernelOp emitKernel(ModuleOp module, const SearchSpace &space,
                           const Candidate &candidate,
                           const WorkloadShape &shape,
                           const KernelContext &context,
                           const BoundTileDecisions &decisions) {
  MLIRContext *ctx = module.getContext();
  Location loc = module.getLoc();

  std::string symName = space.workload + "_" + candidate.id;
  OpBuilder builder(ctx);
  builder.setInsertionPointToEnd(module.getBody());

  // --- the kernel's explicit contract -------------------------------------
  // The kernel reads its operands from outside and produces the output tensor.
  // An argumentless kernel is an analysis artifact: nothing can invoke it, so a
  // predicted cost for it is a claim about code that could not be run.
  llvm::SmallVector<Type, 3> inputTypes{
      RankedTensorType::get({shape.M, shape.K}, context.inputElemType)};
  for (size_t arm = 0; arm < (context.fused ? 2u : 1u); ++arm)
    inputTypes.push_back(
        RankedTensorType::get({shape.K, shape.N}, context.inputElemType));
  llvm::SmallVector<Type, 1> resultTypes{
      RankedTensorType::get({shape.M, shape.N}, context.outputElemType)};

  auto kernel = micro::KernelOp::create(
      builder, loc, symName,
      TypeAttr::get(FunctionType::get(ctx, inputTypes, resultTypes)),
      StringAttr::get(ctx, space.workload),
      /*target=*/StringAttr(), StringAttr::get(ctx, candidate.id),
      IntegerAttr::get(IntegerType::get(ctx, 64), decisions.mBucket));

  // Schedule intent the concrete tile types and op attributes cannot carry
  // themselves. The kernel is what the schedule record and the simulator read.
  kernel->setAttr("memory_path",
                  StringAttr::get(ctx, llvm::join(decisions.memoryPath, ":")));
  kernel->setAttr("owner_mapping",
                  StringAttr::get(ctx, decisions.outerOwner + "/" +
                                           decisions.fragmentOwner));
  kernel->setAttr("tile_layout", StringAttr::get(ctx, decisions.tileLayout));
  kernel->setAttr("mma_shape",
                  DenseI64ArrayAttr::get(ctx, decisions.declaredFragment));
  kernel->setAttr("fragment_shape",
                  DenseI64ArrayAttr::get(ctx, decisions.fragmentShape));
  kernel->setAttr("tail_policy", StringAttr::get(ctx, decisions.tailPolicy));

  Block *kernelBody =
      startRegionBody(builder, kernel.getBody(), loc, inputTypes);

  // --- tile types --------------------------------------------------------
  std::optional<micro::LayoutKind> layout =
      micro::symbolizeLayoutKind(decisions.tileLayout);
  micro::LayoutAttr layoutAttr =
      makeLayoutAttr(ctx, *layout, decisions.vectorWidth);
  micro::OwnerAttr outerOwner =
      micro::OwnerAttr::get(ctx, *micro::symbolizeOwner(decisions.outerOwner));
  micro::OwnerAttr fragmentOwner = micro::OwnerAttr::get(
      ctx, *micro::symbolizeOwner(decisions.fragmentOwner));
  micro::MemorySpaceAttr srcSpace = micro::MemorySpaceAttr::get(
      ctx, *micro::symbolizeMemorySpace(decisions.sourceSpace));
  micro::MemorySpaceAttr stageSpace = micro::MemorySpaceAttr::get(
      ctx, *micro::symbolizeMemorySpace(decisions.stagingSpace));
  micro::MemorySpaceAttr accSpace = micro::MemorySpaceAttr::get(
      ctx, *micro::symbolizeMemorySpace(decisions.accumulatorSpace));

  auto tile = [&](ArrayRef<int64_t> shape, Type element,
                  micro::LayoutAttr tileLayout, micro::MemorySpaceAttr memory,
                  micro::OwnerAttr owner) {
    return micro::TileType::get(ctx, shape, element, tileLayout, memory, owner);
  };

  // --- external tensors --------------------------------------------------
  // The kernel is IsolatedFromAbove, so the operands it reads are its own entry
  // block arguments -- the ones the signature declared.
  const bool fused = context.fused;
  int64_t BM = decisions.workerTile[0];
  int64_t BN = decisions.workerTile[1];
  int64_t BK = decisions.workerTile[2];

  SmallVector<int64_t, 2> lhsExtent{shape.M, shape.K};
  SmallVector<int64_t, 2> rhsExtent{shape.K, shape.N};
  Value lhsTensor = kernelBody->getArgument(0);
  SmallVector<Value, 2> rhsTensors;
  for (size_t arm = 0; arm < (fused ? 2u : 1u); ++arm)
    rhsTensors.push_back(kernelBody->getArgument(1 + arm));

  // --- spatial tiling ----------------------------------------------------
  auto openSpatialLoop = [&](int64_t extent, int64_t step, StringRef ownerName,
                             ValueRange carried) -> micro::SpatialForOp {
    std::optional<micro::MappingTarget> target =
        spatialTargetFor(*micro::symbolizeOwner(ownerName));
    Value lower = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
    Value upper =
        arith::ConstantIndexOp::create(builder, loc, extent).getResult();
    Value by = arith::ConstantIndexOp::create(builder, loc, step).getResult();
    auto loop = micro::SpatialForOp::create(
        builder, loc, carried.getTypes(), lower, upper, by,
        micro::MappingTargetAttr::get(ctx, *target), carried);
    llvm::SmallVector<Type> bodyTypes{IndexType::get(ctx)};
    bodyTypes.append(carried.getTypes().begin(), carried.getTypes().end());
    startRegionBody(builder, loop.getBody(), loc, bodyTypes);
    return loop;
  };

  // --- the output the kernel produces -------------------------------------
  // The kernel writes its output tile by tile from inside the spatial nest, so
  // the output is a value the loops carry: an SSA destination is only updated
  // if the updated destination is what the enclosing loop hands on. It is built
  // in the staging level because a local allocation may not target dram -- what
  // makes it the kernel's *external* output is the declared result.
  micro::MemorySpaceAttr outputSpace =
      micro::MemorySpaceAttr::get(ctx, micro::MemorySpace::sram);
  Type outputTileType = tile({shape.M, shape.N}, context.outputElemType,
                             /*tileLayout=*/layoutAttr, outputSpace,
                             /*owner=*/micro::OwnerAttr());
  Value output =
      micro::TileAllocOp::create(builder, loc, outputTileType).getResult();

  micro::SpatialForOp bmLoop =
      openSpatialLoop(shape.M, BM, decisions.outerOwner, ValueRange{output});
  BlockArgument bm = bmLoop.getBody().front().getArgument(0);
  Value bmCarried = bmLoop.getBody().front().getArgument(1);

  micro::SpatialForOp bnLoop = openSpatialLoop(
      shape.N, BN, decisions.fragmentOwner, ValueRange{bmCarried});
  BlockArgument bn = bnLoop.getBody().front().getArgument(0);
  Value bnCarried = bnLoop.getBody().front().getArgument(1);

  // --- accumulators ------------------------------------------------------
  SmallVector<int64_t, 2> accExtent{BM, BN};
  Type accTileType =
      tile(accExtent, context.accumulatorElemType,
           /*tileLayout=*/micro::LayoutAttr(), accSpace, outerOwner);
  SmallVector<Value, 2> accumulators;
  for (size_t arm = 0; arm < rhsTensors.size(); ++arm)
    accumulators.push_back(
        micro::TileAllocOp::create(builder, loc, accTileType).getResult());

  // --- K loop ------------------------------------------------------------
  Value kLower = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  Value kUpper =
      arith::ConstantIndexOp::create(builder, loc, shape.K).getResult();
  Value kStep = arith::ConstantIndexOp::create(builder, loc, BK).getResult();
  // What an iteration computes is what the next one accumulates into, and what
  // the last one left is the kernel's result. Without the carry the MMA inside
  // the loop would compute dead results while the epilogue read an allocation
  // nothing had written.
  llvm::SmallVector<Type> carriedTypes(accumulators.size(), accTileType);
  auto kLoop = micro::ForOp::create(builder, loc, carriedTypes, kLower, kUpper,
                                    kStep, ValueRange(accumulators));
  llvm::SmallVector<Type> kBodyTypes{IndexType::get(ctx)};
  kBodyTypes.append(carriedTypes);
  Block *kBody = startRegionBody(builder, kLoop.getBody(), loc, kBodyTypes);
  BlockArgument bk = kBody->getArgument(0);
  llvm::SmallVector<BlockArgument> carried;
  for (size_t arm = 0; arm < accumulators.size(); ++arm)
    carried.push_back(kBody->getArgument(arm + 1));

  // `micro.pipeline` goes inside the loop body: that is the position MicroDAG
  // reads as software pipelining. It carries the accumulators out of its own
  // body, because the MMA runs inside it and the loop's terminator could not
  // otherwise see them.
  auto pipeline = micro::PipelineOp::create(
      builder, loc, carriedTypes,
      static_cast<uint64_t>(decisions.pipelineStages));
  startRegionBody(builder, pipeline.getBody(), loc);

  // --- logical views and staged copies -----------------------------------
  SmallVector<int64_t, 2> lhsTileExtent{BM, BK};
  Type lhsViewType = tile(lhsTileExtent, context.inputElemType, layoutAttr,
                          srcSpace, /*owner=*/micro::OwnerAttr());
  Value lhsView = micro::TileViewOp::create(
      builder, loc, lhsViewType, lhsTensor, ValueRange{bm, bk},
      DenseI64ArrayAttr::get(ctx, lhsTileExtent), layoutAttr);

  SmallVector<Value, 2> tokens;
  Type lhsStagedType = tile(lhsTileExtent, context.inputElemType, layoutAttr,
                            stageSpace, outerOwner);
  auto lhsCopy = micro::TileAsyncCopyOp::create(
      builder, loc, lhsStagedType, micro::AsyncTokenType::get(ctx), lhsView,
      stageSpace, outerOwner);
  Value lhsStaged = lhsCopy.getResult();
  tokens.push_back(lhsCopy.getToken());

  SmallVector<int64_t, 2> rhsTileExtent{BK, BN};
  SmallVector<Value, 2> rhsStaged;
  for (Value source : rhsTensors) {
    Type viewType = tile(rhsTileExtent, context.inputElemType, layoutAttr,
                         srcSpace, /*owner=*/micro::OwnerAttr());
    Value view = micro::TileViewOp::create(
        builder, loc, viewType, source, ValueRange{bk, bn},
        DenseI64ArrayAttr::get(ctx, rhsTileExtent), layoutAttr);
    Type stagedType = tile(rhsTileExtent, context.inputElemType, layoutAttr,
                           stageSpace, outerOwner);
    auto copy = micro::TileAsyncCopyOp::create(builder, loc, stagedType,
                                               micro::AsyncTokenType::get(ctx),
                                               view, stageSpace, outerOwner);
    rhsStaged.push_back(copy.getResult());
    tokens.push_back(copy.getToken());
  }

  micro::WaitOp::create(builder, loc, tokens);

  // --- instruction fragments ---------------------------------------------
  // Logical, zero-cost annotations: a partition whose fragment covers its
  // whole parent carries no information and is not emitted. The MMA consumes
  // the staged tiles, not these, because the fragment shape has to stay
  // consistent with the MMA's `shape`.
  int64_t fM = decisions.fragmentShape[0];
  int64_t fN = decisions.fragmentShape[1];
  int64_t fK = decisions.fragmentShape[2];
  SmallVector<int64_t, 2> lhsFragmentExtent{fM, fK};
  SmallVector<int64_t, 2> rhsFragmentExtent{fK, fN};
  if (fM != BM || fK != BK)
    micro::TilePartitionOp::create(
        builder, loc,
        tile(lhsFragmentExtent, context.inputElemType, layoutAttr, stageSpace,
             fragmentOwner),
        lhsStaged, DenseI64ArrayAttr::get(ctx, lhsFragmentExtent),
        fragmentOwner, /*tail=*/BoolAttr());
  if (fK != BK || fN != BN)
    for (Value staged : rhsStaged)
      micro::TilePartitionOp::create(
          builder, loc,
          tile(rhsFragmentExtent, context.inputElemType, layoutAttr, stageSpace,
               fragmentOwner),
          staged, DenseI64ArrayAttr::get(ctx, rhsFragmentExtent), fragmentOwner,
          /*tail=*/BoolAttr());

  // --- MMA ---------------------------------------------------------------
  DenseI64ArrayAttr mmaShape =
      DenseI64ArrayAttr::get(ctx, SmallVector<int64_t, 3>{BM, BN, BK});
  llvm::SmallVector<Value> nextAccumulators;
  for (size_t arm = 0; arm < rhsStaged.size(); ++arm)
    nextAccumulators.push_back(
        micro::MmaOp::create(
            builder, loc, accTileType, lhsStaged, rhsStaged[arm], carried[arm],
            mmaShape,
            micro::DTypeAttr::get(
                ctx, *micro::dtypeOfElementType(context.inputElemType)),
            micro::DTypeAttr::get(
                ctx, *micro::dtypeOfElementType(context.accumulatorElemType)),
            /*engine=*/StringAttr())
            .getResult());

  // The pipeline hands the computed values out of its body, and the loop hands
  // them to the next iteration. Both steps are what make the carry real rather
  // than implied.
  if (auto terminator = dyn_cast<micro::YieldOp>(
          pipeline.getBody().front().getTerminator())) {
    builder.setInsertionPoint(terminator);
    micro::YieldOp::create(builder, loc, nextAccumulators);
    terminator.erase();
  }
  if (auto terminator = dyn_cast<micro::YieldOp>(kBody->getTerminator())) {
    builder.setInsertionPoint(terminator);
    micro::YieldOp::create(builder, loc, pipeline.getResults());
    terminator.erase();
  }

  // Everything after the loop reads what the loop left.
  for (size_t arm = 0; arm < accumulators.size(); ++arm)
    accumulators[arm] = kLoop.getResults()[arm];

  // The epilogue belongs to the output tile, not to one K iteration.
  builder.setInsertionPoint(kLoop->getNextNode());

  // --- elementwise epilogue ----------------------------------------------
  Value epilogue = accumulators[0];
  if (fused) {
    Value gate = micro::VectorOp::create(builder, loc, accTileType, "silu",
                                         ValueRange{accumulators[0]},
                                         /*math_mode=*/StringAttr());
    epilogue = micro::VectorOp::create(builder, loc, accTileType, "mul",
                                       ValueRange{gate, accumulators[1]},
                                       /*math_mode=*/StringAttr());
  }

  // When the accumulator already holds the output element type the conversion
  // would be an identity, so it is not emitted.
  if (context.accumulatorElemType != context.outputElemType) {
    Type convertedType =
        tile(accExtent, context.outputElemType,
             /*tileLayout=*/micro::LayoutAttr(), accSpace, outerOwner);
    epilogue = micro::VectorOp::create(builder, loc, convertedType, "convert",
                                       ValueRange{epilogue},
                                       /*math_mode=*/StringAttr());
  }

  // Writing the tile into the carried output is what updates it: the result is
  // the destination with this tile written at the loop's own coordinates.
  Value written =
      micro::TileStoreOp::create(builder, loc, outputTileType, epilogue,
                                 bnCarried, ValueRange{bm, bn}, outputSpace)
          .getResult();

  if (auto terminator =
          dyn_cast<micro::YieldOp>(bnLoop.getBody().front().getTerminator())) {
    builder.setInsertionPoint(terminator);
    micro::YieldOp::create(builder, loc, ValueRange{written});
    terminator.erase();
  }
  if (auto terminator =
          dyn_cast<micro::YieldOp>(bmLoop.getBody().front().getTerminator())) {
    builder.setInsertionPoint(terminator);
    micro::YieldOp::create(builder, loc, bnLoop.getResults());
    terminator.erase();
  }
  if (auto terminator = dyn_cast<micro::YieldOp>(kernelBody->getTerminator())) {
    builder.setInsertionPoint(terminator);
    micro::YieldOp::create(builder, loc, bmLoop.getResults());
    terminator.erase();
  }

  builder.setInsertionPointToEnd(module.getBody());
  return kernel;
}

} // namespace

llvm::Expected<BoundKernel>
bindCandidateToMicroKernel(mlir::ModuleOp module, const SearchSpace &space,
                           const Candidate &candidate,
                           const WorkloadShape &shape) {
  MLIRContext *context = module.getContext();
  if (!context->getOrLoadDialect<micro::MicroDialect>() ||
      !context->getOrLoadDialect<tensor::TensorDialect>() ||
      !context->getOrLoadDialect<arith::ArithDialect>())
    return invalid("the micro, tensor, and arith dialects must be registered "
                   "in this context");

  llvm::Expected<KernelContext> kernelContext =
      resolveContext(context, space, shape);
  if (!kernelContext)
    return kernelContext.takeError();

  llvm::Expected<BoundTileDecisions> decisions =
      resolveDecisions(context, space, candidate, shape);
  if (!decisions)
    return decisions.takeError();

  BoundKernel bound;
  bound.workload = space.workload;
  bound.decisions = std::move(*decisions);
  micro::KernelOp kernel = emitKernel(module, space, candidate, shape,
                                      *kernelContext, bound.decisions);
  bound.kernel = kernel.getOperation();
  bound.symbolName = kernel.getSymName().str();
  return bound;
}

} // namespace mlir::llk::perf
