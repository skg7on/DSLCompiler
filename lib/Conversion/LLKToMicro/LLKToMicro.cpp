//===- LLKToMicro.cpp - LLK -> concrete micro.kernel lowering -------------===//
//
// Exports the LLK pipeline's semantic structure as tile-centric Micro-IR. One
// concrete `micro.kernel` is appended to the module per supported LLK root
// operation; the original functions are left untouched, so the AVX2/JIT
// pipeline downstream of this pass is unaffected.
//
// Lowering resolves the tile program in the order the Micro-IR design calls
// for:
//
//   external tensor
//     -> logical tile view          micro.tile_view
//     -> materialized memory tile   micro.tile_async_copy + micro.wait
//     -> instruction fragment       micro.tile_partition
//     -> MMA / vector fragment      micro.mma, micro.vector
//     -> write back                 micro.tile_store
//
// Two choices differ from the illustrative example in the M11 spec, and both
// come from the dialect's own verifiers rather than from preference:
//
//   * The MMA runs over the whole materialized tile, and its `shape` names the
//     `[BM, BN, BK]` instruction fragment it covers. MmaOp's verifier requires
//     operands of exactly `[M,K]`, `[K,N]` and `[M,N]`, and the cost model
//     charges `shape` as the work done -- so a sub-tile MMA would both fail
//     verification and silently under-count the kernel's flops.
//   * `micro.tile_partition` is therefore fragment metadata. The dialect
//     defines it as a logical, zero-cost op, and it is emitted only where the
//     instruction fragment is genuinely smaller than the tile it was cut from.
//
// `micro.pipeline` sits inside the K loop rather than around it because that is
// the position MicroDAG reads as loop pipelining; with one stage the two
// spellings are equivalent.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/LLKToMicro/LLKToMicro.h"

#include "LLK/Dialect/LLKEnums.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Dialect/Micro/MicroEnums.h"
#include "LLK/Dialect/Micro/MicroHelpers.h"
#include "LLK/Transforms/Common/ScheduleLoader.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <iterator>

// LLK attribute and op declarations.
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/LLKAttributes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/LLKOps.h.inc"

// Micro attribute, type, and op declarations.
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

using namespace mlir;
using llk::ScheduleEntry;

namespace {

//===----------------------------------------------------------------------===//
// Small helpers
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// Search-space choice builders
//===----------------------------------------------------------------------===//

// The numeric values each dimension is searched over -- the same grid
// `llk-tune` generates candidates from. A search space that offered choices the
// tuner cannot generate, or omitted the ones it can, would describe a different
// search. `BM` is pruned per M bucket rather than used as-is.
constexpr int64_t kSearchBM[] = {1, 4, 8, 16, 32, 64};
constexpr int64_t kSearchBN[] = {16, 32, 64, 128, 256};
constexpr int64_t kSearchBK[] = {32, 64, 128, 256};
constexpr int64_t kSearchVM[] = {1, 2, 4};
constexpr int64_t kSearchVN[] = {4, 8};
constexpr int64_t kSearchVectorWidth[] = {8};
constexpr int64_t kSearchThreads[] = {1, 2, 4, 8};
constexpr int64_t kSearchGrain[] = {1, 2, 4};
constexpr int64_t kSearchStages[] = {1, 2};
constexpr int64_t kSearchPrefetch[] = {1, 2};

/// Builds an integer choice list. The dialect requires integer choices to be
/// positive and strictly increasing, so the values are filtered, sorted, and
/// deduplicated; a non-positive value is dropped because it is not legal.
ArrayAttr makeIntegerChoices(MLIRContext *context, ArrayRef<int64_t> values) {
  SmallVector<int64_t, 8> sorted;
  for (int64_t value : values)
    if (value > 0)
      sorted.push_back(value);
  llvm::sort(sorted);
  sorted.erase(std::unique(sorted.begin(), sorted.end()), sorted.end());

  SmallVector<Attribute, 8> choices;
  for (int64_t value : sorted)
    choices.push_back(IntegerAttr::get(IntegerType::get(context, 64), value));
  return ArrayAttr::get(context, choices);
}

/// Builds a string choice list, keeping the first occurrence of each value and
/// preserving order so the scheduled value can be printed first.
ArrayAttr makeStringChoices(MLIRContext *context,
                            ArrayRef<std::string> values) {
  SmallVector<Attribute, 8> choices;
  llvm::StringSet<> seen;
  for (const std::string &value : values) {
    if (value.empty() || !seen.insert(value).second)
      continue;
    choices.push_back(StringAttr::get(context, value));
  }
  return ArrayAttr::get(context, choices);
}

/// The numeric grid `llk-tune` searches, plus the scheduled value when the grid
/// does not already contain it. Keeping the scheduled value in the space is
/// what lets the tuner reproduce the schedule in hand.
ArrayAttr makeNumericChoices(MLIRContext *context, ArrayRef<int64_t> grid,
                             int64_t scheduled) {
  SmallVector<int64_t, 8> values(grid);
  values.push_back(scheduled);
  return makeIntegerChoices(context, values);
}

/// Legal BM choices for `mBucket`. The llk-tune M-bucket rules become search
/// legality here: a bucket-0 problem is GEMV-like, so only BM = 1 is legal, and
/// the large-M buckets reject tiles no larger than 4.
ArrayAttr makeBmChoices(MLIRContext *context, int64_t scheduled,
                        int64_t mBucket) {
  SmallVector<int64_t, 8> candidates(std::begin(kSearchBM),
                                     std::end(kSearchBM));
  candidates.push_back(scheduled);

  SmallVector<int64_t, 8> legal;
  for (int64_t value : candidates) {
    if (mBucket == 0 && value != 1)
      continue;
    if (mBucket >= 3 && value <= 4)
      continue;
    legal.push_back(value);
  }
  return makeIntegerChoices(context, legal);
}

/// The tile-hierarchy choices: the hierarchy the schedule selected, then the
/// two-level alternatives the export can lower. `outer` and `inner` are the
/// resolved owner names, so a single-owner entry is completed with the
/// machine's innermost compute scope exactly as the concrete export does.
ArrayAttr makeOwnerChoices(MLIRContext *context, StringRef outer,
                           StringRef inner) {
  std::string selected = (outer + "/" + inner).str();
  return makeStringChoices(context,
                           {selected, "worker/lane", "worker/vector_engine"});
}

/// Returns the spatial mapping target of the same name as `owner`.
///
/// `MappingTarget` and `Owner` are independent vocabularies with different
/// enumerator values, so this is a name match and not a numeric cast. Returns
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
// Resolved tile plan
//===----------------------------------------------------------------------===//

/// Everything the kernel body needs, resolved from the operation's shapes and
/// the selected schedule entry.
struct TilePlan {
  // Problem shape and element types.
  int64_t M = 0, N = 0, K = 0;
  Type inputElemType, accumulatorElemType, outputElemType;

  // Tiling.
  int64_t BM = 0, BN = 0, BK = 0;
  /// Instruction fragment, clamped to a divisor of the tile on each axis.
  int64_t fM = 0, fN = 0, fK = 0;
  /// The fragment triple as the schedule stated it, before clamping.
  SmallVector<int64_t, 3> declaredFragment;

  int64_t stages = 1;
  int64_t vectorWidth = 0;
  micro::LayoutKind layoutKind = micro::LayoutKind::row_major;
  micro::Owner outerOwner = micro::Owner::worker;
  micro::Owner innerOwner = micro::Owner::lane;
  micro::Owner fragmentOwner = micro::Owner::vector_engine;
  micro::MappingTarget outerMap = micro::MappingTarget::worker;
  micro::MappingTarget innerMap = micro::MappingTarget::lane;
  micro::MemorySpace srcSpace = micro::MemorySpace::dram;
  micro::MemorySpace stagingSpace = micro::MemorySpace::sram;
  micro::MemorySpace accumulatorSpace = micro::MemorySpace::acc;
};

/// Parses the schedule's memory path into its source and staging levels,
/// checking that it names at least two known spaces and stages through on-chip
/// memory. Shared by both exports so they accept the same memory paths.
LogicalResult parseMemoryPath(Operation *root, const ScheduleEntry &schedule,
                              micro::MemorySpace &src,
                              micro::MemorySpace &staging) {
  auto fail = [&](const Twine &message) {
    root->emitError() << message;
    return failure();
  };

  SmallVector<StringRef, 4> spaces;
  StringRef(schedule.memory_path).split(spaces, ':');
  if (spaces.size() < 2)
    return fail("schedule memory_path '" + schedule.memory_path +
                "' must join at least two memory spaces with ':'");
  for (StringRef space : spaces)
    if (!micro::symbolizeMemorySpace(space))
      return fail("schedule memory_path '" + schedule.memory_path +
                  "' names unknown memory space '" + space + "'");

  src = *micro::symbolizeMemorySpace(spaces[0]);
  staging = *micro::symbolizeMemorySpace(spaces[1]);
  if (staging == micro::MemorySpace::dram)
    return fail("schedule memory_path '" + schedule.memory_path +
                "' stages tiles in dram; the staged level must be on-chip");
  return success();
}

/// Parses the schedule's owner hierarchy into at most two owner levels and
/// their spatial mapping targets. A single owner names the tiled axis; the
/// other tiled axis keeps the machine's innermost compute scope, which is what
/// `lane` is for the AVX2 validation backend.
LogicalResult parseOwnerHierarchy(Operation *root,
                                  const ScheduleEntry &schedule,
                                  micro::Owner &outer, micro::Owner &inner,
                                  micro::MappingTarget &outerMap,
                                  micro::MappingTarget &innerMap) {
  auto fail = [&](const Twine &message) {
    root->emitError() << message;
    return failure();
  };

  // Splitting an empty string yields one empty part, so the empty case is
  // checked before splitting.
  if (schedule.owner_mapping.empty())
    return fail("schedule owner_mapping must name at least one owner");
  SmallVector<StringRef, 4> owners;
  StringRef(schedule.owner_mapping).split(owners, '/');
  if (owners.size() > 2)
    return fail("schedule owner_mapping '" + schedule.owner_mapping +
                "' names more than two owner levels; the export emits one "
                "spatial loop per tiled axis");
  for (StringRef owner : owners)
    if (!micro::symbolizeOwner(owner))
      return fail("schedule owner_mapping '" + schedule.owner_mapping +
                  "' names unknown owner '" + owner + "'");

  outer = *micro::symbolizeOwner(owners[0]);
  inner = *micro::symbolizeOwner(owners.size() > 1 ? owners[1] : "lane");

  std::optional<micro::MappingTarget> outerTarget = spatialTargetFor(outer);
  std::optional<micro::MappingTarget> innerTarget = spatialTargetFor(inner);
  if (!outerTarget)
    return fail("schedule owner_mapping owner '" +
                micro::stringifyOwner(outer) +
                "' has no spatial axis to map onto");
  if (!innerTarget)
    return fail("schedule owner_mapping owner '" +
                micro::stringifyOwner(inner) +
                "' has no spatial axis to map onto");
  outerMap = *outerTarget;
  innerMap = *innerTarget;
  return success();
}

/// Resolves the schedule entry against the operation's shapes, reporting every
/// problem on `root` so the diagnostic points at the operation being lowered.
LogicalResult resolvePlan(Operation *root, const ScheduleEntry &schedule,
                          int64_t M, int64_t N, int64_t K, TilePlan &plan) {
  auto fail = [&](const Twine &message) {
    root->emitError() << message;
    return failure();
  };

  // --- memory path -------------------------------------------------------
  if (failed(parseMemoryPath(root, schedule, plan.srcSpace, plan.stagingSpace)))
    return failure();

  std::optional<micro::MemorySpace> accumulator =
      micro::symbolizeMemorySpace(schedule.accumulator_space);
  if (!accumulator)
    return fail("schedule accumulator_space '" + schedule.accumulator_space +
                "' is not a known memory space");
  if (*accumulator == micro::MemorySpace::dram)
    return fail("schedule accumulator_space must not be dram");
  plan.accumulatorSpace = *accumulator;

  // --- layout ------------------------------------------------------------
  std::optional<micro::LayoutKind> layout =
      micro::symbolizeLayoutKind(schedule.tile_layout);
  if (!layout)
    return fail("schedule tile_layout '" + schedule.tile_layout +
                "' is not a known layout kind");
  plan.layoutKind = *layout;

  // --- owner hierarchy ---------------------------------------------------
  if (failed(parseOwnerHierarchy(root, schedule, plan.outerOwner,
                                 plan.innerOwner, plan.outerMap,
                                 plan.innerMap)))
    return failure();

  std::optional<micro::Owner> fragment =
      micro::symbolizeOwner(schedule.fragment_owner);
  if (!fragment)
    return fail("schedule fragment_owner '" + schedule.fragment_owner +
                "' is not a known owner");
  plan.fragmentOwner = *fragment;

  // --- instruction fragment ---------------------------------------------
  FailureOr<SmallVector<int64_t, 3>> declared =
      parseShapeTriple(schedule.fragment_shape);
  if (failed(declared))
    return fail("schedule fragment_shape '" + schedule.fragment_shape +
                "' must be a positive MxNxK triple");
  plan.declaredFragment = *declared;

  // --- tiling ------------------------------------------------------------
  auto resolveTile = [&](const char *axis, int64_t extent, int64_t requested,
                         int64_t &out) -> LogicalResult {
    if (requested <= 0)
      return fail(Twine("schedule has no ") + axis + " tile size");
    out = std::min(requested, extent);
    if (extent % out != 0)
      return fail(
          Twine(axis) + " tile " + Twine(out) + " does not divide " + axis +
          " = " + Twine(extent) +
          (schedule.enable_tile_masks
               ? "; the export does not emit tile masks, so the schedule's "
               : "; tile masks are disabled for this schedule, so its ") +
          axis + " tile must divide the problem");
    return success();
  };

  plan.M = M;
  plan.N = N;
  plan.K = K;
  if (failed(resolveTile("BM", M, schedule.BM, plan.BM)) ||
      failed(resolveTile("BN", N, schedule.BN, plan.BN)) ||
      failed(resolveTile("BK", K, schedule.BK, plan.BK)))
    return failure();

  plan.fM = largestDivisorAtMost(plan.BM, std::min((*declared)[0], plan.BM));
  plan.fN = largestDivisorAtMost(plan.BN, std::min((*declared)[1], plan.BN));
  plan.fK = largestDivisorAtMost(plan.BK, std::min((*declared)[2], plan.BK));

  plan.stages = std::max<int64_t>(1, schedule.pipeline_stages);
  plan.vectorWidth = std::max<int64_t>(0, schedule.vector_width);
  return success();
}

//===----------------------------------------------------------------------===//
// Kernel emission
//===----------------------------------------------------------------------===//

/// Emits one concrete `micro.kernel` for `root` into `module`.
///
/// The body is a two-level spatial nest over the output tile, a K loop holding
/// the staged copies and the MMAs, and the elementwise epilogue that writes the
/// result back to external memory.
LogicalResult buildKernel(ModuleOp module, Operation *root,
                          const TilePlan &plan, const ScheduleEntry &schedule,
                          StringRef target, int64_t mBucket,
                          SymbolTable &symbols) {
  auto fused = dyn_cast<llk::FusedSwiGLUOp>(root);
  StringRef workload = fused ? "fused_swiglu" : "matmul";
  Location loc = root->getLoc();
  MLIRContext *ctx = module.getContext();

  // --- symbol name -------------------------------------------------------
  std::string base = (workload + "_M" + Twine(plan.M) + "_N" + Twine(plan.N) +
                      "_K" + Twine(plan.K))
                         .str();
  std::string symName = base;
  for (unsigned suffix = 1; symbols.lookup(symName); ++suffix)
    symName = base + "_" + Twine(suffix).str();

  OpBuilder builder(ctx);
  builder.setInsertionPointToEnd(module.getBody());

  // --- the kernel's explicit contract -------------------------------------
  // The kernel reads its operands from outside and produces the output tensor.
  // Naming both is what keeps an internal `tensor.empty` from being mistaken
  // for a caller's buffer, and what gives the write-back somewhere to land: the
  // output is threaded through the spatial nest and yielded.
  llvm::SmallVector<Type, 3> inputTypes{
      RankedTensorType::get({plan.M, plan.K}, plan.inputElemType)};
  for (size_t arm = 0; arm < (fused ? 2u : 1u); ++arm)
    inputTypes.push_back(
        RankedTensorType::get({plan.K, plan.N}, plan.inputElemType));
  llvm::SmallVector<Type, 1> resultTypes{
      RankedTensorType::get({plan.M, plan.N}, plan.outputElemType)};

  auto kernel = micro::KernelOp::create(
      builder, loc, symName,
      TypeAttr::get(FunctionType::get(ctx, inputTypes, resultTypes)),
      StringAttr::get(ctx, workload), StringAttr::get(ctx, target),
      /*candidate=*/StringAttr(),
      IntegerAttr::get(IntegerType::get(ctx, 64), mBucket));

  // Schedule intent that the concrete ops cannot carry themselves. The tile
  // types and op attributes already hold shape, dtype, layout, memory space,
  // and owner.
  kernel->setAttr("memory_path", StringAttr::get(ctx, schedule.memory_path));
  kernel->setAttr(
      "owner_mapping",
      StringAttr::get(ctx, (Twine(micro::stringifyOwner(plan.outerOwner)) +
                            "/" + micro::stringifyOwner(plan.innerOwner))
                               .str()));
  kernel->setAttr("tile_layout", StringAttr::get(ctx, schedule.tile_layout));
  kernel->setAttr("mma_shape",
                  DenseI64ArrayAttr::get(ctx, plan.declaredFragment));
  kernel->setAttr("fragment_shape",
                  DenseI64ArrayAttr::get(
                      ctx, SmallVector<int64_t, 3>{plan.fM, plan.fN, plan.fK}));
  // Non-dividing tiles are rejected instead of masked, so there is no tail.
  kernel->setAttr("tail_policy", StringAttr::get(ctx, "none"));

  // Original workload dimensions before tiling, as generic provenance. A
  // legality rule that needs the whole M/N/K (tail divisibility) reads this
  // rather than reconstructing the workload from an instruction-fragment MMA --
  // which is what lets a non-MMA, or multiple-MMA, kernel still be checked.
  auto dtypeName = [](Type elementType) -> std::string {
    if (std::optional<micro::DType> dtype =
            micro::dtypeOfElementType(elementType))
      return micro::stringifyDType(*dtype).str();
    return "bf16";
  };
  llvm::SmallVector<NamedAttribute, 8> provenance;
  auto addI64 = [&](StringRef name, int64_t value) {
    provenance.push_back(
        NamedAttribute(StringAttr::get(ctx, name),
                       IntegerAttr::get(IntegerType::get(ctx, 64), value)));
  };
  auto addStr = [&](StringRef name, const std::string &value) {
    provenance.push_back(NamedAttribute(StringAttr::get(ctx, name),
                                        StringAttr::get(ctx, value)));
  };
  addI64("M", plan.M);
  addI64("N", plan.N);
  addI64("K", plan.K);
  addStr("input_dtype", dtypeName(plan.inputElemType));
  addStr("weight_dtype", dtypeName(plan.inputElemType));
  addStr("accumulator_dtype", dtypeName(plan.accumulatorElemType));
  addStr("output_dtype", dtypeName(plan.outputElemType));
  kernel->setAttr("original_workload", DictionaryAttr::get(ctx, provenance));

  Block *kernelBody =
      startRegionBody(builder, kernel.getBody(), loc, inputTypes);

  // --- tile types --------------------------------------------------------
  micro::LayoutAttr layout =
      makeLayoutAttr(ctx, plan.layoutKind, plan.vectorWidth);
  micro::OwnerAttr workerOwner = micro::OwnerAttr::get(ctx, plan.outerOwner);
  micro::OwnerAttr fragmentOwner =
      micro::OwnerAttr::get(ctx, plan.fragmentOwner);
  micro::MemorySpaceAttr srcSpace =
      micro::MemorySpaceAttr::get(ctx, plan.srcSpace);
  micro::MemorySpaceAttr stageSpace =
      micro::MemorySpaceAttr::get(ctx, plan.stagingSpace);
  micro::MemorySpaceAttr accSpace =
      micro::MemorySpaceAttr::get(ctx, plan.accumulatorSpace);

  auto tile = [&](ArrayRef<int64_t> shape, Type element,
                  micro::LayoutAttr tileLayout, micro::MemorySpaceAttr memory,
                  micro::OwnerAttr owner) {
    return micro::TileType::get(ctx, shape, element, tileLayout, memory, owner);
  };

  // --- external tensors --------------------------------------------------
  // The kernel is IsolatedFromAbove, so the operands it reads come from its
  // own entry block arguments -- the ones the signature declared -- rather
  // than from a placeholder that only convention says is external.
  Value lhsTensor = kernelBody->getArgument(0);
  SmallVector<Value, 2> rhsTensors;
  for (size_t arm = 0; arm < (fused ? 2u : 1u); ++arm)
    rhsTensors.push_back(kernelBody->getArgument(1 + arm));

  // --- the output the kernel writes back ----------------------------------
  // The kernel writes its output tile by tile from inside the spatial nest, so
  // the output is a value the loops carry: an SSA destination is only updated
  // if the updated destination is what the enclosing loop hands on. Without
  // this the write-back would have nowhere to land.
  // The output is accumulated in the staging level on the way out: a local
  // allocation may not target dram, because a dram operand means a buffer that
  // already belongs to a caller. What makes this the kernel's *external* output
  // is the declared result, not the storage it was built in -- where it finally
  // lands belongs to the ABI.
  micro::MemorySpaceAttr outputSpace =
      micro::MemorySpaceAttr::get(ctx, micro::MemorySpace::sram);
  Type outputTileType = tile({plan.M, plan.N}, plan.outputElemType,
                             /*tileLayout=*/micro::LayoutAttr(), outputSpace,
                             /*owner=*/micro::OwnerAttr());
  Value output =
      micro::TileAllocOp::create(builder, loc, outputTileType).getResult();

  // --- spatial tiling ----------------------------------------------------
  auto openSpatialLoop = [&](int64_t extent, int64_t step,
                             micro::MappingTarget target,
                             ValueRange carried) -> micro::SpatialForOp {
    Value lower = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
    Value upper =
        arith::ConstantIndexOp::create(builder, loc, extent).getResult();
    Value by = arith::ConstantIndexOp::create(builder, loc, step).getResult();
    auto loop = micro::SpatialForOp::create(
        builder, loc, carried.getTypes(), lower, upper, by,
        micro::MappingTargetAttr::get(ctx, target), carried);
    llvm::SmallVector<Type> bodyTypes{IndexType::get(ctx)};
    bodyTypes.append(carried.getTypes().begin(), carried.getTypes().end());
    startRegionBody(builder, loop.getBody(), loc, bodyTypes);
    return loop;
  };

  micro::SpatialForOp bmLoop =
      openSpatialLoop(plan.M, plan.BM, plan.outerMap, ValueRange{output});
  BlockArgument bm = bmLoop.getBody().front().getArgument(0);
  Value bmCarried = bmLoop.getBody().front().getArgument(1);

  micro::SpatialForOp bnLoop =
      openSpatialLoop(plan.N, plan.BN, plan.innerMap, ValueRange{bmCarried});
  BlockArgument bn = bnLoop.getBody().front().getArgument(0);
  Value bnCarried = bnLoop.getBody().front().getArgument(1);

  // --- accumulators ------------------------------------------------------
  SmallVector<int64_t, 2> accExtent{plan.BM, plan.BN};
  Type accTileType =
      tile(accExtent, plan.accumulatorElemType,
           /*tileLayout=*/micro::LayoutAttr(), accSpace, workerOwner);
  SmallVector<Value, 2> accumulators;
  for (size_t arm = 0; arm < rhsTensors.size(); ++arm)
    accumulators.push_back(
        micro::TileAllocOp::create(builder, loc, accTileType).getResult());

  // --- K loop ------------------------------------------------------------
  Value kLower = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  Value kUpper =
      arith::ConstantIndexOp::create(builder, loc, plan.K).getResult();
  Value kStep =
      arith::ConstantIndexOp::create(builder, loc, plan.BK).getResult();
  // The loop carries the accumulators. What an iteration computes is what the
  // next iteration accumulates into, and what the last one left is the kernel's
  // result. Without the carried values the MMA inside the loop would compute
  // dead results while the epilogue read an allocation nothing ever wrote.
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
  // reads as software pipelining. With one stage the two spellings agree. It
  // carries the accumulators out of its own body, because the MMA it overlaps
  // runs inside it and the loop above still has to hand the result on.
  auto pipeline = micro::PipelineOp::create(builder, loc, carriedTypes,
                                            static_cast<uint64_t>(plan.stages));
  startRegionBody(builder, pipeline.getBody(), loc);

  // --- logical views and staged copies -----------------------------------
  // A view names the memory it reads from but allocates nothing; the copy that
  // follows is what materializes it in the staging level.
  SmallVector<int64_t, 2> lhsTileExtent{plan.BM, plan.BK};
  Type lhsViewType = tile(lhsTileExtent, plan.inputElemType, layout, srcSpace,
                          /*owner=*/micro::OwnerAttr());
  Value lhsView = micro::TileViewOp::create(
      builder, loc, lhsViewType, lhsTensor, ValueRange{bm, bk},
      DenseI64ArrayAttr::get(ctx, lhsTileExtent), layout);

  SmallVector<Value, 2> tokens;
  Type lhsStagedType =
      tile(lhsTileExtent, plan.inputElemType, layout, stageSpace, workerOwner);
  auto lhsCopy = micro::TileAsyncCopyOp::create(
      builder, loc, lhsStagedType, micro::AsyncTokenType::get(ctx), lhsView,
      stageSpace, workerOwner);
  Value lhsStaged = lhsCopy.getResult();
  tokens.push_back(lhsCopy.getToken());

  SmallVector<int64_t, 2> rhsTileExtent{plan.BK, plan.BN};
  SmallVector<Value, 2> rhsStaged;
  for (Value source : rhsTensors) {
    Type viewType = tile(rhsTileExtent, plan.inputElemType, layout, srcSpace,
                         /*owner=*/micro::OwnerAttr());
    Value view = micro::TileViewOp::create(
        builder, loc, viewType, source, ValueRange{bk, bn},
        DenseI64ArrayAttr::get(ctx, rhsTileExtent), layout);
    Type stagedType = tile(rhsTileExtent, plan.inputElemType, layout,
                           stageSpace, workerOwner);
    auto copy = micro::TileAsyncCopyOp::create(builder, loc, stagedType,
                                               micro::AsyncTokenType::get(ctx),
                                               view, stageSpace, workerOwner);
    rhsStaged.push_back(copy.getResult());
    tokens.push_back(copy.getToken());
  }

  micro::WaitOp::create(builder, loc, tokens);

  // --- instruction fragments ---------------------------------------------
  // Logical, zero-cost annotations: a partition whose fragment covers its
  // whole parent carries no information and is not emitted. The MMA consumes
  // the staged tiles, not these, because the fragment shape has to stay
  // consistent with the MMA's `shape`.
  SmallVector<int64_t, 2> lhsFragmentExtent{plan.fM, plan.fK};
  SmallVector<int64_t, 2> rhsFragmentExtent{plan.fK, plan.fN};
  if (plan.fM != plan.BM || plan.fK != plan.BK)
    micro::TilePartitionOp::create(
        builder, loc,
        tile(lhsFragmentExtent, plan.inputElemType, layout, stageSpace,
             fragmentOwner),
        lhsStaged, DenseI64ArrayAttr::get(ctx, lhsFragmentExtent),
        fragmentOwner, /*tail=*/BoolAttr());
  if (plan.fK != plan.BK || plan.fN != plan.BN)
    for (Value staged : rhsStaged)
      micro::TilePartitionOp::create(
          builder, loc,
          tile(rhsFragmentExtent, plan.inputElemType, layout, stageSpace,
               fragmentOwner),
          staged, DenseI64ArrayAttr::get(ctx, rhsFragmentExtent), fragmentOwner,
          /*tail=*/BoolAttr());

  // --- MMA ---------------------------------------------------------------
  // Each arm accumulates into the value the loop carried in, not into the
  // original allocation: the carried value is this iteration's accumulator.
  DenseI64ArrayAttr mmaShape = DenseI64ArrayAttr::get(
      ctx, SmallVector<int64_t, 3>{plan.BM, plan.BN, plan.BK});
  llvm::SmallVector<Value> nextAccumulators;
  for (size_t arm = 0; arm < rhsStaged.size(); ++arm)
    nextAccumulators.push_back(
        micro::MmaOp::create(
            builder, loc, accTileType, lhsStaged, rhsStaged[arm], carried[arm],
            mmaShape,
            micro::DTypeAttr::get(
                ctx, *micro::dtypeOfElementType(plan.inputElemType)),
            micro::DTypeAttr::get(
                ctx, *micro::dtypeOfElementType(plan.accumulatorElemType)),
            /*engine=*/StringAttr())
            .getResult());

  // The pipeline hands the computed values out of its body, and the loop hands
  // them to the next iteration. Both steps are what make the carry real rather
  // than implied: the MMA runs inside the pipeline, so without the first the
  // value would not be visible to the loop's own terminator at all.
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

  // Everything after the loop reads what the loop left, so the epilogue and the
  // write-back see the accumulated result rather than the initial allocation.
  for (size_t arm = 0; arm < accumulators.size(); ++arm)
    accumulators[arm] = kLoop.getResults()[arm];

  // The epilogue belongs to the output tile, not to one K iteration.
  builder.setInsertionPoint(kLoop->getNextNode());

  // --- elementwise epilogue ----------------------------------------------
  Value epilogue = accumulators[0];
  if (fused) {
    StringRef mathMode = llk::stringifyMathMode(fused.getMathMode());
    Value gate = micro::VectorOp::create(builder, loc, accTileType, "silu",
                                         ValueRange{accumulators[0]},
                                         StringAttr::get(ctx, mathMode));
    epilogue = micro::VectorOp::create(builder, loc, accTileType, "mul",
                                       ValueRange{gate, accumulators[1]},
                                       /*math_mode=*/StringAttr());
  }

  // When the accumulator already holds the output element type the conversion
  // would be an identity, so it is not emitted.
  if (plan.accumulatorElemType != plan.outputElemType) {
    Type convertedType =
        tile(accExtent, plan.outputElemType,
             /*tileLayout=*/micro::LayoutAttr(), accSpace, workerOwner);
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

  // The inner loop hands the updated output out, the outer loop carries it, and
  // the kernel yields what the nest left behind. Each step is what makes the
  // write-back a value rather than a side effect nothing can observe.
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
  symbols.insert(kernel);
  return success();
}

/// The workload, problem shape, and element types of one supported LLK root
/// operation, resolved and validated. Shared by both exports so they accept
/// exactly the same operations and agree on the schedule.
struct RootInfo {
  std::string workload;
  int64_t M = 0, N = 0, K = 0;
  Type inputElemType, accumulatorElemType, outputElemType;
  int64_t mBucket = 0;
};

/// Checks that `value` is a static rank-2 tensor of a Micro tile dtype.
LogicalResult checkTensor(Operation *root, Value value, const char *what) {
  auto shaped = dyn_cast<ShapedType>(value.getType());
  if (!shaped || !shaped.hasRank() || shaped.getRank() != 2)
    return root->emitError() << what << " must be a rank-2 tensor";
  if (!shaped.hasStaticShape())
    return root->emitError()
           << what
           << " must have a static shape; dynamic shapes are not supported "
              "by the Micro-IR export";
  if (!micro::dtypeOfElementType(shaped.getElementType()))
    return root->emitError()
           << what << " element type is not a Micro tile dtype "
           << "(expected f32, f16, bf16, i32, or i8)";
  return success();
}

/// Validates `root`'s shapes, dtypes, and accumulator type and fills `info`.
LogicalResult inferRootInfo(Operation *root, RootInfo &info) {
  auto fused = dyn_cast<llk::FusedSwiGLUOp>(root);
  auto matmul = dyn_cast<llk::MatmulOp>(root);
  info.workload = fused ? "fused_swiglu" : "matmul";

  Value lhs = fused ? fused.getX() : matmul.getA();
  Value rhs = fused ? fused.getWg() : matmul.getB();
  Value out = fused ? fused.getResult() : matmul.getResult();
  if (failed(checkTensor(root, lhs, "left-hand side")) ||
      failed(checkTensor(root, rhs, "right-hand side")) ||
      failed(checkTensor(root, out, "result")))
    return failure();

  auto lhsShaped = cast<ShapedType>(lhs.getType());
  auto rhsShaped = cast<ShapedType>(rhs.getType());
  auto outShaped = cast<ShapedType>(out.getType());
  info.M = lhsShaped.getDimSize(0);
  info.K = lhsShaped.getDimSize(1);
  info.N = rhsShaped.getDimSize(1);

  if (rhsShaped.getDimSize(0) != info.K)
    return root->emitError()
           << "contraction dimension mismatch: the left-hand side contracts "
              "over "
           << info.K << " but the right-hand side has "
           << rhsShaped.getDimSize(0) << " rows";

  if (outShaped.getDimSize(0) != info.M || outShaped.getDimSize(1) != info.N)
    return root->emitError()
           << "result shape does not match the contraction: expected M = "
           << info.M << " and N = " << info.N;

  if (fused) {
    if (failed(checkTensor(root, fused.getWu(), "up-projection weight")))
      return failure();
    auto wuShaped = cast<ShapedType>(fused.getWu().getType());
    if (wuShaped.getDimSize(0) != info.K || wuShaped.getDimSize(1) != info.N)
      return root->emitError()
             << "the up-projection weight must be shaped [K, N] = [" << info.K
             << ", " << info.N << "]";
    if (fused.getActivation() != llk::Activation::silu)
      return root->emitError()
             << "unsupported SwiGLU activation; the Micro-IR export emits "
                "silu only";
  }

  auto accumulatorTypeAttr = dyn_cast<TypeAttr>(
      fused ? fused.getAccumulatorType() : matmul.getAccumulatorType());
  if (!accumulatorTypeAttr ||
      !micro::dtypeOfElementType(accumulatorTypeAttr.getValue()))
    return root->emitError()
           << "accumulator_type must be a type attribute naming a Micro tile "
              "dtype";

  info.inputElemType = lhsShaped.getElementType();
  info.accumulatorElemType = accumulatorTypeAttr.getValue();
  info.outputElemType = outShaped.getElementType();
  info.mBucket = llk::classifyM(info.M);
  return success();
}

/// Loads the schedule entries matching `info` and selects one, warning when the
/// database has nothing to offer. Shared so the concrete and search-space
/// exports cannot disagree about which schedule an operation runs under.
ScheduleEntry selectSchedule(Operation *root, const RootInfo &info,
                             StringRef dbPath) {
  std::vector<ScheduleEntry> matches =
      llk::loadScheduleDB(dbPath, info.mBucket, info.N, info.K, info.workload);
  if (matches.empty())
    root->emitWarning() << "no schedule entry for " << info.workload
                        << " M_bucket=" << info.mBucket << " in " << dbPath
                        << "; using the built-in conservative schedule";
  return llk::selectBestSchedule(matches, info.N, info.K);
}

/// Validates the root operation's shapes and dtypes, resolves its schedule, and
/// emits its kernel.
LogicalResult lowerRootOp(ModuleOp module, Operation *root, StringRef dbPath,
                          StringRef target, SymbolTable &symbols) {
  RootInfo info;
  if (failed(inferRootInfo(root, info)))
    return failure();
  ScheduleEntry schedule = selectSchedule(root, info, dbPath);

  TilePlan plan;
  plan.inputElemType = info.inputElemType;
  plan.accumulatorElemType = info.accumulatorElemType;
  plan.outputElemType = info.outputElemType;
  if (failed(resolvePlan(root, schedule, info.M, info.N, info.K, plan)))
    return failure();

  return buildKernel(module, root, plan, schedule, target, info.mBucket,
                     symbols);
}

//===----------------------------------------------------------------------===//
// Search-space emission
//===----------------------------------------------------------------------===//

/// Returns true when the module already holds a search space named `name`.
/// Unlike `micro.kernel`, `micro.search_space` is not a symbol, so uniqueness
/// is maintained here rather than by a SymbolTable.
bool searchSpaceNameTaken(ModuleOp module, StringRef name) {
  for (micro::SearchSpaceOp space : module.getOps<micro::SearchSpaceOp>())
    if (space.getSymName() == name)
      return true;
  return false;
}

/// Emits one `micro.search_space` for `root`, holding the legal choices around
/// the schedule the database selected.
///
/// Numeric dimensions come from the tuner grid; symbolic dimensions are
/// schedule-anchored, with the scheduled value printed first so the selected
/// schedule is the space's first candidate.
LogicalResult buildSearchSpace(ModuleOp module, Operation *root,
                               const RootInfo &info,
                               const ScheduleEntry &schedule) {
  // The symbolic choices copy values out of the schedule, so each has to be a
  // value the dialect accepts before it can become a micro.param choice. The
  // checks are the concrete export's, so a search space is only ever emitted
  // for a schedule the concrete export could also lower.
  micro::MemorySpace srcSpace = micro::MemorySpace::dram;
  micro::MemorySpace stagingSpace = micro::MemorySpace::sram;
  if (failed(parseMemoryPath(root, schedule, srcSpace, stagingSpace)))
    return failure();

  micro::Owner outerOwner = micro::Owner::worker;
  micro::Owner innerOwner = micro::Owner::lane;
  micro::MappingTarget outerMap = micro::MappingTarget::worker;
  micro::MappingTarget innerMap = micro::MappingTarget::lane;
  if (failed(parseOwnerHierarchy(root, schedule, outerOwner, innerOwner,
                                 outerMap, innerMap)))
    return failure();

  if (!micro::symbolizeLayoutKind(schedule.tile_layout))
    return root->emitError() << "schedule tile_layout '" << schedule.tile_layout
                             << "' is not a known layout kind";
  if (failed(parseShapeTriple(schedule.fragment_shape)))
    return root->emitError()
           << "schedule fragment_shape '" << schedule.fragment_shape
           << "' must be a positive MxNxK triple";

  MLIRContext *context = module.getContext();
  Location loc = root->getLoc();

  // --- symbol name ---
  std::string base = (info.workload + "_M" + Twine(info.M) + "_N" +
                      Twine(info.N) + "_K" + Twine(info.K))
                         .str();
  std::string symName = base;
  for (unsigned suffix = 1; searchSpaceNameTaken(module, symName); ++suffix)
    symName = base + "_" + Twine(suffix).str();

  OpBuilder builder(context);
  builder.setInsertionPointToEnd(module.getBody());
  auto space = micro::SearchSpaceOp::create(
      builder, loc, symName, StringAttr::get(context, info.workload));
  startRegionBody(builder, space.getBody(), loc);

  auto addParam = [&](StringRef name, StringRef kind, ArrayAttr choices) {
    // No role: the exported space declares one parameter per axis, so each
    // governs its axis as a whole. A space that binds several layouts for
    // different ports would name them.
    micro::ParamOp::create(builder, loc, name, kind, choices,
                           /*role=*/mlir::StringAttr());
  };
  auto addConstraint = [&](StringRef kind, ArrayRef<StringRef> params) {
    SmallVector<Attribute, 4> names;
    for (StringRef param : params)
      names.push_back(StringAttr::get(context, param));
    micro::ConstraintOp::create(builder, loc, kind,
                                ArrayAttr::get(context, names));
  };

  // --- numeric parameters ---
  addParam("BM", "integer", makeBmChoices(context, schedule.BM, info.mBucket));
  addParam("BN", "integer",
           makeNumericChoices(context, kSearchBN, schedule.BN));
  addParam("BK", "integer",
           makeNumericChoices(context, kSearchBK, schedule.BK));
  addParam("VM", "integer",
           makeNumericChoices(context, kSearchVM, schedule.VM));
  addParam("VN", "integer",
           makeNumericChoices(context, kSearchVN, schedule.VN));
  addParam(
      "vector_width", "integer",
      makeNumericChoices(context, kSearchVectorWidth, schedule.vector_width));
  addParam("num_threads", "integer",
           makeNumericChoices(context, kSearchThreads, schedule.num_threads));
  addParam("grain_size", "integer",
           makeNumericChoices(context, kSearchGrain, schedule.grain_size));
  addParam(
      "pipeline_stages", "integer",
      makeNumericChoices(context, kSearchStages, schedule.pipeline_stages));
  // Prefetch distance 0 means "no prefetch" -- a schedule decision rather than
  // a search choice -- so it is not part of the domain.
  addParam(
      "prefetch_distance", "integer",
      makeNumericChoices(context, kSearchPrefetch, schedule.prefetch_distance));

  // --- symbolic parameters ---
  addParam("tile_layout", "layout",
           makeStringChoices(context,
                             {schedule.tile_layout, "row_major", "blocked"}));
  addParam("memory_path", "memory_path",
           makeStringChoices(context, {schedule.memory_path, "dram:sram:acc",
                                       "dram:l2:sram:acc"}));
  addParam("owner_mapping", "owner_mapping",
           makeOwnerChoices(context, micro::stringifyOwner(outerOwner),
                            micro::stringifyOwner(innerOwner)));
  addParam("fragment_shape", "fragment_shape",
           makeStringChoices(context,
                             {schedule.fragment_shape, "16x16x32", "8x8x32"}));
  addParam("tail_policy", "tail_policy", makeStringChoices(context, {"mask"}));

  // --- legality records ---
  // Machine-independent legality only: capacity and compatibility are named
  // here and evaluated against the MachineModel by the tuner.
  addConstraint("sram_capacity", {"BM", "BN", "BK"});
  addConstraint("acc_capacity", {"BM", "BN"});
  addConstraint("mma_compatible", {"BM", "BN", "BK"});
  addConstraint("tile_hierarchy_compatible", {"owner_mapping"});
  addConstraint("layout_supported", {"tile_layout"});
  addConstraint("owner_supported", {"owner_mapping"});
  addConstraint("fragment_compatible", {"fragment_shape", "BM", "BN", "BK"});
  addConstraint("vector_width_supported", {"vector_width"});
  addConstraint("mapping_extent", {"num_threads", "BM", "BN"});
  addConstraint("pipeline_live_tiles",
                {"pipeline_stages", "prefetch_distance", "BM", "BN", "BK"});
  addConstraint("tail_supported", {"BM", "BN", "BK"});

  // --- objective ---
  micro::ObjectiveOp::create(
      builder, loc, "minimize", "latency_cycles",
      ArrayAttr::get(context, {StringAttr::get(context, "matrix_utilization"),
                               StringAttr::get(context, "dram_bytes")}));

  return success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct LLKToMicroPass
    : public PassWrapper<LLKToMicroPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLKToMicroPass)

  LLKToMicroPass() = default;
  LLKToMicroPass(const LLKToMicroPass &other) : PassWrapper(other) {}

  Option<std::string> scheduleDb{
      *this, "schedule-db",
      llvm::cl::desc(
          "Schedule database the export reads its tile choices from. A path "
          "that cannot be read falls back to the built-in schedule."),
      llvm::cl::init("schedules/schedule_db.json")};
  Option<std::string> target{
      *this, "target",
      llvm::cl::desc("Target recorded on every generated micro.kernel"),
      llvm::cl::init("x86-avx2-cpu")};

  StringRef getArgument() const override { return "llk-to-micro"; }
  StringRef getDescription() const override {
    return "Lower LLK root operations to concrete tile-centric micro.kernel IR";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // This pass builds its output programmatically rather than parsing it, so
    // every dialect it emits has to be loaded explicitly: a context loads a
    // dialect on demand, and an export from LLK starts with no micro, tensor,
    // or arith operations in the input to trigger that.
    MLIRContext *context = module.getContext();
    if (!context->getOrLoadDialect<micro::MicroDialect>() ||
        !context->getOrLoadDialect<tensor::TensorDialect>() ||
        !context->getOrLoadDialect<arith::ArithDialect>()) {
      module.emitError()
          << "the micro, tensor, and arith dialects must be registered in "
             "this context";
      signalPassFailure();
      return;
    }

    SmallVector<Operation *> roots;
    module.walk([&](Operation *op) {
      if (isa<llk::FusedSwiGLUOp, llk::MatmulOp>(op))
        roots.push_back(op);
    });
    if (roots.empty())
      return;

    SymbolTable symbols(module);
    for (Operation *root : roots) {
      if (failed(lowerRootOp(module, root, scheduleDb.getValue(),
                             target.getValue(), symbols))) {
        signalPassFailure();
        return;
      }
    }
  }
};

/// Exports the legal choices around the selected schedule. The pass runs on
/// the same root operations the concrete export lowers, and reads the schedule
/// through the same loader, so the search space always contains the schedule
/// the concrete kernel was built from.
struct LLKToMicroSearchSpacePass
    : public PassWrapper<LLKToMicroSearchSpacePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLKToMicroSearchSpacePass)

  LLKToMicroSearchSpacePass() = default;
  LLKToMicroSearchSpacePass(const LLKToMicroSearchSpacePass &other)
      : PassWrapper(other) {}

  Option<std::string> scheduleDb{
      *this, "schedule-db",
      llvm::cl::desc(
          "Schedule database whose selected entry anchors the search space. A "
          "path that cannot be read falls back to the built-in schedule."),
      llvm::cl::init("schedules/schedule_db.json")};

  StringRef getArgument() const override { return "llk-to-micro-search-space"; }
  StringRef getDescription() const override {
    return "Export tile-aware micro.search_space IR from the selected schedule";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // The search space is built programmatically rather than parsed, so the
    // dialect it emits has to be loaded explicitly.
    MLIRContext *context = module.getContext();
    if (!context->getOrLoadDialect<micro::MicroDialect>()) {
      module.emitError()
          << "the micro dialect must be registered in this context";
      signalPassFailure();
      return;
    }

    SmallVector<Operation *> roots;
    module.walk([&](Operation *op) {
      if (isa<llk::FusedSwiGLUOp, llk::MatmulOp>(op))
        roots.push_back(op);
    });
    if (roots.empty())
      return;

    for (Operation *root : roots) {
      RootInfo info;
      if (failed(inferRootInfo(root, info))) {
        signalPassFailure();
        return;
      }
      ScheduleEntry schedule =
          selectSchedule(root, info, scheduleDb.getValue());
      if (failed(buildSearchSpace(module, root, info, schedule))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

namespace mlir {
namespace llk {

std::unique_ptr<Pass> createLLKToMicroPass() {
  return std::make_unique<LLKToMicroPass>();
}

std::unique_ptr<Pass> createLLKToMicroSearchSpacePass() {
  return std::make_unique<LLKToMicroSearchSpacePass>();
}

} // namespace llk
} // namespace mlir
