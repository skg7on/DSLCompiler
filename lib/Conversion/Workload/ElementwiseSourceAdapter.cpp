//===- ElementwiseSourceAdapter.cpp - Bounded elementwise source family ---===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework B5: a structured, non-contraction source family behind the same
// adapter boundary the contraction examples use. Recognition is structural
// (`linalg.generic`), and admission is a small, explicit validator: static
// rank-two f32 tensors, two read inputs, one destination with no incoming
// read, one `arith.addf`/`arith.mulf` yielding its result, identity indexing
// maps and two parallel iterators.
//
// The emitted program is a full-tile MicroIR kernel built from existing
// primitives -- logical views, staged asynchronous copies, one wait and one
// vector operation. No new opcode is introduced and no target name is
// consulted: memory and layout are the target-independent defaults a full-tile
// program can state, and the schedule choices the family does not yet realize
// are reported as such rather than silently ignored.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/Workload/ElementwiseSourceAdapter.h"

#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Dialect/Micro/MicroEnums.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

// Micro attribute, type, and op declarations.
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

using namespace mlir;

namespace mlir::llk::workload {
namespace {

llvm::Error error(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

/// A rejection that names the feature which put the program outside this
/// increment's envelope. The feature is part of the diagnostic so a caller can
/// tell "this family does not support reductions yet" from "this is not this
/// family's source at all".
llvm::Error reject(llvm::StringRef feature, const llvm::Twine &detail) {
  return error("elementwise_source: " + feature + ": " + detail);
}

constexpr llvm::StringLiteral elementwiseKey = "linalg-elementwise-v1";

/// The supported shape of one `linalg.generic` root. Holding the validated
/// facts in one place keeps `describe`, `bindings` and `realize` from
/// disagreeing about what they accepted.
struct ElementwiseProgram {
  linalg::GenericOp generic;
  Operation *scalarOp = nullptr;
  bool multiply = false;
  int64_t d0 = 0;
  int64_t d1 = 0;
  Type elementType;
  SmallVector<Value, 2> inputs;
  Value destination;
  OpResult result;
};

/// The one supported operation is a floating add or multiply. The mnemonic is
/// a primitive key inside this adapter; the generic planner never sees it.
bool isSupportedScalar(Operation *op) {
  return isa<arith::AddFOp, arith::MulFOp>(op);
}

/// Validates `root` against this increment's envelope, returning the facts the
/// contract and the emitter both use. Each rejection names the unsupported
/// feature.
llvm::Expected<ElementwiseProgram> analyze(Operation *root) {
  if (!root)
    return error("an elementwise source root is required");
  auto generic = dyn_cast<linalg::GenericOp>(root);
  if (!generic)
    return reject("structure", "expected a linalg.generic root");

  // Numerical semantics are explicit IR facts. This increment implements
  // strict f32 only, so an approximation mode on the root is rejected rather
  // than dropped.
  if (root->hasAttr("math_mode") || root->hasAttr("llk.math_mode"))
    return reject("numerical",
                  "only strict f32 is supported; approximation modes are not");

  if (generic.getInputs().size() != 2)
    return reject("inputs", "exactly two input operands are supported");
  if (generic.getOutputs().size() != 1)
    return reject("results", "exactly one destination is supported");
  if (generic.getResults().size() != 1)
    return reject("results", "exactly one result is supported");

  // Iterator types first: a reduction is the structural feature that most
  // changes what the program means, so it is named before the maps or shapes
  // that a reduction would also alter.
  for (utils::IteratorType iterator : generic.getIteratorTypesArray())
    if (iterator != utils::IteratorType::parallel)
      return reject("reduction", "only two parallel iterators are supported");

  SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
  if (maps.size() != generic.getInputs().size() + generic.getOutputs().size())
    return reject("indexing", "one identity map per operand is required");
  for (AffineMap map : maps)
    if (!map.isIdentity())
      return reject("indexing", "identity indexing maps are required");

  SmallVector<Value, 3> shaped{generic.getInputs()[0], generic.getInputs()[1],
                               generic.getOutputs()[0]};
  int64_t d0 = 0;
  int64_t d1 = 0;
  Type elementType;
  for (size_t i = 0; i < shaped.size(); ++i) {
    auto type = dyn_cast<RankedTensorType>(shaped[i].getType());
    if (!type)
      return reject("shape", "tensor operands are required");
    if (!type.hasStaticShape())
      return reject("dynamic", "static shapes are required");
    if (type.getRank() != 2)
      return reject("rank", "rank-two tensors are required");
    if (!type.getElementType().isF32())
      return reject("dtype", "only f32 is supported");
    if (i == 0) {
      d0 = type.getShape()[0];
      d1 = type.getShape()[1];
      elementType = type.getElementType();
      // A zero extent is not "unset": it is an unsupported shape and must be
      // rejected here rather than mistaken for the next port's dimensions.
      if (d0 <= 0 || d1 <= 0)
        return reject("shape", "extents must be positive");
    } else if (type.getShape() != ArrayRef<int64_t>{d0, d1}) {
      return reject("shape", "all ports must have the same shape");
    }
  }
  auto resultType =
      dyn_cast<RankedTensorType>(generic.getResults()[0].getType());
  if (!resultType || !resultType.hasStaticShape() ||
      resultType.getShape() != ArrayRef<int64_t>{d0, d1} ||
      resultType.getElementType() != elementType)
    return reject("results", "the result must match the destination port");

  // The body is exactly the supported scalar operation and its yield. The
  // destination block argument must carry no incoming read: a local
  // `tensor.empty` says the destination is written, never read.
  Block &body = generic.getRegion().front();
  if (body.getNumArguments() != 3)
    return reject("body", "the body must take the two inputs and destination");
  if (!body.getArgument(2).use_empty())
    return reject("destination", "the destination operand must not be read");

  Operation *scalarOp = nullptr;
  for (Operation &op : body.without_terminator()) {
    if (scalarOp)
      return reject("body", "exactly one scalar computation is supported");
    scalarOp = &op;
  }
  if (!scalarOp || !isSupportedScalar(scalarOp))
    return reject("body", "expected exactly one arith.addf or arith.mulf");

  auto fastmath = scalarOp->getAttrOfType<arith::FastMathFlagsAttr>("fastmath");
  if (fastmath && fastmath.getValue() != arith::FastMathFlags::none)
    return reject("fastmath", "strict floating-point semantics are required");
  // MicroIR's vector primitive does not carry an explicit rounding mode.
  // Reject it rather than silently changing the source's numerical contract.
  if (scalarOp->hasAttr("roundingmode"))
    return reject("rounding", "explicit floating-point rounding modes are not "
                              "supported");

  auto yield = dyn_cast<linalg::YieldOp>(body.getTerminator());
  if (!yield || yield.getNumOperands() != 1 ||
      yield.getOperand(0) != scalarOp->getResult(0))
    return reject("body", "the scalar operation must yield its result");

  SmallVector<Value, 2> operands(scalarOp->getOperands());
  if (operands.size() != 2)
    return reject("body", "the scalar operation must take two operands");
  bool read[2] = {false, false};
  for (Value operand : operands) {
    auto argument = dyn_cast<BlockArgument>(operand);
    if (!argument || argument.getOwner() != &body ||
        argument.getArgNumber() > 1)
      return reject("body", "the body must read exactly the two input block "
                            "arguments");
    read[argument.getArgNumber()] = true;
  }
  if (!read[0] || !read[1])
    return reject("body",
                  "the body must read exactly the two input block arguments");

  ElementwiseProgram program;
  program.generic = generic;
  program.scalarOp = scalarOp;
  program.multiply = isa<arith::MulFOp>(scalarOp);
  program.d0 = d0;
  program.d1 = d1;
  program.elementType = elementType;
  program.inputs = {generic.getInputs()[0], generic.getInputs()[1]};
  program.destination = generic.getOutputs()[0];
  program.result = generic.getResults()[0];
  return program;
}

/// The symbol of the module-level operation that contains `nested`. The
/// kernel name is derived from it, so a source function yields a stable,
/// recognizable kernel symbol.
std::string enclosingSymbol(Operation *nested) {
  Operation *owner = nested;
  while (owner->getParentOp() && !isa<ModuleOp>(owner->getParentOp()))
    owner = owner->getParentOp();
  StringAttr symbol = SymbolTable::getSymbolName(owner);
  return symbol ? symbol.getValue().str() : std::string();
}

/// The parameter ownership this family declares. Source-owned parameters name
/// the axis they would tile; layout, memory path and owner mapping stay
/// mapping-owned because the source stage has not bound them. Declaring a
/// source parameter is not evidence of realizing it -- `realize` reports the
/// ones this increment cannot emit.
std::vector<ParameterBinding>
elementwiseSchema(const perf::SearchSpace &space) {
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
    else
      binding.role = "axis:" + param.name;
    schema.push_back(std::move(binding));
  }
  return schema;
}

/// The tile each source axis is realized at, plus the declared tail policy. The
/// default is the full extent: a source that offers no tile choice is realized
/// as one tile, not refused.
struct ElementwiseTiles {
  int64_t t0 = 0;
  int64_t t1 = 0;
  std::string tailPolicy = "none";
};

/// Resolves the source-owned tile choices by the axis each one names. A
/// source-owned choice with no matching axis has no realization in this
/// increment and is reported rather than ignored. A tile that does not divide
/// its extent stays rejected unless the source carries the importer's proven
/// tail declaration for exactly this block: the divisibility gate is not
/// silently removed.
llvm::Expected<ElementwiseTiles>
resolveTiles(const ElementwiseProgram &program, const perf::SearchSpace &space,
             const perf::Candidate &candidate) {
  ElementwiseTiles tiles{program.d0, program.d1};
  for (const ParameterBinding &binding : elementwiseSchema(space)) {
    if (binding.owner != BindingOwner::Source)
      continue;
    const perf::SearchParam *param = space.findParam(binding.parameter);
    if (!param)
      return error("axis_not_realized: " + binding.parameter);
    // A declared tail policy is a source fact this family realizes by
    // decomposing the rounded domain into interior and exact edge rectangles.
    if (param->kind == "tail_policy") {
      std::optional<llvm::StringRef> value =
          candidate.symbol(binding.parameter);
      if (!value)
        return error("source_binding: " + binding.parameter +
                     " is not a symbolic binding");
      if (*value != "masked-edge")
        return error("tail_policy: unsupported tail policy '" + value->str() +
                     "'");
      tiles.tailPolicy = value->str();
      continue;
    }
    // A source-owned choice this family does not implement -- a fragment
    // shape, an unrelated integer -- has no realization here.
    if (param->kind != "integer")
      return error("axis_not_realized: " + binding.parameter);
    std::optional<int64_t> value = candidate.integer(binding.parameter);
    if (!value)
      return error("source_binding: " + binding.parameter +
                   " is not an integer binding");
    if (binding.role == "axis:d0")
      tiles.t0 = *value;
    else if (binding.role == "axis:d1")
      tiles.t1 = *value;
    else
      return error("axis_not_realized: " + binding.parameter);
  }
  if (tiles.t0 > 0 && tiles.t1 > 0 && !(program.d0 % tiles.t0) &&
      !(program.d1 % tiles.t1))
    return tiles;
  if (tiles.tailPolicy == "none")
    return error("source_legality: tile_divisibility: T0/T1 must divide d0/d1");
  // The policy is only realizable for the block the importer proved. A
  // different candidate cannot inherit that proof.
  auto proven = program.generic->getAttrOfType<StringAttr>("llk.tail_policy");
  auto block =
      program.generic->getAttrOfType<DenseI64ArrayAttr>("llk.tail_block");
  if (!proven || proven.getValue() != tiles.tailPolicy || !block ||
      block.size() != 2 || block[0] != tiles.t0 || block[1] != tiles.t1)
    return error("tail_policy: the source did not prove this block's tail "
                 "mask");
  return tiles;
}

/// The one structured family this increment supports.
class ElementwiseSourceAdapter final : public SourceAdapter {
public:
  llvm::StringRef key() const override { return elementwiseKey; }

  bool accepts(Operation *root) const override {
    return root && isa<linalg::GenericOp>(root);
  }

  llvm::Expected<WorkloadContract> describe(Operation *root) const override {
    auto program = analyze(root);
    if (!program)
      return program.takeError();

    WorkloadContract described;
    described.axes = {{"d0", program->d0}, {"d1", program->d1}};
    // Ports follow the source's operand order: the two read inputs, the
    // destination (declared, but with no incoming read), and the written
    // result.
    const std::vector<shape::Dimension> shape{program->d0, program->d1};
    const std::string dtype = "f32";
    described.ports.push_back(
        {"input0", PortDirection::Input, PortEffect::Read, shape, dtype});
    described.ports.push_back(
        {"input1", PortDirection::Input, PortEffect::Read, shape, dtype});
    described.ports.push_back(
        {"destination", PortDirection::Input, PortEffect::None, shape, dtype});
    described.ports.push_back(
        {"output", PortDirection::Output, PortEffect::Write, shape, dtype});
    // The first supported numerical mode. Approximation stays an explicit
    // source fact, so this adapter never invents one.
    described.numericalContract = "strict-f32";
    // The primitives realization actually emits: operand movement, the vector
    // operation, and the tile store into the carried output. A rounded domain
    // realizes every rectangle through the bounded `micro.tile_load`, whose
    // `valid[...]` rectangle is clipped before any read, so the contract names
    // that primitive instead of the unbounded tile copy. A required primitive
    // the emitter does not produce would be a false contract.
    described.requiredPrimitives = {"micro.vector", "micro.tile_store"};
    described.requiredPrimitives.push_back(root->hasAttr("llk.tail_policy")
                                               ? "micro.tile_load"
                                               : "micro.tile_async_copy");
    return described;
  }

  llvm::Expected<std::vector<ParameterBinding>>
  bindings(Operation *, const perf::SearchSpace &space) const override {
    return elementwiseSchema(space);
  }

  llvm::Expected<SourceRealization>
  realize(ModuleOp clone, Operation *selectedRoot,
          const WorkloadContract &contract, const perf::SearchSpace &space,
          const perf::Candidate &candidate) const override {
    auto program = analyze(selectedRoot);
    if (!program)
      return program.takeError();

    // Direct callers get the same contract check the driver performs: the
    // adapter's description, not a label, is what it realizes.
    auto described = describe(selectedRoot);
    if (!described)
      return described.takeError();
    if (canonicalWorkloadContract(*described) !=
        canonicalWorkloadContract(contract))
      return error("source_contract_mismatch: elementwise source describes a "
                   "different contract");

    // Resolve the source-owned choices before emitting anything: an axis this
    // increment cannot realize, or a tile that does not divide its extent, is
    // reported instead of being silently ignored or padded.
    auto tiles = resolveTiles(*program, space, candidate);
    if (!tiles)
      return tiles.takeError();

    MLIRContext *context = clone.getContext();
    // Parsing a Linalg-only source does not load the output dialect.
    context->getOrLoadDialect<micro::MicroDialect>();
    Location loc = selectedRoot->getLoc();

    std::string base = enclosingSymbol(selectedRoot);
    base = base.empty() ? std::string("elementwise") : base + "_elementwise";
    SymbolTable symbols(clone);
    std::string symName = base;
    for (unsigned suffix = 1; symbols.lookup(symName); ++suffix)
      symName = base + "_" + std::to_string(suffix);

    // The signature names every root operand and result. The destination is a
    // declared input, but it has no incoming read: `tensor.empty` is never
    // viewed or staged, and an unused block argument is exactly that fact.
    llvm::SmallVector<Type, 3> inputTypes{program->inputs[0].getType(),
                                          program->inputs[1].getType(),
                                          program->destination.getType()};
    llvm::SmallVector<Type, 1> resultTypes{program->result.getType()};

    OpBuilder builder(context);
    builder.setInsertionPointToEnd(clone.getBody());
    auto kernel = micro::KernelOp::create(
        builder, loc, symName,
        TypeAttr::get(FunctionType::get(context, inputTypes, resultTypes)),
        /*workload=*/StringAttr(), /*target=*/StringAttr(),
        /*candidate=*/StringAttr(), /*schedule_id=*/IntegerAttr());

    Block *body = builder.createBlock(&kernel.getBody());
    for (Type input : inputTypes)
      body->addArgument(input, loc);
    builder.setInsertionPointToStart(body);

    // Target-independent placement: external tensors are viewed in dram,
    // staged through sram, and computed into acc. No layout or owner is
    // invented -- those are mapping choices. A source that offers no tile
    // choice is realized as one tile covering the full extent.
    micro::LayoutAttr layout;
    micro::OwnerAttr owner;
    auto dram = micro::MemorySpaceAttr::get(context, micro::MemorySpace::dram);
    auto sram = micro::MemorySpaceAttr::get(context, micro::MemorySpace::sram);
    auto acc = micro::MemorySpaceAttr::get(context, micro::MemorySpace::acc);
    auto tile = [&](ArrayRef<int64_t> shape, micro::MemorySpaceAttr memory) {
      return micro::TileType::get(context, shape, program->elementType, layout,
                                  memory, owner);
    };
    Type tokenType = micro::AsyncTokenType::get(context);
    const SmallVector<int64_t, 2> outputShape{program->d0, program->d1};
    Type outputTileType = tile(outputShape, sram);

    // Tile-by-tile output is SSA-carried: each rectangle updates the
    // destination the next one hands on, and every loop yields what its body
    // produced.
    Value output =
        micro::TileAllocOp::create(builder, loc, outputTileType).getResult();
    /// One tile loop and its body, whichever kind it is: an empty `map` makes
    /// the iterations temporal, a named one makes them spatial.
    struct TileLoop {
      Block *body = nullptr;
      ValueRange results;
    };
    auto openLoop = [&](int64_t lower, int64_t upper, int64_t step,
                        StringRef map, ValueRange carried) -> TileLoop {
      Value lb =
          arith::ConstantIndexOp::create(builder, loc, lower).getResult();
      Value ub =
          arith::ConstantIndexOp::create(builder, loc, upper).getResult();
      Value by = arith::ConstantIndexOp::create(builder, loc, step).getResult();
      llvm::SmallVector<Type> bodyTypes{IndexType::get(context)};
      bodyTypes.append(carried.getTypes().begin(), carried.getTypes().end());
      llvm::SmallVector<Location> bodyLocations(bodyTypes.size(), loc);
      Block *loopBody = nullptr;
      ValueRange results;
      if (map.empty()) {
        auto loop = micro::ForOp::create(builder, loc, carried.getTypes(), lb,
                                         ub, by, carried);
        loopBody =
            builder.createBlock(&loop.getBody(), {}, bodyTypes, bodyLocations);
        results = loop.getResults();
      } else {
        auto loop = micro::SpatialForOp::create(
            builder, loc, carried.getTypes(), lb, ub, by,
            micro::MappingTargetAttr::get(context, map), carried);
        loopBody =
            builder.createBlock(&loop.getBody(), {}, bodyTypes, bodyLocations);
        results = loop.getResults();
      }
      builder.setInsertionPointToEnd(loopBody);
      micro::YieldOp::create(builder, loc);
      builder.setInsertionPoint(loopBody->getTerminator());
      return {loopBody, results};
    };

    // One block rectangle. An empty `valid` is a full interior tile, realized
    // exactly as before. An edge rectangle keeps the block's compile-time
    // capacity but states the exact static fraction of it that is real, so the
    // bounded load copies only in-bounds source elements and the clipped store
    // writes only the declared domain. No view is ever positioned outside its
    // source and then repaired by a later mask.
    // A rounded domain serializes its block walk: every rectangle reuses the
    // previous one's staged, accumulator and carried-output storage, which is
    // what keeps a 5x5 grid of 8x16 tiles -- and the whole rounded output tile
    // each store versions -- inside the machine's capacities. A divisible
    // source keeps the original worker-lane spatial nesting unchanged.
    const bool temporalTiles = tiles->tailPolicy != "none";
    auto emitRectangle =
        [&](Value carried, int64_t rowLower, int64_t rowUpper, int64_t rowStep,
            int64_t columnLower, int64_t columnUpper, int64_t columnStep,
            ArrayRef<int64_t> tileShape, ArrayRef<int64_t> valid) -> Value {
      TileLoop d0Loop = openLoop(rowLower, rowUpper, rowStep,
                                 temporalTiles ? StringRef("") : "worker",
                                 ValueRange{carried});
      BlockArgument d0 = d0Loop.body->getArgument(0);
      Value d0Carried = d0Loop.body->getArgument(1);
      TileLoop d1Loop = openLoop(columnLower, columnUpper, columnStep,
                                 temporalTiles ? StringRef("") : "lane",
                                 ValueRange{d0Carried});
      BlockArgument d1 = d1Loop.body->getArgument(0);
      Value d1Carried = d1Loop.body->getArgument(1);

      llvm::SmallVector<Value, 2> validSizes;
      for (int64_t extent : valid)
        validSizes.push_back(
            arith::ConstantIndexOp::create(builder, loc, extent).getResult());

      llvm::SmallVector<Value, 2> staged;
      llvm::SmallVector<Value, 2> tokens;
      for (unsigned i = 0; i < program->inputs.size(); ++i) {
        // The kernel is IsolatedFromAbove, so the operands it reads are its own
        // entry block arguments -- the source-local values are not in scope.
        Value input = body->getArgument(i);
        if (validSizes.empty()) {
          Value view = micro::TileViewOp::create(
              builder, loc, tile(tileShape, dram), input, ValueRange{d0, d1},
              DenseI64ArrayAttr::get(context, tileShape),
              /*layout=*/micro::LayoutAttr());
          auto copy = micro::TileAsyncCopyOp::create(
              builder, loc, tile(tileShape, sram), tokenType, view, sram,
              /*owner=*/micro::OwnerAttr());
          staged.push_back(copy.getResult());
          tokens.push_back(copy.getToken());
        } else {
          auto load = micro::TileLoadOp::create(
              builder, loc, tile(tileShape, sram), tokenType, input,
              ValueRange{d0, d1}, validSizes, builder.getF32FloatAttr(0.0F),
              sram, /*owner=*/micro::OwnerAttr());
          staged.push_back(load.getResult());
          tokens.push_back(load.getToken());
        }
      }
      micro::WaitOp::create(builder, loc, tokens);

      // The validated scalar operation decides the primitive key inside this
      // adapter only.
      Value result = micro::VectorOp::create(
          builder, loc, tile(tileShape, acc),
          StringAttr::get(context, program->multiply ? "mul" : "add"),
          ValueRange{staged[0], staged[1]}, /*math_mode=*/StringAttr());
      Value written = micro::TileStoreOp::create(
                          builder, loc, outputTileType, result, d1Carried,
                          ValueRange{d0, d1}, sram, validSizes)
                          .getResult();

      if (auto terminator =
              dyn_cast<micro::YieldOp>(d1Loop.body->getTerminator())) {
        builder.setInsertionPoint(terminator);
        micro::YieldOp::create(builder, loc, ValueRange{written});
        terminator.erase();
      }
      if (auto terminator =
              dyn_cast<micro::YieldOp>(d0Loop.body->getTerminator())) {
        builder.setInsertionPoint(terminator);
        micro::YieldOp::create(builder, loc, d1Loop.results);
        terminator.erase();
      }
      // The next rectangle is emitted into the kernel body, not into this
      // loop's body.
      builder.setInsertionPointToEnd(body);
      return d0Loop.results.front();
    };

    // Interior blocks first, then the right, bottom and corner rectangles. The
    // decomposition writes every element exactly once and each rectangle's
    // offsets and static valid extents are inside the source and destination.
    const int64_t fullRows = (program->d0 / tiles->t0) * tiles->t0;
    const int64_t fullColumns = (program->d1 / tiles->t1) * tiles->t1;
    const int64_t rowTail = program->d0 - fullRows;
    const int64_t columnTail = program->d1 - fullColumns;
    const SmallVector<int64_t, 2> blockShape{tiles->t0, tiles->t1};

    // A rounded program realizes even its interior rectangles through the
    // bounded primitive: the contract then names one movement primitive for
    // every rectangle, and a full rectangle states `valid == capacity`.
    ArrayRef<int64_t> fullValid;
    if (temporalTiles)
      fullValid = blockShape;

    Value carried = output;
    if (fullRows > 0 && fullColumns > 0)
      carried = emitRectangle(carried, 0, fullRows, tiles->t0, 0, fullColumns,
                              tiles->t1, blockShape, fullValid);
    if (fullRows > 0 && columnTail > 0)
      carried = emitRectangle(carried, 0, fullRows, tiles->t0, fullColumns,
                              fullColumns + 1, 1, blockShape,
                              ArrayRef<int64_t>{tiles->t0, columnTail});
    if (rowTail > 0 && fullColumns > 0)
      carried = emitRectangle(carried, fullRows, fullRows + 1, 1, 0,
                              fullColumns, tiles->t1, blockShape,
                              ArrayRef<int64_t>{rowTail, tiles->t1});
    if (rowTail > 0 && columnTail > 0)
      carried = emitRectangle(carried, fullRows, fullRows + 1, 1, fullColumns,
                              fullColumns + 1, 1, blockShape,
                              ArrayRef<int64_t>{rowTail, columnTail});

    // The kernel body has no terminator yet: the rectangles were emitted into
    // it in order, so the yielded value is what the last loop left.
    builder.setInsertionPointToEnd(body);
    micro::YieldOp::create(builder, loc, ValueRange{carried});

    // Carry the canonical numerical semantics as an explicit IR fact.
    if (!contract.numericalContract.empty())
      kernel->setAttr("micro.numerical_contract",
                      StringAttr::get(context, contract.numericalContract));

    SourceRealization realization;
    realization.kernel = kernel.getOperation();
    // The receipt names what realization actually produced: every source-owned
    // choice that reached this point was emitted as a tile loop. Mapping-owned
    // choices stay pending for the later stage.
    for (const ParameterBinding &binding : elementwiseSchema(space))
      if (binding.owner == BindingOwner::Source)
        realization.receipt.source.push_back(binding.parameter);
    return realization;
  }
};

} // namespace

llvm::StringRef elementwiseSourceAdapterKey() { return elementwiseKey; }

std::unique_ptr<SourceAdapter> makeElementwiseSourceAdapter() {
  return std::make_unique<ElementwiseSourceAdapter>();
}

} // namespace mlir::llk::workload
