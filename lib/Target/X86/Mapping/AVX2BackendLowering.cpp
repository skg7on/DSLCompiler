//===- AVX2BackendLowering.cpp - Selected AVX2 Linalg lowering ------------===//

#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MathExtras.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>

namespace mlir::llk::target::avx2 {
namespace {

std::optional<int64_t> selectedWidth(DictionaryAttr mapping) {
  std::optional<int64_t> width;
  auto remember = [&](DictionaryAttr parameters) -> bool {
    if (!parameters)
      return true;
    auto raw = parameters.getAs<IntegerAttr>("VW");
    if (!raw)
      return true;
    if (width && *width != raw.getInt())
      return false;
    width = raw.getInt();
    return true;
  };

  if (!remember(mapping.getAs<DictionaryAttr>("bundle_parameters")))
    return std::nullopt;
  auto layouts = mapping.getAs<DictionaryAttr>("layout_parameters");
  if (layouts)
    for (NamedAttribute entry : layouts)
      if (!remember(dyn_cast<DictionaryAttr>(entry.getValue())))
        return std::nullopt;
  return width;
}

SmallVector<StringAttr> groupOperations(Operation *op) {
  SmallVector<StringAttr> result;
  if (auto operations = op->getAttrOfType<ArrayAttr>("micro.group_ops")) {
    for (Attribute value : operations)
      if (auto name = dyn_cast<StringAttr>(value))
        result.push_back(name);
  } else if (auto name = op->getAttrOfType<StringAttr>("micro.op")) {
    result.push_back(name);
  }
  return result;
}

void appendGroupOperations(SmallVectorImpl<Attribute> &operations,
                           Operation *op) {
  if (auto group = op->getAttrOfType<ArrayAttr>("micro.group_ops")) {
    for (Attribute name : group)
      if (!llvm::is_contained(operations, name))
        operations.push_back(name);
    return;
  }
  if (Attribute name = op->getAttr("micro.op"))
    if (!llvm::is_contained(operations, name))
      operations.push_back(name);
}

LogicalResult fuseSelectedElementwiseGroups(ModuleOp module,
                                            IRRewriter &rewriter) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<linalg::GenericOp> selected;
    module.walk([&](linalg::GenericOp generic) {
      if (generic->hasAttr("micro.mapping"))
        selected.push_back(generic);
    });

    for (linalg::GenericOp consumer : selected) {
      auto consumerMapping =
          consumer->getAttrOfType<DictionaryAttr>("micro.mapping");
      auto emitter = consumerMapping.getAs<StringAttr>("emitter");
      auto consumerInstance = consumerMapping.getAs<IntegerAttr>("instance");
      if (!emitter || emitter.getValue() != "avx2_fused_convert_silu_mul" ||
          !consumerInstance)
        continue;

      for (OpOperand &operand : consumer->getOpOperands()) {
        auto producer =
            dyn_cast_or_null<linalg::GenericOp>(operand.get().getDefiningOp());
        if (!producer)
          continue;
        auto producerMapping =
            producer->getAttrOfType<DictionaryAttr>("micro.mapping");
        auto producerEmitter = producerMapping.getAs<StringAttr>("emitter");
        auto producerInstance = producerMapping.getAs<IntegerAttr>("instance");
        if (!producerEmitter ||
            producerEmitter.getValue() != emitter.getValue() ||
            !producerInstance ||
            producerInstance.getInt() != consumerInstance.getInt() ||
            !linalg::areElementwiseOpsFusable(&operand))
          continue;

        Attribute producerMathMode = producer->getAttr("micro.math_mode");
        Attribute consumerMathMode = consumer->getAttr("micro.math_mode");
        if (producerMathMode != consumerMathMode) {
          consumer.emitError("cannot fuse selected AVX2 group instance ")
              << consumerInstance.getInt() << " with inconsistent math modes";
          return failure();
        }

        rewriter.setInsertionPoint(consumer);
        FailureOr<linalg::ElementwiseOpFusionResult> fusion =
            linalg::fuseElementwiseOps(rewriter, &operand);
        if (failed(fusion)) {
          consumer.emitError("failed to fuse selected AVX2 group instance ")
              << consumerInstance.getInt();
          return failure();
        }

        SmallVector<Attribute> operations;
        appendGroupOperations(operations, producer);
        appendGroupOperations(operations, consumer);
        Operation *fused = fusion->fusedOp;
        fused->setAttr("micro.mapping", consumerMapping);
        if (Attribute routes = consumer->getAttr("micro.routes"))
          fused->setAttr("micro.routes", routes);
        fused->setAttr("micro.op",
                       StringAttr::get(module.getContext(), "fused"));
        fused->setAttr("micro.group_ops",
                       ArrayAttr::get(module.getContext(), operations));
        if (consumerMathMode)
          fused->setAttr("micro.math_mode", consumerMathMode);

        for (auto [original, replacement] : fusion->replacements) {
          rewriter.replaceUsesWithIf(
              original, replacement, [&](OpOperand &use) {
                return use.get().getDefiningOp() != producer;
              });
        }
        rewriter.eraseOp(consumer);
        if (llvm::all_of(producer->getResults(),
                         [](Value result) { return result.use_empty(); }))
          rewriter.eraseOp(producer);
        changed = true;
        break;
      }
      if (changed)
        break;
    }
  }
  return success();
}

bool hasFusedStructuralConversion(ModuleOp module, int64_t instance) {
  auto structural =
      module->getAttrOfType<ArrayAttr>("micro.structural_mappings");
  if (!structural)
    return false;
  for (Attribute item : structural) {
    auto record = dyn_cast<DictionaryAttr>(item);
    auto mapping =
        record ? record.getAs<DictionaryAttr>("mapping") : DictionaryAttr{};
    auto emitter =
        mapping ? mapping.getAs<StringAttr>("emitter") : StringAttr{};
    auto recordedInstance =
        mapping ? mapping.getAs<IntegerAttr>("instance") : IntegerAttr{};
    auto op = record ? record.getAs<StringAttr>("op") : StringAttr{};
    if (emitter && emitter.getValue() == "avx2_fused_convert_silu_mul" &&
        recordedInstance && recordedInstance.getInt() == instance && op &&
        op.getValue() == "convert")
      return true;
  }
  return false;
}

void promoteUnitRowArithmetic(ModuleOp module, IRRewriter &rewriter) {
  SmallVector<Operation *> arithmetic;
  module.walk([&](Operation *op) {
    if (!op->hasAttr("llk.avx2.selected_compute"))
      return;
    StringRef dialect = op->getName().getDialectNamespace();
    if ((dialect != "arith" && dialect != "math") || op->getNumResults() != 1 ||
        op->getNumOperands() == 0) {
      op->removeAttr("llk.avx2.selected_compute");
      return;
    }
    auto resultType = dyn_cast<VectorType>(op->getResult(0).getType());
    if (resultType && resultType.getRank() == 2 &&
        resultType.getShape()[0] == 1)
      arithmetic.push_back(op);
    else
      op->removeAttr("llk.avx2.selected_compute");
  });

  for (Operation *op : arithmetic) {
    auto oldType = cast<VectorType>(op->getResult(0).getType());
    auto flatType =
        VectorType::get({oldType.getShape()[1]}, oldType.getElementType());
    rewriter.setInsertionPoint(op);
    SmallVector<Value> operands;
    for (Value operand : op->getOperands()) {
      auto operandType = dyn_cast<VectorType>(operand.getType());
      if (operandType && operandType.getRank() == 2 &&
          operandType.getShape()[0] == 1 &&
          operandType.getShape()[1] == oldType.getShape()[1]) {
        auto flatOperandType = VectorType::get({operandType.getShape()[1]},
                                               operandType.getElementType());
        operands.push_back(vector::ShapeCastOp::create(
            rewriter, op->getLoc(), flatOperandType, operand));
      } else {
        operands.push_back(operand);
      }
    }

    OperationState state(op->getLoc(), op->getName());
    state.addOperands(operands);
    state.addTypes(flatType);
    state.addAttributes(op->getAttrs());
    Operation *flatOp = rewriter.create(state);
    flatOp->removeAttr("llk.avx2.selected_compute");
    Value restored = vector::ShapeCastOp::create(rewriter, op->getLoc(),
                                                 oldType, flatOp->getResult(0));
    rewriter.replaceOp(op, restored);
  }
}

struct AVX2BackendLoweringPass
    : PassWrapper<AVX2BackendLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AVX2BackendLoweringPass)

  AVX2BackendLoweringPass() = default;
  AVX2BackendLoweringPass(int64_t f32Width, int64_t bf16Width)
      : f32VectorWidth(f32Width), bf16VectorWidth(bf16Width) {}

  StringRef getArgument() const final { return "llk-avx2-selected-lowering"; }
  StringRef getDescription() const final {
    return "Realize selected AVX2 elementwise bundles in Vector IR";
  }

  int64_t f32VectorWidth = 0;
  int64_t bf16VectorWidth = 0;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, linalg::LinalgDialect,
                    math::MathDialect, scf::SCFDialect, tensor::TensorDialect,
                    vector::VectorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());
    if (failed(fuseSelectedElementwiseGroups(module, rewriter))) {
      signalPassFailure();
      return;
    }

    llvm::SmallDenseSet<int64_t> realizedInstances;
    SmallVector<linalg::MatmulOp> selectedMatmuls;
    module.walk([&](linalg::MatmulOp matmul) {
      if (matmul->hasAttr("micro.mapping"))
        selectedMatmuls.push_back(matmul);
    });
    for (linalg::MatmulOp matmul : selectedMatmuls) {
      auto mapping = matmul->getAttrOfType<DictionaryAttr>("micro.mapping");
      auto emitter = mapping.getAs<StringAttr>("emitter");
      auto bundle = mapping.getAs<StringAttr>("bundle");
      auto instance = mapping.getAs<IntegerAttr>("instance");
      auto microOp = matmul->getAttrOfType<StringAttr>("micro.op");
      const std::string context =
          "bundle '" + (bundle ? bundle.getValue().str() : "<unknown>") +
          "' instance " +
          (instance ? std::to_string(instance.getInt()) : "<unknown>");
      auto fail = [&](llvm::Twine reason) {
        matmul.emitError("AVX2 selected lowering rejected ")
            << context << ": " << reason;
        signalPassFailure();
      };
      if (!emitter || emitter.getValue() != "avx2_mma" || !instance ||
          !microOp || microOp.getValue() != "mma") {
        fail("Linalg matmul has unsupported or missing selected mapping "
             "provenance");
        return;
      }
      std::optional<int64_t> width = selectedWidth(mapping);
      if (!width || (*width != 4 && *width != 8)) {
        fail("selected VW is missing, conflicting, or unsupported");
        return;
      }
      auto resultType =
          dyn_cast<RankedTensorType>(matmul.getResult(0).getType());
      if (!resultType || resultType.getRank() != 2 ||
          !resultType.hasStaticShape() ||
          resultType.getShape()[1] % *width != 0 ||
          !resultType.getElementType().isF32()) {
        fail("selected contraction needs a static f32 accumulator tile "
             "divisible by VW");
        return;
      }
      SmallVector<int64_t> tileSizes{1, *width, 0};
      linalg::LinalgTilingOptions tiling;
      tiling.setTileSizes(tileSizes);
      rewriter.setInsertionPoint(matmul);
      FailureOr<linalg::TiledLinalgOp> tiled =
          linalg::tileLinalgOp(rewriter, matmul, tiling);
      if (failed(tiled)) {
        fail("Linalg matmul tiling failed for the selected vector width");
        return;
      }
      auto lhsType = cast<RankedTensorType>(matmul.getInputs()[0].getType());
      SmallVector<int64_t> vectorSizes{1, *width, lhsType.getShape()[1]};
      SmallVector<bool> scalableVecDims(3, false);
      FailureOr<linalg::VectorizationResult> vectorized = linalg::vectorize(
          rewriter, tiled->op.getOperation(), vectorSizes, scalableVecDims,
          /*vectorizeNDExtract=*/false,
          /*flatten1DDepthwiseConv=*/false,
          /*assumeDynamicDimsMatchVecSizes=*/false,
          /*createNamedContraction=*/true);
      if (failed(vectorized)) {
        fail("Linalg matmul vectorization failed for the selected width");
        return;
      }
      rewriter.replaceOp(tiled->op, vectorized->replacements);
      rewriter.replaceOp(matmul, tiled->tensorResults);
      realizedInstances.insert(instance.getInt());
    }

    SmallVector<linalg::ReduceOp> selectedReductions;
    module.walk([&](linalg::ReduceOp reduction) {
      if (reduction->hasAttr("micro.mapping"))
        selectedReductions.push_back(reduction);
    });
    for (linalg::ReduceOp reduction : selectedReductions) {
      auto mapping = reduction->getAttrOfType<DictionaryAttr>("micro.mapping");
      auto emitter = mapping.getAs<StringAttr>("emitter");
      auto bundle = mapping.getAs<StringAttr>("bundle");
      auto instance = mapping.getAs<IntegerAttr>("instance");
      auto microOp = reduction->getAttrOfType<StringAttr>("micro.op");
      auto reduceKind =
          reduction->getAttrOfType<StringAttr>("micro.reduce_kind");
      const std::string context =
          "bundle '" + (bundle ? bundle.getValue().str() : "<unknown>") +
          "' instance " +
          (instance ? std::to_string(instance.getInt()) : "<unknown>");
      auto fail = [&](llvm::Twine reason) {
        reduction.emitError("AVX2 selected lowering rejected ")
            << context << ": " << reason;
        signalPassFailure();
      };
      if (!emitter || emitter.getValue() != "avx2_reduce" || !instance ||
          !microOp || microOp.getValue() != "reduce" || !reduceKind ||
          reduceKind.getValue() != "sum") {
        fail("selected reduction has unsupported or missing provenance or "
             "semantics");
        return;
      }
      std::optional<int64_t> width = selectedWidth(mapping);
      if (!width || (*width != 4 && *width != 8)) {
        fail("selected VW is missing, conflicting, or unsupported");
        return;
      }
      if (reduction.getInputs().size() != 1 ||
          reduction->getNumOperands() != 2 || reduction->getNumResults() != 1) {
        fail("selected sum reduction needs one input and one accumulator");
        return;
      }
      auto inputType =
          dyn_cast<RankedTensorType>(reduction.getInputs()[0].getType());
      auto resultType =
          dyn_cast<RankedTensorType>(reduction.getResult(0).getType());
      auto dimensions =
          reduction->getAttrOfType<DenseI64ArrayAttr>("dimensions");
      if (!inputType || !resultType || !inputType.hasStaticShape() ||
          !resultType.hasStaticShape() || inputType.getRank() != 2 ||
          resultType.getRank() != 1 || !inputType.getElementType().isF32() ||
          !resultType.getElementType().isF32() || !dimensions ||
          dimensions.size() != 1 || dimensions[0] < 0 || dimensions[0] >= 2) {
        fail("selected sum needs static rank-two f32 input and rank-one "
             "f32 output with one reduction axis");
        return;
      }
      int64_t reductionAxis = dimensions[0];
      int64_t parallelAxis = reductionAxis == 0 ? 1 : 0;
      if (inputType.getShape()[parallelAxis] % *width != 0) {
        fail("selected VW does not divide the reduction output tile");
        return;
      }
      SmallVector<int64_t> tileSizes{0, 0};
      tileSizes[parallelAxis] = *width;
      linalg::LinalgTilingOptions tiling;
      tiling.setTileSizes(tileSizes);
      rewriter.setInsertionPoint(reduction);
      FailureOr<linalg::TiledLinalgOp> tiled =
          linalg::tileLinalgOp(rewriter, reduction, tiling);
      if (failed(tiled)) {
        fail("Linalg sum tiling failed for the selected vector width");
        return;
      }
      SmallVector<int64_t> vectorSizes(inputType.getShape().begin(),
                                       inputType.getShape().end());
      vectorSizes[parallelAxis] = *width;
      SmallVector<bool> scalableVecDims(2, false);
      FailureOr<linalg::VectorizationResult> vectorized = linalg::vectorize(
          rewriter, tiled->op.getOperation(), vectorSizes, scalableVecDims);
      if (failed(vectorized)) {
        fail("Linalg sum vectorization failed for the selected vector width");
        return;
      }
      rewriter.replaceOp(tiled->op, vectorized->replacements);
      rewriter.replaceOp(reduction, tiled->tensorResults);
      realizedInstances.insert(instance.getInt());
    }

    SmallVector<linalg::GenericOp> selected;
    SmallVector<linalg::GenericOp> selectedTransforms;
    module.walk([&](linalg::GenericOp generic) {
      if (generic->hasAttr("micro.mapping")) {
        selected.push_back(generic);
        return;
      }
      auto microOp = generic->getAttrOfType<StringAttr>("micro.op");
      if (microOp && microOp.getValue() == "transform" &&
          generic->hasAttr("micro.engine"))
        selectedTransforms.push_back(generic);
    });

    std::map<int64_t, unsigned> fusedGroupSizes;
    for (linalg::GenericOp generic : selected) {
      auto mapping = generic->getAttrOfType<DictionaryAttr>("micro.mapping");
      auto emitter = mapping.getAs<StringAttr>("emitter");
      auto instance = mapping.getAs<IntegerAttr>("instance");
      if (emitter && emitter.getValue() == "avx2_fused_convert_silu_mul" &&
          instance)
        ++fusedGroupSizes[instance.getInt()];
    }

    for (linalg::GenericOp generic : selected) {
      auto mapping = generic->getAttrOfType<DictionaryAttr>("micro.mapping");
      auto emitter = mapping.getAs<StringAttr>("emitter");
      auto bundle = mapping.getAs<StringAttr>("bundle");
      auto instance = mapping.getAs<IntegerAttr>("instance");
      auto microOp = generic->getAttrOfType<StringAttr>("micro.op");
      const std::string context =
          "bundle '" + (bundle ? bundle.getValue().str() : "<unknown>") +
          "' instance " +
          (instance ? std::to_string(instance.getInt()) : "<unknown>");
      auto fail = [&](llvm::Twine reason) {
        generic.emitError("AVX2 selected lowering rejected ")
            << context << ": " << reason;
        signalPassFailure();
      };
      if (!emitter || !instance || !microOp) {
        fail("Linalg operation lost its selected mapping provenance");
        return;
      }
      if (emitter.getValue() != "avx2_vector_add" &&
          emitter.getValue() != "avx2_vector_convert" &&
          emitter.getValue() != "avx2_vector_silu" &&
          emitter.getValue() != "avx2_vector_mul" &&
          emitter.getValue() != "avx2_fused_convert_silu_mul") {
        fail("selected compute emitter has no AVX2 Vector implementation");
        return;
      }

      std::optional<int64_t> width = selectedWidth(mapping);
      if (!width || (*width != 4 && *width != 8)) {
        fail("selected VW is missing, conflicting, or unsupported");
        return;
      }
      auto resultType =
          dyn_cast<RankedTensorType>(generic.getResult(0).getType());
      if (!resultType || resultType.getRank() != 2 ||
          !resultType.hasStaticShape() ||
          resultType.getShape()[1] % *width != 0) {
        fail("selected vector width needs a divisible, static rank-two tile");
        return;
      }
      StringRef operation = microOp.getValue();
      if (emitter.getValue() == "avx2_fused_convert_silu_mul") {
        SmallVector<StringAttr> members = groupOperations(generic);
        llvm::SmallDenseSet<StringRef> memberNames;
        for (StringAttr member : members)
          memberNames.insert(member.getValue());
        bool hasConvert =
            memberNames.contains("convert") ||
            hasFusedStructuralConversion(module, instance.getInt());
        if (fusedGroupSizes[instance.getInt()] != 1 || !hasConvert ||
            !memberNames.contains("silu") || !memberNames.contains("mul")) {
          fail("selected fused group did not reach one Vector body with its "
               "structural conversion");
          return;
        }
      } else if (operation != "add" && operation != "convert" &&
                 operation != "silu" && operation != "mul") {
        fail("selected Micro operation has no elementwise Vector lowering");
        return;
      }
      if (!resultType.getElementType().isF32() &&
          !resultType.getElementType().isBF16()) {
        fail("selected element type has no AVX2 Vector arithmetic path");
        return;
      }

      SmallVector<int64_t> tileSizes{1, *width};
      linalg::LinalgTilingOptions tiling;
      tiling.setTileSizes(tileSizes);
      generic.getRegion().walk([&](Operation *inner) {
        StringRef dialect = inner->getName().getDialectNamespace();
        if (dialect == "arith" || dialect == "math")
          inner->setAttr("llk.avx2.selected_compute",
                         UnitAttr::get(&getContext()));
      });
      rewriter.setInsertionPoint(generic);
      FailureOr<linalg::TiledLinalgOp> tiled =
          linalg::tileLinalgOp(rewriter, generic, tiling);
      if (failed(tiled)) {
        fail("Linalg tiling failed for the selected vector width");
        return;
      }

      SmallVector<int64_t> vectorSizes{1, *width};
      FailureOr<linalg::VectorizationResult> vectorized =
          linalg::vectorize(rewriter, tiled->op.getOperation(), vectorSizes);
      if (failed(vectorized)) {
        fail("Linalg vectorization failed for the selected vector width");
        return;
      }
      rewriter.replaceOp(tiled->op, vectorized->replacements);
      rewriter.replaceOp(generic, tiled->tensorResults);
      realizedInstances.insert(instance.getInt());
    }

    llvm::SmallDenseSet<int64_t> realizedConnections;
    unsigned anonymousTransformsRealized = 0;
    for (linalg::GenericOp generic : selectedTransforms) {
      auto engine = generic->getAttrOfType<StringAttr>("micro.engine");
      auto connection = generic->getAttrOfType<IntegerAttr>("micro.connection");
      auto resultType =
          dyn_cast<RankedTensorType>(generic.getResult(0).getType());
      int64_t width = 0;
      if (resultType && resultType.getElementType().isF32())
        width = f32VectorWidth;
      else if (resultType && resultType.getElementType().isBF16())
        width = bf16VectorWidth;
      const std::string context =
          "selected layout transform connection " +
          (connection ? std::to_string(connection.getInt()) : "<unknown>");
      auto fail = [&](llvm::Twine reason) {
        generic.emitError("AVX2 selected lowering rejected ")
            << context << " on engine '"
            << (engine ? engine.getValue() : StringRef("<unknown>"))
            << "': " << reason;
        signalPassFailure();
      };
      if (!engine || engine.getValue().empty() || !resultType ||
          !resultType.hasStaticShape() || resultType.getRank() != 2 ||
          width <= 0) {
        fail("layout transform needs a selected engine, static rank-two tile, "
             "and a declared AVX2 lane width");
        return;
      }
      SmallVector<int64_t> tileSizes{1, width};
      linalg::LinalgTilingOptions tiling;
      tiling.setTileSizes(tileSizes);
      rewriter.setInsertionPoint(generic);
      FailureOr<linalg::TiledLinalgOp> tiled =
          linalg::tileLinalgOp(rewriter, generic, tiling);
      if (failed(tiled)) {
        fail("layout transform tiling failed for the selected width");
        return;
      }
      SmallVector<bool> scalableVecDims(2, false);
      FailureOr<linalg::VectorizationResult> vectorized = linalg::vectorize(
          rewriter, tiled->op.getOperation(), tileSizes, scalableVecDims);
      if (failed(vectorized)) {
        fail("layout transform vectorization failed for the selected width");
        return;
      }
      rewriter.replaceOp(tiled->op, vectorized->replacements);
      rewriter.replaceOp(generic, tiled->tensorResults);
      if (connection)
        realizedConnections.insert(connection.getInt());
      else
        ++anonymousTransformsRealized;
    }

    promoteUnitRowArithmetic(module, rewriter);

    unsigned realizedGroupCount = realizedInstances.size() +
                                  realizedConnections.size() +
                                  anonymousTransformsRealized;
    module->setAttr("llk.backend_groups_realized",
                    IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                     realizedGroupCount));
    module->setAttr("llk.backend_connections_realized",
                    IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                     realizedConnections.size() +
                                         anonymousTransformsRealized));
  }
};

} // namespace

void buildAVX2SelectedBackendPipeline(mlir::OpPassManager &pm,
                                      int64_t f32VectorWidth,
                                      int64_t bf16VectorWidth) {
  pm.addPass(std::make_unique<AVX2BackendLoweringPass>(f32VectorWidth,
                                                       bf16VectorWidth));
}

} // namespace mlir::llk::target::avx2
