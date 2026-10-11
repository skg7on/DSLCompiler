#include "LLK/Conversion/LLKKernel/LLKKernelLowering.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"
using namespace mlir;
TEST(KernelContractTest, FragmentsRetainCallbackAndInitSemantics) {
  MLIRContext ctx;
  ctx.getOrLoadDialect<llk::LLKDialect>();
  ctx.getOrLoadDialect<micro::MicroDialect>();
  ctx.getOrLoadDialect<func::FuncDialect>();
  ctx.getOrLoadDialect<linalg::LinalgDialect>();
  llk::MicroKernelFragment fragment;
  {
    llk::LLKKernelRegistry r;
    llk::populateLLKKernelLowerings(r);
    auto module = parseSourceFile<ModuleOp>(
        std::string(LLK_SOURCE_DIR) +
            "/test/Conversion/LLKToMicro/swiglu_to_micro.mlir",
        &ctx);
    ASSERT_TRUE(module);
    Operation *root = nullptr;
    module->walk([&](Operation *op) {
      if (op->getName().getStringRef() == "llk.fused_swiglu")
        root = op;
    });
    ASSERT_NE(root, nullptr);
    llk::KernelContract c;
    ASSERT_TRUE(
        succeeded(r.lookup("llk.fused_swiglu")->inferContract(root, c)));
    ASSERT_TRUE(succeeded(
        r.lookup("llk.fused_swiglu")->describeMicroKernel(root, c, fragment)));
    EXPECT_EQ(fragment.armCount, 2u);
    EXPECT_FALSE(fragment.hasOutputInit);
    EXPECT_EQ(fragment.rhsOperandTypes.size(), 2u);
  } // source and registry are gone: callback must own the math mode.
  auto type = parseType("!micro.tile<8x32xf32, memory = #micro.memory<acc>, "
                        "owner = #micro.owner<worker>>",
                        &ctx);
  ASSERT_TRUE(type);
  Block block;
  OpBuilder b(&ctx);
  b.setInsertionPointToEnd(&block);
  auto a = block.addArgument(type, b.getUnknownLoc());
  auto u = block.addArgument(type, b.getUnknownLoc());
  Value result =
      fragment.buildEpilogue(b, b.getUnknownLoc(), ArrayRef<Value>{a, u});
  ASSERT_TRUE(result);
  EXPECT_EQ(result.getDefiningOp()->getName().getStringRef(), "micro.vector");
  EXPECT_EQ(result.getDefiningOp()->getAttrOfType<StringAttr>("op").getValue(),
            "mul");
  auto *gate = result.getDefiningOp()->getOperand(0).getDefiningOp();
  EXPECT_EQ(gate->getAttrOfType<StringAttr>("op").getValue(), "silu");
  EXPECT_EQ(gate->getAttrOfType<StringAttr>("math_mode").getValue(),
            "bounded_fast");
}

TEST(KernelContractTest, LinalgMatmulExplicitAccumulatorAndInit) {
  MLIRContext ctx;
  ctx.getOrLoadDialect<linalg::LinalgDialect>();
  OpBuilder b(&ctx);
  Block block;
  b.setInsertionPointToEnd(&block);
  auto loc = b.getUnknownLoc();
  auto a =
      block.addArgument(RankedTensorType::get({2, 4}, b.getBF16Type()), loc);
  auto rhs =
      block.addArgument(RankedTensorType::get({4, 8}, b.getBF16Type()), loc);
  auto init =
      block.addArgument(RankedTensorType::get({2, 8}, b.getBF16Type()), loc);
  auto op =
      linalg::MatmulOp::create(b, loc, ValueRange{a, rhs}, ValueRange{init});
  op->setAttr("accumulator_type", TypeAttr::get(b.getF32Type()));
  llk::LLKKernelRegistry r;
  llk::populateLLKKernelLowerings(r);
  auto *entry = r.lookup("linalg.matmul");
  ASSERT_NE(entry, nullptr);
  llk::KernelContract c;
  ASSERT_TRUE(succeeded(entry->inferContract(op, c)));
  EXPECT_EQ(c.accumulatorElemType, b.getF32Type());
  llk::MicroKernelFragment f;
  ASSERT_TRUE(succeeded(entry->describeMicroKernel(op, c, f)));
  EXPECT_TRUE(f.hasOutputInit);
  EXPECT_EQ(f.armCount, 1u);
  EXPECT_EQ(f.rhsOperandTypes.front(), rhs.getType());
}

namespace {
// Execute the emitted scalar IR using each operation's real constant folder.
// This keeps the numerical regression portable and does not duplicate casts
// or floating-point rounding in a test-only arithmetic implementation.
Attribute foldScalarBody(Block &body, ArrayRef<Attribute> arguments) {
  llvm::DenseMap<Value, Attribute> values;
  for (auto [arg, value] : llvm::zip(body.getArguments(), arguments))
    values[arg] = value;
  for (Operation &op : body.without_terminator()) {
    SmallVector<Attribute> operands;
    for (Value v : op.getOperands())
      operands.push_back(values.lookup(v));
    SmallVector<OpFoldResult> results;
    if (failed(op.fold(operands, results)) ||
        results.size() != op.getNumResults()) {
      llvm::errs() << "Cannot fold " << op << "\n";
      return {};
    }
    for (auto [result, folded] : llvm::zip(op.getResults(), results)) {
      auto attr = dyn_cast<Attribute>(folded);
      if (!attr)
        attr = values.lookup(cast<Value>(folded));
      if (!attr)
        return {};
      values[result] = attr;
    }
  }
  return values.lookup(body.getTerminator()->getOperand(0));
}
} // namespace
TEST(KernelContractTest, MatmulF32AccumulationPreservesNonzeroBF16Init) {
  MLIRContext ctx;
  ctx.getOrLoadDialect<llk::LLKDialect>();
  ctx.getOrLoadDialect<func::FuncDialect>();
  ctx.getOrLoadDialect<linalg::LinalgDialect>();
  ctx.getOrLoadDialect<arith::ArithDialect>();
  ctx.getOrLoadDialect<tensor::TensorDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    func.func @matmul(%a: tensor<1x2xbf16>, %b: tensor<2x1xbf16>, %init: tensor<1x1xbf16>) -> tensor<1x1xbf16> {
      %y = llk.matmul ins(%a,%b : tensor<1x2xbf16>,tensor<2x1xbf16>) outs(%init : tensor<1x1xbf16>)
        {accumulator_type=f32, math_mode=#llk.math_mode<strict>} -> tensor<1x1xbf16>
      return %y : tensor<1x1xbf16>
    }
  )mlir",
                                            &ctx);
  ASSERT_TRUE(module);
  Operation *root = nullptr;
  module->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "llk.matmul")
      root = op;
  });
  ASSERT_NE(root, nullptr);
  llk::LLKKernelRegistry r;
  llk::populateLLKKernelLowerings(r);
  OpBuilder b(root);
  ASSERT_TRUE(succeeded(r.lookup("llk.matmul")->lowerToLinalg(root, b)));
  ASSERT_TRUE(succeeded(verify(*module)));
  linalg::MatmulOp matmul;
  module->walk([&](linalg::MatmulOp op) { matmul = op; });
  ASSERT_TRUE(matmul);
  Attribute acc = b.getFloatAttr(b.getBF16Type(), 256.0);
  if (auto castInit = dyn_cast_or_null<linalg::GenericOp>(
          matmul->getOperand(2).getDefiningOp())) {
    auto outElem =
        cast<ShapedType>(castInit.getResult(0).getType()).getElementType();
    acc = foldScalarBody(castInit.getRegion().front(),
                         {acc, b.getFloatAttr(outElem, 0)});
  }
  ASSERT_TRUE(acc);
  auto one = b.getFloatAttr(b.getBF16Type(), 1.0);
  for (int k = 0; k < 2; ++k) {
    acc = foldScalarBody(matmul.getRegion().front(), {one, one, acc});
    ASSERT_TRUE(acc);
  }
  auto function = *module->getOps<func::FuncOp>().begin();
  Value output = function.getBody().front().getTerminator()->getOperand(0);
  if (auto castOut =
          dyn_cast_or_null<linalg::GenericOp>(output.getDefiningOp()))
    acc = foldScalarBody(castOut.getRegion().front(),
                         {acc, b.getFloatAttr(b.getBF16Type(), 0)});
  ASSERT_TRUE(acc);
  EXPECT_DOUBLE_EQ(cast<FloatAttr>(acc).getValueAsDouble(), 258.0);
}
