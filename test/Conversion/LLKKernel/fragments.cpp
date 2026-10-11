#include "LLK/Conversion/LLKKernel/LLKKernelLowering.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"
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
