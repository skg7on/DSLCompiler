#include "LLK/Conversion/LLKKernel/LLKKernelLowering.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "gtest/gtest.h"
using namespace mlir;
TEST(KernelContractTest, ShapesBucketsAndMicroValidation) {
  MLIRContext ctx;
  OpBuilder b(&ctx);
  Block block;
  OperationState state(b.getUnknownLoc(), "test.contract");
  Operation *root = Operation::create(state);
  ScopedDiagnosticHandler handler(&ctx, [](Diagnostic &) { return success(); });
  for (auto [m, bucket] : {std::pair<int64_t, int>{1, 0},
                           {2, 1},
                           {4, 1},
                           {5, 2},
                           {16, 2},
                           {17, 3},
                           {64, 3},
                           {65, 4}}) {
    auto lhs = block.addArgument(
        RankedTensorType::get({m, 32}, b.getBF16Type()), b.getUnknownLoc());
    auto rhs = block.addArgument(
        RankedTensorType::get({32, 64}, b.getBF16Type()), b.getUnknownLoc());
    auto out = block.addArgument(
        RankedTensorType::get({m, 64}, b.getBF16Type()), b.getUnknownLoc());
    llk::KernelContract c;
    ASSERT_TRUE(succeeded(llk::inferContractionContract(
        root, lhs, rhs, out, TypeAttr::get(b.getF32Type()), c)));
    EXPECT_EQ(c.M, m);
    EXPECT_EQ(c.N, 64);
    EXPECT_EQ(c.K, 32);
    EXPECT_EQ(c.mBucket, bucket);
    EXPECT_EQ(c.inputElemType, b.getBF16Type());
    EXPECT_EQ(c.accumulatorElemType, b.getF32Type());
    EXPECT_TRUE(succeeded(llk::validateMicroContraction(
        root, lhs, rhs, out, TypeAttr::get(b.getF32Type()), c)));
    EXPECT_TRUE(failed(llk::validateMicroContraction(
        root, lhs, rhs, out, b.getStringAttr("f32"), c)));
    auto wrong = block.addArgument(
        RankedTensorType::get({31, 64}, b.getBF16Type()), b.getUnknownLoc());
    EXPECT_TRUE(failed(llk::validateMicroContraction(
        root, lhs, wrong, out, TypeAttr::get(b.getF32Type()), c)));
  }
  auto lhs = block.addArgument(
      RankedTensorType::get({ShapedType::kDynamic, 32}, b.getBF16Type()),
      b.getUnknownLoc());
  auto rhs = block.addArgument(RankedTensorType::get({32, 64}, b.getBF16Type()),
                               b.getUnknownLoc());
  auto out = block.addArgument(
      RankedTensorType::get({ShapedType::kDynamic, 64}, b.getBF16Type()),
      b.getUnknownLoc());
  llk::KernelContract c;
  EXPECT_TRUE(succeeded(llk::inferContractionContract(
      root, lhs, rhs, out, TypeAttr::get(b.getF32Type()), c)));
  EXPECT_EQ(c.mBucket, -1);
  EXPECT_TRUE(failed(llk::validateMicroContraction(
      root, lhs, rhs, out, TypeAttr::get(b.getF32Type()), c)));
  root->destroy();
}
