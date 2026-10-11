#include "LLK/Perf/TileInfo.h"
#include "mlir/IR/MLIRContext.h"
#include "gtest/gtest.h"

namespace mlir::llk::perf {
TEST(OpDAGBuilders, PublicTileDescriptionRetainsTensorSize) {
  mlir::MLIRContext context;
  auto type =
      mlir::RankedTensorType::get({8, 16}, mlir::Float32Type::get(&context));
  TileInfo info = describeType(type);
  EXPECT_FALSE(info.isTile);
  EXPECT_EQ(info.dtype, "f32");
  EXPECT_EQ(info.elements(), 128u);
  EXPECT_EQ(info.bytes(), 512u);
  EXPECT_EQ(shapeString(info.shape), "8x16");
}
TEST(OpDAGBuilders, UnsizableTypesRetainZeroBytes) {
  mlir::MLIRContext context;
  auto dynamic = describeType(mlir::RankedTensorType::get(
      {mlir::ShapedType::kDynamic, 16}, mlir::Float32Type::get(&context)));
  EXPECT_TRUE(dynamic.hasDynamicShape());
  EXPECT_EQ(dynamic.bytes(), 0u);
  EXPECT_EQ(shapeString(dynamic.shape), "?x16");
  auto unsupported = describeType(
      mlir::RankedTensorType::get({8, 16}, mlir::Float64Type::get(&context)));
  EXPECT_EQ(unsupported.elements(), 128u);
  EXPECT_EQ(unsupported.bytes(), 0u);
}
} // namespace mlir::llk::perf

#include "LLK/Perf/OpDAGBuilder.h"
namespace mlir::llk::perf {
namespace {
auto noEvent = [](mlir::Operation &, OpDAGBuildContext &) {
  return llvm::Error::success();
};
void expectError(llvm::Error error, llvm::StringRef fragment) {
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error)).find(fragment.str()),
            std::string::npos);
}
} // namespace
TEST(OpDAGBuilders, DuplicateRegistrationAndAbsentReplacementFail) {
  OpDAGBuilderRegistry builders;
  ASSERT_FALSE(builders.registerBuilder("test.probe", noEvent));
  expectError(builders.registerBuilder("test.probe", noEvent), "test.probe");
  expectError(builders.replaceBuilder("test.absent", noEvent), "test.absent");
  EXPECT_NE(builders.lookup("test.probe"), nullptr);
  EXPECT_EQ(builders.lookup("test.absent"), nullptr);
}
TEST(OpDAGBuilders, RejectsEmptyNamesCallbacksAndStructuralOperations) {
  OpDAGBuilderRegistry builders;
  expectError(builders.registerBuilder("", noEvent), "empty");
  expectError(builders.registerBuilder("test.empty", {}), "callback");
  ASSERT_FALSE(builders.registerBuilder("test.probe", noEvent));
  expectError(builders.replaceBuilder("test.probe", {}), "callback");
  for (llvm::StringRef name :
       {"micro.for", "micro.spatial_for", "micro.pipeline", "micro.yield"}) {
    expectError(builders.registerBuilder(name, noEvent), "structural");
    expectError(builders.replaceBuilder(name, noEvent), "structural");
  }
}
} // namespace mlir::llk::perf
