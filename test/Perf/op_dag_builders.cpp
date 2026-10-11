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
