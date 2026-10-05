//===- tile_facts.cpp - Derived tile bytes and alignment ------------------===//

#include "LLK/Mapping/StoragePlan.h"
#include "LLK/Mapping/TileFacts.h"

#include "LLK/Dialect/Micro/MicroDialect.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>

using namespace mlir::llk::mapping;

namespace {

/// A `!micro.tile<inner>` type, parsed through the loaded Micro dialect so the
/// test exercises the same printed form real workloads carry.
mlir::Type tileType(mlir::MLIRContext &context, llvm::StringRef inner) {
  context.getOrLoadDialect<mlir::micro::MicroDialect>();
  std::string text = "!micro.tile<" + inner.str() + ">";
  return mlir::parseType(text, &context);
}

/// A `tensor<8x32xelement>`. The Micro dialect admits only f32/f16/bf16/i32/i8
/// tile elements, so a 64-bit element -- which the derivation must still size
/// correctly -- is exercised through a modelled tensor; the width path is the
/// same once the head is unwrapped.
mlir::Type tensorType(mlir::MLIRContext &context) {
  return mlir::RankedTensorType::get({8, 32}, mlir::Float64Type::get(&context));
}

} // namespace

TEST(TileFacts, DerivesBytesFromTheTileShapeAndElementWidth) {
  mlir::MLIRContext context;
  // 8 * 32 * 4 bytes = 1024 for f32...
  EXPECT_EQ(tileFactsFor(tileType(context, "8x32xf32")).bytes, 1024u);
  // ...half of it when the element halves to bf16...
  EXPECT_EQ(tileFactsFor(tileType(context, "8x32xbf16")).bytes, 512u);
  // ...and double for a 64-bit element.
  EXPECT_EQ(tileFactsFor(tensorType(context)).bytes, 2048u);
}

TEST(TileFacts, AlignmentFollowsTheElementWidth) {
  mlir::MLIRContext context;
  EXPECT_EQ(tileFactsFor(tileType(context, "8x32xf32")).alignment, 4u);
  EXPECT_EQ(tileFactsFor(tileType(context, "8x32xbf16")).alignment, 2u);
  EXPECT_EQ(tileFactsFor(tensorType(context)).alignment, 8u);
  EXPECT_TRUE(tileFactsFor(tileType(context, "8x32xf32")).known);
}

TEST(TileFacts, ADynamicDimensionIsReportedNotGuessed) {
  mlir::MLIRContext context;
  // A dynamic dimension in either a modelled tensor or a `!micro.tile` head.
  mlir::Type dynamicTensor = mlir::RankedTensorType::get(
      {mlir::ShapedType::kDynamic, 8}, mlir::Float32Type::get(&context));
  EXPECT_FALSE(tileFactsFor(dynamicTensor).known);
  mlir::Type dynamicTile = tileType(context, "8x?xf32");
  ASSERT_TRUE(static_cast<bool>(dynamicTile));
  EXPECT_FALSE(tileFactsFor(dynamicTile).known);
}

TEST(TileFacts, AnUnrecognisedTypeIsReportedNotGuessed) {
  mlir::MLIRContext context;
  EXPECT_FALSE(tileFactsFor(mlir::Type()).known);
  EXPECT_FALSE(tileFactsFor(mlir::NoneType::get(&context)).known);
  EXPECT_FALSE(tileFactsFor(mlir::UnrankedTensorType::get(
                                mlir::Float32Type::get(&context)))
                   .known);
}

TEST(TileFacts, HandlesUnitAndRankZeroShapes) {
  mlir::MLIRContext context;
  // A single-element tile is 1 * 1 * 4 bytes.
  TileFacts unit = tileFactsFor(tileType(context, "1x1xf32"));
  EXPECT_TRUE(unit.known);
  EXPECT_EQ(unit.bytes, 4u);
  EXPECT_EQ(unit.alignment, 4u);
  // A unit extent in only one dimension.
  EXPECT_EQ(tileFactsFor(tileType(context, "1x32xf32")).bytes, 128u);
  // A rank-0 (scalar) shaped value holds exactly one element.
  TileFacts scalar = tileFactsFor(
      mlir::RankedTensorType::get({}, mlir::Float32Type::get(&context)));
  EXPECT_TRUE(scalar.known);
  EXPECT_EQ(scalar.bytes, 4u);
  EXPECT_EQ(scalar.alignment, 4u);
}

TEST(TileFacts, RejectsANonRepresentableByteCount) {
  mlir::MLIRContext context;
  // 2^31 * 2^31 elements of f32 is 2^64 bytes, which is not representable in
  // uint64_t. The count is reported unknown rather than wrapping to a small,
  // wrong size.
  mlir::Type huge = mlir::RankedTensorType::get(
      {int64_t{1} << 31, int64_t{1} << 31}, mlir::Float32Type::get(&context));
  EXPECT_FALSE(tileFactsFor(huge).known);
}

//===----------------------------------------------------------------------===//
// Physical footprint under a layout map
//===----------------------------------------------------------------------===//

namespace {

/// A layout map over a rank-2 index space: `(m, n) -> ...`.
mlir::AffineMap map2(mlir::MLIRContext &context, unsigned dims,
                     llvm::ArrayRef<mlir::AffineExpr> results) {
  return mlir::AffineMap::get(dims, 0, results, &context);
}

} // namespace

TEST(StorageFootprint, PlainValueIsItsLogicalImage) {
  mlir::MLIRContext context;
  // 8x8xf32 = 64 elements * 4 bytes = 256, no map, no padding.
  auto footprint =
      physicalFootprintFor(tileType(context, "8x8xf32"), mlir::AffineMap(), {});
  ASSERT_TRUE(bool(footprint)) << llvm::toString(footprint.takeError());
  EXPECT_EQ(footprint->bytes, 256u);
  EXPECT_TRUE(footprint->known);
}

TEST(StorageFootprint, ABlockedMapWithARaggedExtentPadsTheImage) {
  mlir::MLIRContext context;
  // `(m, n) -> (m, floordiv(n, 4), mod(n, 4))` over an 8x6 index space has
  // physical extents (8, 2, 4) = 64 elements = 256 bytes, larger than the
  // 8*6*4 = 192 logical bytes, because the second dimension rounds up to a
  // whole block.
  mlir::AffineExpr m = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr n = mlir::getAffineDimExpr(1, &context);
  mlir::AffineExpr vw = mlir::getAffineConstantExpr(4, &context);
  mlir::AffineMap blocked = map2(context, 2, {m, n.floorDiv(vw), n % vw});
  auto footprint =
      physicalFootprintFor(tileType(context, "8x6xf32"), blocked, {});
  ASSERT_TRUE(bool(footprint)) << llvm::toString(footprint.takeError());
  EXPECT_EQ(footprint->bytes, 256u);
}

TEST(StorageFootprint, DeclaredPaddingEnlargesTheImage) {
  mlir::MLIRContext context;
  // One extra element of padding on each physical dimension: (8+1)*(8+1)*4.
  auto footprint = physicalFootprintFor(tileType(context, "8x8xf32"),
                                        mlir::AffineMap(), {1, 1});
  ASSERT_FALSE(bool(footprint));
  // Padding is only meaningful alongside a physical map, so the bare form is
  // rejected rather than silently ignored.
  EXPECT_NE(llvm::toString(footprint.takeError()).find("padding"),
            std::string::npos);

  mlir::AffineExpr m = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr n = mlir::getAffineDimExpr(1, &context);
  mlir::AffineMap identity = map2(context, 2, {m, n});
  auto padded =
      physicalFootprintFor(tileType(context, "8x8xf32"), identity, {1, 1});
  ASSERT_TRUE(bool(padded)) << llvm::toString(padded.takeError());
  EXPECT_EQ(padded->bytes, 9u * 9u * 4u);
}

TEST(StorageFootprint, AnAffineStrideMapReportsItsBoundingImage) {
  mlir::MLIRContext context;
  // `(m, n) -> (m, n * 2)` over 4x4 has physical extents (4, 7) = 28 elements,
  // so the stride's holes are charged, not ignored.
  mlir::AffineExpr m = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr n = mlir::getAffineDimExpr(1, &context);
  mlir::AffineMap strided =
      map2(context, 2, {m, n * mlir::getAffineConstantExpr(2, &context)});
  auto footprint =
      physicalFootprintFor(tileType(context, "4x4xf32"), strided, {});
  ASSERT_TRUE(bool(footprint)) << llvm::toString(footprint.takeError());
  EXPECT_EQ(footprint->bytes, 28u * 4u);
}

TEST(StorageFootprint, ADynamicValueIsRejectedNotZeros) {
  mlir::MLIRContext context;
  mlir::Type dynamic = mlir::RankedTensorType::get(
      {mlir::ShapedType::kDynamic, 8}, mlir::Float32Type::get(&context));
  auto footprint = physicalFootprintFor(dynamic, mlir::AffineMap(), {});
  ASSERT_FALSE(bool(footprint));
  EXPECT_NE(llvm::toString(footprint.takeError()).find("unsupported footprint"),
            std::string::npos);
}

TEST(StorageFootprint, AMismatchedMapRankIsRejected) {
  mlir::MLIRContext context;
  mlir::AffineExpr m = mlir::getAffineDimExpr(0, &context);
  // A rank-1 map applied to a rank-2 value states no index relation.
  mlir::AffineMap oneDim = map2(context, 1, {m});
  auto footprint =
      physicalFootprintFor(tileType(context, "8x8xf32"), oneDim, {});
  EXPECT_FALSE(bool(footprint));
}

TEST(StorageFootprint, AnOverflowingImageIsRejected) {
  mlir::MLIRContext context;
  mlir::AffineExpr m = mlir::getAffineDimExpr(0, &context);
  mlir::AffineExpr n = mlir::getAffineDimExpr(1, &context);
  // A stride so large the bounding image's byte count overflows uint64:
  // 8 * (7 * 2^57 + 1) * 4 bytes exceeds 2^64.
  mlir::AffineMap huge =
      map2(context, 2,
           {m, n * mlir::getAffineConstantExpr(int64_t{1} << 57, &context)});
  auto footprint = physicalFootprintFor(tileType(context, "8x8xf32"), huge, {});
  EXPECT_FALSE(bool(footprint));
}
