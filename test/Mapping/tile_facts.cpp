//===- tile_facts.cpp - Derived tile bytes and alignment ------------------===//

#include "LLK/Mapping/TileFacts.h"

#include "LLK/Dialect/Micro/MicroDialect.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include <gtest/gtest.h>

#include <cstdint>

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
