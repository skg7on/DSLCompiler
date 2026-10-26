//===- dynamic_shape_contract.cpp - Symbolic shape contract tests ---------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Behavioral tests for symbolic shape data and the versioned workload
// contract. A `Dimension` is a constant or a symbol; reusing a symbol asserts
// equality. Binding checks every port even when another extent is zero, so an
// empty invocation cannot hide a contradiction.
//
// The version-1 golden rendering below is the byte-exact baseline captured
// from the pre-symbolic serializer. Static artifacts keep that identity.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/LLKKernel/LLKKernelLowering.h"
#include "LLK/Conversion/Workload/WorkloadContract.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Shape/ShapeContract.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

#include "gtest/gtest.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#include <initializer_list>
#include <optional>
#include <string>
#include <vector>

namespace shape = mlir::llk::shape;
namespace workload = mlir::llk::workload;

namespace {

/// A fully dynamic SwiGLU root: the three contraction axes are unresolved.
constexpr llvm::StringLiteral kAllDynamic = R"mlir(
func.func @dynamic(%x: tensor<?x?xbf16>, %wg: tensor<?x?xbf16>,
                   %wu: tensor<?x?xbf16>, %init: tensor<?x?xbf16>)
    -> tensor<?x?xbf16> {
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<?x?xbf16>,
                            tensor<?x?xbf16>, tensor<?x?xbf16>)
      outs(%init : tensor<?x?xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>,
       math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<?x?xbf16>
  return %y : tensor<?x?xbf16>
}
)mlir";

/// Only K is unresolved; M and N are pinned by both the ports and the result.
constexpr llvm::StringLiteral kMixedStatic = R"mlir(
func.func @mixed(%x: tensor<7x?xbf16>, %wg: tensor<?x13xbf16>,
                 %wu: tensor<?x13xbf16>, %init: tensor<7x13xbf16>)
    -> tensor<7x13xbf16> {
  %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<7x?xbf16>,
                            tensor<?x13xbf16>, tensor<?x13xbf16>)
      outs(%init : tensor<7x13xbf16>)
      {accumulator_type = f32, activation = #llk.activation<silu>,
       math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<7x13xbf16>
  return %y : tensor<7x13xbf16>
}
)mlir";

/// The exact version-1 rendering the pre-symbolic serializer produced for the
/// static SwiGLU contract used by Version1IdentityStable.
constexpr llvm::StringLiteral kVersion1Golden =
    "version=1;ports=4;name=3:lhs;direction=5:input;effect=4:read;rank=2;"
    "extent=7;extent=5;element_type=4:bf16;name=4:rhs0;direction=5:input;"
    "effect=4:read;rank=2;extent=5;extent=13;element_type=4:bf16;name=4:rhs1;"
    "direction=5:input;effect=4:read;rank=2;extent=5;extent=13;"
    "element_type=4:bf16;name=6:output;direction=6:output;effect=5:write;"
    "rank=2;extent=7;extent=13;element_type=4:bf16;axes=3;axis=1:K;extent=5;"
    "axis=1:M;extent=7;axis=1:N;extent=13;numerical_contract=10:strict-f32;"
    "primitives=4;primitive=9:micro.mma;primitive=21:micro.tile_async_copy;"
    "primitive=16:micro.tile_store;primitive=12:micro.vector;";

/// Parses a SwiGLU module and resolves its contraction axes through the
/// operator-kernel registry, which is the producer shared code consumes.
mlir::llk::ContractionDimensions inferSwiGLU(llvm::StringRef text) {
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::llk::LLKDialect>();
  context.getOrLoadDialect<mlir::micro::MicroDialect>();
  context.getOrLoadDialect<mlir::func::FuncDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, &context);
  EXPECT_TRUE(bool(module));
  if (!module)
    return {};
  mlir::Operation *root = nullptr;
  module->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "llk.fused_swiglu")
      root = op;
  });
  EXPECT_NE(root, nullptr);
  if (!root)
    return {};
  auto dims = mlir::llk::inferContractionDimensions(root);
  EXPECT_FALSE(mlir::failed(dims));
  if (mlir::failed(dims))
    return {};
  return *dims;
}

/// The four-port SwiGLU shape contract implied by resolved dimensions. A
/// symbolic axis is declared with the default nonnegative constraint.
shape::ShapeContract
swigluContract(const mlir::llk::ContractionDimensions &dims) {
  shape::ShapeContract contract;
  contract.ports = {{"X", shape::PortDirection::Input, {dims.M, dims.K}},
                    {"Wg", shape::PortDirection::Input, {dims.K, dims.N}},
                    {"Wu", shape::PortDirection::Input, {dims.K, dims.N}},
                    {"out", shape::PortDirection::Output, {dims.M, dims.N}}};
  for (const shape::Dimension &dim : {dims.M, dims.N, dims.K})
    if (dim.isSymbol())
      contract.symbols.emplace(dim.getSymbol().str(),
                               shape::SymbolConstraint{});
  return contract;
}

std::vector<shape::PortShapeValues> swigluValues(std::vector<int64_t> x,
                                                 std::vector<int64_t> wg,
                                                 std::vector<int64_t> wu,
                                                 std::vector<int64_t> out) {
  return {{shape::PortDirection::Input, 0, std::move(x)},
          {shape::PortDirection::Input, 1, std::move(wg)},
          {shape::PortDirection::Input, 2, std::move(wu)},
          {shape::PortDirection::Output, 0, std::move(out)}};
}

std::string errorText(llvm::Error error) {
  return llvm::toString(std::move(error));
}

llvm::Expected<workload::WorkloadContract>
roundTrip(const workload::WorkloadContract &contract) {
  mlir::MLIRContext context;
  auto encoded = workload::encodeWorkloadContract(&context, contract);
  if (!encoded)
    return encoded.takeError();
  return workload::decodeWorkloadContract(*encoded);
}

/// The static contract whose rendering is the version-1 golden.
workload::WorkloadContract staticSwiGLUContract() {
  workload::WorkloadContract contract;
  contract.version = 1;
  contract.numericalContract = "strict-f32";
  contract.ports = {{"lhs",
                     workload::PortDirection::Input,
                     workload::PortEffect::Read,
                     {7, 5},
                     "bf16"},
                    {"rhs0",
                     workload::PortDirection::Input,
                     workload::PortEffect::Read,
                     {5, 13},
                     "bf16"},
                    {"rhs1",
                     workload::PortDirection::Input,
                     workload::PortEffect::Read,
                     {5, 13},
                     "bf16"},
                    {"output",
                     workload::PortDirection::Output,
                     workload::PortEffect::Write,
                     {7, 13},
                     "bf16"}};
  contract.axes = {{"K", 5}, {"M", 7}, {"N", 13}};
  contract.requiredPrimitives = {"micro.mma", "micro.vector",
                                 "micro.tile_async_copy", "micro.tile_store"};
  return contract;
}

} // namespace

TEST(ShapeContract, AllDynamicMNK) {
  auto dims = inferSwiGLU(kAllDynamic);
  EXPECT_EQ(dims.M, shape::Dimension::symbol("M"));
  EXPECT_EQ(dims.N, shape::Dimension::symbol("N"));
  EXPECT_EQ(dims.K, shape::Dimension::symbol("K"));

  shape::ShapeContract contract = swigluContract(dims);
  ASSERT_FALSE(static_cast<bool>(shape::validateShapeContract(contract)));

  auto bound = shape::bindShapeValues(
      contract, swigluValues({7, 5}, {5, 13}, {5, 13}, {7, 13}));
  ASSERT_TRUE(bool(bound)) << errorText(bound.takeError());
  EXPECT_EQ(bound->at("M"), 7);
  EXPECT_EQ(bound->at("N"), 13);
  EXPECT_EQ(bound->at("K"), 5);
  EXPECT_EQ(bound->size(), 3u);

  // Wu's reduction extent must agree with X and Wg.
  auto conflict = shape::bindShapeValues(
      contract, swigluValues({7, 5}, {5, 13}, {6, 13}, {7, 13}));
  ASSERT_FALSE(bool(conflict));
  EXPECT_NE(errorText(conflict.takeError()).find("symbol_conflict"),
            std::string::npos);
}

TEST(ShapeContract, MixedStaticEquality) {
  auto dims = inferSwiGLU(kMixedStatic);
  EXPECT_EQ(dims.M, shape::Dimension(7));
  EXPECT_EQ(dims.N, shape::Dimension(13));
  EXPECT_EQ(dims.K, shape::Dimension::symbol("K"));

  shape::ShapeContract contract = swigluContract(dims);
  auto bound = shape::bindShapeValues(
      contract, swigluValues({7, 5}, {5, 13}, {5, 13}, {7, 13}));
  ASSERT_TRUE(bool(bound)) << errorText(bound.takeError());
  EXPECT_EQ(bound->at("K"), 5);

  // A static axis still rejects a disagreeing port extent.
  auto wrongM = shape::bindShapeValues(
      contract, swigluValues({6, 5}, {5, 13}, {5, 13}, {7, 13}));
  ASSERT_FALSE(bool(wrongM));
  EXPECT_NE(errorText(wrongM.takeError()).find("constant_mismatch"),
            std::string::npos);
}

TEST(ShapeContract, RejectConflicts) {
  shape::ShapeContract contract = swigluContract(inferSwiGLU(kAllDynamic));

  // The same symbol supplied different values by two ports.
  auto conflict = shape::bindShapeValues(
      contract, swigluValues({7, 5}, {5, 13}, {5, 13}, {8, 13}));
  ASSERT_FALSE(bool(conflict));
  EXPECT_NE(errorText(conflict.takeError()).find("symbol_conflict"),
            std::string::npos);

  // Two static extents that disagree are a compile-time contradiction, not a
  // relationship to guard at runtime.
  mlir::MLIRContext context;
  mlir::OperationState state(mlir::UnknownLoc::get(&context), "test.contract");
  mlir::Operation *root = mlir::Operation::create(state);
  mlir::ScopedDiagnosticHandler handler(
      &context, [](mlir::Diagnostic &) { return mlir::success(); });
  auto dim = mlir::llk::resolveContractionDimension(root, "M", {7, 8});
  EXPECT_TRUE(mlir::failed(dim));
  root->destroy();
}

TEST(ShapeContract, EmptyStillChecksK) {
  shape::ShapeContract contract = swigluContract(inferSwiGLU(kAllDynamic));

  // M = 0 makes the output empty, but K still has to agree.
  auto empty = shape::bindShapeValues(
      contract, swigluValues({0, 5}, {5, 13}, {5, 13}, {0, 13}));
  ASSERT_TRUE(bool(empty)) << errorText(empty.takeError());
  EXPECT_EQ(empty->at("M"), 0);
  EXPECT_EQ(empty->at("N"), 13);
  EXPECT_EQ(empty->at("K"), 5);

  auto emptyConflict = shape::bindShapeValues(
      contract, swigluValues({0, 5}, {5, 13}, {6, 13}, {0, 13}));
  ASSERT_FALSE(bool(emptyConflict));
  EXPECT_NE(errorText(emptyConflict.takeError()).find("symbol_conflict"),
            std::string::npos);
}

TEST(ShapeContract, RejectNegative) {
  shape::ShapeContract contract = swigluContract(inferSwiGLU(kAllDynamic));

  shape::ShapeContract negative = contract;
  negative.ports[0].shape[0] = shape::Dimension(-1);
  auto invalid = shape::validateShapeContract(negative);
  ASSERT_TRUE(bool(invalid));
  EXPECT_NE(errorText(std::move(invalid)).find("negative_extent"),
            std::string::npos);

  // A concrete extent is rejected before it can reach shape arithmetic.
  auto badValue = shape::bindShapeValues(
      contract, swigluValues({7, -5}, {5, 13}, {5, 13}, {7, 13}));
  ASSERT_FALSE(bool(badValue));
  EXPECT_NE(errorText(badValue.takeError()).find("negative_value"),
            std::string::npos);

  // A declaration that cannot be satisfied is rejected.
  shape::ShapeContract badBounds = contract;
  badBounds.symbols["K"].min = -3;
  auto boundsError = shape::validateShapeContract(badBounds);
  ASSERT_TRUE(bool(boundsError));
  EXPECT_NE(errorText(std::move(boundsError)).find("negative_minimum"),
            std::string::npos);
}

TEST(ShapeContract, Version1IdentityStable) {
  workload::WorkloadContract contract = staticSwiGLUContract();
  ASSERT_FALSE(static_cast<bool>(workload::validateWorkloadContract(contract)));
  EXPECT_EQ(workload::canonicalWorkloadContract(contract), kVersion1Golden);

  auto restored = roundTrip(contract);
  ASSERT_TRUE(bool(restored)) << errorText(restored.takeError());
  EXPECT_EQ(workload::canonicalWorkloadContract(*restored), kVersion1Golden);

  // Version 1 has no symbolic or zero-extent encoding.
  workload::WorkloadContract symbolic = contract;
  symbolic.ports[0].shape[0] = shape::Dimension::symbol("M");
  EXPECT_TRUE(bool(workload::validateWorkloadContract(symbolic)));
  mlir::MLIRContext context;
  EXPECT_FALSE(bool(workload::encodeWorkloadContract(&context, symbolic)));

  workload::WorkloadContract zero = contract;
  zero.ports[0].shape[0] = 0;
  EXPECT_TRUE(bool(workload::validateWorkloadContract(zero)));
}

TEST(ShapeContract, Version2RoundTrip) {
  workload::WorkloadContract contract;
  contract.version = 2;
  contract.numericalContract = "strict-f32";
  contract.ports = {
      {"X",
       workload::PortDirection::Input,
       workload::PortEffect::Read,
       {shape::Dimension::symbol("M"), shape::Dimension::symbol("K")},
       "bf16"},
      {"Wg",
       workload::PortDirection::Input,
       workload::PortEffect::Read,
       {shape::Dimension::symbol("K"), shape::Dimension::symbol("N")},
       "bf16"},
      {"Wu",
       workload::PortDirection::Input,
       workload::PortEffect::Read,
       {shape::Dimension::symbol("K"), shape::Dimension::symbol("N")},
       "bf16"},
      {"out",
       workload::PortDirection::Output,
       workload::PortEffect::Write,
       {shape::Dimension::symbol("M"), shape::Dimension::symbol("N")},
       "bf16"}};
  contract.axes = {{"M", shape::Dimension::symbol("M")},
                   {"N", shape::Dimension::symbol("N")},
                   {"K", shape::Dimension::symbol("K")}};
  contract.symbols = {{"K", shape::SymbolConstraint{1, 4096, 2}},
                      {"M", shape::SymbolConstraint{}},
                      {"N", shape::SymbolConstraint{}}};
  contract.requiredPrimitives = {"micro.mma"};

  ASSERT_FALSE(static_cast<bool>(workload::validateWorkloadContract(contract)));
  const std::string canonical = workload::canonicalWorkloadContract(contract);

  auto restored = roundTrip(contract);
  ASSERT_TRUE(bool(restored)) << errorText(restored.takeError());
  EXPECT_EQ(workload::canonicalWorkloadContract(*restored), canonical);
  ASSERT_EQ(restored->symbols.size(), 3u);
  EXPECT_EQ(restored->symbols.at("K"), shape::SymbolConstraint({1, 4096, 2}));
  EXPECT_EQ(restored->symbols.at("M"), shape::SymbolConstraint{});
  EXPECT_EQ(restored->axes.at("K"), shape::Dimension::symbol("K"));

  // A symbolic artifact has a distinct identity from its static counterpart.
  EXPECT_NE(canonical,
            workload::canonicalWorkloadContract(staticSwiGLUContract()));

  // A declared but unused symbol is rejected, as is an undeclared one.
  workload::WorkloadContract unused = contract;
  unused.symbols["unused"] = shape::SymbolConstraint{};
  EXPECT_TRUE(bool(workload::validateWorkloadContract(unused)));
  workload::WorkloadContract undeclared = contract;
  undeclared.symbols.erase("K");
  EXPECT_TRUE(bool(workload::validateWorkloadContract(undeclared)));
}

TEST(ShapeContract, ElementwiseAxes) {
  // A non-contraction family relates the same rank-two axes across its ports.
  shape::ShapeContract contract;
  contract.ports = {
      {"in0",
       shape::PortDirection::Input,
       {shape::Dimension::symbol("d0"), shape::Dimension::symbol("d1")}},
      {"in1",
       shape::PortDirection::Input,
       {shape::Dimension::symbol("d0"), shape::Dimension::symbol("d1")}},
      {"out",
       shape::PortDirection::Output,
       {shape::Dimension::symbol("d0"), shape::Dimension::symbol("d1")}}};
  contract.symbols = {{"d0", shape::SymbolConstraint{}},
                      {"d1", shape::SymbolConstraint{}}};

  auto bound = shape::bindShapeValues(
      contract, {{shape::PortDirection::Input, 0, {2, 8}},
                 {shape::PortDirection::Input, 1, {2, 8}},
                 {shape::PortDirection::Output, 0, {2, 8}}});
  ASSERT_TRUE(bool(bound)) << errorText(bound.takeError());
  EXPECT_EQ(bound->at("d0"), 2);
  EXPECT_EQ(bound->at("d1"), 8);

  auto mismatch = shape::bindShapeValues(
      contract, {{shape::PortDirection::Input, 0, {2, 8}},
                 {shape::PortDirection::Input, 1, {2, 7}},
                 {shape::PortDirection::Output, 0, {2, 8}}});
  ASSERT_FALSE(bool(mismatch));
  EXPECT_NE(errorText(mismatch.takeError()).find("symbol_conflict"),
            std::string::npos);

  // The workload contract carries the same axes and keeps them symbolic.
  workload::WorkloadContract elementwise;
  elementwise.version = 2;
  elementwise.numericalContract = "strict-f32";
  elementwise.ports = {
      {"lhs",
       workload::PortDirection::Input,
       workload::PortEffect::Read,
       {shape::Dimension::symbol("d0"), shape::Dimension::symbol("d1")},
       "f32"},
      {"rhs",
       workload::PortDirection::Input,
       workload::PortEffect::Read,
       {shape::Dimension::symbol("d0"), shape::Dimension::symbol("d1")},
       "f32"},
      {"out",
       workload::PortDirection::Output,
       workload::PortEffect::Write,
       {shape::Dimension::symbol("d0"), shape::Dimension::symbol("d1")},
       "f32"}};
  elementwise.axes = {{"d0", shape::Dimension::symbol("d0")},
                      {"d1", shape::Dimension::symbol("d1")}};
  elementwise.symbols = {{"d0", shape::SymbolConstraint{}},
                         {"d1", shape::SymbolConstraint{}}};
  ASSERT_FALSE(
      static_cast<bool>(workload::validateWorkloadContract(elementwise)));
  auto elementwiseRestored = roundTrip(elementwise);
  ASSERT_TRUE(bool(elementwiseRestored))
      << errorText(elementwiseRestored.takeError());
  EXPECT_EQ(workload::canonicalWorkloadContract(*elementwiseRestored),
            workload::canonicalWorkloadContract(elementwise));
}
