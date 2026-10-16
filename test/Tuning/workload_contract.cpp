//===- workload_contract.cpp - Rank-neutral workload contract tests -------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Tests for the versioned rank-neutral workload contract (framework B1). A
// contract describes ordered named ports, optional named axes, dtypes,
// numerical semantics and required primitives -- without inventing the M/N/K
// contraction axes that only some workloads have.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/Workload/WorkloadContract.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include "gtest/gtest.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

using namespace mlir::llk::workload;

namespace {

/// A valid three-input, two-result rank-three contract. It deliberately has no
/// axes and no M/N/K: the contract is rank-neutral.
WorkloadContract makeContract() {
  WorkloadContract contract;
  contract.numericalContract = "strict-f32";
  for (unsigned i = 0; i != 3; ++i)
    contract.ports.push_back({"input" + std::to_string(i),
                              PortDirection::Input,
                              PortEffect::Read,
                              {2, 3, 4},
                              "f32"});
  contract.ports.push_back(
      {"sum", PortDirection::Output, PortEffect::Write, {2, 3, 4}, "f32"});
  contract.ports.push_back(
      {"product", PortDirection::Output, PortEffect::Write, {2, 3, 4}, "f32"});
  contract.requiredPrimitives = {"micro.vector"};
  return contract;
}

/// Encodes and decodes `contract` through the MLIR attribute encoding, failing
/// the test on any encode/decode error. The decoded value is plain data, so it
/// outlives the temporary MLIRContext.
llvm::Expected<WorkloadContract> roundTrip(const WorkloadContract &contract) {
  mlir::MLIRContext context;
  return decodeWorkloadContract(encodeWorkloadContract(&context, contract));
}

/// Serializes the validation error into text for a substring assertion.
std::string errorText(llvm::Error error) {
  return llvm::toString(std::move(error));
}

/// Builds a DictionaryAttr from name/value pairs, sorting by name.
mlir::DictionaryAttr
makeDict(mlir::MLIRContext *context,
         std::initializer_list<std::pair<llvm::StringRef, mlir::Attribute>>
             entries) {
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  for (const auto &[name, value] : entries)
    attrs.push_back(
        mlir::NamedAttribute(mlir::StringAttr::get(context, name), value));
  std::sort(
      attrs.begin(), attrs.end(),
      [](const mlir::NamedAttribute &lhs, const mlir::NamedAttribute &rhs) {
        return lhs.getName().getValue() < rhs.getName().getValue();
      });
  return mlir::DictionaryAttr::get(context, attrs);
}

/// Builds a full single-port contract attribute whose port direction is
/// `direction` -- used to exercise the decode path with a bad direction value.
mlir::DictionaryAttr makeSinglePortContractAttr(mlir::MLIRContext *context,
                                                llvm::StringRef direction) {
  llvm::SmallVector<mlir::Attribute> shape;
  shape.push_back(
      mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), 4));

  llvm::SmallVector<mlir::Attribute> ports;
  ports.push_back(makeDict(
      context, {{"name", mlir::StringAttr::get(context, "in")},
                {"direction", mlir::StringAttr::get(context, direction)},
                {"effect", mlir::StringAttr::get(context, "read")},
                {"shape", mlir::ArrayAttr::get(context, shape)},
                {"element_type", mlir::StringAttr::get(context, "f32")}}));

  llvm::SmallVector<mlir::Attribute> noPrimitives;
  llvm::SmallVector<mlir::NamedAttribute> noAxes;

  return makeDict(
      context,
      {{"version",
        mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), 1)},
       {"ports", mlir::ArrayAttr::get(context, ports)},
       {"axes", mlir::DictionaryAttr::get(context, noAxes)},
       {"numerical_contract", mlir::StringAttr::get(context, "strict-f32")},
       {"required_primitives", mlir::ArrayAttr::get(context, noPrimitives)}});
}

} // namespace

TEST(WorkloadContract, ThreeInputsAndTwoResultsNeedNoContractionAxes) {
  WorkloadContract contract = makeContract();
  EXPECT_FALSE(validateWorkloadContract(contract));
  EXPECT_EQ(contract.axes.count("K"), 0u);

  auto restored = roundTrip(contract);
  ASSERT_TRUE(bool(restored)) << llvm::toString(restored.takeError());
  EXPECT_EQ(restored->axes.count("K"), 0u);
  EXPECT_EQ(canonicalWorkloadContract(*restored),
            canonicalWorkloadContract(contract));

  auto changed = contract;
  std::swap(changed.ports[0], changed.ports[1]);
  EXPECT_NE(canonicalWorkloadContract(changed),
            canonicalWorkloadContract(contract));
}

TEST(WorkloadContract, RoundTripsRankOne) {
  WorkloadContract contract;
  contract.numericalContract = "strict-f32";
  contract.ports = {
      {"value", PortDirection::Input, PortEffect::Read, {4}, "f32"}};

  auto restored = roundTrip(contract);
  ASSERT_TRUE(bool(restored)) << llvm::toString(restored.takeError());
  EXPECT_EQ(canonicalWorkloadContract(*restored),
            canonicalWorkloadContract(contract));
  EXPECT_EQ(restored->ports.size(), 1u);
  EXPECT_EQ(restored->ports[0].shape, std::vector<int64_t>({4}));
  EXPECT_EQ(restored->axes.count("K"), 0u);
}

TEST(WorkloadContract, RoundTripsNamedAxes) {
  WorkloadContract contract;
  contract.numericalContract = "strict-f32";
  contract.ports = {
      {"a", PortDirection::Input, PortEffect::Read, {8, 16}, "f32"},
      {"b", PortDirection::Input, PortEffect::Read, {8, 16}, "f32"},
      {"out", PortDirection::Output, PortEffect::Write, {8, 16}, "f32"}};
  contract.axes = {{"d0", 8}, {"d1", 16}};

  auto restored = roundTrip(contract);
  ASSERT_TRUE(bool(restored)) << llvm::toString(restored.takeError());
  EXPECT_EQ(canonicalWorkloadContract(*restored),
            canonicalWorkloadContract(contract));
  EXPECT_EQ(restored->axes.at("d0"), 8);
  EXPECT_EQ(restored->axes.at("d1"), 16);
  EXPECT_EQ(restored->axes.count("K"), 0u);
}

TEST(WorkloadContract, PortOrderChangesIdentity) {
  WorkloadContract a = makeContract();
  WorkloadContract b = a;
  std::swap(b.ports[0], b.ports[1]);
  EXPECT_NE(canonicalWorkloadContract(a), canonicalWorkloadContract(b));
}

TEST(WorkloadContract, DtypeChangesIdentity) {
  WorkloadContract a = makeContract();
  WorkloadContract b = a;
  b.ports[0].elementType = "f16";
  EXPECT_NE(canonicalWorkloadContract(a), canonicalWorkloadContract(b));
}

TEST(WorkloadContract, EffectChangesIdentity) {
  WorkloadContract a = makeContract();
  WorkloadContract b = a;
  b.ports[0].effect = PortEffect::ReadWrite;
  EXPECT_NE(canonicalWorkloadContract(a), canonicalWorkloadContract(b));
}

TEST(WorkloadContract, NumericalContractChangesIdentity) {
  WorkloadContract a = makeContract();
  WorkloadContract b = a;
  b.numericalContract = "approx-f32";
  EXPECT_NE(canonicalWorkloadContract(a), canonicalWorkloadContract(b));
}

TEST(WorkloadContract, RequiredPrimitivesAreCanonicalized) {
  WorkloadContract a = makeContract();
  a.requiredPrimitives = {"micro.vector", "micro.copy", "micro.vector"};
  WorkloadContract b = a;
  b.requiredPrimitives = {"micro.copy", "micro.vector"};
  EXPECT_EQ(canonicalWorkloadContract(a), canonicalWorkloadContract(b));
}

TEST(WorkloadContract, RejectsDynamicExtent) {
  WorkloadContract contract = makeContract();
  contract.ports[0].shape[1] = mlir::ShapedType::kDynamic;
  auto error = validateWorkloadContract(contract);
  ASSERT_TRUE(bool(error));
  EXPECT_NE(
      errorText(std::move(error)).find("workload_contract: dynamic_extent"),
      std::string::npos);
}

TEST(WorkloadContract, RejectsZeroExtent) {
  WorkloadContract contract = makeContract();
  contract.ports[0].shape[1] = 0;
  auto error = validateWorkloadContract(contract);
  ASSERT_TRUE(bool(error));
  EXPECT_NE(errorText(std::move(error)).find("workload_contract: zero_extent"),
            std::string::npos);
}

TEST(WorkloadContract, RejectsNegativeExtent) {
  WorkloadContract contract = makeContract();
  contract.ports[0].shape[1] = -2;
  auto error = validateWorkloadContract(contract);
  ASSERT_TRUE(bool(error));
  EXPECT_NE(
      errorText(std::move(error)).find("workload_contract: negative_extent"),
      std::string::npos);
}

TEST(WorkloadContract, RejectsRankZero) {
  WorkloadContract contract = makeContract();
  contract.ports[0].shape.clear();
  auto error = validateWorkloadContract(contract);
  ASSERT_TRUE(bool(error));
  EXPECT_NE(errorText(std::move(error)).find("workload_contract: rank_zero"),
            std::string::npos);
}

TEST(WorkloadContract, RejectsDuplicatePortNames) {
  WorkloadContract contract = makeContract();
  contract.ports[1].name = contract.ports[0].name;
  auto error = validateWorkloadContract(contract);
  ASSERT_TRUE(bool(error));
  EXPECT_NE(
      errorText(std::move(error)).find("workload_contract: duplicate_port"),
      std::string::npos);
}

TEST(WorkloadContract, RejectsUnknownDtype) {
  WorkloadContract contract = makeContract();
  contract.ports[0].elementType = "q40";
  auto error = validateWorkloadContract(contract);
  ASSERT_TRUE(bool(error));
  EXPECT_NE(
      errorText(std::move(error)).find("workload_contract: unknown_dtype"),
      std::string::npos);
}

TEST(WorkloadContract, RejectsUnknownVersion) {
  WorkloadContract contract = makeContract();
  contract.version = 2;
  auto error = validateWorkloadContract(contract);
  ASSERT_TRUE(bool(error));
  EXPECT_NE(
      errorText(std::move(error)).find("workload_contract: unknown_version"),
      std::string::npos);
}

TEST(WorkloadContract, RejectsMissingVersionAttribute) {
  mlir::MLIRContext context;
  auto attr = mlir::DictionaryAttr::get(&context, {});
  auto decoded = decodeWorkloadContract(attr);
  ASSERT_FALSE(bool(decoded));
  EXPECT_NE(llvm::toString(decoded.takeError()).find("version"),
            std::string::npos);
}

TEST(WorkloadContract, RejectsWrongPortsAttributeType) {
  mlir::MLIRContext context;
  auto attr = makeDict(
      &context, {{"version", mlir::IntegerAttr::get(
                                 mlir::IntegerType::get(&context, 64), 1)},
                 {"ports", mlir::StringAttr::get(&context, "not-an-array")}});
  auto decoded = decodeWorkloadContract(attr);
  ASSERT_FALSE(bool(decoded));
  EXPECT_NE(llvm::toString(decoded.takeError()).find("ports"),
            std::string::npos);
}

TEST(WorkloadContract, RejectsUnknownDirectionAttribute) {
  mlir::MLIRContext context;
  auto decoded =
      decodeWorkloadContract(makeSinglePortContractAttr(&context, "sideways"));
  ASSERT_FALSE(bool(decoded));
  EXPECT_NE(llvm::toString(decoded.takeError()).find("direction"),
            std::string::npos);
}

TEST(WorkloadContract, DelimiterNamesRoundTripAndDoNotCollide) {
  WorkloadContract a;
  a.numericalContract = "strict-f32";
  a.ports = {{"a:b;c=2", PortDirection::Input, PortEffect::Read, {2}, "f32"}};
  WorkloadContract b;
  b.numericalContract = "strict-f32";
  b.ports = {{"a:b;c=3", PortDirection::Input, PortEffect::Read, {2}, "f32"}};

  EXPECT_NE(canonicalWorkloadContract(a), canonicalWorkloadContract(b));

  auto restored = roundTrip(a);
  ASSERT_TRUE(bool(restored)) << llvm::toString(restored.takeError());
  EXPECT_EQ(restored->ports[0].name, "a:b;c=2");
  EXPECT_EQ(canonicalWorkloadContract(*restored), canonicalWorkloadContract(a));
}
