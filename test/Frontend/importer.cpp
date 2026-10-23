//===- importer.cpp - Proven structured frontend import tests ------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "FrontendTestSupport.h"
#include "LLK/Conversion/Workload/SourceAdapter.h"
#include "LLK/Perf/Candidate.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

using namespace frontend_test;
using namespace llk::frontend;
using namespace mlir;

TEST(FrontendImporter, VerifiedAddAndMultiplyRoundTripWithoutTriton) {
  for (bool multiply : {false, true}) {
    auto value = data();
    if (multiply)
      for (auto &op : operations(value))
        if (op.getAsObject()->getString("opcode") == "add" &&
            op.getAsObject()
                    ->getArray("results")
                    ->front()
                    .getAsObject()
                    ->getString("dtype") == "f32")
          (*op.getAsObject())["opcode"] = "mul";
    auto request = parseRequest(render(value));
    ASSERT_TRUE(bool(request));
    MLIRContext context;
    auto imported = importProgram(*request, context);
    ASSERT_TRUE(bool(imported));
    EXPECT_TRUE(succeeded(verify(*imported->module)));
    unsigned genericCount = 0, scalarCount = 0;
    imported->module->walk([&](Operation *op) {
      EXPECT_TRUE(op->getName().isRegistered());
      EXPECT_FALSE(op->getName().getStringRef().starts_with("tt."));
      genericCount += isa<linalg::GenericOp>(op);
      scalarCount += multiply ? isa<arith::MulFOp>(op) : isa<arith::AddFOp>(op);
    });
    EXPECT_EQ(genericCount, 1u);
    EXPECT_EQ(scalarCount, 1u);
    EXPECT_EQ(imported->selection.symbol, "add");
    ASSERT_EQ(imported->contract.ports.size(), 4u);
    EXPECT_EQ(imported->contract.ports[2].effect,
              mlir::llk::workload::PortEffect::None);
    EXPECT_EQ(imported->contract.ports[3].effect,
              mlir::llk::workload::PortEffect::Write);
    ASSERT_EQ(imported->publicToNativePorts.size(), 3u);
    EXPECT_EQ(imported->publicToNativePorts[2].nativePort, 3u);
    EXPECT_EQ(imported->internalDestinationPort, 2u);
    std::string text;
    llvm::raw_string_ostream os(text);
    imported->module->print(os);
    auto again = parseSourceString<ModuleOp>(text, &context);
    ASSERT_TRUE(bool(again));
    EXPECT_TRUE(succeeded(verify(*again)));
  }
}

TEST(FrontendImporter, LivePythonOffsetsMasksAndMultiplyPassProof) {
  for (auto name : {"add", "mul"}) {
    auto request = parseRequest(
        read(std::string(LLK_FRONTEND_CAPTURE_DIR) + "/" + name + ".json"));
    ASSERT_TRUE(bool(request));
    MLIRContext context;
    auto imported = importProgram(*request, context);
    if (!imported)
      FAIL() << llvm::toString(imported.takeError());
    EXPECT_TRUE(succeeded(verify(*imported->module)));
  }
}

TEST(FrontendImporter, RejectsIncompleteDuplicateGridAndWrites) {
  for (unsigned variant = 0; variant != 4; ++variant) {
    auto value = data();
    auto *grid = program(value).getObject("launch")->getArray("grid");
    if (variant == 0)
      (*grid)[0] = 3;
    if (variant == 1)
      (*grid)[2] = 2;
    if (variant == 2) {
      auto &ops = operations(value);
      ops.insert(ops.end() - 1, ops[ops.size() - 2]);
    }
    if (variant == 3) {
      for (auto &op : operations(value))
        if (op.getAsObject()->getString("opcode") == "store")
          (*op.getAsObject()->getArray("operands"))[2] = "mask";
      for (auto &op : operations(value))
        if (op.getAsObject()->getString("opcode") == "constant" &&
            op.getAsObject()
                    ->getArray("results")
                    ->front()
                    .getAsObject()
                    ->getString("dtype") == "i1")
          (*op.getAsObject()->getObject("attributes"))["value"] = false;
    }
    auto request = parseRequest(render(value));
    ASSERT_TRUE(bool(request));
    MLIRContext context;
    auto imported = importProgram(*request, context);
    ASSERT_FALSE(bool(imported));
    EXPECT_EQ(errorCode(imported.takeError()), "invalid_launch_coverage");
  }
}

TEST(FrontendImporter, RejectsUnsafeIndexMap) {
  auto value = data();
  auto &ops = operations(value);
  (*ops[4].getAsObject()->getArray("operands"))[0] = "pn";
  auto request = parseRequest(render(value));
  ASSERT_TRUE(bool(request));
  MLIRContext context;
  auto imported = importProgram(*request, context);
  ASSERT_FALSE(bool(imported));
  EXPECT_EQ(errorCode(imported.takeError()), "invalid_launch_coverage");
}

TEST(FrontendImporter, SourceBlockDomainsAndOwnershipAreNotRetuned) {
  auto request = parseRequest(fixture());
  ASSERT_TRUE(bool(request));
  MLIRContext context;
  auto imported = importProgram(*request, context);
  ASSERT_TRUE(bool(imported));
  ASSERT_EQ(imported->space.params.size(), 2u);
  EXPECT_EQ(imported->space.params[0].role, "axis:d0");
  EXPECT_EQ(imported->space.params[1].role, "axis:d1");
  ASSERT_EQ(imported->space.params[0].choices.size(), 1u);
  EXPECT_EQ(imported->space.params[0].choices[0].integer(), 8);
  EXPECT_EQ(imported->space.params[1].choices[0].integer(), 16);
  auto registry = mlir::llk::workload::makeDefaultSourceAdapters();
  auto root = mlir::llk::workload::selectSourceRoot(
      *imported->module, imported->selection, registry, imported->adapterKey);
  ASSERT_TRUE(bool(root));
  const auto *adapter = registry.findByKey(imported->adapterKey);
  ASSERT_NE(adapter, nullptr);
  auto schema = adapter->bindings(*root, imported->space);
  ASSERT_TRUE(bool(schema));
  EXPECT_EQ(schema->front().owner, mlir::llk::workload::BindingOwner::Source);
  mlir::llk::perf::Candidate candidate;
  for (const auto &param : imported->space.params)
    candidate.values[param.name] = param.choices.front().integer();
  ASSERT_FALSE(bool(mlir::llk::workload::validateCandidateBindings(
      imported->space, candidate)));
  auto realized = adapter->realize(*imported->module, *root, imported->contract,
                                   imported->space, candidate);
  if (!realized)
    FAIL() << llvm::toString(realized.takeError());
  ASSERT_NE(realized->kernel, nullptr);
  EXPECT_TRUE(succeeded(verify(*imported->module)));
  EXPECT_EQ(realized->receipt.source.size(), 2u);
  EXPECT_TRUE(realized->receipt.mapping.empty());
  auto receiptError = mlir::llk::workload::validateBindingReceipt(
      imported->space, candidate, *schema, realized->receipt,
      mlir::llk::workload::ReceiptStage::Source);
  EXPECT_FALSE(bool(receiptError));
  llvm::consumeError(std::move(receiptError));
  candidate.values[imported->space.params[0].name] = 4;
  auto changed = mlir::llk::workload::validateCandidateBindings(imported->space,
                                                                candidate);
  EXPECT_TRUE(bool(changed));
  llvm::consumeError(std::move(changed));
}

TEST(FrontendImporter, RejectsRelaxedNumericalPolicyBeforeConstructingIR) {
  auto value = data();
  (*program(value).getObject("numerical_contract"))["fma"] = true;
  (*value.getAsObject()->getObject("options")->getObject(
      "numerical_contract"))["fma"] = true;
  auto request = parseRequest(render(value));
  ASSERT_TRUE(bool(request));
  MLIRContext context;
  auto imported = importProgram(*request, context);
  ASSERT_FALSE(bool(imported));
  EXPECT_EQ(errorCode(imported.takeError()), "unsupported_numerical_contract");
  EXPECT_FALSE(context.getLoadedDialect<linalg::LinalgDialect>());
}

TEST(FrontendImporter, RejectsNonAffineOverflowAndOutOfBoundsIndices) {
  for (unsigned variant = 0; variant != 3; ++variant) {
    auto value = data();
    auto &ops = operations(value);
    if (variant == 0)
      (*ops[8].getAsObject())["opcode"] = "mul";
    if (variant == 1)
      (*ops[2].getAsObject()->getObject("attributes"))["value"] = int64_t(1)
                                                                  << 62;
    if (variant == 2) {
      (*ops[6].getAsObject()->getObject("attributes"))["start"] = 1;
      (*ops[6].getAsObject()->getObject("attributes"))["end"] = 9;
    }
    auto request = parseRequest(render(value));
    ASSERT_TRUE(bool(request));
    MLIRContext context;
    auto imported = importProgram(*request, context);
    ASSERT_FALSE(bool(imported));
    EXPECT_EQ(errorCode(imported.takeError()), "invalid_launch_coverage");
  }
}

TEST(FrontendImporter,
     PublicPortOrderMapsWithoutAliasingTheInternalDestination) {
  auto value = data();
  auto *tensors = program(value).getArray("tensors");
  std::swap((*tensors)[0], (*tensors)[2]);
  auto request = parseRequest(render(value));
  ASSERT_TRUE(bool(request));
  MLIRContext context;
  auto imported = importProgram(*request, context);
  ASSERT_TRUE(bool(imported));
  EXPECT_EQ(imported->publicToNativePorts[0].publicPort, 0u);
  EXPECT_EQ(imported->publicToNativePorts[0].nativePort, 3u);
  EXPECT_EQ(imported->publicToNativePorts[1].nativePort, 0u);
  EXPECT_EQ(imported->publicToNativePorts[2].nativePort, 1u);
  auto function =
      imported->module->lookupSymbol<func::FuncOp>(imported->selection.symbol);
  ASSERT_TRUE(bool(function));
  auto generic = *function.getOps<linalg::GenericOp>().begin();
  auto scalar = mlir::cast<arith::AddFOp>(generic.getRegion().front().front());
  EXPECT_EQ(scalar.getLhs(), generic.getRegion().front().getArgument(1));
  EXPECT_EQ(scalar.getRhs(), generic.getRegion().front().getArgument(0));
}

TEST(FrontendImporter, RejectsZeroResultCountedLoopWithoutConstructingIR) {
  auto value = data();
  auto &ops = operations(value);
  auto span = *ops[0].getAsObject()->get("span");
  llvm::json::Object yield{{"opcode", "yield"},
                           {"operands", llvm::json::Array{}},
                           {"results", llvm::json::Array{}},
                           {"attributes", llvm::json::Object{}},
                           {"regions", llvm::json::Array{}},
                           {"span", span}};
  llvm::json::Object loop{
      {"opcode", "counted_loop"},
      {"operands", llvm::json::Array{*ops[2]
                                          .getAsObject()
                                          ->getArray("results")
                                          ->front()
                                          .getAsObject()
                                          ->get("id"),
                                     *ops[3]
                                          .getAsObject()
                                          ->getArray("results")
                                          ->front()
                                          .getAsObject()
                                          ->get("id")}},
      {"results", llvm::json::Array{}},
      {"attributes", llvm::json::Object{{"step", 1}}},
      {"regions", llvm::json::Array{llvm::json::Object{
                      {"arguments", llvm::json::Array{llvm::json::Object{
                                        {"id", "iv"},
                                        {"dtype", "index"},
                                        {"shape", llvm::json::Array{}}}}},
                      {"operations", llvm::json::Array{std::move(yield)}}}}},
      {"span", span}};
  ops.insert(ops.end() - 1, llvm::json::Value(std::move(loop)));
  auto request = parseRequest(render(value));
  ASSERT_TRUE(bool(request));
  MLIRContext context;
  auto imported = importProgram(*request, context);
  ASSERT_FALSE(bool(imported));
  EXPECT_EQ(errorCode(imported.takeError()), "unsupported_frontend_program");
  EXPECT_FALSE(context.getLoadedDialect<linalg::LinalgDialect>());
}
