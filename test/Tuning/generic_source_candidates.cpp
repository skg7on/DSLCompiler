//===- generic_source_candidates.cpp - Direct MicroIR source tests --------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "FrameworkTestSupport.h"
#include "LLK/Conversion/MicroMapping/CandidateInstantiation.h"
#include "LLK/Conversion/Workload/WorkloadContract.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/Error.h"
#include <algorithm>
#include <string>
#include <utility>

using namespace mlir;
using namespace mlir::llk;

namespace {
constexpr llvm::StringLiteral addSource = R"mlir(
module {
  micro.kernel @add(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>)
      -> tensor<8x8xf32> attributes {micro.numerical_contract = "strict-f32"} {
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>}
      : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b {shape = array<i64: 8, 8>}
      : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %s = micro.vector "add" %ta, %tb
      : !micro.tile<8x8xf32, memory = #micro.memory<sram>>,
        !micro.tile<8x8xf32, memory = #micro.memory<sram>>
        -> !micro.tile<8x8xf32, memory = #micro.memory<acc>>
    micro.yield %s : !micro.tile<8x8xf32, memory = #micro.memory<acc>>
  }
})mlir";

workload::WorkloadContract addContract() {
  workload::WorkloadContract c;
  c.numericalContract = "strict-f32";
  c.ports = {{"input0",
              workload::PortDirection::Input,
              workload::PortEffect::Read,
              {8, 8},
              "f32"},
             {"input1",
              workload::PortDirection::Input,
              workload::PortEffect::Read,
              {8, 8},
              "f32"},
             {"output0",
              workload::PortDirection::Output,
              workload::PortEffect::Write,
              {8, 8},
              "f32"}};
  c.requiredPrimitives = {"micro.vector"};
  return c;
}

perf::SearchSpace singleton() {
  perf::SearchSpace s;
  s.name = "direct";
  s.workload = "diagnostic-only-label";
  return s;
}
perf::Candidate candidate() {
  perf::Candidate c;
  c.id = perf::computeCandidateId({}, {});
  return c;
}
tuning::CandidateInstantiationOptions options(std::string symbol = "add") {
  tuning::CandidateInstantiationOptions o;
  o.sourceMode = tuning::SourceMode::ConcreteMicro;
  o.sourceSymbol = std::move(symbol);
  return o;
}

void expectError(llvm::Expected<tuning::CandidateInstance> result,
                 llvm::StringRef reason) {
  ASSERT_FALSE(bool(result));
  EXPECT_NE(llvm::toString(result.takeError()).find(reason.str()),
            std::string::npos);
}
} // namespace

TEST_F(FrameworkTest, DirectElementwiseNeedsNoMmaFacts) {
  auto source = parse(addSource);
  ASSERT_TRUE(source);
  const auto before = print(source->getOperation());
  auto result = tuning::instantiateCandidate(*source, singleton(), candidate(),
                                             addContract(), options());
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(result->bindingFacts.contractions.empty());
  EXPECT_FALSE(result->bindingFacts.originalWorkload);
  EXPECT_NE(result->sourceGraphHash, 0u);
  ASSERT_EQ(result->workloadGraph.getNodes().size(), 1u);
  EXPECT_EQ(result->workloadGraph.getNodes()[0].opName, "micro.vector");
  EXPECT_EQ(print(source->getOperation()), before);
  EXPECT_NE(result->kernel, SymbolTable::lookupSymbolIn(*source, "add"));
  EXPECT_TRUE(succeeded(verify(*result->module)));
  auto attr =
      result->kernel->getAttrOfType<DictionaryAttr>("micro.workload_contract");
  auto restored = workload::decodeWorkloadContract(attr);
  ASSERT_TRUE(bool(restored)) << llvm::toString(restored.takeError());
  EXPECT_EQ(workload::canonicalWorkloadContract(*restored),
            workload::canonicalWorkloadContract(addContract()));
  EXPECT_EQ(workload::canonicalWorkloadContract(result->workloadContract),
            workload::canonicalWorkloadContract(addContract()));
}

TEST_F(FrameworkTest, RejectsReadInputsMarkedAsNone) {
  auto source = parse(addSource);
  ASSERT_TRUE(source);
  const auto before = print(source->getOperation());
  for (unsigned input = 0; input != 2; ++input) {
    SCOPED_TRACE(input);
    auto contract = addContract();
    contract.ports[input].effect = workload::PortEffect::None;
    expectError(tuning::instantiateCandidate(*source, singleton(), candidate(),
                                             contract, options()),
                "effect none");
    EXPECT_EQ(print(source->getOperation()), before);
  }
}

TEST_F(FrameworkTest, AcceptsUnusedInputMarkedAsNone) {
  std::string text = addSource.str();
  const std::string arg = "%b: tensor<8x8xf32>";
  text.replace(text.find(arg), arg.size(), arg + ", %init: tensor<8x8xf32>");
  auto source = parse(text);
  ASSERT_TRUE(source);
  auto contract = addContract();
  contract.ports.insert(contract.ports.begin() + 2,
                        {"init",
                         workload::PortDirection::Input,
                         workload::PortEffect::None,
                         {8, 8},
                         "f32"});
  auto result = tuning::instantiateCandidate(*source, singleton(), candidate(),
                                             contract, options());
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  EXPECT_TRUE(succeeded(verify(*result->module)));
  EXPECT_TRUE(result->kernel->getRegion(0).front().getArgument(2).use_empty());
}

TEST_F(FrameworkTest, RejectsMismatchedOrderedPorts) {
  auto source = parse(addSource);
  ASSERT_TRUE(source);
  const auto before = print(source->getOperation());
  for (unsigned variant = 0; variant != 7; ++variant) {
    SCOPED_TRACE(variant);
    auto c = addContract();
    switch (variant) {
    case 0:
      c.ports.pop_back();
      break;
    case 1:
      c.ports[0].shape = {4, 16};
      break;
    case 2:
      c.ports[1].elementType = "f16";
      break;
    case 3:
      std::swap(c.ports[0], c.ports[2]);
      break;
    case 4:
      c.ports[0].effect = workload::PortEffect::ReadWrite;
      break;
    case 5:
      c.ports[2].effect = workload::PortEffect::Read;
      break;
    case 6:
      c.ports[2].shape = {64};
      break;
    }
    expectError(tuning::instantiateCandidate(*source, singleton(), candidate(),
                                             c, options()),
                "port");
    EXPECT_EQ(print(source->getOperation()), before);
  }
}

TEST_F(FrameworkTest, RejectsMissingOrMismatchedNumericalMetadata) {
  auto source = parse(addSource);
  ASSERT_TRUE(source);
  auto c = addContract();
  c.numericalContract = "approx-f32";
  expectError(tuning::instantiateCandidate(*source, singleton(), candidate(), c,
                                           options()),
              "numerical");
  SymbolTable::lookupSymbolIn(*source, "add")
      ->removeAttr("micro.numerical_contract");
  expectError(tuning::instantiateCandidate(*source, singleton(), candidate(),
                                           addContract(), options()),
              "numerical");
}

TEST_F(FrameworkTest, RejectsUnknownOrAmbiguousKernelSymbols) {
  auto source = parse(addSource);
  ASSERT_TRUE(source);
  expectError(tuning::instantiateCandidate(*source, singleton(), candidate(),
                                           addContract(), options("missing")),
              "symbol");
  auto *other = SymbolTable::lookupSymbolIn(*source, "add")->clone();
  other->setAttr("sym_name", StringAttr::get(context.get(), "other"));
  source->getBody()->push_back(other);
  expectError(tuning::instantiateCandidate(*source, singleton(), candidate(),
                                           addContract(), options("")),
              "ambiguous");
  auto exact = tuning::instantiateCandidate(*source, singleton(), candidate(),
                                            addContract(), options("other"));
  ASSERT_TRUE(bool(exact)) << llvm::toString(exact.takeError());
  EXPECT_EQ(SymbolTable::getSymbolName(exact->kernel).getValue(), "other");
  other->setAttr("sym_name", StringAttr::get(context.get(), "add"));
  expectError(tuning::instantiateCandidate(*source, singleton(), candidate(),
                                           addContract(), options()),
              "ambiguous");
}

TEST_F(FrameworkTest, SelectsSingletonWithoutSymbol) {
  auto source = parse(addSource);
  ASSERT_TRUE(source);
  auto result = tuning::instantiateCandidate(*source, singleton(), candidate(),
                                             addContract(), options(""));
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  auto o = options();
  o.sourceRootOrdinal = 1;
  expectError(tuning::instantiateCandidate(*source, singleton(), candidate(),
                                           addContract(), o),
              "ordinal");
}

TEST_F(FrameworkTest, RejectsUnrealizedParametersAndUndeclaredBindings) {
  auto source = parse(addSource);
  ASSERT_TRUE(source);
  auto s = singleton();
  s.params = {{"T0", "integer", {perf::SearchChoice(int64_t{4})}}};
  auto c = candidate();
  c.values["T0"] = 4;
  expectError(
      tuning::instantiateCandidate(*source, s, c, addContract(), options()),
      "axis_not_realized: T0");
  expectError(tuning::instantiateCandidate(*source, singleton(), c,
                                           addContract(), options()),
              "binding");
}

TEST_F(FrameworkTest, DiagnosticLabelCannotChangeSemanticsOrIdentity) {
  auto source = parse(addSource);
  ASSERT_TRUE(source);
  auto first = tuning::instantiateCandidate(*source, singleton(), candidate(),
                                            addContract(), options());
  auto s = singleton();
  s.workload = "fused_swiglu";
  auto second = tuning::instantiateCandidate(*source, s, candidate(),
                                             addContract(), options());
  ASSERT_TRUE(bool(first)) << llvm::toString(first.takeError());
  ASSERT_TRUE(bool(second)) << llvm::toString(second.takeError());
  EXPECT_EQ(first->sourceGraphHash, second->sourceGraphHash);
  EXPECT_EQ(print(first->kernel), print(second->kernel));
}

TEST_F(FrameworkTest, FullSourceAndContractContentChangeIdentity) {
  auto source = parse(addSource);
  ASSERT_TRUE(source);
  auto first = tuning::instantiateCandidate(*source, singleton(), candidate(),
                                            addContract(), options());
  ASSERT_TRUE(bool(first)) << llvm::toString(first.takeError());
  auto c = addContract();
  c.ports[0].name = "renamed";
  auto changedContract = tuning::instantiateCandidate(
      *source, singleton(), candidate(), c, options());
  ASSERT_TRUE(bool(changedContract))
      << llvm::toString(changedContract.takeError());
  EXPECT_NE(first->sourceGraphHash, changedContract->sourceGraphHash);
  source->getOperation()->setAttr("micro.semantic_tag",
                                  StringAttr::get(context.get(), "changed"));
  auto changedSource = tuning::instantiateCandidate(
      *source, singleton(), candidate(), addContract(), options());
  ASSERT_TRUE(bool(changedSource)) << llvm::toString(changedSource.takeError());
  EXPECT_NE(first->sourceGraphHash, changedSource->sourceGraphHash);
  auto *kernel = SymbolTable::lookupSymbolIn(*source, "add");
  kernel->setAttr("micro.numerical_contract",
                  StringAttr::get(context.get(), "approx-f32"));
  c = addContract();
  c.numericalContract = "approx-f32";
  auto changedMath = tuning::instantiateCandidate(*source, singleton(),
                                                  candidate(), c, options());
  ASSERT_TRUE(bool(changedMath)) << llvm::toString(changedMath.takeError());
  EXPECT_NE(changedSource->sourceGraphHash, changedMath->sourceGraphHash);
}

TEST_F(FrameworkTest, RejectsAbsentRequiredPrimitive) {
  auto source = parse(addSource);
  ASSERT_TRUE(source);
  auto c = addContract();
  c.requiredPrimitives = {"micro.mma"};
  expectError(tuning::instantiateCandidate(*source, singleton(), candidate(), c,
                                           options()),
              "primitive");
}

TEST_F(FrameworkTest, RequiredPrimitivesMustCoverActualWorkloadGraph) {
  auto source = parse(addSource);
  ASSERT_TRUE(source);
  auto c = addContract();
  c.requiredPrimitives.clear();
  const auto before = print(source->getOperation());
  expectError(tuning::instantiateCandidate(*source, singleton(), candidate(), c,
                                           options()),
              "micro.vector");
  EXPECT_EQ(print(source->getOperation()), before);
}

TEST_F(FrameworkTest, RejectsMalformedNumericalMetadataEvenForEmptyContract) {
  auto source = parse(addSource);
  ASSERT_TRUE(source);
  auto *kernel = SymbolTable::lookupSymbolIn(*source, "add");
  kernel->setAttr("micro.numerical_contract",
                  IntegerAttr::get(IntegerType::get(context.get(), 64), 1));
  auto c = addContract();
  c.numericalContract.clear();
  expectError(tuning::instantiateCandidate(*source, singleton(), candidate(), c,
                                           options()),
              "numerical");
}

TEST_F(FrameworkTest,
       RetainedContractCannotBeOverwrittenWithDifferentSemantics) {
  auto source = parse(addSource);
  ASSERT_TRUE(source);
  auto retained = addContract();
  retained.ports[0].elementType = "f16";
  SymbolTable::lookupSymbolIn(*source, "add")
      ->setAttr("micro.workload_contract",
                workload::encodeWorkloadContract(context.get(), retained));
  const auto before = print(source->getOperation());
  expectError(tuning::instantiateCandidate(*source, singleton(), candidate(),
                                           addContract(), options()),
              "retained");
  EXPECT_EQ(print(source->getOperation()), before);
}

TEST_F(FrameworkTest, UnusedInputsRemainOrderedAndParticipateInIdentity) {
  std::string text = addSource.str();
  const std::string arg = "%b: tensor<8x8xf32>";
  text.replace(text.find(arg), arg.size(), arg + ", %init: tensor<4x16xf16>");
  auto source = parse(text);
  ASSERT_TRUE(source);
  auto c = addContract();
  c.ports.insert(c.ports.begin() + 2, {"init",
                                       workload::PortDirection::Input,
                                       workload::PortEffect::Read,
                                       {4, 16},
                                       "f16"});
  const auto before = print(source->getOperation());
  auto result = tuning::instantiateCandidate(*source, singleton(), candidate(),
                                             c, options());
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  EXPECT_EQ(print(source->getOperation()), before);
  auto type = cast<FunctionType>(
      result->kernel->getAttrOfType<TypeAttr>("function_type").getValue());
  EXPECT_EQ(type.getNumInputs(), 3u);
  EXPECT_TRUE(
      cast<RankedTensorType>(type.getInput(2)).getElementType().isF16());
  std::swap(c.ports[0], c.ports[2]);
  expectError(tuning::instantiateCandidate(*source, singleton(), candidate(), c,
                                           options()),
              "port");
  auto base = parse(addSource);
  ASSERT_TRUE(base);
  auto original = tuning::instantiateCandidate(*base, singleton(), candidate(),
                                               addContract(), options());
  ASSERT_TRUE(bool(original)) << llvm::toString(original.takeError());
  EXPECT_NE(original->sourceGraphHash, result->sourceGraphHash);
}

TEST_F(FrameworkTest, RejectsUnsupportedSourceModeAndNullModule) {
  auto source = parse(addSource);
  ASSERT_TRUE(source);
  auto o = options();
  o.sourceMode = tuning::SourceMode::SemanticSource;
  // A concrete MicroIR module is not a semantic source: the adapter path
  // selects a source *function*, so a kernel symbol is not a function.
  expectError(tuning::instantiateCandidate(*source, singleton(), candidate(),
                                           addContract(), o),
              "@add' is not a function");
  // The synthetic binder is explicitly requested legacy behavior; the contract
  // path does not realize it.
  auto synthetic = options();
  synthetic.sourceMode = tuning::SourceMode::Synthetic;
  expectError(tuning::instantiateCandidate(*source, singleton(), candidate(),
                                           addContract(), synthetic),
              "synthetic");
  expectError(tuning::instantiateCandidate(ModuleOp{}, singleton(), candidate(),
                                           addContract(), options()),
              "module");
}
