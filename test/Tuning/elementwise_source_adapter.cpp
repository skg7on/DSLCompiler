//===- elementwise_source_adapter.cpp - Bounded elementwise adapter tests -===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework B5: a deliberately bounded non-contraction source family. The
// adapter recognizes `linalg.generic` structurally and then validates exactly
// one supported shape of it: static rank-two f32 add/mul. Everything outside
// that envelope is rejected with the feature that made it unsupported, so a
// broader Linalg program cannot slip through as if it were this workload.
//
//===----------------------------------------------------------------------===//

#include "FrameworkTestSupport.h"

#include "LLK/Conversion/MicroMapping/CandidateInstantiation.h"
#include "LLK/Conversion/MicroToLinalg.h"
#include "LLK/Conversion/Workload/SourceAdapter.h"
#include "LLK/Conversion/Workload/WorkloadContract.h"
#include "LLK/Perf/Candidate.h"
#include "LLK/Perf/SearchSpace.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::llk;

namespace {

constexpr llvm::StringLiteral elementwiseKey = "linalg-elementwise-v1";

perf::SearchSpace singletonSpace() {
  perf::SearchSpace space;
  space.name = "elementwise";
  return space;
}

perf::Candidate singletonCandidate() {
  perf::Candidate candidate;
  candidate.id = perf::computeCandidateId({}, {});
  return candidate;
}

tuning::CandidateInstantiationOptions sourceOptions() {
  tuning::CandidateInstantiationOptions options;
  options.sourceMode = tuning::SourceMode::SemanticSource;
  options.sourceSymbol = "elementwise";
  options.adapterKey = elementwiseKey.str();
  return options;
}

void expectDescribeError(const std::string &text, llvm::StringRef reason) {
  // Parsing is the test's own business: a variant that does not parse would
  // not exercise the adapter's validator at all.
  static DialectRegistry dialects = [] {
    DialectRegistry registry;
    registry.insert<micro::MicroDialect, llk::LLKDialect, func::FuncDialect,
                    linalg::LinalgDialect, tensor::TensorDialect,
                    arith::ArithDialect>();
    return registry;
  }();
  static MLIRContext context(dialects);
  auto source = parseSourceString<ModuleOp>(text, &context);
  ASSERT_TRUE(source) << text;
  workload::SourceAdapterRegistry registry =
      workload::makeDefaultSourceAdapters();
  auto root = workload::selectSourceRoot(*source, {"elementwise", 0}, registry,
                                         elementwiseKey);
  ASSERT_TRUE(bool(root)) << llvm::toString(root.takeError());
  auto adapter = registry.resolve(*root, elementwiseKey);
  ASSERT_TRUE(bool(adapter)) << llvm::toString(adapter.takeError());
  auto described = (*adapter)->describe(*root);
  ASSERT_FALSE(bool(described)) << text;
  EXPECT_NE(llvm::toString(described.takeError()).find(reason.str()),
            std::string::npos)
      << "expected '" << reason.str()
      << "' in: " << llvm::toString(described.takeError());
}

/// One negative structure per feature the bounded adapter must reject.
struct NegativeVariant {
  const char *feature;
  llvm::StringLiteral text;
  llvm::StringLiteral reason;
};

constexpr llvm::StringLiteral reductionSource = R"mlir(
module {
  func.func @elementwise(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>)
      -> tensor<8xf32> {
    %init = tensor.empty() : tensor<8xf32>
    %out = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>],
      iterator_types = ["parallel", "reduction"]
    } ins(%a, %b : tensor<8x16xf32>, tensor<8x16xf32>)
      outs(%init : tensor<8xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<8xf32>
    return %out : tensor<8xf32>
  }
})mlir";

constexpr llvm::StringLiteral nonidentitySource = R"mlir(
#id = affine_map<(d0, d1) -> (d0, d1)>
#transpose = affine_map<(d0, d1) -> (d1, d0)>
module {
  func.func @elementwise(%a: tensor<16x8xf32>, %b: tensor<8x16xf32>)
      -> tensor<8x16xf32> {
    %init = tensor.empty() : tensor<8x16xf32>
    %out = linalg.generic {
      indexing_maps = [#transpose, #id, #id],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<16x8xf32>, tensor<8x16xf32>)
      outs(%init : tensor<8x16xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<8x16xf32>
    return %out : tensor<8x16xf32>
  }
})mlir";

constexpr llvm::StringLiteral dynamicSource = R"mlir(
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @elementwise(%n: index, %a: tensor<?x16xf32>,
                         %b: tensor<?x16xf32>) -> tensor<?x16xf32> {
    %init = tensor.empty(%n) : tensor<?x16xf32>
    %out = linalg.generic {
      indexing_maps = [#id, #id, #id],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<?x16xf32>, tensor<?x16xf32>)
      outs(%init : tensor<?x16xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<?x16xf32>
    return %out : tensor<?x16xf32>
  }
})mlir";

constexpr llvm::StringLiteral zeroExtentSource = R"mlir(
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @elementwise(%a: tensor<0x16xf32>, %b: tensor<0x16xf32>)
      -> tensor<0x16xf32> {
    %init = tensor.empty() : tensor<0x16xf32>
    %out = linalg.generic {
      indexing_maps = [#id, #id, #id],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<0x16xf32>, tensor<0x16xf32>)
      outs(%init : tensor<0x16xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum : f32
    } -> tensor<0x16xf32>
    return %out : tensor<0x16xf32>
  }
})mlir";

constexpr llvm::StringLiteral f64Source = R"mlir(
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @elementwise(%a: tensor<8x16xf64>, %b: tensor<8x16xf64>)
      -> tensor<8x16xf64> {
    %init = tensor.empty() : tensor<8x16xf64>
    %out = linalg.generic {
      indexing_maps = [#id, #id, #id],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<8x16xf64>, tensor<8x16xf64>)
      outs(%init : tensor<8x16xf64>) {
    ^bb0(%x: f64, %y: f64, %unused: f64):
      %sum = arith.addf %x, %y : f64
      linalg.yield %sum : f64
    } -> tensor<8x16xf64>
    return %out : tensor<8x16xf64>
  }
})mlir";

constexpr llvm::StringLiteral extraBodySource = R"mlir(
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @elementwise(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>)
      -> tensor<8x16xf32> {
    %init = tensor.empty() : tensor<8x16xf32>
    %out = linalg.generic {
      indexing_maps = [#id, #id, #id],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<8x16xf32>, tensor<8x16xf32>)
      outs(%init : tensor<8x16xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %one = arith.constant 1.0 : f32
      %sum = arith.addf %x, %one : f32
      linalg.yield %sum : f32
    } -> tensor<8x16xf32>
    return %out : tensor<8x16xf32>
  }
})mlir";

constexpr llvm::StringLiteral initReadSource = R"mlir(
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @elementwise(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>)
      -> tensor<8x16xf32> {
    %init = tensor.empty() : tensor<8x16xf32>
    %out = linalg.generic {
      indexing_maps = [#id, #id, #id],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<8x16xf32>, tensor<8x16xf32>)
      outs(%init : tensor<8x16xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %unused : f32
      linalg.yield %sum : f32
    } -> tensor<8x16xf32>
    return %out : tensor<8x16xf32>
  }
})mlir";

constexpr llvm::StringLiteral multipleResultsSource = R"mlir(
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @elementwise(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>)
      -> (tensor<8x16xf32>, tensor<8x16xf32>) {
    %init0 = tensor.empty() : tensor<8x16xf32>
    %init1 = tensor.empty() : tensor<8x16xf32>
    %out:2 = linalg.generic {
      indexing_maps = [#id, #id, #id, #id],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<8x16xf32>, tensor<8x16xf32>)
      outs(%init0, %init1 : tensor<8x16xf32>, tensor<8x16xf32>) {
    ^bb0(%x: f32, %y: f32, %unused0: f32, %unused1: f32):
      %sum = arith.addf %x, %y : f32
      linalg.yield %sum, %sum : f32, f32
    } -> (tensor<8x16xf32>, tensor<8x16xf32>)
    return %out#0, %out#1 : tensor<8x16xf32>, tensor<8x16xf32>
  }
})mlir";

constexpr llvm::StringLiteral fastmathSource = R"mlir(
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @elementwise(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>)
      -> tensor<8x16xf32> {
    %init = tensor.empty() : tensor<8x16xf32>
    %out = linalg.generic {
      indexing_maps = [#id, #id, #id],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<8x16xf32>, tensor<8x16xf32>)
      outs(%init : tensor<8x16xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %sum = arith.addf %x, %y fastmath<fast> : f32
      linalg.yield %sum : f32
    } -> tensor<8x16xf32>
    return %out : tensor<8x16xf32>
  }
})mlir";

} // namespace

/// The fixture loads `test/Tuning/Inputs/framework/elementwise.mlir`, a
/// deliberately plain `linalg.generic` add over two rank-two f32 operands.
class ElementwiseSourceAdapter : public FrameworkTest {
protected:
  OwningOpRef<ModuleOp> loadFixture(llvm::StringRef name) {
    std::string path = std::string(LLK_SOURCE_DIR) +
                       "/test/Tuning/Inputs/framework/" + name.str();
    auto buffer = llvm::MemoryBuffer::getFile(path);
    if (!buffer)
      return {};
    return parse((*buffer)->getBuffer());
  }
};

TEST_F(ElementwiseSourceAdapter, DescribesBoundedElementwiseContract) {
  auto source = loadFixture("elementwise.mlir");
  ASSERT_TRUE(source);
  const std::string before = print(source->getOperation());

  workload::SourceAdapterRegistry registry =
      workload::makeDefaultSourceAdapters();
  auto root = workload::selectSourceRoot(*source, {"elementwise", 0}, registry,
                                         elementwiseKey);
  ASSERT_TRUE(bool(root)) << llvm::toString(root.takeError());
  auto adapter = registry.resolve(*root, elementwiseKey);
  ASSERT_TRUE(bool(adapter)) << llvm::toString(adapter.takeError());
  EXPECT_EQ((*adapter)->key(), elementwiseKey);

  auto described = (*adapter)->describe(*root);
  ASSERT_TRUE(bool(described)) << llvm::toString(described.takeError());

  // Axes are the source's iteration space, not contraction M/N/K.
  EXPECT_EQ(described->axes.at("d0"), 8);
  EXPECT_EQ(described->axes.at("d1"), 16);
  EXPECT_EQ(described->axes.count("K"), 0u);

  // Two read inputs, the destination with no incoming read, and one written
  // result, in the source's operand order.
  ASSERT_EQ(described->ports.size(), 4u);
  EXPECT_EQ(described->ports[0].direction, workload::PortDirection::Input);
  EXPECT_EQ(described->ports[0].effect, workload::PortEffect::Read);
  EXPECT_EQ(described->ports[0].shape, (std::vector<int64_t>{8, 16}));
  EXPECT_EQ(described->ports[0].elementType, "f32");
  EXPECT_EQ(described->ports[1].direction, workload::PortDirection::Input);
  EXPECT_EQ(described->ports[1].effect, workload::PortEffect::Read);
  EXPECT_EQ(described->ports[1].shape, (std::vector<int64_t>{8, 16}));
  EXPECT_EQ(described->ports[2].direction, workload::PortDirection::Input);
  EXPECT_EQ(described->ports[2].effect, workload::PortEffect::None);
  EXPECT_EQ(described->ports[2].shape, (std::vector<int64_t>{8, 16}));
  EXPECT_EQ(described->ports[3].direction, workload::PortDirection::Output);
  EXPECT_EQ(described->ports[3].effect, workload::PortEffect::Write);
  EXPECT_EQ(described->ports[3].shape, (std::vector<int64_t>{8, 16}));

  // Strict f32 is the first supported numerical mode; the primitive the
  // emitter actually produces is a vector add.
  EXPECT_EQ(described->numericalContract, "strict-f32");
  EXPECT_NE(std::find(described->requiredPrimitives.begin(),
                      described->requiredPrimitives.end(), "micro.vector"),
            described->requiredPrimitives.end());
  EXPECT_NE(std::find(described->requiredPrimitives.begin(),
                      described->requiredPrimitives.end(),
                      "micro.tile_async_copy"),
            described->requiredPrimitives.end());

  EXPECT_EQ(print(source->getOperation()), before);
}

TEST_F(ElementwiseSourceAdapter, EmitsOneVectorAddOverAStagedFullTile) {
  auto source = loadFixture("elementwise.mlir");
  ASSERT_TRUE(source);
  const std::string before = print(source->getOperation());

  workload::SourceAdapterRegistry registry =
      workload::makeDefaultSourceAdapters();
  auto root = workload::selectSourceRoot(*source, {"elementwise", 0}, registry,
                                         elementwiseKey);
  ASSERT_TRUE(bool(root)) << llvm::toString(root.takeError());
  auto adapter = registry.resolve(*root, elementwiseKey);
  ASSERT_TRUE(bool(adapter)) << llvm::toString(adapter.takeError());
  auto contract = (*adapter)->describe(*root);
  ASSERT_TRUE(bool(contract)) << llvm::toString(contract.takeError());

  auto instance = tuning::instantiateCandidate(*source, singletonSpace(),
                                               singletonCandidate(), *contract,
                                               sourceOptions(), registry);
  ASSERT_TRUE(bool(instance)) << llvm::toString(instance.takeError());
  ASSERT_TRUE(instance->kernel);
  EXPECT_TRUE(succeeded(verify(*instance->module)));
  EXPECT_EQ(print(source->getOperation()), before);

  int vectorOps = 0;
  int mmaOps = 0;
  int movements = 0;
  int semanticOps = 0;
  instance->kernel->walk([&](Operation *op) {
    llvm::StringRef name = op->getName().getStringRef();
    if (name == "micro.vector") {
      ++vectorOps;
      EXPECT_EQ(op->getAttrOfType<StringAttr>("op").getValue(), "add");
    }
    if (name == "micro.mma")
      ++mmaOps;
    if (name == "micro.tile_async_copy")
      ++movements;
    if (name.starts_with("linalg.") || name.starts_with("llk."))
      ++semanticOps;
  });
  EXPECT_EQ(vectorOps, 1);
  EXPECT_EQ(mmaOps, 0);
  EXPECT_EQ(movements, 2);
  EXPECT_EQ(semanticOps, 0);

  // The signature names every root operand and result, so the destination is a
  // declared input -- but an unused `tensor.empty` is never read or staged.
  auto type = cast<FunctionType>(
      instance->kernel->getAttrOfType<TypeAttr>("function_type").getValue());
  ASSERT_EQ(type.getNumInputs(), 3u);
  ASSERT_EQ(type.getNumResults(), 1u);
  Block &entry = instance->kernel->getRegion(0).front();
  EXPECT_TRUE(entry.getArgument(2).use_empty());

  EXPECT_TRUE(instance->receipt.source.empty());
  EXPECT_TRUE(instance->receipt.mapping.empty());
}

TEST_F(ElementwiseSourceAdapter, MultiplyBecomesOneVectorMul) {
  std::string text = R"mlir(
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @elementwise(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>)
      -> tensor<8x16xf32> {
    %init = tensor.empty() : tensor<8x16xf32>
    %out = linalg.generic {
      indexing_maps = [#id, #id, #id],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<8x16xf32>, tensor<8x16xf32>)
      outs(%init : tensor<8x16xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %product = arith.mulf %x, %y : f32
      linalg.yield %product : f32
    } -> tensor<8x16xf32>
    return %out : tensor<8x16xf32>
  }
})mlir";
  auto source = parse(text);
  ASSERT_TRUE(source);

  workload::SourceAdapterRegistry registry =
      workload::makeDefaultSourceAdapters();
  auto root = workload::selectSourceRoot(*source, {"elementwise", 0}, registry,
                                         elementwiseKey);
  ASSERT_TRUE(bool(root)) << llvm::toString(root.takeError());
  auto adapter = registry.resolve(*root, elementwiseKey);
  ASSERT_TRUE(bool(adapter)) << llvm::toString(adapter.takeError());
  auto contract = (*adapter)->describe(*root);
  ASSERT_TRUE(bool(contract)) << llvm::toString(contract.takeError());

  auto instance = tuning::instantiateCandidate(*source, singletonSpace(),
                                               singletonCandidate(), *contract,
                                               sourceOptions(), registry);
  ASSERT_TRUE(bool(instance)) << llvm::toString(instance.takeError());
  int vectorOps = 0;
  instance->kernel->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "micro.vector")
      return;
    ++vectorOps;
    EXPECT_EQ(op->getAttrOfType<StringAttr>("op").getValue(), "mul");
  });
  EXPECT_EQ(vectorOps, 1);
}

TEST_F(ElementwiseSourceAdapter, FullTileLowersThroughReferencePath) {
  auto source = loadFixture("elementwise.mlir");
  ASSERT_TRUE(source);
  workload::SourceAdapterRegistry registry =
      workload::makeDefaultSourceAdapters();
  auto root = workload::selectSourceRoot(*source, {"elementwise", 0}, registry,
                                         elementwiseKey);
  ASSERT_TRUE(bool(root)) << llvm::toString(root.takeError());
  auto adapter = registry.resolve(*root, elementwiseKey);
  ASSERT_TRUE(bool(adapter)) << llvm::toString(adapter.takeError());
  auto contract = (*adapter)->describe(*root);
  ASSERT_TRUE(bool(contract)) << llvm::toString(contract.takeError());
  auto instance = tuning::instantiateCandidate(*source, singletonSpace(),
                                               singletonCandidate(), *contract,
                                               sourceOptions(), registry);
  ASSERT_TRUE(bool(instance)) << llvm::toString(instance.takeError());

  // The emitted full tile is ordinary MicroIR: the portable reference path
  // must lower it without a special case.
  PassManager passes(context.get());
  passes.addPass(llk::createMicroToLinalgPass());
  ASSERT_TRUE(succeeded(passes.run(*instance->module)));
  EXPECT_TRUE(succeeded(verify(*instance->module)));

  int microOps = 0;
  instance->module->walk([&](Operation *op) {
    if (op->getName().getStringRef().starts_with("micro."))
      ++microOps;
  });
  EXPECT_EQ(microOps, 0);
}

TEST_F(ElementwiseSourceAdapter, RejectsUnsupportedStructures) {
  const NegativeVariant variants[] = {
      {"reduction", reductionSource, "reduction"},
      {"nonidentity map", nonidentitySource, "identity"},
      {"dynamic shape", dynamicSource, "dynamic"},
      {"zero extent", zeroExtentSource, "positive"},
      {"f64", f64Source, "f32"},
      {"extra body computation", extraBodySource, "body"},
      {"destination read", initReadSource, "destination"},
      {"multiple results", multipleResultsSource, "result"},
      {"fastmath", fastmathSource, "fastmath"},
  };
  for (const NegativeVariant &variant : variants) {
    SCOPED_TRACE(variant.feature);
    expectDescribeError(variant.text.str(), variant.reason);
  }
}

TEST_F(ElementwiseSourceAdapter, RejectsExplicitRoundingModes) {
  auto fixture = loadFixture("elementwise.mlir");
  ASSERT_TRUE(fixture);
  const std::string addText = print(fixture->getOperation());
  const arith::RoundingMode modes[] = {
      arith::RoundingMode::to_nearest_even, arith::RoundingMode::downward,
      arith::RoundingMode::upward, arith::RoundingMode::toward_zero,
      arith::RoundingMode::to_nearest_away};
  for (llvm::StringRef operation : {"arith.addf", "arith.mulf"}) {
    SCOPED_TRACE(operation.str());
    std::string text = addText;
    text.replace(text.find("arith.addf"), std::string("arith.addf").size(),
                 operation.str());
    for (arith::RoundingMode mode : modes) {
      SCOPED_TRACE(arith::stringifyRoundingMode(mode).str());
      auto source = parse(text);
      ASSERT_TRUE(source);
      auto registry = workload::makeDefaultSourceAdapters();
      auto root = workload::selectSourceRoot(*source, {"elementwise", 0},
                                             registry, elementwiseKey);
      ASSERT_TRUE(bool(root)) << llvm::toString(root.takeError());
      Operation &scalar = (*root)->getRegion(0).front().front();
      scalar.setAttr("roundingmode",
                     arith::RoundingModeAttr::get(context.get(), mode));
      ASSERT_TRUE(succeeded(verify(*source)));
      const std::string before = print(source->getOperation());
      auto adapter = registry.resolve(*root, elementwiseKey);
      ASSERT_TRUE(bool(adapter)) << llvm::toString(adapter.takeError());
      auto described = (*adapter)->describe(*root);
      ASSERT_FALSE(bool(described));
      EXPECT_NE(llvm::toString(described.takeError()).find("rounding"),
                std::string::npos);
      EXPECT_EQ(print(source->getOperation()), before);
    }
  }
}

TEST_F(ElementwiseSourceAdapter, SourceTileRoleIsDeclaredButNotYetRealized) {
  auto source = loadFixture("elementwise.mlir");
  ASSERT_TRUE(source);
  workload::SourceAdapterRegistry registry =
      workload::makeDefaultSourceAdapters();
  auto root = workload::selectSourceRoot(*source, {"elementwise", 0}, registry,
                                         elementwiseKey);
  ASSERT_TRUE(bool(root)) << llvm::toString(root.takeError());
  auto adapter = registry.resolve(*root, elementwiseKey);
  ASSERT_TRUE(bool(adapter)) << llvm::toString(adapter.takeError());
  auto contract = (*adapter)->describe(*root);
  ASSERT_TRUE(bool(contract)) << llvm::toString(contract.takeError());

  perf::SearchSpace space = singletonSpace();
  perf::SearchParam tile;
  tile.name = "T0";
  tile.kind = "integer";
  tile.role = "axis:d0";
  tile.choices = {perf::SearchChoice(int64_t{8})};
  space.params.push_back(tile);

  auto schema = (*adapter)->bindings(*root, space);
  ASSERT_TRUE(bool(schema)) << llvm::toString(schema.takeError());
  ASSERT_EQ(schema->size(), 1u);
  EXPECT_EQ((*schema)[0].owner, workload::BindingOwner::Source);
  EXPECT_EQ((*schema)[0].role, "axis:d0");

  perf::Candidate candidate;
  candidate.values["T0"] = 8;
  candidate.id = perf::computeCandidateId(candidate.values, {});
  auto instance = tuning::instantiateCandidate(
      *source, space, candidate, *contract, sourceOptions(), registry);
  ASSERT_FALSE(bool(instance));
  EXPECT_NE(llvm::toString(instance.takeError()).find("axis_not_realized: T0"),
            std::string::npos);
}
