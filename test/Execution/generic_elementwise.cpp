//===- generic_elementwise.cpp - Generic source execution acceptance ------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework B7: the generic, non-contraction source path has to end in running
// code, not merely in a selected plan. These tests take the bounded structured
// elementwise source through the same adapter, mapping search and mapped
// compiler the contraction examples use, and check every output element
// against the scalar expression.
//
// Two facts are kept apart on purpose:
//
//   * the reference backend is portable and proves the generated program
//     computes what the source says, and
//   * the selected-target backend, on an AVX2 host, proves the selected
//     bundles and their width actually carried the computation.
//
// The rank gates are checked separately: rank-neutral analysis and lowering
// stay available, while the rank-two mapped ABI v1 refuses what it cannot
// describe with an explicit `runtime_capability:` diagnostic.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Conversion/MicroMapping/CandidateInstantiation.h"
#include "LLK/Conversion/MicroMapping/MappedTuningSession.h"
#include "LLK/Conversion/Workload/ElementwiseSourceAdapter.h"
#include "LLK/Conversion/Workload/SourceAdapter.h"
#include "LLK/Conversion/Workload/WorkloadContract.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/WorkloadGraph.h"
#include "LLK/Runtime/MappedExecutable.h"
#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TargetParser/Host.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifndef LLK_SOURCE_DIR
#error "LLK_SOURCE_DIR must name the repository root"
#endif

namespace {

using mlir::llk::mapping::MappingSearchOptions;
using mlir::llk::mapping::MappingSearchResult;
using mlir::llk::mapping::SearchMode;
using mlir::llk::workload::PortDirection;
using mlir::llk::workload::PortEffect;
using mlir::llk::workload::WorkloadContract;

mlir::DialectRegistry buildRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::scf::SCFDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect>();
  return registry;
}

mlir::OwningOpRef<mlir::ModuleOp> parseSource(mlir::MLIRContext &context,
                                              llvm::StringRef text) {
  return mlir::parseSourceString<mlir::ModuleOp>(text, &context);
}

std::string printModule(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  stream.flush();
  return text;
}

/// A rank-three kernel whose tensors the mapped ABI v1 cannot describe. Its
/// body is otherwise ordinary: analysis and lowering are rank-neutral, and only
/// execution preparation applies the ABI envelope.
constexpr llvm::StringLiteral kRankThreeKernel = R"mlir(
module {
  micro.kernel @rank3(%a: tensor<2x4x8xf32>, %b: tensor<2x4x8xf32>)
      -> tensor<2x4x8xf32> {
    %ta = micro.tile_view %a {shape = array<i64: 2, 4, 8>}
        : tensor<2x4x8xf32> -> !micro.tile<2x4x8xf32, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b {shape = array<i64: 2, 4, 8>}
        : tensor<2x4x8xf32> -> !micro.tile<2x4x8xf32, memory = #micro.memory<sram>>
    %sum = micro.vector "add" %ta, %tb
        : !micro.tile<2x4x8xf32, memory = #micro.memory<sram>>,
          !micro.tile<2x4x8xf32, memory = #micro.memory<sram>>
       -> !micro.tile<2x4x8xf32, memory = #micro.memory<acc>>
    micro.yield %sum : !micro.tile<2x4x8xf32, memory = #micro.memory<acc>>
  }
}
)mlir";

/// The versioned contract a direct MicroIR module may carry. Rank-three ports
/// are valid contract data; the execution envelope is a separate limit.
WorkloadContract rankThreeContract() {
  WorkloadContract contract;
  const std::vector<int64_t> shape{2, 4, 8};
  contract.ports = {
      {"input0", PortDirection::Input, PortEffect::Read, shape, "f32"},
      {"input1", PortDirection::Input, PortEffect::Read, shape, "f32"},
      {"output", PortDirection::Output, PortEffect::Write, shape, "f32"},
  };
  contract.numericalContract = "strict-f32";
  return contract;
}

/// A rank-two kernel whose retained contract claims an input is read-write.
/// The mapped ABI v1 hands inputs to callers as read-only buffers, so the
/// contract and the execution envelope disagree even though every port is
/// rank two.
constexpr llvm::StringLiteral kReadWriteInputKernel = R"mlir(
module {
  micro.kernel @readwrite(%a: tensor<2x8xf32>, %b: tensor<2x8xf32>)
      -> tensor<2x8xf32> {
    %ta = micro.tile_view %a {shape = array<i64: 2, 8>}
        : tensor<2x8xf32> -> !micro.tile<2x8xf32, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b {shape = array<i64: 2, 8>}
        : tensor<2x8xf32> -> !micro.tile<2x8xf32, memory = #micro.memory<sram>>
    %sum = micro.vector "add" %ta, %tb
        : !micro.tile<2x8xf32, memory = #micro.memory<sram>>,
          !micro.tile<2x8xf32, memory = #micro.memory<sram>>
       -> !micro.tile<2x8xf32, memory = #micro.memory<acc>>
    micro.yield %sum : !micro.tile<2x8xf32, memory = #micro.memory<acc>>
  }
}
)mlir";

bool avx2DisabledByTestOverride() {
  const char *disabled = std::getenv("LLK_TEST_DISABLE_AVX2");
  return disabled && std::strcmp(disabled, "1") == 0;
}

bool selectedAvx2Available() {
  if (avx2DisabledByTestOverride())
    return false;
  const std::string triple = llvm::sys::getDefaultTargetTriple();
  if (triple.rfind("x86_64-", 0) != 0)
    return false;
  const auto features = llvm::sys::getHostCPUFeatures();
  auto avx2 = features.find("avx2");
  return avx2 != features.end() && avx2->second;
}

//===----------------------------------------------------------------------===//
// Structured elementwise source
//===----------------------------------------------------------------------===//

/// The bounded structured fixture, parameterized by the one scalar operation
/// this source family admits.
std::string elementwiseSource(bool multiply) {
  const llvm::StringRef operation = multiply ? "arith.mulf" : "arith.addf";
  return R"mlir(
module {
  func.func @elementwise(%a: tensor<8x16xf32>, %b: tensor<8x16xf32>)
      -> tensor<8x16xf32> {
    %init = tensor.empty() : tensor<8x16xf32>
    %out = linalg.generic {
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : tensor<8x16xf32>, tensor<8x16xf32>)
      outs(%init : tensor<8x16xf32>) {
    ^bb0(%x: f32, %y: f32, %unused: f32):
      %result = )mlir" +
         operation.str() + R"mlir( %x, %y : f32
      linalg.yield %result : f32
    } -> tensor<8x16xf32>
    return %out : tensor<8x16xf32>
  }
}
)mlir";
}

llvm::Expected<WorkloadContract> describeElementwise(mlir::ModuleOp source) {
  mlir::llk::workload::SourceAdapterRegistry registry =
      mlir::llk::workload::makeDefaultSourceAdapters();
  auto root = mlir::llk::workload::selectSourceRoot(
      source, {"elementwise", 0}, registry,
      mlir::llk::workload::elementwiseSourceAdapterKey());
  if (!root)
    return root.takeError();
  auto adapter = registry.resolve(
      *root, mlir::llk::workload::elementwiseSourceAdapterKey());
  if (!adapter)
    return adapter.takeError();
  return (*adapter)->describe(*root);
}

/// Four bindings come from two one-choice axes; each test selects exactly one.
mlir::llk::perf::SearchSpace elementwiseSpace(int64_t t0, int64_t t1) {
  using namespace mlir::llk::perf;
  SearchSpace space;
  space.name = "elementwise";
  space.workload = "elementwise";
  SearchParam axis0;
  axis0.name = "T0";
  axis0.kind = "integer";
  axis0.role = "axis:d0";
  axis0.choices = {SearchChoice(t0)};
  SearchParam axis1;
  axis1.name = "T1";
  axis1.kind = "integer";
  axis1.role = "axis:d1";
  axis1.choices = {SearchChoice(t1)};
  space.params = {std::move(axis0), std::move(axis1)};
  space.objective.direction = ObjectiveDirection::Minimize;
  space.objective.primaryMetric = "latency_cycles";
  return space;
}

struct FloatBuffer {
  std::vector<float> storage;
  MemRef2D descriptor;

  FloatBuffer(int64_t rows, int64_t columns, float fill)
      : storage(static_cast<size_t>(rows * columns), fill) {
    descriptor =
        MemRef2D{storage.data(), storage.data(), 0, rows, columns, columns, 1};
  }

  llk::InvocationBuffer2D checked() const {
    return {descriptor, llk::InvocationElementType::F32,
            storage.size() * sizeof(float)};
  }
};

/// Everything one execution produced, kept next to the compiler evidence that
/// produced it so a selected run cannot be mistaken for a reference run.
struct ElementwiseExecution {
  std::unique_ptr<llk::MappedExecutable> executable;
  unsigned selectedGroupsVerified = 0;
  unsigned backendGroupsRealized = 0;
  unsigned referenceGroupsLowered = 0;
  std::string loweredIR;
};

llvm::Expected<ElementwiseExecution>
compileElementwise(mlir::MLIRContext &context, bool multiply, int64_t t0,
                   int64_t t1, llk::MappedBackend backend,
                   llk::MappedStop stop = llk::MappedStop::Executable) {
  using namespace mlir::llk;
  auto source = parseSource(context, elementwiseSource(multiply));
  if (!source)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot parse the elementwise fixture");
  auto target = target::avx2::createMappingTarget(LLK_SOURCE_DIR);
  if (!target)
    return target.takeError();
  auto contract = describeElementwise(*source);
  if (!contract)
    return contract.takeError();

  perf::SearchSpace space = elementwiseSpace(t0, t1);
  tuning::MappedTuningOptions options;
  options.source.sourceSymbol = "elementwise";
  options.source.adapterKey = workload::elementwiseSourceAdapterKey().str();
  options.mapping.mode = mapping::SearchMode::Exact;
  options.mapping.topK = 1;
  options.executable = true;
  options.backend = backend;

  auto report = tuning::runMappedTuningSession(*source, space, *contract,
                                               **target, options);
  if (!report)
    return report.takeError();
  if (report->ranked.empty()) {
    std::string reason = "no mapped plan for the elementwise source";
    if (!report->rejected.empty())
      reason += ": " + report->rejected.front().rejectionReason;
    return llvm::createStringError(llvm::inconvertibleErrorCode(), reason);
  }
  const tuning::MappedTuningCandidate &selected = report->ranked.front();

  tuning::CandidateInstantiationOptions sourceOptions = options.source;
  sourceOptions.machine = &(*target)->machine();
  auto instance = tuning::instantiateCandidate(
      *source, space, selected.ranking.candidate, *contract, sourceOptions);
  if (!instance)
    return instance.takeError();
  if (instance->sourceGraphHash != selected.sourceGraphHash ||
      instance->bindingHash != selected.bindingHash ||
      selected.plan.sourceBindingHash != selected.bindingHash)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "re-instantiated source identity changed");

  auto symbol = instance->kernel->getAttrOfType<mlir::StringAttr>("sym_name");
  if (!symbol)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "instantiated kernel has no symbol");

  ElementwiseExecution execution;
  llk::MappedCompileOptions compileOptions;
  compileOptions.entrySymbol = symbol.getValue().str();
  compileOptions.backend = backend;

  if (stop == llk::MappedStop::Lowered) {
    compileOptions.stop = llk::MappedStop::Lowered;
    auto lowered = llk::compileMappedKernel(instance->module.get(), **target,
                                            selected.plan, compileOptions);
    if (!lowered)
      return lowered.takeError();
    execution.selectedGroupsVerified = lowered->selectedGroupsVerified;
    execution.backendGroupsRealized = lowered->backendGroupsRealized;
    execution.referenceGroupsLowered = lowered->referenceGroupsLowered;
    execution.loweredIR = printModule(*lowered->module);
    return execution;
  }

  compileOptions.stop = llk::MappedStop::Executable;
  auto compiled = llk::compileMappedKernel(instance->module.get(), **target,
                                           selected.plan, compileOptions);
  if (!compiled)
    return compiled.takeError();
  if (!compiled->executable)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the elementwise kernel did not compile");
  execution.selectedGroupsVerified = compiled->selectedGroupsVerified;
  execution.backendGroupsRealized = compiled->backendGroupsRealized;
  execution.referenceGroupsLowered = compiled->referenceGroupsLowered;

  // The lowered form is separate evidence: it names the width the selected
  // backend reached, which the callable ABI never exposes.
  compileOptions.stop = llk::MappedStop::Lowered;
  auto lowered = llk::compileMappedKernel(instance->module.get(), **target,
                                          selected.plan, compileOptions);
  if (!lowered)
    return lowered.takeError();
  execution.loweredIR = printModule(*lowered->module);
  execution.executable = std::move(compiled->executable);
  return execution;
}

/// Runs the compiled kernel on literal inputs and checks every output element.
/// `destinationFill` is deliberately caller-chosen: the elementwise source
/// writes its full result, so the incoming destination must not change it.
void expectElementwiseResult(llk::MappedExecutable &executable, bool multiply,
                             float destinationFill) {
  constexpr int64_t rows = 8;
  constexpr int64_t columns = 16;
  constexpr size_t count = static_cast<size_t>(rows * columns);

  FloatBuffer a(rows, columns, 0.0f);
  FloatBuffer b(rows, columns, 0.0f);
  FloatBuffer destination(rows, columns, destinationFill);
  for (size_t i = 0; i < count; ++i) {
    a.storage[i] = static_cast<float>(i) / 8.0f;
    b.storage[i] = static_cast<float>(127 - i) / 16.0f;
  }
  const std::vector<float> originalA = a.storage;
  const std::vector<float> originalB = b.storage;
  const std::vector<float> originalDestination = destination.storage;

  // Sentinel words on both sides of the aligned output catch an off-by-one
  // write-back rather than hiding it in a larger buffer.
  std::vector<float> output(count + 2, -99.0f);
  MemRef2D outputDescriptor{output.data(), output.data() + 1, 0, rows,
                            columns,       columns,           1};
  llk::InvocationBuffer2D checkedOutput{outputDescriptor,
                                        llk::InvocationElementType::F32,
                                        output.size() * sizeof(float)};

  llvm::Error error = executable.invoke(
      {a.checked(), b.checked(), destination.checked()}, {checkedOutput});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));

  for (size_t i = 0; i < count; ++i) {
    const float expected =
        multiply ? a.storage[i] * b.storage[i] : a.storage[i] + b.storage[i];
    EXPECT_FLOAT_EQ(output[i + 1], expected) << "element " << i;
  }
  EXPECT_EQ(output.front(), -99.0f);
  EXPECT_EQ(output.back(), -99.0f);
  EXPECT_EQ(a.storage, originalA);
  EXPECT_EQ(b.storage, originalB);
  EXPECT_EQ(destination.storage, originalDestination);
}

//===----------------------------------------------------------------------===//
// Rank gating
//===----------------------------------------------------------------------===//

TEST(GenericElementwise, RankThreeContractIsAnalyzableButNotExecutable) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = parseSource(context, kRankThreeKernel);
  ASSERT_TRUE(module);

  // Static contract reporting works: the contract validates, canonicalizes and
  // can be retained on a kernel whose rank the execution ABI does not support.
  WorkloadContract contract = rankThreeContract();
  ASSERT_FALSE(static_cast<bool>(
      mlir::llk::workload::validateWorkloadContract(contract)));
  const std::string canonical =
      mlir::llk::workload::canonicalWorkloadContract(contract);
  EXPECT_FALSE(canonical.empty());
  module->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() != "micro.kernel")
      return;
    op->setAttr(
        "micro.workload_contract",
        mlir::llk::workload::encodeWorkloadContract(&context, contract));
  });

  llk::MappedCompileOptions options;
  options.entrySymbol = "rank3";
  options.backend = llk::MappedBackend::Reference;

  // Analysis and lowering are rank-neutral: this must reach lowered IR.
  options.stop = llk::MappedStop::Lowered;
  llvm::Expected<llk::MappedCompilation> lowered =
      llk::compileConcreteMicroKernel(*module, options);
  ASSERT_TRUE(static_cast<bool>(lowered))
      << llvm::toString(lowered.takeError());

  // The rank-two mapped ABI v1 is an execution envelope, not an analysis
  // limit. Asking for code fails explicitly and names the incompatibility.
  options.stop = llk::MappedStop::Executable;
  llvm::Expected<llk::MappedCompilation> executable =
      llk::compileConcreteMicroKernel(*module, options);
  ASSERT_FALSE(static_cast<bool>(executable));
  const std::string message = llvm::toString(executable.takeError());
  EXPECT_NE(message.find("runtime_capability:"), std::string::npos) << message;
  EXPECT_NE(message.find("rank-two"), std::string::npos) << message;
}

/// A rank-one reduction's selected lowering is valid IR: the selected emitter
/// carries it and nothing about the analysis depends on rank two. Only the
/// callable ABI refuses it, and `requireExecutable` on the binding does not
/// change that until code is actually requested.
TEST(GenericElementwise, RankOneSelectedLoweringIsAllowedButNotExecutable) {
  auto buffer = llvm::MemoryBuffer::getFile(
      std::string(LLK_SOURCE_DIR) +
      "/test/Execution/Inputs/issue129/reduce_sum.mlir");
  ASSERT_TRUE(static_cast<bool>(buffer)) << buffer.getError().message();
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module =
      mlir::parseSourceString<mlir::ModuleOp>((*buffer)->getBuffer(), &context);
  ASSERT_TRUE(module);

  auto target = mlir::llk::target::avx2::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  mlir::Operation *kernel = nullptr;
  module->walk([&](mlir::Operation *op) {
    if (!kernel && op->getName().getStringRef() == "micro.kernel")
      kernel = op;
  });
  ASSERT_NE(kernel, nullptr);
  llvm::Expected<mlir::llk::mapping::WorkloadGraph> graph =
      mlir::llk::mapping::extractWorkloadGraph(kernel);
  ASSERT_TRUE(static_cast<bool>(graph)) << llvm::toString(graph.takeError());

  mlir::llk::mapping::LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "f32";
  MappingSearchOptions searchOptions;
  searchOptions.mode = SearchMode::Exact;
  mlir::llk::mapping::CoveringSearch search(*graph, **target, context,
                                            layoutContext, searchOptions);
  llvm::Expected<MappingSearchResult> found = search.search();
  ASSERT_TRUE(static_cast<bool>(found)) << llvm::toString(found.takeError());
  ASSERT_FALSE(found->plans.empty());

  llk::MappedCompileOptions options;
  options.entrySymbol = "reduce_sum";
  options.backend = llk::MappedBackend::SelectedTarget;
  options.stop = llk::MappedStop::Lowered;
  // The binding contract stays complete; only the invocation envelope is
  // rank-two.
  options.requireExecutable = true;
  llvm::Expected<llk::MappedCompilation> lowered = llk::compileMappedKernel(
      *module, **target, found->plans.front(), options);
  ASSERT_TRUE(static_cast<bool>(lowered))
      << llvm::toString(lowered.takeError());
  EXPECT_GT(lowered->backendGroupsRealized, 0u);
  EXPECT_EQ(lowered->referenceGroupsLowered, 0u);

  options.stop = llk::MappedStop::Executable;
  llvm::Expected<llk::MappedCompilation> executable = llk::compileMappedKernel(
      *module, **target, found->plans.front(), options);
  ASSERT_FALSE(static_cast<bool>(executable));
  const std::string message = llvm::toString(executable.takeError());
  EXPECT_NE(message.find("runtime_capability:"), std::string::npos) << message;
  EXPECT_NE(message.find("rank-two"), std::string::npos) << message;
}

TEST(GenericElementwise, ReadWriteInputContractIsRejectedBeforeCode) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = parseSource(context, kReadWriteInputKernel);
  ASSERT_TRUE(module);
  module->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() != "micro.kernel")
      return;
    auto contract = rankThreeContract();
    contract.ports[0].shape = {2, 8};
    contract.ports[1].shape = {2, 8};
    contract.ports[2].shape = {2, 8};
    contract.ports[0].effect = PortEffect::ReadWrite;
    op->setAttr(
        "micro.workload_contract",
        mlir::llk::workload::encodeWorkloadContract(&context, contract));
  });

  llk::MappedCompileOptions options;
  options.entrySymbol = "readwrite";
  options.backend = llk::MappedBackend::Reference;
  options.stop = llk::MappedStop::Lowered;
  llvm::Expected<llk::MappedCompilation> lowered =
      llk::compileConcreteMicroKernel(*module, options);
  ASSERT_TRUE(static_cast<bool>(lowered))
      << llvm::toString(lowered.takeError());

  options.stop = llk::MappedStop::Executable;
  llvm::Expected<llk::MappedCompilation> executable =
      llk::compileConcreteMicroKernel(*module, options);
  ASSERT_FALSE(static_cast<bool>(executable));
  const std::string message = llvm::toString(executable.takeError());
  EXPECT_NE(message.find("runtime_capability:"), std::string::npos) << message;
  EXPECT_NE(message.find("read-only"), std::string::npos) << message;
}

//===----------------------------------------------------------------------===//
// Numerical acceptance
//===----------------------------------------------------------------------===//

TEST(GenericElementwise, SelectedAvx2LowersFullAndTiledAddAndMultiply) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  struct Case {
    bool multiply;
    int64_t t0;
    int64_t t1;
  };
  for (const Case &test : {Case{false, 8, 16}, Case{true, 8, 16},
                           Case{false, 4, 8}, Case{true, 4, 8}}) {
    SCOPED_TRACE((test.multiply ? "mul" : "add") +
                 std::string(test.t0 == 8 ? " full" : " tiled"));
    llvm::Expected<ElementwiseExecution> execution = compileElementwise(
        context, test.multiply, test.t0, test.t1,
        llk::MappedBackend::SelectedTarget, llk::MappedStop::Lowered);
    ASSERT_TRUE(static_cast<bool>(execution))
        << llvm::toString(execution.takeError());
    EXPECT_GT(execution->selectedGroupsVerified, 0u);
    EXPECT_GT(execution->backendGroupsRealized, 0u);
    EXPECT_EQ(execution->referenceGroupsLowered, 0u);
    EXPECT_NE(execution->loweredIR.find("vector<8xf32>"), std::string::npos)
        << execution->loweredIR;
  }
}

TEST(GenericElementwise, ReferenceBackendExecutesFullAndTiledAddAndMultiply) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  struct Case {
    bool multiply;
    int64_t t0;
    int64_t t1;
  };
  for (const Case &test : {Case{false, 8, 16}, Case{true, 8, 16},
                           Case{false, 4, 8}, Case{true, 4, 8}}) {
    SCOPED_TRACE((test.multiply ? "mul" : "add") +
                 std::string(test.t0 == 8 ? " full" : " tiled"));
    llvm::Expected<ElementwiseExecution> execution =
        compileElementwise(context, test.multiply, test.t0, test.t1,
                           llk::MappedBackend::Reference);
    ASSERT_TRUE(static_cast<bool>(execution))
        << llvm::toString(execution.takeError());
    EXPECT_NE(
        execution->executable->executionIdentity().find("backend=reference"),
        llvm::StringRef::npos);
    // A changed caller destination is not an input to the computation: the
    // full result is written regardless of what the buffer carried.
    expectElementwiseResult(*execution->executable, test.multiply, -77.0f);
    expectElementwiseResult(*execution->executable, test.multiply, 123.5f);
  }
}

TEST(GenericElementwise, SelectedAvx2ExecutesFullAndTiledAddAndMultiply) {
  if (!selectedAvx2Available()) {
#ifdef LLK_REQUIRE_SELECTED_TARGET
    if (avx2DisabledByTestOverride())
      FAIL() << "test override disabled required selected AVX2 execution";
    else
      FAIL() << "required selected AVX2 execution needs an x86_64 AVX2 host";
#else
    GTEST_SKIP() << "selected AVX2 execution requires an x86_64 AVX2 host";
#endif
  }

  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  struct Case {
    bool multiply;
    int64_t t0;
    int64_t t1;
  };
  for (const Case &test : {Case{false, 8, 16}, Case{true, 8, 16},
                           Case{false, 4, 8}, Case{true, 4, 8}}) {
    SCOPED_TRACE((test.multiply ? "mul" : "add") +
                 std::string(test.t0 == 8 ? " full" : " tiled"));
    llvm::Expected<ElementwiseExecution> execution =
        compileElementwise(context, test.multiply, test.t0, test.t1,
                           llk::MappedBackend::SelectedTarget);
    ASSERT_TRUE(static_cast<bool>(execution))
        << llvm::toString(execution.takeError());
    EXPECT_NE(execution->executable->executionIdentity().find(
                  "backend=selected-target"),
              llvm::StringRef::npos);
    // The selected bundles carried the computation: verified groups exist,
    // none fell back to the reference bridge, and the AVX2 lane width reached
    // the lowered IR.
    EXPECT_GT(execution->selectedGroupsVerified, 0u);
    EXPECT_GT(execution->backendGroupsRealized, 0u);
    EXPECT_EQ(execution->referenceGroupsLowered, 0u);
    EXPECT_NE(execution->loweredIR.find("vector<8xf32>"), std::string::npos)
        << execution->loweredIR;
    expectElementwiseResult(*execution->executable, test.multiply, -77.0f);
  }
}

} // namespace

int main(int argc, char **argv) {
  bool requireSelectedTarget = false;
  for (int index = 1; index < argc;) {
    if (std::strcmp(argv[index], "--require-selected-target") == 0) {
      requireSelectedTarget = true;
      for (int move = index; move + 1 < argc; ++move)
        argv[move] = argv[move + 1];
      --argc;
      continue;
    }
    ++index;
  }

#ifdef LLK_REQUIRE_SELECTED_TARGET
  if (!requireSelectedTarget) {
    std::fprintf(stderr, "GenericElementwiseAVX2Test requires "
                         "--require-selected-target\n");
    return 2;
  }
#else
  if (requireSelectedTarget) {
    std::fprintf(stderr, "--require-selected-target is only valid for the "
                         "selected-target executable\n");
    return 2;
  }
#endif

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
