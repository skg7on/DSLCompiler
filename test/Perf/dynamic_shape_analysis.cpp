//===- dynamic_shape_analysis.cpp - Sampled dynamic analysis tests -------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dynamic plans are rankable only on explicit concrete shape samples. These
// tests pin the facts that make that ranking trustworthy:
//
//   * an incomplete binding cannot produce a rankable one-iteration score;
//   * a complete sample resolves port dimensions and guards, while each
//     unrolled iteration still evaluates its own boundary extents;
//   * clipped external traffic is never inflated by padded local capacity;
//   * an all-zero extent is a known empty execution, not an unknown one;
//   * a sample whose unrolled event count exceeds the simulator's bound fails
//     explicitly instead of being truncated;
//   * weights combine per-sample costs deterministically;
//   * a malformed profile fails deterministically.
//
// The contraction fixtures run the real LLK-to-Micro export, so the shape
// arithmetic under test is the arithmetic the compiler emits.

#include "CompletePlanEvaluation.h"
#include "LLK/Conversion/LLKToMicro/LLKToMicro.h"
#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Conversion/MicroMapping/ShapeSpecialization.h"
#include "LLK/Conversion/Workload/SourceAdapter.h"
#include "LLK/Conversion/Workload/WorkloadContract.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Machine/MachineModelLoader.h"
#include "LLK/Mapping/CostModel.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/LlkMap.h"
#include "LLK/Mapping/PlanReport.h"
#include "LLK/Mapping/ShapeEvaluation.h"
#include "LLK/Mapping/WorkloadGraph.h"
#include "LLK/Perf/MicroDAG.h"
#include "LLK/Perf/MicroPerfReport.h"
#include "LLK/Perf/SelectedKernelAnalysis.h"
#include "LLK/Perf/ShapeProfile.h"
#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <string>

#ifndef LLK_MACHINE_DIR
#error "LLK_MACHINE_DIR must name the repository's machines directory"
#endif

namespace {

namespace shape = mlir::llk::shape;
namespace mapping = mlir::llk::mapping;
namespace perf = mlir::llk::perf;
namespace workload = mlir::llk::workload;

using mlir::ModuleOp;
using mlir::Operation;
using mlir::OwningOpRef;

constexpr llvm::StringLiteral kDynamicMatmul = R"mlir(
module {
  func.func @matmul_dynamic_root(%a: tensor<?x?xbf16>, %b: tensor<?x?xbf16>,
                                 %init: tensor<?x?xbf16>)
      -> tensor<?x?xbf16> {
    %y = llk.matmul ins(%a, %b : tensor<?x?xbf16>, tensor<?x?xbf16>)
        outs(%init : tensor<?x?xbf16>)
        {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
        -> tensor<?x?xbf16>
    return %y : tensor<?x?xbf16>
  }
}
)mlir";

constexpr llvm::StringLiteral kDynamicSwiglu = R"mlir(
module {
  func.func @swiglu_dynamic_root(%x: tensor<?x?xbf16>, %wg: tensor<?x?xbf16>,
                                 %wu: tensor<?x?xbf16>,
                                 %init: tensor<?x?xbf16>)
      -> tensor<?x?xbf16> {
    %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<?x?xbf16>,
                              tensor<?x?xbf16>, tensor<?x?xbf16>)
        outs(%init : tensor<?x?xbf16>)
        {accumulator_type = f32, activation = #llk.activation<silu>,
         math_mode = #llk.math_mode<bounded_fast>}
        -> tensor<?x?xbf16>
    return %y : tensor<?x?xbf16>
  }
}
)mlir";

/// One exported dynamic contraction, its source contract and the plain shape
/// contract the samples bind.
struct Exported {
  std::unique_ptr<mlir::MLIRContext> context;
  OwningOpRef<ModuleOp> module;
  Operation *kernel = nullptr;
  workload::WorkloadContract workloadContract;
  shape::ShapeContract contract;
};

llvm::Expected<Exported> exportSource(llvm::StringRef sourceText,
                                      llvm::StringRef sourceSymbol) {
  auto exported = std::make_unique<Exported>();
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::micro::MicroDialect>();
  registry.insert<mlir::llk::LLKDialect>();
  exported->context = std::make_unique<mlir::MLIRContext>(registry);

  exported->module = mlir::parseSourceString<ModuleOp>(
      sourceText, mlir::ParserConfig(exported->context.get()));
  if (!exported->module)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the dynamic source did not parse");

  workload::SourceAdapterRegistry adapters =
      workload::makeDefaultSourceAdapters();
  llvm::Expected<Operation *> root = workload::selectSourceRoot(
      *exported->module, {sourceSymbol.str(), 0}, adapters);
  if (!root)
    return root.takeError();
  llvm::Expected<const workload::SourceAdapter *> adapter =
      adapters.resolve(*root);
  if (!adapter)
    return adapter.takeError();
  llvm::Expected<workload::WorkloadContract> contract =
      (*adapter)->describe(*root);
  if (!contract)
    return contract.takeError();
  exported->workloadContract = std::move(*contract);

  mlir::PassManager passes(exported->context.get());
  passes.addPass(mlir::llk::createLLKToMicroPass());
  if (mlir::failed(passes.run(*exported->module)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the LLK-to-Micro export failed");

  exported->module->walk([&](Operation *op) {
    if (!exported->kernel && op->getName().getStringRef() == "micro.kernel")
      exported->kernel = op;
  });
  if (!exported->kernel)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the export produced no micro.kernel");

  llvm::Expected<mlir::DictionaryAttr> encoded =
      workload::encodeWorkloadContract(exported->context.get(),
                                       exported->workloadContract);
  if (!encoded)
    return encoded.takeError();
  exported->kernel->setAttr("micro.workload_contract", *encoded);

  llvm::Expected<shape::ShapeContract> described =
      mapping::shapeContractOf(exported->kernel);
  if (!described)
    return described.takeError();
  exported->contract = std::move(*described);
  return std::move(*exported);
}

llvm::Expected<Exported> exportMatmul(llvm::StringRef sourceText) {
  return exportSource(sourceText, "matmul_dynamic_root");
}

llvm::Expected<mlir::llk::machine::MachineModel> avx2Machine() {
  return mlir::llk::machine::loadMachineModel(std::string(LLK_MACHINE_DIR) +
                                              "/x86-avx2-v2.yaml");
}

/// The shipped AVX2 profile plus the one link its bounded contraction export
/// needs: the read initializer is loaded straight into the accumulator, and
/// the profile models no `dram -> acc` edge. The analysis is required to see a
/// concrete engine for every movement, so the fixture declares the modeled
/// path rather than accepting the deliberately-partial abstract `dma` pool.
llvm::Expected<std::string> analysisMachinePath() {
  std::string sourcePath = std::string(LLK_MACHINE_DIR) + "/x86-avx2-v2.yaml";
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(sourcePath);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "cannot read the machine profile");
  llvm::SmallString<256> path;
  int fd = -1;
  if (std::error_code error = llvm::sys::fs::createTemporaryFile(
          "llk-dynamic-shape", "yaml", fd, path))
    return llvm::createStringError(error, "cannot create a machine profile");
  // The bounded baseline keeps eight workers' accumulators live at once; the
  // shipped 8 KiB accumulator is smaller than that residency. The fixture
  // raises the local capacities so the analysis under test is the shape
  // analysis, not a capacity rejection of this particular schedule.
  std::string profile = (*buffer)->getBuffer().str();
  for (const char *oldValue :
       {"capacity_bytes: 8192", "capacity_bytes: 32768"}) {
    size_t position = 0;
    while ((position = profile.find(oldValue, position)) != std::string::npos) {
      profile.replace(position, std::strlen(oldValue),
                      "capacity_bytes: 1048576");
      position += std::strlen("capacity_bytes: 1048576");
    }
  }
  llvm::raw_fd_ostream stream(fd, /*shouldClose=*/true);
  stream << profile;
  stream << "  - id: dram_to_acc.0\n"
            "    source: dram.0\n"
            "    destination: acc.0\n"
            "    bandwidth_bytes_per_cycle: 64\n"
            "    latency_cycles: 220\n"
            "    transaction_bytes: 64\n"
            "    transfer_engines: [dma.0]\n"
            "  - id: acc_to_dram.0\n"
            "    source: acc.0\n"
            "    destination: dram.0\n"
            "    bandwidth_bytes_per_cycle: 64\n"
            "    latency_cycles: 220\n"
            "    transaction_bytes: 64\n"
            "    transfer_engines: [dma.0]\n";
  stream.close();
  if (stream.has_error())
    return llvm::createStringError(stream.error(),
                                   "cannot write the machine profile");
  return path.str().str();
}

shape::ShapeValues values(std::initializer_list<int64_t> mnik) {
  return {{"M", *mnik.begin()},
          {"N", *(mnik.begin() + 1)},
          {"K", *(mnik.begin() + 2)}};
}

/// One sample materialized and bound to the shipped AVX2 target. Analysis runs
/// on the bound kernel, exactly as `micro-perf` sees a selected plan: an
/// unmapped movement deliberately keeps the partial `dma` pool and is not
/// rankable.
struct PreparedSample {
  std::unique_ptr<mapping::MappingTarget> target;
  std::unique_ptr<llk::MappedCompilation> compilation;
  Operation *kernel = nullptr;
};

llvm::Expected<PreparedSample> prepareSample(const Exported &exported,
                                             const shape::ShapeValues &sample) {
  PreparedSample prepared;
  llvm::Expected<std::string> machinePath = analysisMachinePath();
  if (!machinePath)
    return machinePath.takeError();
  llvm::Expected<std::unique_ptr<mapping::MappingTarget>> target =
      mlir::llk::target::avx2::createMappingTarget(LLK_SOURCE_DIR,
                                                   *machinePath);
  if (!target)
    return target.takeError();
  prepared.target = std::move(*target);

  llvm::Expected<OwningOpRef<ModuleOp>> clone = mapping::materializeShapeSample(
      exported.module.get(), exported.contract, sample);
  if (!clone)
    return clone.takeError();
  Operation *kernel = nullptr;
  (*clone)->walk([&](Operation *op) {
    if (!kernel && op->getName().getStringRef() == "micro.kernel")
      kernel = op;
  });
  if (!kernel)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the sampled clone has no micro.kernel");

  llvm::Expected<mapping::WorkloadGraph> graph =
      mapping::extractWorkloadGraph(kernel);
  if (!graph)
    return graph.takeError();

  mapping::LayoutContext layout;
  layout.rank = 2;
  layout.elementType = exported.workloadContract.ports.front().elementType;
  mapping::MappingSearchOptions options;
  options.mode = mapping::SearchMode::Exact;
  mapping::CoveringSearch search(*graph, *prepared.target, *exported.context,
                                 layout, options);
  llvm::Expected<mapping::MappingSearchResult> result = search.search();
  if (!result)
    return result.takeError();
  if (result->plans.empty()) {
    std::string reason = "the sampled kernel has no complete plan";
    for (const mapping::Diagnostic &diagnostic : result->frontier.diagnostics)
      reason += "; " + diagnostic.message;
    return llvm::createStringError(llvm::inconvertibleErrorCode(), reason);
  }

  llk::MappedCompileOptions compileOptions;
  compileOptions.requireExecutable = true;
  compileOptions.stop = llk::MappedStop::MappedMicro;
  llvm::Expected<llk::MappedCompilation> compilation = llk::compileMappedKernel(
      clone->get(), *prepared.target, result->plans.front(), compileOptions);
  if (!compilation)
    return compilation.takeError();
  prepared.compilation =
      std::make_unique<llk::MappedCompilation>(std::move(*compilation));
  prepared.compilation->module->walk([&](Operation *op) {
    if (!prepared.kernel && op->getName().getStringRef() == "micro.kernel")
      prepared.kernel = op;
  });
  if (!prepared.kernel)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the bound sample has no micro.kernel");
  return prepared;
}

/// Analyzes `kernel` at `sample`, returning the complete shared analysis.
llvm::Expected<perf::SelectedKernelAnalysis>
analyzeSample(const Exported &exported, const shape::ShapeValues &sample) {
  llvm::Expected<PreparedSample> prepared = prepareSample(exported, sample);
  if (!prepared)
    return prepared.takeError();
  return perf::analyzeSelectedKernel(prepared->kernel,
                                     prepared->target->machine(),
                                     /*requireComplete=*/true);
}

//===----------------------------------------------------------------------===//
// Profile binding
//===----------------------------------------------------------------------===//

TEST(DynamicShapeAnalysis, PartialValuesRejected) {
  llvm::Expected<Exported> exported = exportMatmul(kDynamicMatmul);
  ASSERT_TRUE(bool(exported)) << llvm::toString(exported.takeError());

  // Missing a declared symbol is not "unknown work"; it is an invalid profile.
  perf::ShapeProfile missing;
  missing.samples.push_back({{{"M", 7}, {"N", 13}}, 1.0});
  llvm::Error missingError =
      perf::validateShapeProfile(exported->contract, missing);
  ASSERT_TRUE(bool(missingError));
  EXPECT_NE(llvm::toString(std::move(missingError)).find("K"),
            std::string::npos);

  // An unknown symbol is equally invalid: a profile cannot name a value the
  // contract never declared.
  perf::ShapeProfile unknown;
  unknown.samples.push_back({{{"M", 7}, {"N", 13}, {"K", 5}, {"X", 1}}, 1.0});
  llvm::Error unknownError =
      perf::validateShapeProfile(exported->contract, unknown);
  ASSERT_TRUE(bool(unknownError));
  EXPECT_NE(llvm::toString(std::move(unknownError)).find("unknown symbol"),
            std::string::npos);

  // A negative value is rejected before any unsigned conversion.
  EXPECT_TRUE(bool(shape::validateShapeValues(
      exported->contract, {{"M", -1}, {"N", 1}, {"K", 1}})));

  // A profile with no observations asks for a ranking it cannot support.
  perf::ShapeProfile empty;
  llvm::Error emptyError =
      perf::validateShapeProfile(exported->contract, empty);
  ASSERT_TRUE(bool(emptyError));
  EXPECT_NE(llvm::toString(std::move(emptyError)).find("no samples"),
            std::string::npos);
}

TEST(DynamicShapeAnalysis, MalformedProfile) {
  llvm::Expected<Exported> exported = exportMatmul(kDynamicMatmul);
  ASSERT_TRUE(bool(exported)) << llvm::toString(exported.takeError());

  auto parseAndValidate = [&](llvm::StringRef json) -> std::string {
    llvm::Expected<perf::ShapeProfile> parsed = perf::parseShapeProfile(json);
    if (!parsed)
      return llvm::toString(parsed.takeError());
    llvm::Error error = perf::validateShapeProfile(exported->contract, *parsed);
    return error ? llvm::toString(std::move(error)) : std::string();
  };

  EXPECT_NE(parseAndValidate("not json").find("invalid JSON"),
            std::string::npos);
  EXPECT_NE(parseAndValidate(R"({"samples":[]})").find("version is required"),
            std::string::npos);
  EXPECT_NE(parseAndValidate(
                R"({"version":2,"samples":[{"values":{"M":7,"N":13,"K":5}}]})")
                .find("unsupported version"),
            std::string::npos);
  EXPECT_NE(
      parseAndValidate(R"({"version":1,"samples":[],"aggregation":"median"})")
          .find("unsupported aggregation"),
      std::string::npos);
  EXPECT_NE(
      parseAndValidate(
          R"({"version":1,"samples":[{"values":{"M":7,"N":13,"K":5}}],"aggregation":"median"})")
          .find("unsupported aggregation"),
      std::string::npos);
  EXPECT_NE(
      parseAndValidate(
          R"({"version":1,"samples":[{"values":{"M":7,"N":13,"K":5}}],"max_variants":0})")
          .find("max_variants"),
      std::string::npos);
  EXPECT_NE(
      parseAndValidate(
          R"({"version":1,"samples":[{"values":{"M":7,"N":13,"K":5},"weight":0}]})")
          .find("finite and positive"),
      std::string::npos);
  EXPECT_NE(
      parseAndValidate(
          R"({"version":1,"samples":[{"values":{"M":7,"N":13,"K":5}}],"extra":1})")
          .find("unknown field"),
      std::string::npos);
  EXPECT_NE(
      parseAndValidate(
          R"({"version":1,"samples":[{"values":{"M":7.5,"N":13,"K":5}}]})")
          .find("must be an integer"),
      std::string::npos);
  EXPECT_NE(
      parseAndValidate(
          R"({"version":1,"samples":[{"values":{"M":7,"M":8,"N":13,"K":5}}]})")
          .find("duplicate key 'M'"),
      std::string::npos);

  // A non-finite weight is rejected by semantic validation even when it came
  // from a constructed profile rather than JSON.
  perf::ShapeProfile nonFinite;
  nonFinite.samples.push_back(
      {values({7, 13, 5}), std::numeric_limits<double>::quiet_NaN()});
  EXPECT_TRUE(bool(perf::validateShapeProfile(exported->contract, nonFinite)));

  // The version-1 schema parses and validates as written in the design.
  llvm::Expected<perf::ShapeProfile> good = perf::parseShapeProfile(
      R"({"version":1,"samples":[{"values":{"M":7,"N":13,"K":5},"weight":1.0}],"aggregation":"weighted_mean","max_variants":1})");
  ASSERT_TRUE(bool(good)) << llvm::toString(good.takeError());
  EXPECT_FALSE(bool(perf::validateShapeProfile(exported->contract, *good)));
}

TEST(DynamicShapeAnalysis, FusedSampleAnalyzes) {
  // The machinery is operator-independent: the same sample binding, clipped
  // boundary analysis and exact reduction facts apply to the two-arm fused
  // contraction.
  llvm::Expected<Exported> exported =
      exportSource(kDynamicSwiglu, "swiglu_dynamic_root");
  ASSERT_TRUE(bool(exported)) << llvm::toString(exported.takeError());
  llvm::Expected<perf::SelectedKernelAnalysis> sample =
      analyzeSample(*exported, values({7, 13, 5}));
  ASSERT_TRUE(bool(sample)) << llvm::toString(sample.takeError());
  EXPECT_TRUE(sample->complete);
  EXPECT_FALSE(sample->events.events.empty());
  EXPECT_GT(sample->cost.dramBytes, 0u);
}

//===----------------------------------------------------------------------===//
// No sample, no rankable score
//===----------------------------------------------------------------------===//

TEST(DynamicShapeAnalysis, NoValuesIncomplete) {
  llvm::Expected<Exported> exported = exportMatmul(kDynamicMatmul);
  ASSERT_TRUE(bool(exported)) << llvm::toString(exported.takeError());
  llvm::Expected<mlir::llk::machine::MachineModel> machine = avx2Machine();
  ASSERT_TRUE(bool(machine)) << llvm::toString(machine.takeError());

  // The runtime loop bounds cannot be charged. A strict analysis refuses the
  // stream instead of presenting one iteration as the whole kernel.
  llvm::Expected<perf::SelectedKernelAnalysis> strict =
      perf::analyzeSelectedKernel(exported->kernel, *machine,
                                  /*requireComplete=*/true);
  ASSERT_FALSE(bool(strict));
  const std::string reason = llvm::toString(strict.takeError());
  EXPECT_NE(reason.find("incomplete"), std::string::npos) << reason;
  EXPECT_NE(reason.find("non-static bounds"), std::string::npos) << reason;

  // The lenient report keeps the partial cost but labels it incomplete.
  llvm::Expected<perf::MicroPerfReport> report =
      perf::analyzeKernel(exported->kernel, *machine, /*level=*/1);
  ASSERT_TRUE(bool(report)) << llvm::toString(report.takeError());
  EXPECT_FALSE(report->complete);
  bool namesBounds = false;
  for (const std::string &incomplete : report->incompleteReasons)
    namesBounds |= incomplete.find("non-static bounds") != std::string::npos;
  EXPECT_TRUE(namesBounds);
}

//===----------------------------------------------------------------------===//
// Sample materialization and per-iteration boundary facts
//===----------------------------------------------------------------------===//

TEST(DynamicShapeAnalysis, SampleMatchesStaticClone) {
  llvm::Expected<Exported> exported = exportMatmul(kDynamicMatmul);
  ASSERT_TRUE(bool(exported)) << llvm::toString(exported.takeError());

  llvm::Expected<OwningOpRef<ModuleOp>> clone = mapping::materializeShapeSample(
      exported->module.get(), exported->contract, values({7, 13, 5}));
  ASSERT_TRUE(bool(clone)) << llvm::toString(clone.takeError());

  // Only shape computations were resolved: the executable tile types keep their
  // static capacity and the original source keeps its dynamic dimensions.
  unsigned dims = 0;
  unsigned loops = 0;
  unsigned dynamicLoops = 0;
  (*clone)->walk([&](Operation *op) {
    llvm::StringRef name = op->getName().getStringRef();
    dims += name == "tensor.dim";
    if (name != "micro.for" && name != "micro.spatial_for")
      return;
    ++loops;
    // The sampled clone is an exact-shape analysis artifact: every loop bound
    // is folded, while the loop structure itself is preserved.
    if (!mlir::getConstantIntValue(op->getOperand(1)))
      ++dynamicLoops;
  });
  EXPECT_EQ(dims, 0u);
  EXPECT_GT(loops, 0u);
  EXPECT_EQ(dynamicLoops, 0u);

  llvm::Expected<perf::SelectedKernelAnalysis> sample =
      analyzeSample(*exported, values({7, 13, 5}));
  ASSERT_TRUE(bool(sample)) << llvm::toString(sample.takeError());
  EXPECT_TRUE(sample->complete);
  EXPECT_FALSE(sample->events.events.empty());

  // The same sampled workload must produce the same work/bytes/capacity twice.
  llvm::Expected<perf::SelectedKernelAnalysis> again =
      analyzeSample(*exported, values({7, 13, 5}));
  ASSERT_TRUE(bool(again)) << llvm::toString(again.takeError());
  EXPECT_EQ(sample->cost.latencyCycles, again->cost.latencyCycles);
  EXPECT_EQ(sample->cost.dramBytes, again->cost.dramBytes);
  EXPECT_EQ(sample->cost.localBytes, again->cost.localBytes);
  EXPECT_EQ(sample->peakBytes, again->peakBytes);
  EXPECT_EQ(sample->trafficBytes, again->trafficBytes);
}

TEST(DynamicShapeAnalysis, EdgeTrafficAndValidK) {
  llvm::Expected<Exported> exported = exportMatmul(kDynamicMatmul);
  ASSERT_TRUE(bool(exported)) << llvm::toString(exported.takeError());

  llvm::Expected<PreparedSample> prepared =
      prepareSample(*exported, values({9, 33, 35}));
  ASSERT_TRUE(bool(prepared)) << llvm::toString(prepared.takeError());

  llvm::Expected<perf::MicroDAG> dag =
      perf::buildMicroDAG(prepared->kernel, prepared->target->machine());
  ASSERT_TRUE(bool(dag)) << llvm::toString(dag.takeError());

  // External traffic is the valid rectangle, exactly the sum the compiler's
  // clipped accesses touch. The padded local tiles remain full capacity.
  uint64_t externalBytes = 0;
  uint64_t fillElements = 0;
  bool sawTailLoad = false;
  for (const perf::MicroEvent &event : dag->events) {
    externalBytes += event.sourceBytes ? event.sourceBytes : 0;
    fillElements += event.fillElements;
    if (event.kind == perf::EventKind::Load && event.sourceBytes == 1 * 3 * 2 &&
        event.destinationBytes == 8 * 32 * 2)
      sawTailLoad = true;
  }
  EXPECT_TRUE(sawTailLoad) << "the M=1,K=3 edge load must copy 6 bytes into a "
                              "full 8x32 tile";
  EXPECT_GT(fillElements, 0u);
  EXPECT_EQ(externalBytes, 3534u * 2u);

  // Only the real K terms are reduction work: the full 32-term iteration and
  // the 3-term tail both appear, and the padded 32-term count never replaces
  // the tail.
  bool sawFullK = false;
  bool sawTailK = false;
  for (const perf::MicroEvent &event : dag->events) {
    if (event.kind != perf::EventKind::Mma)
      continue;
    sawFullK |= event.workItems == 8u * 32u * 32u;
    sawTailK |= event.workItems == 8u * 32u * 3u;
  }
  EXPECT_TRUE(sawFullK);
  EXPECT_TRUE(sawTailK);
}

TEST(DynamicShapeAnalysis, EmptyHasZeroEvents) {
  llvm::Expected<Exported> exported = exportMatmul(kDynamicMatmul);
  ASSERT_TRUE(bool(exported)) << llvm::toString(exported.takeError());
  llvm::Expected<mlir::llk::machine::MachineModel> machine = avx2Machine();
  ASSERT_TRUE(bool(machine)) << llvm::toString(machine.takeError());

  llvm::Expected<PreparedSample> prepared =
      prepareSample(*exported, values({0, 13, 5}));
  ASSERT_TRUE(bool(prepared)) << llvm::toString(prepared.takeError());

  llvm::Expected<perf::SelectedKernelAnalysis> analysis =
      perf::analyzeSelectedKernel(prepared->kernel, prepared->target->machine(),
                                  /*requireComplete=*/true);
  ASSERT_TRUE(bool(analysis)) << llvm::toString(analysis.takeError());
  EXPECT_TRUE(analysis->complete);
  EXPECT_TRUE(analysis->events.events.empty());
  EXPECT_EQ(analysis->cost.latencyCycles, 0.0);
  EXPECT_EQ(analysis->cost.localBytes, 0u);

  // A known zero trip count is not an unknown one: extraction records zero
  // occurrences rather than leaving the multiplicity unset.
  llvm::Expected<mapping::WorkloadGraph> graph =
      mapping::extractWorkloadGraph(prepared->kernel);
  ASSERT_TRUE(bool(graph)) << llvm::toString(graph.takeError());
  bool sawZero = false;
  bool sawUnknown = false;
  for (const mapping::WorkloadNode &node : graph->getNodes()) {
    sawZero |= node.executionMultiplicity && *node.executionMultiplicity == 0;
    sawUnknown |= !node.executionMultiplicity;
  }
  EXPECT_TRUE(sawZero);
  EXPECT_FALSE(sawUnknown);
}

TEST(DynamicShapeAnalysis, ResidencyIndependentOfGrid) {
  llvm::Expected<Exported> exported = exportMatmul(kDynamicMatmul);
  ASSERT_TRUE(bool(exported)) << llvm::toString(exported.takeError());

  llvm::Expected<perf::SelectedKernelAnalysis> small =
      analyzeSample(*exported, values({8, 32, 32}));
  ASSERT_TRUE(bool(small)) << llvm::toString(small.takeError());
  llvm::Expected<perf::SelectedKernelAnalysis> large =
      analyzeSample(*exported, values({1024, 64, 32}));
  ASSERT_TRUE(bool(large)) << llvm::toString(large.takeError());

  // Local materialized capacity is a function of the static tile and pipeline
  // shape, never of the number of logical tiles the runtime grid will produce.
  EXPECT_EQ(small->peakBytes, large->peakBytes);
  EXPECT_LT(small->cost.dramBytes, large->cost.dramBytes);
}

TEST(DynamicShapeAnalysis, EventLimitExplicit) {
  llvm::Expected<Exported> exported = exportMatmul(kDynamicMatmul);
  ASSERT_TRUE(bool(exported)) << llvm::toString(exported.takeError());
  llvm::Expected<mlir::llk::machine::MachineModel> machine = avx2Machine();
  ASSERT_TRUE(bool(machine)) << llvm::toString(machine.takeError());

  llvm::Expected<PreparedSample> prepared =
      prepareSample(*exported, values({600, 600, 600}));
  ASSERT_TRUE(bool(prepared)) << llvm::toString(prepared.takeError());

  llvm::Expected<perf::MicroDAG> dag =
      perf::buildMicroDAG(prepared->kernel, *machine);
  ASSERT_FALSE(bool(dag));
  const std::string reason = llvm::toString(dag.takeError());
  EXPECT_NE(reason.find("more than"), std::string::npos) << reason;
  EXPECT_NE(reason.find(llvm::Twine(perf::kMaxEvents).str()), std::string::npos)
      << reason;
}

//===----------------------------------------------------------------------===//
// Plan and materialized analysis agree at each sample
//===----------------------------------------------------------------------===//

TEST(DynamicShapeAnalysis, PlanMaterializedParity) {
  llvm::Expected<Exported> exported = exportMatmul(kDynamicMatmul);
  ASSERT_TRUE(bool(exported)) << llvm::toString(exported.takeError());

  llvm::Expected<mapping::WorkloadGraph> graph =
      mapping::extractWorkloadGraph(exported->kernel);
  ASSERT_TRUE(bool(graph)) << llvm::toString(graph.takeError());
  llvm::Expected<std::string> machinePath = analysisMachinePath();
  ASSERT_TRUE(bool(machinePath)) << llvm::toString(machinePath.takeError());
  llvm::Expected<std::unique_ptr<mapping::MappingTarget>> target =
      mlir::llk::target::avx2::createMappingTarget(LLK_SOURCE_DIR,
                                                   *machinePath);
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());

  mapping::LayoutContext layout;
  layout.rank = 2;
  layout.elementType = exported->workloadContract.ports.front().elementType;
  mapping::MappingSearchOptions options;
  options.mode = mapping::SearchMode::Exact;
  mapping::CoveringSearch search(*graph, **target, *exported->context, layout,
                                 options);
  llvm::Expected<mapping::MappingSearchResult> found = search.search();
  ASSERT_TRUE(bool(found)) << llvm::toString(found.takeError());
  ASSERT_FALSE(found->plans.empty());

  perf::ShapeProfile shapeProfile;
  shapeProfile.samples.push_back({values({7, 13, 5}), 1.0});
  mapping::ShapeSampleOptions sampled;
  sampled.contract = &exported->contract;
  sampled.profile = &shapeProfile;

  llvm::Expected<mapping::CompletePlanEvaluation> evaluated =
      mapping::evaluateCompletePlan(
          exported->module.get(), *graph, **target, found->plans.front(),
          mapping::BindContract::Executable, std::nullopt,
          mapping::AnalysisRequirement::Complete, sampled);
  ASSERT_TRUE(bool(evaluated)) << llvm::toString(evaluated.takeError());
  ASSERT_TRUE(evaluated->plan.has_value())
      << (evaluated->rejection ? evaluated->rejection->message : "no plan");
  const mapping::CoveringPlan &plan = *evaluated->plan;
  ASSERT_EQ(plan.shapeSamples.size(), 1u);
  EXPECT_EQ(plan.shapeAggregation, "weighted_mean");
  EXPECT_EQ(plan.shapeProfileCanonical,
            perf::canonicalShapeProfile(shapeProfile));
  EXPECT_EQ(plan.scoreSource, mapping::PlanScoreSource::Schedule);

  // The plan's sampled cost is the materialized IR's own analysis.
  llvm::Expected<perf::SelectedKernelAnalysis> materialized =
      analyzeSample(*exported, values({7, 13, 5}));
  ASSERT_TRUE(bool(materialized)) << llvm::toString(materialized.takeError());
  EXPECT_DOUBLE_EQ(plan.shapeSamples.front().cost.latencyCycles,
                   materialized->cost.latencyCycles);
  EXPECT_EQ(plan.shapeSamples.front().cost.dramBytes,
            materialized->cost.dramBytes);
  EXPECT_EQ(plan.shapeSamples.front().cost.localBytes,
            materialized->cost.localBytes);
  EXPECT_EQ(plan.shapeSamples.front().fillElements, materialized->fillElements);
  EXPECT_FALSE(plan.shapeSamples.front().peakBytes.empty());
  EXPECT_FALSE(materialized->peakBytes.empty());
  EXPECT_DOUBLE_EQ(plan.totalCost.latencyCycles,
                   plan.shapeSamples.front().cost.latencyCycles);

  // The plan report records the profile identity, policy, bounds and the
  // observation itself, so a reader never has to guess what the score meant.
  mapping::MappingSearchResult reported;
  reported.plans.push_back(plan);
  const std::string report = mapping::writePlanReport(
      reported, (*target)->machine(), **target, options, /*moduleHash=*/0);
  EXPECT_NE(report.find("\"shapeProfile\""), std::string::npos);
  EXPECT_NE(report.find(perf::canonicalShapeProfile(shapeProfile)),
            std::string::npos);
  EXPECT_NE(report.find("\"fillElements\""), std::string::npos);
}

//===----------------------------------------------------------------------===//
// Weighted aggregation
//===----------------------------------------------------------------------===//

TEST(DynamicShapeAnalysis, WeightedRanking) {
  mapping::Cost planA = {100.0, 10, 100, 0, 0.1, 0.1};
  mapping::Cost planB = {90.0, 10, 100, 0, 0.1, 0.1};

  // Plan A: 100 cycles on the light sample, 90 on the heavy one.
  llvm::Expected<mapping::Cost> weightedA =
      mapping::weightedMeanCost({{planA, 1.0}, {planB, 4.0}});
  ASSERT_TRUE(bool(weightedA)) << llvm::toString(weightedA.takeError());
  EXPECT_DOUBLE_EQ(weightedA->latencyCycles, 92.0);

  // Plan B: 80 cycles on the light sample, 98 on the heavy one. Its unweighted
  // mean is better, but the heavy observation is the one that decides.
  mapping::Cost fastLight = {80.0, 10, 100, 0, 0.1, 0.1};
  mapping::Cost slowHeavy = {98.0, 10, 100, 0, 0.1, 0.1};
  llvm::Expected<mapping::Cost> weightedB =
      mapping::weightedMeanCost({{fastLight, 1.0}, {slowHeavy, 4.0}});
  ASSERT_TRUE(bool(weightedB)) << llvm::toString(weightedB.takeError());
  EXPECT_DOUBLE_EQ(weightedB->latencyCycles, 94.4);

  llvm::Expected<mapping::Cost> unweightedA =
      mapping::weightedMeanCost({{planA, 1.0}, {planB, 1.0}});
  llvm::Expected<mapping::Cost> unweightedB =
      mapping::weightedMeanCost({{fastLight, 1.0}, {slowHeavy, 1.0}});
  ASSERT_TRUE(bool(unweightedA));
  ASSERT_TRUE(bool(unweightedB));

  mapping::ObjectiveOrder order;
  EXPECT_TRUE(mapping::costLess(*weightedA, *weightedB, order));
  EXPECT_TRUE(mapping::costLess(*unweightedB, *unweightedA, order));

  EXPECT_FALSE(
      bool(mapping::weightedMeanCost(llvm::ArrayRef<mapping::WeightedCost>{})));
  EXPECT_FALSE(bool(mapping::weightedMeanCost({{planA, 0.0}})));
  EXPECT_FALSE(bool(mapping::weightedMeanCost({{planA, -1.0}})));
  EXPECT_FALSE(bool(mapping::weightedMeanCost(
      {{planA, std::numeric_limits<double>::quiet_NaN()}})));
}

//===----------------------------------------------------------------------===//
// Shape index evaluation
//===----------------------------------------------------------------------===//

TEST(DynamicShapeAnalysis, ShapeEvaluationBindsTheIteration) {
  // A boundary extent `min(8, 13 - i*8)` is concrete only once the induction
  // variable is bound. This is the exact form the bounded exporter emits.
  Exported exported; // only used for its MLIR context
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  exported.context = std::make_unique<mlir::MLIRContext>(registry);
  OwningOpRef<ModuleOp> module = mlir::parseSourceString<ModuleOp>(
      R"mlir(
module {
  func.func @f(%i: index, %d: index) -> index {
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %off = arith.muli %i, %c8 : index
    %rem = arith.subi %d, %off : index
    %valid = arith.minsi %c8, %rem : index
    return %valid : index
  }
}
)mlir",
      mlir::ParserConfig(exported.context.get()));
  ASSERT_TRUE(bool(module));
  auto function = *module->getBody()->getOps<mlir::func::FuncOp>().begin();
  Operation *validOp = nullptr;
  function.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "arith.minsi")
      validOp = op;
  });
  ASSERT_NE(validOp, nullptr);

  mapping::ShapeEvaluationContext context;
  context.bind(function.getArgument(0), 0);
  context.bind(function.getArgument(1), 13);
  llvm::Expected<int64_t> first =
      mapping::evaluateShapeIndex(validOp->getResult(0), context);
  ASSERT_TRUE(bool(first)) << llvm::toString(first.takeError());
  EXPECT_EQ(*first, 8);

  context.bind(function.getArgument(0), 1);
  llvm::Expected<int64_t> second =
      mapping::evaluateShapeIndex(validOp->getResult(0), context);
  ASSERT_TRUE(bool(second)) << llvm::toString(second.takeError());
  EXPECT_EQ(*second, 5);

  // An unsupported expression is an explicit error, never a default value.
  OwningOpRef<ModuleOp> unsupported = mlir::parseSourceString<ModuleOp>(
      R"mlir(
module {
  func.func @f(%i: index) -> index {
    %c3 = arith.constant 3 : index
    %x = arith.divsi %i, %c3 : index
    return %x : index
  }
}
)mlir",
      mlir::ParserConfig(exported.context.get()));
  ASSERT_TRUE(bool(unsupported));
  auto f2 = *unsupported->getBody()->getOps<mlir::func::FuncOp>().begin();
  Operation *division = nullptr;
  f2.walk([&](Operation *op) {
    if (op->getName().getStringRef() == "arith.divsi")
      division = op;
  });
  ASSERT_NE(division, nullptr);
  EXPECT_FALSE(bool(mapping::evaluateShapeIndex(
      division->getResult(0), mapping::ShapeEvaluationContext{})));
}

} // namespace
