//===- e2e_workflow.cpp - The whole mapping workflow, end to end ---------===//
//
// Issue #53: one test that runs the documented chain on the same fixture
// against two targets, so a break anywhere along it -- extraction, matching,
// placement, connections, covering, binding, verification -- fails here rather
// than in a per-layer test that no longer reflects the whole.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/PlanBinder.h"
#include "LLK/Runtime/MappedExecutable.h"
#include "LLK/Target/GenericAccelerator/Mapping/GenericAcceleratorMappingTarget.h"
#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include "LLK/Dialect/Micro/MicroDialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

using namespace mlir;
using namespace mlir::llk::mapping;

namespace avx2_mapping = mlir::llk::target::avx2;
namespace accel_mapping = mlir::llk::target::generic_accel;

#ifndef LLK_SOURCE_DIR
#error "LLK_SOURCE_DIR must name the repository root"
#endif

namespace {

/// A concrete kernel with both kinds of work the mapper has to place: a staged
/// copy and a vector add.
constexpr llvm::StringLiteral kKernel = R"mlir(
module {
  micro.kernel @mapped {
    %ext = tensor.empty() : tensor<8x8xf32>
    %t, %tok = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    %v = micro.tile_view %t {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %v, %v : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)mlir";

struct Parsed {
  std::unique_ptr<MLIRContext> context;
  OwningOpRef<ModuleOp> module;
  Operation *kernel = nullptr;
};

Parsed parseKernel(llvm::StringRef text) {
  Parsed parsed;
  parsed.context = std::make_unique<MLIRContext>();
  parsed.context->getOrLoadDialect<micro::MicroDialect>();
  parsed.context->getOrLoadDialect<tensor::TensorDialect>();
  parsed.module = parseSourceString<ModuleOp>(text, parsed.context.get());
  if (parsed.module)
    parsed.module->walk([&](Operation *op) {
      if (!parsed.kernel && op->getName().getStringRef() == "micro.kernel")
        parsed.kernel = op;
    });
  return parsed;
}

/// Everything the documented chain produces, so one test can assert on all of
/// it.
struct Workflow {
  WorkloadGraph graph;
  MappingSearchResult search;
  BoundPlan bound;
  std::string mappedText;
};

/// Runs the chain: extract the workload graph, search for a plan, bind it, and
/// print the result. Fails the test at whichever step fails.
llvm::Expected<Workflow> runWorkflow(MLIRContext &context, ModuleOp module,
                                     MappingTarget &target) {
  Workflow workflow;

  llvm::Expected<WorkloadGraph> graph = extractWorkloadGraph([&] {
    Operation *kernel = nullptr;
    module->walk([&](Operation *op) {
      if (!kernel && op->getName().getStringRef() == "micro.kernel")
        kernel = op;
    });
    return kernel;
  }());
  if (!graph)
    return graph.takeError();
  workflow.graph = std::move(*graph);

  LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "f32";
  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(workflow.graph, target, context, layoutContext,
                        options);
  llvm::Expected<MappingSearchResult> result = search.search();
  if (!result)
    return result.takeError();
  workflow.search = std::move(*result);
  if (workflow.search.plans.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the workflow produced no plan");

  llvm::Expected<BoundPlan> bound =
      bindPlan(module, workflow.search.plans.front(), target);
  if (!bound)
    return bound.takeError();
  workflow.bound = std::move(*bound);

  llvm::raw_string_ostream stream(workflow.mappedText);
  workflow.bound.module->print(stream);
  return std::move(workflow);
}

} // namespace

TEST(E2EWorkflow, MapsTheSameKernelOnAVX2) {
  Parsed parsed = parseKernel(kKernel);
  ASSERT_TRUE(parsed.module);

  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      avx2_mapping::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  llvm::Expected<Workflow> workflow =
      runWorkflow(*parsed.context, *parsed.module, **target);
  ASSERT_TRUE(static_cast<bool>(workflow))
      << llvm::toString(workflow.takeError());

  // The workload graph found the copy and the vector, with the view folded
  // away.
  EXPECT_EQ(workflow->graph.getNodes().size(), 2u);
  // The plan covers both, and the binder recorded it.
  EXPECT_EQ(workflow->search.plans.front().placements.size(), 2u);
  EXPECT_TRUE(workflow->bound.kernel->hasAttr("micro.plan"));
  EXPECT_TRUE(workflow->bound.kernel->hasAttr("micro.routes"));
  // Layered verification passes on what was written.
  EXPECT_FALSE(static_cast<bool>(
      verifyMappedMicroIR(*workflow->bound.module, **target)));
  // The source module was not touched.
  EXPECT_FALSE(parsed.kernel->hasAttr("micro.plan"));
}

TEST(E2EWorkflow, MapsTheSameKernelOnASecondTarget) {
  Parsed parsed = parseKernel(kKernel);
  ASSERT_TRUE(parsed.module);

  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      accel_mapping::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  llvm::Expected<Workflow> workflow =
      runWorkflow(*parsed.context, *parsed.module, **target);
  ASSERT_TRUE(static_cast<bool>(workflow))
      << llvm::toString(workflow.takeError());

  EXPECT_EQ(workflow->graph.getNodes().size(), 2u);
  EXPECT_EQ(workflow->search.plans.front().placements.size(), 2u);
  EXPECT_FALSE(static_cast<bool>(
      verifyMappedMicroIR(*workflow->bound.module, **target)));

  // The rules it selected are the accelerator's, not AVX2's -- which is the
  // point: the same kernel, the same chain, no generic code changed.
  for (const PlanPlacement &placement :
       workflow->search.plans.front().placements)
    EXPECT_EQ(placement.rule.rfind("accel.", 0), 0u) << placement.rule;
}

TEST(E2EWorkflow, RepeatedRunsProduceIdenticalPlansAndIr) {
  Parsed first = parseKernel(kKernel);
  Parsed second = parseKernel(kKernel);
  ASSERT_TRUE(first.module);
  ASSERT_TRUE(second.module);

  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      avx2_mapping::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(static_cast<bool>(target));

  llvm::Expected<Workflow> a =
      runWorkflow(*first.context, *first.module, **target);
  llvm::Expected<Workflow> b =
      runWorkflow(*second.context, *second.module, **target);
  ASSERT_TRUE(static_cast<bool>(a)) << llvm::toString(a.takeError());
  ASSERT_TRUE(static_cast<bool>(b)) << llvm::toString(b.takeError());

  EXPECT_EQ(a->search.plans.front().id, b->search.plans.front().id);
  EXPECT_EQ(a->mappedText, b->mappedText);
}

//===----------------------------------------------------------------------===//
// Stage C5: the chain runs to code
//===----------------------------------------------------------------------===//
//
// The chain above stops at mapped Micro-IR. These tests take the same plan all
// the way to a JIT-compiled program and run it, which is the only way to tell
// whether a selected schedule means anything.

namespace {

/// The same shape of work, but with the explicit signature the lowering bridge
/// requires: a staged copy and a vector add, producing the add's result.
///
/// The result is `v + v` where `v` is a view of the copied input, so a caller
/// that fills the input with 1.0 reads back 2.0 in every element -- an answer
/// that depends on the copy actually having happened.
constexpr llvm::StringLiteral kLowerableKernel = R"mlir(
module {
  micro.kernel @lowerable(%ext: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %t, %tok = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    %v = micro.tile_view %t {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %v, %v : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<acc>>
    micro.yield %r : !micro.tile<8x8xf32, memory = #micro.memory<acc>>
  }
}
)mlir";

/// Searches `module` for a plan on `target`, which is the schedule-
/// instantiation half the caller owns.
llvm::Expected<CoveringPlan> planFor(MLIRContext &context, ModuleOp module,
                                     MappingTarget &target) {
  Operation *kernel = nullptr;
  module->walk([&](Operation *op) {
    if (!kernel && op->getName().getStringRef() == "micro.kernel")
      kernel = op;
  });
  llvm::Expected<WorkloadGraph> graph = extractWorkloadGraph(kernel);
  if (!graph)
    return graph.takeError();

  LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "f32";
  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(*graph, target, context, layoutContext, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  if (!result)
    return result.takeError();
  if (result->plans.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the search produced no plan");
  return result->plans.front();
}

} // namespace

TEST(E2EWorkflow, CompilesAMappedKernelToARunnableExecutable) {
  Parsed parsed = parseKernel(kLowerableKernel);
  ASSERT_TRUE(parsed.module);

  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      avx2_mapping::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  llvm::Expected<CoveringPlan> plan =
      planFor(*parsed.context, *parsed.module, **target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());

  ::llk::MappedCompileOptions options;
  options.entrySymbol = "lowerable";
  llvm::Expected<::llk::MappedCompilation> compiled =
      ::llk::compileMappedKernel(*parsed.module, **target, *plan, options);
  ASSERT_TRUE(static_cast<bool>(compiled))
      << llvm::toString(compiled.takeError());

  EXPECT_EQ(compiled->stopped, ::llk::MappedStop::Executable);
  ASSERT_TRUE(compiled->executable);
  ASSERT_EQ(compiled->executable->abi().inputs.size(), 1u);
  ASSERT_EQ(compiled->executable->abi().outputs.size(), 1u);
  EXPECT_EQ(compiled->executable->abi().inputs[0].shape,
            (std::vector<int64_t>{8, 8}));

  // Both selected groups are handed to their concrete consumers: vector
  // arithmetic to the AVX2 Vector pass and the copy to the host movement ABI.
  EXPECT_EQ(compiled->targetLowered, 2u);
  EXPECT_EQ(compiled->referenceLowered, 0u);

  // The source module is untouched: binding clones.
  EXPECT_FALSE(parsed.kernel->hasAttr("micro.plan"));

  std::vector<float> input(64, 1.0f);
  std::vector<float> output(64, -1.0f);
  ::llk::InvocationBuffer2D in{{input.data(), input.data(), 0, 8, 8, 8, 1},
                               ::llk::InvocationElementType::F32,
                               input.size() * sizeof(float)};
  ::llk::InvocationBuffer2D out{{output.data(), output.data(), 0, 8, 8, 8, 1},
                                ::llk::InvocationElementType::F32,
                                output.size() * sizeof(float)};
  llvm::Error error = compiled->executable->invoke({in}, {out});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));

  // v + v over an input of ones is two everywhere -- and it is only two if the
  // staged copy actually delivered the input to the add.
  for (size_t i = 0; i < output.size(); ++i)
    EXPECT_EQ(output[i], 2.0f) << "element " << i;
}

TEST(E2EWorkflow, RejectsSelectedTargetUntilItsBackendExists) {
  Parsed parsed = parseKernel(kLowerableKernel);
  ASSERT_TRUE(parsed.module);

  ::llk::MappedCompileOptions options;
  options.backend = ::llk::MappedBackend::SelectedTarget;
  auto compiled = ::llk::compileConcreteMicroKernel(*parsed.module, options);
  ASSERT_FALSE(static_cast<bool>(compiled));
  EXPECT_NE(llvm::toString(compiled.takeError()).find("selected-target"),
            std::string::npos);
}

TEST(E2EWorkflow, StopsBeforeExecutingWhenAsked) {
  Parsed parsed = parseKernel(kLowerableKernel);
  ASSERT_TRUE(parsed.module);

  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      avx2_mapping::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  llvm::Expected<CoveringPlan> plan =
      planFor(*parsed.context, *parsed.module, **target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());

  // Mapped Micro: the selected state is recorded and no target has touched the
  // operations yet.
  {
    ::llk::MappedCompileOptions options;
    options.stop = ::llk::MappedStop::MappedMicro;
    llvm::Expected<::llk::MappedCompilation> compiled =
        ::llk::compileMappedKernel(*parsed.module, **target, *plan, options);
    ASSERT_TRUE(static_cast<bool>(compiled))
        << llvm::toString(compiled.takeError());
    EXPECT_EQ(compiled->stopped, ::llk::MappedStop::MappedMicro);
    EXPECT_TRUE(compiled->module);
    EXPECT_FALSE(compiled->executable);
    EXPECT_TRUE(compiled->module
                    ->walk([](Operation *op) {
                      return op->hasAttr("micro.plan") ? WalkResult::interrupt()
                                                       : WalkResult::advance();
                    })
                    .wasInterrupted());
  }

  // Target-lowered: the AVX2 emitters have rewritten the operations they
  // implement, which for the vector family means the result tile carries the
  // physical vector width the bundle selected.
  {
    ::llk::MappedCompileOptions options;
    options.stop = ::llk::MappedStop::TargetLowered;
    llvm::Expected<::llk::MappedCompilation> compiled =
        ::llk::compileMappedKernel(*parsed.module, **target, *plan, options);
    ASSERT_TRUE(static_cast<bool>(compiled))
        << llvm::toString(compiled.takeError());
    EXPECT_EQ(compiled->stopped, ::llk::MappedStop::TargetLowered);
    EXPECT_EQ(compiled->targetLowered, 2u);
    EXPECT_EQ(compiled->referenceLowered, 0u);

    std::string text;
    llvm::raw_string_ostream stream(text);
    compiled->module->print(stream);
    EXPECT_NE(text.find("vectorized"), std::string::npos) << text;
  }
}

TEST(E2EWorkflow, RefusesToCompileWhenThePlanIsNotExecutable) {
  Parsed parsed = parseKernel(kLowerableKernel);
  ASSERT_TRUE(parsed.module);

  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      avx2_mapping::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  llvm::Expected<CoveringPlan> plan =
      planFor(*parsed.context, *parsed.module, **target);
  ASSERT_TRUE(static_cast<bool>(plan)) << llvm::toString(plan.takeError());

  // Compiling to an executable needs the entry symbol, and does not guess one:
  // a lowered module may hold both the source function and the kernel that
  // replaced it.
  ::llk::MappedCompileOptions options;
  options.entrySymbol = "";
  llvm::Expected<::llk::MappedCompilation> compiled =
      ::llk::compileMappedKernel(*parsed.module, **target, *plan, options);
  ASSERT_FALSE(static_cast<bool>(compiled));
  EXPECT_NE(llvm::toString(compiled.takeError()).find("entry symbol"),
            std::string::npos);
}
