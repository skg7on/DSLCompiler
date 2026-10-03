//===- e2e_workflow.cpp - The whole mapping workflow, end to end ---------===//
//
// Issue #53: one test that runs the documented chain on the same fixture
// against two targets, so a break anywhere along it -- extraction, matching,
// placement, connections, covering, binding, verification -- fails here rather
// than in a per-layer test that no longer reflects the whole.
//
//===----------------------------------------------------------------------===//

#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/PlanBinder.h"
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
