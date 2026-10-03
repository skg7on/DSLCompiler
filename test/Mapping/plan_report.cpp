//===- plan_report.cpp - Versioned JSON plan report (§22.2) ---------------===//
//
// The plan report is reproducibility and diagnostic metadata: it names the
// exact inputs a search ran on (module, machine, layout and rule libraries),
// the options it used, how much of the space it covered, and the plans it
// retained. It must never change the IR.
//
// This test pins the report's shape and, above all, its byte-for-byte
// determinism: two runs over identical inputs have to produce the same bytes
// (§29.12), so no key may depend on iteration order, addresses, or the clock.
//
//===----------------------------------------------------------------------===//

#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/LayoutConstraints.h"
#include "LLK/Mapping/MappingRules.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/PlanReport.h"
#include "LLK/Mapping/StableHash.h"
#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include "LLK/Dialect/Micro/MicroDialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include <gtest/gtest.h>

#include <memory>
#include <string>

using namespace mlir;
using namespace mlir::llk::mapping;

namespace avx2_mapping = mlir::llk::target::avx2;

#ifndef LLK_SOURCE_DIR
#error "LLK_SOURCE_DIR must name the repository root"
#endif

namespace {

/// Two nodes -- a staged copy and a vector add -- so the search has real work
/// to place and route.
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

/// A search of `kernel` at the given options. Fails the test at whichever step
/// fails, returning the result.
MappingSearchResult searchKernel(MLIRContext &context, Operation *kernel,
                                 MappingTarget &target,
                                 const MappingSearchOptions &options) {
  llvm::Expected<WorkloadGraph> graph = extractWorkloadGraph(kernel);
  EXPECT_TRUE(static_cast<bool>(graph));
  if (!graph)
    return {};

  LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "f32";
  CoveringSearch search(*graph, target, context, layoutContext, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  EXPECT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  if (!result)
    return {};
  return std::move(*result);
}

} // namespace

// The report carries every field §22.2 names and is byte-identical across runs.
TEST(MappingPlanReportTest, EmitsRequiredFieldsAndIsByteIdentical) {
  Parsed parsed = parseKernel(kKernel);
  ASSERT_TRUE(parsed.module);

  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      avx2_mapping::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  MappingSearchResult result =
      searchKernel(*parsed.context, parsed.kernel, **target, options);
  ASSERT_FALSE(result.plans.empty());

  const uint64_t moduleHash = stableHash(kKernel);

  std::string first = writePlanReport(result, (**target).machine(), **target,
                                      options, moduleHash);
  std::string second = writePlanReport(result, (**target).machine(), **target,
                                       options, moduleHash);
  EXPECT_EQ(first, second);

  llvm::Expected<llvm::json::Value> json = llvm::json::parse(first);
  ASSERT_TRUE(static_cast<bool>(json)) << llvm::toString(json.takeError());
  const llvm::json::Object *root = json->getAsObject();
  ASSERT_TRUE(root);

  EXPECT_TRUE(root->getInteger("version").has_value());
  EXPECT_TRUE(root->getString("compilerVersion").has_value());
  EXPECT_TRUE(root->getInteger("costModelVersion").has_value());
  EXPECT_TRUE(root->getString("inputModuleHash").has_value());
  EXPECT_TRUE(root->getString("sourceBindingHash").has_value());
  EXPECT_TRUE(root->getString("sourceBinding").has_value());
  EXPECT_TRUE(root->getString("machineHash").has_value());
  EXPECT_TRUE(root->getString("layoutLibraryHash").has_value());
  EXPECT_TRUE(root->getString("ruleLibraryHash").has_value());
  EXPECT_TRUE(root->getBoolean("searchTruncated").has_value());
  EXPECT_TRUE(root->getString("selectedPlanId").has_value());

  // The input hash names the module the caller supplied.
  EXPECT_EQ(*root->getString("inputModuleHash"), hexId(moduleHash));

  // Search options are recorded, not just their effect.
  const llvm::json::Object *searchOptions = root->getObject("searchOptions");
  ASSERT_TRUE(searchOptions);
  EXPECT_EQ(*searchOptions->getString("mode"), "deterministic");
  EXPECT_TRUE(searchOptions->getInteger("topK").has_value());

  // Counts are present and cohere with the retained plans.
  const llvm::json::Object *counts = root->getObject("counts");
  ASSERT_TRUE(counts);
  EXPECT_TRUE(counts->getInteger("candidates").has_value());
  EXPECT_TRUE(counts->getInteger("instances").has_value());
  EXPECT_TRUE(counts->getInteger("routes").has_value());
  EXPECT_TRUE(counts->getInteger("plans").has_value());
  EXPECT_GE(*counts->getInteger("plans"),
            static_cast<int64_t>(result.plans.size()));
  EXPECT_GE(*counts->getInteger("instances"), 1);

  // Rejections are grouped by stable code: each code appears at most once.
  ASSERT_TRUE(root->getArray("rejections"));

  // Top-K plans with their component costs.
  const llvm::json::Array *plans = root->getArray("plans");
  ASSERT_TRUE(plans);
  ASSERT_EQ(plans->size(), result.plans.size());
  const llvm::json::Object *best = (*plans)[0].getAsObject();
  ASSERT_TRUE(best);
  EXPECT_EQ(*best->getString("id"), hexId(result.plans.front().id));
  EXPECT_TRUE(best->getString("totalCost").has_value());
  const llvm::json::Object *components = best->getObject("costComponents");
  ASSERT_TRUE(components);
  EXPECT_TRUE(components->getString("latencyCycles").has_value());
  EXPECT_TRUE(components->getInteger("dramBytes").has_value());
  EXPECT_TRUE(components->getString("computeUtilization").has_value());

  // The selected plan is the best plan.
  EXPECT_EQ(*root->getString("selectedPlanId"), hexId(result.plans.front().id));
}

// Rejections are grouped by code, each code once, counts positive.
TEST(MappingPlanReportTest, RejectionCountsAreGroupedByCode) {
  Parsed parsed = parseKernel(kKernel);
  ASSERT_TRUE(parsed.module);

  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      avx2_mapping::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  // A zero candidate cap rejects every node, so the search records the same
  // stable codes on both nodes: the report must group them into one entry per
  // code, with the occurrence count, not one entry per event.
  MappingSearchOptions options;
  options.mode = SearchMode::Beam;
  options.topK = 1;
  options.maxCandidatesPerNode = 0;
  MappingSearchResult result =
      searchKernel(*parsed.context, parsed.kernel, **target, options);

  ASSERT_FALSE(result.frontier.codeCounts.empty());
  std::string report = writePlanReport(result, (**target).machine(), **target,
                                       options, stableHash(kKernel));
  llvm::Expected<llvm::json::Value> json = llvm::json::parse(report);
  ASSERT_TRUE(static_cast<bool>(json)) << llvm::toString(json.takeError());
  const llvm::json::Array *rejections =
      json->getAsObject()->getArray("rejections");
  ASSERT_TRUE(rejections);
  ASSERT_FALSE(rejections->empty());

  llvm::StringSet<> seen;
  bool sawTruncated = false;
  for (const llvm::json::Value &entry : *rejections) {
    const llvm::json::Object *object = entry.getAsObject();
    ASSERT_TRUE(object);
    std::optional<llvm::StringRef> code = object->getString("code");
    std::optional<int64_t> count = object->getInteger("count");
    ASSERT_TRUE(code.has_value());
    ASSERT_TRUE(count.has_value());
    EXPECT_GT(*count, 0);
    EXPECT_TRUE(seen.insert(*code).second) << "duplicate code " << code->str();
    if (*code == "search_truncated")
      sawTruncated = true;
  }
  EXPECT_TRUE(sawTruncated);
}

// The library hashes are content-derived and stable, and differ between the
// two registries (a layout library is not a rule library).
TEST(MappingPlanReportTest, RegistryHashesAreContentDerivedAndStable) {
  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      avx2_mapping::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());

  uint64_t layoutHash = (**target).layouts().computeContentHash();
  uint64_t ruleHash = (**target).rules().computeContentHash();
  EXPECT_NE(layoutHash, 0u);
  EXPECT_NE(ruleHash, 0u);
  EXPECT_NE(layoutHash, ruleHash);
  EXPECT_EQ(layoutHash, (**target).layouts().computeContentHash());
}
