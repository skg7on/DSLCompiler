//===- search_space_loader.cpp - micro.search_space -> SearchSpace tests --===//
//
// Covers issue #49, loader half:
//   - integer and symbolic choices load in declaration order
//   - a parameter's dialect kind is preserved, so symbolic roles can be
//     resolved by domain instead of by name
//   - constraints load as typed kinds referencing declared parameters
//   - the objective loads with its direction, primary, and secondary metrics
//   - symbol lookup selects a search space, and rejects missing or ambiguous
//     names
//
// The fixture is what the M11 exporter emits, so the loader is exercised on
// the shape of IR the tuner will actually see.
//
//===----------------------------------------------------------------------===//

#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Perf/SearchSpace.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

namespace mlir::llk::perf {
namespace {

/// The export's shape: every parameter kind, typed constraints, and an
/// objective. Two spaces are present so symbol lookup has something to choose.
const char *kFixture = R"MLIR(
micro.search_space @fused_swiglu_M8_N64_K64 attributes {workload = "fused_swiglu"} {
  micro.param "BM" {kind = "integer", choices = [8 : i64, 16 : i64, 32 : i64]}
  micro.param "BN" {kind = "integer", choices = [64 : i64, 128 : i64]}
  micro.param "BK" {kind = "integer", choices = [64 : i64]}
  micro.param "pipeline_stages" {kind = "integer", choices = [1 : i64, 2 : i64]}
  micro.param "tile_layout" {kind = "layout", choices = ["row_major", "blocked"]}
  micro.param "memory_path" {kind = "memory_path", choices = ["dram:sram:acc", "dram:l2:sram:acc"]}
  micro.param "owner_mapping" {kind = "owner_mapping", choices = ["worker/vector_engine"]}
  micro.param "fragment_shape" {kind = "fragment_shape", choices = ["16x16x32"]}
  micro.param "tail_policy" {kind = "tail_policy", choices = ["mask"]}
  micro.constraint "sram_capacity" {params = ["BM", "BN", "BK"]}
  micro.constraint "layout_supported" {params = ["tile_layout"]}
  micro.constraint "pipeline_live_tiles" {params = ["pipeline_stages", "BM", "BN", "BK"]}
  micro.candidate @candidate_0 {bindings = {BM = 8 : i64, BN = 64 : i64, BK = 64 : i64, pipeline_stages = 1 : i64, tile_layout = "row_major", memory_path = "dram:sram:acc", owner_mapping = "worker/vector_engine", fragment_shape = "16x16x32", tail_policy = "mask"}}
  micro.objective {direction = "minimize", metric = "latency_cycles", secondary = ["matrix_utilization", "dram_bytes"]}
}

micro.search_space @matmul_M16_N64_K64 attributes {workload = "matmul"} {
  micro.param "BM" {kind = "integer", choices = [16 : i64]}
  micro.constraint "sram_capacity" {params = ["BM"]}
  micro.objective {direction = "minimize", metric = "latency_cycles"}
}
)MLIR";

struct Parsed {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

std::unique_ptr<Parsed> parse(llvm::StringRef source) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::micro::MicroDialect>();

  auto parsed = std::make_unique<Parsed>();
  parsed->context = std::make_unique<mlir::MLIRContext>(registry);
  parsed->module = mlir::parseSourceString<mlir::ModuleOp>(
      source, mlir::ParserConfig(parsed->context.get()));
  if (!parsed->module) {
    ADD_FAILURE() << "the fixture does not parse";
    return nullptr;
  }
  if (mlir::failed(mlir::verify(*parsed->module))) {
    ADD_FAILURE() << "the fixture does not verify";
    return nullptr;
  }
  return parsed;
}

TEST(SearchSpaceLoader, LoadsIntegerAndSymbolicChoicesInDeclarationOrder) {
  auto parsed = parse(kFixture);
  ASSERT_TRUE(parsed);

  auto space = loadSearchSpace(parsed->module.get(), "fused_swiglu_M8_N64_K64");
  ASSERT_TRUE(static_cast<bool>(space)) << llvm::toString(space.takeError());

  ASSERT_EQ(space->name, "fused_swiglu_M8_N64_K64");
  ASSERT_EQ(space->workload, "fused_swiglu");
  ASSERT_EQ(space->params.size(), 9u);

  // Declaration order is preserved: BM is first, tail_policy last.
  EXPECT_EQ(space->params.front().name, "BM");
  EXPECT_EQ(space->params.back().name, "tail_policy");

  const SearchParam *bm = space->findParam("BM");
  ASSERT_NE(bm, nullptr);
  EXPECT_EQ(bm->kind, "integer");
  ASSERT_EQ(bm->choices.size(), 3u);
  EXPECT_TRUE(bm->choices[0].isInteger());
  EXPECT_EQ(bm->choices[0].integer(), 8);
  EXPECT_EQ(bm->choices[1].integer(), 16);
  EXPECT_EQ(bm->choices[2].integer(), 32);

  const SearchParam *layout = space->findParam("tile_layout");
  ASSERT_NE(layout, nullptr);
  EXPECT_EQ(layout->kind, "layout");
  ASSERT_EQ(layout->choices.size(), 2u);
  EXPECT_TRUE(layout->choices[0].isSymbolic());
  EXPECT_EQ(layout->choices[0].symbol(), "row_major");
  EXPECT_EQ(layout->choices[1].symbol(), "blocked");

  // Symbolic roles are resolved by dialect kind, not by parameter name.
  EXPECT_EQ(space->findParamOfKind("layout"), layout);
  EXPECT_EQ(space->findParamOfKind("fragment_shape")->name, "fragment_shape");
  EXPECT_EQ(space->findParamOfKind("integer"), nullptr);
}

TEST(SearchSpaceLoader, LoadsTypedConstraints) {
  auto parsed = parse(kFixture);
  ASSERT_TRUE(parsed);

  auto space = loadSearchSpace(parsed->module.get(), "fused_swiglu_M8_N64_K64");
  ASSERT_TRUE(static_cast<bool>(space)) << llvm::toString(space.takeError());

  ASSERT_EQ(space->constraints.size(), 3u);

  EXPECT_EQ(space->constraints[0].kind, ConstraintKind::SramCapacity);
  EXPECT_EQ(space->constraints[0].params,
            (std::vector<std::string>{"BM", "BN", "BK"}));

  EXPECT_EQ(space->constraints[1].kind, ConstraintKind::LayoutSupported);
  EXPECT_EQ(space->constraints[1].params,
            (std::vector<std::string>{"tile_layout"}));

  EXPECT_EQ(space->constraints[2].kind, ConstraintKind::PipelineLiveTiles);
  EXPECT_EQ(space->constraints[2].params,
            (std::vector<std::string>{"pipeline_stages", "BM", "BN", "BK"}));
}

TEST(SearchSpaceLoader, LoadsObjective) {
  auto parsed = parse(kFixture);
  ASSERT_TRUE(parsed);

  auto space = loadSearchSpace(parsed->module.get(), "fused_swiglu_M8_N64_K64");
  ASSERT_TRUE(static_cast<bool>(space)) << llvm::toString(space.takeError());

  EXPECT_EQ(space->objective.direction, ObjectiveDirection::Minimize);
  EXPECT_EQ(space->objective.primaryMetric, "latency_cycles");
  EXPECT_EQ(space->objective.secondaryMetrics,
            (std::vector<std::string>{"matrix_utilization", "dram_bytes"}));
}

TEST(SearchSpaceLoader, DeclaredCandidatesAreNotTuningParameters) {
  // micro.candidate ops record pre-baked bindings; the generator produces its
  // own from the parameters, so loading must not mistake one for a parameter.
  auto parsed = parse(kFixture);
  ASSERT_TRUE(parsed);

  auto space = loadSearchSpace(parsed->module.get(), "fused_swiglu_M8_N64_K64");
  ASSERT_TRUE(static_cast<bool>(space)) << llvm::toString(space.takeError());

  EXPECT_EQ(space->params.size(), 9u);
  EXPECT_EQ(space->findParam("candidate_0"), nullptr);
}

TEST(SearchSpaceLoader, ObjectiveWithoutSecondaryLoads) {
  auto parsed = parse(kFixture);
  ASSERT_TRUE(parsed);

  auto space = loadSearchSpace(parsed->module.get(), "matmul_M16_N64_K64");
  ASSERT_TRUE(static_cast<bool>(space)) << llvm::toString(space.takeError());

  EXPECT_EQ(space->objective.primaryMetric, "latency_cycles");
  EXPECT_TRUE(space->objective.secondaryMetrics.empty());
}

TEST(SearchSpaceLoader, EmptySymbolRequiresASingleSpace) {
  auto parsed = parse(kFixture);
  ASSERT_TRUE(parsed);

  auto space = loadSearchSpace(parsed->module.get());
  ASSERT_FALSE(static_cast<bool>(space));
  std::string error = llvm::toString(space.takeError());
  EXPECT_NE(error.find("more than one micro.search_space"), std::string::npos)
      << error;
}

TEST(SearchSpaceLoader, UnknownSymbolIsRejected) {
  auto parsed = parse(kFixture);
  ASSERT_TRUE(parsed);

  auto space = loadSearchSpace(parsed->module.get(), "nope");
  ASSERT_FALSE(static_cast<bool>(space));
  std::string error = llvm::toString(space.takeError());
  EXPECT_NE(error.find("no micro.search_space named 'nope'"), std::string::npos)
      << error;
}

TEST(SearchSpaceLoader, RepeatedSymbolicChoiceIsRejected) {
  // The dialect orders integer choices but does not require symbolic ones to be
  // unique, so the loader is the layer that catches a repeated string choice.
  auto parsed = parse(R"MLIR(
micro.search_space @dup attributes {workload = "matmul"} {
  micro.param "tile_layout" {kind = "layout", choices = ["row_major", "row_major"]}
}
)MLIR");
  ASSERT_TRUE(parsed);

  auto space = loadSearchSpace(parsed->module.get(), "dup");
  ASSERT_FALSE(static_cast<bool>(space));
  std::string error = llvm::toString(space.takeError());
  EXPECT_NE(error.find("repeats a choice"), std::string::npos) << error;
}

TEST(SearchSpaceLoader, ConstraintKindRoundTripsThroughItsSpelling) {
  for (ConstraintKind kind :
       {ConstraintKind::SramCapacity, ConstraintKind::AccCapacity,
        ConstraintKind::MmaCompatible, ConstraintKind::MappingExtent,
        ConstraintKind::TailSupported, ConstraintKind::VectorWidthSupported,
        ConstraintKind::TileHierarchyCompatible,
        ConstraintKind::LayoutSupported, ConstraintKind::OwnerSupported,
        ConstraintKind::FragmentCompatible,
        ConstraintKind::PipelineLiveTiles}) {
    auto text = stringifyConstraintKind(kind);
    EXPECT_FALSE(text.empty());
    EXPECT_EQ(symbolizeConstraintKind(text), kind);
  }
  EXPECT_EQ(stringifyConstraintKind(ConstraintKind::PipelineLiveTiles),
            "pipeline_live_tiles");
  EXPECT_EQ(symbolizeConstraintKind("not_a_rule"), std::nullopt);
}

} // namespace
} // namespace mlir::llk::perf
