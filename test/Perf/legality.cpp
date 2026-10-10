//===- legality.cpp - Machine-aware legality tests ------------------------===//
//
// Covers issue #49, legality half. Every constraint kind the dialect accepts
// has a rejection case here, plus a legal baseline the exporter's constraint
// set accepts and the first candidate of a space shaped like the M11 export's
// output for both roots it supports:
//   - SRAM and accumulator capacity
//   - MMA dtype and tile-shape compatibility
//   - mapping extent (thread count)
//   - tail support
//   - vector width support
//   - tile-hierarchy divisibility
//   - layout support on the staged memory
//   - owner existence and nesting
//   - fragment compatibility with the execution tile
//   - pipeline live-tile capacity and outstanding async copies
//
// Rejection tests use a space with a single constraint so the stable reason
// string is the one under test; the baseline uses the exporter's full set.
//
//===----------------------------------------------------------------------===//

#include "LLK/Perf/Legality.h"
#include "LLK/Machine/MachineModelLoader.h"
#include "LLK/Perf/CandidateGenerator.h"

#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mlir::llk::perf {
namespace {

using llvm::StringRef;

SearchParam integerParam(std::string name, std::vector<SearchChoice> choices) {
  return SearchParam{std::move(name), "integer", std::move(choices),
                     std::string{}};
}

SearchParam symbolicParam(std::string name, std::string kind,
                          std::vector<SearchChoice> choices) {
  return SearchParam{std::move(name), std::move(kind), std::move(choices),
                     std::string{}};
}

/// A symbolic parameter that names the role (port) it governs. Two such
/// parameters of one kind are what a per-role layout constraint must evaluate
/// individually.
SearchParam roleParam(std::string name, std::string kind,
                      std::vector<SearchChoice> choices, std::string role) {
  return SearchParam{std::move(name), std::move(kind), std::move(choices),
                     std::move(role)};
}

std::vector<SearchParam> fullParams(bool withTail = true) {
  std::vector<SearchParam> params = {
      integerParam("BM", {SearchChoice(int64_t{8})}),
      integerParam("BN", {SearchChoice(int64_t{64})}),
      integerParam("BK", {SearchChoice(int64_t{64})}),
      integerParam("pipeline_stages", {SearchChoice(int64_t{1})}),
      integerParam("prefetch_distance", {SearchChoice(int64_t{1})}),
      integerParam("vector_width", {SearchChoice(int64_t{8})}),
      integerParam("num_threads", {SearchChoice(int64_t{8})}),
      symbolicParam("tile_layout", "layout", {SearchChoice("row_major")}),
      symbolicParam("memory_path", "memory_path",
                    {SearchChoice("dram:sram:acc")}),
      symbolicParam("owner_mapping", "owner_mapping",
                    {SearchChoice("worker/vector_engine")}),
      symbolicParam("fragment_shape", "fragment_shape",
                    {SearchChoice("16x16x32")}),
  };
  if (withTail)
    params.push_back(
        symbolicParam("tail_policy", "tail_policy", {SearchChoice("pad")}));
  return params;
}

/// A space whose only constraint is `kind` over `referenced`.
SearchSpace spaceWith(ConstraintKind kind, std::vector<std::string> referenced,
                      std::vector<SearchParam> params,
                      llvm::StringRef workload = "fused_swiglu") {
  SearchSpace space;
  space.name = "legality_fixture";
  space.workload = workload.str();
  space.params = std::move(params);
  space.constraints.push_back(
      SearchConstraint{kind, std::move(referenced), {}});
  return space;
}

Candidate makeCandidate(std::map<std::string, int64_t> values,
                        std::map<std::string, std::string> symbolic) {
  Candidate candidate;
  candidate.values = std::move(values);
  candidate.symbolicValues = std::move(symbolic);
  candidate.id = computeCandidateId(candidate.values, candidate.symbolicValues);
  return candidate;
}

/// The exported space's preferred candidate, which must be legal.
Candidate preferred() {
  return makeCandidate({{"BM", 8},
                        {"BN", 64},
                        {"BK", 64},
                        {"pipeline_stages", 1},
                        {"prefetch_distance", 1},
                        {"vector_width", 8},
                        {"num_threads", 8}},
                       {{"tile_layout", "row_major"},
                        {"memory_path", "dram:sram:acc"},
                        {"owner_mapping", "worker/vector_engine"},
                        {"fragment_shape", "16x16x32"},
                        {"tail_policy", "pad"}});
}

WorkloadShape swigluShape() {
  WorkloadShape shape;
  shape.M = 8;
  shape.N = 64;
  shape.K = 64;
  return shape;
}

const machine::MachineModel &avx2() {
  static std::optional<machine::MachineModel> model = [] {
    auto loaded = machine::loadMachineModel(std::string(LLK_MACHINE_DIR) +
                                            "/x86-avx2-v2.yaml");
    if (!loaded) {
      ADD_FAILURE() << llvm::toString(loaded.takeError());
      return std::optional<machine::MachineModel>();
    }
    return std::optional<machine::MachineModel>(std::move(*loaded));
  }();
  return *model;
}

/// A machine whose only on-chip memory accepts `row_major`, so a layout
/// constraint is observably decided by the value bound to each referenced
/// parameter.
const machine::MachineModel &rowMajorOnlyMachine() {
  static constexpr llvm::StringLiteral kYaml = R"yaml(
schema: llk.machine.v2
target: layout_probe
executors:
  - id: worker.0
    kind: worker
memories:
  - id: sram.0
    kind: sram
    visible_from: worker.0
    capacity_bytes: 1048576
    alignment_bytes: 64
    supported_layouts: [row_major]
compute:
  - id: mxu
    kind: matrix_engine
    refines: [matrix]
    attached_to: worker.0
    element_types: [f32, bf16]
    accumulator_dtypes: [f32]
    shapes: [[4, 8, 8]]
)yaml";
  static std::optional<machine::MachineModel> model = [] {
    auto loaded = machine::parseMachineModel(kYaml, "<legality-test>");
    if (!loaded) {
      ADD_FAILURE() << llvm::toString(loaded.takeError());
      return std::optional<machine::MachineModel>();
    }
    return std::optional<machine::MachineModel>(std::move(*loaded));
  }();
  return *model;
}

std::vector<SearchConstraint> exporterConstraints();

/// A space shaped like the M11 export's output: the tuner grid for the numeric
/// parameters, the schedule-anchored symbolic choices the export prefers, and
/// the exporter's constraint set. This is the IR the tuner actually receives,
/// so the first generated candidate must be legal for both roots the export
/// supports.
SearchSpace exportedShapedSpace(std::string workload) {
  SearchSpace space;
  space.name = workload + "_M8_N64_K64";
  space.workload = std::move(workload);
  space.params = {
      integerParam("BM", {SearchChoice(int64_t{1}), SearchChoice(int64_t{4}),
                          SearchChoice(int64_t{8}), SearchChoice(int64_t{16})}),
      integerParam("BN", {SearchChoice(int64_t{16}), SearchChoice(int64_t{32}),
                          SearchChoice(int64_t{64})}),
      integerParam("BK",
                   {SearchChoice(int64_t{32}), SearchChoice(int64_t{64})}),
      integerParam("pipeline_stages",
                   {SearchChoice(int64_t{1}), SearchChoice(int64_t{2})}),
      integerParam("prefetch_distance",
                   {SearchChoice(int64_t{1}), SearchChoice(int64_t{2})}),
      integerParam("vector_width", {SearchChoice(int64_t{8})}),
      integerParam("num_threads",
                   {SearchChoice(int64_t{1}), SearchChoice(int64_t{2}),
                    SearchChoice(int64_t{4}), SearchChoice(int64_t{8})}),
      integerParam("grain_size",
                   {SearchChoice(int64_t{1}), SearchChoice(int64_t{2})}),
      symbolicParam("tile_layout", "layout",
                    {SearchChoice("row_major"), SearchChoice("blocked")}),
      symbolicParam(
          "memory_path", "memory_path",
          {SearchChoice("dram:sram:acc"), SearchChoice("dram:l2:sram:acc")}),
      symbolicParam(
          "owner_mapping", "owner_mapping",
          {SearchChoice("worker/lane"), SearchChoice("worker/vector_engine")}),
      symbolicParam("fragment_shape", "fragment_shape",
                    {SearchChoice("16x16x32"), SearchChoice("8x8x32")}),
      symbolicParam("tail_policy", "tail_policy", {SearchChoice("pad")}),
  };
  space.constraints = exporterConstraints();
  return space;
}

/// The full exporter constraint set, in the order the export emits it.
std::vector<SearchConstraint> exporterConstraints() {
  auto constraint = [](ConstraintKind kind, std::vector<std::string> params) {
    return SearchConstraint{kind, std::move(params), {}};
  };
  return {
      constraint(ConstraintKind::SramCapacity, {"BM", "BN", "BK"}),
      constraint(ConstraintKind::AccCapacity, {"BM", "BN"}),
      constraint(ConstraintKind::MmaCompatible, {"BM", "BN", "BK"}),
      constraint(ConstraintKind::TileHierarchyCompatible, {"owner_mapping"}),
      constraint(ConstraintKind::LayoutSupported, {"tile_layout"}),
      constraint(ConstraintKind::OwnerSupported, {"owner_mapping"}),
      constraint(ConstraintKind::FragmentCompatible,
                 {"fragment_shape", "BM", "BN", "BK"}),
      constraint(ConstraintKind::VectorWidthSupported, {"vector_width"}),
      constraint(ConstraintKind::MappingExtent, {"num_threads", "BM", "BN"}),
      constraint(ConstraintKind::PipelineLiveTiles,
                 {"pipeline_stages", "prefetch_distance", "BM", "BN", "BK"}),
      constraint(ConstraintKind::TailSupported, {"BM", "BN", "BK"}),
  };
}

void expectRejected(const LegalityResult &result, llvm::StringRef prefix,
                    llvm::StringRef detail) {
  ASSERT_FALSE(result.legal);
  EXPECT_EQ(result.reason.substr(0, prefix.size()), prefix) << result.reason;
  EXPECT_NE(result.reason.find(detail), std::string::npos) << result.reason;
}

//===----------------------------------------------------------------------===//
// Baseline
//===----------------------------------------------------------------------===//

TEST(Legality, PreferredCandidateSatisfiesTheExporterConstraintSet) {
  SearchSpace space;
  space.params = fullParams();
  space.constraints = exporterConstraints();

  LegalityResult result =
      checkLegality(space, preferred(), swigluShape(), avx2());
  EXPECT_TRUE(result.legal) << result.reason;
  EXPECT_TRUE(result.reason.empty());
}

TEST(Legality, AccCapacityAtExactlyTheLimitIsLegal) {
  // 2 * 8 * 64 * 4 = 4096, exactly the acc capacity.
  SearchSpace space =
      spaceWith(ConstraintKind::AccCapacity, {"BM", "BN"}, fullParams());
  EXPECT_TRUE(checkLegality(space, preferred(), swigluShape(), avx2()).legal);
}

//===----------------------------------------------------------------------===//
// Capacity
//===----------------------------------------------------------------------===//

TEST(Legality, SramCapacityOverflowIsRejected) {
  SearchSpace space =
      spaceWith(ConstraintKind::SramCapacity, {"BM", "BN", "BK"}, fullParams());
  Candidate candidate = preferred();
  candidate.values["BM"] = 128;
  candidate.values["BN"] = 256;
  candidate.values["BK"] = 256;

  expectRejected(checkLegality(space, candidate, swigluShape(), avx2()),
                 "sram_capacity:", "327680");
}

TEST(Legality, DoubleBufferingCountsAgainstSram) {
  // The same tile fits at one stage and overflows when the pipeline doubles it:
  // 8x64x64 bf16 needs 17408 bytes, and 34816 with two stages.
  SearchSpace space =
      spaceWith(ConstraintKind::SramCapacity, {"BM", "BN", "BK"}, fullParams());
  Candidate candidate = preferred();
  EXPECT_TRUE(checkLegality(space, candidate, swigluShape(), avx2()).legal);

  candidate.values["pipeline_stages"] = 2;
  expectRejected(checkLegality(space, candidate, swigluShape(), avx2()),
                 "sram_capacity:", "requires");
}

TEST(Legality, MatmulSramCapacityCountsOneWeightTile) {
  // The tiles that overflow for the fused contraction fit a plain matmul:
  // 8x64x64 at two stages is 34816 bytes with two weight tiles, 18432 with one.
  SearchSpace space = spaceWith(ConstraintKind::SramCapacity,
                                {"BM", "BN", "BK"}, fullParams(), "matmul");
  Candidate candidate = preferred();
  candidate.values["pipeline_stages"] = 2;

  EXPECT_TRUE(checkLegality(space, candidate, swigluShape(), avx2()).legal);
}

TEST(Legality, MatmulAccCapacityCountsOneAccumulator) {
  // 16x64 f32 is exactly one 4096-byte accumulator; a fused contraction's two
  // accumulators of the same shape would need 8192 and be rejected.
  SearchSpace space = spaceWith(ConstraintKind::AccCapacity, {"BM", "BN"},
                                fullParams(), "matmul");
  Candidate candidate = preferred();
  candidate.values["BM"] = 16;
  candidate.values["BN"] = 64;

  EXPECT_TRUE(checkLegality(space, candidate, swigluShape(), avx2()).legal);
}

TEST(Legality, AccCapacityOverflowIsRejected) {
  SearchSpace space =
      spaceWith(ConstraintKind::AccCapacity, {"BM", "BN"}, fullParams());
  Candidate candidate = preferred();
  candidate.values["BM"] = 64;
  candidate.values["BN"] = 64;

  expectRejected(checkLegality(space, candidate, swigluShape(), avx2()),
                 "acc_capacity:", "32768");
}

//===----------------------------------------------------------------------===//
// MMA
//===----------------------------------------------------------------------===//

TEST(Legality, UnsupportedMmaInputDtypeIsRejected) {
  SearchSpace space = spaceWith(ConstraintKind::MmaCompatible,
                                {"BM", "BN", "BK"}, fullParams());
  WorkloadShape shape = swigluShape();
  shape.inputDType = "f16"; // the AVX2 FMA model accepts f32 and bf16 only

  expectRejected(checkLegality(space, preferred(), shape, avx2()),
                 "mma_compatible:", "f16");
}

TEST(Legality, TileNotAMultipleOfTheMmaFragmentIsRejected) {
  SearchSpace space =
      spaceWith(ConstraintKind::MmaCompatible, {"BM", "BN", "BK"},
                fullParams(/*withTail=*/false));
  Candidate candidate = preferred();
  candidate.values["BM"] = 3;

  expectRejected(checkLegality(space, candidate, swigluShape(), avx2()),
                 "mma_compatible:", "4x8x8");
}

//===----------------------------------------------------------------------===//
// Mapping and tail
//===----------------------------------------------------------------------===//

TEST(Legality, MoreThreadsThanTheMachineModelsIsRejected) {
  SearchSpace space = spaceWith(ConstraintKind::MappingExtent,
                                {"num_threads", "BM", "BN"}, fullParams());
  Candidate candidate = preferred();
  candidate.values["num_threads"] = 16;

  expectRejected(checkLegality(space, candidate, swigluShape(), avx2()),
                 "mapping_extent:", "16");
}

TEST(Legality, TailWithoutAMaskPolicyIsRejected) {
  SearchSpace space =
      spaceWith(ConstraintKind::TailSupported, {"BM", "BN", "BK"},
                fullParams(/*withTail=*/false));
  WorkloadShape shape = swigluShape();
  shape.M = 17; // not a multiple of BM = 8

  expectRejected(checkLegality(space, preferred(), shape, avx2()),
                 "tail_supported:", "17");
}

TEST(Legality, TailWithMaskPolicyIsRejected) {
  SearchSpace space = spaceWith(ConstraintKind::TailSupported,
                                {"BM", "BN", "BK"}, fullParams());
  WorkloadShape shape = swigluShape();
  shape.M = 17;

  Candidate candidate = preferred();
  candidate.symbolicValues["tail_policy"] = "mask";
  expectRejected(checkLegality(space, candidate, shape, avx2()),
                 "tail_supported:", "tail_policy 'mask' is unsupported");
}

//===----------------------------------------------------------------------===//
// Vector width and layout
//===----------------------------------------------------------------------===//

TEST(Legality, VectorWidthBeyondTheEngineLanesIsRejected) {
  SearchSpace space = spaceWith(ConstraintKind::VectorWidthSupported,
                                {"vector_width"}, fullParams());
  Candidate candidate = preferred();
  candidate.values["vector_width"] = 32; // bf16 lanes are 16

  expectRejected(checkLegality(space, candidate, swigluShape(), avx2()),
                 "vector_width_supported:", "32");
}

TEST(Legality, LayoutSupportedByAnyMemoryOnThePathIsLegal) {
  // `blocked` is a legal layout for sram. The multi-level path stages in l2,
  // whose modelled layout list is narrower, but the machine's memory system as
  // a whole accepts the layout, so the candidate is not rejected here -- the
  // simulator is where a specific kernel's layout mismatch would surface.
  SearchSpace space =
      spaceWith(ConstraintKind::LayoutSupported, {"tile_layout"}, fullParams());
  Candidate candidate = preferred();
  candidate.symbolicValues["tile_layout"] = "blocked";

  // Both multi-level paths the export offers, including the three-space form
  // the schedule database anchors on.
  candidate.symbolicValues["memory_path"] = "dram:l2:sram:acc";
  EXPECT_TRUE(checkLegality(space, candidate, swigluShape(), avx2()).legal);

  candidate.symbolicValues["memory_path"] = "dram:l2:sram";
  EXPECT_TRUE(checkLegality(space, candidate, swigluShape(), avx2()).legal);
}

TEST(Legality, LayoutNoMemorySupportsIsRejected) {
  SearchSpace space =
      spaceWith(ConstraintKind::LayoutSupported, {"tile_layout"}, fullParams());
  Candidate candidate = preferred();
  candidate.symbolicValues["tile_layout"] = "swizzled";

  expectRejected(checkLegality(space, candidate, swigluShape(), avx2()),
                 "layout_supported:", "swizzled");
}

TEST(Legality, BlockedLayoutOnSramIsLegal) {
  SearchSpace space =
      spaceWith(ConstraintKind::LayoutSupported, {"tile_layout"}, fullParams());
  Candidate candidate = preferred();
  candidate.symbolicValues["tile_layout"] = "blocked";

  EXPECT_TRUE(checkLegality(space, candidate, swigluShape(), avx2()).legal);
}

//===----------------------------------------------------------------------===//
// Owner
//===----------------------------------------------------------------------===//

TEST(Legality, UnmodeledOwnerIsRejected) {
  SearchSpace space = spaceWith(ConstraintKind::OwnerSupported,
                                {"owner_mapping"}, fullParams());
  Candidate candidate = preferred();
  candidate.symbolicValues["owner_mapping"] = "worker/pe"; // no pe on AVX2

  expectRejected(checkLegality(space, candidate, swigluShape(), avx2()),
                 "owner_supported:", "pe");
}

TEST(Legality, OwnerNotNestedUnderItsParentIsRejected) {
  SearchSpace space = spaceWith(ConstraintKind::OwnerSupported,
                                {"owner_mapping"}, fullParams());
  Candidate candidate = preferred();
  // vector_engine nests under lane under worker, so worker is not its child.
  candidate.symbolicValues["owner_mapping"] = "vector_engine/worker";

  expectRejected(checkLegality(space, candidate, swigluShape(), avx2()),
                 "owner_supported:", "vector_engine");
}

//===----------------------------------------------------------------------===//
// Fragment and tile hierarchy
//===----------------------------------------------------------------------===//

TEST(Legality, FragmentNotDividingTheExecutionTileIsRejected) {
  SearchSpace space = spaceWith(ConstraintKind::FragmentCompatible,
                                {"fragment_shape", "BM", "BN", "BK"},
                                fullParams(/*withTail=*/false));
  Candidate candidate = preferred();
  candidate.symbolicValues["fragment_shape"] = "32x32x64";

  expectRejected(checkLegality(space, candidate, swigluShape(), avx2()),
                 "fragment_compatible:", "32x32x64");
}

TEST(Legality, FragmentDividingTheExecutionTileIsLegal) {
  SearchSpace space = spaceWith(ConstraintKind::FragmentCompatible,
                                {"fragment_shape", "BM", "BN", "BK"},
                                fullParams(/*withTail=*/false));
  Candidate candidate = preferred();
  candidate.symbolicValues["fragment_shape"] = "8x8x32";

  EXPECT_TRUE(checkLegality(space, candidate, swigluShape(), avx2()).legal);
}

TEST(Legality, ChildTileNotDividingItsParentIsRejected) {
  SearchSpace space = spaceWith(ConstraintKind::TileHierarchyCompatible,
                                {"fragment_shape", "BM", "BN", "BK"},
                                fullParams(/*withTail=*/false));
  Candidate candidate = preferred();
  // fragment M = 16 does not divide BM = 8.
  expectRejected(checkLegality(space, candidate, swigluShape(), avx2()),
                 "tile_hierarchy_compatible:", "16x16x32");
}

//===----------------------------------------------------------------------===//
// Pipeline
//===----------------------------------------------------------------------===//

TEST(Legality, TooManyOutstandingCopiesForThePipelineIsRejected) {
  SearchSpace space = spaceWith(
      ConstraintKind::PipelineLiveTiles,
      {"pipeline_stages", "prefetch_distance", "BM", "BN", "BK"}, fullParams());
  Candidate candidate = preferred();
  candidate.values["pipeline_stages"] = 2;
  candidate.values["prefetch_distance"] = 2; // dma max_outstanding is 1

  expectRejected(checkLegality(space, candidate, swigluShape(), avx2()),
                 "pipeline_live_tiles:", "outstanding");
}

TEST(Legality, PipelineLiveTilesOverflowIsRejected) {
  SearchSpace space = spaceWith(
      ConstraintKind::PipelineLiveTiles,
      {"pipeline_stages", "prefetch_distance", "BM", "BN", "BK"}, fullParams());
  Candidate candidate = preferred();
  candidate.values["BM"] = 64;
  candidate.values["BN"] = 128;
  candidate.values["BK"] = 128;
  candidate.values["pipeline_stages"] = 2;

  expectRejected(checkLegality(space, candidate, swigluShape(), avx2()),
                 "pipeline_live_tiles:", "requires");
}

//===----------------------------------------------------------------------===//
// Structural rules
//===----------------------------------------------------------------------===//

TEST(Legality, UnboundReferencedParameterIsRejected) {
  SearchSpace space =
      spaceWith(ConstraintKind::SramCapacity, {"BM", "BN", "BK"}, fullParams());
  Candidate candidate = preferred();
  candidate.values.erase("BK");

  expectRejected(checkLegality(space, candidate, swigluShape(), avx2()),
                 "sram_capacity:", "does not bind parameter 'BK'");
}

TEST(Legality, FirstFailingConstraintInDeclarationOrderWins) {
  SearchSpace space;
  space.params = fullParams();
  space.constraints = exporterConstraints();
  Candidate candidate = preferred();
  candidate.values["BM"] = 128;
  candidate.values["BN"] = 256;
  candidate.values["BK"] = 256;

  // sram_capacity is declared before acc_capacity.
  expectRejected(checkLegality(space, candidate, swigluShape(), avx2()),
                 "sram_capacity:", "requires");
}

TEST(Legality, EmptyConstraintListIsLegal) {
  SearchSpace space;
  space.params = fullParams();
  EXPECT_TRUE(checkLegality(space, preferred(), swigluShape(), avx2()).legal);
}

TEST(Legality, GeneratedCandidatesBindEveryParameterTheRulesRead) {
  // The seam between generation and legality is the binding names: a candidate
  // the generator produced must satisfy the rules without any hand-written
  // values. The first grid point is every parameter's preferred choice, which
  // the baseline test above proves legal, so it must be legal here too.
  SearchSpace space;
  space.params = fullParams();
  space.constraints = exporterConstraints();

  std::vector<Candidate> candidates = generateGridCandidates(space);
  ASSERT_FALSE(candidates.empty());
  LegalityResult result =
      checkLegality(space, candidates.front(), swigluShape(), avx2());
  EXPECT_TRUE(result.legal) << result.reason;
}

TEST(Legality, ExportedMatmulSpaceFirstCandidateIsLegal) {
  SearchSpace space = exportedShapedSpace("matmul");
  std::vector<Candidate> candidates = generateGridCandidates(space);
  ASSERT_FALSE(candidates.empty());
  LegalityResult result =
      checkLegality(space, candidates.front(), swigluShape(), avx2());
  EXPECT_TRUE(result.legal) << result.reason;
}

TEST(Legality, ExportedSwiGLUSpaceFirstCandidateIsLegal) {
  SearchSpace space = exportedShapedSpace("fused_swiglu");
  std::vector<Candidate> candidates = generateGridCandidates(space);
  ASSERT_FALSE(candidates.empty());
  LegalityResult result =
      checkLegality(space, candidates.front(), swigluShape(), avx2());
  EXPECT_TRUE(result.legal) << result.reason;
}

//===----------------------------------------------------------------------===//
// Constraint facts (A7): evaluate only what each constraint needs
//===----------------------------------------------------------------------===//

// A layout constraint must evaluate every layout parameter it references.
// Two role-carrying layout parameters used to make the unique-kind lookup
// return null, so the rule treated the ambiguity as "no layout to check" and
// silently passed both values. Now the second port's value decides.
TEST(Legality, LayoutConstraintEvaluatesEveryReferencedLayoutParameter) {
  const machine::MachineModel &machine = rowMajorOnlyMachine();

  // Machine SRAM supports row_major only. layout_supported referencing both
  // parameters must reject rhs_layout, rather than treat ambiguity as absence.
  SearchSpace space;
  space.params.push_back(SearchParam{
      "lhs_layout", "layout", {SearchChoice("row_major")}, "operand0"});
  space.params.push_back(SearchParam{
      "rhs_layout", "layout", {SearchChoice("blocked")}, "operand1"});
  space.constraints.push_back(SearchConstraint{
      ConstraintKind::LayoutSupported, {"lhs_layout", "rhs_layout"}, {}});

  Candidate c;
  c.symbolicValues["lhs_layout"] = "row_major";
  c.symbolicValues["rhs_layout"] = "blocked";

  LegalityResult result = checkLegality(space, c, BindingFacts{}, machine);
  expectRejected(result, "layout_supported:", "blocked");
  EXPECT_NE(result.reason.find("operand1"), std::string::npos) << result.reason;

  // The first parameter is checked independently of the second, so a legal
  // second layout does not simply short-circuit the rule.
  c.symbolicValues["rhs_layout"] = "row_major";
  EXPECT_TRUE(checkLegality(space, c, BindingFacts{}, machine).legal);
}

// A role declared by two layout parameters leaves "which one governs it"
// unanswerable, so the space is rejected rather than picking one.
TEST(Legality, DuplicateLayoutRolesAreRejected) {
  const machine::MachineModel &machine = rowMajorOnlyMachine();

  SearchSpace space;
  space.params = {
      roleParam("a_layout", "layout", {SearchChoice("row_major")}, "operand0"),
      roleParam("b_layout", "layout", {SearchChoice("row_major")}, "operand0"),
  };
  space.constraints.push_back(
      SearchConstraint{ConstraintKind::LayoutSupported, {"a_layout"}, {}});

  Candidate candidate =
      makeCandidate({}, {{"a_layout", "row_major"}, {"b_layout", "row_major"}});
  expectRejected(checkLegality(space, candidate, BindingFacts{}, machine),
                 "layout_supported:", "more than one");
}

// A reference that names neither a parameter nor a role is unresolved: it is a
// rejection, not a silent pass. This must hold even when the candidate cannot
// bind the name -- the space is malformed, not the candidate.
TEST(Legality, UnresolvedLayoutReferenceIsRejected) {
  const machine::MachineModel &machine = rowMajorOnlyMachine();

  SearchSpace space;
  space.params = {
      roleParam("lhs_layout", "layout", {SearchChoice("row_major")},
                "operand0"),
  };
  space.constraints.push_back(
      SearchConstraint{ConstraintKind::LayoutSupported, {"operand9"}, {}});

  Candidate candidate = makeCandidate({}, {{"lhs_layout", "row_major"}});
  expectRejected(checkLegality(space, candidate, BindingFacts{}, machine),
                 "layout_supported:", "operand9");
}

// A reference may name a role the parameter declares. `Candidate` has no role
// lookup -- it is keyed by parameter name -- so the role must resolve to its
// parameter *before* the binding check: a valid role reference is admitted and
// evaluated, not rejected as an unbound parameter named after the role.
TEST(Legality, ResolvedRoleReferenceIsEvaluated) {
  const machine::MachineModel &machine = rowMajorOnlyMachine();

  SearchSpace space;
  space.params.push_back(
      SearchParam{"lhs_layout",
                  "layout",
                  {SearchChoice("row_major"), SearchChoice("blocked")},
                  "operand0"});
  space.constraints.push_back(
      SearchConstraint{ConstraintKind::LayoutSupported, {"operand0"}, {}});

  Candidate c;
  c.symbolicValues["lhs_layout"] = "row_major";
  EXPECT_TRUE(checkLegality(space, c, BindingFacts{}, machine).legal);

  c.symbolicValues["lhs_layout"] = "blocked";
  LegalityResult result = checkLegality(space, c, BindingFacts{}, machine);
  expectRejected(result, "layout_supported:", "blocked");
  EXPECT_NE(result.reason.find("operand0"), std::string::npos) << result.reason;
}

// `mapping_extent` reads the thread count and the machine's capacity only, so
// it must evaluate with no workload facts at all -- the review's vector-only
// regression. The constraint is still enforced, not skipped.
TEST(Legality, ShapeIndependentConstraintNeedsNoWorkloadFacts) {
  SearchSpace space;
  space.params = {
      integerParam("num_threads",
                   {SearchChoice(int64_t{1}), SearchChoice(int64_t{2}),
                    SearchChoice(int64_t{16})}),
  };
  space.constraints.push_back(
      SearchConstraint{ConstraintKind::MappingExtent, {"num_threads"}, {}});

  Candidate ok = makeCandidate({{"num_threads", 2}}, {});
  EXPECT_TRUE(checkLegality(space, ok, BindingFacts{}, avx2()).legal);

  Candidate wide = makeCandidate({{"num_threads", 16}}, {});
  expectRejected(checkLegality(space, wide, BindingFacts{}, avx2()),
                 "mapping_extent:", "16");
}

// Every contraction in the facts is checked, not merely the first: a second
// MMA with an unsupported input dtype is a rejection.
TEST(Legality, EveryContractionIsCheckedForMmaCompatibility) {
  SearchSpace space;
  space.params = {integerParam("BM", {SearchChoice(int64_t{8})}),
                  integerParam("BN", {SearchChoice(int64_t{8})}),
                  integerParam("BK", {SearchChoice(int64_t{16})})};
  space.constraints.push_back(
      SearchConstraint{ConstraintKind::MmaCompatible, {"BM", "BN", "BK"}, {}});

  WorkloadShape f32;
  f32.M = 16;
  f32.N = 16;
  f32.K = 32;
  f32.inputDType = "f32";
  f32.accumulatorDType = "f32";

  WorkloadShape f16 = f32;
  f16.inputDType = "f16";

  BindingFacts facts;
  facts.contractions = {f32, f16};
  Candidate candidate = makeCandidate({{"BM", 8}, {"BN", 8}, {"BK", 16}}, {});

  expectRejected(checkLegality(space, candidate, facts, rowMajorOnlyMachine()),
                 "mma_compatible:", "f16");
}

// Tail divisibility is a whole-workload question: it reads the original
// dimensions recorded before tiling, never a contraction's fragment shape.
// When those dimensions are absent, the constraint is reported as unevaluable
// instead of passing by omission.
TEST(Legality, TailConstraintReadsTheOriginalWorkload) {
  SearchSpace space;
  space.params = {integerParam("BM", {SearchChoice(int64_t{8})}),
                  integerParam("BN", {SearchChoice(int64_t{64})}),
                  integerParam("BK", {SearchChoice(int64_t{64})})};
  space.constraints.push_back(
      SearchConstraint{ConstraintKind::TailSupported, {"BM", "BN", "BK"}, {}});
  Candidate candidate = makeCandidate({{"BM", 8}, {"BN", 64}, {"BK", 64}}, {});

  WorkloadShape original;
  original.M = 17;
  original.N = 64;
  original.K = 64;
  BindingFacts withOriginal;
  withOriginal.originalWorkload = original;
  withOriginal.contractions = {swigluShape()};
  expectRejected(checkLegality(space, candidate, withOriginal, avx2()),
                 "tail_supported:", "17");

  expectRejected(checkLegality(space, candidate, BindingFacts{}, avx2()),
                 "tail_supported:", "cannot be evaluated");
}

} // namespace
} // namespace mlir::llk::perf
