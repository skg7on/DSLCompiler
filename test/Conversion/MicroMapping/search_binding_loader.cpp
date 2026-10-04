//===- search_binding_loader.cpp - micro.candidate -> SearchBinding ----===//
//
// Covers phase-4 task 1 (epic #67): the producer for mapping::SearchBinding.
// A `micro.candidate` is a complete assignment of a space's parameters, so
// loading one is a *binding-level* global legality check (ruling S2): every
// declared parameter is bound, every bound name is declared, and every value
// is inside its parameter's declared domain. A candidate outside its domain is
// a startup diagnostic, not a late search failure, so the loader rejects it.
//
// The positive cases read the checked-in fixture. Some rejection cases come
// straight from source strings, because the structures they exploit are legal
// IR the dialect verifier does not reject: a candidate name shared across two
// spaces (symbols are unique only within one space) and a candidate with no
// enclosing search space. Other rejection cases cannot come from a source
// string: the parser verifies a module's invariants after building it, so the
// dialect's own candidate checks would fire first. Those parse a valid space
// and then *mutate* a candidate's `bindings`, which is the state the loader
// must survive when no verifier has blessed the IR -- exactly the startup path
// ruling S2 is about.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/MicroMapping/SearchBindingLoader.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Machine/MachineModelLoader.h"
#include "LLK/Mapping/SearchBinding.h"
#include "MicroMappingCommon.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#include "gtest/gtest.h"

#include <memory>
#include <string>

using namespace mlir::llk::mapping;

namespace {
using mlir::ModuleOp;

struct Parsed {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

/// Parses `source`. parseSourceString already verifies the module's
/// invariants, so a successful parse is also a verified module.
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
  return parsed;
}

/// The named `micro.candidate` op, found by symbol. Kept generic on purpose:
/// the test drives the loader through the IR, not through the dialect's C++
/// API, so a change to the op's accessors cannot mask a loader bug.
mlir::Operation *findCandidate(mlir::ModuleOp module, llvm::StringRef name) {
  mlir::Operation *found = nullptr;
  module.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() != "micro.candidate")
      return;
    auto symbol = op->getAttrOfType<mlir::StringAttr>("sym_name");
    if (symbol && symbol.getValue() == name)
      found = op;
  });
  return found;
}

/// Replaces a candidate's `bindings` dictionary, bypassing the dialect
/// verifier so the loader's own domain check is what the test measures.
void setBindings(
    mlir::Operation *candidate, mlir::MLIRContext *context,
    llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Attribute>> entries) {
  mlir::Builder builder(context);
  llvm::SmallVector<mlir::NamedAttribute> named;
  for (const auto &entry : entries)
    named.push_back(builder.getNamedAttr(entry.first, entry.second));
  candidate->setAttr("bindings", builder.getDictionaryAttr(named));
}

std::unique_ptr<Parsed> parseFixture() {
  std::string error;
  auto source = llvm::MemoryBuffer::getFile(
      std::string(LLK_SOURCE_DIR) +
      "/test/Conversion/MicroMapping/search_binding.mlir");
  if (!source) {
    ADD_FAILURE() << "cannot read the search_binding.mlir fixture";
    return nullptr;
  }
  return parse((*source)->getBuffer());
}

/// The error text of a failed `loadSearchBinding`, or "" when it succeeded.
template <typename T> std::string takeError(llvm::Expected<T> &value) {
  EXPECT_FALSE(static_cast<bool>(value));
  if (value)
    return "";
  return llvm::toString(value.takeError());
}

const char *kNoCandidate = R"MLIR(
micro.search_space @space attributes {workload = "w"} {
  micro.param "BM" {kind = "integer", choices = [32 : i64, 64 : i64]}
}
)MLIR";

/// Two spaces, each declaring `@candidate_17`. The dialect makes candidate
/// symbols unique only *within* one space, so this parses and verifies -- the
/// cross-space ambiguity check is the loader's own, and it is reachable from
/// valid IR a user can write in a `.mlir` file.
const char *kTwoSpacesShareCandidateName = R"MLIR(
micro.search_space @a attributes {workload = "w"} {
  micro.param "BM" {kind = "integer", choices = [32 : i64, 64 : i64]}
  micro.candidate @candidate_17 {bindings = {BM = 32 : i64}}
}
micro.search_space @b attributes {workload = "w"} {
  micro.param "BK" {kind = "integer", choices = [16 : i64, 32 : i64]}
  micro.candidate @candidate_17 {bindings = {BK = 16 : i64}}
}
)MLIR";

/// A candidate with no enclosing search space. Neither the dialect verifier
/// nor the module verifier forbids this, so the loader is the layer that
/// rejects it -- another rejection reachable from valid parsed IR.
const char *kOrphanCandidate = R"MLIR(
micro.candidate @orphan {bindings = {BM = 32 : i64}}
)MLIR";

TEST(SearchBindingLoader, LoadsEveryBoundParameterOfANamedCandidate) {
  auto parsed = parseFixture();
  ASSERT_TRUE(parsed);

  auto binding = loadSearchBinding(parsed->module.get(), "candidate_17");
  ASSERT_TRUE(static_cast<bool>(binding))
      << llvm::toString(binding.takeError());

  EXPECT_EQ(binding->candidateId, "candidate_17");
  ASSERT_EQ(binding->values.size(), 2u);

  auto bm = binding->values.find("BM");
  ASSERT_NE(bm, binding->values.end());
  EXPECT_EQ(std::get<int64_t>(bm->second), 64);

  auto layout = binding->values.find("tile_layout");
  ASSERT_NE(layout, binding->values.end());
  EXPECT_EQ(std::get<std::string>(layout->second), "blocked");

  // The hash and canonical form come from the phase-1 helpers, so a loaded
  // binding is recognisable as the same point a generated one would produce.
  llvm::StringMap<SearchValue> expectedValues;
  expectedValues["BM"] = int64_t{64};
  expectedValues["tile_layout"] = std::string("blocked");
  SearchBinding expected =
      makeSearchBinding("candidate_17", std::move(expectedValues));
  EXPECT_EQ(binding->stableHash, expected.stableHash);
  EXPECT_EQ(canonicalSearchBindingString(*binding),
            canonicalSearchBindingString(expected));
}

TEST(SearchBindingLoader, LoadsADifferentCandidateOfTheSameSpace) {
  auto parsed = parseFixture();
  ASSERT_TRUE(parsed);

  auto binding = loadSearchBinding(parsed->module.get(), "candidate_18");
  ASSERT_TRUE(static_cast<bool>(binding))
      << llvm::toString(binding.takeError());

  EXPECT_EQ(binding->candidateId, "candidate_18");
  EXPECT_EQ(std::get<int64_t>(binding->values["BM"]), 128);
  EXPECT_EQ(std::get<std::string>(binding->values["tile_layout"]), "row_major");
}

TEST(SearchBindingLoader, EmptySymbolSelectsTheSoleCandidate) {
  auto parsed = parse(R"MLIR(
micro.search_space @space attributes {workload = "w"} {
  micro.param "BM" {kind = "integer", choices = [32 : i64, 64 : i64]}
  micro.param "tile_layout" {kind = "layout", choices = ["row_major", "blocked"]}
  micro.candidate @solo {bindings = {BM = 32 : i64, tile_layout = "row_major"}}
}
)MLIR");
  ASSERT_TRUE(parsed);

  auto binding = loadSearchBinding(parsed->module.get());
  ASSERT_TRUE(static_cast<bool>(binding))
      << llvm::toString(binding.takeError());
  EXPECT_EQ(binding->candidateId, "solo");
}

TEST(SearchBindingLoader, RejectsAnUnknownCandidateSymbol) {
  auto parsed = parseFixture();
  ASSERT_TRUE(parsed);

  auto binding = loadSearchBinding(parsed->module.get(), "nope");
  std::string error = takeError(binding);
  EXPECT_NE(error.find("no micro.candidate named 'nope'"), std::string::npos)
      << error;
}

TEST(SearchBindingLoader, RejectsACandidateThatDoesNotBindEveryParameter) {
  auto parsed = parseFixture();
  ASSERT_TRUE(parsed);

  mlir::Operation *candidate = findCandidate(*parsed->module, "candidate_17");
  ASSERT_NE(candidate, nullptr);
  mlir::Builder builder(parsed->context.get());
  setBindings(candidate, parsed->context.get(),
              {{"BM", builder.getI64IntegerAttr(64)}});

  auto binding = loadSearchBinding(parsed->module.get(), "candidate_17");
  std::string error = takeError(binding);
  EXPECT_NE(error.find("does not bind parameter 'tile_layout'"),
            std::string::npos)
      << error;
}

TEST(SearchBindingLoader, RejectsABoundNameNoParameterDeclares) {
  auto parsed = parseFixture();
  ASSERT_TRUE(parsed);

  mlir::Operation *candidate = findCandidate(*parsed->module, "candidate_17");
  ASSERT_NE(candidate, nullptr);
  mlir::Builder builder(parsed->context.get());
  setBindings(candidate, parsed->context.get(),
              {{"BM", builder.getI64IntegerAttr(64)},
               {"NOPE", builder.getStringAttr("blocked")}});

  auto binding = loadSearchBinding(parsed->module.get(), "candidate_17");
  std::string error = takeError(binding);
  EXPECT_NE(error.find("binds unknown parameter 'NOPE'"), std::string::npos)
      << error;
}

TEST(SearchBindingLoader, RejectsAValueOutsideTheDeclaredDomain) {
  // Ruling S2: an out-of-domain candidate is a binding-level global check at
  // load, not a late search failure.
  auto parsed = parseFixture();
  ASSERT_TRUE(parsed);

  mlir::Operation *candidate = findCandidate(*parsed->module, "candidate_17");
  ASSERT_NE(candidate, nullptr);
  mlir::Builder builder(parsed->context.get());
  setBindings(candidate, parsed->context.get(),
              {{"BM", builder.getI64IntegerAttr(999)},
               {"tile_layout", builder.getStringAttr("blocked")}});

  auto binding = loadSearchBinding(parsed->module.get(), "candidate_17");
  std::string error = takeError(binding);
  // The loader mirrors the verifier's wording on purpose (see the header).
  EXPECT_NE(error.find("is not one of the declared choices"), std::string::npos)
      << error;
  EXPECT_NE(error.find("BM"), std::string::npos) << error;
}

TEST(SearchBindingLoader, RejectsACandidateNameSharedAcrossSearchSpaces) {
  // Reachable from valid parsed IR: the dialect only makes candidate symbols
  // unique within one space, so this ambiguity is the loader's to catch.
  auto parsed = parse(kTwoSpacesShareCandidateName);
  ASSERT_TRUE(parsed);

  auto binding = loadSearchBinding(parsed->module.get(), "candidate_17");
  std::string error = takeError(binding);
  EXPECT_NE(error.find("more than one micro.candidate named 'candidate_17'"),
            std::string::npos)
      << error;
}

TEST(SearchBindingLoader, EmptySymbolRejectsMoreThanOneCandidate) {
  // The same two-space fixture with no symbol: the module has two candidates,
  // so neither may be selected implicitly.
  auto parsed = parse(kTwoSpacesShareCandidateName);
  ASSERT_TRUE(parsed);

  auto binding = loadSearchBinding(parsed->module.get());
  std::string error = takeError(binding);
  EXPECT_NE(error.find("more than one micro.candidate"), std::string::npos)
      << error;
}

TEST(SearchBindingLoader, RejectsACandidateWithNoEnclosingSearchSpace) {
  auto parsed = parse(kOrphanCandidate);
  ASSERT_TRUE(parsed);

  auto binding = loadSearchBinding(parsed->module.get(), "orphan");
  std::string error = takeError(binding);
  EXPECT_NE(error.find("is not nested in a micro.search_space"),
            std::string::npos)
      << error;
}

TEST(SearchBindingLoader, AnAmbiguousModuleRequiresASymbol) {
  auto parsed = parseFixture();
  ASSERT_TRUE(parsed);

  auto binding = loadSearchBinding(parsed->module.get());
  std::string error = takeError(binding);
  EXPECT_NE(error.find("more than one micro.candidate"), std::string::npos)
      << error;
}

TEST(SearchBindingLoader, AModuleWithoutACandidateIsRejected) {
  auto parsed = parse(kNoCandidate);
  ASSERT_TRUE(parsed);

  auto binding = loadSearchBinding(parsed->module.get());
  std::string error = takeError(binding);
  EXPECT_NE(error.find("no micro.candidate"), std::string::npos) << error;
}

//===----------------------------------------------------------------------===//
// loadBoundLayout (phase-4 task 4, carried item A)
//===----------------------------------------------------------------------===//

TEST(SearchBindingLoader, ResolvesTheLayoutParameterByKindNotName) {
  // The layout parameter is called `block_shape`, not `layout` or
  // `tile_layout`: resolution must go through the declared `kind`, because the
  // name is the space's to choose. It declares no role, so it governs the
  // layout axis as a whole and is keyed by the empty role.
  auto parsed = parse(R"MLIR(
micro.search_space @space attributes {workload = "w"} {
  micro.param "VW" {kind = "integer", choices = [4 : i64, 8 : i64]}
  micro.param "block_shape" {kind = "layout", choices = ["blocked", "row_major"]}
  micro.candidate @c {bindings = {VW = 8 : i64, block_shape = "blocked"}}
}
)MLIR");
  ASSERT_TRUE(parsed);

  auto binding = loadSearchBinding(parsed->module.get(), "c");
  ASSERT_TRUE(static_cast<bool>(binding))
      << llvm::toString(binding.takeError());

  auto layouts = loadBoundLayouts(parsed->module.get(), *binding);
  ASSERT_TRUE(static_cast<bool>(layouts))
      << llvm::toString(layouts.takeError());
  ASSERT_EQ(layouts->size(), 1u);
  EXPECT_EQ(layouts->lookup(""), "blocked");
}

TEST(SearchBindingLoader, ResolvesOneLayoutPerRole) {
  // Two layout parameters, each naming the port it governs, so both resolve
  // and the binding constrains the two operands differently. This is what a
  // role-less parameter cannot express.
  auto parsed = parse(R"MLIR(
micro.search_space @space attributes {workload = "w"} {
  micro.param "lhs_layout" {kind = "layout", role = "lhs", choices = ["blocked"]}
  micro.param "rhs_layout" {kind = "layout", role = "rhs", choices = ["row_major"]}
  micro.candidate @c {bindings = {lhs_layout = "blocked", rhs_layout = "row_major"}}
}
)MLIR");
  ASSERT_TRUE(parsed);

  auto binding = loadSearchBinding(parsed->module.get(), "c");
  ASSERT_TRUE(static_cast<bool>(binding))
      << llvm::toString(binding.takeError());

  auto layouts = loadBoundLayouts(parsed->module.get(), *binding);
  ASSERT_TRUE(static_cast<bool>(layouts))
      << llvm::toString(layouts.takeError());
  ASSERT_EQ(layouts->size(), 2u);
  EXPECT_EQ(layouts->lookup("lhs"), "blocked");
  EXPECT_EQ(layouts->lookup("rhs"), "row_major");
}

TEST(SearchBindingLoader, TwoLayoutsForOneRoleCannotBeResolved) {
  // Two layout parameters claiming the *same* role: which one selects that
  // role's layout is unanswerable, so it is reported rather than silently
  // picking one -- a value the caller bound would otherwise be ignored.
  auto parsed = parse(R"MLIR(
micro.search_space @space attributes {workload = "w"} {
  micro.param "lhs" {kind = "layout", choices = ["blocked"]}
  micro.param "rhs" {kind = "layout", choices = ["row_major"]}
  micro.candidate @c {bindings = {lhs = "blocked", rhs = "row_major"}}
}
)MLIR");
  ASSERT_TRUE(parsed);

  auto binding = loadSearchBinding(parsed->module.get(), "c");
  ASSERT_TRUE(static_cast<bool>(binding))
      << llvm::toString(binding.takeError());

  auto layouts = loadBoundLayouts(parsed->module.get(), *binding);
  std::string error = takeError(layouts);
  EXPECT_NE(error.find("more than one layout-kind parameter for role"),
            std::string::npos)
      << error;
}

TEST(SearchBindingLoader, NoLayoutKindParameterLeavesTheAxisUnbound) {
  auto parsed = parse(R"MLIR(
micro.search_space @space attributes {workload = "w"} {
  micro.param "VW" {kind = "integer", choices = [4 : i64, 8 : i64]}
  micro.candidate @c {bindings = {VW = 8 : i64}}
}
)MLIR");
  ASSERT_TRUE(parsed);

  auto binding = loadSearchBinding(parsed->module.get(), "c");
  ASSERT_TRUE(static_cast<bool>(binding))
      << llvm::toString(binding.takeError());

  auto layouts = loadBoundLayouts(parsed->module.get(), *binding);
  ASSERT_TRUE(static_cast<bool>(layouts))
      << llvm::toString(layouts.takeError());
  EXPECT_TRUE(layouts->empty());
}

TEST(SearchBindingLoader, RejectsABindingWhoseCandidateIsGone) {
  auto parsed = parseFixture();
  ASSERT_TRUE(parsed);

  llvm::StringMap<SearchValue> values;
  values["BM"] = int64_t{64};
  values["tile_layout"] = std::string("blocked");
  SearchBinding binding = makeSearchBinding("nope", std::move(values));

  auto layouts = loadBoundLayouts(parsed->module.get(), binding);
  std::string error = takeError(layouts);
  EXPECT_NE(error.find("no micro.candidate named 'nope'"), std::string::npos)
      << error;
}

//===----------------------------------------------------------------------===//
// Objective selection (§17.1, issue #109 defect 6)
//===----------------------------------------------------------------------===//

/// Two search spaces, each with its own objective and one candidate. The
/// objectives differ in direction, so which one is read is observable.
constexpr llvm::StringLiteral kTwoSpacesTwoObjectives = R"mlir(
module {
  micro.search_space @space_a attributes {workload = "a"} {
    micro.param "N" {kind = "integer", choices = [1 : i64]}
    micro.candidate @cand_a {bindings = {N = 1 : i64}}
    micro.objective {direction = "minimize", metric = "latency_cycles"}
  }
  micro.search_space @space_b attributes {workload = "b"} {
    micro.param "N" {kind = "integer", choices = [1 : i64]}
    micro.candidate @cand_b {bindings = {N = 1 : i64}}
    micro.objective {direction = "maximize", metric = "dram_bytes"}
  }
}
)mlir";

// The objective must come from the search space the candidate belongs to. The
// old code took the module's *first* objective, so a candidate in the second
// space was ranked by the first space's order.
TEST(SearchObjective, UsesTheSelectedSpacesObjective) {
  auto parsed = parse(kTwoSpacesTwoObjectives);
  ASSERT_TRUE(parsed);

  auto a = mlir::llk::micro_mapping_detail::objectiveOrderFromModule(
      *parsed->module, "cand_a");
  ASSERT_TRUE(static_cast<bool>(a)) << llvm::toString(a.takeError());
  ASSERT_TRUE(a->has_value());
  EXPECT_TRUE((*a)->minimize);
  EXPECT_EQ((*a)->primary, CostMetric::LatencyCycles);

  auto b = mlir::llk::micro_mapping_detail::objectiveOrderFromModule(
      *parsed->module, "cand_b");
  ASSERT_TRUE(static_cast<bool>(b)) << llvm::toString(b.takeError());
  ASSERT_TRUE(b->has_value());
  EXPECT_FALSE((*b)->minimize);
  EXPECT_EQ((*b)->primary, CostMetric::DramBytes);
}

// With no selector, several declared objectives are ambiguous: rejecting is
// what keeps a candidate from being silently ranked by another space's order.
TEST(SearchObjective, SeveralObjectivesWithoutASelectorAreRejected) {
  auto parsed = parse(kTwoSpacesTwoObjectives);
  ASSERT_TRUE(parsed);

  auto order = mlir::llk::micro_mapping_detail::objectiveOrderFromModule(
      *parsed->module);
  std::string error = takeError(order);
  EXPECT_NE(error.find("micro.objectives"), std::string::npos) << error;
}

//===----------------------------------------------------------------------===//
// Persistent global constraints (issue #109 item 5)
//===----------------------------------------------------------------------===//

namespace {
/// A machine with one vector engine, so a `vector_width_supported` constraint
/// has a lane count to compare against.
constexpr llvm::StringLiteral kVectorMachine = R"yaml(
schema: llk.machine.v2
target: constrained
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
  - id: vpu
    kind: vector_engine
    attached_to: worker.0
    element_types: [f32]
    shapes: [[8]]
    lanes: {f32: 8}
)yaml";

/// A space whose `vector_width_supported` constraint distinguishes its two
/// candidates: `vector_width = 8` matches the engine's lanes, `16` exceeds
/// them.
constexpr llvm::StringLiteral kConstrainedSpace = R"mlir(
module {
  micro.kernel @gemm {
    %a = micro.tile_alloc : !micro.tile<16x32xf32, memory = #micro.memory<sram>>
    %b = micro.tile_alloc : !micro.tile<32x16xf32, memory = #micro.memory<sram>>
    %c = micro.tile_alloc : !micro.tile<16x16xf32, memory = #micro.memory<sram>>
    %r = micro.mma %a, %b, %c {shape = array<i64: 16, 16, 32>, input = #micro.dtype<f32>, accumulator = #micro.dtype<f32>} : !micro.tile<16x32xf32, memory = #micro.memory<sram>>, !micro.tile<32x16xf32, memory = #micro.memory<sram>>, !micro.tile<16x16xf32, memory = #micro.memory<sram>> -> !micro.tile<16x16xf32, memory = #micro.memory<sram>>
    micro.yield
  }
  micro.search_space @space attributes {workload = "gemm"} {
    micro.param "vector_width" {kind = "integer", choices = [8 : i64, 16 : i64]}
    micro.constraint "vector_width_supported" {params = ["vector_width"]}
    micro.candidate @ok {bindings = {vector_width = 8 : i64}}
    micro.candidate @wide {bindings = {vector_width = 16 : i64}}
  }
}
)mlir";

/// The same constrained space over a kernel with no MMA, so no workload shape
/// can be derived.
constexpr llvm::StringLiteral kConstrainedSpaceWithoutMma = R"mlir(
module {
  micro.kernel @plain {
    %t = micro.tile_alloc : !micro.tile<8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
  micro.search_space @space attributes {workload = "plain"} {
    micro.param "vector_width" {kind = "integer", choices = [8 : i64, 16 : i64]}
    micro.constraint "vector_width_supported" {params = ["vector_width"]}
    micro.candidate @ok {bindings = {vector_width = 8 : i64}}
  }
}
)mlir";

/// The first `micro.kernel` in `module`, or null.
mlir::Operation *firstKernel(mlir::ModuleOp module) {
  mlir::Operation *kernel = nullptr;
  module.walk([&](mlir::Operation *op) {
    if (!kernel && op->getName().getStringRef() == "micro.kernel")
      kernel = op;
  });
  return kernel;
}

llvm::Expected<mlir::llk::machine::MachineModel> vectorMachine() {
  return mlir::llk::machine::parseMachineModel(kVectorMachine, "<test>");
}
} // namespace

// A binding that violates a `micro.constraint` is rejected before the search:
// the constraint is the space's persistent global legality rule, so the plan
// must never be selected.
TEST(SearchBindingLoader, EnforcesTheSpacesConstraintsOnABinding) {
  auto parsed = parse(kConstrainedSpace);
  ASSERT_TRUE(parsed);
  llvm::Expected<mlir::llk::machine::MachineModel> machine = vectorMachine();
  ASSERT_TRUE(static_cast<bool>(machine))
      << llvm::toString(machine.takeError());

  auto ok = loadSearchBinding(parsed->module.get(), "ok");
  ASSERT_TRUE(static_cast<bool>(ok)) << llvm::toString(ok.takeError());
  EXPECT_FALSE(static_cast<bool>(verifyBindingLegality(
      *parsed->module, *ok, firstKernel(*parsed->module), *machine)));

  auto wide = loadSearchBinding(parsed->module.get(), "wide");
  ASSERT_TRUE(static_cast<bool>(wide)) << llvm::toString(wide.takeError());
  llvm::Error error = verifyBindingLegality(
      *parsed->module, *wide, firstKernel(*parsed->module), *machine);
  ASSERT_TRUE(static_cast<bool>(error));
  std::string text = llvm::toString(std::move(error));
  EXPECT_NE(text.find("violates the search space's constraints"),
            std::string::npos)
      << text;
  EXPECT_NE(text.find("vector_width 16 exceeds engine 'vpu' lanes 8"),
            std::string::npos)
      << text;
}

// A space that declares constraints over a kernel with no MMA cannot be
// evaluated: failing is required, because an unenforced constraint must not
// look like an enforced one.
TEST(SearchBindingLoader, AConstrainedSpaceWithoutAShapeIsRejected) {
  auto parsed = parse(kConstrainedSpaceWithoutMma);
  ASSERT_TRUE(parsed);
  llvm::Expected<mlir::llk::machine::MachineModel> machine = vectorMachine();
  ASSERT_TRUE(static_cast<bool>(machine));

  auto binding = loadSearchBinding(parsed->module.get(), "ok");
  ASSERT_TRUE(static_cast<bool>(binding))
      << llvm::toString(binding.takeError());
  llvm::Error error = verifyBindingLegality(
      *parsed->module, *binding, firstKernel(*parsed->module), *machine);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("cannot be evaluated"),
            std::string::npos);
}

// A space with no constraints is always legal, and the check costs nothing --
// it does not even need a shape (a null kernel is fine).
TEST(SearchBindingLoader, ASpaceWithoutConstraintsIsAlwaysLegal) {
  auto parsed = parse(kTwoSpacesTwoObjectives);
  ASSERT_TRUE(parsed);
  llvm::Expected<mlir::llk::machine::MachineModel> machine = vectorMachine();
  ASSERT_TRUE(static_cast<bool>(machine));

  auto binding = loadSearchBinding(parsed->module.get(), "cand_a");
  ASSERT_TRUE(static_cast<bool>(binding))
      << llvm::toString(binding.takeError());
  EXPECT_FALSE(static_cast<bool>(
      verifyBindingLegality(*parsed->module, *binding, nullptr, *machine)));
}

} // namespace
