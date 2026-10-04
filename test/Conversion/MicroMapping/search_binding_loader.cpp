//===- search_binding_loader.cpp - micro.candidate -> SearchBinding ----===//
//
// Covers phase-4 task 1 (epic #67): the producer for mapping::SearchBinding.
// A `micro.candidate` is a complete assignment of a space's parameters, so
// loading one is a *binding-level* global legality check (ruling S2): every
// declared parameter is bound, every bound name is declared, and every value
// is inside its parameter's declared domain. A candidate outside its domain is
// a startup diagnostic, not a late search failure, so the loader rejects it.
//
// The positive cases read the checked-in fixture. The rejection cases cannot
// come from a source string: the parser verifies a module's invariants after
// building it, so the dialect's own candidate checks would fire first. They
// instead parse a valid space and then *mutate* a candidate's `bindings`, which
// is the state the loader must survive when no verifier has blessed the IR --
// exactly the startup path ruling S2 is about.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/MicroMapping/SearchBindingLoader.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Mapping/SearchBinding.h"

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

} // namespace
