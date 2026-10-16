//===- source_adapter_registry.cpp - Source adapter dispatch tests --------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework B3: source adapters are registered explicitly, resolved by exact
// root selection, and every declared choice is accounted for by a realization
// receipt. These tests use a test-only adapter: the production adapters arrive
// in later increments, but the dispatch and receipt contract is enforced here.
//
//===----------------------------------------------------------------------===//

#include "FrameworkTestSupport.h"
#include "LLK/Conversion/MicroMapping/CandidateInstantiation.h"
#include "LLK/Conversion/Workload/SourceAdapter.h"
#include "LLK/Conversion/Workload/WorkloadContract.h"
#include "LLK/Perf/Candidate.h"
#include "LLK/Perf/SearchSpace.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;
using namespace mlir::llk;

namespace {

constexpr llvm::StringLiteral elementwiseSource = R"mlir(
module {
  func.func @elementwise(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>)
      -> tensor<8x8xf32> {
    %init = tensor.empty() : tensor<8x8xf32>
    %r = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0, d1)>],
                         iterator_types = ["parallel", "parallel"]}
        ins(%a, %b : tensor<8x8xf32>, tensor<8x8xf32>)
        outs(%init : tensor<8x8xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %s = arith.addf %x, %y : f32
      linalg.yield %s : f32
    } -> tensor<8x8xf32>
    return %r : tensor<8x8xf32>
  }
})mlir";

/// Two accepted roots in one function, for exact-ordinal selection.
constexpr llvm::StringLiteral twoRootSource = R"mlir(
module {
  func.func @pair(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>)
      -> (tensor<8x8xf32>, tensor<8x8xf32>) {
    %init = tensor.empty() : tensor<8x8xf32>
    %first = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                              affine_map<(d0, d1) -> (d0, d1)>,
                                              affine_map<(d0, d1) -> (d0, d1)>],
                             iterator_types = ["parallel", "parallel"]}
        ins(%a, %b : tensor<8x8xf32>, tensor<8x8xf32>)
        outs(%init : tensor<8x8xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %s = arith.addf %x, %y : f32
      linalg.yield %s : f32
    } -> tensor<8x8xf32>
    %second = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                               affine_map<(d0, d1) -> (d0, d1)>,
                                               affine_map<(d0, d1) -> (d0, d1)>],
                              iterator_types = ["parallel", "parallel"]}
        ins(%first, %b : tensor<8x8xf32>, tensor<8x8xf32>)
        outs(%init : tensor<8x8xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %s = arith.mulf %x, %y : f32
      linalg.yield %s : f32
    } -> tensor<8x8xf32>
    return %first, %second : tensor<8x8xf32>, tensor<8x8xf32>
  }
})mlir";

/// The same accepted root under two function symbols, so ordinals must stay
/// inside the selected function.
constexpr llvm::StringLiteral twoFunctionSource = R"mlir(
module {
  func.func @left(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>)
      -> tensor<8x8xf32> {
    %init = tensor.empty() : tensor<8x8xf32>
    %r = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0, d1)>],
                         iterator_types = ["parallel", "parallel"]}
        ins(%a, %b : tensor<8x8xf32>, tensor<8x8xf32>)
        outs(%init : tensor<8x8xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %s = arith.addf %x, %y : f32
      linalg.yield %s : f32
    } -> tensor<8x8xf32>
    return %r : tensor<8x8xf32>
  }
  func.func @right(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>)
      -> tensor<8x8xf32> {
    %init = tensor.empty() : tensor<8x8xf32>
    %r = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0, d1)>,
                                          affine_map<(d0, d1) -> (d0, d1)>],
                         iterator_types = ["parallel", "parallel"]}
        ins(%a, %b : tensor<8x8xf32>, tensor<8x8xf32>)
        outs(%init : tensor<8x8xf32>) {
    ^bb0(%x: f32, %y: f32, %o: f32):
      %s = arith.mulf %x, %y : f32
      linalg.yield %s : f32
    } -> tensor<8x8xf32>
    return %r : tensor<8x8xf32>
  }
})mlir";

/// The kernel a test adapter emits. Its workload graph has exactly one node
/// (`micro.vector`), which the test contract declares.
constexpr llvm::StringLiteral emittedKernel = R"mlir(
module {
  micro.kernel @emitted(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>)
      -> tensor<8x8xf32> attributes {micro.numerical_contract = "strict-f32"} {
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>}
      : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b {shape = array<i64: 8, 8>}
      : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %s = micro.vector "add" %ta, %tb
      : !micro.tile<8x8xf32, memory = #micro.memory<sram>>,
        !micro.tile<8x8xf32, memory = #micro.memory<sram>>
        -> !micro.tile<8x8xf32, memory = #micro.memory<acc>>
    micro.yield %s : !micro.tile<8x8xf32, memory = #micro.memory<acc>>
  }
})mlir";

workload::WorkloadContract addContract() {
  workload::WorkloadContract contract;
  contract.numericalContract = "strict-f32";
  contract.ports = {{"input0",
                     workload::PortDirection::Input,
                     workload::PortEffect::Read,
                     {8, 8},
                     "f32"},
                    {"input1",
                     workload::PortDirection::Input,
                     workload::PortEffect::Read,
                     {8, 8},
                     "f32"},
                    {"output0",
                     workload::PortDirection::Output,
                     workload::PortEffect::Write,
                     {8, 8},
                     "f32"}};
  contract.requiredPrimitives = {"micro.vector"};
  return contract;
}

perf::SearchParam integerParam(std::string name, std::string role,
                               std::vector<int64_t> choices) {
  perf::SearchParam param;
  param.name = std::move(name);
  param.kind = "integer";
  param.role = std::move(role);
  for (int64_t choice : choices)
    param.choices.emplace_back(choice);
  return param;
}

perf::SearchParam symbolicParam(std::string name, std::string kind,
                                std::string role, std::string choice) {
  perf::SearchParam param;
  param.name = std::move(name);
  param.kind = std::move(kind);
  param.role = std::move(role);
  param.choices.emplace_back(std::move(choice));
  return param;
}

perf::SearchSpace spaceWith(std::vector<perf::SearchParam> params) {
  perf::SearchSpace space;
  space.name = "adapter";
  space.workload = "diagnostic-only-label";
  space.params = std::move(params);
  return space;
}

perf::Candidate candidateWith(std::map<std::string, int64_t> values,
                              std::map<std::string, std::string> symbols) {
  perf::Candidate candidate;
  candidate.values = std::move(values);
  candidate.symbolicValues = std::move(symbols);
  candidate.id =
      perf::computeCandidateId(candidate.values, candidate.symbolicValues);
  return candidate;
}

/// A process-wide context for the tests that need no fixture-owned state.
MLIRContext &sourceTestContext() {
  static DialectRegistry dialects = [] {
    DialectRegistry registry;
    registry.insert<micro::MicroDialect, llk::LLKDialect, func::FuncDialect,
                    linalg::LinalgDialect, tensor::TensorDialect,
                    arith::ArithDialect>();
    return registry;
  }();
  static MLIRContext context(dialects);
  return context;
}

OwningOpRef<ModuleOp> parseIn(llvm::StringRef text) {
  return parseSourceString<ModuleOp>(text, &sourceTestContext());
}

std::string messageOf(llvm::Error error) {
  return llvm::toString(std::move(error));
}

std::string messageOf(llvm::Expected<const workload::SourceAdapter *> adapter) {
  return llvm::toString(adapter.takeError());
}

std::string messageOf(llvm::Expected<Operation *> root) {
  return llvm::toString(root.takeError());
}

void expectDiagnostic(llvm::Error error, llvm::StringRef expected) {
  ASSERT_TRUE(bool(error));
  EXPECT_NE(messageOf(std::move(error)).find(expected.str()), std::string::npos)
      << "expected diagnostic '" << expected.str() << "'";
}

/// A minimal, test-only adapter. It recognizes one operation name, describes a
/// contract the test supplies, declares a schema the test supplies, and emits a
/// parsed template kernel. `realized` names the source-owned parameters it
/// actually realizes: a receipt entry appears only for those, which is what
/// makes "an offered choice cannot disappear" observable rather than assumed.
class TestAdapter final : public workload::SourceAdapter {
public:
  TestAdapter(std::string key, std::string acceptsOp,
              workload::WorkloadContract described,
              std::vector<workload::ParameterBinding> schema,
              std::vector<std::string> realized)
      : adapterKey(std::move(key)), acceptsOp(std::move(acceptsOp)),
        described(std::move(described)), schema(std::move(schema)),
        realized(std::move(realized)) {}

  llvm::StringRef key() const override { return adapterKey; }

  bool accepts(Operation *op) const override {
    return op->getName().getStringRef() == acceptsOp;
  }

  llvm::Expected<workload::WorkloadContract>
  describe(Operation *) const override {
    return described;
  }

  llvm::Expected<std::vector<workload::ParameterBinding>>
  bindings(Operation *, const perf::SearchSpace &) const override {
    return schema;
  }

  llvm::Expected<workload::SourceRealization>
  realize(ModuleOp clone, Operation *, const workload::WorkloadContract &,
          const perf::SearchSpace &, const perf::Candidate &) const override {
    mlir::OwningOpRef<ModuleOp> parsed =
        mlir::parseSourceString<ModuleOp>(emittedKernel, clone.getContext());
    if (!parsed)
      return llvm::make_error<llvm::StringError>(
          "test adapter cannot parse its emitted kernel",
          llvm::inconvertibleErrorCode());
    // Move the parsed kernel into the caller's clone, then find it there.
    clone.getBody()->getOperations().splice(
        clone.getBody()->getOperations().end(),
        parsed->getBody()->getOperations());
    Operation *kernel = SymbolTable::lookupSymbolIn(clone, "emitted");
    if (!kernel)
      return llvm::make_error<llvm::StringError>(
          "test adapter emitted no kernel", llvm::inconvertibleErrorCode());
    workload::SourceRealization realization;
    realization.kernel = kernel;
    for (const workload::ParameterBinding &binding : schema)
      if (binding.owner == workload::BindingOwner::Source &&
          std::find(realized.begin(), realized.end(), binding.parameter) !=
              realized.end())
        realization.receipt.source.push_back(binding.parameter);
    return realization;
  }

private:
  std::string adapterKey;
  std::string acceptsOp;
  workload::WorkloadContract described;
  std::vector<workload::ParameterBinding> schema;
  std::vector<std::string> realized;
};

workload::ParameterBinding sourceBinding(std::string parameter,
                                         std::string role) {
  return workload::ParameterBinding{
      std::move(parameter), workload::BindingOwner::Source, std::move(role)};
}

workload::ParameterBinding mappingBinding(std::string parameter,
                                          std::string role) {
  return workload::ParameterBinding{
      std::move(parameter), workload::BindingOwner::Mapping, std::move(role)};
}

std::unique_ptr<TestAdapter>
acceptingElementwise(std::string key,
                     std::vector<workload::ParameterBinding> schema = {},
                     std::vector<std::string> realized = {}) {
  return std::make_unique<TestAdapter>(std::move(key), "linalg.generic",
                                       addContract(), std::move(schema),
                                       std::move(realized));
}

/// One source-owned axis (`T0` on `axis:d0`) and one mapping-owned choice
/// (`L0`, a layout). The candidate binds both; the source stage realizes only
/// the axis.
struct SplitOwnership {
  perf::SearchSpace space;
  perf::Candidate candidate;
  std::vector<workload::ParameterBinding> schema;

  SplitOwnership()
      : space(spaceWith({integerParam("T0", "axis:d0", {4}),
                         symbolicParam("L0", "layout", "", "row_major")})),
        candidate(candidateWith({{"T0", 4}}, {{"L0", "row_major"}})),
        schema(
            {sourceBinding("T0", "axis:d0"), mappingBinding("L0", "layout")}) {}
};

} // namespace

//===----------------------------------------------------------------------===//
// BindingSchema
//===----------------------------------------------------------------------===//

TEST(BindingSchema, AnOfferedChoiceCannotDisappear) {
  perf::SearchSpace space = spaceWith({integerParam("T0", "axis:d0", {4})});
  perf::Candidate candidate = candidateWith({{"T0", 4}}, {});
  std::vector<workload::ParameterBinding> schema = {
      sourceBinding("T0", "axis:d0")};

  workload::BindingReceipt receipt;
  expectDiagnostic(
      workload::validateBindingReceipt(space, candidate, schema, receipt),
      "axis_not_realized: T0");

  receipt.source = {"T0"};
  EXPECT_FALSE(
      workload::validateBindingReceipt(space, candidate, schema, receipt));

  receipt.mapping = {"T0"};
  expectDiagnostic(
      workload::validateBindingReceipt(space, candidate, schema, receipt),
      "unexpected_receipt: T0");
}

TEST(BindingSchema, EveryParameterHasExactlyOneOwner) {
  perf::SearchSpace space = spaceWith(
      {integerParam("T0", "axis:d0", {4}), integerParam("T1", "axis:d1", {2})});
  perf::Candidate candidate = candidateWith({{"T0", 4}, {"T1", 2}}, {});
  workload::BindingReceipt receipt;
  receipt.source = {"T0", "T1"};

  expectDiagnostic(
      workload::validateBindingReceipt(space, candidate,
                                       {sourceBinding("T0", "axis:d0"),
                                        sourceBinding("T0", "axis:d0"),
                                        sourceBinding("T1", "axis:d1")},
                                       receipt),
      "duplicate_owner: T0");
  expectDiagnostic(
      workload::validateBindingReceipt(
          space, candidate, {sourceBinding("T0", "axis:d0")}, receipt),
      "unowned_parameter: T1");
}

TEST(BindingSchema, UnknownAndOutOfDomainChoicesFail) {
  perf::SearchSpace space =
      spaceWith({integerParam("T0", "axis:d0", {4}),
                 symbolicParam("L0", "layout", "", "row_major")});
  workload::BindingReceipt receipt;
  receipt.source = {"T0", "L0"};

  expectDiagnostic(
      workload::validateBindingReceipt(
          space, candidateWith({{"T0", 4}}, {{"L0", "row_major"}}),
          {sourceBinding("T0", "axis:d0"), sourceBinding("TX", "axis:d0"),
           mappingBinding("L0", "layout")},
          receipt),
      "unknown_parameter: TX");

  expectDiagnostic(
      workload::validateBindingReceipt(
          space, candidateWith({{"T0", 5}}, {{"L0", "row_major"}}),
          {sourceBinding("T0", "axis:d0"), mappingBinding("L0", "layout")},
          receipt),
      "out_of_domain: T0");

  // A symbolic binding where the space declares an integer domain is not an
  // acceptable substitute, even though the parameter name matches.
  expectDiagnostic(
      workload::validateBindingReceipt(
          space, candidateWith({}, {{"L0", "row_major"}, {"T0", "4"}}),
          {sourceBinding("T0", "axis:d0"), mappingBinding("L0", "layout")},
          receipt),
      "out_of_domain: T0");
}

TEST(BindingSchema, RolesComeFromDeclarationsNotParameterSpelling) {
  perf::SearchSpace space = spaceWith(
      {integerParam("T0", "axis:d0", {4}), integerParam("T1", "", {2})});
  perf::Candidate candidate = candidateWith({{"T0", 4}, {"T1", 2}}, {});
  workload::BindingReceipt receipt;
  receipt.source = {"T0", "T1"};

  // "axis:0" happens to look like the parameter name's digits, but the space
  // declares "axis:d0": the declared role wins.
  expectDiagnostic(
      workload::validateBindingReceipt(
          space, candidate,
          {sourceBinding("T0", "axis:0"), sourceBinding("T1", "axis:d1")},
          receipt),
      "role_mismatch: T0");

  // A parameter that declares no role still needs one, and a source axis is
  // never inferred from the parameter's spelling.
  expectDiagnostic(
      workload::validateBindingReceipt(
          space, candidate,
          {sourceBinding("T0", "axis:d0"), sourceBinding("T1", "T1")}, receipt),
      "role_mismatch: T1");
}

TEST(BindingSchema, MappingRolesFollowTheDeclaredKind) {
  perf::SearchSpace space =
      spaceWith({symbolicParam("L0", "layout", "", "row_major"),
                 symbolicParam("MP0", "memory_path", "", "sram_direct")});
  perf::Candidate candidate =
      candidateWith({}, {{"L0", "row_major"}, {"MP0", "sram_direct"}});
  workload::BindingReceipt receipt;
  receipt.mapping = {"L0", "MP0"};

  // With no declared role, a mapping choice's role is its declared kind.
  EXPECT_FALSE(workload::validateBindingReceipt(
      space, candidate,
      {mappingBinding("L0", "layout"), mappingBinding("MP0", "memory_path")},
      receipt));
  expectDiagnostic(
      workload::validateBindingReceipt(
          space, candidate,
          {mappingBinding("L0", "layout"), mappingBinding("MP0", "layout")},
          receipt),
      "role_mismatch: MP0");

  // A declared role wins over the kind, so several parameters can share a kind
  // while still naming distinct ports.
  perf::SearchSpace ports =
      spaceWith({symbolicParam("L0", "layout", "operand0", "row_major"),
                 symbolicParam("L1", "layout", "operand1", "row_major")});
  perf::Candidate bothPorts =
      candidateWith({}, {{"L0", "row_major"}, {"L1", "row_major"}});
  workload::BindingReceipt bothReceipt;
  bothReceipt.mapping = {"L0", "L1"};
  EXPECT_FALSE(workload::validateBindingReceipt(
      ports, bothPorts,
      {mappingBinding("L0", "operand0"), mappingBinding("L1", "operand1")},
      bothReceipt));
  expectDiagnostic(
      workload::validateBindingReceipt(
          ports, bothPorts,
          {mappingBinding("L0", "operand0"), mappingBinding("L1", "layout")},
          bothReceipt),
      "role_mismatch: L1");

  // Two ports that share one role are one role claiming two choices; the
  // declared roles agree, so the duplicate role is what rejects.
  perf::SearchSpace shared =
      spaceWith({symbolicParam("L0", "layout", "operand0", "row_major"),
                 symbolicParam("L1", "layout", "operand0", "row_major")});
  expectDiagnostic(
      workload::validateBindingReceipt(
          shared, bothPorts,
          {mappingBinding("L0", "operand0"), mappingBinding("L1", "operand0")},
          bothReceipt),
      "duplicate_role: operand0");
}

TEST(BindingSchema, OneRoleIdentifiesOneAxisOrPort) {
  perf::SearchSpace space = spaceWith(
      {integerParam("T0", "axis:d0", {4}), integerParam("T1", "axis:d0", {2})});
  perf::Candidate candidate = candidateWith({{"T0", 4}, {"T1", 2}}, {});
  workload::BindingReceipt receipt;
  receipt.source = {"T0", "T1"};

  expectDiagnostic(
      workload::validateBindingReceipt(
          space, candidate,
          {sourceBinding("T0", "axis:d0"), sourceBinding("T1", "axis:d0")},
          receipt),
      "duplicate_role: axis:d0");
}

TEST(BindingSchema, SourceStageKeepsMappingChoicesPending) {
  SplitOwnership split;
  workload::BindingReceipt receipt;
  receipt.source = {"T0"};

  EXPECT_FALSE(workload::validateBindingReceipt(
      split.space, split.candidate, split.schema, receipt,
      workload::ReceiptStage::Source));
  // The same receipt is not complete evidence: the mapping choice is offered,
  // owned by the mapping stage, and still unrealized.
  expectDiagnostic(workload::validateBindingReceipt(
                       split.space, split.candidate, split.schema, receipt,
                       workload::ReceiptStage::Complete),
                   "axis_not_realized: L0");

  receipt.mapping = {"L0"};
  EXPECT_FALSE(workload::validateBindingReceipt(
      split.space, split.candidate, split.schema, receipt,
      workload::ReceiptStage::Complete));
  // A mapping choice cannot be claimed before the mapping stage runs.
  expectDiagnostic(workload::validateBindingReceipt(
                       split.space, split.candidate, split.schema, receipt,
                       workload::ReceiptStage::Source),
                   "unexpected_receipt: L0");
}

TEST(BindingSchema, ReceiptEntriesMustNameSchemaParameters) {
  perf::SearchSpace space = spaceWith({integerParam("T0", "axis:d0", {4})});
  perf::Candidate candidate = candidateWith({{"T0", 4}}, {});
  workload::BindingReceipt receipt;
  receipt.source = {"T0", "TX"};

  expectDiagnostic(
      workload::validateBindingReceipt(
          space, candidate, {sourceBinding("T0", "axis:d0")}, receipt),
      "unknown_receipt: TX");

  workload::BindingReceipt duplicated;
  duplicated.source = {"T0", "T0"};
  expectDiagnostic(
      workload::validateBindingReceipt(
          space, candidate, {sourceBinding("T0", "axis:d0")}, duplicated),
      "duplicate_receipt: T0");
}

TEST(BindingSchema, CandidateKeysMustBeDeclaredParameterNames) {
  perf::SearchSpace space = spaceWith({integerParam("T0", "axis:d0", {4})});
  std::vector<workload::ParameterBinding> schema = {
      sourceBinding("T0", "axis:d0")};
  workload::BindingReceipt receipt;
  receipt.source = {"T0"};

  expectDiagnostic(workload::validateBindingReceipt(
                       space, candidateWith({}, {}), schema, receipt),
                   "missing_binding: T0");
  expectDiagnostic(
      workload::validateBindingReceipt(
          space, candidateWith({{"T0", 4}, {"TX", 1}}, {}), schema, receipt),
      "unknown_binding: TX");
}

//===----------------------------------------------------------------------===//
// SourceAdapterRegistry
//===----------------------------------------------------------------------===//

TEST(SourceAdapterRegistry, RegistrationRejectsNullAndDuplicateKeys) {
  workload::SourceAdapterRegistry registry;
  EXPECT_FALSE(registry.add(acceptingElementwise("first")));
  EXPECT_FALSE(registry.add(acceptingElementwise("second")));
  expectDiagnostic(registry.add(acceptingElementwise("first")),
                   "duplicate_source_adapter: first");
  expectDiagnostic(registry.add(nullptr), "null_source_adapter");
}

TEST(SourceAdapterRegistry, ZeroMatchesIsUnsupported) {
  auto source = parseIn(elementwiseSource);
  ASSERT_TRUE(source);
  Operation *root = nullptr;
  (*source).walk([&](Operation *op) {
    if (!root && op->getName().getStringRef() == "linalg.generic")
      root = op;
  });
  ASSERT_TRUE(root);

  workload::SourceAdapterRegistry defaults =
      workload::makeDefaultSourceAdapters();
  auto unmatched = defaults.resolve(root);
  ASSERT_FALSE(bool(unmatched));
  EXPECT_NE(messageOf(std::move(unmatched)).find("unsupported_source_adapter"),
            std::string::npos);

  // An adapter keyed on a different family does not claim this root either.
  workload::SourceAdapterRegistry other;
  ASSERT_FALSE(other.add(std::make_unique<TestAdapter>(
      "matmul-only", "linalg.matmul", addContract(),
      std::vector<workload::ParameterBinding>{}, std::vector<std::string>{})));
  auto stillUnmatched = other.resolve(root);
  ASSERT_FALSE(bool(stillUnmatched));
  EXPECT_NE(
      messageOf(std::move(stillUnmatched)).find("unsupported_source_adapter"),
      std::string::npos);
}

TEST(SourceAdapterRegistry, TwoMatchesRequireAnExplicitKey) {
  auto source = parseIn(elementwiseSource);
  ASSERT_TRUE(source);
  Operation *root = nullptr;
  (*source).walk([&](Operation *op) {
    if (!root && op->getName().getStringRef() == "linalg.generic")
      root = op;
  });
  ASSERT_TRUE(root);

  workload::SourceAdapterRegistry registry;
  ASSERT_FALSE(registry.add(acceptingElementwise("first")));
  ASSERT_FALSE(registry.add(acceptingElementwise("second")));

  auto ambiguous = registry.resolve(root);
  ASSERT_FALSE(bool(ambiguous));
  EXPECT_NE(messageOf(std::move(ambiguous)).find("ambiguous_source_adapter"),
            std::string::npos);

  auto explicitKey = registry.resolve(root, "second");
  ASSERT_TRUE(bool(explicitKey)) << messageOf(std::move(explicitKey));
  EXPECT_EQ((*explicitKey)->key(), "second");

  auto unknown = registry.resolve(root, "missing");
  ASSERT_FALSE(bool(unknown));
  EXPECT_NE(
      messageOf(std::move(unknown)).find("unknown_source_adapter: missing"),
      std::string::npos);
}

TEST(SourceAdapterRegistry, AnAdapterOnlyAnswersForItsOwnFamily) {
  auto source = parseIn(elementwiseSource);
  ASSERT_TRUE(source);
  Operation *generic = nullptr;
  Operation *empty = nullptr;
  (*source).walk([&](Operation *op) {
    if (!generic && op->getName().getStringRef() == "linalg.generic")
      generic = op;
    if (!empty && op->getName().getStringRef() == "tensor.empty")
      empty = op;
  });
  ASSERT_TRUE(generic);
  ASSERT_TRUE(empty);

  workload::SourceAdapterRegistry registry;
  ASSERT_FALSE(registry.add(acceptingElementwise("elementwise-v1")));
  ASSERT_TRUE(bool(registry.resolve(generic)));
  auto mismatch = registry.resolve(empty, "elementwise-v1");
  ASSERT_FALSE(bool(mismatch));
  EXPECT_NE(messageOf(std::move(mismatch))
                .find("adapter_rejects_source: elementwise-v1"),
            std::string::npos);
}

//===----------------------------------------------------------------------===//
// selectSourceRoot
//===----------------------------------------------------------------------===//

TEST(SourceSelection, CountsOnlyAcceptedTopLevelRoots) {
  auto source = parseIn(elementwiseSource);
  ASSERT_TRUE(source);
  workload::SourceAdapterRegistry registry;
  ASSERT_FALSE(registry.add(acceptingElementwise("elementwise-v1")));

  auto selected =
      workload::selectSourceRoot(*source, {"elementwise", 0}, registry);
  ASSERT_TRUE(bool(selected)) << messageOf(std::move(selected));
  EXPECT_EQ((*selected)->getName().getStringRef(), "linalg.generic");

  // The nested scalar body and the terminator are not roots.
  expectDiagnostic(
      workload::selectSourceRoot(*source, {"elementwise", 1}, registry)
          .takeError(),
      "source_root_ordinal 1 is out of range for @elementwise");
}

TEST(SourceSelection, SelectsTheExactOrdinalWithinTheFunction) {
  auto source = parseIn(twoRootSource);
  ASSERT_TRUE(source);
  workload::SourceAdapterRegistry registry;
  ASSERT_FALSE(registry.add(acceptingElementwise("elementwise-v1")));

  auto first = workload::selectSourceRoot(*source, {"pair", 0}, registry);
  ASSERT_TRUE(bool(first)) << messageOf(std::move(first));
  auto second = workload::selectSourceRoot(*source, {"pair", 1}, registry);
  ASSERT_TRUE(bool(second)) << messageOf(std::move(second));
  EXPECT_NE(*first, *second);

  // Identifies which root the ordinal picked by the op its body yields.
  auto bodyOf = [](Operation *root) -> llvm::StringRef {
    if (root->getName().getStringRef() != "linalg.generic" ||
        root->getNumRegions() == 0 || root->getRegion(0).empty())
      return {};
    for (Operation &op : root->getRegion(0).front().getOperations())
      if (op.getName().getStringRef() == "linalg.yield")
        return op.getOperand(0).getDefiningOp()->getName().getStringRef();
    return {};
  };
  EXPECT_EQ(bodyOf(*first), "arith.addf");
  EXPECT_EQ(bodyOf(*second), "arith.mulf");
}

TEST(SourceSelection, UnrelatedFunctionsDoNotContributeOrdinals) {
  auto source = parseIn(twoFunctionSource);
  ASSERT_TRUE(source);
  workload::SourceAdapterRegistry registry;
  ASSERT_FALSE(registry.add(acceptingElementwise("elementwise-v1")));

  auto left = workload::selectSourceRoot(*source, {"left", 0}, registry);
  ASSERT_TRUE(bool(left)) << messageOf(std::move(left));
  expectDiagnostic(
      workload::selectSourceRoot(*source, {"left", 1}, registry).takeError(),
      "source_root_ordinal 1 is out of range for @left");
  auto right = workload::selectSourceRoot(*source, {"right", 0}, registry);
  ASSERT_TRUE(bool(right)) << messageOf(std::move(right));
  EXPECT_NE(*left, *right);

  expectDiagnostic(
      workload::selectSourceRoot(*source, {"nosuch", 0}, registry).takeError(),
      "unknown source function '@nosuch'");
  // An empty symbol only works when the module declares one function.
  expectDiagnostic(
      workload::selectSourceRoot(*source, {"", 0}, registry).takeError(),
      "source function");
}

//===----------------------------------------------------------------------===//
// Adapter dispatch through candidate instantiation
//===----------------------------------------------------------------------===//

TEST_F(FrameworkTest, AdapterRealizesOfferedSourceBindings) {
  auto source = parseIn(elementwiseSource);
  ASSERT_TRUE(source);
  const std::string before = print(source->getOperation());
  SplitOwnership split;

  workload::SourceAdapterRegistry registry;
  ASSERT_FALSE(registry.add(
      acceptingElementwise("elementwise-v1", split.schema, {"T0"})));

  tuning::CandidateInstantiationOptions options;
  options.sourceMode = tuning::SourceMode::SemanticSource;
  options.sourceSymbol = "elementwise";
  auto instance = tuning::instantiateCandidate(
      *source, split.space, split.candidate, addContract(), options, registry);
  ASSERT_TRUE(bool(instance)) << llvm::toString(instance.takeError());
  EXPECT_EQ(instance->receipt.source, std::vector<std::string>{"T0"});
  EXPECT_TRUE(instance->receipt.mapping.empty());
  EXPECT_EQ(print(source->getOperation()), before);
  EXPECT_NE(instance->sourceGraphHash, 0u);
  ASSERT_EQ(instance->workloadGraph.getNodes().size(), 1u);
  EXPECT_EQ(instance->workloadGraph.getNodes()[0].opName, "micro.vector");
  EXPECT_EQ(
      instance->kernel->getAttrOfType<StringAttr>("source_adapter").getValue(),
      "elementwise-v1");
  EXPECT_EQ(
      instance->kernel->getAttrOfType<StringAttr>("source_symbol").getValue(),
      "elementwise");

  // The returned receipt is source-stage evidence: completing it is the mapping
  // stage's business, and the mapping choice is still pending here.
  expectDiagnostic(workload::validateBindingReceipt(
                       split.space, split.candidate, split.schema,
                       instance->receipt, workload::ReceiptStage::Complete),
                   "axis_not_realized: L0");
}

TEST_F(FrameworkTest, OfferedSourceBindingCannotDisappear) {
  auto source = parseIn(elementwiseSource);
  ASSERT_TRUE(source);
  SplitOwnership split;

  workload::SourceAdapterRegistry registry;
  ASSERT_FALSE(registry.add(
      acceptingElementwise("elementwise-v1", split.schema, /*realized=*/{})));

  tuning::CandidateInstantiationOptions options;
  options.sourceMode = tuning::SourceMode::SemanticSource;
  options.sourceSymbol = "elementwise";
  auto instance = tuning::instantiateCandidate(
      *source, split.space, split.candidate, addContract(), options, registry);
  ASSERT_FALSE(bool(instance));
  EXPECT_NE(llvm::toString(instance.takeError()).find("axis_not_realized: T0"),
            std::string::npos);
}

TEST_F(FrameworkTest, AmbiguousAdaptersRefuseToDispatch) {
  auto source = parseIn(elementwiseSource);
  ASSERT_TRUE(source);
  SplitOwnership split;

  workload::SourceAdapterRegistry registry;
  ASSERT_FALSE(
      registry.add(acceptingElementwise("first", split.schema, {"T0"})));
  ASSERT_FALSE(
      registry.add(acceptingElementwise("second", split.schema, {"T0"})));

  tuning::CandidateInstantiationOptions options;
  options.sourceMode = tuning::SourceMode::SemanticSource;
  options.sourceSymbol = "elementwise";
  auto ambiguous = tuning::instantiateCandidate(
      *source, split.space, split.candidate, addContract(), options, registry);
  ASSERT_FALSE(bool(ambiguous));
  EXPECT_NE(
      llvm::toString(ambiguous.takeError()).find("ambiguous_source_adapter"),
      std::string::npos);

  options.adapterKey = "second";
  auto explicitKey = tuning::instantiateCandidate(
      *source, split.space, split.candidate, addContract(), options, registry);
  ASSERT_TRUE(bool(explicitKey)) << llvm::toString(explicitKey.takeError());
  EXPECT_EQ(explicitKey->kernel->getAttrOfType<StringAttr>("source_adapter")
                .getValue(),
            "second");

  options.adapterKey = "missing";
  auto unknown = tuning::instantiateCandidate(
      *source, split.space, split.candidate, addContract(), options, registry);
  ASSERT_FALSE(bool(unknown));
  EXPECT_NE(llvm::toString(unknown.takeError())
                .find("unknown_source_adapter: missing"),
            std::string::npos);
}

TEST_F(FrameworkTest, DescribedContractMustMatchTheSuppliedContract) {
  auto source = parseIn(elementwiseSource);
  ASSERT_TRUE(source);
  SplitOwnership split;

  workload::WorkloadContract other = addContract();
  other.numericalContract = "approximate-f32";

  workload::SourceAdapterRegistry registry;
  ASSERT_FALSE(registry.add(std::make_unique<TestAdapter>(
      "elementwise-v1", "linalg.generic", other, split.schema,
      std::vector<std::string>{"T0"})));

  tuning::CandidateInstantiationOptions options;
  options.sourceMode = tuning::SourceMode::SemanticSource;
  options.sourceSymbol = "elementwise";
  auto instance = tuning::instantiateCandidate(
      *source, split.space, split.candidate, addContract(), options, registry);
  ASSERT_FALSE(bool(instance));
  EXPECT_NE(
      llvm::toString(instance.takeError()).find("source_contract_mismatch"),
      std::string::npos);
}

TEST_F(FrameworkTest, DefaultAdaptersRejectASemanticSource) {
  auto source = parseIn(elementwiseSource);
  ASSERT_TRUE(source);
  tuning::CandidateInstantiationOptions options;
  options.sourceMode = tuning::SourceMode::SemanticSource;
  options.sourceSymbol = "elementwise";

  // Without injected adapters the descriptor overload constructs the defaults;
  // an unrecognized family is unsupported rather than silently first-matched.
  auto instance = tuning::instantiateCandidate(
      *source, spaceWith({}), candidateWith({}, {}), addContract(), options);
  ASSERT_FALSE(bool(instance));
  EXPECT_NE(
      llvm::toString(instance.takeError()).find("unsupported_source_adapter"),
      std::string::npos);

  options.sourceMode = tuning::SourceMode::Synthetic;
  auto synthetic = tuning::instantiateCandidate(
      *source, spaceWith({}), candidateWith({}, {}), addContract(), options);
  ASSERT_FALSE(bool(synthetic));
  EXPECT_NE(llvm::toString(synthetic.takeError()).find("synthetic"),
            std::string::npos);
}
