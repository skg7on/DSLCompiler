//===- resource_regression_fixture.cpp - Issue #129 shared test fixtures --===//
//
// See resource_regression_fixture.h. Every case is a `CaseSpec`: Micro-IR text,
// LLKMap rule text, an in-memory MachineModel, and the emitter keys the target
// declares. Nothing here consults the production search, storage planner or
// cost model to decide what a case *should* produce -- the test states that in
// literal assertions.
//
//===----------------------------------------------------------------------===//

#include "resource_regression_fixture.h"

#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Machine/MachineModel.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/LayoutConstraints.h"
#include "LLK/Mapping/MappingRules.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/WorkloadGraph.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>
#include <string>
#include <vector>

namespace issue129 {

namespace {

using namespace mlir;
using namespace mlir::llk::mapping;

namespace machine = mlir::llk::machine;

llvm::Error fixtureError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

/// The `micro.kernel` in `module`. The generated op classes are not part of the
/// dialect's public headers, so it is found by name.
Operation *findKernel(ModuleOp module) {
  Operation *kernel = nullptr;
  module->walk([&](Operation *op) {
    if (!kernel && op->getName().getStringRef() == "micro.kernel")
      kernel = op;
  });
  return kernel;
}

/// One case's inputs, in the order a case author writes them: the source
/// Micro-IR, the rules that map it, the machine those rules are placed on, and
/// the emitter keys the target declares. Later tasks may extend this (a case
/// that needs a second machine, for instance) without touching `buildCase`'s
/// callers.
struct CaseSpec {
  llvm::StringRef source;
  llvm::StringRef rules;
  machine::MachineModel machine;
  std::vector<std::string> emitters;
};

/// Parses `spec` and loads its target. The MLIR context owns `source`, and the
/// graph is extracted from the parsed kernel -- so `ResourceCase::graph` is the
/// fixture's *actual* source graph, not a hand-built stand-in.
llvm::Expected<ResourceCase> buildCase(const CaseSpec &spec) {
  ResourceCase c;
  c.context = std::make_unique<MLIRContext>();
  c.context->getOrLoadDialect<micro::MicroDialect>();
  c.context->getOrLoadDialect<tensor::TensorDialect>();

  c.source = parseSourceString<ModuleOp>(spec.source, c.context.get());
  if (!c.source)
    return fixtureError("issue129 fixture: the case source did not parse");
  Operation *kernel = findKernel(*c.source);
  if (!kernel)
    return fixtureError(
        "issue129 fixture: the case source has no micro.kernel");
  llvm::Expected<WorkloadGraph> graph = extractWorkloadGraph(kernel);
  if (!graph)
    return graph.takeError();
  c.graph = std::move(*graph);

  llvm::Expected<RuleRegistry> rules = parseRuleText(spec.rules, "<issue129>");
  if (!rules)
    return rules.takeError();
  c.target = std::make_unique<FileMappingTarget>(
      "issue129", spec.machine, LayoutRegistry{}, std::move(*rules),
      spec.emitters);
  return c;
}

//===----------------------------------------------------------------------===//
// R1: `two-compute`
//===----------------------------------------------------------------------===//

/// One worker `e0` with two *attached* vector engines `vpu.a` and `vpu.b`, both
/// concurrency one. Two attached engines of one kind are two legal placements
/// of one rule, so a search that loses the selection collapses both into the
/// executor's first engine.
///
/// The machine also carries two capabilities the rule can never select, so a
/// regression can prove the *whole* recorded-selection check rather than only
/// its happy path:
///   * `mxu.a`, a `matrix_engine` attached to `e0` -- recording it for the
///     `vector_engine` requirement is a wrong-kind selection;
///   * `vpu.c`, a `vector_engine` attached to `dsp.0`, whose executor kind is
///     not `worker` -- recording it is an unattached selection.
/// Neither changes the search's result: `require executor kind worker` excludes
/// `dsp.0`, and `require compute kind vector_engine` excludes `mxu.a`.
machine::MachineModel twoComputeMachine() {
  machine::MachineModel model;
  model.target = "issue129.two-compute";
  model.description = "one worker, two attached vector engines";
  model.executors = {{"e0", "worker", std::nullopt, {}, 1, {}},
                     {"dsp.0", "dsp", std::nullopt, {}, 1, {}}};
  machine::ComputeNode a;
  a.id = "vpu.a";
  a.kind = "vector_engine";
  a.attachedTo = "e0";
  a.concurrency = 1;
  machine::ComputeNode b;
  b.id = "vpu.b";
  b.kind = "vector_engine";
  b.attachedTo = "e0";
  b.concurrency = 1;
  machine::ComputeNode wrongKind;
  wrongKind.id = "mxu.a";
  wrongKind.kind = "matrix_engine";
  wrongKind.attachedTo = "e0";
  wrongKind.concurrency = 1;
  machine::ComputeNode elsewhere;
  elsewhere.id = "vpu.c";
  elsewhere.kind = "vector_engine";
  elsewhere.attachedTo = "dsp.0";
  elsewhere.concurrency = 1;
  model.computes = {a, b, wrongKind, elsewhere};
  machine::MemoryNode sram;
  sram.id = "sram.0";
  sram.kind = "sram";
  sram.visibleFrom = "e0";
  sram.capacityBytes = 1u << 20;
  sram.alignmentBytes = 64;
  model.memories = {sram};
  return model;
}

/// An `8x8xf32` elementwise add: one workload node, so a search over it yields
/// exactly one placement per selected engine.
constexpr llvm::StringLiteral kTwoComputeSource = R"mlir(
module {
  micro.kernel @two_compute {
    %a = tensor.empty() : tensor<8x8xf32>
    %b = tensor.empty() : tensor<8x8xf32>
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %ta, %tb : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)mlir";

/// A rule that requires the `vector_engine` capability and a `worker` executor,
/// so its only legal placements differ in *which* attached engine is selected.
constexpr llvm::StringLiteral kTwoComputeRules = R"llkmap(
rule issue129.vector_add v1 {
  match micro.vector(op = "add");
  require executor kind worker;
  require compute kind vector_engine;
  input "operand0";
  input "operand1";
  output "result";
  bundle "issue129.vector.add.f32";
  emit "issue129_vector_add";
  cost 4;
}
)llkmap";

CaseSpec twoComputeSpec() {
  CaseSpec spec;
  spec.source = kTwoComputeSource;
  spec.rules = kTwoComputeRules;
  spec.machine = twoComputeMachine();
  spec.emitters = {"issue129_vector_add"};
  return spec;
}

//===----------------------------------------------------------------------===//
// R3: `missing-memory`, `named-ports`
//===----------------------------------------------------------------------===//

/// One worker `e0` with a vector engine and *two* same-kind SRAM nodes, both
/// visible from `e0`. Two nodes of one kind are the ambiguity endpoint-only
/// resolution must detect: a rule that never names a port cannot say which of
/// them an occurrence's value lives in.
machine::MachineModel twoMemoryMachine() {
  machine::MachineModel model;
  model.target = "issue129.two-memory";
  model.description = "one worker, two visible sram nodes";
  model.executors = {{"e0", "worker", std::nullopt, {}, 1, {}}};
  machine::ComputeNode vpu;
  vpu.id = "vpu.a";
  vpu.kind = "vector_engine";
  vpu.attachedTo = "e0";
  vpu.concurrency = 1;
  model.computes = {vpu};
  machine::MemoryNode a;
  a.id = "sram.a";
  a.kind = "sram";
  a.visibleFrom = "e0";
  a.capacityBytes = 1u << 20;
  a.alignmentBytes = 64;
  machine::MemoryNode b;
  b.id = "sram.b";
  b.kind = "sram";
  b.visibleFrom = "e0";
  b.capacityBytes = 1u << 20;
  b.alignmentBytes = 64;
  model.memories = {a, b};
  return model;
}

/// An `8x8xf32` elementwise add over SRAM tiles: one workload node, whose
/// operand and result tiles both state `memory = #micro.memory<sram>`. The
/// kernel is the same for both R3 cases; only the rule differs.
constexpr llvm::StringLiteral kTwoMemorySource = R"mlir(
module {
  micro.kernel @two_memory {
    %a = micro.tile_alloc : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %a, %a : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)mlir";

/// A rule that declares *no* memory requirement at all. The occurrence's tile
/// states an SRAM kind, and the machine offers two SRAM nodes to the selected
/// executor, so a strict binding has no fact that names one of them: the
/// endpoint is ambiguous and must be refused rather than bound to the first.
constexpr llvm::StringLiteral kMissingMemoryRules = R"llkmap(
rule issue129.vector_add_nomem {
  match micro.vector(op = "add");
  require executor kind worker;
  require compute kind vector_engine;
  input "operand0";
  input "operand1";
  output "result";
  bundle "issue129.vector.add.nomem";
  emit "issue129_vector_add";
  cost 4;
}
)llkmap";

/// The same add, but every occurrence is named: both inputs and the result each
/// carry their own `require memory ... kind sram`. The two same-kind nodes are
/// then a *choice* the search enumerates and records per occurrence, so a
/// strict binding is unambiguous and the selections must survive the round
/// trip.
constexpr llvm::StringLiteral kNamedPortMemoryRules = R"llkmap(
rule issue129.vector_add_named {
  match micro.vector(op = "add");
  require executor kind worker;
  require compute kind vector_engine;
  require memory input "operand0" kind sram;
  require memory input "operand1" kind sram;
  require memory output "result" kind sram;
  input "operand0";
  input "operand1";
  output "result";
  bundle "issue129.vector.add.named";
  emit "issue129_vector_add";
  cost 4;
}
)llkmap";

CaseSpec missingMemorySpec() {
  CaseSpec spec;
  spec.source = kTwoMemorySource;
  spec.rules = kMissingMemoryRules;
  spec.machine = twoMemoryMachine();
  spec.emitters = {"issue129_vector_add"};
  return spec;
}

CaseSpec namedPortsSpec() {
  CaseSpec spec;
  spec.source = kTwoMemorySource;
  spec.rules = kNamedPortMemoryRules;
  spec.machine = twoMemoryMachine();
  spec.emitters = {"issue129_vector_add"};
  return spec;
}

//===----------------------------------------------------------------------===//
// Case table
//===----------------------------------------------------------------------===//

/// A named case builder. Later tasks append one entry and own the names they
/// add; the constructor below stays unchanged.
struct CaseEntry {
  llvm::StringLiteral name;
  llvm::Expected<ResourceCase> (*build)();
};

llvm::Expected<ResourceCase> buildTwoCompute() {
  return buildCase(twoComputeSpec());
}

llvm::Expected<ResourceCase> buildMissingMemory() {
  return buildCase(missingMemorySpec());
}

llvm::Expected<ResourceCase> buildNamedPorts() {
  return buildCase(namedPortsSpec());
}

llvm::ArrayRef<CaseEntry> caseTable() {
  static const CaseEntry table[] = {
      {"two-compute", &buildTwoCompute},
      {"missing-memory", &buildMissingMemory},
      {"named-ports", &buildNamedPorts},
  };
  return table;
}

} // namespace

llvm::Expected<ResourceCase> resourceCase(llvm::StringRef name) {
  for (const CaseEntry &entry : caseTable())
    if (entry.name == name)
      return entry.build();
  std::string known;
  for (const CaseEntry &entry : caseTable()) {
    if (!known.empty())
      known += ", ";
    known += entry.name.str();
  }
  return fixtureError("issue129 fixture: unknown resource case '" + name +
                      "'; known cases are " + known);
}

llvm::Expected<MappingSearchResult>
searchCase(ResourceCase &c, const MappingSearchOptions &options) {
  if (!c.target)
    return fixtureError("issue129 fixture: the case has no target");
  if (!c.context)
    return fixtureError("issue129 fixture: the case has no MLIR context");
  // The fixture's values are 2-D `f32` tiles, so the layout context the search
  // solves any layout requirement against names that rank and element type.
  LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "f32";
  CoveringSearch search(c.graph, *c.target, *c.context, layoutContext, options);
  return search.search();
}

} // namespace issue129
