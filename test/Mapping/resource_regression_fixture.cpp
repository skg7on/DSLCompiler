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
  /// The LLKMap layout text the target declares, empty for a case whose rules
  /// require no layout (issue #129, task R5).
  llvm::StringRef layouts{};
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
  LayoutRegistry layouts;
  if (!spec.layouts.empty()) {
    llvm::Expected<LayoutRegistry> parsed =
        parseLayoutText(spec.layouts, "<issue129 layouts>");
    if (!parsed)
      return parsed.takeError();
    layouts = std::move(*parsed);
  }
  c.target = std::make_unique<FileMappingTarget>(
      "issue129", spec.machine, std::move(layouts), std::move(*rules),
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
// R4: `two-hop`
//===----------------------------------------------------------------------===//

/// Two visibility scopes, so the value the vector writes in `sram.0` cannot be
/// read by the store placed where `dram.0` lives: the edge is a real transfer,
/// and the only legal route from SRAM to DRAM is sram.0 -> l2.0 -> dram.0.
///
/// Each link names its *own* transfer engine (`dma.a` for the first hop,
/// `dma.b` for the second), because routing records only the first engine a
/// link lists -- an engine no link names is dead, and one link naming both
/// would collapse the two hops onto a single engine.
///
/// `l2CapacityBytes` is a parameter so the same fixture can state the
/// intermediate's capacity literally: 256 bytes holds the 8x8xf32 value, 128
/// does not.
machine::MachineModel twoHopMachine(uint64_t l2CapacityBytes) {
  machine::MachineModel model;
  model.target = "issue129.two-hop";
  model.description = "sram and dram in two scopes, l2 between them";
  machine::ExecutorNode clusterA;
  clusterA.id = "cluster.a";
  clusterA.kind = "cluster";
  clusterA.refines = {"group"};
  clusterA.concurrency = 1;
  machine::ExecutorNode workerA;
  workerA.id = "worker.a";
  workerA.kind = "worker";
  workerA.parent = "cluster.a";
  workerA.concurrency = 1;
  machine::ExecutorNode clusterB;
  clusterB.id = "cluster.b";
  clusterB.kind = "cluster";
  clusterB.refines = {"group"};
  clusterB.concurrency = 1;
  machine::ExecutorNode workerB;
  workerB.id = "worker.b";
  workerB.kind = "worker";
  workerB.parent = "cluster.b";
  workerB.concurrency = 1;
  model.executors = {clusterA, workerA, clusterB, workerB};

  machine::ComputeNode vpu;
  vpu.id = "vpu.a";
  vpu.kind = "vector_engine";
  vpu.attachedTo = "worker.a";
  vpu.concurrency = 1;
  model.computes = {vpu};

  machine::MemoryNode sram;
  sram.id = "sram.0";
  sram.kind = "sram";
  sram.visibleFrom = "cluster.a";
  sram.capacityBytes = 1u << 20;
  sram.alignmentBytes = 64;
  machine::MemoryNode l2;
  l2.id = "l2.0";
  l2.kind = "l2";
  l2.visibleFrom = "cluster.b";
  l2.capacityBytes = l2CapacityBytes;
  l2.alignmentBytes = 64;
  machine::MemoryNode dram;
  dram.id = "dram.0";
  dram.kind = "dram";
  dram.visibleFrom = "cluster.b";
  dram.capacityBytes = 1u << 30;
  dram.alignmentBytes = 64;
  model.memories = {sram, l2, dram};

  machine::TransferEngineNode dmaA;
  dmaA.id = "dma.a";
  dmaA.kind = "dma";
  dmaA.refines = {"transfer"};
  dmaA.attachedTo = "cluster.a";
  dmaA.count = 1;
  dmaA.maxOutstanding = 1;
  machine::TransferEngineNode dmaB;
  dmaB.id = "dma.b";
  dmaB.kind = "dma";
  dmaB.refines = {"transfer"};
  dmaB.attachedTo = "cluster.b";
  dmaB.count = 1;
  dmaB.maxOutstanding = 1;
  model.transferEngines = {dmaA, dmaB};

  machine::LinkEdge toL2;
  toL2.id = "sram_to_l2.0";
  toL2.source = "sram.0";
  toL2.destination = "l2.0";
  toL2.bandwidthBytesPerCycle = 64;
  toL2.latencyCycles = 12;
  toL2.transactionBytes = 64;
  toL2.transferEngines = {"dma.a"};
  machine::LinkEdge toDram;
  toDram.id = "l2_to_dram.0";
  toDram.source = "l2.0";
  toDram.destination = "dram.0";
  toDram.bandwidthBytesPerCycle = 32;
  toDram.latencyCycles = 220;
  toDram.transactionBytes = 64;
  toDram.transferEngines = {"dma.b"};
  model.links = {toL2, toDram};
  return model;
}

/// A vector add in SRAM whose result is consumed by a store placed in DRAM: one
/// `8x8xf32` value (256 bytes) crossing the hierarchy. The store's own
/// `dst_memory` is L2 so that the rewired operand (which the movement lands in
/// DRAM) and the store's destination stay two distinct memories, as the dialect
/// requires.
constexpr llvm::StringLiteral kTwoHopSource = R"mlir(
module {
  micro.kernel @two_hop {
    %a = micro.tile_alloc : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %a, %a : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.tile_store %r {dst_memory = #micro.memory<l2>} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)mlir";

/// The vector's work lives in SRAM and the store's in DRAM, so the value
/// between them must move.
constexpr llvm::StringLiteral kTwoHopRules = R"llkmap(
rule issue129.vector_add_sram {
  match micro.vector(op = "add");
  require executor kind worker;
  require compute kind vector_engine;
  require memory kind sram;
  input "operand0";
  input "operand1";
  output "result";
  bundle "issue129.two-hop.vector";
  emit "issue129_vector_add";
  cost 4;
}
rule issue129.store_dram {
  match micro.tile_store();
  require executor kind worker;
  require memory kind dram;
  input "source";
  bundle "issue129.two-hop.store";
  emit "issue129_store";
  cost 4;
}
)llkmap";

CaseSpec twoHopSpec() {
  CaseSpec spec;
  spec.source = kTwoHopSource;
  spec.rules = kTwoHopRules;
  spec.machine = twoHopMachine(256);
  spec.emitters = {"issue129_vector_add", "issue129_store"};
  return spec;
}

//===----------------------------------------------------------------------===//
// R5: `sequential`, `pipeline-four`, `parallel-overlap`, `padded-layout`
//===----------------------------------------------------------------------===//

/// One worker with a vector engine and a *small* L2 the 8x8xf32 results live
/// in, plus a DRAM the operands are borrowed from. The two memories are told
/// apart by the tiles' own stated kinds, so a result is charged to L2 and a
/// borrowed operand to DRAM -- which is what lets a liveness assertion name one
/// memory and mean it.
///
/// `l2CapacityBytes` is a parameter so a case can state the intermediate's
/// capacity literally: 256 holds one 256-byte result and not two.
machine::MachineModel livenessMachine(uint64_t l2CapacityBytes) {
  machine::MachineModel model;
  model.target = "issue129.liveness";
  model.description = "one worker, a small l2 and a borrowed dram operand";
  machine::ExecutorNode worker;
  worker.id = "worker.0";
  worker.kind = "worker";
  worker.concurrency = 1;
  model.executors = {worker};
  machine::ComputeNode vpu;
  vpu.id = "vpu.0";
  vpu.kind = "vector_engine";
  vpu.attachedTo = "worker.0";
  vpu.concurrency = 1;
  model.computes = {vpu};
  machine::MemoryNode l2;
  l2.id = "l2.0";
  l2.kind = "l2";
  l2.visibleFrom = "worker.0";
  l2.capacityBytes = l2CapacityBytes;
  l2.alignmentBytes = 64;
  machine::MemoryNode dram;
  dram.id = "dram.0";
  dram.kind = "dram";
  dram.visibleFrom = "worker.0";
  dram.capacityBytes = 1u << 30;
  dram.alignmentBytes = 64;
  model.memories = {l2, dram};
  return model;
}

/// The one rule every liveness case maps: an `8x8xf32` element-wise add whose
/// operands are borrowed from DRAM and whose result lives in L2. Nothing
/// declares how many times it runs -- the enclosing structural ops are what
/// say that, and liveness is what turns them into residency.
constexpr llvm::StringLiteral kLivenessRules = R"llkmap(
rule issue129.liveness_add {
  match micro.vector(op = "add");
  require executor kind worker;
  require compute kind vector_engine;
  input "operand0";
  input "operand1";
  output "result";
  bundle "issue129.liveness.add";
  emit "issue129_vector_add";
  cost 4;
}
)llkmap";

/// A widened row-major layout: every logical column occupies every other
/// physical column, so the physical image of an `8x8xf32` value is 8x15 -- 120
/// elements, 480 bytes -- against the logical 64 elements, 256 bytes. A value's
/// span is its physical image, never its index space, so that is what a live
/// range must reserve. (A row *stride* that exceeds the row is a padding the
/// layout map cannot express -- its image is bounded as a box, so only a
/// genuinely wider image changes the byte count; a padding clause in the layout
/// grammar is a separate change.)
constexpr llvm::StringLiteral kLivenessLayouts = R"llkmap(
layout issue129.padded_row_major(int S) {
  param S in [1..4];
  require rank == 2;
  require S == 2;
  implements row_major;
  map (m, n) -> (m, n * S);
}
)llkmap";

/// The same add with the padded layout required of its result, so the case's
/// occupancy reflects the physical image.
constexpr llvm::StringLiteral kPaddedLivenessRules = R"llkmap(
rule issue129.liveness_add_padded {
  match micro.vector(op = "add");
  require executor kind worker;
  require compute kind vector_engine;
  require layout result satisfies issue129.padded_row_major;
  input "operand0";
  input "operand1";
  output "result";
  bundle "issue129.liveness.add.padded";
  emit "issue129_vector_add";
  cost 4;
}
)llkmap";

/// Four *temporal* iterations of one add: the loop reuses the result buffer
/// rather than keeping four of them, so the L2 peak is one 256-byte value.
constexpr llvm::StringLiteral kSequentialSource = R"mlir(
module {
  micro.kernel @sequential {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %a = tensor.empty() : tensor<8x8xf32>
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<dram>>
    micro.for %i = %c0 to %c4 step %c1 {
      %r = micro.vector "add" %ta, %ta : !micro.tile<8x8xf32, memory = #micro.memory<dram>>, !micro.tile<8x8xf32, memory = #micro.memory<dram>> -> !micro.tile<8x8xf32, memory = #micro.memory<l2>>
    }
    micro.yield
  }
}
)mlir";

/// The same four iterations, but the add runs inside a four-stage pipeline: the
/// stages overlap, so four versions of the result are resident at once and the
/// peak is four 256-byte values.
constexpr llvm::StringLiteral kPipelineFourSource = R"mlir(
module {
  micro.kernel @pipeline_four {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %a = tensor.empty() : tensor<8x8xf32>
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<dram>>
    micro.for %i = %c0 to %c4 step %c1 {
      micro.pipeline stages = 4 {
        %r = micro.vector "add" %ta, %ta : !micro.tile<8x8xf32, memory = #micro.memory<dram>>, !micro.tile<8x8xf32, memory = #micro.memory<dram>> -> !micro.tile<8x8xf32, memory = #micro.memory<l2>>
      }
    }
    micro.yield
  }
}
)mlir";

/// Two *spatial* occurrences of the add: the owners run side by side, so both
/// 256-byte results are resident and the peak is 512 -- which a 256-byte L2
/// cannot hold.
constexpr llvm::StringLiteral kParallelOverlapSource = R"mlir(
module {
  micro.kernel @parallel_overlap {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %a = tensor.empty() : tensor<8x8xf32>
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<dram>>
    micro.spatial_for %i = %c0 to %c2 step %c1 map = #micro.map<worker> {
      %r = micro.vector "add" %ta, %ta : !micro.tile<8x8xf32, memory = #micro.memory<dram>>, !micro.tile<8x8xf32, memory = #micro.memory<dram>> -> !micro.tile<8x8xf32, memory = #micro.memory<l2>>
    }
    micro.yield
  }
}
)mlir";

/// The padded case is the sequential kernel: one result, whose padded physical
/// image is what its L2 reservation must cover.
CaseSpec sequentialSpec() {
  CaseSpec spec;
  spec.source = kSequentialSource;
  spec.rules = kLivenessRules;
  spec.machine = livenessMachine(256);
  spec.emitters = {"issue129_vector_add"};
  return spec;
}

CaseSpec pipelineFourSpec() {
  CaseSpec spec;
  spec.source = kPipelineFourSource;
  spec.rules = kLivenessRules;
  spec.machine = livenessMachine(1024);
  spec.emitters = {"issue129_vector_add"};
  return spec;
}

CaseSpec parallelOverlapSpec() {
  CaseSpec spec;
  spec.source = kParallelOverlapSource;
  spec.rules = kLivenessRules;
  spec.machine = livenessMachine(512);
  spec.emitters = {"issue129_vector_add"};
  return spec;
}

CaseSpec paddedLayoutSpec() {
  CaseSpec spec;
  spec.source = kSequentialSource;
  spec.rules = kPaddedLivenessRules;
  spec.layouts = kLivenessLayouts;
  spec.machine = livenessMachine(512);
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

llvm::Expected<ResourceCase> buildTwoHop() { return buildCase(twoHopSpec()); }

llvm::Expected<ResourceCase> buildSequential() {
  return buildCase(sequentialSpec());
}

llvm::Expected<ResourceCase> buildPipelineFour() {
  return buildCase(pipelineFourSpec());
}

llvm::Expected<ResourceCase> buildParallelOverlap() {
  return buildCase(parallelOverlapSpec());
}

llvm::Expected<ResourceCase> buildPaddedLayout() {
  return buildCase(paddedLayoutSpec());
}

llvm::ArrayRef<CaseEntry> caseTable() {
  static const CaseEntry table[] = {
      {"two-compute", &buildTwoCompute},
      {"missing-memory", &buildMissingMemory},
      {"named-ports", &buildNamedPorts},
      {"two-hop", &buildTwoHop},
      {"sequential", &buildSequential},
      {"pipeline-four", &buildPipelineFour},
      {"parallel-overlap", &buildParallelOverlap},
      {"padded-layout", &buildPaddedLayout},
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
