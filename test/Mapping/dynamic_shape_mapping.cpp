//===- dynamic_shape_mapping.cpp - Dynamic plan legality and capacity -----===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Dynamic Shape Task 3 (issue #194). Certifies that a plan whose workload
// extent is a runtime value is still a *plan*: its local tile capacities are
// static, its simultaneous residency is bounded by the selected executor pool
// and the recovered pipeline/spatial facts rather than by the logical grid, and
// every boundary behaviour it needs is a typed requirement the selected emitter
// must implement.
//
// The tests deliberately separate three facts that used to collapse into one
// rejection:
//
//   * capacity/storage legality -- certified for the whole declared domain;
//   * boundary realization -- certified by the emitter the rule selected;
//   * performance ranking -- *not* certified without concrete samples, so a
//     dynamic plan is legal and unranked rather than illegal.

#include "LLK/Mapping/BoundaryRequirements.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/PlanBinder.h"
#include "LLK/Mapping/StoragePlan.h"
#include "LLK/Mapping/WorkloadGraph.h"
#include "LLK/Shape/ShapeContract.h"

#include "LLK/Target/GenericAccelerator/Mapping/GenericAcceleratorMappingTarget.h"
#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include "LLK/Perf/SelectedKernelAnalysis.h"

// MicroDialect/MicroEnums/MicroHelpers are prerequisites for the generated op
// declarations below, not direct uses here.
#include "LLK/Dialect/Micro/MicroDialect.h" // IWYU pragma: keep
#include "LLK/Dialect/Micro/MicroEnums.h"   // IWYU pragma: keep
#include "LLK/Dialect/Micro/MicroHelpers.h" // IWYU pragma: keep

#include "BoundaryDerivation.h"
#include "MicroMappingCommon.h"

#include "resource_regression_fixture.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/Bytecode/BytecodeOpInterface.h" // IWYU pragma: keep
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Builders.h"          // IWYU pragma: keep
#include "mlir/IR/BuiltinAttributes.h" // IWYU pragma: keep
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h" // IWYU pragma: keep
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpImplementation.h"       // IWYU pragma: keep
#include "mlir/Interfaces/CallInterfaces.h" // IWYU pragma: keep
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

// Micro attribute, type, and op declarations.
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <vector>

using namespace mlir::llk::machine;
using namespace mlir::llk::mapping;

#ifndef LLK_SOURCE_DIR
#error "LLK_SOURCE_DIR must name the repository root"
#endif

namespace {

namespace avx2_mapping = mlir::llk::target::avx2;
namespace accel_mapping = mlir::llk::target::generic_accel;

//===----------------------------------------------------------------------===//
// Fixtures
//===----------------------------------------------------------------------===//

/// One worker whose pool has `concurrency` slots, a vector engine attached to
/// it, SRAM and DRAM. The pool is the certified bound on a spatial loop whose
/// extent is a runtime value.
MachineModel dynamicMachine(uint32_t concurrency, uint64_t sramCapacity) {
  MachineModel model;
  model.target = "dynamic-shape";
  model.executors = {{"worker.0", "worker", std::nullopt, {}, concurrency, {}}};
  ComputeNode vpu;
  vpu.id = "vpu.0";
  vpu.kind = "vector_engine";
  vpu.attachedTo = "worker.0";
  vpu.concurrency = concurrency;
  vpu.lanes = {{"f32", 8}};
  vpu.elementTypes = {"f32"};
  vpu.issueCycles = 1;
  vpu.latencyCycles = 4;
  model.computes = {vpu};
  MemoryNode sram;
  sram.id = "sram.0";
  sram.kind = "sram";
  sram.visibleFrom = "worker.0";
  sram.capacityBytes = sramCapacity;
  sram.alignmentBytes = 64;
  MemoryNode dram;
  dram.id = "dram.0";
  dram.kind = "dram";
  dram.visibleFrom = "worker.0";
  dram.capacityBytes = 1u << 30;
  dram.alignmentBytes = 64;
  model.memories = {sram, dram};
  return model;
}

/// The rules every case maps with: one elementwise vector rule, one bounded
/// load rule, and one store rule. None names a workload.
constexpr llvm::StringLiteral kDynamicRules = R"llkmap(
rule dyn.vector_add {
  match micro.vector(op = "add");
  require executor kind worker;
  input "operand0";
  input "operand1";
  output "result";
  bundle "dyn.vector.add";
  emit "e1";
  cost 4;
}

rule dyn.tile_load {
  match micro.tile_load();
  require executor kind worker;
  require memory input "source" kind dram;
  input "source";
  output "result";
  bundle "dyn.tile.load";
  emit "e1";
  cost 2;
}

rule dyn.tile_store {
  match micro.tile_store();
  require executor kind worker;
  input "source";
  bundle "dyn.tile.store";
  emit "e1";
  cost 2;
}
)llkmap";

llvm::Expected<std::unique_ptr<MappingTarget>>
dynamicTarget(MachineModel machine) {
  llvm::Expected<RuleRegistry> rules = parseRuleText(kDynamicRules, "<test>");
  if (!rules)
    return rules.takeError();
  return std::make_unique<FileMappingTarget>(
      "dynamic-shape", std::move(machine), LayoutRegistry{}, std::move(*rules),
      std::vector<std::string>{"e1"});
}

struct ParsedKernel {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::Operation *kernel = nullptr;
};

ParsedKernel parseKernel(llvm::StringRef text) {
  ParsedKernel parsed;
  parsed.context = std::make_unique<mlir::MLIRContext>();
  parsed.context->getOrLoadDialect<mlir::micro::MicroDialect>();
  parsed.context->getOrLoadDialect<mlir::arith::ArithDialect>();
  parsed.context->getOrLoadDialect<mlir::tensor::TensorDialect>();
  parsed.context->getOrLoadDialect<mlir::func::FuncDialect>();
  parsed.module =
      mlir::parseSourceString<mlir::ModuleOp>(text, parsed.context.get());
  if (parsed.module)
    parsed.module->walk([&](mlir::Operation *op) {
      if (op->getName().getStringRef() == "micro.kernel")
        parsed.kernel = op;
    });
  return parsed;
}

std::optional<CoveringPlan> searchOne(const WorkloadGraph &graph,
                                      const MappingTarget &target,
                                      mlir::MLIRContext &context) {
  MappingSearchOptions options;
  options.mode = SearchMode::Deterministic;
  CoveringSearch search(graph, target, context, LayoutContext{}, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  if (!result || result->plans.empty())
    return std::nullopt;
  return result->plans.front();
}

/// One `micro.vector "add"` over an 8x8xf32 tile inside a temporal loop whose
/// upper bound is `UPPER`. The kernel always defines `%n`, so a case can select
/// a small constant, a large constant, or a runtime bound.
constexpr llvm::StringLiteral kLoopedVectorKernel = R"mlir(
module {
  micro.kernel @looped {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c800 = arith.constant 800 : index
    %n = arith.addi %c4, %c1 : index
    %a = tensor.empty() : tensor<8x8xf32>
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<dram>>
    micro.for %i = %c0 to UPPER step %c1 {
      %r = micro.vector "add" %ta, %ta : !micro.tile<8x8xf32, memory = #micro.memory<dram>>, !micro.tile<8x8xf32, memory = #micro.memory<dram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    }
    micro.yield
  }
}
)mlir";

std::string loopedVectorKernel(llvm::StringRef upper) {
  std::string text = kLoopedVectorKernel.str();
  text.replace(text.find("UPPER"), 5, upper.str());
  return text;
}

/// The same vector body inside a `micro.spatial_for` with a computed bound: the
/// per-owner concurrency is a runtime value, so only the selected pool bounds
/// residency.
constexpr llvm::StringLiteral kSpatialVectorKernel = R"mlir(
module {
  micro.kernel @spatial {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %n = arith.addi %c4, %c1 : index
    %a = tensor.empty() : tensor<8x8xf32>
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<dram>>
    micro.spatial_for %i = %c0 to %n step %c1 map = #micro.map<worker> {
      %r = micro.vector "add" %ta, %ta : !micro.tile<8x8xf32, memory = #micro.memory<dram>>, !micro.tile<8x8xf32, memory = #micro.memory<dram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    }
    micro.yield
  }
}
)mlir";

struct CertifiedPlan {
  ParsedKernel parsed;
  WorkloadGraph graph;
  CoveringPlan plan;
};

/// Extracts, searches, and finalizes `text` against `machine`. Returns the
/// finalization error rather than asserting, so a test can inspect it.
llvm::Expected<CertifiedPlan> certify(llvm::StringRef text,
                                      MachineModel machine,
                                      bool *capacityExceeded = nullptr) {
  CertifiedPlan certified;
  certified.parsed = parseKernel(text);
  if (!certified.parsed.module || !certified.parsed.kernel)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "fixture did not parse");
  llvm::Expected<WorkloadGraph> graph =
      extractWorkloadGraph(certified.parsed.kernel);
  if (!graph)
    return graph.takeError();
  certified.graph = std::move(*graph);
  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      dynamicTarget(std::move(machine));
  if (!target)
    return target.takeError();
  std::optional<CoveringPlan> plan =
      searchOne(certified.graph, **target, *certified.parsed.context);
  if (!plan)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "fixture searched no plan");
  certified.plan = std::move(*plan);
  if (llvm::Error error =
          finalizeStoragePlan(certified.graph, certified.plan,
                              (*target)->machine(), capacityExceeded))
    return std::move(error);
  return std::move(certified);
}

/// A comparable, order-independent summary of a finalized plan's local storage:
/// memory, bytes and simultaneous residency of every allocation.
std::vector<std::string> storageShape(const CoveringPlan &plan) {
  std::vector<std::string> entries;
  for (const StorageAllocation &allocation : plan.allocations)
    entries.push_back(allocation.memory + ":" +
                      std::to_string(allocation.bytes) + "x" +
                      std::to_string(allocation.simultaneousOccurrences) +
                      (allocation.borrowed ? ":borrowed" : ":owned"));
  std::sort(entries.begin(), entries.end());
  return entries;
}

/// The owned (compiler-reserved) allocation of `plan` in `memory`, or null.
const StorageAllocation *ownedAllocation(const CoveringPlan &plan,
                                         llvm::StringRef memory) {
  for (const StorageAllocation &allocation : plan.allocations)
    if (!allocation.borrowed && allocation.memory == memory)
      return &allocation;
  return nullptr;
}

/// Every operation named `name` in `parsed`.
std::vector<mlir::Operation *> opsNamed(ParsedKernel &parsed,
                                        llvm::StringRef name) {
  std::vector<mlir::Operation *> ops;
  if (!parsed.module)
    return ops;
  parsed.module->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == name)
      ops.push_back(op);
  });
  return ops;
}

/// The single operation named `name`, or null.
mlir::Operation *oneOp(ParsedKernel &parsed, llvm::StringRef name) {
  std::vector<mlir::Operation *> ops = opsNamed(parsed, name);
  return ops.size() == 1 ? ops.front() : nullptr;
}

} // namespace

//===----------------------------------------------------------------------===//
// Fixed capacity is independent of the declared domain
//===----------------------------------------------------------------------===//

TEST(DynamicShapeMapping, FixedCapacityIndependentOfMNK) {
  // The same static tile capacity, three external domains: a small constant
  // trip count, a large constant one, and a bound only the program knows at
  // runtime. A temporal loop reuses one buffer, so none of them may grow the
  // local footprint.
  llvm::Expected<CertifiedPlan> small =
      certify(loopedVectorKernel("%c8"), dynamicMachine(1, 1u << 20));
  ASSERT_TRUE(bool(small)) << llvm::toString(small.takeError());
  llvm::Expected<CertifiedPlan> large =
      certify(loopedVectorKernel("%c800"), dynamicMachine(1, 1u << 20));
  ASSERT_TRUE(bool(large)) << llvm::toString(large.takeError());
  llvm::Expected<CertifiedPlan> runtime =
      certify(loopedVectorKernel("%n"), dynamicMachine(1, 1u << 20));
  ASSERT_TRUE(bool(runtime)) << llvm::toString(runtime.takeError());

  EXPECT_EQ(storageShape(small->plan), storageShape(large->plan));
  EXPECT_EQ(storageShape(small->plan), storageShape(runtime->plan));

  const StorageAllocation *owned = ownedAllocation(small->plan, "sram.0");
  ASSERT_NE(owned, nullptr);
  EXPECT_EQ(owned->bytes, 256u); // 8x8xf32, one occurrence
  EXPECT_EQ(owned->simultaneousOccurrences, 1u);

  // The runtime domain is carried, not pretended away: the plan says the work
  // count is a runtime value, and the capacity verdict does not depend on it.
  bool runtimeCarried = false;
  for (const std::string &note : runtime->plan.diagnostics.storageNotes)
    runtimeCarried |= note.find("runtime multiplicity") != std::string::npos;
  EXPECT_TRUE(runtimeCarried);
}

//===----------------------------------------------------------------------===//
// Residency is bounded by the selected pool, not the logical grid
//===----------------------------------------------------------------------===//

TEST(DynamicShapeMapping, BoundedResidency) {
  // Four pool slots: at most four occurrences of the vector's 256-byte result
  // can be live together, whatever the spatial extent turns out to be.
  llvm::Expected<CertifiedPlan> fits =
      certify(kSpatialVectorKernel, dynamicMachine(4, 1024));
  ASSERT_TRUE(bool(fits)) << llvm::toString(fits.takeError());
  const StorageAllocation *owned = ownedAllocation(fits->plan, "sram.0");
  ASSERT_NE(owned, nullptr);
  EXPECT_EQ(owned->bytes, 256u);
  EXPECT_EQ(owned->simultaneousOccurrences, 4u)
      << "residency must be the selected pool's concurrency";

  // One byte short of the pool's four occurrences is a capacity rejection, and
  // it is reported as one.
  bool capacityExceeded = false;
  llvm::Expected<CertifiedPlan> tooSmall =
      certify(kSpatialVectorKernel, dynamicMachine(4, 1023), &capacityExceeded);
  ASSERT_FALSE(bool(tooSmall));
  EXPECT_TRUE(capacityExceeded);
  EXPECT_NE(llvm::toString(tooSmall.takeError()).find("capacity"),
            std::string::npos);

  // With no selected executor there is no pool to bound residency: strict
  // planning refuses rather than inventing a bound.
  llvm::Expected<CertifiedPlan> unbounded =
      certify(kSpatialVectorKernel, dynamicMachine(4, 1024));
  ASSERT_TRUE(bool(unbounded)) << llvm::toString(unbounded.takeError());
  ASSERT_FALSE(unbounded->plan.placements.empty());
  unbounded->plan.placements.front().executor.clear();
  llvm::Error noPool = finalizeStoragePlan(unbounded->graph, unbounded->plan,
                                           dynamicMachine(4, 1024));
  ASSERT_TRUE(bool(noPool));
  const std::string message = llvm::toString(std::move(noPool));
  EXPECT_NE(message.find("simultaneous residency"), std::string::npos)
      << message;
}

//===----------------------------------------------------------------------===//
// Boundary requirements are typed and the default emitter rejects them
//===----------------------------------------------------------------------===//

TEST(DynamicShapeMapping, MissingBoundaryEmitter) {
  // A rule-matched operation whose boundary needs a clipped access asks for a
  // typed requirement. The default emitter declares no boundary implementation
  // and must refuse -- declaring an emitter key is not implementing one.
  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      dynamicTarget(dynamicMachine(1, 1u << 20));
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  std::unique_ptr<TargetEmitter> emitter = (*target)->createEmitter("e1");
  ASSERT_NE(emitter, nullptr);
  TargetBundle bundle;
  bundle.name = "dyn.vector.add";
  bundle.emitterKey = "e1";

  BoundaryRequirements none;
  EXPECT_FALSE(bool(emitter->verifyBoundary(bundle, none)));

  BoundaryRequirements partial;
  partial.partialK = true;
  llvm::Error refusal = emitter->verifyBoundary(bundle, partial);
  ASSERT_TRUE(bool(refusal));
  const std::string message = llvm::toString(std::move(refusal));
  EXPECT_NE(message.find("boundary_unsupported"), std::string::npos);
  EXPECT_NE(message.find("partial_k"), std::string::npos);
  EXPECT_EQ(renderBoundaryRequirements(partial), "partial_k");
  EXPECT_EQ(renderBoundaryRequirements(BoundaryRequirements{}), "none");
}

//===----------------------------------------------------------------------===//
// Boundary facts are derived from the operations, not from a workload name
//===----------------------------------------------------------------------===//

TEST(DynamicShapeMapping, ElementwiseBoundary) {
  // A direct elementwise kernel -- no contraction anywhere -- with a bounded
  // load, an elementwise add, and a terminal store, all inside a loop with a
  // runtime bound. The derived requirement is the same typed fact a
  // contraction would produce, and the plan is certified (storage verdict)
  // while the performance analysis refuses to rank it.
  constexpr llvm::StringLiteral kElementwise = R"mlir(
module {
  micro.kernel @elementwise(%src: tensor<8x8xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %n = arith.addi %c4, %c1 : index
    %valid = arith.addi %c8, %c0 : index
    %a = tensor.empty() : tensor<8x8xf32>
    %dstTile = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.for %i = %c0 to %valid step %c1 {
      %t, %tok = micro.tile_load %src[%c0, %c0] valid[%valid, %valid] {dst_memory = #micro.memory<sram>, padding = 0.000000e+00 : f32} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.async_token
      micro.wait %tok
      %r = micro.vector "add" %t, %t : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
      micro.tile_store %r {dst_memory = #micro.memory<dram>} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    }
    micro.yield
  }
}
)mlir";

  ParsedKernel parsed = parseKernel(kElementwise);
  ASSERT_TRUE(parsed.module);
  ASSERT_NE(parsed.kernel, nullptr);
  llvm::Expected<WorkloadGraph> graph = extractWorkloadGraph(parsed.kernel);
  ASSERT_TRUE(bool(graph)) << llvm::toString(graph.takeError());
  // The bounded load is a workload node like any other; the token and the index
  // operands (offset, valid extent, induction variables) are not ports.
  bool sawLoad = false;
  for (const WorkloadNode &node : graph->getNodes())
    sawLoad |= node.opName == "micro.tile_load";
  EXPECT_TRUE(sawLoad);

  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      dynamicTarget(dynamicMachine(1, 1u << 20));
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  std::optional<CoveringPlan> plan =
      searchOne(*graph, **target, *parsed.context);
  ASSERT_TRUE(plan.has_value());
  ASSERT_FALSE(bool(finalizeStoragePlan(*graph, *plan, (*target)->machine())));

  // Storage is certified even though the loop bound is a runtime value; the
  // plan is not rankable because the work count is unknown.
  llvm::Expected<mlir::llk::perf::SelectedKernelAnalysis> strict =
      mlir::llk::perf::analyzeSelectedKernel(
          parsed.kernel, (*target)->machine(),
          /*requireComplete=*/true, *plan, *graph);
  ASSERT_FALSE(bool(strict));
  EXPECT_NE(llvm::toString(strict.takeError()).find("non-static bounds"),
            std::string::npos);

  // Bind the plan and check the materialized IR keeps the boundary operands and
  // the runtime guards rather than widening them into full-tile accesses.
  std::unique_ptr<PlanMaterializer> materializer =
      mlir::llk::micro_mapping_detail::createCanonicalPlanMaterializer();
  llvm::Expected<BoundPlan> bound =
      bindPlan(parsed.module.get(), *plan, **target, BindContract::Executable,
               materializer.get());
  ASSERT_TRUE(bool(bound)) << llvm::toString(bound.takeError());
  ASSERT_TRUE(bound->unmaterialized.empty());

  mlir::Operation *load = nullptr;
  bound->module->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "micro.tile_load")
      load = op;
  });
  ASSERT_NE(load, nullptr);
  EXPECT_TRUE(load->hasAttr("micro.mapping"));
  // The runtime guards (`valid[...]`) and offsets are still operands of the
  // materialized op -- the plan retains them; it does not bake in the tile size.
  auto typedLoad = llvm::cast<mlir::micro::TileLoadOp>(load);
  EXPECT_EQ(typedLoad.getOffsets().size(), 2u);
  EXPECT_EQ(typedLoad.getValidSizes().size(), 2u);
  for (mlir::Value valid : typedLoad.getValidSizes())
    EXPECT_TRUE(mlir::getConstantIntValue(valid) == std::nullopt)
        << "the runtime guard must survive binding";

  // The typed requirement is derived from the elementwise operations alone, and
  // the movement emitter selected by the rule decides whether it can realize it.
  BoundaryRequirements requirements = boundaryRequirementsFor({load});
  EXPECT_TRUE(requirements.boundedLoad);
  EXPECT_FALSE(requirements.boundedStore);
  EXPECT_FALSE(requirements.partialK);
  // The default emitter on this target has no boundary implementation, so the
  // coverage check refuses the plan with the typed reason.
  TargetBundle bundle;
  bundle.name = "dyn.tile.load";
  bundle.emitterKey = "e1";
  EXPECT_TRUE(bool(
      (*target)->createEmitter("e1")->verifyBoundary(bundle, requirements)));
}

//===----------------------------------------------------------------------===//
// Boundary derivation: proven full-tile accesses ask for nothing
//===----------------------------------------------------------------------===//

TEST(DynamicShapeMapping, BoundaryDerivationDistinguishesGuards) {
  // A clipped load/store and a partial reduction are requirements; the same
  // operations with constant full-tile extents are not.
  constexpr llvm::StringLiteral kBounded = R"mlir(
module {
  func.func @bounded(%src: tensor<8x32xf32>, %i: index, %j: index,
                     %vm: index, %vk: index,
                     %s: !micro.tile<8x32xf32, memory = #micro.memory<acc>>,
                     %d: !micro.tile<16x64xf32, memory = #micro.memory<dram>>) {
    %t, %tok = micro.tile_load %src[%i, %j] valid[%vm, %vk] {dst_memory = #micro.memory<sram>, padding = 0.000000e+00 : f32} : tensor<8x32xf32> -> !micro.tile<8x32xf32, memory = #micro.memory<sram>>, !micro.async_token
    %w = micro.tile_store %s into %d[%i, %j] valid[%vm, %vk] {dst_memory = #micro.memory<dram>} : !micro.tile<8x32xf32, memory = #micro.memory<acc>>, !micro.tile<16x64xf32, memory = #micro.memory<dram>> -> !micro.tile<16x64xf32, memory = #micro.memory<dram>>
    return
  }
}
)mlir";
  constexpr llvm::StringLiteral kFull = R"mlir(
module {
  func.func @full(%src: tensor<8x32xf32>, %i: index, %j: index,
                  %s: !micro.tile<8x32xf32, memory = #micro.memory<acc>>,
                  %d: !micro.tile<16x64xf32, memory = #micro.memory<dram>>,
                  %a: !micro.tile<16x32xbf16>, %b: !micro.tile<32x16xbf16>,
                  %c: !micro.tile<16x16xf32>) {
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %t, %tok = micro.tile_load %src[%i, %j] valid[%c8, %c32] {dst_memory = #micro.memory<sram>, padding = 0.000000e+00 : f32} : tensor<8x32xf32> -> !micro.tile<8x32xf32, memory = #micro.memory<sram>>, !micro.async_token
    %w = micro.tile_store %s into %d[%i, %j] valid[%c8, %c32] {dst_memory = #micro.memory<dram>} : !micro.tile<8x32xf32, memory = #micro.memory<acc>>, !micro.tile<16x64xf32, memory = #micro.memory<dram>> -> !micro.tile<16x64xf32, memory = #micro.memory<dram>>
    %r = micro.mma %a, %b, %c valid_k %c32 {shape = array<i64: 16, 16, 32>, input = #micro.dtype<bf16>, accumulator = #micro.dtype<f32>} : !micro.tile<16x32xbf16>, !micro.tile<32x16xbf16>, !micro.tile<16x16xf32> -> !micro.tile<16x16xf32>
    return
  }
}
)mlir";

  ParsedKernel bounded = parseKernel(kBounded);
  ASSERT_TRUE(bounded.module);
  BoundaryRequirements guardRequirements = boundaryRequirementsFor(
      {oneOp(bounded, "micro.tile_load"), oneOp(bounded, "micro.tile_store")});
  EXPECT_TRUE(guardRequirements.boundedLoad);
  EXPECT_TRUE(guardRequirements.boundedStore);
  EXPECT_FALSE(guardRequirements.partialK);

  ParsedKernel full = parseKernel(kFull);
  ASSERT_TRUE(full.module);
  BoundaryRequirements provenRequirements = boundaryRequirementsFor(
      {oneOp(full, "micro.tile_load"), oneOp(full, "micro.tile_store"),
       oneOp(full, "micro.mma")});
  EXPECT_FALSE(provenRequirements.any())
      << "a constant full-tile access needs no boundary implementation";
}

//===----------------------------------------------------------------------===//
// Partial-K capability is declared per emitter
//===----------------------------------------------------------------------===//

TEST(DynamicShapeMapping, PartialKCapability) {
  // The AVX2 movement/contraction emitter implements the boundary behaviours
  // through the shared lowering the selected pipeline runs first.
  llvm::Expected<std::unique_ptr<MappingTarget>> avx2 =
      avx2_mapping::createMappingTarget(LLK_SOURCE_DIR, "");
  ASSERT_TRUE(bool(avx2)) << llvm::toString(avx2.takeError());
  std::unique_ptr<TargetEmitter> mma = (*avx2)->createEmitter("avx2_mma");
  ASSERT_NE(mma, nullptr);
  TargetBundle avx2Bundle;
  avx2Bundle.name = "avx2.mma.f32";
  avx2Bundle.emitterKey = "avx2_mma";
  BoundaryRequirements partial;
  partial.partialK = true;
  EXPECT_FALSE(bool(mma->verifyBoundary(avx2Bundle, partial)))
      << "AVX2 realizes a runtime reduction bound";

  // The capability is per emitter, not per target: the arithmetic key does not
  // carry a contraction, so it cannot claim the partial-K implementation.
  std::unique_ptr<TargetEmitter> add =
      (*avx2)->createEmitter("avx2_vector_add");
  ASSERT_NE(add, nullptr);
  TargetBundle addBundle;
  addBundle.name = "avx2.vector.add.f32";
  addBundle.emitterKey = "avx2_vector_add";
  EXPECT_TRUE(bool(add->verifyBoundary(addBundle, partial)));

  // The generic accelerator declares its limits explicitly: bounded movement is
  // modeled by the reference lowering, a partial-K remainder implementation is
  // not declared at all.
  llvm::Expected<std::unique_ptr<MappingTarget>> accel =
      accel_mapping::createMappingTarget(LLK_SOURCE_DIR, "");
  ASSERT_TRUE(bool(accel)) << llvm::toString(accel.takeError());
  std::unique_ptr<TargetEmitter> mxu = (*accel)->createEmitter("accel_mxu");
  ASSERT_NE(mxu, nullptr);
  TargetBundle accelBundle;
  accelBundle.name = "accel.mma.bf16";
  accelBundle.emitterKey = "accel_mxu";

  BoundaryRequirements movement;
  movement.boundedLoad = true;
  movement.boundedStore = true;
  EXPECT_FALSE(bool(mxu->verifyBoundary(accelBundle, movement)));

  llvm::Error refusal = mxu->verifyBoundary(accelBundle, partial);
  ASSERT_TRUE(bool(refusal));
  const std::string message = llvm::toString(std::move(refusal));
  EXPECT_NE(message.find("partial-k remainder"), std::string::npos) << message;
}

//===----------------------------------------------------------------------===//
// Routing, intermediates and layout padding are real storage
//===----------------------------------------------------------------------===//

TEST(DynamicShapeMapping, RouteAndPaddingStorage) {
  // A two-hop movement reserves an intermediate in each hop's destination
  // memory; the intermediate is one buffer shared by both hops.
  llvm::Expected<issue129::ResourceCase> twoHop =
      issue129::resourceCase("two-hop");
  ASSERT_TRUE(bool(twoHop)) << llvm::toString(twoHop.takeError());
  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  llvm::Expected<MappingSearchResult> result =
      issue129::searchCase(*twoHop, options);
  ASSERT_TRUE(bool(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());
  CoveringPlan plan = result->plans.front();
  ASSERT_FALSE(bool(
      finalizeStoragePlan(twoHop->graph, plan, twoHop->target->machine())));
  ASSERT_FALSE(plan.connectionPlans.empty());
  const PlanConnection &movement = plan.connectionPlans.front();
  ASSERT_EQ(movement.hops.size(), 2u);
  EXPECT_TRUE(llvm::any_of(plan.allocations, [](const StorageAllocation &a) {
    return a.memory == "l2.0" && a.bytes == 256;
  })) << "the intermediate hop's memory must be reserved";
  EXPECT_TRUE(llvm::any_of(plan.allocations, [](const StorageAllocation &a) {
    return a.memory == "dram.0" && a.bytes == 256;
  }));

  // A padded layout's physical image -- not its logical element count -- is
  // what a live range reserves.
  llvm::Expected<issue129::ResourceCase> padded =
      issue129::resourceCase("padded-layout");
  ASSERT_TRUE(bool(padded)) << llvm::toString(padded.takeError());
  llvm::Expected<MappingSearchResult> paddedResult =
      issue129::searchCase(*padded, options);
  ASSERT_TRUE(bool(paddedResult)) << llvm::toString(paddedResult.takeError());
  ASSERT_FALSE(paddedResult->plans.empty());
  CoveringPlan paddedPlan = paddedResult->plans.front();
  ASSERT_FALSE(bool(finalizeStoragePlan(padded->graph, paddedPlan,
                                        padded->target->machine())));
  EXPECT_TRUE(llvm::any_of(paddedPlan.allocations,
                           [](const StorageAllocation &a) {
                             return a.memory == "l2.0" && a.bytes == 480;
                           }))
      << "8x8xf32 under a padded row-major layout is a 480-byte image";
}

//===----------------------------------------------------------------------===//
// Unbounded dynamic local storage is rejected
//===----------------------------------------------------------------------===//

TEST(DynamicShapeMapping, RejectUnknownLocalAllocation) {
  // A locally materialized tile whose extent is a runtime sentinel has no
  // compile-time capacity. Strict planning must refuse it: an unknown footprint
  // is not a smaller one.
  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::micro::MicroDialect>();
  mlir::Type dynamicTile = mlir::parseType(
      "!micro.tile<?x?xf32, memory = #micro.memory<sram>>", &context);
  ASSERT_TRUE(dynamicTile);

  WorkloadGraph graph;
  WorkloadValueId in =
      graph.addValue(WorkloadValue{0, dynamicTile, "in", /*external=*/true});
  WorkloadValueId out =
      graph.addValue(WorkloadValue{0, dynamicTile, "out", /*external=*/false});
  WorkloadNode node;
  node.opName = "micro.vector";
  node.sourceOrdinal = 0;
  node.executionMultiplicity = 1;
  node.attributes = mlir::DictionaryAttr::get(
      &context, {mlir::NamedAttribute(mlir::StringAttr::get(&context, "op"),
                                      mlir::StringAttr::get(&context, "add"))});
  node.inputs.push_back(WorkloadPort{in, dynamicTile, std::nullopt});
  node.outputs.push_back(WorkloadPort{out, dynamicTile, std::nullopt});
  graph.addNode(std::move(node));
  graph.finalize();

  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      dynamicTarget(dynamicMachine(1, 1u << 20));
  ASSERT_TRUE(bool(target)) << llvm::toString(target.takeError());
  std::optional<CoveringPlan> plan = searchOne(graph, **target, context);
  ASSERT_TRUE(plan.has_value());

  llvm::Error error = finalizeStoragePlan(graph, *plan, (*target)->machine());
  ASSERT_TRUE(bool(error));
  const std::string message = llvm::toString(std::move(error));
  EXPECT_NE(message.find("footprint"), std::string::npos) << message;
  EXPECT_NE(message.find("static"), std::string::npos) << message;

  // An explicit analysis artifact may still report the gap, but it never
  // reports it as a verified fit.
  plan->materialized = false;
  ASSERT_FALSE(bool(finalizeStoragePlan(graph, *plan, (*target)->machine())));
  bool reported = false;
  for (const std::string &note : plan->diagnostics.storageNotes)
    reported |= note.find("footprint") != std::string::npos;
  EXPECT_TRUE(reported);
}

//===----------------------------------------------------------------------===//
// Shape predicates are classified, not guessed
//===----------------------------------------------------------------------===//

TEST(DynamicShapeMapping, ShapePredicateVerdict) {
  mlir::llk::shape::ShapeContract contract;
  contract.symbols.emplace(
      "M", mlir::llk::shape::SymbolConstraint{0, std::nullopt, 8});
  contract.symbols.emplace("N", mlir::llk::shape::SymbolConstraint{4, 4, 1});

  using Kind = mlir::llk::shape::ShapePredicate::Kind;
  using Verdict = mlir::llk::shape::PredicateVerdict;

  // Symbolic divisibility the contract already declares is proven; a divisor
  // the congruence does not imply stays a guard until the invocation proves it.
  EXPECT_EQ(mlir::llk::shape::classifyPredicate({Kind::DivisibleBy, "M", 8},
                                                contract),
            Verdict::Proven);
  EXPECT_EQ(mlir::llk::shape::classifyPredicate({Kind::DivisibleBy, "M", 3},
                                                contract),
            Verdict::RuntimeGuard);
  // A single-point domain decides equality either way.
  EXPECT_EQ(
      mlir::llk::shape::classifyPredicate({Kind::EqualTo, "N", 4}, contract),
      Verdict::Proven);
  EXPECT_EQ(
      mlir::llk::shape::classifyPredicate({Kind::EqualTo, "N", 5}, contract),
      Verdict::Disproven);
  EXPECT_EQ(
      mlir::llk::shape::classifyPredicate({Kind::AtLeast, "M", 1}, contract),
      Verdict::RuntimeGuard);
  EXPECT_EQ(
      mlir::llk::shape::classifyPredicate({Kind::AtLeast, "N", 5}, contract),
      Verdict::Disproven);
  EXPECT_EQ(std::string(mlir::llk::shape::stringifyPredicateVerdict(
                Verdict::RuntimeGuard)),
            "runtime_guard");
}
