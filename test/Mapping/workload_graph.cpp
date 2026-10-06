//===- workload_graph.cpp - Workload graph extraction (issue #80 / D1) ---===//

#include "LLK/Mapping/WorkloadGraph.h"

#include "LLK/Dialect/Micro/MicroDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include <gtest/gtest.h>

#include <memory>
#include <optional>

using namespace mlir;
using namespace mlir::llk::mapping;

namespace {

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
  parsed.context->getOrLoadDialect<arith::ArithDialect>();
  parsed.module = parseSourceString<ModuleOp>(text, parsed.context.get());
  if (parsed.module)
    parsed.module->walk([&](Operation *op) {
      if (op->getName().getStringRef() == "micro.kernel")
        parsed.kernel = op;
    });
  return parsed;
}

/// A tiled GEMM, copied from the micro-perf end-to-end fixture: two staged
/// copies, a view of each, one MMA fragment, and a store.
constexpr llvm::StringRef kGemm = R"mlir(
module {
  micro.kernel @gemm_tile attributes {workload = "gemm"} {
    %a_ext = tensor.empty() : tensor<16x32xbf16>
    %b_ext = tensor.empty() : tensor<32x16xbf16>
    %a_tile, %a_tok = micro.async_copy %a_ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<16x32xbf16> -> tensor<16x32xbf16>, !micro.async_token
    %b_tile, %b_tok = micro.async_copy %b_ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<32x16xbf16> -> tensor<32x16xbf16>, !micro.async_token
    micro.wait %a_tok, %b_tok
    %a_frag = micro.tile_view %a_tile {shape = array<i64: 16, 32>} : tensor<16x32xbf16> -> !micro.tile<16x32xbf16, memory = #micro.memory<sram>>
    %b_frag = micro.tile_view %b_tile {shape = array<i64: 32, 16>} : tensor<32x16xbf16> -> !micro.tile<32x16xbf16, memory = #micro.memory<sram>>
    %acc = micro.tile_alloc : !micro.tile<16x16xf32, memory = #micro.memory<acc>>
    %result = micro.mma %a_frag, %b_frag, %acc {shape = array<i64: 16, 16, 32>, input = #micro.dtype<bf16>, accumulator = #micro.dtype<f32>} : !micro.tile<16x32xbf16, memory = #micro.memory<sram>>, !micro.tile<32x16xbf16, memory = #micro.memory<sram>>, !micro.tile<16x16xf32, memory = #micro.memory<acc>> -> !micro.tile<16x16xf32, memory = #micro.memory<acc>>
    micro.tile_store %result {dst_memory = #micro.memory<dram>} : !micro.tile<16x16xf32, memory = #micro.memory<acc>>
    micro.yield
  }
}
)mlir";

/// A tile view feeding a vector op: the view is logical, so both vector
/// operands must resolve to the copy that produced the viewed tile.
constexpr llvm::StringRef kVector = R"mlir(
module {
  micro.kernel @vec {
    %ext = tensor.empty() : tensor<8x8xf32>
    %t, %tok = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    %v = micro.tile_view %t {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %v, %v : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)mlir";

const WorkloadNode *findNodeByOp(const WorkloadGraph &graph,
                                 llvm::StringRef op) {
  for (const WorkloadNode &node : graph.getNodes())
    if (node.opName == op)
      return &node;
  return nullptr;
}

/// A view with constant offsets: the consumer's index space is the source's,
/// shifted by (1, 2).
constexpr llvm::StringRef kShiftedView = R"mlir(
module {
  micro.kernel @shift {
    %ext = tensor.empty() : tensor<8x8xf32>
    %t, %tok = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %v = micro.tile_view %t[%c1, %c2] {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %v, %v : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)mlir";

/// A non-constant view offset: no affine map can be extracted from it alone.
constexpr llvm::StringRef kSymbolicView = R"mlir(
module {
  micro.kernel @symbolic {
    %ext = tensor.empty() : tensor<8x8xf32>
    %t, %tok = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    %c3 = arith.constant 3 : index
    %offset = arith.addi %c3, %c3 : index
    %v = micro.tile_view %t[%offset, %offset] {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %v, %v : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)mlir";

/// A partition names a fragment, not an index relationship to its parent.
constexpr llvm::StringRef kPartitioned = R"mlir(
module {
  micro.kernel @partition {
    %t = micro.tile_alloc : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %p = micro.tile_partition %t {shape = array<i64: 4, 4>} : !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<4x4xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %p, %p : !micro.tile<4x4xf32, memory = #micro.memory<sram>>, !micro.tile<4x4xf32, memory = #micro.memory<sram>> -> !micro.tile<4x4xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
)mlir";

} // namespace

TEST(WorkloadGraph, KeepsOnlyExecutionOpsAsNodes) {
  Parsed parsed = parseKernel(kGemm);
  ASSERT_TRUE(parsed.module);
  auto graph = extractWorkloadGraph(parsed.kernel);
  ASSERT_TRUE(static_cast<bool>(graph));
  // Two copies, one mma, one store. wait, tile_view, and tile_alloc are not
  // work to place on the target.
  EXPECT_EQ(graph->getNodes().size(), 4u);
  EXPECT_EQ(findNodeByOp(*graph, "micro.mma")->inputs.size(), 3u);
  EXPECT_EQ(findNodeByOp(*graph, "micro.mma")->outputs.size(), 1u);
  EXPECT_EQ(findNodeByOp(*graph, "micro.tile_store")->outputs.size(), 0u);
}

TEST(WorkloadGraph, TileViewIsTransparent) {
  Parsed parsed = parseKernel(kVector);
  ASSERT_TRUE(parsed.module);
  auto graph = extractWorkloadGraph(parsed.kernel);
  ASSERT_TRUE(static_cast<bool>(graph));

  const WorkloadNode *copy = findNodeByOp(*graph, "micro.async_copy");
  const WorkloadNode *vector = findNodeByOp(*graph, "micro.vector");
  ASSERT_NE(copy, nullptr);
  ASSERT_NE(vector, nullptr);
  ASSERT_EQ(copy->outputs.size(), 1u);
  ASSERT_EQ(vector->inputs.size(), 2u);
  // Both operands resolve through the logical view to the copy's output.
  EXPECT_EQ(vector->inputs[0].value, copy->outputs[0].value);
  EXPECT_EQ(vector->inputs[1].value, copy->outputs[0].value);
}

TEST(WorkloadGraph, WholeSourceViewGivesAnIdentityAccessMap) {
  Parsed parsed = parseKernel(kVector);
  ASSERT_TRUE(parsed.module);
  auto graph = extractWorkloadGraph(parsed.kernel);
  ASSERT_TRUE(static_cast<bool>(graph));

  const WorkloadNode *vector = findNodeByOp(*graph, "micro.vector");
  ASSERT_NE(vector, nullptr);
  ASSERT_EQ(vector->inputs.size(), 2u);
  ASSERT_TRUE(vector->inputs[0].accessMap.has_value());
  // A view of the whole source is the identity relation.
  EXPECT_EQ(*vector->inputs[0].accessMap,
            AffineMap::getMultiDimIdentityMap(2, parsed.context.get()));
  EXPECT_EQ(*vector->inputs[1].accessMap,
            AffineMap::getMultiDimIdentityMap(2, parsed.context.get()));
}

TEST(WorkloadGraph, ConstantOffsetViewGivesAShiftedAccessMap) {
  Parsed parsed = parseKernel(kShiftedView);
  ASSERT_TRUE(parsed.module) << "fixture must parse";
  auto graph = extractWorkloadGraph(parsed.kernel);
  ASSERT_TRUE(static_cast<bool>(graph));

  MLIRContext &context = *parsed.context;
  const WorkloadNode *vector = findNodeByOp(*graph, "micro.vector");
  ASSERT_NE(vector, nullptr);
  ASSERT_EQ(vector->inputs.size(), 2u);
  ASSERT_TRUE(vector->inputs[0].accessMap.has_value());
  AffineMap expected = AffineMap::get(
      2, 0,
      {getAffineDimExpr(0, &context) + getAffineConstantExpr(1, &context),
       getAffineDimExpr(1, &context) + getAffineConstantExpr(2, &context)},
      &context);
  EXPECT_EQ(*vector->inputs[0].accessMap, expected);
}

TEST(WorkloadGraph, NonConstantViewOffsetHasNoAccessMap) {
  Parsed parsed = parseKernel(kSymbolicView);
  ASSERT_TRUE(parsed.module) << "fixture must parse";
  auto graph = extractWorkloadGraph(parsed.kernel);
  ASSERT_TRUE(static_cast<bool>(graph));

  const WorkloadNode *vector = findNodeByOp(*graph, "micro.vector");
  ASSERT_NE(vector, nullptr);
  ASSERT_FALSE(vector->inputs.empty());
  // The offset is not a constant this layer can turn into an affine map, so
  // none is invented.
  EXPECT_FALSE(vector->inputs[0].accessMap.has_value());
}

TEST(WorkloadGraph, PartitionHasNoAccessMap) {
  Parsed parsed = parseKernel(kPartitioned);
  ASSERT_TRUE(parsed.module) << "fixture must parse";
  auto graph = extractWorkloadGraph(parsed.kernel);
  ASSERT_TRUE(static_cast<bool>(graph));

  const WorkloadNode *vector = findNodeByOp(*graph, "micro.vector");
  ASSERT_NE(vector, nullptr);
  ASSERT_FALSE(vector->inputs.empty());
  // A fragment's position within its parent is not an index relation the op
  // alone states, so no map is invented.
  EXPECT_FALSE(vector->inputs[0].accessMap.has_value());
}

TEST(WorkloadGraph, ExternalInputsAreMarked) {
  Parsed parsed = parseKernel(kVector);
  ASSERT_TRUE(parsed.module);
  auto graph = extractWorkloadGraph(parsed.kernel);
  ASSERT_TRUE(static_cast<bool>(graph));
  const WorkloadNode *copy = findNodeByOp(*graph, "micro.async_copy");
  ASSERT_EQ(copy->inputs.size(), 1u);
  const WorkloadValue *value = graph->findValue(copy->inputs[0].value);
  ASSERT_NE(value, nullptr);
  EXPECT_TRUE(value->external);
}

TEST(WorkloadGraph, IdenticalKernelsProduceIdenticalCanonicalStrings) {
  Parsed first = parseKernel(kGemm);
  Parsed second = parseKernel(kGemm);
  auto a = extractWorkloadGraph(first.kernel);
  auto b = extractWorkloadGraph(second.kernel);
  ASSERT_TRUE(static_cast<bool>(a));
  ASSERT_TRUE(static_cast<bool>(b));
  EXPECT_EQ(a->canonicalString(), b->canonicalString());
}

TEST(WorkloadGraph, CanonicalIdsIgnoreInsertionOrder) {
  MLIRContext context;
  context.getOrLoadDialect<micro::MicroDialect>();
  Type f32 = Float32Type::get(&context);

  auto build = [&](bool reverse) -> WorkloadGraph {
    WorkloadGraph graph;
    auto addNode = [&](llvm::StringRef op, llvm::StringRef operandName) {
      WorkloadValueId input = graph.addValue(
          WorkloadValue{0, f32, operandName.str(), /*external=*/true});
      WorkloadValueId output =
          graph.addValue(WorkloadValue{0, f32, (op + ".out").str(), false});
      WorkloadNode node;
      node.opName = op.str();
      node.inputs.push_back(WorkloadPort{input, f32, std::nullopt});
      node.outputs.push_back(WorkloadPort{output, f32, std::nullopt});
      graph.addNode(std::move(node));
      return output;
    };
    if (!reverse) {
      addNode("micro.async_copy", "a");
      addNode("micro.vector", "b");
    } else {
      addNode("micro.vector", "b");
      addNode("micro.async_copy", "a");
    }
    graph.finalize();
    return graph;
  };

  WorkloadGraph forward = build(/*reverse=*/false);
  WorkloadGraph backward = build(/*reverse=*/true);
  EXPECT_EQ(forward.canonicalString(), backward.canonicalString());
  EXPECT_EQ(findNodeByOp(forward, "micro.vector")->id,
            findNodeByOp(backward, "micro.vector")->id);
  EXPECT_EQ(findNodeByOp(forward, "micro.async_copy")->id,
            findNodeByOp(backward, "micro.async_copy")->id);
}

TEST(WorkloadGraph, EndpointIdentityDistinguishesRepeatedUses) {
  PortRef lhs{7, PortDirection::Input, 0};
  PortRef rhs{7, PortDirection::Input, 1};
  EXPECT_NE(canonicalPortRefString(lhs), canonicalPortRefString(rhs));
  EXPECT_EQ(canonicalPortRefString(lhs), "node=7,input=0");
}

TEST(WorkloadGraph, EndpointLookupDistinguishesRepeatedUses) {
  Parsed parsed = parseKernel(kVector);
  ASSERT_TRUE(parsed.module);
  auto graph = extractWorkloadGraph(parsed.kernel);
  ASSERT_TRUE(static_cast<bool>(graph));

  const WorkloadNode *vector = findNodeByOp(*graph, "micro.vector");
  ASSERT_NE(vector, nullptr);
  ASSERT_EQ(vector->inputs.size(), 2u);

  // Both operand uses carry the same SSA value, but they are distinct
  // occurrences: the endpoint, not the value id, tells them apart.
  const WorkloadPort *first =
      lookupPort(*graph, PortRef{vector->id, PortDirection::Input, 0});
  const WorkloadPort *second =
      lookupPort(*graph, PortRef{vector->id, PortDirection::Input, 1});
  ASSERT_NE(first, nullptr);
  ASSERT_NE(second, nullptr);
  EXPECT_NE(first, second);
  EXPECT_EQ(first->value, second->value);
  EXPECT_NE(canonicalPortRefString({vector->id, PortDirection::Input, 0}),
            canonicalPortRefString({vector->id, PortDirection::Input, 1}));
}

TEST(WorkloadGraph, EndpointLookupRejectsInvalidReferences) {
  Parsed parsed = parseKernel(kVector);
  ASSERT_TRUE(parsed.module);
  auto graph = extractWorkloadGraph(parsed.kernel);
  ASSERT_TRUE(static_cast<bool>(graph));

  const WorkloadNode *vector = findNodeByOp(*graph, "micro.vector");
  ASSERT_NE(vector, nullptr);
  ASSERT_EQ(vector->inputs.size(), 2u);
  ASSERT_EQ(vector->outputs.size(), 1u);

  // An unknown node, an input index past the node's ports, an output index past
  // the node's ports, and a direction that is not one of the two ranks all
  // resolve to nothing rather than inventing a port.
  EXPECT_EQ(lookupPort(*graph, PortRef{9999, PortDirection::Input, 0}),
            nullptr);
  EXPECT_EQ(lookupPort(*graph, PortRef{vector->id, PortDirection::Input, 2}),
            nullptr);
  EXPECT_EQ(lookupPort(*graph, PortRef{vector->id, PortDirection::Output, 1}),
            nullptr);
  EXPECT_EQ(
      lookupPort(*graph, PortRef{vector->id, static_cast<PortDirection>(7), 0}),
      nullptr);
}

TEST(WorkloadGraph, EndpointLookupResolvesExternalSource) {
  Parsed parsed = parseKernel(kVector);
  ASSERT_TRUE(parsed.module);
  auto graph = extractWorkloadGraph(parsed.kernel);
  ASSERT_TRUE(static_cast<bool>(graph));

  const WorkloadNode *copy = findNodeByOp(*graph, "micro.async_copy");
  ASSERT_NE(copy, nullptr);
  ASSERT_EQ(copy->inputs.size(), 1u);

  // The copy reads the external `tensor.empty`; its endpoint resolves, and the
  // value it carries stays marked external.
  const WorkloadPort *port =
      lookupPort(*graph, PortRef{copy->id, PortDirection::Input, 0});
  ASSERT_NE(port, nullptr);
  const WorkloadValue *value = graph->findValue(port->value);
  ASSERT_NE(value, nullptr);
  EXPECT_TRUE(value->external);
}

TEST(WorkloadGraph, ReversedInsertionKeepsEndpointValueRelationships) {
  MLIRContext context;
  context.getOrLoadDialect<micro::MicroDialect>();
  Type f32 = Float32Type::get(&context);

  // One producer whose result feeds both operand ports of one consumer. The
  // node ids differ before finalization, so endpoints must be resolved after.
  auto build = [&](bool reverse) -> WorkloadGraph {
    WorkloadGraph graph;
    WorkloadValueId source =
        graph.addValue(WorkloadValue{0, f32, "src", /*external=*/true});
    WorkloadValueId tile =
        graph.addValue(WorkloadValue{0, f32, "tile", /*external=*/false});
    WorkloadValueId result =
        graph.addValue(WorkloadValue{0, f32, "result", /*external=*/false});

    WorkloadNode copy;
    copy.opName = "micro.async_copy";
    copy.inputs.push_back(WorkloadPort{source, f32, std::nullopt});
    copy.outputs.push_back(WorkloadPort{tile, f32, std::nullopt});

    WorkloadNode vector;
    vector.opName = "micro.vector";
    vector.inputs.push_back(WorkloadPort{tile, f32, std::nullopt});
    vector.inputs.push_back(WorkloadPort{tile, f32, std::nullopt});
    vector.outputs.push_back(WorkloadPort{result, f32, std::nullopt});

    if (!reverse) {
      graph.addNode(std::move(copy));
      graph.addNode(std::move(vector));
    } else {
      graph.addNode(std::move(vector));
      graph.addNode(std::move(copy));
    }
    graph.finalize();
    return graph;
  };

  WorkloadGraph forward = build(/*reverse=*/false);
  WorkloadGraph backward = build(/*reverse=*/true);
  EXPECT_EQ(forward.canonicalString(), backward.canonicalString());

  const WorkloadNode *forwardProducer =
      findNodeByOp(forward, "micro.async_copy");
  const WorkloadNode *forwardConsumer = findNodeByOp(forward, "micro.vector");
  const WorkloadNode *backwardProducer =
      findNodeByOp(backward, "micro.async_copy");
  const WorkloadNode *backwardConsumer = findNodeByOp(backward, "micro.vector");
  ASSERT_NE(forwardProducer, nullptr);
  ASSERT_NE(forwardConsumer, nullptr);
  ASSERT_NE(backwardProducer, nullptr);
  ASSERT_NE(backwardConsumer, nullptr);

  // The producer's output endpoint and both consumer input endpoints agree on
  // the value they carry, and that relationship is identical after finalizing
  // either insertion order.
  auto resolve = [&](const WorkloadGraph &graph, WorkloadNodeId node,
                     PortDirection direction, uint32_t index) {
    const WorkloadPort *port =
        lookupPort(graph, PortRef{node, direction, index});
    EXPECT_NE(port, nullptr);
    return port ? port->value : WorkloadValueId(-1);
  };

  WorkloadValueId forwardProducerValue =
      resolve(forward, forwardProducer->id, PortDirection::Output, 0);
  for (uint32_t index = 0; index < 2; ++index)
    EXPECT_EQ(
        resolve(forward, forwardConsumer->id, PortDirection::Input, index),
        forwardProducerValue);
  EXPECT_EQ(resolve(backward, backwardProducer->id, PortDirection::Output, 0),
            forwardProducerValue);
  for (uint32_t index = 0; index < 2; ++index)
    EXPECT_EQ(
        resolve(backward, backwardConsumer->id, PortDirection::Input, index),
        forwardProducerValue);
}
