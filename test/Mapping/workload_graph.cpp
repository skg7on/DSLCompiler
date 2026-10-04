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
