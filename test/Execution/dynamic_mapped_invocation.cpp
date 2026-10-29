//===- dynamic_mapped_invocation.cpp - Checked dynamic mapped ABI --------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dynamic Shape Task 5 (issue #196). ABI v2 records the canonical symbolic
// contract and the explicit argument/result port bindings, so a caller-owned
// output destination is an argument rather than a returned aggregate. These
// tests check the two halves that make that safe:
//
//   * descriptor resolution binds every declared symbol before entry, rejects
//     negative or overflowing extents before any unsigned conversion, accepts a
//     genuinely empty invocation, and still refuses a contradiction that hides
//     behind a zero extent; and
//   * lowering a bound kernel turns its output destination into a caller
//     memref argument, verifies that the result is that destination, and never
//     materialises an output-sized compiler buffer.
//
// Nothing here runs the JIT: the prepared module is the artifact under test.
// Runtime allocation and ownership behavior is covered by the JIT lifetime
// tests. Version 1 keeps its bytes, validation and hash; the last case pins the
// exact v1 hash so a later change cannot silently rewrite older artifacts.

#include "LLK/Conversion/LLKToMicro/LLKToMicro.h"
#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Conversion/MappedKernelAbi.h"
#include "LLK/Conversion/Workload/WorkloadContract.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/WorkloadGraph.h"
#include "LLK/Runtime/KernelAbi.h"
#include "LLK/Runtime/MappedInvocation.h"
#include "LLK/Runtime/PreparedMappedKernel.h"
#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

namespace shape = ::mlir::llk::shape;

/// Fails unless `error` carries `needle`.
void expectError(llvm::Error error, llvm::StringRef needle) {
  ASSERT_TRUE(static_cast<bool>(error));
  const std::string message = llvm::toString(std::move(error));
  EXPECT_NE(message.find(needle.str()), std::string::npos) << message;
}

template <typename T>
void expectError(llvm::Expected<T> value, llvm::StringRef needle) {
  ASSERT_FALSE(static_cast<bool>(value));
  expectError(value.takeError(), needle);
}

llk::InvocationBuffer2D buffer(void *data, int64_t rows, int64_t columns,
                               llk::InvocationElementType type,
                               uint64_t allocationBytes) {
  return {{data, data, 0, rows, columns, columns, 1}, type, allocationBytes};
}

/// M x N x K matmul ABI v2: lhs (M x K), rhs (K x N), read initializer
/// (M x N), caller-owned destination (M x N) bound to the output port.
llk::KernelAbi symbolicMatmulAbi() {
  using Role = llk::AbiPortRole;
  const shape::Dimension m = shape::Dimension::symbol("M");
  const shape::Dimension n = shape::Dimension::symbol("N");
  const shape::Dimension k = shape::Dimension::symbol("K");
  llk::KernelAbi abi;
  abi.version = llk::kMappedAbiVersionV2;
  abi.inputs = {{{}, "f32", {m, k}}, {{}, "f32", {k, n}}, {{}, "f32", {m, n}}};
  abi.outputs = {{{}, "f32", {m, n}}};
  abi.arguments = {
      {Role::Input, 0}, {Role::Input, 1}, {Role::Input, 2}, {Role::Output, 0}};
  abi.results = {0};
  abi.symbols = {{"M", {}}, {"N", {}}, {"K", {}}};
  return abi;
}

struct MatmulBuffers {
  std::vector<float> lhs;
  std::vector<float> rhs;
  std::vector<float> init;
  std::vector<float> destination;
  llk::InvocationBuffer2D lhsBuffer;
  llk::InvocationBuffer2D rhsBuffer;
  llk::InvocationBuffer2D initBuffer;
  llk::InvocationBuffer2D destinationBuffer;
};

MatmulBuffers makeMatmulBuffers(int64_t m, int64_t n, int64_t k) {
  MatmulBuffers buffers;
  buffers.lhs.resize(static_cast<size_t>(m * k));
  buffers.rhs.resize(static_cast<size_t>(k * n));
  buffers.init.resize(static_cast<size_t>(m * n));
  buffers.destination.resize(static_cast<size_t>(m * n));
  buffers.lhsBuffer =
      buffer(buffers.lhs.data(), m, k, llk::InvocationElementType::F32,
             buffers.lhs.size() * sizeof(float));
  buffers.rhsBuffer =
      buffer(buffers.rhs.data(), k, n, llk::InvocationElementType::F32,
             buffers.rhs.size() * sizeof(float));
  buffers.initBuffer =
      buffer(buffers.init.data(), m, n, llk::InvocationElementType::F32,
             buffers.init.size() * sizeof(float));
  buffers.destinationBuffer =
      buffer(buffers.destination.data(), m, n, llk::InvocationElementType::F32,
             buffers.destination.size() * sizeof(float));
  return buffers;
}

TEST(DynamicMappedInvocation, BindsEverySymbolFromConcreteDescriptors) {
  const llk::KernelAbi abi = symbolicMatmulAbi();

  MatmulBuffers first = makeMatmulBuffers(7, 5, 13);
  llvm::Expected<shape::ShapeValues> values = llk::resolveMappedShapeValues(
      abi, {first.lhsBuffer, first.rhsBuffer, first.initBuffer},
      {first.destinationBuffer});
  ASSERT_TRUE(static_cast<bool>(values)) << llvm::toString(values.takeError());
  EXPECT_EQ(*values, (shape::ShapeValues{{"M", 7}, {"N", 5}, {"K", 13}}));

  // The same compiled ABI serves a second, unrelated invocation shape: the
  // symbols are bound per call, never frozen into the artifact.
  MatmulBuffers second = makeMatmulBuffers(9, 33, 35);
  values = llk::resolveMappedShapeValues(
      abi, {second.lhsBuffer, second.rhsBuffer, second.initBuffer},
      {second.destinationBuffer});
  ASSERT_TRUE(static_cast<bool>(values)) << llvm::toString(values.takeError());
  EXPECT_EQ(*values, (shape::ShapeValues{{"M", 9}, {"N", 33}, {"K", 35}}));

  EXPECT_FALSE(static_cast<bool>(llk::validateMappedInvocation(
      abi, {second.lhsBuffer, second.rhsBuffer, second.initBuffer},
      {second.destinationBuffer})));
}

TEST(DynamicMappedInvocation, StaticMismatch) {
  llk::KernelAbi abi;
  abi.version = llk::kMappedAbiVersionV2;
  abi.inputs = {{{}, "f32", {shape::Dimension(8), shape::Dimension(8)}}};
  abi.outputs = {{{}, "f32", {shape::Dimension(8), shape::Dimension(8)}}};
  abi.arguments = {{llk::AbiPortRole::Input, 0}, {llk::AbiPortRole::Output, 0}};
  abi.results = {0};

  std::vector<float> input(64), output(64);
  llk::InvocationBuffer2D in =
      buffer(input.data(), 8, 8, llk::InvocationElementType::F32, 256);
  llk::InvocationBuffer2D out =
      buffer(output.data(), 8, 8, llk::InvocationElementType::F32, 256);
  EXPECT_FALSE(
      static_cast<bool>(llk::validateMappedInvocation(abi, {in}, {out})));

  llk::InvocationBuffer2D wrong = in;
  wrong.descriptor.size0 = 7;
  wrong.descriptor.stride0 = 8;
  expectError(llk::validateMappedInvocation(abi, {wrong}, {out}), "extent");
  wrong = out;
  wrong.descriptor.size1 = 4;
  wrong.descriptor.stride0 = 4;
  expectError(llk::validateMappedInvocation(abi, {in}, {wrong}), "extent");
}

TEST(DynamicMappedInvocation, SharedSymbolMismatch) {
  const llk::KernelAbi abi = symbolicMatmulAbi();
  MatmulBuffers buffers = makeMatmulBuffers(7, 5, 13);

  // lhs declares K = 13; a rhs with 12 rows contradicts the shared symbol.
  std::vector<float> rhs12(12 * 5);
  llk::InvocationBuffer2D rhs =
      buffer(rhs12.data(), 12, 5, llk::InvocationElementType::F32,
             rhs12.size() * sizeof(float));
  expectError(llk::resolveMappedShapeValues(
                  abi, {buffers.lhsBuffer, rhs, buffers.initBuffer},
                  {buffers.destinationBuffer}),
              "conflict");

  // N is shared by rhs, the initializer and the destination.
  std::vector<float> init74(7 * 4);
  llk::InvocationBuffer2D init =
      buffer(init74.data(), 7, 4, llk::InvocationElementType::F32,
             init74.size() * sizeof(float));
  expectError(llk::resolveMappedShapeValues(
                  abi, {buffers.lhsBuffer, buffers.rhsBuffer, init},
                  {buffers.destinationBuffer}),
              "conflict");
}

TEST(DynamicMappedInvocation, EmptyValid) {
  const llk::KernelAbi abi = symbolicMatmulAbi();
  // M = 0: the lhs and every M x N buffer touches no byte, so a null pointer
  // with zero bytes is the honest descriptor.
  auto empty = buffer(nullptr, 0, 13, llk::InvocationElementType::F32, 0);
  auto emptySquare = buffer(nullptr, 0, 5, llk::InvocationElementType::F32, 0);
  std::vector<float> rhs13x5(13 * 5);
  llk::InvocationBuffer2D rhs =
      buffer(rhs13x5.data(), 13, 5, llk::InvocationElementType::F32,
             rhs13x5.size() * sizeof(float));

  llvm::Expected<shape::ShapeValues> values = llk::resolveMappedShapeValues(
      abi, {empty, rhs, emptySquare}, {emptySquare});
  ASSERT_TRUE(static_cast<bool>(values)) << llvm::toString(values.takeError());
  EXPECT_EQ(*values, (shape::ShapeValues{{"M", 0}, {"N", 5}, {"K", 13}}));
}

TEST(DynamicMappedInvocation, EmptyStillRejectsMismatch) {
  const llk::KernelAbi abi = symbolicMatmulAbi();
  auto emptyLhs = buffer(nullptr, 0, 13, llk::InvocationElementType::F32, 0);
  auto emptySquare = buffer(nullptr, 0, 5, llk::InvocationElementType::F32, 0);
  std::vector<float> rhs12x5(12 * 5);
  llk::InvocationBuffer2D rhs =
      buffer(rhs12x5.data(), 12, 5, llk::InvocationElementType::F32,
             rhs12x5.size() * sizeof(float));

  // An empty lhs still declares K = 13, so the contradiction with rhs is
  // checked even though no element is touched.
  expectError(llk::resolveMappedShapeValues(abi, {emptyLhs, rhs, emptySquare},
                                            {emptySquare}),
              "conflict");

  // A zero-sized destination whose N disagrees with the initializer is a
  // contradiction, not an empty invocation.
  std::vector<float> rhs13x5(13 * 5);
  llk::InvocationBuffer2D goodRhs =
      buffer(rhs13x5.data(), 13, 5, llk::InvocationElementType::F32,
             rhs13x5.size() * sizeof(float));
  auto emptySquare6 = buffer(nullptr, 0, 6, llk::InvocationElementType::F32, 0);
  expectError(llk::resolveMappedShapeValues(
                  abi, {emptyLhs, goodRhs, emptySquare}, {emptySquare6}),
              "conflict");
}

TEST(DynamicMappedInvocation, NegativeSize) {
  const llk::KernelAbi abi = symbolicMatmulAbi();
  MatmulBuffers buffers = makeMatmulBuffers(7, 5, 13);

  llk::InvocationBuffer2D negative = buffers.lhsBuffer;
  negative.descriptor.size0 = -7;
  expectError(llk::resolveMappedShapeValues(
                  abi, {negative, buffers.rhsBuffer, buffers.initBuffer},
                  {buffers.destinationBuffer}),
              "negative");

  negative = buffers.destinationBuffer;
  negative.descriptor.size1 = -1;
  expectError(llk::resolveMappedShapeValues(
                  abi,
                  {buffers.lhsBuffer, buffers.rhsBuffer, buffers.initBuffer},
                  {negative}),
              "negative");
}

TEST(DynamicMappedInvocation, SizeOverflow) {
  const llk::KernelAbi abi = symbolicMatmulAbi();
  MatmulBuffers buffers = makeMatmulBuffers(7, 5, 13);

  llk::InvocationBuffer2D huge = buffers.rhsBuffer;
  huge.descriptor.size0 = std::numeric_limits<int64_t>::max();
  huge.descriptor.size1 = 2;
  huge.descriptor.stride0 = 2;
  huge.descriptor.stride1 = 1;
  huge.allocationBytes = 0;
  expectError(llk::resolveMappedShapeValues(
                  abi, {buffers.lhsBuffer, huge, buffers.initBuffer},
                  {buffers.destinationBuffer}),
              "overflow");
}

TEST(DynamicMappedInvocation, InsufficientAllocation) {
  const llk::KernelAbi abi = symbolicMatmulAbi();
  MatmulBuffers buffers = makeMatmulBuffers(7, 5, 13);

  llk::InvocationBuffer2D shortBuffer = buffers.rhsBuffer;
  shortBuffer.allocationBytes = buffers.rhs.size() * sizeof(float) - 1;
  expectError(llk::resolveMappedShapeValues(
                  abi, {buffers.lhsBuffer, shortBuffer, buffers.initBuffer},
                  {buffers.destinationBuffer}),
              "allocation");

  // The aligned pointer must stay inside the caller's allocation.
  llk::InvocationBuffer2D shifted = buffers.initBuffer;
  shifted.descriptor.aligned = buffers.init.data() + 4;
  shifted.allocationBytes = buffers.init.size() * sizeof(float);
  expectError(llk::resolveMappedShapeValues(
                  abi, {buffers.lhsBuffer, buffers.rhsBuffer, shifted},
                  {buffers.destinationBuffer}),
              "allocation");
}

TEST(DynamicMappedInvocation, DtypeStrideAlias) {
  const llk::KernelAbi abi = symbolicMatmulAbi();
  MatmulBuffers buffers = makeMatmulBuffers(7, 5, 13);

  llk::InvocationBuffer2D wrongDtype = buffers.lhsBuffer;
  wrongDtype.elementType = llk::InvocationElementType::BF16;
  expectError(llk::resolveMappedShapeValues(
                  abi, {wrongDtype, buffers.rhsBuffer, buffers.initBuffer},
                  {buffers.destinationBuffer}),
              "dtype");

  llk::InvocationBuffer2D strided = buffers.lhsBuffer;
  strided.descriptor.stride0 = 14;
  expectError(llk::resolveMappedShapeValues(
                  abi, {strided, buffers.rhsBuffer, buffers.initBuffer},
                  {buffers.destinationBuffer}),
              "compact");

  llk::InvocationBuffer2D offset = buffers.initBuffer;
  offset.descriptor.offset = 1;
  expectError(llk::resolveMappedShapeValues(
                  abi, {buffers.lhsBuffer, buffers.rhsBuffer, offset},
                  {buffers.destinationBuffer}),
              "offset");

  // The caller-owned destination must not alias a read-only input.
  llk::InvocationBuffer2D aliased = buffers.destinationBuffer;
  aliased.descriptor.allocated = buffers.lhs.data();
  aliased.descriptor.aligned = buffers.lhs.data();
  aliased.allocationBytes = buffers.lhs.size() * sizeof(float);
  expectError(llk::resolveMappedShapeValues(
                  abi,
                  {buffers.lhsBuffer, buffers.rhsBuffer, buffers.initBuffer},
                  {aliased}),
              "overlap");
}

TEST(DynamicMappedInvocation, Version1HashStable) {
  llk::KernelAbi abi;
  abi.inputs.push_back({{2, 2}, "f32"});
  abi.outputs.push_back({{2, 2}, "f32"});
  // The exact FNV-1a value version 1 has always produced for this contract.
  EXPECT_EQ(llk::computeKernelAbiHash(abi), 1484757976495943717ULL);

  // The same ports under version 2 are a different artifact with a different
  // identity, not a reinterpretation of the version 1 bytes.
  llk::KernelAbi v2 = symbolicMatmulAbi();
  EXPECT_NE(llk::computeKernelAbiHash(v2), llk::computeKernelAbiHash(abi));
  EXPECT_EQ(llk::computeKernelAbiHash(v2), llk::computeKernelAbiHash(v2));
  llk::KernelAbi changed = v2;
  changed.symbols["K"].multipleOf = 4;
  EXPECT_NE(llk::computeKernelAbiHash(v2), llk::computeKernelAbiHash(changed));

  // Version 1 declares no symbols, so resolution keeps its descriptor checks
  // and reports an empty binding rather than inventing symbol names.
  std::vector<float> input(4), output(4);
  llvm::Expected<shape::ShapeValues> resolved = llk::resolveMappedShapeValues(
      abi, {buffer(input.data(), 2, 2, llk::InvocationElementType::F32, 16)},
      {buffer(output.data(), 2, 2, llk::InvocationElementType::F32, 16)});
  ASSERT_TRUE(static_cast<bool>(resolved))
      << llvm::toString(resolved.takeError());
  EXPECT_TRUE(resolved->empty());
}

TEST(DynamicMappedInvocation, AbiEnvelope) {
  // Rank two remains the execution boundary in both versions.
  llk::KernelAbi rank3;
  rank3.version = llk::kMappedAbiVersionV2;
  rank3.inputs = {
      {{},
       "f32",
       {shape::Dimension(2), shape::Dimension(2), shape::Dimension(2)}}};
  rank3.outputs = {{{}, "f32", {shape::Dimension(2), shape::Dimension(2)}}};
  rank3.arguments = {{llk::AbiPortRole::Input, 0},
                     {llk::AbiPortRole::Output, 0}};
  rank3.results = {0};
  expectError(llk::validateKernelAbi(rank3), "rank two");

  // Twelve descriptors is still the ceiling.
  llk::KernelAbi thirteen;
  thirteen.version = llk::kMappedAbiVersionV2;
  for (unsigned i = 0; i < 13; ++i)
    thirteen.inputs.push_back(
        {{}, "f32", {shape::Dimension(2), shape::Dimension(2)}});
  expectError(llk::validateKernelAbi(thirteen), "12 descriptors");

  // A symbol used by a port must be declared, and every output port is bound
  // by a caller-owned argument.
  llk::KernelAbi undeclared = symbolicMatmulAbi();
  undeclared.symbols.erase("K");
  expectError(llk::validateKernelAbi(undeclared), "undeclared");

  llk::KernelAbi unbound = symbolicMatmulAbi();
  unbound.arguments.pop_back();
  unbound.results.clear();
  expectError(llk::validateKernelAbi(unbound), "output port");

  llk::KernelAbi unknown = symbolicMatmulAbi();
  unknown.version = 3;
  expectError(llk::validateKernelAbi(unknown),
              "unsupported mapped ABI version");
}

//===----------------------------------------------------------------------===//
// Output destination lowering
//===----------------------------------------------------------------------===//

/// A fully dynamic copy whose output is an explicit caller-owned destination.
/// The destination is bound to the output port, so the kernel writes through
/// it instead of returning an aggregate the runtime would have to copy.
constexpr llvm::StringLiteral kDynamicCopyKernel = R"mlir(
module {
  micro.kernel @copy_dynamic(%src: tensor<?x?xf32>, %dst: tensor<?x?xf32>)
      -> tensor<?x?xf32>
      attributes {
        argument_bindings = [{port = 0 : i64, role = "input"},
                             {port = 1 : i64, role = "output"}],
        result_bindings = [1]
      } {
    %c0 = arith.constant 0 : index
    %m = tensor.dim %src, %c0 : tensor<?x?xf32>
    %c1 = arith.constant 1 : index
    %n = tensor.dim %src, %c1 : tensor<?x?xf32>
    %sv = micro.tile_view %src {shape = array<i64: -9223372036854775808, -9223372036854775808>}
        : tensor<?x?xf32>
       -> !micro.tile<?x?xf32, memory = #micro.memory<sram>>
    %dv = micro.tile_view %dst {shape = array<i64: -9223372036854775808, -9223372036854775808>}
        : tensor<?x?xf32>
       -> !micro.tile<?x?xf32, memory = #micro.memory<dram>>
    %r = micro.tile_store %sv into %dv[%c0, %c0] valid[%m, %n]
        {dst_memory = #micro.memory<dram>}
        : !micro.tile<?x?xf32, memory = #micro.memory<sram>>,
          !micro.tile<?x?xf32, memory = #micro.memory<dram>>
       -> !micro.tile<?x?xf32, memory = #micro.memory<dram>>
    micro.yield %r : !micro.tile<?x?xf32, memory = #micro.memory<dram>>
  }
}
)mlir";

mlir::DialectRegistry buildRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::scf::SCFDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                  mlir::cf::ControlFlowDialect>();
  return registry;
}

/// Attaches the canonical symbolic contract the kernel implements.
void attachContract(mlir::Operation *kernel, mlir::MLIRContext &context,
                    mlir::llk::workload::WorkloadContract contract) {
  llvm::Expected<mlir::DictionaryAttr> encoded =
      mlir::llk::workload::encodeWorkloadContract(&context, contract);
  ASSERT_TRUE(static_cast<bool>(encoded))
      << llvm::toString(encoded.takeError());
  kernel->setAttr("micro.workload_contract", *encoded);
}

/// The dynamic copy's contract: one read input and one caller-owned output,
/// both M x N.
void attachCopyContract(mlir::ModuleOp module, mlir::MLIRContext &context) {
  using mlir::llk::workload::PortDirection;
  using mlir::llk::workload::PortEffect;
  namespace workload = mlir::llk::workload;
  const shape::Dimension m = shape::Dimension::symbol("M");
  const shape::Dimension n = shape::Dimension::symbol("N");
  workload::WorkloadContract contract;
  contract.version = 2;
  contract.ports = {
      {"src", PortDirection::Input, PortEffect::Read, {m, n}, "f32"},
      {"dst", PortDirection::Output, PortEffect::Write, {m, n}, "f32"},
  };
  contract.symbols = {{"M", {}}, {"N", {}}};
  module.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "micro.kernel")
      attachContract(op, context, contract);
  });
}

/// The contraction's contract, in the exporter's port order: lhs, rhs, read
/// initializer, caller-owned output. Every port reuses M/N/K, so the ABI binds
/// the same symbol across ports and refuses descriptors that disagree.
void attachContractionContract(mlir::Operation *kernel,
                               mlir::MLIRContext &context) {
  using mlir::llk::workload::PortDirection;
  using mlir::llk::workload::PortEffect;
  namespace workload = mlir::llk::workload;
  const shape::Dimension m = shape::Dimension::symbol("M");
  const shape::Dimension n = shape::Dimension::symbol("N");
  const shape::Dimension k = shape::Dimension::symbol("K");
  workload::WorkloadContract contract;
  contract.version = 2;
  contract.ports = {
      {"lhs", PortDirection::Input, PortEffect::Read, {m, k}, "bf16"},
      {"rhs0", PortDirection::Input, PortEffect::Read, {k, n}, "bf16"},
      {"init", PortDirection::Input, PortEffect::Read, {m, n}, "bf16"},
      {"output", PortDirection::Output, PortEffect::Write, {m, n}, "bf16"},
  };
  contract.symbols = {{"M", {}}, {"N", {}}, {"K", {}}};
  attachContract(kernel, context, contract);
}

llk::MappedCompilation compileToLowered(mlir::ModuleOp module,
                                        llvm::StringRef symbol) {
  llk::MappedCompileOptions options;
  options.entrySymbol = symbol.str();
  options.backend = llk::MappedBackend::Reference;
  options.stop = llk::MappedStop::Lowered;
  llvm::Expected<llk::MappedCompilation> compiled =
      llk::compileConcreteMicroKernel(module, options);
  if (!compiled) {
    ADD_FAILURE() << llvm::toString(compiled.takeError());
    return {};
  }
  return std::move(*compiled);
}

TEST(DynamicMappedInvocation, LowersCallerOwnedDestinationToArgument) {
  mlir::MLIRContext context(buildRegistry());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kDynamicCopyKernel, &context);
  ASSERT_TRUE(module);
  attachCopyContract(*module, context);

  llk::MappedCompilation lowered = compileToLowered(*module, "copy_dynamic");
  ASSERT_TRUE(lowered.module);

  // Bufferization must keep the destination in the caller's storage: a
  // dynamic-shaped allocation would be the whole output materialized in
  // compiler-owned memory.
  bool dynamicAllocation = false;
  lowered.module->walk([&](mlir::memref::AllocOp alloc) {
    dynamicAllocation |= !alloc.getType().hasStaticShape();
  });
  EXPECT_FALSE(dynamicAllocation);

  llvm::Expected<llk::PreparedMappedKernel> prepared =
      llk::prepareMappedKernelForInvocation(*lowered.module, "copy_dynamic");
  ASSERT_TRUE(static_cast<bool>(prepared))
      << llvm::toString(prepared.takeError());

  // ABI v2: three symbolic ports are bound by four signature positions, and
  // the output port is the caller-owned destination.
  EXPECT_EQ(prepared->abi.version, llk::kMappedAbiVersionV2);
  ASSERT_EQ(prepared->abi.inputs.size(), 1u);
  ASSERT_EQ(prepared->abi.outputs.size(), 1u);
  ASSERT_TRUE(prepared->abi.symbols.count("M"));
  ASSERT_TRUE(prepared->abi.symbols.count("N"));
  ASSERT_EQ(prepared->abi.outputs[0].dimensions.size(), 2u);
  EXPECT_TRUE(prepared->abi.outputs[0].dimensions[0].isSymbol());
  EXPECT_EQ(prepared->abi.outputs[0].dimensions[0].getSymbol(), "M");
  ASSERT_EQ(prepared->abi.arguments.size(), 2u);
  EXPECT_EQ(prepared->abi.arguments[1].role, llk::AbiPortRole::Output);

  // The prepared function returns void: the result was verified to be the
  // destination and then eliminated, rather than copied into a fresh argument.
  auto function =
      prepared->module->lookupSymbol<mlir::func::FuncOp>("copy_dynamic");
  ASSERT_TRUE(function);
  EXPECT_EQ(function.getFunctionType().getNumInputs(), 2u);
  EXPECT_EQ(function.getFunctionType().getNumResults(), 0u);
  auto destination =
      mlir::dyn_cast<mlir::MemRefType>(function.getFunctionType().getInput(1));
  ASSERT_TRUE(destination);
  EXPECT_TRUE(destination.isDynamicDim(0));
  EXPECT_TRUE(destination.isDynamicDim(1));

  // Dynamic dimensions survive in the descriptor: the caller supplies the
  // extents, and no extent was frozen at compile time.
  bool dynamicDeallocOfDestination = false;
  function.walk([&](mlir::memref::DeallocOp dealloc) {
    for (mlir::BlockArgument argument : function.getArguments())
      if (dealloc.getMemref() == argument)
        dynamicDeallocOfDestination = true;
  });
  EXPECT_FALSE(dynamicDeallocOfDestination);
}

//===----------------------------------------------------------------------===//
// The mapped compiler's own path
//===----------------------------------------------------------------------===//

/// A fully dynamic M x N x K contraction, exported and mapped rather than
/// written as Micro-IR by hand. The canonical symbolic contract is attached
/// the way the mapping pipeline attaches it, so ABI v2 records M, N and K
/// instead of positional placeholders.
constexpr llvm::StringLiteral kDynamicMatmulSource = R"mlir(
module {
  func.func @matmul_dynamic_root(%a: tensor<?x?xbf16>, %b: tensor<?x?xbf16>,
                                 %init: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
    %y = llk.matmul ins(%a, %b : tensor<?x?xbf16>, tensor<?x?xbf16>)
        outs(%init : tensor<?x?xbf16>)
        {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
        -> tensor<?x?xbf16>
    return %y : tensor<?x?xbf16>
  }
}
)mlir";

llk::InvocationBuffer2D bf16Buffer(std::vector<uint16_t> &storage, int64_t rows,
                                   int64_t columns) {
  return {{storage.data(), storage.data(), 0, rows, columns, columns, 1},
          llk::InvocationElementType::BF16,
          storage.size() * sizeof(uint16_t)};
}

TEST(DynamicMappedInvocation,
     CompilesExportedDynamicContractionThroughMapping) {
  mlir::MLIRContext context(buildRegistry());
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kDynamicMatmulSource, &context);
  ASSERT_TRUE(module);

  mlir::PassManager pm(&context);
  pm.addPass(mlir::llk::createLLKToMicroPass("/nonexistent/llk-schedule.json"));
  ASSERT_TRUE(mlir::succeeded(pm.run(*module)));

  // Keep only the kernel: the semantic source is the reference the export was
  // made from, not a program this path compiles.
  llvm::SmallVector<mlir::Operation *> semantic;
  module->walk([&](mlir::func::FuncOp function) {
    bool hasLLK = false;
    function.walk([&](mlir::Operation *op) {
      hasLLK |= op->getName().getStringRef().starts_with("llk.");
    });
    if (hasLLK)
      semantic.push_back(function);
  });
  for (mlir::Operation *function : semantic)
    function->erase();

  mlir::Operation *kernel = nullptr;
  module->walk([&](mlir::Operation *op) {
    if (!kernel && op->getName().getStringRef() == "micro.kernel")
      kernel = op;
  });
  ASSERT_TRUE(kernel);
  attachContractionContract(kernel, context);
  auto symbol = kernel->getAttrOfType<mlir::StringAttr>("sym_name");
  ASSERT_TRUE(symbol);

  llvm::Expected<std::unique_ptr<mlir::llk::mapping::MappingTarget>> target =
      mlir::llk::target::avx2::createMappingTarget(LLK_SOURCE_DIR);
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  llvm::Expected<mlir::llk::mapping::WorkloadGraph> graph =
      mlir::llk::mapping::extractWorkloadGraph(kernel);
  ASSERT_TRUE(static_cast<bool>(graph)) << llvm::toString(graph.takeError());

  mlir::llk::mapping::LayoutContext layout;
  layout.rank = 2;
  layout.elementType = "bf16";
  mlir::llk::mapping::MappingSearchOptions searchOptions;
  searchOptions.mode = mlir::llk::mapping::SearchMode::Exact;
  mlir::llk::mapping::CoveringSearch search(*graph, **target, context, layout,
                                            searchOptions);
  llvm::Expected<mlir::llk::mapping::MappingSearchResult> result =
      search.search();
  ASSERT_TRUE(static_cast<bool>(result)) << llvm::toString(result.takeError());
  ASSERT_FALSE(result->plans.empty());

  llk::MappedCompileOptions options;
  options.entrySymbol = symbol.getValue().str();
  options.backend = llk::MappedBackend::Reference;
  options.stop = llk::MappedStop::Lowered;
  llvm::Expected<llk::MappedCompilation> compiled = llk::compileMappedKernel(
      *module, **target, result->plans.front(), options);
  ASSERT_TRUE(static_cast<bool>(compiled))
      << llvm::toString(compiled.takeError());
  ASSERT_TRUE(compiled->module);

  llvm::Expected<llk::PreparedMappedKernel> prepared =
      llk::prepareMappedKernelForInvocation(*compiled->module,
                                            options.entrySymbol);
  ASSERT_TRUE(static_cast<bool>(prepared))
      << llvm::toString(prepared.takeError());

  // The canonical contract survives export, search and binding: three read
  // inputs and one caller-owned destination, with M/N/K as shared symbols.
  EXPECT_EQ(prepared->abi.version, llk::kMappedAbiVersionV2);
  ASSERT_EQ(prepared->abi.inputs.size(), 3u);
  ASSERT_EQ(prepared->abi.outputs.size(), 1u);
  for (llvm::StringRef name : {"M", "N", "K"})
    EXPECT_TRUE(prepared->abi.symbols.count(name.str())) << name.str();
  ASSERT_EQ(prepared->abi.outputs[0].dimensions.size(), 2u);
  EXPECT_EQ(prepared->abi.outputs[0].dimensions[0].getSymbol(), "M");
  EXPECT_EQ(prepared->abi.outputs[0].dimensions[1].getSymbol(), "N");
  ASSERT_EQ(prepared->abi.arguments.size(), 4u);
  EXPECT_EQ(prepared->abi.arguments.back().role, llk::AbiPortRole::Output);

  auto function =
      prepared->module->lookupSymbol<mlir::func::FuncOp>(options.entrySymbol);
  ASSERT_TRUE(function);
  EXPECT_EQ(function.getFunctionType().getNumInputs(), 4u);
  EXPECT_EQ(function.getFunctionType().getNumResults(), 0u);

  // No output-sized compiler buffer: every remaining allocation is a static
  // tile, never the dynamic M x N output.
  bool dynamicAllocation = false;
  prepared->module->walk([&](mlir::memref::AllocOp alloc) {
    dynamicAllocation |= !alloc.getType().hasStaticShape();
  });
  EXPECT_FALSE(dynamicAllocation);

  // The dynamic signature reaches the descriptor transport: a 7x5 output with
  // K = 13 resolves M, N and K from the caller's buffers without recompiling.
  std::vector<uint16_t> lhs(7 * 13), rhs(13 * 5), init(7 * 5), dst(7 * 5);
  llvm::Expected<shape::ShapeValues> values = llk::resolveMappedShapeValues(
      prepared->abi,
      {bf16Buffer(lhs, 7, 13), bf16Buffer(rhs, 13, 5), bf16Buffer(init, 7, 5)},
      {bf16Buffer(dst, 7, 5)});
  ASSERT_TRUE(static_cast<bool>(values)) << llvm::toString(values.takeError());
  EXPECT_EQ(*values, (shape::ShapeValues{{"M", 7}, {"N", 5}, {"K", 13}}));
}
} // namespace
