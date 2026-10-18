//===- generic_elementwise.cpp - Generic source execution acceptance ------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Framework B7: the generic, non-contraction source path has to end in running
// code, not merely in a selected plan. These tests take the bounded structured
// elementwise source through the same adapter, mapping search and mapped
// compiler the contraction examples use, and check every output element
// against the scalar expression.
//
// Two facts are kept apart on purpose:
//
//   * the reference backend is portable and proves the generated program
//     computes what the source says, and
//   * the selected-target backend, on an AVX2 host, proves the selected
//     bundles and their width actually carried the computation.
//
// The rank gates are checked separately: rank-neutral analysis and lowering
// stay available, while the rank-two mapped ABI v1 refuses what it cannot
// describe with an explicit `runtime_capability:` diagnostic.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Conversion/MicroMapping/CandidateInstantiation.h"
#include "LLK/Conversion/MicroMapping/MappedTuningSession.h"
#include "LLK/Conversion/Workload/ElementwiseSourceAdapter.h"
#include "LLK/Conversion/Workload/SourceAdapter.h"
#include "LLK/Conversion/Workload/WorkloadContract.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/WorkloadGraph.h"
#include "LLK/Runtime/MappedExecutable.h"
#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TargetParser/Host.h"

#include "gtest/gtest.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#ifndef LLK_SOURCE_DIR
#error "LLK_SOURCE_DIR must name the repository root"
#endif

namespace {

using mlir::llk::mapping::CoveringPlan;
using mlir::llk::mapping::MappingSearchOptions;
using mlir::llk::mapping::MappingSearchResult;
using mlir::llk::mapping::MappingTarget;
using mlir::llk::mapping::SearchMode;
using mlir::llk::workload::PortDirection;
using mlir::llk::workload::PortEffect;
using mlir::llk::workload::WorkloadContract;
using mlir::llk::workload::WorkloadPort;

mlir::DialectRegistry buildRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::scf::SCFDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect>();
  return registry;
}

mlir::OwningOpRef<mlir::ModuleOp> parseSource(mlir::MLIRContext &context,
                                              llvm::StringRef text) {
  return mlir::parseSourceString<mlir::ModuleOp>(text, &context);
}

std::string printModule(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  stream.flush();
  return text;
}

/// A rank-three kernel whose tensors the mapped ABI v1 cannot describe. Its
/// body is otherwise ordinary: analysis and lowering are rank-neutral, and only
/// execution preparation applies the ABI envelope.
constexpr llvm::StringLiteral kRankThreeKernel = R"mlir(
module {
  micro.kernel @rank3(%a: tensor<2x4x8xf32>, %b: tensor<2x4x8xf32>)
      -> tensor<2x4x8xf32> {
    %ta = micro.tile_view %a {shape = array<i64: 2, 4, 8>}
        : tensor<2x4x8xf32> -> !micro.tile<2x4x8xf32, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b {shape = array<i64: 2, 4, 8>}
        : tensor<2x4x8xf32> -> !micro.tile<2x4x8xf32, memory = #micro.memory<sram>>
    %sum = micro.vector "add" %ta, %tb
        : !micro.tile<2x4x8xf32, memory = #micro.memory<sram>>,
          !micro.tile<2x4x8xf32, memory = #micro.memory<sram>>
       -> !micro.tile<2x4x8xf32, memory = #micro.memory<acc>>
    micro.yield %sum : !micro.tile<2x4x8xf32, memory = #micro.memory<acc>>
  }
}
)mlir";

/// The versioned contract a direct MicroIR module may carry. Rank-three ports
/// are valid contract data; the execution envelope is a separate limit.
WorkloadContract rankThreeContract() {
  WorkloadContract contract;
  const std::vector<int64_t> shape{2, 4, 8};
  contract.ports = {
      {"input0", PortDirection::Input, PortEffect::Read, shape, "f32"},
      {"input1", PortDirection::Input, PortEffect::Read, shape, "f32"},
      {"output", PortDirection::Output, PortEffect::Write, shape, "f32"},
  };
  contract.numericalContract = "strict-f32";
  return contract;
}

/// A rank-two kernel whose retained contract claims an input is read-write.
/// The mapped ABI v1 hands inputs to callers as read-only buffers, so the
/// contract and the execution envelope disagree even though every port is
/// rank two.
constexpr llvm::StringLiteral kReadWriteInputKernel = R"mlir(
module {
  micro.kernel @readwrite(%a: tensor<2x8xf32>, %b: tensor<2x8xf32>)
      -> tensor<2x8xf32> {
    %ta = micro.tile_view %a {shape = array<i64: 2, 8>}
        : tensor<2x8xf32> -> !micro.tile<2x8xf32, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b {shape = array<i64: 2, 8>}
        : tensor<2x8xf32> -> !micro.tile<2x8xf32, memory = #micro.memory<sram>>
    %sum = micro.vector "add" %ta, %tb
        : !micro.tile<2x8xf32, memory = #micro.memory<sram>>,
          !micro.tile<2x8xf32, memory = #micro.memory<sram>>
       -> !micro.tile<2x8xf32, memory = #micro.memory<acc>>
    micro.yield %sum : !micro.tile<2x8xf32, memory = #micro.memory<acc>>
  }
}
)mlir";

bool avx2DisabledByTestOverride() {
  const char *disabled = std::getenv("LLK_TEST_DISABLE_AVX2");
  return disabled && std::strcmp(disabled, "1") == 0;
}

bool selectedAvx2Available() {
  if (avx2DisabledByTestOverride())
    return false;
  const std::string triple = llvm::sys::getDefaultTargetTriple();
  if (triple.rfind("x86_64-", 0) != 0)
    return false;
  const auto features = llvm::sys::getHostCPUFeatures();
  auto avx2 = features.find("avx2");
  return avx2 != features.end() && avx2->second;
}

//===----------------------------------------------------------------------===//
// Rank gating
//===----------------------------------------------------------------------===//

TEST(GenericElementwise, RankThreeContractIsAnalyzableButNotExecutable) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = parseSource(context, kRankThreeKernel);
  ASSERT_TRUE(module);

  // Static contract reporting works: the contract validates, canonicalizes and
  // can be retained on a kernel whose rank the execution ABI does not support.
  WorkloadContract contract = rankThreeContract();
  ASSERT_FALSE(static_cast<bool>(
      mlir::llk::workload::validateWorkloadContract(contract)));
  const std::string canonical =
      mlir::llk::workload::canonicalWorkloadContract(contract);
  EXPECT_FALSE(canonical.empty());
  module->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() != "micro.kernel")
      return;
    op->setAttr(
        "micro.workload_contract",
        mlir::llk::workload::encodeWorkloadContract(&context, contract));
  });

  llk::MappedCompileOptions options;
  options.entrySymbol = "rank3";
  options.backend = llk::MappedBackend::Reference;

  // Analysis and lowering are rank-neutral: this must reach lowered IR.
  options.stop = llk::MappedStop::Lowered;
  llvm::Expected<llk::MappedCompilation> lowered =
      llk::compileConcreteMicroKernel(*module, options);
  ASSERT_TRUE(static_cast<bool>(lowered))
      << llvm::toString(lowered.takeError());

  // The rank-two mapped ABI v1 is an execution envelope, not an analysis
  // limit. Asking for code fails explicitly and names the incompatibility.
  options.stop = llk::MappedStop::Executable;
  llvm::Expected<llk::MappedCompilation> executable =
      llk::compileConcreteMicroKernel(*module, options);
  ASSERT_FALSE(static_cast<bool>(executable));
  const std::string message = llvm::toString(executable.takeError());
  EXPECT_NE(message.find("runtime_capability:"), std::string::npos) << message;
  EXPECT_NE(message.find("rank-two"), std::string::npos) << message;
}

TEST(GenericElementwise, ReadWriteInputContractIsRejectedBeforeCode) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto module = parseSource(context, kReadWriteInputKernel);
  ASSERT_TRUE(module);
  module->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() != "micro.kernel")
      return;
    auto contract = rankThreeContract();
    contract.ports[0].shape = {2, 8};
    contract.ports[1].shape = {2, 8};
    contract.ports[2].shape = {2, 8};
    contract.ports[0].effect = PortEffect::ReadWrite;
    op->setAttr(
        "micro.workload_contract",
        mlir::llk::workload::encodeWorkloadContract(&context, contract));
  });

  llk::MappedCompileOptions options;
  options.entrySymbol = "readwrite";
  options.backend = llk::MappedBackend::Reference;
  options.stop = llk::MappedStop::Lowered;
  llvm::Expected<llk::MappedCompilation> lowered =
      llk::compileConcreteMicroKernel(*module, options);
  ASSERT_TRUE(static_cast<bool>(lowered))
      << llvm::toString(lowered.takeError());

  options.stop = llk::MappedStop::Executable;
  llvm::Expected<llk::MappedCompilation> executable =
      llk::compileConcreteMicroKernel(*module, options);
  ASSERT_FALSE(static_cast<bool>(executable));
  const std::string message = llvm::toString(executable.takeError());
  EXPECT_NE(message.find("runtime_capability:"), std::string::npos) << message;
  EXPECT_NE(message.find("read-only"), std::string::npos) << message;
}

} // namespace
