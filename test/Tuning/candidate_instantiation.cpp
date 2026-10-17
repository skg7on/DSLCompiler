//===- candidate_instantiation.cpp - Tests for candidate instantiation ----===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/LLKToMicro/LLKToMicro.h"
#include "LLK/Conversion/MicroMapping/CandidateInstantiation.h"
#include "LLK/Conversion/Workload/SourceAdapter.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Perf/Candidate.h"
#include "LLK/Perf/SearchSpace.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <string>
#include <vector>

#ifndef LLK_SOURCE_DIR
#error "LLK_SOURCE_DIR must name the repository root"
#endif

namespace {

std::string printModule(mlir::ModuleOp module) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  module.print(stream);
  stream.flush();
  return text;
}

mlir::llk::perf::SearchSpace testSpace() {
  using namespace mlir::llk::perf;
  SearchSpace space;
  space.workload = "matmul";
  space.params = {
      {"BM", "integer", {SearchChoice(int64_t{4}), SearchChoice(int64_t{8})}},
      {"BN", "integer", {SearchChoice(int64_t{16})}},
      {"BK", "integer", {SearchChoice(int64_t{32})}},
      {"tile_layout", "layout", {SearchChoice("row_major")}},
      {"memory_path", "memory_path", {SearchChoice("dram:sram:acc")}},
      {"owner_mapping",
       "owner_mapping",
       {SearchChoice("worker/vector_engine")}},
      {"fragment_shape", "fragment_shape", {SearchChoice("8x16x32")}},
      {"tail_policy", "tail_policy", {SearchChoice("none")}},
      {"pipeline_stages", "integer", {SearchChoice(int64_t{1})}},
      {"vector_width", "integer", {SearchChoice(int64_t{8})}},
      {"num_threads", "integer", {SearchChoice(int64_t{1})}},
      {"grain_size", "integer", {SearchChoice(int64_t{1})}},
  };
  return space;
}

mlir::llk::perf::Candidate candidate(int64_t BM) {
  using namespace mlir::llk::perf;
  Candidate result;
  result.values = {{"BM", BM},          {"BN", 16},
                   {"BK", 32},          {"pipeline_stages", 1},
                   {"vector_width", 8}, {"num_threads", 1},
                   {"grain_size", 1}};
  result.symbolicValues = {{"tile_layout", "row_major"},
                           {"memory_path", "dram:sram:acc"},
                           {"owner_mapping", "worker/vector_engine"},
                           {"fragment_shape", "8x16x32"},
                           {"tail_policy", "none"}};
  result.id = computeCandidateId(result.values, result.symbolicValues);
  return result;
}

TEST(CandidateInstantiationTest, SelectsTheExactRootBeforeWriting) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  std::string path = std::string(LLK_SOURCE_DIR) +
                     "/test/Tuning/Inputs/issue129/two_roots.mlir";
  auto buffer = llvm::MemoryBuffer::getFile(path);
  ASSERT_TRUE(bool(buffer)) << buffer.getError().message();
  auto module =
      mlir::parseSourceString<mlir::ModuleOp>((*buffer)->getBuffer(), &context);
  ASSERT_TRUE(module);
  const std::string original = printModule(*module);

  mlir::llk::ScheduleEntry schedule;
  schedule.BM = 8;
  schedule.BN = 16;
  schedule.BK = 32;
  schedule.fragment_shape = "8x16x32";
  auto exported = mlir::llk::exportMicroKernelFromSchedule(*module, "two_roots",
                                                           1, schedule);
  ASSERT_TRUE(bool(exported)) << llvm::toString(exported.takeError());
  EXPECT_EQ((*exported)->getName().getStringRef(), "micro.kernel");
  EXPECT_EQ(printModule(*module).find("micro.kernel"),
            printModule(*module).rfind("micro.kernel"));

  const std::string withKernel = printModule(*module);
  auto invalid = mlir::llk::exportMicroKernelFromSchedule(*module, "two_roots",
                                                          2, schedule);
  ASSERT_FALSE(bool(invalid));
  llvm::consumeError(invalid.takeError());
  EXPECT_EQ(printModule(*module), withKernel);
  EXPECT_NE(original, withKernel);
}

TEST(CandidateInstantiationTest, PreservesSourceAndBindsSemanticCandidates) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  std::string path = std::string(LLK_SOURCE_DIR) +
                     "/test/Tuning/Inputs/issue129/two_roots.mlir";
  auto buffer = llvm::MemoryBuffer::getFile(path);
  ASSERT_TRUE(bool(buffer)) << buffer.getError().message();
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>((*buffer)->getBuffer(), &context);
  ASSERT_TRUE(source);
  const std::string original = printModule(*source);

  mlir::llk::perf::WorkloadShape shape;
  shape.M = 8;
  shape.N = 16;
  shape.K = 32;
  shape.inputDType = "bf16";
  shape.weightDType = "bf16";
  shape.accumulatorDType = "f32";
  shape.outputDType = "bf16";
  mlir::llk::machine::MachineModel machine;
  mlir::llk::tuning::CandidateInstantiationOptions options;
  options.sourceSymbol = "two_roots";
  options.sourceRootOrdinal = 1;
  options.machine = &machine;

  auto one = mlir::llk::tuning::instantiateCandidate(
      *source, testSpace(), candidate(8), shape, options);
  ASSERT_TRUE(bool(one)) << llvm::toString(one.takeError());
  auto two = mlir::llk::tuning::instantiateCandidate(
      *source, testSpace(), candidate(4), shape, options);
  ASSERT_TRUE(bool(two)) << llvm::toString(two.takeError());
  EXPECT_NE(one->bindingHash, two->bindingHash);
  EXPECT_NE(one->sourceGraphHash, 0u);
  EXPECT_EQ(one->sourceGraphHash, two->sourceGraphHash);
  EXPECT_EQ(printModule(*source), original);
  EXPECT_EQ(
      one->kernel->getAttrOfType<mlir::StringAttr>("candidate").getValue(),
      candidate(8).id);
  EXPECT_EQ(
      one->kernel->getAttrOfType<mlir::StringAttr>("source_symbol").getValue(),
      "two_roots");
  EXPECT_EQ(one->kernel->getAttrOfType<mlir::IntegerAttr>("source_root_ordinal")
                .getInt(),
            1);
  EXPECT_TRUE(one->kernel->getAttr("source_math_mode"));

  auto kernelType = mlir::dyn_cast<mlir::FunctionType>(
      one->kernel->getAttrOfType<mlir::TypeAttr>("function_type").getValue());
  ASSERT_TRUE(kernelType);
  ASSERT_EQ(kernelType.getNumInputs(), 3u);
  auto lhsType = mlir::cast<mlir::ShapedType>(kernelType.getInput(0));
  auto rhsType = mlir::cast<mlir::ShapedType>(kernelType.getInput(1));
  EXPECT_EQ(lhsType.getDimSize(0), 8);
  EXPECT_EQ(lhsType.getDimSize(1), 32);
  EXPECT_EQ(rhsType.getDimSize(0), 32);
  EXPECT_EQ(rhsType.getDimSize(1), 16);
  EXPECT_EQ(kernelType.getInput(2), kernelType.getResult(0));
  mlir::BlockArgument initArgument =
      one->kernel->getRegion(0).front().getArgument(2);
  bool readsInit = false;
  one->kernel->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "micro.tile_view" &&
        op->getNumOperands() > 0 && op->getOperand(0) == initArgument)
      readsInit = true;
  });
  EXPECT_TRUE(readsInit) << "the source out/init tensor seeds the accumulator";

  auto duplicate = mlir::llk::tuning::instantiateCandidate(
      *source, testSpace(), candidate(8), shape, options);
  ASSERT_TRUE(bool(duplicate)) << llvm::toString(duplicate.takeError());
  EXPECT_EQ(printModule(one->module.get()),
            printModule(duplicate->module.get()));
}

TEST(CandidateInstantiationTest,
     SemanticAttributesParticipateInSourceIdentity) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  std::string path = std::string(LLK_SOURCE_DIR) +
                     "/test/Tuning/Inputs/issue129/two_roots.mlir";
  auto buffer = llvm::MemoryBuffer::getFile(path);
  ASSERT_TRUE(bool(buffer));
  std::string changed = (*buffer)->getBuffer().str();
  size_t mode = changed.find("bounded_fast");
  ASSERT_NE(mode, std::string::npos);
  mode =
      changed.find("bounded_fast", mode + std::string("bounded_fast").size());
  ASSERT_NE(mode, std::string::npos);
  changed.replace(mode, std::string("bounded_fast").size(), "strict");
  auto changedSource =
      mlir::parseSourceString<mlir::ModuleOp>(changed, &context);
  ASSERT_TRUE(changedSource);
  auto originalSource = llvm::MemoryBuffer::getFile(path);
  ASSERT_TRUE(bool(originalSource));
  auto original = mlir::parseSourceString<mlir::ModuleOp>(
      (*originalSource)->getBuffer(), &context);
  ASSERT_TRUE(original);

  mlir::llk::perf::WorkloadShape shape;
  shape.M = 8;
  shape.N = 16;
  shape.K = 32;
  mlir::llk::machine::MachineModel machine;
  mlir::llk::tuning::CandidateInstantiationOptions options;
  options.sourceSymbol = "two_roots";
  options.sourceRootOrdinal = 1;
  options.machine = &machine;
  auto normal = mlir::llk::tuning::instantiateCandidate(
      *original, testSpace(), candidate(8), shape, options);
  auto changedMode = mlir::llk::tuning::instantiateCandidate(
      *changedSource, testSpace(), candidate(8), shape, options);
  ASSERT_TRUE(bool(normal)) << llvm::toString(normal.takeError());
  ASSERT_TRUE(bool(changedMode)) << llvm::toString(changedMode.takeError());
  EXPECT_NE(normal->sourceGraphHash, changedMode->sourceGraphHash);
}

TEST(CandidateInstantiationTest, RetainsSwiGLUProjectionAndEpilogueOrder) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::tensor::TensorDialect,
                  mlir::arith::ArithDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  std::string path =
      std::string(LLK_SOURCE_DIR) + "/test/Tuning/Inputs/issue129/swiglu.mlir";
  auto buffer = llvm::MemoryBuffer::getFile(path);
  ASSERT_TRUE(bool(buffer));
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>((*buffer)->getBuffer(), &context);
  ASSERT_TRUE(source);
  const std::string original = printModule(*source);
  auto space = testSpace();
  space.workload = "fused_swiglu";
  mlir::llk::perf::WorkloadShape shape;
  shape.M = 8;
  shape.N = 16;
  shape.K = 32;
  mlir::llk::machine::MachineModel machine;
  mlir::llk::tuning::CandidateInstantiationOptions options;
  options.sourceSymbol = "swiglu";
  options.machine = &machine;
  auto instance = mlir::llk::tuning::instantiateCandidate(
      *source, space, candidate(8), shape, options);
  ASSERT_TRUE(bool(instance)) << llvm::toString(instance.takeError());
  EXPECT_EQ(instance->bindingFacts.contractions.size(), 2u);
  unsigned mmaCount = 0;
  std::vector<std::string> vectorOps;
  instance->kernel->walk([&](mlir::Operation *op) {
    auto name = op->getName().getStringRef();
    if (name == "micro.mma")
      ++mmaCount;
    if (name == "micro.vector") {
      auto kind = op->getAttrOfType<mlir::StringAttr>("op");
      if (kind)
        vectorOps.push_back(kind.getValue().str());
    }
  });
  EXPECT_EQ(mmaCount, 2u);
  auto silu = std::find(vectorOps.begin(), vectorOps.end(), "silu");
  auto multiply = std::find(vectorOps.begin(), vectorOps.end(), "mul");
  ASSERT_NE(silu, vectorOps.end());
  ASSERT_NE(multiply, vectorOps.end());
  EXPECT_LT(silu, multiply);
  // The gated projection is two arms and one silu/mul epilogue followed by the
  // accumulator-to-output conversion, not a single widened contraction.
  auto conversion = std::find(vectorOps.begin(), vectorOps.end(), "convert");
  ASSERT_NE(conversion, vectorOps.end());
  EXPECT_LT(multiply, conversion);
  EXPECT_EQ(printModule(*source), original);
}

/// The built-in contraction family is what the WorkloadShape entry point now
/// dispatches to. Both overloads must describe the same source root and emit
/// the same kernel, because the shape overload is only a translation of the
/// supplied workload into the generic contract.
TEST(CandidateInstantiationTest,
     ShapeAndContractOverloadsAgreeOnTheSelectedRoot) {
  mlir::DialectRegistry dialects;
  dialects.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
  mlir::MLIRContext context(dialects);
  context.loadAllAvailableDialects();
  std::string path = std::string(LLK_SOURCE_DIR) +
                     "/test/Tuning/Inputs/issue129/two_roots.mlir";
  auto buffer = llvm::MemoryBuffer::getFile(path);
  ASSERT_TRUE(bool(buffer)) << buffer.getError().message();
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>((*buffer)->getBuffer(), &context);
  ASSERT_TRUE(source);
  const std::string originalSource = printModule(*source);

  // Resolve and describe the selected root through the default registry: the
  // adapter is the authority on what the source implements.
  mlir::llk::workload::SourceAdapterRegistry registry =
      mlir::llk::workload::makeDefaultSourceAdapters();
  EXPECT_FALSE(registry.empty()) << "the contraction family must be registered";
  auto root = mlir::llk::workload::selectSourceRoot(*source, {"two_roots", 1},
                                                    registry);
  ASSERT_TRUE(bool(root)) << llvm::toString(root.takeError());
  auto adapter = registry.resolve(*root);
  ASSERT_TRUE(bool(adapter)) << llvm::toString(adapter.takeError());
  EXPECT_EQ((*adapter)->key(), "llk-contraction-v1");
  auto described = (*adapter)->describe(*root);
  ASSERT_TRUE(bool(described)) << llvm::toString(described.takeError());

  // The described contract is a contraction of M=8, N=16, K=32 with three read
  // inputs and a written output, in the emitted kernel's order.
  EXPECT_EQ(described->axes.at("M"), 8);
  EXPECT_EQ(described->axes.at("N"), 16);
  EXPECT_EQ(described->axes.at("K"), 32);
  ASSERT_EQ(described->ports.size(), 4u);
  EXPECT_EQ(described->ports[0].shape, (std::vector<int64_t>{8, 32}));
  EXPECT_EQ(described->ports[1].shape, (std::vector<int64_t>{32, 16}));
  EXPECT_EQ(described->ports[2].shape, (std::vector<int64_t>{8, 16}));
  EXPECT_EQ(described->ports[3].direction,
            mlir::llk::workload::PortDirection::Output);
  EXPECT_EQ(described->ports[3].effect, mlir::llk::workload::PortEffect::Write);

  mlir::llk::perf::WorkloadShape shape;
  shape.M = 8;
  shape.N = 16;
  shape.K = 32;
  shape.inputDType = "bf16";
  shape.weightDType = "bf16";
  shape.accumulatorDType = "f32";
  shape.outputDType = "bf16";
  mlir::llk::machine::MachineModel machine;
  mlir::llk::tuning::CandidateInstantiationOptions options;
  options.sourceSymbol = "two_roots";
  options.sourceRootOrdinal = 1;
  options.machine = &machine;

  auto legacy = mlir::llk::tuning::instantiateCandidate(
      *source, testSpace(), candidate(8), shape, options);
  ASSERT_TRUE(bool(legacy)) << llvm::toString(legacy.takeError());
  auto generic = mlir::llk::tuning::instantiateCandidate(
      *source, testSpace(), candidate(8), *described, options, registry);
  ASSERT_TRUE(bool(generic)) << llvm::toString(generic.takeError());

  // Ordered signature types, primitive sequences, contraction facts, init
  // reads and source numerical attributes all survive the translation.
  EXPECT_EQ(legacy->bindingFacts.contractions.size(),
            generic->bindingFacts.contractions.size());
  EXPECT_EQ(legacy->kernel->getAttr("function_type"),
            generic->kernel->getAttr("function_type"));
  EXPECT_EQ(legacy->kernel->getAttr("source_math_mode"),
            generic->kernel->getAttr("source_math_mode"));
  EXPECT_EQ(printModule(*legacy->module), printModule(*generic->module));
  EXPECT_EQ(legacy->receipt.source, generic->receipt.source);
  EXPECT_EQ(printModule(*source), originalSource);

  // The versioned generic provenance names the adapter and the workload
  // contract, and the compatibility metadata is still there beside it.
  EXPECT_EQ(legacy->kernel->getAttrOfType<mlir::StringAttr>("source_adapter")
                .getValue(),
            "llk-contraction-v1");
  EXPECT_TRUE(legacy->kernel->getAttr("micro.workload_contract"));
  EXPECT_TRUE(legacy->kernel->getAttr("original_workload"));
  EXPECT_EQ(legacy->bindingFacts.contractions.size(), 1u);
  EXPECT_EQ(legacy->bindingFacts.contractions[0].M, 8);
  EXPECT_EQ(legacy->bindingFacts.contractions[0].N, 16);
  EXPECT_EQ(legacy->bindingFacts.contractions[0].K, 32);

  // Changed contract metadata cannot be substituted for the contract the root
  // describes: a record recorded against the old contract is not silently
  // reused under a new one.
  auto relabelled = *described;
  relabelled.ports[0].name = "renamed";
  auto mismatch = mlir::llk::tuning::instantiateCandidate(
      *source, testSpace(), candidate(8), relabelled, options, registry);
  ASSERT_FALSE(bool(mismatch));
  EXPECT_NE(
      llvm::toString(mismatch.takeError()).find("source_contract_mismatch"),
      std::string::npos);
}

/// A contraction whose output element type equals the accumulator needs no
/// conversion, and an operator without an activation epilogue emits no vector
/// operation at all. The described primitives must say so, because a required
/// primitive that the export never emits is a false contract.
TEST(CandidateInstantiationTest,
     DescribedPrimitivesMatchWhatEachContractionRootEmits) {
  mlir::DialectRegistry dialects;
  dialects.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
  mlir::MLIRContext context(dialects);
  context.loadAllAvailableDialects();
  mlir::llk::workload::SourceAdapterRegistry registry =
      mlir::llk::workload::makeDefaultSourceAdapters();

  auto describeRoot = [&](llvm::StringRef file,
                          llvm::StringRef symbol) -> mlir::Attribute {
    std::string path = std::string(LLK_SOURCE_DIR) +
                       "/test/Tuning/Inputs/issue129/" + file.str();
    auto buffer = llvm::MemoryBuffer::getFile(path);
    EXPECT_TRUE(bool(buffer)) << path;
    if (!buffer)
      return {};
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        (*buffer)->getBuffer(), &context);
    EXPECT_TRUE(bool(module));
    if (!module)
      return {};
    auto root = mlir::llk::workload::selectSourceRoot(
        *module, {symbol.str(), 0}, registry);
    EXPECT_TRUE(bool(root)) << llvm::toString(root.takeError());
    if (!root)
      return {};
    auto adapter = registry.resolve(*root);
    EXPECT_TRUE(bool(adapter)) << llvm::toString(adapter.takeError());
    if (!adapter)
      return {};
    auto described = (*adapter)->describe(*root);
    EXPECT_TRUE(bool(described)) << llvm::toString(described.takeError());
    if (!described)
      return {};
    return mlir::llk::workload::encodeWorkloadContract(&context, *described);
  };

  // linalg.matmul: accumulator and output are both f32, and there is no
  // epilogue, so no `micro.vector` is emitted.
  auto linalg = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
      describeRoot("linalg_matmul.mlir", "linalg_source"));
  ASSERT_TRUE(linalg);
  auto linalgContract = mlir::llk::workload::decodeWorkloadContract(linalg);
  ASSERT_TRUE(bool(linalgContract))
      << llvm::toString(linalgContract.takeError());
  EXPECT_EQ(linalgContract->requiredPrimitives,
            (std::vector<std::string>{"micro.mma", "micro.tile_async_copy",
                                      "micro.tile_store"}));

  // llk.matmul: bf16 output over an f32 accumulator needs a conversion.
  auto llkMatmul = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
      describeRoot("two_roots.mlir", "two_roots"));
  ASSERT_TRUE(llkMatmul);
  auto matmulContract = mlir::llk::workload::decodeWorkloadContract(llkMatmul);
  ASSERT_TRUE(bool(matmulContract))
      << llvm::toString(matmulContract.takeError());
  EXPECT_EQ(matmulContract->requiredPrimitives,
            (std::vector<std::string>{"micro.mma", "micro.tile_async_copy",
                                      "micro.tile_store", "micro.vector"}));

  // SwiGLU: two arms, no initialized destination, and a silu/mul epilogue.
  auto swiglu = mlir::dyn_cast_or_null<mlir::DictionaryAttr>(
      describeRoot("swiglu.mlir", "swiglu"));
  ASSERT_TRUE(swiglu);
  auto swigluContract = mlir::llk::workload::decodeWorkloadContract(swiglu);
  ASSERT_TRUE(bool(swigluContract))
      << llvm::toString(swigluContract.takeError());
  EXPECT_EQ(swigluContract->axes.at("K"), 32);
  ASSERT_EQ(swigluContract->ports.size(), 4u);
  for (const mlir::llk::workload::WorkloadPort &port : swigluContract->ports)
    EXPECT_NE(port.shape, (std::vector<int64_t>{8, 16, 32}))
        << "a contraction port is rank two, never a fabricated rank three";
  EXPECT_TRUE(
      llvm::is_contained(swigluContract->requiredPrimitives, "micro.vector"));
}

/// Generic dispatch is structural: the workload *label* never picks an adapter
/// and never changes what is emitted. Two spaces that differ only in their
/// diagnostic label must produce byte-identical kernels.
TEST(CandidateInstantiationTest, WorkloadLabelCannotChangeTheAdapterOrKernel) {
  mlir::DialectRegistry dialects;
  dialects.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
  mlir::MLIRContext context(dialects);
  context.loadAllAvailableDialects();
  std::string path = std::string(LLK_SOURCE_DIR) +
                     "/test/Tuning/Inputs/issue129/two_roots.mlir";
  auto buffer = llvm::MemoryBuffer::getFile(path);
  ASSERT_TRUE(bool(buffer));
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>((*buffer)->getBuffer(), &context);
  ASSERT_TRUE(source);

  mlir::llk::perf::WorkloadShape shape;
  shape.M = 8;
  shape.N = 16;
  shape.K = 32;
  mlir::llk::machine::MachineModel machine;
  mlir::llk::tuning::CandidateInstantiationOptions options;
  options.sourceSymbol = "two_roots";
  options.sourceRootOrdinal = 1;
  options.machine = &machine;

  auto matmulSpace = testSpace();
  matmulSpace.workload = "matmul";
  auto swigluSpace = testSpace();
  swigluSpace.workload = "fused_swiglu";

  auto first = mlir::llk::tuning::instantiateCandidate(
      *source, matmulSpace, candidate(8), shape, options);
  auto second = mlir::llk::tuning::instantiateCandidate(
      *source, swigluSpace, candidate(8), shape, options);
  ASSERT_TRUE(bool(first)) << llvm::toString(first.takeError());
  ASSERT_TRUE(bool(second)) << llvm::toString(second.takeError());
  EXPECT_EQ(printModule(*first->module), printModule(*second->module));
  EXPECT_EQ(first->sourceGraphHash, second->sourceGraphHash);
  EXPECT_EQ(first->kernel->getAttrOfType<mlir::StringAttr>("source_adapter")
                .getValue(),
            second->kernel->getAttrOfType<mlir::StringAttr>("source_adapter")
                .getValue());
}

TEST(CandidateInstantiationTest, AcceptsAnExactLinalgMatmulSourceRoot) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  std::string path = std::string(LLK_SOURCE_DIR) +
                     "/test/Tuning/Inputs/issue129/linalg_matmul.mlir";
  auto buffer = llvm::MemoryBuffer::getFile(path);
  ASSERT_TRUE(bool(buffer));
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>((*buffer)->getBuffer(), &context);
  ASSERT_TRUE(source);
  auto shape = mlir::llk::perf::WorkloadShape{};
  shape.M = 8;
  shape.N = 16;
  shape.K = 32;
  shape.inputDType = "bf16";
  shape.weightDType = "bf16";
  shape.accumulatorDType = "f32";
  shape.outputDType = "f32";
  mlir::llk::machine::MachineModel machine;
  mlir::llk::tuning::CandidateInstantiationOptions options;
  options.sourceSymbol = "linalg_source";
  options.machine = &machine;
  auto instance = mlir::llk::tuning::instantiateCandidate(
      *source, testSpace(), candidate(8), shape, options);
  ASSERT_TRUE(bool(instance)) << llvm::toString(instance.takeError());
  EXPECT_EQ(
      instance->kernel->getAttrOfType<mlir::StringAttr>("workload").getValue(),
      "matmul");
  ASSERT_TRUE(
      instance->kernel->getAttrOfType<mlir::StringAttr>("source_semantics"));
  EXPECT_NE(
      instance->kernel->getAttrOfType<mlir::StringAttr>("source_semantics")
          .getValue()
          .find("linalg.matmul"),
      llvm::StringRef::npos);
}

TEST(CandidateInstantiationTest, RejectsAnOfferedAxisItCannotRealize) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  std::string path = std::string(LLK_SOURCE_DIR) +
                     "/test/Tuning/Inputs/issue129/two_roots.mlir";
  auto buffer = llvm::MemoryBuffer::getFile(path);
  ASSERT_TRUE(bool(buffer));
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>((*buffer)->getBuffer(), &context);
  ASSERT_TRUE(source);
  auto space = testSpace();
  space.params.push_back(
      {"retile_axis", "integer", {mlir::llk::perf::SearchChoice(int64_t{1})}});
  auto point = candidate(8);
  point.values["retile_axis"] = 1;
  point.id =
      mlir::llk::perf::computeCandidateId(point.values, point.symbolicValues);
  mlir::llk::perf::WorkloadShape shape;
  shape.M = 8;
  shape.N = 16;
  shape.K = 32;
  mlir::llk::machine::MachineModel machine;
  mlir::llk::tuning::CandidateInstantiationOptions options;
  options.sourceSymbol = "two_roots";
  options.machine = &machine;
  auto result = mlir::llk::tuning::instantiateCandidate(*source, space, point,
                                                        shape, options);
  ASSERT_FALSE(bool(result));
  EXPECT_NE(
      llvm::toString(result.takeError()).find("axis_not_realized: retile_axis"),
      std::string::npos);

  auto parallelSpace = testSpace();
  for (auto &param : parallelSpace.params)
    if (param.name == "num_threads")
      param.choices.push_back(mlir::llk::perf::SearchChoice(int64_t{2}));
  auto parallelPoint = candidate(8);
  parallelPoint.values["num_threads"] = 2;
  parallelPoint.id = mlir::llk::perf::computeCandidateId(
      parallelPoint.values, parallelPoint.symbolicValues);
  result = mlir::llk::tuning::instantiateCandidate(
      *source, parallelSpace, parallelPoint, shape, options);
  ASSERT_FALSE(bool(result));
  EXPECT_NE(
      llvm::toString(result.takeError()).find("axis_not_realized: num_threads"),
      std::string::npos);

  auto grainSpace = testSpace();
  for (auto &param : grainSpace.params)
    if (param.name == "grain_size")
      param.choices.push_back(mlir::llk::perf::SearchChoice(int64_t{2}));
  auto grainPoint = candidate(8);
  grainPoint.values["grain_size"] = 2;
  grainPoint.id = mlir::llk::perf::computeCandidateId(
      grainPoint.values, grainPoint.symbolicValues);
  result = mlir::llk::tuning::instantiateCandidate(*source, grainSpace,
                                                   grainPoint, shape, options);
  ASSERT_FALSE(bool(result));
  EXPECT_NE(
      llvm::toString(result.takeError()).find("axis_not_realized: grain_size"),
      std::string::npos);
}

TEST(CandidateInstantiationTest,
     SyntheticModeIsExplicitAndUsesTheLegacyBinder) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::arith::ArithDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  std::string path = std::string(LLK_SOURCE_DIR) +
                     "/test/Tuning/Inputs/issue129/two_roots.mlir";
  auto buffer = llvm::MemoryBuffer::getFile(path);
  ASSERT_TRUE(bool(buffer));
  auto source =
      mlir::parseSourceString<mlir::ModuleOp>((*buffer)->getBuffer(), &context);
  ASSERT_TRUE(source);
  const std::string original = printModule(*source);
  mlir::llk::perf::WorkloadShape shape;
  shape.M = 8;
  shape.N = 16;
  shape.K = 32;
  mlir::llk::tuning::CandidateInstantiationOptions options;
  options.sourceMode = mlir::llk::tuning::SourceMode::Synthetic;
  auto instance = mlir::llk::tuning::instantiateCandidate(
      *source, testSpace(), candidate(8), shape, options);
  ASSERT_TRUE(bool(instance)) << llvm::toString(instance.takeError());
  EXPECT_EQ(instance->sourceMode, mlir::llk::tuning::SourceMode::Synthetic);
  EXPECT_EQ(instance->kernel->getName().getStringRef(), "micro.kernel");
  EXPECT_EQ(printModule(*source), original);
}

} // namespace
