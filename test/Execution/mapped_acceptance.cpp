//===- mapped_acceptance.cpp - Numeric acceptance for mapped kernels ------===//
//
// Stage C8's numeric chains. Everything before this proves that IR traverses,
// that a plan binds, that an ABI is declared; none of it runs a *compiler-
// generated* kernel and checks what it computed. These tests do exactly that,
// and only that: the LLK source below is the same source the IR fixtures use,
// so the kernel under test is produced live by the export rather than
// hand-authored, and every element of every result is checked.
//
// The chains are host-portable on purpose. The exported program is Linalg, SCF
// and memref, and the JIT compiles it for whatever machine is running, so the
// numbers are the same on arm64 and on x86. What cannot be checked here is that
// AVX2 *instructions* were emitted -- that needs an AVX2 runner, and this file
// does not pretend otherwise.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/LLKToMicro/LLKToMicro.h"
#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/WorkloadGraph.h"
#include "LLK/Runtime/MappedExecutable.h"
#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/Error.h"
#include "llvm/TargetParser/Host.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
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

/// `bfloat16` is the top half of an `f32`, which is all these chains need: the
/// operands are ones and every expected result is a power of two.
uint16_t toBf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
}

float fromBf16(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

/// The LLK source of `test/Conversion/MicroMapping/matmul_e2e.mlir`. The
/// fixture's whole point is that the kernel is *exported* rather than written
/// by hand, so the acceptance chain starts from the same place the IR fixture
/// does.
constexpr llvm::StringLiteral kMatmulSource = R"mlir(
module {
  func.func @matmul(%a: tensor<16x64xbf16>, %b: tensor<64x64xbf16>,
                    %init: tensor<16x64xbf16>) -> tensor<16x64xbf16> {
    %y = llk.matmul ins(%a, %b : tensor<16x64xbf16>, tensor<64x64xbf16>)
        outs(%init : tensor<16x64xbf16>)
        {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
        -> tensor<16x64xbf16>
    return %y : tensor<16x64xbf16>
  }
}
)mlir";

/// The LLK source of `test/Conversion/MicroMapping/swiglu_e2e.mlir`.
constexpr llvm::StringLiteral kSwigluSource = R"mlir(
module {
  func.func @swiglu(%x: tensor<16x64xbf16>, %wg: tensor<64x64xbf16>,
                    %wu: tensor<64x64xbf16>, %init: tensor<16x64xbf16>)
      -> tensor<16x64xbf16> {
    %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<16x64xbf16>,
                              tensor<64x64xbf16>, tensor<64x64xbf16>)
        outs(%init : tensor<16x64xbf16>)
        {accumulator_type = f32, activation = #llk.activation<silu>,
         math_mode = #llk.math_mode<bounded_fast>}
        -> tensor<16x64xbf16>
    return %y : tensor<16x64xbf16>
  }
}
)mlir";

mlir::DialectRegistry buildRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::scf::SCFDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect>();
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  return registry;
}

/// Runs the whole chain on `source` and returns the executable it produced:
/// export to concrete Micro, search the shipped AVX2 target, and compile the
/// plan the search selected.
llvm::Expected<std::unique_ptr<llk::MappedExecutable>>
compileChain(mlir::MLIRContext &context, llvm::StringRef sourceText,
             llvm::StringRef entrySymbol,
             llk::MappedBackend backend = llk::MappedBackend::Reference,
             llvm::StringRef scheduleDb = {}) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &context);
  if (!module)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the LLK source did not parse");

  // The kernel is exported here, not written by hand.
  mlir::PassManager pm(&context);
  if (scheduleDb.empty())
    pm.addPass(mlir::llk::createLLKToMicroPass());
  else
    pm.addPass(mlir::llk::createLLKToMicroPass(scheduleDb));
  if (mlir::failed(pm.run(*module)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the LLK-to-Micro export failed");

  mlir::Operation *kernel = nullptr;
  module->walk([&](mlir::Operation *op) {
    if (!kernel && op->getName().getStringRef() == "micro.kernel")
      kernel = op;
  });
  if (!kernel)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the export produced no micro.kernel");

  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      mlir::llk::target::avx2::createMappingTarget(LLK_SOURCE_DIR);
  if (!target)
    return target.takeError();

  llvm::Expected<mlir::llk::mapping::WorkloadGraph> graph =
      mlir::llk::mapping::extractWorkloadGraph(kernel);
  if (!graph)
    return graph.takeError();

  mlir::llk::mapping::LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "bf16";
  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  mlir::llk::mapping::CoveringSearch search(*graph, **target, context,
                                            layoutContext, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  if (!result)
    return result.takeError();
  if (result->plans.empty()) {
    std::string detail =
        "the search found no complete plan (nodesWithoutRules=" +
        std::to_string(result->frontier.nodesWithoutRules) +
        ", candidatesWithoutPlacement=" +
        std::to_string(result->frontier.candidatesWithoutPlacement) +
        ", incompatibleInstancePairs=" +
        std::to_string(result->frontier.incompatibleInstancePairs) + ")";
    for (const auto &diagnostic : result->frontier.diagnostics)
      detail += "; " + diagnostic.message;
    return llvm::createStringError(llvm::inconvertibleErrorCode(), detail);
  }

  llk::MappedCompileOptions compileOptions;
  compileOptions.entrySymbol = entrySymbol.str();
  compileOptions.backend = backend;
  llvm::Expected<llk::MappedCompilation> compiled = llk::compileMappedKernel(
      *module, **target, result->plans.front(), compileOptions);
  if (!compiled)
    return compiled.takeError();
  if (!compiled->executable)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the compilation produced no executable");
  return std::move(compiled->executable);
}

/// One all-ones bf16 buffer of `rows` x `columns`.
struct Buffer {
  std::vector<uint16_t> storage;
  MemRef2D descriptor;

  Buffer(int64_t rows, int64_t columns, float fill)
      : storage(static_cast<size_t>(rows * columns), toBf16(fill)) {
    descriptor =
        MemRef2D{storage.data(), storage.data(), 0, rows, columns, columns, 1};
  }

  llk::InvocationBuffer2D checked() const {
    return {descriptor, llk::InvocationElementType::BF16,
            storage.size() * sizeof(uint16_t)};
  }
};

std::string matmulSource(int64_t M, int64_t N, int64_t K) {
  const std::string lhs =
      "tensor<" + std::to_string(M) + "x" + std::to_string(K) + "xbf16>";
  const std::string rhs =
      "tensor<" + std::to_string(K) + "x" + std::to_string(N) + "xbf16>";
  const std::string output =
      "tensor<" + std::to_string(M) + "x" + std::to_string(N) + "xf32>";
  return "module {\n  func.func @matmul(%a: " + lhs + ", %b: " + rhs +
         ", %init: " + output + ") -> " + output +
         " {\n"
         "    %y = llk.matmul ins(%a, %b : " +
         lhs + ", " + rhs + ") outs(%init : " + output +
         ") {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>} "
         "-> " +
         output +
         "\n"
         "    return %y : " +
         output + "\n  }\n}\n";
}

void fillSignedBf16(Buffer &buffer, int64_t seed) {
  for (size_t i = 0; i < buffer.storage.size(); ++i) {
    int64_t numerator = (static_cast<int64_t>(i) * 17 + seed * 11) % 31 - 15;
    buffer.storage[i] = toBf16(static_cast<float>(numerator) / 8.0f);
  }
}

float storedBf16(const Buffer &buffer, size_t index) {
  return fromBf16(buffer.storage[index]);
}

struct FloatBuffer {
  std::vector<float> storage;
  MemRef2D descriptor;

  FloatBuffer(int64_t rows, int64_t columns, float fill)
      : storage(static_cast<size_t>(rows * columns), fill) {
    descriptor =
        MemRef2D{storage.data(), storage.data(), 0, rows, columns, columns, 1};
  }

  llk::InvocationBuffer2D checked() const {
    return {descriptor, llk::InvocationElementType::F32,
            storage.size() * sizeof(float)};
  }
};

void fillSigned(FloatBuffer &buffer, int64_t seed) {
  for (size_t i = 0; i < buffer.storage.size(); ++i) {
    int64_t numerator = (static_cast<int64_t>(i) * 13 + seed * 7) % 19 - 9;
    buffer.storage[i] = static_cast<float>(numerator) / 8.0f;
  }
}

void expectPaddedMatmul(mlir::MLIRContext &context, llk::MappedBackend backend,
                        int64_t M, int64_t N, int64_t K) {
  const std::string source = matmulSource(M, N, K);
  const std::string schedule =
      std::string(LLK_SOURCE_DIR) +
      "/test/Conversion/MicroMapping/tail_acceptance_schedule.json";
  const std::string entry = "matmul_M" + std::to_string(M) + "_N" +
                            std::to_string(N) + "_K" + std::to_string(K);
  llvm::Expected<std::unique_ptr<llk::MappedExecutable>> executable =
      compileChain(context, source, entry, backend, schedule);
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());
  const llk::KernelAbi &abi = (*executable)->abi();
  ASSERT_EQ(abi.inputs.size(), 3u);
  ASSERT_EQ(abi.outputs.size(), 1u);
  EXPECT_EQ(abi.inputs[0].shape, (std::vector<int64_t>{M, K}));
  EXPECT_EQ(abi.inputs[1].shape, (std::vector<int64_t>{K, N}));
  EXPECT_EQ(abi.inputs[2].shape, (std::vector<int64_t>{M, N}));
  EXPECT_EQ(abi.outputs[0].shape, (std::vector<int64_t>{M, N}));

  Buffer a(M, K, 0.0f);
  Buffer b(K, N, 0.0f);
  fillSignedBf16(a, 3);
  fillSignedBf16(b, 11);
  FloatBuffer init(M, N, 0.0f);
  fillSigned(init, 5);

  std::vector<float> guarded(static_cast<size_t>(M * N) + 2, -77.0f);
  MemRef2D descriptor{guarded.data(), guarded.data() + 1, 0, M, N, N, 1};
  llk::InvocationBuffer2D checkedOutput{descriptor,
                                        llk::InvocationElementType::F32,
                                        guarded.size() * sizeof(float)};
  llvm::Error error =
      (*executable)
          ->invoke({a.checked(), b.checked(), init.checked()}, {checkedOutput});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));

  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      size_t index = static_cast<size_t>(m * N + n);
      float expected = init.storage[index];
      for (int64_t k = 0; k < K; ++k) {
        size_t aIndex = static_cast<size_t>(m * K + k);
        size_t bIndex = static_cast<size_t>(k * N + n);
        expected += storedBf16(a, aIndex) * storedBf16(b, bIndex);
      }
      EXPECT_NEAR(guarded[index + 1], expected,
                  2e-4f + 2e-4f * std::abs(expected))
          << "shape " << M << "x" << N << "x" << K << " at " << m << "," << n;
    }
  }
  EXPECT_EQ(guarded.front(), -77.0f);
  EXPECT_EQ(guarded.back(), -77.0f);
}

} // namespace

TEST(MappedAcceptance, InvokesTheCompilerGeneratedMatmulAndChecksEveryElement) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::Expected<std::unique_ptr<llk::MappedExecutable>> executable =
      compileChain(context, kMatmulSource, "matmul_M16_N64_K64");
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());
  EXPECT_NE((*executable)->executionIdentity().find("backend=reference"),
            llvm::StringRef::npos);
  EXPECT_NE((*executable)->executionIdentity().find("abi="),
            llvm::StringRef::npos);
  EXPECT_NE((*executable)->executionIdentity().find("lowered-ir-sha256="),
            llvm::StringRef::npos);

  // The exported contract: both operands, the caller's initialized output,
  // and the output tile.
  const llk::KernelAbi &abi = (*executable)->abi();
  ASSERT_EQ(abi.inputs.size(), 3u);
  ASSERT_EQ(abi.outputs.size(), 1u);
  EXPECT_EQ(abi.inputs[0].shape, (std::vector<int64_t>{16, 64}));
  EXPECT_EQ(abi.inputs[1].shape, (std::vector<int64_t>{64, 64}));
  EXPECT_EQ(abi.inputs[2].shape, (std::vector<int64_t>{16, 64}));
  EXPECT_EQ(abi.outputs[0].shape, (std::vector<int64_t>{16, 64}));

  Buffer a(16, 64, 1.0f);
  Buffer b(64, 64, 1.0f);
  Buffer init(16, 64, 7.0f);
  Buffer out(16, 64, -1.0f);

  llvm::Error error =
      (*executable)
          ->invoke({a.checked(), b.checked(), init.checked()}, {out.checked()});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));

  // Every output is a full K=64 reduction of ones, so every one of the 1024
  // elements is exactly 71, including the nonzero output initializer, --
  // which is only true if the staged copies, the
  // contraction, the accumulator that the K loop threads, the narrowing
  // epilogue and the write-back all happened, in that order.
  for (size_t i = 0; i < out.storage.size(); ++i)
    EXPECT_EQ(fromBf16(out.storage[i]), 71.0f) << "element " << i;
}

TEST(MappedAcceptance, InvokesTheCompilerGeneratedSwiGLUAndChecksEveryElement) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::Expected<std::unique_ptr<llk::MappedExecutable>> executable =
      compileChain(context, kSwigluSource, "fused_swiglu_M16_N64_K64");
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());

  const llk::KernelAbi &abi = (*executable)->abi();
  ASSERT_EQ(abi.inputs.size(), 3u);
  ASSERT_EQ(abi.outputs.size(), 1u);
  EXPECT_EQ(abi.outputs[0].shape, (std::vector<int64_t>{16, 64}));

  Buffer x(16, 64, 1.0f);
  Buffer wg(64, 64, 1.0f);
  Buffer wu(64, 64, 1.0f);
  Buffer out(16, 64, -1.0f);

  llvm::Error error =
      (*executable)
          ->invoke({x.checked(), wg.checked(), wu.checked()}, {out.checked()});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));

  // The gate and the up projection each reduce K=64 ones to 64, `silu(64)` is
  // 64 to the precision of the activation, and the gating multiply gives 4096.
  // That is a number only a fused kernel that ran both projections *and* the
  // activation can produce.
  for (size_t i = 0; i < out.storage.size(); ++i)
    EXPECT_EQ(fromBf16(out.storage[i]), 4096.0f) << "element " << i;
}

TEST(MappedAcceptance, LeavesTheCallersBuffersOwnedByTheCaller) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::Expected<std::unique_ptr<llk::MappedExecutable>> executable =
      compileChain(context, kMatmulSource, "matmul_M16_N64_K64");
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());

  // The output buffer is handed over inside a larger allocation with sentinels
  // on both sides: the kernel writes the region it was given and nothing else,
  // and it does not adopt the pointer.
  const int64_t rows = 16, columns = 64;
  std::vector<uint16_t> storage(static_cast<size_t>(rows * columns) + 32,
                                toBf16(-7.0f));
  const size_t lead = 16;
  MemRef2D out{
      storage.data(), storage.data() + lead, 0, rows, columns, columns, 1};

  Buffer a(rows, columns, 1.0f);
  Buffer b(64, 64, 1.0f);
  Buffer init(rows, columns, 0.0f);

  llk::InvocationBuffer2D checkedOut{out, llk::InvocationElementType::BF16,
                                     storage.size() * sizeof(uint16_t)};
  llvm::Error error =
      (*executable)
          ->invoke({a.checked(), b.checked(), init.checked()}, {checkedOut});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));

  for (int64_t i = 0; i < rows * columns; ++i)
    EXPECT_EQ(fromBf16(storage[lead + static_cast<size_t>(i)]), 64.0f)
        << "element " << i;
  EXPECT_EQ(fromBf16(storage[lead - 1]), -7.0f);
  EXPECT_EQ(fromBf16(storage[lead + static_cast<size_t>(rows * columns)]),
            -7.0f);
}

TEST(MappedAcceptance, ExecutesPaddedTailsWithNonzeroAccumulators) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  for (const std::array<int64_t, 3> shape :
       {std::array<int64_t, 3>{5, 9, 7}, {1, 17, 3}, {17, 65, 63}})
    expectPaddedMatmul(context, llk::MappedBackend::Reference, shape[0],
                       shape[1], shape[2]);
}

TEST(MappedAcceptance, SelectedAvx2ExecutesThePaddedTailFixtures) {
  const std::string hostTriple = llvm::sys::getDefaultTargetTriple();
  if (hostTriple.rfind("x86_64-", 0) != 0)
    GTEST_SKIP() << "selected AVX2 invocation requires an x86_64 host";
  const auto hostFeatures = llvm::sys::getHostCPUFeatures();
  auto avx2 = hostFeatures.find("avx2");
  if (avx2 == hostFeatures.end() || !avx2->second)
    GTEST_SKIP() << "selected AVX2 invocation requires host AVX2";

  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  for (const std::array<int64_t, 3> shape :
       {std::array<int64_t, 3>{5, 9, 7}, {1, 17, 3}, {17, 65, 63}})
    expectPaddedMatmul(context, llk::MappedBackend::SelectedTarget, shape[0],
                       shape[1], shape[2]);
}

TEST(MappedAcceptance, SelectedAvx2BackendExecutesNumerically) {
  const std::string hostTriple = llvm::sys::getDefaultTargetTriple();
  if (hostTriple.rfind("x86_64-", 0) != 0) {
    GTEST_SKIP() << "selected AVX2 invocation requires an x86_64 host";
  }
  const auto hostFeatures = llvm::sys::getHostCPUFeatures();
  auto avx2 = hostFeatures.find("avx2");
  if (avx2 == hostFeatures.end() || !avx2->second) {
    GTEST_SKIP() << "selected AVX2 invocation requires host AVX2";
  }

  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::Expected<std::unique_ptr<llk::MappedExecutable>> executable =
      compileChain(context, kMatmulSource, "matmul_M16_N64_K64",
                   llk::MappedBackend::SelectedTarget);
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());
  EXPECT_NE((*executable)->executionIdentity().find("backend=selected-target"),
            llvm::StringRef::npos);
  EXPECT_NE((*executable)->executionIdentity().find("features=avx2"),
            llvm::StringRef::npos);

  const int64_t rows = 16, columns = 64;
  std::vector<uint16_t> output(static_cast<size_t>(rows * columns), 0);
  MemRef2D out{output.data(), output.data(), 0, rows, columns, columns, 1};
  Buffer a(rows, columns, 1.0f);
  Buffer b(64, 64, 1.0f);
  Buffer init(rows, columns, 0.0f);
  llk::InvocationBuffer2D checkedOut{out, llk::InvocationElementType::BF16,
                                     output.size() * sizeof(uint16_t)};
  llvm::Error error =
      (*executable)
          ->invoke({a.checked(), b.checked(), init.checked()}, {checkedOut});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));
  for (uint16_t value : output)
    EXPECT_EQ(fromBf16(value), 64.0f);
}
