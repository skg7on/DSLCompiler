//===- contractions.cpp - Counted contractions and f32 SwiGLU -------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// PF11. A captured counted-K loop with one or two dot accumulators is recovered
// as an `llk.matmul` or `llk.fused_swiglu` root only when every shape, access,
// shared-input and effect predicate holds. The recovery is checked here both
// through the importer's contract/search-space evidence and through native
// execution of the compiler-produced kernel: reference always, selected AVX2
// when the host provides it. Unsupported patterns are rejected by name rather
// than lowered approximately.
//
//===----------------------------------------------------------------------===//
#include "FrontendTestSupport.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Frontend/Compiler.h"

#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/TargetParser/Host.h"

#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace frontend_test;
using namespace llk::frontend;
using mlir::DialectRegistry;
using mlir::MLIRContext;
using mlir::ModuleOp;

namespace {

constexpr float kGuard = -777.0F;

/// The fixture as a request, with the immutable configuration snapshot spliced
/// in the same way the compiler tests do: the request names content, not live
/// files.
llvm::json::Value withConfiguration(llvm::StringRef name) {
  auto parsed = llvm::json::parse(
      read(std::string(LLK_FRONTEND_CAPTURE_DIR) + "/" + name.str()));
  EXPECT_TRUE(bool(parsed)) << name.str();
  if (!parsed)
    return llvm::json::Value(nullptr);
  auto *options = parsed->getAsObject()->getObject("options");
  (*options)["config_root"] = "/no/live/configuration/should/be/read";
  llvm::json::Object files;
  for (auto path :
       {"machines/x86-avx2-v2.yaml", "mapping/x86-avx2/layouts.llkmap",
        "mapping/x86-avx2/rules.llkmap"})
    files[path] = read(std::string(LLK_FRONTEND_SOURCE_DIR) + "/" + path);
  (*options)["config_files"] = std::move(files);
  return std::move(*parsed);
}

std::optional<FrontendRequest> requestFor(llvm::StringRef name) {
  auto request = parseRequest(render(withConfiguration(name)));
  EXPECT_TRUE(bool(request)) << name.str();
  if (!request) {
    llvm::consumeError(request.takeError());
    return std::nullopt;
  }
  return std::move(*request);
}

/// The imported module borrows its context, so they are returned together.
struct ImportedCase {
  std::unique_ptr<MLIRContext> context;
  std::optional<ImportedProgram> imported;
  std::string error;

  bool ok() const { return bool(imported); }
};

ImportedCase importFixture(llvm::StringRef name) {
  ImportedCase result;
  auto request = requestFor(name);
  if (!request) {
    result.error = "request did not parse";
    return result;
  }
  DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::micro::MicroDialect, mlir::llk::LLKDialect>();
  result.context = std::make_unique<MLIRContext>(registry);
  auto imported = importProgram(*request, *result.context);
  if (!imported) {
    result.error = llvm::toString(imported.takeError());
    return result;
  }
  result.imported = std::move(*imported);
  return result;
}

std::string importedText(ImportedProgram &imported) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  imported.module->print(stream);
  return text;
}

std::unique_ptr<MLIRContext> makeContext() {
  DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::micro::MicroDialect, mlir::llk::LLKDialect>();
  return std::make_unique<MLIRContext>(registry);
}

unsigned countOps(ModuleOp module, llvm::StringRef name) {
  unsigned count = 0;
  module.walk([&](mlir::Operation *op) {
    count += op->getName().getStringRef() == name;
  });
  return count;
}

#if LLK_FRONTEND_ENABLE_EXECUTION
std::optional<FrontendCompilation>
compileFixture(llvm::StringRef name, llvm::StringRef backend,
               std::string *errorOut = nullptr,
               std::string *messageOut = nullptr) {
  auto value = withConfiguration(name);
  (*value.getAsObject()->getObject("options"))["backend"] = backend;
  auto request = parseRequest(render(value));
  EXPECT_TRUE(bool(request)) << name.str();
  if (!request)
    return std::nullopt;
  auto compiled = compileFrontend(*request);
  if (!compiled) {
    std::string code, message;
    llvm::handleAllErrors(
        compiled.takeError(),
        [&](const FrontendError &error) {
          code = error.diagnostic.code;
          message = error.diagnostic.code + ": " + error.diagnostic.message;
        },
        [&](const llvm::ErrorInfoBase &error) { message = error.message(); });
    if (errorOut)
      *errorOut = code;
    if (messageOut)
      *messageOut = message;
    return std::nullopt;
  }
  return std::move(*compiled);
}

bool selectedTargetAvailable() {
  const auto features = llvm::sys::getHostCPUFeatures();
  return llvm::sys::getDefaultTargetTriple().starts_with("x86_64-") &&
         features.lookup("avx2");
}

/// Caller-owned compact f32 storage with one guard word on each side.
struct Guarded2D {
  int64_t rows = 0;
  int64_t columns = 0;
  std::vector<float> storage;

  Guarded2D(int64_t rows, int64_t columns)
      : rows(rows), columns(columns),
        storage(static_cast<size_t>(rows * columns) + 2, kGuard) {}

  float &at(int64_t row, int64_t column) {
    return storage[size_t(row * columns + column) + 1];
  }
  float at(int64_t row, int64_t column) const {
    return storage[size_t(row * columns + column) + 1];
  }
  llk::InvocationBuffer2D checked() {
    return {{storage.data(), storage.data() + 1, 0, rows, columns, columns, 1},
            llk::InvocationElementType::F32,
            storage.size() * sizeof(float)};
  }
  void expectGuards() const {
    EXPECT_EQ(storage.front(), kGuard);
    EXPECT_EQ(storage.back(), kGuard);
  }
};

float inputA(int64_t index) { return float(index % 7) - 3.0F; }
float inputB(int64_t index) { return float(index % 5) - 2.0F; }

void fill(Guarded2D &buffer, float (*value)(int64_t)) {
  for (int64_t row = 0; row != buffer.rows; ++row)
    for (int64_t column = 0; column != buffer.columns; ++column)
      buffer.at(row, column) = value(row * buffer.columns + column);
}

/// The source semantics: out = A @ B, computed in double only to give the
/// comparison a reference that is not the compiler's own arithmetic.
std::vector<double> matmulReference(const Guarded2D &a, const Guarded2D &b,
                                    const Guarded2D *init) {
  const int64_t M = a.rows, K = a.columns, N = b.columns;
  std::vector<double> out(size_t(M * N), 0.0);
  for (int64_t i = 0; i != M; ++i)
    for (int64_t j = 0; j != N; ++j) {
      double sum = init ? init->at(i, j) : 0.0;
      for (int64_t k = 0; k != K; ++k)
        sum += double(a.at(i, k)) * double(b.at(k, j));
      out[size_t(i * N + j)] = sum;
    }
  return out;
}

std::vector<double> swigluReference(const Guarded2D &x, const Guarded2D &wg,
                                    const Guarded2D &wu) {
  const int64_t M = x.rows, K = x.columns, N = wg.columns;
  std::vector<double> out(size_t(M * N), 0.0);
  for (int64_t i = 0; i != M; ++i)
    for (int64_t j = 0; j != N; ++j) {
      double gate = 0.0, up = 0.0;
      for (int64_t k = 0; k != K; ++k) {
        gate += double(x.at(i, k)) * double(wg.at(k, j));
        up += double(x.at(i, k)) * double(wu.at(k, j));
      }
      const double silu = gate / (1.0 + std::exp(-gate));
      out[size_t(i * N + j)] = silu * up;
    }
  return out;
}

void expectClose(float actual, double expected) {
  const double tolerance = 1e-5 + 1e-4 * std::fabs(expected);
  EXPECT_NEAR(double(actual), expected, tolerance)
      << "actual=" << actual << " expected=" << expected;
}

void expectValues(const Guarded2D &out, const std::vector<double> &expected) {
  ASSERT_EQ(expected.size(), size_t(out.rows * out.columns));
  for (int64_t row = 0; row != out.rows; ++row)
    for (int64_t column = 0; column != out.columns; ++column)
      expectClose(out.at(row, column),
                  expected[size_t(row * out.columns + column)]);
}

std::string reportField(const FrontendCompilation &compilation,
                        llvm::StringRef field) {
  auto report = llvm::json::parse(compilation.report());
  EXPECT_TRUE(bool(report));
  if (!report)
    return {};
  auto value = report->getAsObject()->getString(field);
  EXPECT_TRUE(bool(value)) << field.str();
  return value ? value->str() : std::string();
}
#endif

} // namespace

//===----------------------------------------------------------------------===//
// Import: the recovered root and its typed evidence.
//===----------------------------------------------------------------------===//

TEST(FrontendContractions, RecoversCanonicalMatmulAsOneContractionRoot) {
  auto fixture = importFixture("matmul_32x64x16.json");
  ASSERT_TRUE(fixture.ok()) << fixture.error;
  ImportedProgram &imported = *fixture.imported;
  auto text = importedText(imported);
  auto context = makeContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
  ASSERT_TRUE(bool(module));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  EXPECT_EQ(countOps(*module, "llk.matmul"), 1u);
  EXPECT_EQ(countOps(*module, "linalg.generic"), 0u);
  EXPECT_EQ(countOps(*module, "micro.kernel"), 0u)
      << "the source root stays source-level; realization binds the kernel";

  EXPECT_EQ(imported.adapterKey, "llk-contraction-v1");
  ASSERT_EQ(imported.contract.ports.size(), 4u);
  EXPECT_EQ(imported.contract.axes.at("M").getConstant(), 32);
  EXPECT_EQ(imported.contract.axes.at("N").getConstant(), 64);
  EXPECT_EQ(imported.contract.axes.at("K").getConstant(), 16);
  // A zero accumulator has no caller buffer: the compiler owns the init port.
  EXPECT_EQ(imported.internalDestinationPort, std::optional<unsigned>(2));
  ASSERT_EQ(imported.publicToNativePorts.size(), 2u);
  EXPECT_EQ(imported.publicToNativePorts[0].nativePort, 0u);
  EXPECT_EQ(imported.publicToNativePorts[1].nativePort, 1u);
  // The search space exposes the tile axes the contraction family realizes.
  EXPECT_TRUE(imported.space.findParam("BM") != nullptr);
  EXPECT_TRUE(imported.space.findParam("BN") != nullptr);
  EXPECT_TRUE(imported.space.findParam("BK") != nullptr);
}

TEST(FrontendContractions, RecoversFusedSwiGLUOverOneSharedInput) {
  auto fixture = importFixture("swiglu_16x64x16.json");
  ASSERT_TRUE(fixture.ok()) << fixture.error;
  ImportedProgram &imported = *fixture.imported;
  auto text = importedText(imported);
  auto context = makeContext();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
  ASSERT_TRUE(bool(module));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*module)));
  EXPECT_EQ(countOps(*module, "llk.fused_swiglu"), 1u);
  EXPECT_EQ(countOps(*module, "llk.matmul"), 0u);
  EXPECT_EQ(imported.adapterKey, "llk-contraction-v1");
  // lhs + two right-hand arms + output: the fused op reads no accumulator.
  ASSERT_EQ(imported.contract.ports.size(), 4u);
  EXPECT_EQ(imported.contract.ports[1].name, "rhs0");
  EXPECT_EQ(imported.contract.ports[2].name, "rhs1");
  EXPECT_EQ(imported.internalDestinationPort, std::nullopt)
      << "every fused input is caller-owned";
  ASSERT_EQ(imported.publicToNativePorts.size(), 3u);
  for (unsigned port = 0; port != 3; ++port)
    EXPECT_EQ(imported.publicToNativePorts[port].nativePort, port);
}

TEST(FrontendContractions, PreservesACallerAccumulatorAsAPublicPort) {
  auto fixture = importFixture("matmul_acc_32x64x16.json");
  ASSERT_TRUE(fixture.ok()) << fixture.error;
  ImportedProgram &imported = *fixture.imported;
  EXPECT_EQ(imported.internalDestinationPort, std::nullopt);
  ASSERT_EQ(imported.publicToNativePorts.size(), 3u);
  EXPECT_EQ(imported.publicToNativePorts[2].nativePort, 2u)
      << "the caller's init tensor is the native init port";
  // The init port reads: a nonzero accumulator cannot be silently replaced.
  ASSERT_GE(imported.contract.ports.size(), 3u);
  EXPECT_EQ(imported.contract.ports[2].direction,
            mlir::llk::workload::PortDirection::Input);
  EXPECT_EQ(imported.contract.ports[2].effect,
            mlir::llk::workload::PortEffect::Read);
}

TEST(FrontendContractions, RejectsPatternsOutsideTheDeclaredFamily) {
  struct Rejection {
    const char *fixture;
    const char *code;
  };
  const Rejection rejections[] = {
      // Two different left operands are two contractions, not SwiGLU.
      {"swiglu_distinct_x_16x64x16.json", "unsupported_frontend_program"},
      // A rounded intermediate is a different numerical contract than the
      // fused f32 epilogue realizes.
      {"swiglu_rounded_16x64x16.json", "unsupported_numerical_contract"},
      // A second writer is not one complete tile store.
      {"matmul_extra_store_32x64x16.json", "unsupported_frontend_program"},
      // The counted loop must traverse exactly the declared K extent.
      {"matmul_wrong_extent_32x64x16.json", "unsupported_frontend_program"},
  };
  for (const Rejection &rejection : rejections) {
    SCOPED_TRACE(rejection.fixture);
    auto request = requestFor(rejection.fixture);
    ASSERT_TRUE(bool(request));
    MLIRContext context;
    auto imported = importProgram(*request, context);
    ASSERT_FALSE(bool(imported));
    EXPECT_EQ(errorCode(imported.takeError()), rejection.code);
  }
}

#if LLK_FRONTEND_ENABLE_EXECUTION
//===----------------------------------------------------------------------===//
// Execution: reference always, selected AVX2 when the host provides it.
//===----------------------------------------------------------------------===//

TEST(FrontendContractions, ReferenceExecutesCanonicalMatmul) {
  struct Shape {
    const char *fixture;
    int64_t M, N, K;
  };
  const Shape cases[] = {
      {"matmul_32x64x16.json", 32, 64, 16},
      {"matmul_7x9x5.json", 7, 9, 5},
  };
  for (const Shape &test : cases) {
    SCOPED_TRACE(test.fixture);
    std::string message;
    auto compiled =
        compileFixture(test.fixture, "reference", nullptr, &message);
    if (!compiled)
      FAIL() << test.fixture << " did not compile: " << message;
    Guarded2D a(test.M, test.K), b(test.K, test.N), out(test.M, test.N);
    fill(a, inputA);
    fill(b, inputB);
    const auto expected = matmulReference(a, b, nullptr);
    auto error = compiled->invoke({a.checked(), b.checked()}, {out.checked()});
    ASSERT_FALSE(bool(error)) << llvm::toString(std::move(error));
    expectValues(out, expected);
    a.expectGuards();
    b.expectGuards();
    out.expectGuards();
    EXPECT_EQ(reportField(*compiled, "source_adapter"), "llk-contraction-v1");
    EXPECT_EQ(reportField(*compiled, "backend"), "reference");
  }
}

TEST(FrontendContractions, ReferenceExecutesFusedSwiGLU) {
  std::string message;
  auto compiled =
      compileFixture("swiglu_16x64x16.json", "reference", nullptr, &message);
  if (!compiled)
    FAIL() << "swiglu did not compile: " << message;
  constexpr int64_t M = 16, N = 64, K = 16;
  Guarded2D x(M, K), wg(K, N), wu(K, N), out(M, N);
  fill(x, inputA);
  fill(wg, inputB);
  fill(wu, [](int64_t index) { return float(index % 3) - 1.0F; });
  const auto expected = swigluReference(x, wg, wu);
  auto error = compiled->invoke({x.checked(), wg.checked(), wu.checked()},
                                {out.checked()});
  ASSERT_FALSE(bool(error)) << llvm::toString(std::move(error));
  expectValues(out, expected);
  x.expectGuards();
  wg.expectGuards();
  wu.expectGuards();
  out.expectGuards();
  EXPECT_EQ(reportField(*compiled, "source_adapter"), "llk-contraction-v1");
  // The fused kernel exports one vector epilogue over the two accumulators.
  auto micro = compiled->inspect("micro");
  ASSERT_TRUE(bool(micro));
  EXPECT_TRUE(micro->contains("micro.mma"));
  EXPECT_TRUE(micro->contains("\"silu\""));
}

TEST(FrontendContractions, ReferencePreservesACallerAccumulator) {
  std::string message;
  auto compiled = compileFixture("matmul_acc_32x64x16.json", "reference",
                                 nullptr, &message);
  if (!compiled)
    FAIL() << "accumulator fixture did not compile: " << message;
  constexpr int64_t M = 32, N = 64, K = 16;
  Guarded2D a(M, K), b(K, N), init(M, N), out(M, N);
  fill(a, inputA);
  fill(b, inputB);
  fill(init, [](int64_t index) { return float(index % 11) * 0.25F - 1.0F; });
  const auto expected = matmulReference(a, b, &init);
  const auto initBefore = init.storage;
  auto error = compiled->invoke({a.checked(), b.checked(), init.checked()},
                                {out.checked()});
  ASSERT_FALSE(bool(error)) << llvm::toString(std::move(error));
  expectValues(out, expected);
  EXPECT_EQ(init.storage, initBefore) << "the accumulator input is read-only";
  out.expectGuards();
}

TEST(FrontendContractions, RelaxedActivationIsADifferentNumericalIdentity) {
  std::string exactMessage, relaxedMessage;
  auto exact = compileFixture("swiglu_16x64x16.json", "reference", nullptr,
                              &exactMessage);
  auto relaxed = compileFixture("swiglu_native_16x64x16.json", "reference",
                                nullptr, &relaxedMessage);
  if (!exact || !relaxed)
    FAIL() << exactMessage << relaxedMessage;
  EXPECT_NE(reportField(*exact, "workload_identity"),
            reportField(*relaxed, "workload_identity"));
  EXPECT_NE(reportField(*exact, "native_request_identity"),
            reportField(*relaxed, "native_request_identity"));

  constexpr int64_t M = 16, N = 64, K = 16;
  Guarded2D x(M, K), wg(K, N), wu(K, N), out(M, N);
  fill(x, inputA);
  fill(wg, inputB);
  fill(wu, [](int64_t index) { return float(index % 3) - 1.0F; });
  const auto expected = swigluReference(x, wg, wu);
  auto error = relaxed->invoke({x.checked(), wg.checked(), wu.checked()},
                               {out.checked()});
  ASSERT_FALSE(bool(error)) << llvm::toString(std::move(error));
  expectValues(out, expected);
  out.expectGuards();
}

TEST(FrontendContractions, NaNPropagatesThroughTheFusedEpilogue) {
  std::string message;
  auto compiled =
      compileFixture("swiglu_16x64x16.json", "reference", nullptr, &message);
  if (!compiled)
    FAIL() << "swiglu did not compile: " << message;
  constexpr int64_t M = 16, N = 64, K = 16;
  Guarded2D x(M, K), wg(K, N), wu(K, N), out(M, N);
  fill(x, inputA);
  fill(wg, inputB);
  fill(wu, [](int64_t index) { return float(index % 3) - 1.0F; });
  x.at(0, 0) = std::numeric_limits<float>::quiet_NaN();
  auto error = compiled->invoke({x.checked(), wg.checked(), wu.checked()},
                                {out.checked()});
  ASSERT_FALSE(bool(error)) << llvm::toString(std::move(error));
  EXPECT_TRUE(std::isnan(out.at(0, 0)));
  out.expectGuards();
}

TEST(FrontendContractions, NaNAndInfinityFollowTheDeclaredArithmetic) {
  // A non-finite source element must not be clamped, zeroed or skipped by the
  // contraction. Row 0 carries the non-finite value and row 1 stays finite, so
  // the check pins both the propagation and that the rest of the domain is
  // unaffected.
  struct Case {
    float value;
    bool expectNaN;
    const char *name;
  };
  const Case cases[] = {
      {std::numeric_limits<float>::infinity(), false, "inf"},
      {std::numeric_limits<float>::quiet_NaN(), true, "nan"},
  };
  for (const Case &test : cases) {
    SCOPED_TRACE(test.name);
    std::string message;
    auto compiled =
        compileFixture("matmul_32x64x16.json", "reference", nullptr, &message);
    if (!compiled)
      FAIL() << "matmul did not compile: " << message;
    constexpr int64_t M = 32, N = 64, K = 16;
    Guarded2D a(M, K), b(K, N), out(M, N);
    fill(a, [](int64_t) { return 0.0F; });
    fill(b, [](int64_t) { return 1.0F; });
    a.at(0, 0) = test.value;
    // The reference stays finite: every other row is a finite dot product.
    for (int64_t column = 0; column != N; ++column)
      b.at(0, column) = 2.0F;
    auto error = compiled->invoke({a.checked(), b.checked()}, {out.checked()});
    ASSERT_FALSE(bool(error)) << llvm::toString(std::move(error));
    if (test.expectNaN)
      EXPECT_TRUE(std::isnan(out.at(0, 0)));
    else
      EXPECT_TRUE(std::isinf(out.at(0, 0)) && out.at(0, 0) > 0.0F);
    for (int64_t column = 0; column != N; ++column)
      EXPECT_EQ(out.at(1, column), 0.0F) << column;
    out.expectGuards();
  }
}

TEST(FrontendContractions, SelectedTargetExecutesOrReportsUnavailableHost) {
  const bool available = selectedTargetAvailable();
  for (auto name : {"matmul_32x64x16.json", "swiglu_16x64x16.json"}) {
    SCOPED_TRACE(name);
    std::string error;
    auto compiled = compileFixture(name, "selected-target", &error);
    if (!available) {
      ASSERT_FALSE(bool(compiled));
      EXPECT_EQ(error, "native_backend_unavailable");
      continue;
    }
    if (!compiled)
      FAIL() << name << " failed selected compilation: " << error;
    if (std::string_view(name) == "matmul_32x64x16.json") {
      Guarded2D a(32, 16), b(16, 64), out(32, 64);
      fill(a, inputA);
      fill(b, inputB);
      const auto expected = matmulReference(a, b, nullptr);
      auto invoked =
          compiled->invoke({a.checked(), b.checked()}, {out.checked()});
      ASSERT_FALSE(bool(invoked)) << llvm::toString(std::move(invoked));
      expectValues(out, expected);
      out.expectGuards();
    } else {
      Guarded2D x(16, 16), wg(16, 64), wu(16, 64), out(16, 64);
      fill(x, inputA);
      fill(wg, inputB);
      fill(wu, [](int64_t index) { return float(index % 3) - 1.0F; });
      const auto expected = swigluReference(x, wg, wu);
      auto invoked = compiled->invoke({x.checked(), wg.checked(), wu.checked()},
                                      {out.checked()});
      ASSERT_FALSE(bool(invoked)) << llvm::toString(std::move(invoked));
      expectValues(out, expected);
      out.expectGuards();
    }
    auto report = llvm::json::parse(compiled->report());
    ASSERT_TRUE(bool(report));
    auto *object = report->getAsObject();
    EXPECT_EQ(object->getString("backend"), "selected-target");
    EXPECT_GT(object->getInteger("selected_group_count").value_or(0), 0);
    EXPECT_EQ(object->getInteger("reference_group_count"), 0);
  }
}
#endif
