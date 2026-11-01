//===- tails.cpp - Bounded elementwise tails: numerical and canary proof -===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// PF10. A 33x65 (and 1x7) add/multiply launched with 8x16 blocks needs a tail:
// the grid rounds the domain up, and the declared in-bounds mask is what makes
// the rounded blocks safe. The importer proves that mask, the elementwise
// adapter realizes the interior and the static edge rectangles with bounded
// movement, and both the reference and the selected AVX2 lanes execute the
// result. Guard words on both sides of every payload catch an out-of-bounds
// write that a numerically correct output would otherwise hide.
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

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

#if LLK_FRONTEND_ENABLE_EXECUTION
using namespace frontend_test;
using namespace llk::frontend;

namespace {

constexpr float kGuard = -777.0F;

std::string captured(llvm::StringRef name) {
  return read(std::string(LLK_FRONTEND_CAPTURE_DIR) + "/" + name.str());
}

/// Splice the immutable configuration snapshot into a captured request, the
/// same way the compiler tests do: the request names content, not live files.
llvm::json::Value withConfiguration(llvm::StringRef name) {
  auto parsed = llvm::json::parse(captured(name));
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
struct GuardedBuffer {
  int64_t rows = 0;
  int64_t columns = 0;
  std::vector<float> storage;

  GuardedBuffer(int64_t rows, int64_t columns)
      : rows(rows), columns(columns),
        storage(static_cast<size_t>(rows * columns) + 2, kGuard) {}

  float &payload(int64_t index) { return storage[size_t(index) + 1]; }
  const float &payload(int64_t index) const {
    return storage[size_t(index) + 1];
  }
  llk::InvocationBuffer2D checked() {
    return {{storage.data(), storage.data() + 1, 0, rows, columns, columns, 1},
            llk::InvocationElementType::F32,
            storage.size() * sizeof(float)};
  }
};

float inputA(int64_t index) { return float(index % 7) - 3.0F; }
float inputB(int64_t index) { return float(index % 5) - 2.0F; }
float expectedValue(int64_t index, bool multiply) {
  return multiply ? inputA(index) * inputB(index)
                  : inputA(index) + inputB(index);
}

void executeTail(FrontendCompilation &compilation, bool multiply, int64_t rows,
                 int64_t columns) {
  GuardedBuffer a(rows, columns), b(rows, columns), out(rows, columns);
  for (int64_t index = 0; index != rows * columns; ++index) {
    a.payload(index) = inputA(index);
    b.payload(index) = inputB(index);
  }
  auto error = compilation.invoke({a.checked(), b.checked()}, {out.checked()});
  ASSERT_FALSE(bool(error)) << llvm::toString(std::move(error));
  for (int64_t index = 0; index != rows * columns; ++index)
    EXPECT_EQ(out.payload(index), expectedValue(index, multiply)) << index;
  // Both guards survive: the rounded blocks wrote no element outside the
  // declared domain. Inputs stay untouched.
  EXPECT_EQ(out.storage.front(), kGuard);
  EXPECT_EQ(out.storage.back(), kGuard);
  for (int64_t index = 0; index != rows * columns; ++index) {
    EXPECT_EQ(a.payload(index), inputA(index));
    EXPECT_EQ(b.payload(index), inputB(index));
  }
}

void expectRealizedBoundaries(const FrontendCompilation &compilation) {
  auto micro = compilation.inspect("micro");
  ASSERT_TRUE(bool(micro));
  EXPECT_TRUE(micro->contains("micro.kernel"));
  // The interior stays a full tile; the edge rectangles are bounded movement
  // with static valid extents, not an out-of-bounds view.
  EXPECT_TRUE(micro->contains("micro.tile_load"));
  EXPECT_TRUE(micro->contains("valid["));
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::micro::MicroDialect, mlir::llk::LLKDialect>();
  mlir::MLIRContext context(registry);
  auto parsed = mlir::parseSourceString<mlir::ModuleOp>(*micro, &context);
  ASSERT_TRUE(bool(parsed));
  EXPECT_TRUE(mlir::succeeded(mlir::verify(*parsed)));
}

void expectTailReceipt(const FrontendCompilation &compilation) {
  auto report = llvm::json::parse(compilation.report());
  ASSERT_TRUE(bool(report));
  auto *object = report->getAsObject();
  ASSERT_NE(object, nullptr);
  EXPECT_EQ(object->getString("backend"), "reference");
  EXPECT_EQ(object->getString("source_adapter"), "linalg-elementwise-v1");
  auto *bindings = object->getObject("source_bindings");
  ASSERT_NE(bindings, nullptr);
  auto policy = bindings->getString("frontend_tail_policy");
  ASSERT_TRUE(bool(policy));
  EXPECT_EQ(*policy, "masked-edge");
  auto *receipt = object->getObject("binding_receipt")->getArray("source");
  ASSERT_NE(receipt, nullptr);
  std::vector<std::string> names;
  for (const auto &entry : *receipt)
    if (auto name = entry.getAsString())
      names.push_back(name->str());
  EXPECT_NE(std::find(names.begin(), names.end(), "frontend_tail_policy"),
            names.end());
}

struct TailCase {
  const char *fixture;
  bool multiply;
  int64_t rows;
  int64_t columns;
};

const TailCase kTailCases[] = {
    {"tail_add_33x65.json", false, 33, 65},
    {"tail_mul_33x65.json", true, 33, 65},
    {"tail_add_1x7.json", false, 1, 7},
    {"tail_mul_1x7.json", true, 1, 7},
};

} // namespace

TEST(FrontendTails, ReferenceExecutesRoundedBlocksWithGuards) {
  for (const TailCase &test : kTailCases) {
    SCOPED_TRACE(test.fixture);
    std::string message;
    auto compiled =
        compileFixture(test.fixture, "reference", nullptr, &message);
    if (!compiled)
      FAIL() << test.fixture << " did not compile: " << message;
    executeTail(*compiled, test.multiply, test.rows, test.columns);
    expectRealizedBoundaries(*compiled);
    expectTailReceipt(*compiled);
    // Scratch is owned by the executable, so a second invocation is safe.
    executeTail(*compiled, test.multiply, test.rows, test.columns);
  }
}

TEST(FrontendTails, SelectedTargetRealizesEdgesOrReportsUnavailableHost) {
  const bool available = selectedTargetAvailable();
  for (const TailCase &test : kTailCases) {
    SCOPED_TRACE(test.fixture);
    std::string error;
    auto compiled = compileFixture(test.fixture, "selected-target", &error);
    if (!available) {
      ASSERT_FALSE(bool(compiled));
      EXPECT_EQ(error, "native_backend_unavailable");
      continue;
    }
    if (!compiled)
      FAIL() << test.fixture << " failed selected compilation: " << error;
    executeTail(*compiled, test.multiply, test.rows, test.columns);
    expectRealizedBoundaries(*compiled);
    auto report = llvm::json::parse(compiled->report());
    ASSERT_TRUE(bool(report));
    auto *object = report->getAsObject();
    EXPECT_EQ(object->getString("backend"), "selected-target");
    EXPECT_GT(object->getInteger("selected_group_count").value_or(0), 0);
    EXPECT_EQ(object->getInteger("reference_group_count"), 0);
  }
}

TEST(FrontendTails, RejectsUnprovedMasksAndMalformedAccesses) {
  struct Rejection {
    const char *fixture;
    const char *code;
  };
  const Rejection rejections[] = {
      // The present divisibility gate is not silently removed: a rounded
      // launch with no declared tail mask is an unsupported tail policy.
      {"tail_unmasked_33x65.json", "unsupported_tail_policy"},
      {"tail_false_mask_33x65.json", "unsupported_tail_policy"},
      {"tail_short_mask_33x65.json", "unsupported_tail_policy"},
      {"tail_wide_mask_33x65.json", "unsupported_tail_policy"},
      // An address that starts before the domain is a coverage defect, not a
      // mask-policy question.
      {"tail_negative_offset_33x65.json", "invalid_launch_coverage"},
  };
  for (const Rejection &rejection : rejections) {
    SCOPED_TRACE(rejection.fixture);
    std::string error;
    auto compiled = compileFixture(rejection.fixture, "reference", &error);
    EXPECT_FALSE(bool(compiled));
    EXPECT_EQ(error, rejection.code);
  }
}

TEST(FrontendTails, UnderCoveringGridStillFailsBeforeAnyTailPolicy) {
  auto value = withConfiguration("tail_add_33x65.json");
  auto *launch =
      value.getAsObject()->getObject("program")->getObject("launch")->getArray(
          "grid");
  (*launch)[0] = 4; // 4 * 8 = 32 < 33: the domain is not covered at all.
  auto request = parseRequest(render(value));
  ASSERT_TRUE(bool(request));
  auto compiled = compileFrontend(*request);
  ASSERT_FALSE(bool(compiled));
  EXPECT_EQ(errorCode(compiled.takeError()), "invalid_launch_coverage");
}
#endif
