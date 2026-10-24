//===- compiler.cpp - Shared frontend compilation and invocation tests ---===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "LLK/Frontend/Compiler.h"
#include "FrontendTestSupport.h"
#include "LLK/Conversion/MicroMapping/MappedTuningSession.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/TargetParser/Host.h"
#include <array>
#include <bit>
#include <cmath>
#include <limits>

#if LLK_FRONTEND_ENABLE_EXECUTION
using namespace frontend_test;
using namespace llk::frontend;
namespace native = mlir::llk;
namespace {
// A lost public output mapping, aliased internal destination, arithmetic change
// or incorrect source tile realization must change these literal results.
llvm::json::Value compileData(bool multiply = false, bool tiled = true) {
  auto value = data();
  const int64_t rows = tiled ? 4 : 8, columns = tiled ? 8 : 16;
  for (auto &tensor : *program(value).getArray("tensors")) {
    (*tensor.getAsObject())["shape"] = llvm::json::Array{8, 16};
    (*tensor.getAsObject())["strides"] = llvm::json::Array{16, 1};
  }
  auto *launch = program(value).getObject("launch");
  (*launch)["grid"] = llvm::json::Array{tiled ? 2 : 1, tiled ? 2 : 1, 1};
  (*launch)["block"] = llvm::json::Array{rows, columns, 1};
  auto *constants = program(value).getObject("constexprs");
  (*constants)["BLOCK_M"] = rows;
  (*constants)["BLOCK_N"] = columns;
  for (auto &op : operations(value)) {
    auto &o = *op.getAsObject();
    auto *results = o.getArray("results");
    for (auto &result : *results) {
      auto *shape = result.getAsObject()->getArray("shape");
      for (auto &dim : *shape) {
        if (dim.getAsInteger() == 8)
          dim = rows;
        else if (dim.getAsInteger() == 16)
          dim = columns;
      }
    }
    if (o.getString("opcode") == "range") {
      auto *attrs = o.getObject("attributes");
      (*attrs)["end"] = attrs->getInteger("end") == 8 ? rows : columns;
    }
    if (o.getString("opcode") == "constant" && !results->empty()) {
      auto id = results->front().getAsObject()->getString("id");
      if (id == "bm")
        (*o.getObject("attributes"))["value"] = rows;
      if (id == "bn")
        (*o.getObject("attributes"))["value"] = columns;
    }
    if (multiply && o.getString("opcode") == "add" &&
        results->front().getAsObject()->getString("dtype") == "f32")
      o["opcode"] = "mul";
  }
  auto *options = value.getAsObject()->getObject("options");
  (*options)["config_root"] = "/no/live/configuration/should/be/read";
  llvm::json::Object files;
  for (auto path :
       {"machines/x86-avx2-v2.yaml", "mapping/x86-avx2/layouts.llkmap",
        "mapping/x86-avx2/rules.llkmap"})
    files[path] = read(std::string(LLK_FRONTEND_SOURCE_DIR) + "/" + path);
  (*options)["config_files"] = std::move(files);
  return value;
}
struct Buffer {
  std::vector<float> storage = std::vector<float>(130, -777.f);
  llk::InvocationBuffer2D checked() {
    return {{storage.data(), storage.data() + 1, 0, 8, 16, 16, 1},
            llk::InvocationElementType::F32,
            storage.size() * sizeof(float)};
  }
};
void execute(FrontendCompilation &compilation, bool multiply) {
  Buffer a, b, out;
  constexpr float expectedAdd[] = {
      258, 261, 264, 267, 270, 273, 276, 279, 282, 285, 288, 291, 294, 297, 300,
      303, 306, 309, 312, 315, 318, 321, 324, 327, 330, 333, 336, 339, 342, 345,
      348, 351, 354, 357, 360, 363, 366, 369, 372, 375, 378, 381, 384, 387, 390,
      393, 396, 399, 402, 405, 408, 411, 414, 417, 420, 423, 426, 429, 432, 435,
      438, 441, 444, 447, 450, 453, 456, 459, 462, 465, 468, 471, 474, 477, 480,
      483, 486, 489, 492, 495, 498, 501, 504, 507, 510, 513, 516, 519, 522, 525,
      528, 531, 534, 537, 540, 543, 546, 549, 552, 555, 558, 561, 564, 567, 570,
      573, 576, 579, 582, 585, 588, 591, 594, 597, 600, 603, 606, 609, 612, 615,
      618, 621, 624, 627, 630, 633, 636, 639};
  constexpr float expectedMul[] = {
      257,   518,   783,   1052,  1325,  1602,  1883,  2168,  2457,  2750,
      3047,  3348,  3653,  3962,  4275,  4592,  4913,  5238,  5567,  5900,
      6237,  6578,  6923,  7272,  7625,  7982,  8343,  8708,  9077,  9450,
      9827,  10208, 10593, 10982, 11375, 11772, 12173, 12578, 12987, 13400,
      13817, 14238, 14663, 15092, 15525, 15962, 16403, 16848, 17297, 17750,
      18207, 18668, 19133, 19602, 20075, 20552, 21033, 21518, 22007, 22500,
      22997, 23498, 24003, 24512, 25025, 25542, 26063, 26588, 27117, 27650,
      28187, 28728, 29273, 29822, 30375, 30932, 31493, 32058, 32627, 33200,
      33777, 34358, 34943, 35532, 36125, 36722, 37323, 37928, 38537, 39150,
      39767, 40388, 41013, 41642, 42275, 42912, 43553, 44198, 44847, 45500,
      46157, 46818, 47483, 48152, 48825, 49502, 50183, 50868, 51557, 52250,
      52947, 53648, 54353, 55062, 55775, 56492, 57213, 57938, 58667, 59400,
      60137, 60878, 61623, 62372, 63125, 63882, 64643, 65408};
  for (size_t i = 0; i != 128; ++i) {
    a.storage[i + 1] = float(i + 1);
    b.storage[i + 1] = float(257 + 2 * i);
  }
  auto error = compilation.invoke({a.checked(), b.checked()}, {out.checked()});
  ASSERT_FALSE(bool(error)) << llvm::toString(std::move(error));
  for (size_t i = 0; i != 128; ++i)
    EXPECT_EQ(out.storage[i + 1], multiply ? expectedMul[i] : expectedAdd[i]);
  EXPECT_EQ(out.storage.front(), -777);
  EXPECT_EQ(out.storage.back(), -777);
  for (size_t i = 0; i != 128; ++i) {
    EXPECT_EQ(a.storage[i + 1], float(i + 1));
    EXPECT_EQ(b.storage[i + 1], float(257 + 2 * i));
  }
}
} // namespace

TEST(FrontendCompiler, FullAndTiledAddMultiplyExecuteLiteralResults) {
  for (bool tiled : {false, true})
    for (bool multiply : {false, true}) {
      auto request = parseRequest(render(compileData(multiply, tiled)));
      ASSERT_TRUE(bool(request));
      auto compiled = compileFrontend(*request);
      if (!compiled)
        FAIL() << llvm::toString(compiled.takeError());
      execute(*compiled, multiply);
      execute(*compiled,
              multiply); // New public outputs with the same owned scratch.
      auto report = llvm::json::parse(compiled->report());
      ASSERT_TRUE(bool(report));
      auto *object = report->getAsObject();
      ASSERT_NE(object, nullptr);
      EXPECT_EQ(object->getString("backend"), "reference");
      EXPECT_EQ(object->getString("source_adapter"), "linalg-elementwise-v1");
      EXPECT_EQ(object->getInteger("selected_group_count"), 0);
      EXPECT_TRUE(object->getString("execution_identity")
                      .value_or("")
                      .contains("backend=reference"));
      EXPECT_GT(object->getInteger("selected_groups_verified").value_or(0), 0);
      EXPECT_FALSE(
          object->getString("native_request_identity").value_or("").empty());
      EXPECT_EQ(object->getObject("adaptation")
                    ->getInteger("internal_destination_port"),
                2);
      EXPECT_EQ(object->getObject("search")->getInteger("retained"), 1);
      EXPECT_EQ(object->getObject("search")->getInteger("generated"), 1);
      auto contract = object->getString("workload_identity").value_or("");
      EXPECT_FALSE(contract.contains("matmul"));
      EXPECT_FALSE(contract.contains("swiglu"));
      EXPECT_EQ(compiled->nativeABI().inputs.size(), 3u);
      auto micro = compiled->inspect("micro");
      ASSERT_TRUE(bool(micro));
      EXPECT_TRUE(micro->contains("micro.kernel"));
      auto *bindings = object->getObject("source_bindings");
      ASSERT_NE(bindings, nullptr);
      EXPECT_EQ(bindings->getInteger("frontend_block_0"), tiled ? 4 : 8);
      EXPECT_EQ(bindings->getInteger("frontend_block_1"), tiled ? 8 : 16);
      EXPECT_EQ(
          object->getObject("binding_receipt")->getArray("source")->size(), 2u);
      mlir::DialectRegistry registry;
      mlir::registerAllDialects(registry);
      registry.insert<mlir::micro::MicroDialect, mlir::llk::LLKDialect>();
      mlir::MLIRContext context(registry);
      auto parsed = mlir::parseSourceString<mlir::ModuleOp>(*micro, &context);
      ASSERT_TRUE(bool(parsed));
      EXPECT_TRUE(mlir::succeeded(mlir::verify(*parsed)));
      auto unavailable = compiled->inspect("asm");
      ASSERT_FALSE(bool(unavailable));
      EXPECT_EQ(errorCode(unavailable.takeError()), "inspection_unavailable");
    }
}

TEST(FrontendCompiler,
     CheckedInvocationRejectsBadBuffersAndAliasingWithoutWrites) {
  auto request = parseRequest(render(compileData()));
  ASSERT_TRUE(bool(request));
  auto compiled = compileFrontend(*request);
  if (!compiled)
    FAIL() << llvm::toString(compiled.takeError());
  Buffer a, b, out;
  auto shortBuffer = a.checked();
  shortBuffer.allocationBytes = 1;
  for (unsigned variant = 0; variant != 4; ++variant) {
    llvm::Error error = llvm::Error::success();
    if (variant == 0)
      error = compiled->invoke({a.checked()}, {out.checked()});
    if (variant == 1)
      error = compiled->invoke({shortBuffer, b.checked()}, {out.checked()});
    if (variant == 2)
      error = compiled->invoke({a.checked(), b.checked()}, {a.checked()});
    if (variant == 3) {
      auto wrong = a.checked();
      wrong.elementType = llk::InvocationElementType::I32;
      error = compiled->invoke({wrong, b.checked()}, {out.checked()});
    }
    ASSERT_TRUE(bool(error));
    EXPECT_EQ(errorCode(std::move(error)), "invalid_frontend_buffer");
    EXPECT_TRUE(
        llvm::all_of(out.storage, [](float value) { return value == -777.f; }));
  }
  execute(*compiled, false); // Failed calls leave a reusable executable.
}

TEST(FrontendCompiler, SnapshotAndRetentionOptionsNeverFallBackSilently) {
  auto value = compileData();
  (*value.getAsObject()->getObject("options"))["config_files"] =
      llvm::json::Object{};
  auto request = parseRequest(render(value));
  ASSERT_TRUE(bool(request));
  auto compiled = compileFrontend(*request);
  ASSERT_FALSE(bool(compiled));
  EXPECT_EQ(errorCode(compiled.takeError()), "invalid_frontend_configuration");
  value = compileData();
  (*value.getAsObject()->getObject("options"))["top_k"] = 2;
  request = parseRequest(render(value));
  ASSERT_TRUE(bool(request));
  compiled = compileFrontend(*request);
  ASSERT_FALSE(bool(compiled));
  EXPECT_EQ(errorCode(compiled.takeError()), "unsupported_frontend_options");
}

TEST(FrontendCompiler, FullRequestIdentityIsStableAndIncludesSnapshotContents) {
  std::string identity;
  for (unsigned variant = 0; variant != 3; ++variant) {
    auto value = compileData();
    if (variant == 2) {
      auto *files =
          value.getAsObject()->getObject("options")->getObject("config_files");
      auto text = files->getString("machines/x86-avx2-v2.yaml")->str();
      (*files)["machines/x86-avx2-v2.yaml"] =
          text + "\n# distinct frozen request snapshot\n";
    }
    auto request = parseRequest(render(value));
    ASSERT_TRUE(bool(request));
    auto compiled = compileFrontend(*request);
    if (!compiled)
      FAIL() << llvm::toString(compiled.takeError());
    auto report = llvm::json::parse(compiled->report());
    ASSERT_TRUE(bool(report));
    auto key = report->getAsObject()->getString("native_request_identity");
    ASSERT_TRUE(bool(key));
    if (variant == 0)
      identity = key->str();
    if (variant == 1)
      EXPECT_EQ(*key, identity);
    if (variant == 2)
      EXPECT_NE(*key, identity);
    auto *capability = report->getAsObject()->getObject("capabilities");
    ASSERT_NE(capability, nullptr);
    EXPECT_EQ(capability->getInteger("version"), 2);
    EXPECT_EQ(capability->getString("abi_preparation"), "supported");
    EXPECT_EQ(capability->getString("host_compatibility"), "supported");
    EXPECT_EQ(capability->getString("invocation"), "not_checked");
  }
}

TEST(FrontendCompiler, PublicOutputFirstPreservesCheckedPortOrder) {
  auto value = compileData();
  auto *ports = program(value).getArray("tensors");
  std::swap((*ports)[0], (*ports)[2]);
  auto request = parseRequest(render(value));
  ASSERT_TRUE(bool(request));
  auto compiled = compileFrontend(*request);
  if (!compiled)
    FAIL() << llvm::toString(compiled.takeError());
  execute(*compiled, false);
  auto report = llvm::json::parse(compiled->report());
  ASSERT_TRUE(bool(report));
  auto *portsMap = report->getAsObject()
                       ->getObject("adaptation")
                       ->getArray("public_to_native");
  ASSERT_EQ(portsMap->size(), 3u);
  EXPECT_EQ((*portsMap)[0].getAsObject()->getInteger("native"), 3);
  EXPECT_EQ((*portsMap)[1].getAsObject()->getInteger("native"), 0);
  EXPECT_EQ((*portsMap)[2].getAsObject()->getInteger("native"), 1);
}

TEST(FrontendCompiler, StrictF32SignedZerosNonfinitePrecisionAndSubnormals) {
  constexpr float inf = std::numeric_limits<float>::infinity();
  constexpr float nan = std::numeric_limits<float>::quiet_NaN();
  const std::array<float, 8> lhs = {-0.f, 0.f, inf,           -inf,
                                    inf,  nan, 1.0009765625f, 0x1p-149f};
  const std::array<float, 8> rhs = {-0.f, -1.f, 2.f,         2.f,
                                    -inf, 1.f,  2.00390625f, 0x1p-149f};
  // Independently specified IEEE-f32 outcomes. NaN payload/sign is unspecified;
  // finite values (including both zero signs and subnormals) compare by bits.
  const std::array<float, 8> expectedAdd = {
      -0.f, -1.f, inf, -inf, nan, nan, 3.0048828125f, 0x1p-148f};
  const std::array<float, 8> expectedMul = {
      0.f, -0.f, inf, -inf, -inf, nan, 2.005863189697265625f, 0.f};
  const auto features = llvm::sys::getHostCPUFeatures();
  const bool selectedAvailable =
      llvm::sys::getDefaultTargetTriple().starts_with("x86_64-") &&
      features.lookup("avx2");
  for (auto backend : {"reference", "selected-target"})
    for (bool tiled : {false, true})
      for (bool multiply : {false, true}) {
        auto value = compileData(multiply, tiled);
        (*value.getAsObject()->getObject("options"))["backend"] = backend;
        auto request = parseRequest(render(value));
        ASSERT_TRUE(bool(request));
        auto compiled = compileFrontend(*request);
        if (llvm::StringRef(backend) == "selected-target" &&
            !selectedAvailable) {
          ASSERT_FALSE(bool(compiled));
          EXPECT_EQ(errorCode(compiled.takeError()),
                    "native_backend_unavailable");
          continue;
        }
        if (!compiled)
          FAIL() << llvm::toString(compiled.takeError());
        Buffer a, b, out;
        for (size_t i = 0; i != 128; ++i) {
          a.storage[i + 1] = lhs[i % lhs.size()];
          b.storage[i + 1] = rhs[i % rhs.size()];
        }
        auto error =
            compiled->invoke({a.checked(), b.checked()}, {out.checked()});
        ASSERT_FALSE(bool(error)) << llvm::toString(std::move(error));
        const auto &expected = multiply ? expectedMul : expectedAdd;
        for (size_t i = 0; i != 128; ++i) {
          const float want = expected[i % expected.size()];
          const float got = out.storage[i + 1];
          if (std::isnan(want))
            EXPECT_TRUE(std::isnan(got));
          else
            EXPECT_EQ(std::bit_cast<uint32_t>(got),
                      std::bit_cast<uint32_t>(want));
        }
        EXPECT_EQ(out.storage.front(), -777.f);
        EXPECT_EQ(out.storage.back(), -777.f);
      }
}

TEST(FrontendCompiler,
     SelectionRejectsChangedSourceBindingTargetPlanAndReceipt) {
  auto request = parseRequest(render(compileData()));
  ASSERT_TRUE(bool(request));
  mlir::MLIRContext context;
  auto imported = importProgram(*request, context);
  ASSERT_TRUE(bool(imported));
  auto target =
      native::target::avx2::createMappingTarget(LLK_FRONTEND_SOURCE_DIR);
  ASSERT_TRUE(bool(target));
  native::tuning::MappedTuningOptions options;
  options.source.sourceSymbol = imported->selection.symbol;
  options.source.adapterKey = imported->adapterKey;
  options.backend = llk::MappedBackend::Reference;
  options.mapping.mode = native::mapping::SearchMode::Deterministic;
  options.mapping.topK = 1;
  options.topK = 1;
  options.measureTop = 0;
  auto report = native::tuning::runMappedTuningSession(
      *imported->module, imported->space, imported->contract, **target,
      options);
  ASSERT_TRUE(bool(report));
  ASSERT_EQ(report->ranked.size(), 1u);
  for (unsigned variant = 0; variant != 5; ++variant) {
    auto selected = report->ranked.front();
    auto source = mlir::OwningOpRef<mlir::ModuleOp>(
        mlir::cast<mlir::ModuleOp>((*imported->module)->clone()));
    auto space = imported->space;
    auto machine = (*target)->machine();
    machine.clockHz = machine.clockHz.value_or(1) + 1;
    std::vector<std::string> keys;
    for (auto key : native::target::avx2::emitterKeys())
      keys.push_back(key.str());
    native::mapping::FileMappingTarget changedTarget(
        "x86-avx2", machine, (*target)->layouts(), (*target)->rules(), keys);
    if (variant == 0)
      source->getOperation()->setAttr("llk.changed_source",
                                      mlir::UnitAttr::get(&context));
    if (variant == 1) {
      space.params[0].choices.emplace_back(2);
      selected.ranking.candidate.values[space.params[0].name] = 2;
    }
    if (variant == 3)
      selected.plan.id ^= 1;
    if (variant == 4)
      selected.receipt.source.clear();
    auto compiled = native::tuning::compileMappedTuningSelection(
        *source, space, imported->contract,
        variant == 2 ? changedTarget : **target, selected, options);
    ASSERT_FALSE(bool(compiled)) << variant;
    EXPECT_TRUE(llvm::StringRef(llvm::toString(compiled.takeError()))
                    .contains("selection"));
  }
}

TEST(FrontendCompiler,
     SelectedTargetRunsOrReportsUnavailableHostWithoutFallback) {
  for (bool tiled : {false, true})
    for (bool multiply : {false, true}) {
      auto value = compileData(multiply, tiled);
      (*value.getAsObject()->getObject("options"))["backend"] =
          "selected-target";
      auto request = parseRequest(render(value));
      ASSERT_TRUE(bool(request));
      auto compiled = compileFrontend(*request);
      auto features = llvm::sys::getHostCPUFeatures();
      bool supported =
          llvm::sys::getDefaultTargetTriple().starts_with("x86_64-") &&
          features.lookup("avx2");
      if (!supported) {
        ASSERT_FALSE(bool(compiled));
        EXPECT_EQ(errorCode(compiled.takeError()),
                  "native_backend_unavailable");
        continue;
      }
      if (!compiled)
        FAIL() << llvm::toString(compiled.takeError());
      execute(*compiled, multiply);
      auto report = llvm::json::parse(compiled->report());
      ASSERT_TRUE(bool(report));
      EXPECT_EQ(report->getAsObject()->getString("backend"), "selected-target");
      EXPECT_GT(
          report->getAsObject()->getInteger("selected_group_count").value_or(0),
          0);
      EXPECT_EQ(report->getAsObject()->getInteger("reference_group_count"), 0);
    }
}
#endif
