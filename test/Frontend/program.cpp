//===- program.cpp - Strict schema-v1 native request parsing tests --------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "FrontendTestSupport.h"

using namespace frontend_test;
using namespace llk::frontend;

TEST(FrontendProgram, CanonicalFixturePreservesOrderedContract) {
  auto request = parseRequest(fixture());
  ASSERT_TRUE(bool(request));
  const auto &p = request->program();
  ASSERT_EQ(p.tensors.size(), 3u);
  EXPECT_EQ(p.tensors[0].name, "a");
  EXPECT_EQ(p.tensors[1].name, "b");
  EXPECT_EQ(p.tensors[2].name, "out");
  EXPECT_EQ(p.tensors[2].effect, "write");
  EXPECT_EQ(p.launch.grid, (std::array<int64_t, 3>{4, 4, 1}));
  EXPECT_EQ(request->options().backend, "reference");
  EXPECT_EQ(request->canonicalJSON(), llvm::StringRef(fixture()).trim());
  auto again = parseRequest(request->canonicalJSON());
  ASSERT_TRUE(bool(again));
  EXPECT_EQ(request->canonicalJSON(), again->canonicalJSON());
}

TEST(FrontendProgram, RejectsUnknownMissingAndDuplicateFields) {
  for (unsigned variant = 0; variant != 4; ++variant) {
    auto value = data();
    if (variant == 0)
      program(value)["version"] = 2;
    if (variant == 1)
      program(value).erase("source");
    if (variant == 2)
      (*operations(value)[0].getAsObject()->getObject("span"))["extra"] = true;
    std::string text = variant == 3 ? "{\"program\":null," + fixture().substr(1)
                                    : render(value);
    auto request = parseRequest(text);
    ASSERT_FALSE(bool(request));
    EXPECT_EQ(errorCode(request.takeError()), "invalid_frontend_schema");
  }
}

TEST(FrontendProgram, RejectsBoolFractionalOverflowAndNonfiniteNumbers) {
  for (auto text :
       {replaceOnce(fixture(), "\"axis\":0", "\"axis\":true"),
        replaceOnce(fixture(), "\"axis\":0", "\"axis\":0.0"),
        replaceOnce(fixture(), "\"BLOCK_M\":8",
                    "\"BLOCK_M\":9223372036854775808"),
        replaceOnce(fixture(), "\"value\":0.0", "\"value\":1e999")}) {
    auto request = parseRequest(text);
    ASSERT_FALSE(bool(request));
    EXPECT_EQ(errorCode(request.takeError()), "invalid_frontend_schema");
  }
}

TEST(FrontendProgram, ConfigSnapshotIsExplicitAndNeverReadFromDisk) {
  auto value = data();
  auto *options = value.getAsObject()->getObject("options");
  (*options)["config_root"] = "/this/path/does/not/exist";
  auto request = parseRequest(render(value));
  ASSERT_TRUE(bool(request));
  (*options)["config_files"] = nullptr;
  request = parseRequest(render(value));
  ASSERT_FALSE(bool(request));
  EXPECT_EQ(errorCode(request.takeError()), "invalid_frontend_schema");
}

TEST(FrontendProgram, RejectsUndefinedDuplicateMasksAndEffects) {
  for (unsigned variant = 0; variant != 6; ++variant) {
    auto value = data();
    auto &ops = operations(value);
    if (variant == 0)
      (*ops[15].getAsObject()->getArray("operands"))[0] = "missing";
    if (variant == 1)
      (*ops[0].getAsObject()->getArray("results"))[0].getAsObject()->operator[](
          "id") = "a";
    if (variant == 2)
      (*ops[15].getAsObject()->getArray("operands"))[1] = "zero";
    if (variant == 3)
      (*program(value).getArray("tensors"))[2].getAsObject()->operator[](
          "effect") = "readwrite";
    if (variant == 4)
      (*(*ops[12].getAsObject()->getArray("results"))[0]
            .getAsObject())["shape"] = llvm::json::Array{8, 1};
    if (variant == 5)
      (*ops[14].getAsObject()->getArray("operands"))[0] = "out";
    auto request = parseRequest(render(value));
    ASSERT_FALSE(bool(request));
    EXPECT_EQ(errorCode(request.takeError()), "invalid_frontend_schema");
  }
}

TEST(FrontendProgram, ParsesLivePythonCountedRegionsAndCastSites) {
  auto request =
      parseRequest(read(std::string(LLK_FRONTEND_CAPTURE_DIR) + "/loop.json"));
  ASSERT_TRUE(bool(request));
  EXPECT_EQ(request->program().numericalContract.castSites.size(), 2u);
  bool found = false;
  for (const auto &op : request->program().operations)
    if (op.opcode == "counted_loop") {
      found = true;
      ASSERT_EQ(op.regions.size(), 1u);
      EXPECT_EQ(op.regions[0].arguments.size(), 2u);
      EXPECT_EQ(op.regions[0].operations.back().opcode, "yield");
    }
  EXPECT_TRUE(found);
}

TEST(FrontendProgram, RejectsNonJsonNumericSyntaxAndEscapedDuplicateKeys) {
  for (const auto &token : {"+0", "00", "0.", "0e"}) {
    auto request = parseRequest(
        replaceOnce(fixture(), "\"axis\":0", std::string("\"axis\":") + token));
    ASSERT_FALSE(bool(request)) << token;
    EXPECT_EQ(errorCode(request.takeError()), "invalid_frontend_schema");
  }
  auto request = parseRequest("{\"\\u0070rogram\":null," + fixture().substr(1));
  ASSERT_FALSE(bool(request));
  EXPECT_EQ(errorCode(request.takeError()), "invalid_frontend_schema");
}

TEST(FrontendProgram, CanonicalKeysExactLargeIntegersAndNegativeZeroSurvive) {
  auto value = data();
  (*program(value).getObject("constexprs"))["large"] =
      int64_t(9007199254740993LL);
  auto text = render(value);
  auto request = parseRequest(text);
  ASSERT_TRUE(bool(request));
  EXPECT_TRUE(request->canonicalJSON().contains("9007199254740993"));
  auto reordered = llvm::json::parse(text);
  ASSERT_TRUE(bool(reordered));
  auto again = parseRequest(render(*reordered));
  ASSERT_TRUE(bool(again));
  EXPECT_EQ(request->canonicalJSON(), again->canonicalJSON());
  auto negative =
      parseRequest(replaceOnce(fixture(), "\"value\":0.0", "\"value\":-0.0"));
  ASSERT_TRUE(bool(negative));
  EXPECT_TRUE(negative->canonicalJSON().contains("-0.0"));
}

TEST(FrontendProgram, BoundsNestingAndRejectsNonJsonWhitespace) {
  auto deep =
      parseRequest(std::string(1000, '[') + "0" + std::string(1000, ']'));
  ASSERT_FALSE(bool(deep));
  EXPECT_EQ(errorCode(deep.takeError()), "invalid_frontend_schema");
  auto verticalTab = parseRequest(std::string(1, '\v') + fixture());
  ASSERT_FALSE(bool(verticalTab));
  EXPECT_EQ(errorCode(verticalTab.takeError()), "invalid_frontend_schema");
}

TEST(FrontendProgram, RejectsEscapingRegionsBadYieldsAndLostCastMetadata) {
  auto text = read(std::string(LLK_FRONTEND_CAPTURE_DIR) + "/loop.json");
  for (unsigned variant = 0; variant != 4; ++variant) {
    auto parsed = llvm::json::parse(text);
    ASSERT_TRUE(bool(parsed));
    auto &value = *parsed;
    llvm::json::Object *loop = nullptr;
    for (auto &op : operations(value))
      if (op.getAsObject()->getString("opcode") == "counted_loop")
        loop = op.getAsObject();
    ASSERT_NE(loop, nullptr);
    auto *body = loop->getArray("regions")->front().getAsObject();
    auto *arguments = body->getArray("arguments");
    auto *bodyOps = body->getArray("operations");
    if (variant == 0)
      (*arguments)[0].getAsObject()->operator[]("id") = "a";
    if (variant == 1)
      (*bodyOps->back().getAsObject()->getArray("operands"))[0] = "a";
    if (variant == 2) {
      auto &ops = operations(value);
      (*ops[ops.size() - 2].getAsObject()->getArray("operands"))[1] =
          *(*arguments)[1].getAsObject()->get("id");
    }
    if (variant == 3) {
      (*program(value).getObject("numerical_contract"))["cast_sites"] =
          llvm::json::Array{};
      (*value.getAsObject()->getObject("options")->getObject(
          "numerical_contract"))["cast_sites"] = llvm::json::Array{};
    }
    auto request = parseRequest(render(value));
    ASSERT_FALSE(bool(request)) << variant;
    EXPECT_EQ(errorCode(request.takeError()), "invalid_frontend_schema");
  }
}

TEST(FrontendProgram, RejectsProgramOptionsNumericalMismatch) {
  auto value = data();
  (*program(value).getObject("numerical_contract"))["fma"] = true;
  auto request = parseRequest(render(value));
  ASSERT_FALSE(bool(request));
  EXPECT_EQ(errorCode(request.takeError()), "invalid_frontend_schema");
}
