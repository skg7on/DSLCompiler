//===- frontend_errors.cpp - C ABI error, ownership and concurrency tests -===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// These cases pin the parts of the C contract a ctypes caller depends on:
// structured errors with stable codes, borrowed text that stays valid until the
// owner is released, output-handle initialization on every failure path, and
// serialized invocation on one handle from several threads.
//===----------------------------------------------------------------------===//

#include "LLK/CAPI/Frontend.h"

#include "gtest/gtest.h"

#include <array>
#include <atomic>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

namespace {
constexpr int64_t Rows = 8, Columns = 16, Guard = 2;
constexpr size_t Elements = static_cast<size_t>(Rows * Columns);
constexpr float GuardValue = -777.0f;

// Read the fixture with the C++ standard library: this consumer must not depend
// on LLVM, which is exactly what a ctypes caller sees.
std::string fixtureText() {
  std::ifstream file(LLK_CAPI_FIXTURE, std::ios::binary);
  EXPECT_TRUE(file.good()) << LLK_CAPI_FIXTURE;
  return std::string(std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>());
}

struct Storage {
  std::vector<float> data =
      std::vector<float>(Elements + 2 * Guard, GuardValue);
  float *payload() { return data.data() + Guard; }
};

LLKBuffer2D described(Storage &storage) {
  LLKBuffer2D buffer{};
  buffer.allocated = storage.payload();
  buffer.aligned = storage.payload();
  buffer.offset = 0;
  buffer.sizes[0] = Rows;
  buffer.sizes[1] = Columns;
  buffer.strides[0] = Columns;
  buffer.strides[1] = 1;
  buffer.element_type = LLK_ELEMENT_F32;
  buffer.allocation_bytes = Elements * sizeof(float);
  return buffer;
}

std::string errorJson(LLKError *error) {
  LLKString text = llk_frontend_error_json_v1(error);
  return text.data ? std::string(text.data, text.length) : std::string();
}

struct Handle {
  LLKExecutable *handle = nullptr;
  LLKError *error = nullptr;
  explicit Handle(const std::string &request) {
    int status = llk_frontend_compile_v1(request.data(), request.size(),
                                         &handle, &error);
    EXPECT_EQ(status, 0) << (error ? errorJson(error) : "");
    if (error) {
      llk_frontend_error_free_v1(error);
      error = nullptr;
    }
  }
  ~Handle() { llk_frontend_release_v1(handle); }
  Handle(const Handle &) = delete;
  Handle &operator=(const Handle &) = delete;
};

std::string rejected(LLKExecutable *handle, const LLKBuffer2D *inputs,
                     size_t inputCount, const LLKBuffer2D *outputs,
                     size_t outputCount) {
  LLKError *error = (LLKError *)(uintptr_t)1;
  int status = llk_frontend_invoke_v1(handle, inputs, inputCount, outputs,
                                      outputCount, &error);
  EXPECT_EQ(status, 1);
  EXPECT_NE(error, nullptr);
  EXPECT_NE(error, (LLKError *)(uintptr_t)1);
  if (!error)
    return {};
  std::string json = errorJson(error);
  llk_frontend_error_free_v1(error);
  return json;
}
} // namespace

TEST(FrontendCAPIErrorPaths, MalformedRequestReportsStructuredDiagnostic) {
  std::string request = fixtureText();
  ASSERT_FALSE(request.empty());

  // An unregistered opcode is a schema error carrying the source span.
  auto at = request.find("\"opcode\":\"program_id\"");
  ASSERT_NE(at, std::string::npos);
  std::string broken =
      request.substr(0, at) + "\"opcode\":\"not_an_opcode\"" +
      request.substr(at + std::string("\"opcode\":\"program_id\"").size());

  LLKExecutable *handle = (LLKExecutable *)(uintptr_t)1;
  LLKError *error = nullptr;
  ASSERT_EQ(
      llk_frontend_compile_v1(broken.data(), broken.size(), &handle, &error),
      1);
  EXPECT_EQ(handle, nullptr);
  ASSERT_NE(error, nullptr);
  std::string json = errorJson(error);
  EXPECT_NE(json.find("\"code\":\"invalid_frontend_schema\""),
            std::string::npos)
      << json;
  EXPECT_NE(json.find("\"file\":\"add_kernel.py\""), std::string::npos) << json;
  EXPECT_NE(json.find("\"stage\":"), std::string::npos) << json;
  llk_frontend_error_free_v1(error);

  // Truncated text never reaches compilation and never returns a handle.
  handle = (LLKExecutable *)(uintptr_t)1;
  error = (LLKError *)(uintptr_t)1;
  EXPECT_EQ(llk_frontend_compile_v1(request.data(), request.size() / 2, &handle,
                                    &error),
            1);
  EXPECT_EQ(handle, nullptr);
  EXPECT_NE(error, (LLKError *)(uintptr_t)1);
  llk_frontend_error_free_v1(error);

  // A null request text is rejected instead of being dereferenced.
  handle = (LLKExecutable *)(uintptr_t)1;
  error = nullptr;
  EXPECT_EQ(llk_frontend_compile_v1(nullptr, 8, &handle, &error), 1);
  EXPECT_EQ(handle, nullptr);
  llk_frontend_error_free_v1(error);
}

TEST(FrontendCAPIErrorPaths, InvocationRejectsEveryBufferContractViolation) {
  Handle compiled(fixtureText());
  ASSERT_NE(compiled.handle, nullptr);

  Storage a, b, out;
  std::array<LLKBuffer2D, 2> inputs{described(a), described(b)};
  std::array<LLKBuffer2D, 1> outputs{described(out)};

  auto expectRejected = [&](const char *label) {
    SCOPED_TRACE(label);
    std::string json = rejected(compiled.handle, inputs.data(), inputs.size(),
                                outputs.data(), outputs.size());
    EXPECT_NE(json.find("invalid_frontend_buffer"), std::string::npos) << json;
  };

  // Arity mismatches are checked before any descriptor is read.
  {
    SCOPED_TRACE("input arity");
    std::string json =
        rejected(compiled.handle, inputs.data(), 1, outputs.data(), 1);
    EXPECT_NE(json.find("invalid_frontend_buffer"), std::string::npos) << json;
  }
  {
    SCOPED_TRACE("output arity");
    std::string json =
        rejected(compiled.handle, inputs.data(), 2, outputs.data(), 2);
    EXPECT_NE(json.find("invalid_frontend_buffer"), std::string::npos) << json;
  }

  auto save = inputs;
  inputs[0].element_type = LLK_ELEMENT_BF16;
  expectRejected("dtype tag");
  inputs = save;

  inputs[1].sizes[0] = Rows + 1;
  expectRejected("shape");
  inputs = save;

  inputs[0].strides[0] = 1;
  inputs[0].strides[1] = Columns;
  expectRejected("strides");
  inputs = save;

  outputs[0].offset = 4;
  expectRejected("offset");
  outputs[0] = described(out);

  inputs[0].allocation_bytes = Elements * sizeof(float) - sizeof(float);
  expectRejected("allocation bytes");
  inputs = save;

  // The element tag is a fixed ABI value, not the native enum's ordinal.
  inputs[0].element_type = 0;
  expectRejected("unknown element tag");
  inputs = save;

  // Writing the first input is not representable: outputs are disjoint.
  std::string json =
      rejected(compiled.handle, inputs.data(), inputs.size(), inputs.data(), 1);
  EXPECT_NE(json.find("overlaps"), std::string::npos) << json;
}

TEST(FrontendCAPILifetime, ReportAndInspectionAreBorrowedUntilRelease) {
  std::string request = fixtureText();
  LLKExecutable *handle = nullptr;
  LLKError *error = nullptr;
  ASSERT_EQ(
      llk_frontend_compile_v1(request.data(), request.size(), &handle, &error),
      0);
  ASSERT_NE(handle, nullptr);
  ASSERT_EQ(error, nullptr);

  LLKString first{}, second{};
  ASSERT_EQ(llk_frontend_report_v1(handle, &first, &error), 0);
  ASSERT_EQ(llk_frontend_report_v1(handle, &second, &error), 0);
  EXPECT_EQ(first.data, second.data);
  EXPECT_EQ(first.length, second.length);
  EXPECT_NE(
      std::string(first.data, first.length).find("\"execution_identity\":"),
      std::string::npos);

  LLKString micro{};
  ASSERT_EQ(llk_frontend_inspect_v1(handle, "micro", &micro, &error), 0);
  EXPECT_NE(std::string(micro.data, micro.length).find("micro.kernel"),
            std::string::npos);

  // An unavailable stage reports an error and leaves the output untouched.
  LLKString empty{};
  LLKError *stageError = nullptr;
  EXPECT_EQ(llk_frontend_inspect_v1(handle, "asm", &empty, &stageError), 1);
  ASSERT_NE(stageError, nullptr);
  EXPECT_NE(errorJson(stageError).find("inspection_unavailable"),
            std::string::npos);
  llk_frontend_error_free_v1(stageError);
  EXPECT_EQ(empty.data, nullptr);
  EXPECT_EQ(empty.length, 0u);

  // The report stays valid across an invocation; release happens last.
  Storage a, b, out;
  std::array<LLKBuffer2D, 2> inputs{described(a), described(b)};
  std::array<LLKBuffer2D, 1> outputs{described(out)};
  ASSERT_EQ(llk_frontend_invoke_v1(handle, inputs.data(), inputs.size(),
                                   outputs.data(), outputs.size(), &error),
            0);
  LLKString after{};
  ASSERT_EQ(llk_frontend_report_v1(handle, &after, &error), 0);
  EXPECT_EQ(first.data, after.data);

  llk_frontend_release_v1(handle);
  // Releasing nothing is a no-op, so an already-released owner is never
  // dereferenced by mistake.
  llk_frontend_release_v1(nullptr);
  llk_frontend_error_free_v1(nullptr);
}

TEST(FrontendCAPIConcurrency, SimultaneousInvokesOnDistinctOutputs) {
  Handle compiled(fixtureText());
  ASSERT_NE(compiled.handle, nullptr);

  constexpr size_t Threads = 4;
  std::vector<Storage> inputsA(Threads), inputsB(Threads), outputs(Threads);
  std::vector<std::array<LLKBuffer2D, 2>> inputs(Threads);
  std::vector<std::array<LLKBuffer2D, 1>> results(Threads);
  std::atomic<size_t> succeeded{0}, failed{0};

  for (size_t t = 0; t != Threads; ++t) {
    auto &a = inputsA[t].data;
    auto &b = inputsB[t].data;
    for (size_t i = 0; i != Elements; ++i) {
      a[i + Guard] = float(i + 1);
      b[i + Guard] = float(257 + 2 * i);
    }
    inputs[t] = {described(inputsA[t]), described(inputsB[t])};
    results[t] = {described(outputs[t])};
  }

  std::vector<std::thread> workers;
  for (size_t t = 0; t != Threads; ++t)
    workers.emplace_back([&, t] {
      LLKError *error = nullptr;
      int status = llk_frontend_invoke_v1(compiled.handle, inputs[t].data(),
                                          inputs[t].size(), results[t].data(),
                                          results[t].size(), &error);
      if (status == 0 && !error)
        ++succeeded;
      else
        ++failed;
      llk_frontend_error_free_v1(error);
    });
  for (auto &worker : workers)
    worker.join();

  EXPECT_EQ(failed.load(), 0u);
  EXPECT_EQ(succeeded.load(), Threads);
  for (size_t t = 0; t != Threads; ++t)
    for (size_t i = 0; i != Elements; ++i)
      EXPECT_EQ(outputs[t].data[i + Guard], float(258 + 3 * i))
          << "thread " << t;
}
