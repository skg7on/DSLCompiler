//===- Program.h - Immutable parsed frontend request contract -------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLK_FRONTEND_PROGRAM_H
#define LLK_FRONTEND_PROGRAM_H
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace llk::frontend {
struct SourceSpan {
  std::string file;
  int64_t line = 1;
  int64_t column = 0;
};
struct Diagnostic {
  std::string code;
  std::string message;
  std::optional<SourceSpan> span;
  std::string stage;
};
class FrontendError final : public llvm::ErrorInfo<FrontendError> {
public:
  static char ID;
  explicit FrontendError(Diagnostic diagnostic)
      : diagnostic(std::move(diagnostic)) {}
  void log(llvm::raw_ostream &os) const override;
  std::error_code convertToErrorCode() const override;
  Diagnostic diagnostic;
};
llvm::Error frontendError(llvm::StringRef code, llvm::Twine message,
                          std::optional<SourceSpan> span = {},
                          llvm::StringRef stage = "validate");
struct TensorSpec {
  std::string name;
  std::vector<int64_t> shape;
  std::string dtype;
  std::vector<int64_t> strides;
  std::string direction;
  std::string effect;
};
struct LaunchSpec {
  std::array<int64_t, 3> grid{};
  std::array<int64_t, 3> block{};
};
struct FrontendValue {
  std::string id;
  std::string dtype;
  std::vector<int64_t> shape;
};
struct CastSite {
  std::string valueId;
  std::string sourceDtype;
  std::string targetDtype;
  std::string rounding;
  bool operator==(const CastSite &) const = default;
};
struct NumericalContract {
  std::string storageDtype;
  std::string computeDtype;
  std::string accumulatorDtype;
  std::vector<CastSite> castSites;
  std::string rounding;
  std::string activationExpression;
  std::string activationApproximation;
  std::string reductionOrder;
  bool reassociation = false;
  bool fma = false;
  std::string subnormal;
  llvm::json::Object metadata;
};
struct FrontendRegion;
struct FrontendOperation {
  std::string opcode;
  std::vector<std::string> operands;
  std::vector<FrontendValue> results;
  llvm::json::Object attributes;
  std::vector<FrontendRegion> regions;
  SourceSpan span;
};
struct FrontendRegion {
  std::vector<FrontendValue> arguments;
  std::vector<FrontendOperation> operations;
};
struct FrontendProgram {
  std::string entry;
  std::vector<TensorSpec> tensors;
  llvm::json::Object constexprs;
  llvm::json::Object capturedConstants;
  LaunchSpec launch;
  std::vector<FrontendOperation> operations;
  NumericalContract numericalContract;
  int64_t version = 1;
  std::string source;
};
struct CompileOptions {
  std::string target;
  std::string backend;
  std::string searchMode;
  int64_t maxCandidates = 64;
  int64_t topK = 1;
  NumericalContract numericalContract;
  std::string configRoot;
  llvm::json::Object configFiles;
};
struct FrontendRequestData {
  FrontendProgram program;
  CompileOptions options;
  std::string canonicalJSON;
};
/// Only parseRequest constructs this immutable, validated request. All strings,
/// arrays and JSON metadata are owned; no view into the input text escapes.
class FrontendRequest {
public:
  const FrontendProgram &program() const { return data->program; }
  const CompileOptions &options() const { return data->options; }
  llvm::StringRef canonicalJSON() const { return data->canonicalJSON; }

private:
  explicit FrontendRequest(std::shared_ptr<const FrontendRequestData> data)
      : data(std::move(data)) {}
  std::shared_ptr<const FrontendRequestData> data;
  friend llvm::Expected<FrontendRequest> parseRequest(llvm::StringRef);
};
llvm::Expected<FrontendRequest> parseRequest(llvm::StringRef json);
} // namespace llk::frontend
#endif
