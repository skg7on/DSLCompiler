//===- FrontendTestSupport.h - Typed frontend fixture helpers -------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLK_TEST_FRONTEND_SUPPORT_H
#define LLK_TEST_FRONTEND_SUPPORT_H

#include "LLK/Frontend/Importer.h"
#include "LLK/Frontend/Program.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "gtest/gtest.h"
#include <string>

namespace frontend_test {
inline std::string read(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  EXPECT_TRUE(bool(buffer)) << path.str();
  return buffer ? (*buffer)->getBuffer().str() : std::string();
}
inline std::string fixture() { return read(LLK_FRONTEND_FIXTURE); }
inline llvm::json::Value data() {
  auto parsed = llvm::json::parse(fixture());
  EXPECT_TRUE(bool(parsed));
  return parsed ? std::move(*parsed) : llvm::json::Value(nullptr);
}
inline llvm::json::Object &program(llvm::json::Value &data) {
  return *data.getAsObject()->getObject("program");
}
inline llvm::json::Array &operations(llvm::json::Value &data) {
  return *program(data).getArray("operations");
}
inline std::string render(const llvm::json::Value &data) {
  return llvm::formatv("{0}", data).str();
}
inline std::string errorCode(llvm::Error error) {
  std::string code;
  llvm::handleAllErrors(
      std::move(error),
      [&](const llk::frontend::FrontendError &error) {
        code = error.diagnostic.code;
      },
      [&](const llvm::ErrorInfoBase &error) {
        ADD_FAILURE() << error.message();
      });
  return code;
}
inline std::string replaceOnce(std::string text, llvm::StringRef from,
                               llvm::StringRef to) {
  size_t at = text.find(from.str());
  EXPECT_NE(at, std::string::npos);
  if (at != std::string::npos)
    text.replace(at, from.size(), to.str());
  return text;
}
} // namespace frontend_test
#endif
