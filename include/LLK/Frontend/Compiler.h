//===- Compiler.h - Owned frontend compilation and checked invocation ----===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLK_FRONTEND_COMPILER_H
#define LLK_FRONTEND_COMPILER_H
#include "LLK/Frontend/Program.h"
#include "LLK/Runtime/MappedInvocation.h"
#include <memory>

namespace llk::frontend {
/// Owns native code, immutable report/inspection text and private destination
/// storage. No MLIR context or borrowed operation escapes compilation. Calls on
/// one executable are serialized while its private destination is in use.
class FrontendCompilation {
public:
  ~FrontendCompilation();
  FrontendCompilation(FrontendCompilation &&) noexcept;
  FrontendCompilation &operator=(FrontendCompilation &&) noexcept;
  FrontendCompilation(const FrontendCompilation &) = delete;
  FrontendCompilation &operator=(const FrontendCompilation &) = delete;

  llvm::StringRef report() const;
  llvm::Expected<llvm::StringRef> inspect(llvm::StringRef stage) const;
  const KernelAbi &nativeABI() const;
  llvm::Error invoke(llvm::ArrayRef<InvocationBuffer2D> inputs,
                     llvm::ArrayRef<InvocationBuffer2D> outputs);

private:
  struct State;
  explicit FrontendCompilation(std::unique_ptr<State> state);
  std::unique_ptr<State> state;
  friend llvm::Expected<FrontendCompilation>
  compileFrontend(const FrontendRequest &request);
};

llvm::Expected<FrontendCompilation>
compileFrontend(const FrontendRequest &request);
} // namespace llk::frontend
#endif
