//===- mapped_jit_test_factory.h - Test factory -----------------*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLK_TEST_EXECUTION_MAPPEDJITTESTFACTORY_H
#define LLK_TEST_EXECUTION_MAPPEDJITTESTFACTORY_H

#include "LLK/Runtime/MappedExecutable.h"

#include <utility>

namespace llk::testing {
struct TestAllocatorHooks {
  void *(*allocate)(size_t) = nullptr;
  void *(*alignedAllocate)(size_t, size_t) = nullptr;
  void (*release)(void *) = nullptr;
};

class MappedExecutableTestFactory {
public:
  static llvm::Expected<std::unique_ptr<MappedExecutable>>
  create(PreparedMappedKernel prepared, const MappedJitOptions &options,
         TestAllocatorHooks hooks);
};

inline llvm::Expected<std::unique_ptr<MappedExecutable>>
createMappedExecutableForTest(PreparedMappedKernel prepared,
                              const MappedJitOptions &options,
                              TestAllocatorHooks hooks) {
  return MappedExecutableTestFactory::create(std::move(prepared), options,
                                             hooks);
}

} // namespace llk::testing

#endif // LLK_TEST_EXECUTION_MAPPEDJITTESTFACTORY_H
