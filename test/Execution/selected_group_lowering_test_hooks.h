//===- selected_group_lowering_test_hooks.h - Test hooks --------*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLK_TEST_EXECUTION_SELECTEDGROUPLOWERINGTESTHOOKS_H
#define LLK_TEST_EXECUTION_SELECTEDGROUPLOWERINGTESTHOOKS_H

#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Mapping/MappingTarget.h"

namespace llk::testing {
llvm::Error
lowerSelectedOperationsForTest(mlir::ModuleOp module,
                               const mlir::llk::mapping::MappingTarget &target,
                               const mlir::llk::mapping::CoveringPlan &plan,
                               MappedCompilation &compilation);
} // namespace llk::testing

#endif // LLK_TEST_EXECUTION_SELECTEDGROUPLOWERINGTESTHOOKS_H
