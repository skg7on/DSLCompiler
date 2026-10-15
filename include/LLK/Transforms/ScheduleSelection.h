//===- ScheduleSelection.h - Select a persisted schedule --------*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLK_TRANSFORMS_SCHEDULESELECTION_H
#define LLK_TRANSFORMS_SCHEDULESELECTION_H

#include <memory>

namespace mlir {
class Pass;
}

namespace mlir {
namespace llk {

std::unique_ptr<mlir::Pass> createScheduleSelectionPass();

} // namespace llk
} // namespace mlir

#endif
