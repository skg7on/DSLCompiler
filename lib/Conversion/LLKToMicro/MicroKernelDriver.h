//===- MicroKernelDriver.h - Micro kernel and space driver ------*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MICROKERNELDRIVER_H
#define LLK_MICROKERNELDRIVER_H
#include "LLK/Conversion/LLKKernel/LLKKernelLowering.h"
#include "LLK/Dialect/Micro/MicroEnums.h"
#include "LLK/Transforms/Common/ScheduleLoader.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
namespace mlir::llk::kernel_driver {
struct TilePlan {
  // Problem extent and element types. An axis that resolves statically is a
  // constant; the integer fields are filled only for those axes, so an unknown
  // extent never reaches numeric shape arithmetic.
  int64_t M = 0, N = 0, K = 0;
  int64_t paddedM = 0, paddedN = 0, paddedK = 0;
  ContractionDimensions dimensions;
  /// True when at least one axis is a symbol rather than a constant.
  bool dynamic = false;
  /// True when the plan realizes its boundary with fixed-capacity tiles and
  /// clipped accesses. Dynamic plans are always bounded; a static plan can
  /// also select it explicitly.
  bool bounded = false;
  std::string tailPolicy{"none"};
  Type inputElemType, accumulatorElemType, outputElemType;

  // Tiling.
  int64_t BM = 0, BN = 0, BK = 0;
  /// Instruction fragment, clamped to a divisor of the tile on each axis.
  int64_t fM = 0, fN = 0, fK = 0;
  /// The fragment triple as the schedule stated it, before clamping.
  SmallVector<int64_t, 3> declaredFragment;

  int64_t stages = 1;
  int64_t vectorWidth = 0;
  micro::LayoutKind layoutKind = micro::LayoutKind::row_major;
  // Owner and spatial-axis spellings are target vocabulary, carried verbatim
  // from the schedule and recorded in the IR as *symbols*. The export does not
  // resolve them to a concrete class -- it has no target loaded -- so it
  // neither normalizes nor rejects a spelling here; the owning target's machine
  // model resolves it before execution. The defaults are the legacy schedule
  // spellings the built-in conservative entry uses, preserved so an export
  // without an explicit owner_mapping has the same owner path it always had.
  std::string outerOwner = "worker";
  std::string innerOwner = "lane";
  std::string fragmentOwner = "vector_engine";
  micro::MemorySpace srcSpace = micro::MemorySpace::dram;
  micro::MemorySpace stagingSpace = micro::MemorySpace::sram;
  micro::MemorySpace accumulatorSpace = micro::MemorySpace::acc;
};
LogicalResult resolvePlan(Operation *, const ScheduleEntry &,
                          const ContractionDimensions &, TilePlan &);
/// Selects a schedule entry for `info`. A root with any symbolic extent does
/// not consult the database -- a fixed-shape entry cannot be proven legal for
/// an unknown domain -- and uses the explicit bounded baseline instead.
ScheduleEntry selectSchedule(Operation *, const KernelContract &, bool dynamic,
                             StringRef);
LogicalResult buildKernel(ModuleOp, Operation *, const TilePlan &,
                          const ScheduleEntry &, StringRef, int64_t,
                          SymbolTable &, const KernelContract &,
                          const MicroKernelFragment &);
LogicalResult buildSearchSpace(ModuleOp, Operation *, const KernelContract &,
                               const ContractionDimensions &,
                               const ScheduleEntry &);
} // namespace mlir::llk::kernel_driver
#endif
