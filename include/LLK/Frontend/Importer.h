//===- Importer.h - Proven frontend import into structured source IR ------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLK_FRONTEND_IMPORTER_H
#define LLK_FRONTEND_IMPORTER_H
#include "LLK/Conversion/Workload/SourceAdapter.h"
#include "LLK/Frontend/Program.h"
#include "LLK/Machine/MachineModel.h"
#include "LLK/Transforms/Common/ScheduleLoader.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include <optional>
namespace llk::frontend {
struct PublicPortMap {
  unsigned publicPort = 0;
  unsigned nativePort = 0;
};
struct ImportedProgram {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::llk::workload::SourceSelection selection;
  mlir::llk::workload::WorkloadContract contract;
  mlir::llk::perf::SearchSpace space;
  std::string adapterKey;
  std::vector<PublicPortMap> publicToNativePorts;
  /// The native input port the compiler owns and fills itself. A zero-seeded
  /// contraction destination is the only such port today; a fused SwiGLU and a
  /// caller-accumulator contraction have none.
  std::optional<unsigned> internalDestinationPort;
};
/// The schedule a contraction of `M x N x K` is anchored on for `machine`: the
/// source is tiled with exact divisors of its extents so the static `pad`
/// boundary policy materializes no extra element, and the largest such tile
/// that stays inside the machine's staging capacity is chosen. A shape with no
/// feasible tile is an explicit error, not a silently padded copy that exceeds
/// the machine.
///
/// The machine's largest rank-three matrix fragment is the instruction shape
/// the anchor names; a machine that declares none cannot realize a contraction.
llvm::Expected<mlir::llk::ScheduleEntry>
anchorScheduleFor(const mlir::llk::machine::MachineModel &machine, int64_t M,
                  int64_t N, int64_t K, unsigned arms);

/// Proves the bounded access/effect/coverage contract before creating IR.
/// The caller's context must outlive the returned owning module.
llvm::Expected<ImportedProgram> importProgram(const FrontendRequest &request,
                                              mlir::MLIRContext &context);

/// Imports for a specific target machine. The elementwise family has no tiled
/// schedule and ignores it; a contraction anchors its tile search on it.
llvm::Expected<ImportedProgram>
importProgram(const FrontendRequest &request, mlir::MLIRContext &context,
              const mlir::llk::machine::MachineModel &machine);
} // namespace llk::frontend
#endif
