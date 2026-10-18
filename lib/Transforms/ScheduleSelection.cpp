//===- ScheduleSelection.cpp - Deterministic schedule lookup from DB ------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Loads schedule_db.json and selects the best-matching schedule entry for
// each registered LLK kernel based on its shared contract.
//
// Fallback order: exact match → relax K → relax N → bucket default → built-in.

#include "LLK/Transforms/ScheduleSelection.h"
#include "LLK/Transforms/Common/ScheduleLoader.h"

#include "LLK/Conversion/LLKKernel/LLKKernelLowering.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/raw_ostream.h" // IWYU pragma: keep

using namespace mlir;

// ---------------------------------------------------------------------------
// Shared schedule loader
// ---------------------------------------------------------------------------

using llk::loadScheduleDB;
using llk::ScheduleEntry;
using llk::selectBestSchedule;

// ---------------------------------------------------------------------------
// ScheduleSelectionPass
// ---------------------------------------------------------------------------

namespace {

struct ScheduleSelectionPass
    : public PassWrapper<ScheduleSelectionPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ScheduleSelectionPass)

  StringRef getArgument() const override { return "select-schedule"; }
  StringRef getDescription() const override {
    return "Select schedule from database based on shape + target";
  }

  void runOnOperation() override {
    llk::LLKKernelRegistry registry;
    llk::populateLLKKernelLowerings(registry);
    getOperation().walk([&](Operation *op) {
      if (op->getName().getDialectNamespace() != "llk")
        return;
      auto *entry = registry.lookup(op->getName().getStringRef());
      if (!entry)
        return;
      llk::KernelContract contract;
      if (failed(entry->inferContract(op, contract)))
        return;
      if (contract.mBucket < 0) {
        op->emitWarning() << "Cannot select schedule for dynamic shape";
        return;
      }
      StringRef opName = contract.workload;
      int64_t N = contract.N, K = contract.K;
      int bucket = contract.mBucket;

      auto matches =
          loadScheduleDB("schedules/schedule_db.json", bucket, N, K, opName);

      if (matches.empty()) {
        op->emitWarning() << "No schedule entry for " << opName
                          << " M_bucket=" << bucket
                          << " in schedules/schedule_db.json; using fallback";
      }

      ScheduleEntry selected = selectBestSchedule(matches, N, K);

      op->emitRemark() << "selected schedule for " << opName
                       << ": BM=" << selected.BM << " BN=" << selected.BN
                       << " BK=" << selected.BK << " VM=" << selected.VM
                       << " VN=" << selected.VN
                       << " threads=" << selected.num_threads
                       << " grain=" << selected.grain_size
                       << " axis=" << selected.parallel_axis
                       << " (M_bucket=" << bucket << ")";
    });
  }
};

} // namespace

namespace mlir {
namespace llk {

std::unique_ptr<Pass> createScheduleSelectionPass() {
  return std::make_unique<ScheduleSelectionPass>();
}

} // namespace llk
} // namespace mlir
