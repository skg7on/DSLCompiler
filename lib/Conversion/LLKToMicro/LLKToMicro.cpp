//===- LLKToMicro.cpp - LLK -> concrete micro.kernel lowering -------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Exports the LLK pipeline's semantic structure as tile-centric Micro-IR. One
// concrete `micro.kernel` is appended to the module per supported LLK root
// operation; the original functions are left untouched, so the AVX2/JIT
// pipeline downstream of this pass is unaffected.
//
// Lowering resolves the tile program in the order the Micro-IR design calls
// for:
//
//   external tensor
//     -> logical tile view          micro.tile_view
//     -> materialized memory tile   micro.tile_async_copy + micro.wait
//     -> instruction fragment       micro.tile_partition
//     -> MMA / vector fragment      micro.mma, micro.vector
//     -> write back                 micro.tile_store
//
// Two choices differ from the illustrative example in the M11 spec, and both
// come from the dialect's own verifiers rather than from preference:
//
//   * The MMA runs over the whole materialized tile, and its `shape` names the
//     `[BM, BN, BK]` instruction fragment it covers. MmaOp's verifier requires
//     operands of exactly `[M,K]`, `[K,N]` and `[M,N]`, and the cost model
//     charges `shape` as the work done -- so a sub-tile MMA would both fail
//     verification and silently under-count the kernel's flops.
//   * `micro.tile_partition` is therefore fragment metadata. The dialect
//     defines it as a logical, zero-cost op, and it is emitted only where the
//     instruction fragment is genuinely smaller than the tile it was cut from.
//
// `micro.pipeline` sits inside the K loop rather than around it because that is
// the position MicroDAG reads as loop pipelining; with one stage the two
// spellings are equivalent.

#include "LLK/Conversion/LLKToMicro/LLKToMicro.h"
#include "LLK/Conversion/MicroMapping/TileExtent.h"
#include "MicroKernelDriver.h"

#include "LLK/Dialect/LLKEnums.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Dialect/Micro/MicroEnums.h"
#include "LLK/Dialect/Micro/MicroHelpers.h"
#include "LLK/Transforms/Common/ScheduleLoader.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <iterator>
#include <limits>

// Micro attribute, type, and op declarations.
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

using namespace mlir;
using llk::ScheduleEntry;

using namespace mlir::llk::kernel_driver;
namespace {
LogicalResult inferMicroContract(const llk::LLKKernelRegistry &registry,
                                 Operation *root, llk::KernelContract &info) {
  auto *entry = registry.lookup(root->getName().getStringRef());
  if (!entry || !entry->hasMicroLowering())
    return failure();
  llk::MicroKernelFragment fragment;
  if (failed(entry->inferContract(root, info)))
    return failure();
  return entry->describeMicroKernel(root, info, fragment);
}
struct LLKToMicroPass
    : public PassWrapper<LLKToMicroPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLKToMicroPass)

  LLKToMicroPass() = default;
  explicit LLKToMicroPass(StringRef scheduleDbPath) {
    scheduleDb = scheduleDbPath.str();
  }
  LLKToMicroPass(const LLKToMicroPass &other) : PassWrapper(other) {}

  Option<std::string> scheduleDb{
      *this, "schedule-db",
      llvm::cl::desc(
          "Schedule database the export reads its tile choices from. A path "
          "that cannot be read falls back to the built-in schedule."),
      llvm::cl::init("schedules/schedule_db.json")};
  Option<std::string> target{
      *this, "target",
      llvm::cl::desc("Target recorded on every generated micro.kernel"),
      llvm::cl::init("x86-avx2-cpu")};

  StringRef getArgument() const override { return "llk-to-micro"; }
  StringRef getDescription() const override {
    return "Lower LLK root operations to concrete tile-centric micro.kernel IR";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // This pass builds its output programmatically rather than parsing it, so
    // every dialect it emits has to be loaded explicitly: a context loads a
    // dialect on demand, and an export from LLK starts with no micro, tensor,
    // or arith operations in the input to trigger that.
    MLIRContext *context = module.getContext();
    if (!context->getOrLoadDialect<micro::MicroDialect>() ||
        !context->getOrLoadDialect<tensor::TensorDialect>() ||
        !context->getOrLoadDialect<arith::ArithDialect>()) {
      module.emitError()
          << "the micro, tensor, and arith dialects must be registered in "
             "this context";
      signalPassFailure();
      return;
    }

    llk::LLKKernelRegistry registry;
    llk::populateLLKKernelLowerings(registry);
    SmallVector<Operation *> roots;
    module.walk([&](Operation *op) {
      if (auto *entry = registry.lookup(op->getName().getStringRef());
          entry && entry->hasMicroLowering())
        roots.push_back(op);
    });
    if (roots.empty())
      return;

    SymbolTable symbols(module);
    llvm::DenseMap<Operation *, uint64_t> ordinals;
    for (Operation *root : roots) {
      Operation *function = root->getParentOp();
      while (function && function->getName().getStringRef() != "func.func")
        function = function->getParentOp();
      if (!function) {
        root->emitError("supported root is not nested in a symbol table");
        signalPassFailure();
        return;
      }
      auto symbol = SymbolTable::getSymbolName(function);
      if (!symbol) {
        root->emitError("containing function has no symbol name");
        signalPassFailure();
        return;
      }
      llk::KernelContract info;
      if (failed(inferMicroContract(registry, root, info))) {
        signalPassFailure();
        return;
      }
      ScheduleEntry schedule =
          selectSchedule(root, info, scheduleDb.getValue());
      TilePlan validatedPlan;
      validatedPlan.inputElemType = info.inputElemType;
      validatedPlan.accumulatorElemType = info.accumulatorElemType;
      validatedPlan.outputElemType = info.outputElemType;
      if (failed(resolvePlan(root, schedule, info.M, info.N, info.K,
                             validatedPlan))) {
        signalPassFailure();
        return;
      }
      auto exported = exportMicroKernelFromSchedule(
          module, symbol.getValue(), ordinals[function]++, schedule);
      if (!exported) {
        root->emitError() << llvm::toString(exported.takeError());
        signalPassFailure();
        return;
      }
      (*exported)->setAttr("target",
                           StringAttr::get(context, target.getValue()));
    }
  }
};

/// Exports the legal choices around the selected schedule. The pass runs on
/// the same root operations the concrete export lowers, and reads the schedule
/// through the same loader, so the search space always contains the schedule
/// the concrete kernel was built from.
struct LLKToMicroSearchSpacePass
    : public PassWrapper<LLKToMicroSearchSpacePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LLKToMicroSearchSpacePass)

  LLKToMicroSearchSpacePass() = default;
  LLKToMicroSearchSpacePass(const LLKToMicroSearchSpacePass &other)
      : PassWrapper(other) {}

  Option<std::string> scheduleDb{
      *this, "schedule-db",
      llvm::cl::desc(
          "Schedule database whose selected entry anchors the search space. A "
          "path that cannot be read falls back to the built-in schedule."),
      llvm::cl::init("schedules/schedule_db.json")};

  StringRef getArgument() const override { return "llk-to-micro-search-space"; }
  StringRef getDescription() const override {
    return "Export tile-aware micro.search_space IR from the selected schedule";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // The search space is built programmatically rather than parsed, so the
    // dialect it emits has to be loaded explicitly.
    MLIRContext *context = module.getContext();
    if (!context->getOrLoadDialect<micro::MicroDialect>()) {
      module.emitError()
          << "the micro dialect must be registered in this context";
      signalPassFailure();
      return;
    }

    llk::LLKKernelRegistry registry;
    llk::populateLLKKernelLowerings(registry);
    SmallVector<Operation *> roots;
    module.walk([&](Operation *op) {
      if (auto *entry = registry.lookup(op->getName().getStringRef());
          entry && entry->hasMicroLowering())
        roots.push_back(op);
    });
    if (roots.empty())
      return;

    for (Operation *root : roots) {
      llk::KernelContract info;
      if (failed(inferMicroContract(registry, root, info))) {
        signalPassFailure();
        return;
      }
      ScheduleEntry schedule =
          selectSchedule(root, info, scheduleDb.getValue());
      if (failed(buildSearchSpace(module, root, info, schedule))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

namespace mlir {
namespace llk {

llvm::Expected<Operation *>
exportMicroKernelFromSchedule(ModuleOp module, StringRef sourceSymbol,
                              uint64_t sourceRootOrdinal,
                              const ScheduleEntry &schedule) {
  auto fail = [](const llvm::Twine &message) -> llvm::Expected<Operation *> {
    return llvm::make_error<llvm::StringError>(message.str(),
                                               llvm::inconvertibleErrorCode());
  };
  if (!module || sourceSymbol.empty())
    return fail("a module and non-empty source function symbol are required");
  Operation *function = SymbolTable::lookupSymbolIn(module, sourceSymbol);
  if (!function || function->getName().getStringRef() != "func.func")
    return fail("unknown source function '" + sourceSymbol + "'");

  llk::LLKKernelRegistry registry;
  llk::populateLLKKernelLowerings(registry);
  SmallVector<Operation *> roots;
  function->walk([&](Operation *op) {
    if (auto *entry = registry.lookup(op->getName().getStringRef());
        entry && entry->hasMicroLowering())
      roots.push_back(op);
  });
  if (sourceRootOrdinal >= roots.size())
    return fail("source function @" + sourceSymbol + " has " +
                Twine(roots.size()) + " supported roots; ordinal " +
                Twine(sourceRootOrdinal) + " is out of range");

  Operation *root = roots[sourceRootOrdinal];
  llk::KernelContract info;
  if (failed(inferMicroContract(registry, root, info)))
    return fail("selected source root is not supported");
  TilePlan plan;
  plan.inputElemType = info.inputElemType;
  plan.accumulatorElemType = info.accumulatorElemType;
  plan.outputElemType = info.outputElemType;
  if (failed(resolvePlan(root, schedule, info.M, info.N, info.K, plan)))
    return fail("selected schedule is not valid for the source root");

  // The checks above are read-only. Build only after exact selection and full
  // plan validation have succeeded, so an invalid ordinal/schedule leaves the
  // supplied clone byte-identical.
  if (!module.getContext()->getOrLoadDialect<micro::MicroDialect>() ||
      !module.getContext()->getOrLoadDialect<tensor::TensorDialect>() ||
      !module.getContext()->getOrLoadDialect<arith::ArithDialect>())
    return fail("the micro, tensor, and arith dialects must be registered");
  SymbolTable symbols(module);
  llk::MicroKernelFragment fragment;
  if (failed(registry.lookup(root->getName().getStringRef())
                 ->describeMicroKernel(root, info, fragment)))
    return fail("selected source root is not supported");
  if (failed(buildKernel(module, root, plan, schedule, "x86-avx2-cpu",
                         info.mBucket, symbols, info, fragment)))
    return fail("failed to emit concrete Micro-IR kernel");
  Operation *result = nullptr;
  module.walk([&](micro::KernelOp kernel) { result = kernel.getOperation(); });
  if (!result)
    return fail("concrete Micro-IR export emitted no kernel");
  return result;
}

std::unique_ptr<Pass> createLLKToMicroPass() {
  return std::make_unique<LLKToMicroPass>();
}

std::unique_ptr<Pass> createLLKToMicroPass(StringRef scheduleDb) {
  return std::make_unique<LLKToMicroPass>(scheduleDb);
}

std::unique_ptr<Pass> createLLKToMicroSearchSpacePass() {
  return std::make_unique<LLKToMicroSearchSpacePass>();
}

} // namespace llk
} // namespace mlir
