//===- TritonCPUVerifier.cpp - reject unsupported GPU ops -----------------===//
//
// Runs before Stage 2. Checks every Triton IR op against a CPU-support
// whitelist. Emits actionable errors for GPU-only constructs.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/TritonToLLK/TritonCPUVerifier.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringSet.h"

using namespace mlir;

namespace {

// Ops supported on CPU. Only tt.* names are checked by the pass below; the
// non-Triton entries document the broader set of ops the CPU pipeline is
// expected to handle so the whitelist stays a single source of truth.
static const llvm::StringSet<> kSupportedOps = {
    "tt.dot",
    "tt.load",
    "tt.store",
    "tt.make_block_ptr",
    "tt.advance",
    "tt.alloc",
    "tt.async_copy",
    "tt.await",
    "tt.atomic_add",
    "tt.atomic_max",
    "tt.atomic_min",
    "tt.atomic_cas",
    "tt.get_program_id",
    "tt.make_range",
    "tt.splat",
    "tt.reduce",
    "tt.broadcast",
    "tt.trans",
    "arith.constant",
    "arith.addf",
    "arith.mulf",
    "arith.divf",
    "arith.subf",
    "arith.negf",
    "arith.select",
    "arith.truncf",
    "arith.extf",
    "arith.cmpi",
    "arith.addi",
    "arith.muli",
    "arith.divsi",
    "arith.remsi",
    "arith.ceildivsi",
    "math.exp",
    "math.exp2",
    "math.log",
    "math.log2",
    "scf.for",
    "scf.if",
    "scf.yield",
    "func.func",
    "func.return",
    "memref.alloc",
    "memref.dealloc",
    "memref.copy",
    "memref.subview",
};

struct TritonCPUVerifierPass
    : public PassWrapper<TritonCPUVerifierPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TritonCPUVerifierPass)

  StringRef getArgument() const override { return "triton-cpu-verify"; }
  StringRef getDescription() const override {
    return "Verify no unsupported GPU ops before CPU lowering";
  }

  void runOnOperation() override {
    getOperation().walk([&](Operation *op) {
      StringRef opName = op->getName().getStringRef();

      // Only check Triton dialect ops (tt.*).
      if (!opName.starts_with("tt."))
        return;

      if (!kSupportedOps.contains(opName)) {
        op->emitError("Triton op '")
            << opName << "' is unsupported on CPU. "
            << "See the CPU compatibility guide for supported operations.";
        signalPassFailure();
      }
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::llk::createTritonCPUVerifierPass() {
  return std::make_unique<TritonCPUVerifierPass>();
}
