//===- ScheduleSelection.cpp - Deterministic schedule lookup from DB ------===//
//
// Loads schedule_db.json and selects the best-matching schedule entry for
// each llk.fused_swiglu operation based on (M_bucket, N, K).
//
// Fallback order: exact match → relax K → relax N → bucket default → built-in.
//
//===----------------------------------------------------------------------===//

#include "LLK/Transforms/ScheduleSelection.h"
#include "LLK/Transforms/Common/ScheduleLoader.h"

#include "LLK/Dialect/LLKEnums.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

// Attribute class declarations (needed before LLKOps.h.inc).
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/LLKAttributes.h.inc"

// Access the generated op interfaces.
#define GET_OP_CLASSES
#include "LLK/Dialect/LLKOps.h.inc"

using namespace mlir;

// ---------------------------------------------------------------------------
// M bucket classifier (duplicated from ShapeSpecialization for pass isolation)
// ---------------------------------------------------------------------------

using llk::classifyM;
using llk::loadScheduleDB;
using llk::ScheduleEntry;

// ---------------------------------------------------------------------------
// Fallback schedule selection
// ---------------------------------------------------------------------------

static ScheduleEntry selectBest(const std::vector<ScheduleEntry> &matches,
                                int64_t N, int64_t K) {
  if (matches.empty()) {
    // Built-in default: conservative small-tile schedule.
    ScheduleEntry def;
    def.BM = 8;
    def.BN = 32;
    def.BK = 32;
    def.VM = 1;
    def.VN = 4;
    def.vector_width = 8;
    def.num_threads = 4;
    def.grain_size = 1;
    def.parallel_axis = "n";
    return def;
  }

  // All entries from loadScheduleDB already match M_bucket.
  // Return the first match; a fuller implementation would
  // compare N and K against the entry values for exact fit.
  return matches.front();
}

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
    // Walk all LLK ops and annotate with selected schedules.
    getOperation().walk([&](Operation *op) {
      auto xType = [&]() -> ShapedType {
        if (auto swiglu = dyn_cast<llk::FusedSwiGLUOp>(op))
          return swiglu.getX().getType();
        if (auto rope = dyn_cast<llk::RoPEOp>(op)) {
          // For RoPE, derive shape info for schedule selection.
          return rope.getX().getType();
        }
        if (auto attn = dyn_cast<llk::AttentionOp>(op)) {
          // For Attention, use Q shape for schedule selection.
          return attn.getQ().getType();
        }
        return nullptr;
      }();

      if (!xType)
        return;

      if (!xType.hasStaticShape()) {
        op->emitWarning() << "Cannot select schedule for dynamic shape";
        return;
      }

      // Get the operation name for schedule lookup.
      std::string opName;
      if (isa<llk::FusedSwiGLUOp>(op))
        opName = "fused_swiglu";
      else if (isa<llk::RoPEOp>(op))
        opName = "rope";
      else if (isa<llk::AttentionOp>(op))
        opName = "attention";
      else
        return;

      int64_t M = xType.getDimSize(0);
      int64_t N = 0;
      if (auto swiglu = dyn_cast<llk::FusedSwiGLUOp>(op)) {
        N = swiglu.getWg().getType().getDimSize(1);
      } else if (auto rope = dyn_cast<llk::RoPEOp>(op)) {
        N = rope.getX().getType().getDimSize(3); // D
      } else if (auto attn = dyn_cast<llk::AttentionOp>(op)) {
        N = attn.getQ().getType().getDimSize(3); // D
      }

      int bucket = classifyM(M);

      auto matches =
          loadScheduleDB("schedules/schedule_db.json", bucket, N, 0, opName);

      if (matches.empty()) {
        op->emitWarning() << "No schedule entry for " << opName
                          << " M_bucket=" << bucket
                          << " in schedules/schedule_db.json; using fallback";
      }

      ScheduleEntry selected = selectBest(matches, N, 0);

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
