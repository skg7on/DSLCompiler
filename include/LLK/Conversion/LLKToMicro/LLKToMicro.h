//===- LLKToMicro.h - LLK -> Micro-IR export --------------------*- C++ -*-===//
//
// Export path from the LLK scheduling pipeline into tile-centric Micro-IR.
//
// Two passes, both non-destructive: they add IR to the module and leave the
// original functions alone, so the AVX2/JIT pipeline downstream of them is
// untouched.
//
//   * LLKToMicroPass exports one concrete `micro.kernel` per supported LLK
//     root operation -- the selected schedule bound to execution tiles.
//   * LLKToMicroSearchSpacePass exports one `micro.search_space` per root
//     operation -- the legal choices around that schedule, for the tuner.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CONVERSION_LLKTOMICRO_LLKTOMICRO_H
#define LLK_CONVERSION_LLKTOMICRO_LLKTOMICRO_H

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/StringRef.h"

#include <memory>

namespace mlir {
namespace llk {

/// Lowers each supported LLK root operation to a concrete `micro.kernel`.
std::unique_ptr<mlir::Pass> createLLKToMicroPass();

/// Lowers roots using an explicit schedule database path.
std::unique_ptr<mlir::Pass> createLLKToMicroPass(llvm::StringRef scheduleDb);

/// Exports a tile-aware `micro.search_space` for each supported LLK root
/// operation, holding the legal choices around the schedule it selects.
std::unique_ptr<mlir::Pass> createLLKToMicroSearchSpacePass();

} // namespace llk
} // namespace mlir

#endif // LLK_CONVERSION_LLKTOMICRO_LLKTOMICRO_H
