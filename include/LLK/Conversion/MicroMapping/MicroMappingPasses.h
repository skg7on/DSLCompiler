//===- MicroMappingPasses.h - Mapping entry points (design §21) -----------===//
//
// The two composable, user-facing entry points into the target-independent
// mapping engine (epic #67, design §21):
//
//   * `micro-map`      -- runs the whole chain (extract -> search -> bind) and
//                         binds the best plan directly;
//   * `micro-bind-plan`-- binds one plan selected by its stable,
//   content-derived
//                         id.
//
// Both passes are strictly target-neutral: a target is named and loaded from
// files on disk (`target`/`machine`/`layouts`/`rules`/`emitters`), and no pass
// branch ever mentions a specific backend. The target plugin gives its own ids
// meaning, exactly as the mapping core does.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CONVERSION_MICROMAPPING_MICROMAPPINGPASSES_H
#define LLK_CONVERSION_MICROMAPPING_MICROMAPPINGPASSES_H

#include "mlir/Pass/Pass.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mlir {
namespace llk {

/// Everything `micro-map` needs to load a target and search it. The same fields
/// are exposed on the command line as `--micro-map="target=<name>
/// machine=<path> layouts=<path> rules=<path> emitters=<csv>
/// mode=<deterministic|beam|exact> top-k=<n>"`.
struct MicroMapOptions {
  /// Opaque target label, recorded on the selected plan; never interpreted by
  /// generic code.
  std::string target;
  std::string machinePath;
  std::string layoutPath;
  std::string rulePath;
  /// Emitter keys the target plugin understands, comma-separated on the
  /// command line and split here.
  std::vector<std::string> emitterKeys;
  /// `deterministic`, `beam`, or `exact`.
  std::string mode = "beam";
  unsigned topK = 8;
  unsigned beamWidth = 64;
  /// When non-empty, the pass writes the versioned JSON plan report (design
  /// §22.2) to this path. The report is metadata: it never changes the IR.
  std::string reportPath;
};

/// `micro-bind-plan` runs the same search deterministically and binds the one
/// plan whose id the caller names, so it carries a whole `MicroMapOptions` plus
/// that id. A plan id is a content hash, so the only way to reproduce it is to
/// re-run the search -- plans are never persisted between passes.
struct MicroBindPlanOptions {
  MicroMapOptions search;
  uint64_t planId = 0;
};

/// Binds the best plan produced by a mapping search onto the module's
/// `micro.kernel`.
std::unique_ptr<Pass> createMicroMapPass();
std::unique_ptr<Pass> createMicroMapPass(const MicroMapOptions &options);

/// Binds the plan with the requested stable id onto the module's
/// `micro.kernel`.
std::unique_ptr<Pass> createMicroBindPlanPass();
std::unique_ptr<Pass>
createMicroBindPlanPass(const MicroBindPlanOptions &options);

} // namespace llk
} // namespace mlir

#endif // LLK_CONVERSION_MICROMAPPING_MICROMAPPINGPASSES_H
