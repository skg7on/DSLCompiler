//===- MicroMappingPasses.h - Mapping entry points (design §21) -----------===//
//
// The composable, user-facing entry points into the target-independent mapping
// engine (epic #67, design §21):
//
//   * `micro-map`          -- runs the whole chain (extract -> search -> bind)
//                             and binds the best plan directly;
//   * `micro-bind-plan`    -- binds one plan selected by its stable,
//   content-derived
//                             id;
//   * `micro-verify-mapping`-- verifies already-mapped Micro-IR against a
//                             target: phase 2 of the layered verification
//                             (design §18.3), resolving the ids the binder
//                             recorded.
//
// All passes are strictly target-neutral: a target is named and loaded from
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
/// mode=<deterministic|beam|exact> beam-width=<n> top-k=<n> report=<path>
/// report-only=<bool> candidate=<sym>"`.
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
  /// When set, the pass runs the same search and writes the same report but
  /// does not bind the selected plan, leaving the module exactly as it was read
  /// (design §21: "emitting a plan report without modifying input IR"). It
  /// requires `reportPath`: a report-only run with nothing to report into would
  /// discard its own result. A search that finds no plan still fails the pass,
  /// report-only or not, so the mode cannot hide a real failure.
  bool reportOnly = false;
  /// The `micro.candidate` symbol to search at, or empty for a binding-free
  /// search. The candidate is loaded from the module (phase-4 task 1): its
  /// values pin the rule parameters of the same name, and its `layout`-kind
  /// parameter -- resolved by `kind`, never by name -- becomes the layout every
  /// rule must offer. A binding changes which plans are *legal*, so a candidate
  /// no rule can satisfy is a search failure carrying the frontier's
  /// diagnostics, never a silently different plan (design §8.3/§9.5).
  std::string candidate;
};

/// `micro-bind-plan` runs the same search and binds the one plan whose id the
/// caller names, so it carries a whole `MicroMapOptions` plus that id. A plan
/// id is a content hash, so the only way to reproduce it is to re-run the
/// search -- plans are never persisted between passes. The search options
/// (mode, beam-width, top-k) must be the ones the id was produced with, or the
/// search can order or cap the plans differently and the id will not be found.
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

/// Everything `micro-verify-mapping` needs to resolve a mapped kernel's ids.
/// Phase 2 does not search, so it takes only the target configuration -- the
/// same five keys `micro-map` loads its target with -- and no search options
/// (mode, top-k, beam-width) and no plan id. Exposed on the command line as
/// `--micro-verify-mapping="target=<name> machine=<path> layouts=<path>
/// rules=<path> emitters=<csv>"`; the design's short `machine=<path>` spelling
/// (design §21) is not enough on its own, because resolving rule, layout, and
/// emitter ids needs those registries.
struct MicroVerifyMappingOptions {
  /// Opaque target label. It is only a label -- verification never interprets
  /// it -- but it is required, exactly as `micro-map` requires it, so the two
  /// entry points take the same target configuration.
  std::string target;
  std::string machinePath;
  std::string layoutPath;
  std::string rulePath;
  /// Emitter keys the target plugin understands, comma-separated on the command
  /// line and split here.
  std::vector<std::string> emitterKeys;
};

/// Verifies the module's mapped `micro.kernel` against the target named by
/// `options`: every recorded rule, executor, memory and visibility, layout,
/// route node/link, and emitter must resolve. The pass changes nothing -- a
/// successful run is a report, not a rewrite -- and fails with a stable §22.3
/// diagnostic on the first violation.
std::unique_ptr<Pass> createMicroVerifyMappingPass();
std::unique_ptr<Pass>
createMicroVerifyMappingPass(const MicroVerifyMappingOptions &options);

} // namespace llk
} // namespace mlir

#endif // LLK_CONVERSION_MICROMAPPING_MICROMAPPINGPASSES_H
