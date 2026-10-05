//===- StoragePlan.h - Physical footprints and value live ranges ----------===//
//
// Task B3 (issue #67, stage B). Two contracts live here:
//
//   * `physicalFootprintFor` sizes a value the way memory actually holds it --
//     its static logical shape under a concrete layout map, plus any declared
//     padding -- not by its logical element count. A dynamic or unmodelled
//     value is an explicit *unsupported-footprint* error, never a zero and
//     never an admitted lower bound.
//
//   * `finalizeStoragePlan` builds a deterministic, dependency-ordered
//   plan-step
//     DAG from a selected covering, assigns each storage allocation the
//     interval it is live over, and checks every memory's capacity against real
//     occupancy. `computePeakStorage` summarizes the result: the largest
//     simultaneous occupancy of each memory over those intervals.
//
// The core stays target-neutral: memory identities are machine node ids, never
// target-name branches. All arithmetic is checked; overflow is a rejection.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_STORAGEPLAN_H
#define LLK_MAPPING_STORAGEPLAN_H

#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/WorkloadGraph.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <optional>
#include <vector>

namespace mlir::llk::machine {
struct MachineModel;
} // namespace mlir::llk::machine

namespace mlir::llk::mapping {

/// The physical image a value occupies: the byte count of the bounding image of
/// its layout map over its logical index space, padded as declared. `known` is
/// true only when every part was derived from static facts.
struct PhysicalFootprint {
  uint64_t bytes = 0;
  /// The per-dimension extent of the physical image. Empty for a value with no
  /// layout map (its image is its logical shape).
  llvm::SmallVector<int64_t, 4> physicalShape;
  bool known = false;
};

/// The largest logical point count a semi-affine layout image is enumerated
/// over before it is reported unsupported. A layout map containing `floordiv`,
/// `ceildiv`, or `mod` is not affine, so its bounding image cannot be read off
/// the index-space corners and must be enumerated; beyond this bound the
/// footprint is refused rather than guessed.
inline constexpr uint64_t kFootprintEnumerationLimit = 1u << 16;

/// The largest plan-step span `computePeakStorage` will summarize. Steps come
/// from `finalizeStoragePlan`, so this bound only rejects a hand-built interval
/// set with an absurd range instead of iterating forever.
inline constexpr uint64_t kMaxStorageSteps = 1u << 20;

/// The physical image of `valueType` under `map` plus `padding`.
///
/// `map` is the concrete logical-to-physical layout map (the solved map of the
/// value's layout, if any), or a null map for a value with no layout -- in
/// which case its image is its logical shape and `padding` must be empty. Each
/// `padding[j]` adds that many elements to physical dimension `j`.
///
/// An affine map's image is bounded exactly by evaluating its results at the
/// index-space corners; a semi-affine map (containing `floordiv`/`ceildiv`/
/// `mod`) is enumerated exactly when small enough. A dynamic extent, an
/// unmodelled element type, a rank mismatch, or any overflowing product is an
/// error -- never a zero and never a lower bound.
llvm::Expected<PhysicalFootprint>
physicalFootprintFor(mlir::Type valueType, mlir::AffineMap map,
                     llvm::ArrayRef<int64_t> padding = {});

/// The peak simultaneous occupancy of each memory over the live ranges of
/// `allocations`, keyed by memory node id. An allocation's storage is the root
/// of its `aliasOf` chain: aliased allocations share that root's bytes rather
/// than summing, and the alias must prove compatibility (same memory, no larger
/// than its target). Rejects a duplicate id, a broken or cyclic alias chain,
/// and any summation that overflows.
///
/// This summarizes a schedule's live ranges; it does not invent a schedule or
/// prove arbitrary concurrent orderings safe. `finalizeStoragePlan` is what
/// builds and validates the dependency-aware intervals first.
llvm::Expected<std::map<MemoryNodeId, uint64_t>>
computePeakStorage(llvm::ArrayRef<StorageAllocation> allocations);

/// Builds and validates the selected plan's storage plan.
///
/// It constructs a deterministic plan-step DAG (a compute step per placement
/// and a hop/transform/wait step per connection, ordered so every consumer
/// follows the producers and connections feeding it), assigns each value a
/// storage allocation live from its first writer through its last real reader,
/// records the synchronization decisions the movements imply, and checks each
/// memory's capacity against the occupancy the intervals produce. The occupancy
/// notes are recorded in `plan.diagnostics.warnings`.
///
/// A value's footprint is its physical image under the layout the placement
/// solved for it, multiplied by its producer's execution multiplicity. When
/// `plan.materialized` (strict executable planning) an unknown multiplicity or
/// an unsupported footprint is an error; a non-materialized (analysis) plan
/// uses a reported conservative fallback instead. Every plan memory must be
/// modeled by `machine`, and the graph must be acyclic and fully covered.
llvm::Error finalizeStoragePlan(const WorkloadGraph &graph, CoveringPlan &plan,
                                const machine::MachineModel &machine);

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_STORAGEPLAN_H
