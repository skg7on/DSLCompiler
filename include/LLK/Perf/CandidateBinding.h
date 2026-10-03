//===- CandidateBinding.h - Bind a candidate into a concrete kernel -------===//
//
// Part of the M12 tuning core (issue #50).
//
// A Candidate is one point in a search space: numbers for the tile sizes,
// pipeline depth, and vector width, and names for the layout, memory path,
// owner hierarchy, fragment shape, and tail policy. Binding is the step that
// turns that point into IR the performance model can cost:
//
//   micro.search_space + Candidate -> concrete micro.kernel
//
// The bound kernel is emitted with C++ builders and carries no micro.param,
// micro.constraint, micro.objective, or symbolic choice. Every search decision
// that the concrete dialect can express lands in tile metadata -- shapes in the
// loop steps and tensor extents, layout in `#micro.layout`, memory in the tile
// memory spaces, owner in `#micro.owner` and the spatial loop maps, fragment in
// `micro.tile_partition` and the MMA shape, pipeline depth in `micro.pipeline`.
//
// The decisions are returned alongside the kernel because the schedule record
// (#50) has to persist them: the YAML output names the tile hierarchy, layout,
// memory placement, owner mapping, and fragment that the bound kernel embodies.
//
// Bindings the concrete dialect has no place for -- num_threads, grain_size,
// and the VM/VN vector tile sizes -- stay in the schedule record as candidate
// values. Nothing in this dialect represents thread count today, so a kernel
// cannot carry it; legality has already checked it against the machine.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_PERF_CANDIDATEBINDING_H
#define LLK_PERF_CANDIDATEBINDING_H

#include "LLK/Perf/Candidate.h"
#include "LLK/Perf/SearchSpace.h"

#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir {
class ModuleOp;
class Operation;
} // namespace mlir

namespace mlir::llk::perf {

/// The concrete tile decisions a candidate resolved to. These are the values
/// the bound kernel embodies and the schedule record persists.
struct BoundTileDecisions {
  /// The M bucket the workload shape falls in, as `llk::classifyM` computes it.
  int64_t mBucket = 0;

  /// Worker tile `[BM, BN, BK]`, already clamped to the problem extents.
  std::vector<int64_t> workerTile;
  /// Instruction fragment as the candidate bound it `[fM, fN, fK]`.
  std::vector<int64_t> declaredFragment;
  /// The fragment the kernel actually partitions to, clamped to divide the
  /// worker tile on every axis.
  std::vector<int64_t> fragmentShape;

  /// Resolved layout kind, applied to every materialized tile.
  std::string tileLayout;

  /// The memory path as declared, plus its resolved roles. `sourceSpace` is
  /// where external tensors live, `stagingSpace` where tiles are copied to,
  /// and `accumulatorSpace` where the accumulator is allocated.
  std::vector<std::string> memoryPath;
  std::string sourceSpace;
  std::string stagingSpace;
  std::string accumulatorSpace;

  /// Owner hierarchy: the outer scope maps the tiled axes, the fragment owner
  /// issues the instruction fragments.
  std::string outerOwner;
  std::string fragmentOwner;

  /// From the tail_policy parameter; "none" when the space declares none.
  std::string tailPolicy;
  int64_t pipelineStages = 1;
  int64_t vectorWidth = 0;
};

/// A bound kernel and the decisions it resolved to.
struct BoundKernel {
  /// The emitted `micro.kernel`. Owned by the module it was emitted into, so
  /// the pointer is only valid while that module lives.
  mlir::Operation *kernel = nullptr;
  /// Symbol of the emitted `micro.kernel`.
  std::string symbolName;
  std::string workload;
  BoundTileDecisions decisions;
};

/// Resolves `candidate` against `space` and `shape`, then appends one concrete
/// `micro.kernel` to `module`.
///
/// Fails when the candidate does not bind every parameter binding needs, when
/// a symbolic value names an unknown Micro layout/owner/memory space, when the
/// memory path is malformed, or when the problem shape is not positive. Tile
/// sizes are clamped to the problem extents exactly as the LLKToMicro export
/// clamps them, so a candidate whose tile exceeds its extent binds the extent.
/// A tile that still does not divide its extent is rejected: the export lowers
/// no masks, so a partial tile has no concrete form.
llvm::Expected<BoundKernel>
bindCandidateToMicroKernel(mlir::ModuleOp module, const SearchSpace &space,
                           const Candidate &candidate,
                           const WorkloadShape &shape);

} // namespace mlir::llk::perf

#endif // LLK_PERF_CANDIDATEBINDING_H
