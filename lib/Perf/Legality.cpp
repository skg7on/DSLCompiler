//===- Legality.cpp - Machine-aware candidate legality --------------------===//
//
// Part of the M12 tuning core (issue #49). See Legality.h.
//
// Each rule below reads the parameters it needs and compares them against the
// machine. Integer roles are conventional parameter names (BM/BN/BK,
// vector_width, pipeline_stages, prefetch_distance, num_threads); symbolic
// roles are found through the dialect `kind` on the parameter, which is what
// that kind is for. A rule never asks for a parameter the export does not
// declare; when an optional input is absent the rule degrades to its
// machine-only core rather than guessing.
//
// The rules deliberately overlap the simulator's diagnostics, because they
// answer different questions. Legality asks "could this ever be bound?" before
// any IR exists; the simulator asks "did this particular kernel fit?" after.
// Where they disagree, the simulator is authoritative.
//
//===----------------------------------------------------------------------===//

#include "LLK/Perf/Legality.h"

#include "LLK/Dialect/Micro/MicroEnums.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::llk::perf {

using llvm::StringRef;

namespace {

LegalityResult illegal(ConstraintKind kind, const llvm::Twine &detail) {
  return LegalityResult{false,
                        (stringifyConstraintKind(kind) + ": " + detail).str()};
}

LegalityResult legal() { return LegalityResult{true, ""}; }

/// Bytes per element for a micro dtype name, or nullopt when it is not a dtype
/// the dialect models.
std::optional<int64_t> dtypeBytes(StringRef name) {
  std::optional<micro::DType> dtype = micro::symbolizeDType(name);
  if (!dtype)
    return std::nullopt;
  switch (*dtype) {
  case micro::DType::f32:
    return 4;
  case micro::DType::f16:
  case micro::DType::bf16:
    return 2;
  case micro::DType::i32:
    return 4;
  case micro::DType::i8:
    return 1;
  }
  return std::nullopt;
}

/// Parses an `MxNxK` fragment shape, requiring all three extents positive.
std::optional<std::array<int64_t, 3>> parseShapeTriple(StringRef text) {
  llvm::SmallVector<StringRef, 3> parts;
  text.split(parts, 'x');
  if (parts.size() != 3)
    return std::nullopt;

  std::array<int64_t, 3> shape{};
  for (size_t i = 0; i < parts.size(); ++i) {
    int64_t value = 0;
    if (parts[i].getAsInteger(10, value) || value <= 0)
      return std::nullopt;
    shape[i] = value;
  }
  return shape;
}

std::string tripleText(llvm::ArrayRef<int64_t> shape) {
  return (llvm::Twine(shape[0]) + "x" + llvm::Twine(shape[1]) + "x" +
          llvm::Twine(shape[2]))
      .str();
}

/// The largest fragment an engine can issue, measured by volume. This is the
/// fragment the cost model divides an execution tile by, so legality uses the
/// same one.
const std::array<int64_t, 3> &
largestTileShape(const MatrixEngineModel &engine) {
  const std::array<int64_t, 3> *best = &engine.tileShapes.front();
  auto volume = [](const std::array<int64_t, 3> &shape) {
    return shape[0] * shape[1] * shape[2];
  };
  for (const std::array<int64_t, 3> &shape : engine.tileShapes)
    if (volume(shape) > volume(*best))
      best = &shape;
  return *best;
}

/// The bound value of the parameter declared with `kind`, resolved through the
/// space so a rule can find "the layout" without knowing its name.
std::optional<StringRef> symbolOfKind(const SearchSpace &space,
                                      const Candidate &candidate,
                                      StringRef kind) {
  const SearchParam *param = space.findParamOfKind(kind);
  if (!param)
    return std::nullopt;
  return candidate.symbol(param->name);
}

/// True when the candidate selects masked tails, which makes tile divisibility
/// unnecessary.
bool isMasked(const SearchSpace &space, const Candidate &candidate) {
  std::optional<StringRef> policy =
      symbolOfKind(space, candidate, "tail_policy");
  return policy && *policy == "mask";
}

/// Reads required integer parameters, remembering the first that is missing so
/// a rule can report it instead of dividing by zero.
struct IntResolver {
  const Candidate &candidate;
  std::string error;

  std::optional<int64_t> get(StringRef name) {
    std::optional<int64_t> value = candidate.integer(name);
    if (!value && error.empty())
      error = ("candidate does not bind parameter '" + name + "'").str();
    return value;
  }
};

/// How many weight tiles one K step stages for `workload`. The fused SwiGLU
/// contraction reads two weight matrices (gate and up); every other root the
/// M11 export lowers, matmul included, reads one. The workload name is the
/// search space's own `workload` attribute, not a guess from the parameters.
int weightTileCount(StringRef workload) {
  return workload == "fused_swiglu" ? 2 : 1;
}

/// How many accumulator tiles are live for `workload`. A fused contraction
/// accumulates gate and up side by side; a plain matmul accumulates one.
int accumulatorCount(StringRef workload) {
  return workload == "fused_swiglu" ? 2 : 1;
}

/// Bytes of the tiles one K step stages: the input tile plus each weight tile
/// the workload reads. Nullopt when either dtype is not modeled.
std::optional<uint64_t> stagedTileBytes(const SearchSpace &space,
                                        const WorkloadShape &shape, int64_t BM,
                                        int64_t BN, int64_t BK) {
  std::optional<int64_t> inputBytes = dtypeBytes(shape.inputDType);
  std::optional<int64_t> weightBytes = dtypeBytes(shape.weightDType);
  if (!inputBytes || !weightBytes)
    return std::nullopt;
  return uint64_t(BM) * BK * *inputBytes +
         uint64_t(weightTileCount(space.workload)) * uint64_t(BK) * BN *
             *weightBytes;
}

bool nestedUnder(const MachineModel &machine, StringRef inner,
                 StringRef outer) {
  const OwnerModel *current = machine.findOwner(inner);
  llvm::StringSet<> visited;
  while (current && current->parent) {
    if (*current->parent == outer)
      return true;
    if (!visited.insert(*current->parent).second)
      break; // a cycle would otherwise spin; verifyMachineModel rejects these
    current = machine.findOwner(*current->parent);
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Rules
//===----------------------------------------------------------------------===//

LegalityResult checkSramCapacity(const SearchSpace &space,
                                 const Candidate &candidate,
                                 const WorkloadShape &shape,
                                 const MachineModel &machine) {
  ConstraintKind kind = ConstraintKind::SramCapacity;
  const MemoryLevelModel *sram = machine.findMemory("sram");
  if (!sram)
    return illegal(kind, "machine '" + machine.name +
                             "' does not model memory 'sram'");

  IntResolver resolver{candidate, {}};
  std::optional<int64_t> BM = resolver.get("BM");
  std::optional<int64_t> BN = resolver.get("BN");
  std::optional<int64_t> BK = resolver.get("BK");
  if (!resolver.error.empty())
    return illegal(kind, resolver.error);

  std::optional<uint64_t> tileBytes =
      stagedTileBytes(space, shape, *BM, *BN, *BK);
  if (!tileBytes)
    return illegal(kind, "input or weight dtype is not a micro dtype");

  // A multi-stage pipeline double-buffers the staged tiles.
  int64_t stages =
      std::max<int64_t>(1, candidate.integer("pipeline_stages").value_or(1));
  uint64_t factor = stages > 1 ? 2 : 1;
  uint64_t required = factor * *tileBytes;
  uint64_t limit =
      static_cast<uint64_t>(sram->capacityBytes * kSramUtilizationLimit);
  if (required > limit)
    return illegal(kind, "requires " + llvm::Twine(required) +
                             " bytes but sram holds " +
                             llvm::Twine(sram->capacityBytes) +
                             " bytes (limit " + llvm::Twine(limit) + ")");
  return legal();
}

LegalityResult checkAccCapacity(const SearchSpace &space,
                                const Candidate &candidate,
                                const WorkloadShape &shape,
                                const MachineModel &machine) {
  ConstraintKind kind = ConstraintKind::AccCapacity;
  const MemoryLevelModel *acc = machine.findMemory("acc");
  if (!acc)
    return illegal(kind, "machine '" + machine.name +
                             "' does not model memory 'acc'");

  std::optional<int64_t> accBytes = dtypeBytes(shape.accumulatorDType);
  if (!accBytes)
    return illegal(kind, "accumulator dtype '" + shape.accumulatorDType +
                             "' is not a micro dtype");

  IntResolver resolver{candidate, {}};
  std::optional<int64_t> BM = resolver.get("BM");
  std::optional<int64_t> BN = resolver.get("BN");
  if (!resolver.error.empty())
    return illegal(kind, resolver.error);

  // Every accumulator the workload keeps live must fit at once.
  uint64_t required =
      uint64_t(accumulatorCount(space.workload)) * *BM * *BN * *accBytes;
  if (required > acc->capacityBytes)
    return illegal(kind, "requires " + llvm::Twine(required) +
                             " bytes but acc holds " +
                             llvm::Twine(acc->capacityBytes) + " bytes");
  return legal();
}

LegalityResult checkMmaCompatible(const SearchSpace &space,
                                  const Candidate &candidate,
                                  const WorkloadShape &shape,
                                  const MachineModel &machine) {
  ConstraintKind kind = ConstraintKind::MmaCompatible;
  if (machine.matrixEngines.empty())
    return illegal(kind,
                   "machine '" + machine.name + "' declares no matrix engine");

  // Any engine that accepts the dtype pair can run the contraction; the first
  // matching one is used, so a machine is not rejected because its *first*
  // engine happens to have a different dtype mix.
  const MatrixEngineModel *engine = nullptr;
  for (const MatrixEngineModel &candidateEngine : machine.matrixEngines)
    if (llvm::is_contained(candidateEngine.inputDTypes, shape.inputDType) &&
        llvm::is_contained(candidateEngine.accumulatorDTypes,
                           shape.accumulatorDType)) {
      engine = &candidateEngine;
      break;
    }
  if (!engine) {
    const MatrixEngineModel &first = machine.matrixEngines.front();
    if (!llvm::is_contained(first.inputDTypes, shape.inputDType))
      return illegal(kind, "engine '" + first.name +
                               "' does not support input dtype '" +
                               shape.inputDType + "' (supported: " +
                               llvm::join(first.inputDTypes, ", ") + ")");
    return illegal(kind, "no matrix engine supports accumulator dtype '" +
                             shape.accumulatorDType + "' (engine '" +
                             first.name + "' supports: " +
                             llvm::join(first.accumulatorDTypes, ", ") + ")");
  }

  if (isMasked(space, candidate))
    return legal();

  IntResolver resolver{candidate, {}};
  std::optional<int64_t> BM = resolver.get("BM");
  std::optional<int64_t> BN = resolver.get("BN");
  std::optional<int64_t> BK = resolver.get("BK");
  if (!resolver.error.empty())
    return illegal(kind, resolver.error);

  const std::array<int64_t, 3> &fragment = largestTileShape(*engine);
  if (*BM % fragment[0] != 0 || *BN % fragment[1] != 0 ||
      *BK % fragment[2] != 0)
    return illegal(kind, "tile " + tripleText({*BM, *BN, *BK}) +
                             " is not a multiple of engine '" + engine->name +
                             "' fragment " + tripleText(fragment) +
                             " and tail_policy is not 'mask'");
  return legal();
}

LegalityResult checkMappingExtent(const SearchSpace &,
                                  const Candidate &candidate,
                                  const WorkloadShape &,
                                  const MachineModel &machine) {
  ConstraintKind kind = ConstraintKind::MappingExtent;
  std::optional<int64_t> threads = candidate.integer("num_threads");
  if (!threads)
    return illegal(kind, "candidate does not bind parameter 'num_threads'");
  if (*threads < 1)
    return illegal(kind, "num_threads must be positive");
  // The spec's second condition -- the spatial map's worker extent -- is not a
  // search parameter in the current export, so only the thread count is bound.
  if (uint64_t(*threads) > machine.workerThreads)
    return illegal(kind, "num_threads " + llvm::Twine(*threads) +
                             " exceeds machine '" + machine.name +
                             "' worker_threads " +
                             llvm::Twine(machine.workerThreads));
  return legal();
}

LegalityResult checkTailSupported(const SearchSpace &space,
                                  const Candidate &candidate,
                                  const WorkloadShape &shape,
                                  const MachineModel &) {
  ConstraintKind kind = ConstraintKind::TailSupported;
  if (isMasked(space, candidate))
    return legal();

  IntResolver resolver{candidate, {}};
  std::optional<int64_t> BM = resolver.get("BM");
  std::optional<int64_t> BN = resolver.get("BN");
  std::optional<int64_t> BK = resolver.get("BK");
  if (!resolver.error.empty())
    return illegal(kind, resolver.error);

  if (shape.M % *BM != 0 || shape.N % *BN != 0 || shape.K % *BK != 0)
    return illegal(kind, "shape " + tripleText({shape.M, shape.N, shape.K}) +
                             " is not divisible by tile " +
                             tripleText({*BM, *BN, *BK}) +
                             " and tail_policy is not 'mask'");
  return legal();
}

LegalityResult checkVectorWidthSupported(const SearchSpace &,
                                         const Candidate &candidate,
                                         const WorkloadShape &shape,
                                         const MachineModel &machine) {
  ConstraintKind kind = ConstraintKind::VectorWidthSupported;
  std::optional<int64_t> width = candidate.integer("vector_width");
  if (!width)
    return illegal(kind, "candidate does not bind parameter 'vector_width'");
  if (*width < 1)
    return illegal(kind, "vector_width must be positive");

  if (machine.vectorEngines.empty())
    return illegal(kind,
                   "machine '" + machine.name + "' declares no vector engine");

  const VectorEngineModel *best = nullptr;
  int64_t lanes = 0;
  for (const VectorEngineModel &engine : machine.vectorEngines) {
    auto it = engine.lanes.find(shape.inputDType);
    if (it != engine.lanes.end() && it->second > lanes) {
      lanes = it->second;
      best = &engine;
    }
  }
  if (!best)
    return illegal(kind, "no vector engine supports dtype '" +
                             shape.inputDType + "'");
  if (*width > lanes)
    return illegal(kind, "vector_width " + llvm::Twine(*width) +
                             " exceeds engine '" + best->name + "' lanes " +
                             llvm::Twine(lanes) + " for dtype '" +
                             shape.inputDType + "'");
  return legal();
}

LegalityResult checkLayoutSupported(const SearchSpace &space,
                                    const Candidate &candidate,
                                    const WorkloadShape &,
                                    const MachineModel &machine) {
  ConstraintKind kind = ConstraintKind::LayoutSupported;
  std::optional<StringRef> layout = symbolOfKind(space, candidate, "layout");
  if (!layout)
    return legal(); // the space declares no layout choice to check

  // The layout must be materializable somewhere in the target's memory system.
  // The source level (dram) is excluded because it holds caller-owned buffers
  // whose layout the kernel does not choose; any other level on the path that
  // accepts the layout is enough. The machine's per-level layout lists are
  // coarse, and a specific kernel's layout mismatches are reported by the
  // simulator as warnings, so this rule is deliberately a floor, not a
  // per-tile verdict.
  llvm::SmallVector<StringRef, 4> path;
  if (std::optional<StringRef> text =
          symbolOfKind(space, candidate, "memory_path"))
    text->split(path, ':');
  else
    path.push_back("sram");

  bool sawModeledLevel = false;
  for (StringRef level : path) {
    if (level == "dram")
      continue;
    const MemoryLevelModel *model = machine.findMemory(level);
    if (!model)
      continue;
    sawModeledLevel = true;
    if (llvm::is_contained(model->supportedLayouts, *layout))
      return legal();
  }

  if (!sawModeledLevel)
    return illegal(kind, "machine '" + machine.name +
                             "' models no memory on the path '" +
                             llvm::join(path, ":") + "'");
  return illegal(kind, "no memory supports layout '" + *layout + "'");
}

LegalityResult checkOwnerSupported(const SearchSpace &space,
                                   const Candidate &candidate,
                                   const WorkloadShape &,
                                   const MachineModel &machine) {
  ConstraintKind kind = ConstraintKind::OwnerSupported;
  std::optional<StringRef> mapping =
      symbolOfKind(space, candidate, "owner_mapping");
  if (!mapping)
    return legal();

  llvm::SmallVector<StringRef, 4> owners;
  mapping->split(owners, '/');
  if (owners.empty())
    return illegal(kind, "owner_mapping is empty");

  for (StringRef owner : owners)
    if (!machine.findOwner(owner))
      return illegal(kind, "owner '" + owner + "' is not modeled by machine '" +
                               machine.name + "'");

  // Outer comes first: each owner must sit inside the one to its left.
  for (size_t i = 1; i < owners.size(); ++i)
    if (!nestedUnder(machine, owners[i], owners[i - 1]))
      return illegal(kind, "owner '" + owners[i] + "' is not nested under '" +
                               owners[i - 1] + "'");
  return legal();
}

LegalityResult checkFragmentCompatible(const SearchSpace &space,
                                       const Candidate &candidate,
                                       const WorkloadShape &,
                                       const MachineModel &machine) {
  ConstraintKind kind = ConstraintKind::FragmentCompatible;
  std::optional<StringRef> text =
      symbolOfKind(space, candidate, "fragment_shape");
  if (!text)
    return legal();

  std::optional<std::array<int64_t, 3>> fragment = parseShapeTriple(*text);
  if (!fragment)
    return illegal(kind, "fragment_shape '" + *text +
                             "' is not a positive MxNxK triple");

  bool supported = false;
  for (const MatrixEngineModel &engine : machine.matrixEngines) {
    const std::array<int64_t, 3> &shape = largestTileShape(engine);
    if ((*fragment)[0] >= shape[0] && (*fragment)[1] >= shape[1] &&
        (*fragment)[2] >= shape[2]) {
      supported = true;
      break;
    }
  }
  if (!supported)
    return illegal(kind, "fragment '" + *text +
                             "' is not supported by any matrix engine");

  if (isMasked(space, candidate))
    return legal();

  IntResolver resolver{candidate, {}};
  std::optional<int64_t> BM = resolver.get("BM");
  std::optional<int64_t> BN = resolver.get("BN");
  std::optional<int64_t> BK = resolver.get("BK");
  if (!resolver.error.empty())
    return illegal(kind, resolver.error);

  if (*BM % (*fragment)[0] != 0 || *BN % (*fragment)[1] != 0 ||
      *BK % (*fragment)[2] != 0)
    return illegal(kind, "fragment '" + *text + "' does not divide execution " +
                             "tile " + tripleText({*BM, *BN, *BK}));
  return legal();
}

LegalityResult checkTileHierarchyCompatible(const SearchSpace &space,
                                            const Candidate &candidate,
                                            const WorkloadShape &,
                                            const MachineModel &) {
  ConstraintKind kind = ConstraintKind::TileHierarchyCompatible;
  std::optional<StringRef> text =
      symbolOfKind(space, candidate, "fragment_shape");
  if (!text)
    return legal();

  std::optional<std::array<int64_t, 3>> fragment = parseShapeTriple(*text);
  if (!fragment)
    return illegal(kind, "fragment_shape '" + *text +
                             "' is not a positive MxNxK triple");
  if (isMasked(space, candidate))
    return legal();

  IntResolver resolver{candidate, {}};
  std::optional<int64_t> BM = resolver.get("BM");
  std::optional<int64_t> BN = resolver.get("BN");
  std::optional<int64_t> BK = resolver.get("BK");
  if (!resolver.error.empty())
    return illegal(kind, resolver.error);

  if (*BM % (*fragment)[0] != 0 || *BN % (*fragment)[1] != 0 ||
      *BK % (*fragment)[2] != 0)
    return illegal(kind, "child fragment '" + *text + "' does not divide " +
                             "parent tile " + tripleText({*BM, *BN, *BK}) +
                             " and tail_policy is not 'mask'");
  return legal();
}

LegalityResult checkPipelineLiveTiles(const SearchSpace &space,
                                      const Candidate &candidate,
                                      const WorkloadShape &shape,
                                      const MachineModel &machine) {
  ConstraintKind kind = ConstraintKind::PipelineLiveTiles;
  const MemoryLevelModel *sram = machine.findMemory("sram");
  if (!sram)
    return illegal(kind, "machine '" + machine.name +
                             "' does not model memory 'sram'");

  int64_t stages =
      std::max<int64_t>(1, candidate.integer("pipeline_stages").value_or(1));
  int64_t prefetch = candidate.integer("prefetch_distance").value_or(0);

  // Only a pipelined kernel keeps copies in flight: a single-stage loop
  // completes each movement before issuing the next.
  uint64_t outstanding = stages > 1 ? static_cast<uint64_t>(prefetch) : 0;
  if (outstanding > machine.dma.maxOutstanding)
    return illegal(kind, llvm::Twine(outstanding) +
                             " outstanding async copies exceed machine '" +
                             machine.name + "' dma.max_outstanding " +
                             llvm::Twine(machine.dma.maxOutstanding));

  IntResolver resolver{candidate, {}};
  std::optional<int64_t> BM = resolver.get("BM");
  std::optional<int64_t> BN = resolver.get("BN");
  std::optional<int64_t> BK = resolver.get("BK");
  if (!resolver.error.empty())
    return illegal(kind, resolver.error);

  std::optional<uint64_t> tileBytes =
      stagedTileBytes(space, shape, *BM, *BN, *BK);
  if (!tileBytes)
    return illegal(kind, "input or weight dtype is not a micro dtype");

  uint64_t live = static_cast<uint64_t>(stages) * *tileBytes;
  uint64_t limit =
      static_cast<uint64_t>(sram->capacityBytes * kSramUtilizationLimit);
  if (live > limit)
    return illegal(kind, "requires " + llvm::Twine(live) +
                             " live bytes but sram holds " +
                             llvm::Twine(sram->capacityBytes) +
                             " bytes (limit " + llvm::Twine(limit) + ")");
  return legal();
}

} // namespace

LegalityResult checkConstraint(const SearchConstraint &constraint,
                               const SearchSpace &space,
                               const Candidate &candidate,
                               const WorkloadShape &shape,
                               const MachineModel &machine) {
  // A rule cannot be evaluated against a value the candidate never bound; that
  // is a malformed candidate, not a legal one.
  for (const std::string &name : constraint.params)
    if (!candidate.integer(name) && !candidate.symbol(name))
      return illegal(constraint.kind,
                     "candidate does not bind parameter '" + name + "'");

  switch (constraint.kind) {
  case ConstraintKind::SramCapacity:
    return checkSramCapacity(space, candidate, shape, machine);
  case ConstraintKind::AccCapacity:
    return checkAccCapacity(space, candidate, shape, machine);
  case ConstraintKind::MmaCompatible:
    return checkMmaCompatible(space, candidate, shape, machine);
  case ConstraintKind::MappingExtent:
    return checkMappingExtent(space, candidate, shape, machine);
  case ConstraintKind::TailSupported:
    return checkTailSupported(space, candidate, shape, machine);
  case ConstraintKind::VectorWidthSupported:
    return checkVectorWidthSupported(space, candidate, shape, machine);
  case ConstraintKind::TileHierarchyCompatible:
    return checkTileHierarchyCompatible(space, candidate, shape, machine);
  case ConstraintKind::LayoutSupported:
    return checkLayoutSupported(space, candidate, shape, machine);
  case ConstraintKind::OwnerSupported:
    return checkOwnerSupported(space, candidate, shape, machine);
  case ConstraintKind::FragmentCompatible:
    return checkFragmentCompatible(space, candidate, shape, machine);
  case ConstraintKind::PipelineLiveTiles:
    return checkPipelineLiveTiles(space, candidate, shape, machine);
  }
  llvm_unreachable("unhandled constraint kind");
}

LegalityResult checkLegality(const SearchSpace &space,
                             const Candidate &candidate,
                             const WorkloadShape &shape,
                             const MachineModel &machine) {
  for (const SearchConstraint &constraint : space.constraints) {
    LegalityResult result =
        checkConstraint(constraint, space, candidate, shape, machine);
    if (!result.legal)
      return result;
  }
  return legal();
}

} // namespace mlir::llk::perf
