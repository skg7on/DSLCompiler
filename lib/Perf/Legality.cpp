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
const std::vector<int64_t> &
largestTileShape(const machine::ComputeNode &engine) {
  auto volume = [](llvm::ArrayRef<int64_t> shape) {
    int64_t product = 1;
    for (int64_t extent : shape)
      product *= extent;
    return product;
  };
  const std::vector<int64_t> *best = &engine.shapes.front();
  for (const std::vector<int64_t> &shape : engine.shapes)
    if (shape.size() == best->size() && volume(shape) > volume(*best))
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

/// The shapes a dtype/shape-dependent rule must evaluate: every contraction the
/// kernel performs, or -- when it performs none -- the original workload. Empty
/// when neither fact is available, which the caller reports as unevaluable.
llvm::SmallVector<const WorkloadShape *>
applicableShapes(const BindingFacts &facts) {
  llvm::SmallVector<const WorkloadShape *> shapes;
  if (!facts.contractions.empty()) {
    for (const WorkloadShape &contraction : facts.contractions)
      shapes.push_back(&contraction);
    return shapes;
  }
  if (facts.originalWorkload)
    shapes.push_back(&*facts.originalWorkload);
  return shapes;
}

/// The shape whose element type a dtype-only rule reads: the original workload
/// when the export recorded one, otherwise the first contraction. Null when
/// the kernel supplies no dtype-bearing fact at all.
const WorkloadShape *dtypeShape(const BindingFacts &facts) {
  if (facts.originalWorkload)
    return &*facts.originalWorkload;
  if (!facts.contractions.empty())
    return &facts.contractions.front();
  return nullptr;
}

/// Resolves the symbolic parameters a constraint references against `kind`,
/// appending each to `out` in declaration order. A reference that names neither
/// a declared parameter nor a declared role of that kind is unresolved; a
/// reference to a parameter of a different kind is a kind mismatch. Both are
/// returned as an error string rather than silently skipped, because "the
/// space declares no layout choice to check" and "the constraint names a role
/// nothing declares" must not answer the same way. A role declared by more than
/// one parameter of `kind` is likewise rejected: which one governs that role is
/// then unanswerable.
std::optional<std::string>
resolveReferencedParams(const SearchSpace &space,
                        const SearchConstraint &constraint, StringRef kind,
                        llvm::SmallVectorImpl<const SearchParam *> &out) {
  llvm::StringSet<> declaredRoles;
  for (const SearchParam &param : space.params) {
    if (param.kind != kind || param.role.empty())
      continue;
    if (!declaredRoles.insert(param.role).second)
      return std::string("more than one ") + kind.str() +
             " parameter declares role '" + param.role + "'";
  }

  for (const std::string &name : constraint.params) {
    const SearchParam *param = space.findParam(name);
    if (!param) {
      // A reference may name the role rather than the parameter itself.
      for (const SearchParam &candidateParam : space.params)
        if (candidateParam.kind == kind && candidateParam.role == name) {
          param = &candidateParam;
          break;
        }
    }
    if (!param)
      return std::string("references '") + name +
             "', which the space does not declare";
    if (param->kind != kind)
      return std::string("references '") + name + "', which is not a " +
             kind.str() + " parameter";
    out.push_back(param);
  }
  return std::nullopt;
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

bool isOfKind(const machine::MachineModel &machine,
              const machine::ExecutorNode &executor, StringRef kind) {
  return machine.ownerMatches(kind, executor.id);
}

bool nestedUnder(const machine::MachineModel &machine, StringRef inner,
                 StringRef outer) {
  // v1 asked whether one owner *kind* sat under another in a kind hierarchy;
  // v2 asks the same question of concrete executors, since that is where
  // containment now lives.
  for (const machine::ExecutorNode &executor : machine.executors) {
    if (!isOfKind(machine, executor, inner))
      continue;
    const machine::ExecutorNode *current = &executor;
    llvm::StringSet<> visited;
    while (current && current->parent) {
      const machine::ExecutorNode *parent =
          machine.findExecutor(*current->parent);
      if (!parent)
        break;
      if (isOfKind(machine, *parent, outer))
        return true;
      if (!visited.insert(parent->id).second)
        break; // a cycle would otherwise spin; verifyMachineModel rejects these
      current = parent;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Rules
//===----------------------------------------------------------------------===//

LegalityResult checkSramCapacity(const SearchSpace &space,
                                 const Candidate &candidate,
                                 const BindingFacts &facts,
                                 const machine::MachineModel &machine) {
  ConstraintKind kind = ConstraintKind::SramCapacity;
  const machine::MemoryNode *sram = machine.findMemoryOfKind("sram");
  if (!sram)
    return illegal(kind, "machine '" + machine.target +
                             "' does not model memory 'sram'");

  IntResolver resolver{candidate, {}};
  std::optional<int64_t> BM = resolver.get("BM");
  std::optional<int64_t> BN = resolver.get("BN");
  std::optional<int64_t> BK = resolver.get("BK");
  if (!resolver.error.empty())
    return illegal(kind, resolver.error);

  llvm::SmallVector<const WorkloadShape *> shapes = applicableShapes(facts);
  if (shapes.empty())
    return illegal(kind, "cannot be evaluated: the kernel supplies no workload "
                         "shape (no micro.mma and no original-workload "
                         "provenance)");

  uint64_t limit =
      static_cast<uint64_t>(sram->capacityBytes * kSramUtilizationLimit);
  for (const WorkloadShape *shape : shapes) {
    std::optional<uint64_t> tileBytes =
        stagedTileBytes(space, *shape, *BM, *BN, *BK);
    if (!tileBytes)
      return illegal(kind, "input or weight dtype is not a micro dtype");

    // A multi-stage pipeline double-buffers the staged tiles.
    int64_t stages =
        std::max<int64_t>(1, candidate.integer("pipeline_stages").value_or(1));
    uint64_t factor = stages > 1 ? 2 : 1;
    uint64_t required = factor * *tileBytes;
    if (required > limit)
      return illegal(kind, "requires " + llvm::Twine(required) +
                               " bytes but sram holds " +
                               llvm::Twine(sram->capacityBytes) +
                               " bytes (limit " + llvm::Twine(limit) + ")");
  }
  return legal();
}

LegalityResult checkAccCapacity(const SearchSpace &space,
                                const Candidate &candidate,
                                const BindingFacts &facts,
                                const machine::MachineModel &machine) {
  ConstraintKind kind = ConstraintKind::AccCapacity;
  const machine::MemoryNode *acc = machine.findMemoryOfKind("acc");
  if (!acc)
    return illegal(kind, "machine '" + machine.target +
                             "' does not model memory 'acc'");

  IntResolver resolver{candidate, {}};
  std::optional<int64_t> BM = resolver.get("BM");
  std::optional<int64_t> BN = resolver.get("BN");
  if (!resolver.error.empty())
    return illegal(kind, resolver.error);

  llvm::SmallVector<const WorkloadShape *> shapes = applicableShapes(facts);
  if (shapes.empty())
    return illegal(kind, "cannot be evaluated: the kernel supplies no workload "
                         "shape (no micro.mma and no original-workload "
                         "provenance)");

  for (const WorkloadShape *shape : shapes) {
    std::optional<int64_t> accBytes = dtypeBytes(shape->accumulatorDType);
    if (!accBytes)
      return illegal(kind, "accumulator dtype '" + shape->accumulatorDType +
                               "' is not a micro dtype");

    // Every accumulator the workload keeps live must fit at once.
    uint64_t required =
        uint64_t(accumulatorCount(space.workload)) * *BM * *BN * *accBytes;
    if (required > acc->capacityBytes)
      return illegal(kind, "requires " + llvm::Twine(required) +
                               " bytes but acc holds " +
                               llvm::Twine(acc->capacityBytes) + " bytes");
  }
  return legal();
}

/// Reports that `engine` cannot run `shape`; shared by each contraction so the
/// diagnostic names the *first* offending dtype pair.
LegalityResult mmaUnsupported(const machine::ComputeNode &engine,
                              const WorkloadShape &shape) {
  ConstraintKind kind = ConstraintKind::MmaCompatible;
  if (!llvm::is_contained(engine.elementTypes, shape.inputDType))
    return illegal(kind, "engine '" + engine.id +
                             "' does not support input dtype '" +
                             shape.inputDType + "' (supported: " +
                             llvm::join(engine.elementTypes, ", ") + ")");
  return illegal(kind, "no matrix engine supports accumulator dtype '" +
                           shape.accumulatorDType + "' (engine '" + engine.id +
                           "' supports: " +
                           llvm::join(engine.accumulatorDTypes, ", ") + ")");
}

LegalityResult checkMmaCompatible(const SearchSpace &space,
                                  const Candidate &candidate,
                                  const BindingFacts &facts,
                                  const machine::MachineModel &machine) {
  ConstraintKind kind = ConstraintKind::MmaCompatible;
  if (machine.computesOfKind("matrix_engine").empty())
    return illegal(kind, "machine '" + machine.target +
                             "' declares no matrix engine");

  // Every contraction the kernel performs must be runnable, not merely the
  // first: a second `micro.mma` with an unsupported dtype pair is a rejection.
  llvm::SmallVector<const WorkloadShape *> shapes = applicableShapes(facts);
  if (shapes.empty())
    return illegal(kind, "cannot be evaluated: the kernel performs no "
                         "contraction (micro.mma) to check");

  bool masked = isMasked(space, candidate);
  IntResolver resolver{candidate, {}};
  std::optional<int64_t> BM;
  std::optional<int64_t> BN;
  std::optional<int64_t> BK;
  if (!masked) {
    BM = resolver.get("BM");
    BN = resolver.get("BN");
    BK = resolver.get("BK");
    if (!resolver.error.empty())
      return illegal(kind, resolver.error);
  }

  for (const WorkloadShape *shape : shapes) {
    // Any engine that accepts the dtype pair can run this contraction; the
    // first matching one is used, so a machine is not rejected because its
    // *first* engine happens to have a different dtype mix.
    const machine::ComputeNode *engine = nullptr;
    for (const machine::ComputeNode *candidateEngine :
         machine.computesOfKind("matrix_engine"))
      if (llvm::is_contained(candidateEngine->elementTypes,
                             shape->inputDType) &&
          llvm::is_contained(candidateEngine->accumulatorDTypes,
                             shape->accumulatorDType)) {
        engine = candidateEngine;
        break;
      }
    if (!engine)
      return mmaUnsupported(*machine.computesOfKind("matrix_engine").front(),
                            *shape);
    if (masked)
      continue;

    const std::vector<int64_t> &fragment = largestTileShape(*engine);
    if (*BM % fragment[0] != 0 || *BN % fragment[1] != 0 ||
        *BK % fragment[2] != 0)
      return illegal(kind, "tile " + tripleText({*BM, *BN, *BK}) +
                               " is not a multiple of engine '" + engine->id +
                               "' fragment " + tripleText(fragment) +
                               " and tail_policy is not 'mask'");
  }
  return legal();
}

LegalityResult checkMappingExtent(const SearchSpace &,
                                  const Candidate &candidate,
                                  const machine::MachineModel &machine) {
  ConstraintKind kind = ConstraintKind::MappingExtent;
  std::optional<int64_t> threads = candidate.integer("num_threads");
  if (!threads)
    return illegal(kind, "candidate does not bind parameter 'num_threads'");
  if (*threads < 1)
    return illegal(kind, "num_threads must be positive");
  // The spec's second condition -- the spatial map's worker extent -- is not a
  // search parameter in the current export, so only the thread count is bound.
  // Nothing here needs a workload shape, which is why a vector-only kernel can
  // satisfy a mapping_extent constraint.
  if (uint64_t(*threads) > machine.workerThreads)
    return illegal(kind, "num_threads " + llvm::Twine(*threads) +
                             " exceeds machine '" + machine.target +
                             "' worker_threads " +
                             llvm::Twine(machine.workerThreads));
  return legal();
}

LegalityResult checkTailSupported(const SearchSpace &space,
                                  const Candidate &candidate,
                                  const BindingFacts &facts,
                                  const machine::MachineModel &) {
  ConstraintKind kind = ConstraintKind::TailSupported;
  if (isMasked(space, candidate))
    return legal();

  // Divisibility is a whole-workload question, so it reads the original
  // dimensions the export recorded before tiling -- never a contraction's
  // instruction-fragment shape.
  if (!facts.originalWorkload)
    return illegal(kind, "cannot be evaluated: the kernel supplies no original "
                         "workload dimensions (no original-workload "
                         "provenance)");
  const WorkloadShape &shape = *facts.originalWorkload;

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
                                         const BindingFacts &facts,
                                         const machine::MachineModel &machine) {
  ConstraintKind kind = ConstraintKind::VectorWidthSupported;
  std::optional<int64_t> width = candidate.integer("vector_width");
  if (!width)
    return illegal(kind, "candidate does not bind parameter 'vector_width'");
  if (*width < 1)
    return illegal(kind, "vector_width must be positive");

  // The check is about the element type the engine must vectorize, so it needs
  // a dtype-bearing fact but no contraction per se.
  const WorkloadShape *shape = dtypeShape(facts);
  if (!shape)
    return illegal(kind, "cannot be evaluated: the kernel supplies no input "
                         "dtype (no original-workload provenance and no "
                         "micro.mma)");

  if (machine.computesOfKind("vector_engine").empty())
    return illegal(kind, "machine '" + machine.target +
                             "' declares no vector engine");

  const machine::ComputeNode *best = nullptr;
  int64_t lanes = 0;
  for (const machine::ComputeNode *engine :
       machine.computesOfKind("vector_engine")) {
    auto it = engine->lanes.find(shape->inputDType);
    if (it != engine->lanes.end() && it->second > lanes) {
      lanes = it->second;
      best = engine;
    }
  }
  if (!best)
    return illegal(kind, "no vector engine supports dtype '" +
                             shape->inputDType + "'");
  if (*width > lanes)
    return illegal(kind, "vector_width " + llvm::Twine(*width) +
                             " exceeds engine '" + best->id + "' lanes " +
                             llvm::Twine(lanes) + " for dtype '" +
                             shape->inputDType + "'");
  return legal();
}

LegalityResult checkLayoutSupported(const SearchSpace &space,
                                    const Candidate &candidate,
                                    const SearchConstraint &constraint,
                                    const machine::MachineModel &machine) {
  ConstraintKind kind = ConstraintKind::LayoutSupported;

  // Evaluate every layout parameter the constraint references, by name or by
  // declared role. A space that binds one layout per port must have each of
  // them checked; resolving "the layout" by unique kind silently skipped a
  // second parameter, treating ambiguity as absence.
  llvm::SmallVector<const SearchParam *, 2> layouts;
  if (std::optional<std::string> error =
          resolveReferencedParams(space, constraint, "layout", layouts))
    return illegal(kind, *error);
  if (layouts.empty())
    return legal();

  // The layout must be materializable somewhere in the target's memory system.
  // The source level (dram) is excluded because it holds caller-owned buffers
  // whose layout the kernel does not choose; any other level on the path that
  // accepts the layout is enough. The machine's per-level layout lists are
  // coarse, and a specific kernel's layout mismatches are reported by the
  // simulator as warnings, so this rule is deliberately a floor, not a
  // per-tile verdict. `memory_path` is supporting context, not a referenced
  // axis, so it is read best-effort.
  llvm::SmallVector<StringRef, 4> path;
  if (std::optional<StringRef> text =
          symbolOfKind(space, candidate, "memory_path"))
    text->split(path, ':');
  else
    path.push_back("sram");

  for (const SearchParam *param : layouts) {
    std::optional<StringRef> layout = candidate.symbol(param->name);
    if (!layout)
      return illegal(kind,
                     "candidate does not bind parameter '" + param->name + "'");

    bool supported = false;
    bool sawModeledLevel = false;
    for (StringRef level : path) {
      if (level == "dram")
        continue;
      const machine::MemoryNode *model = machine.findMemoryOfKind(level);
      if (!model)
        continue;
      sawModeledLevel = true;
      if (llvm::is_contained(model->supportedLayouts, *layout)) {
        supported = true;
        break;
      }
    }

    // Name the referenced parameter (and role) when the constraint governs
    // more than one layout, so the rejection says which port failed.
    std::string where;
    if (layouts.size() > 1 || !param->role.empty()) {
      std::string label = param->role.empty() ? param->name : param->role;
      where = "(role '" + label + "') ";
    }
    if (!supported) {
      if (!sawModeledLevel)
        return illegal(kind, where + "machine '" + machine.target +
                                 "' models no memory on the path '" +
                                 llvm::join(path, ":") + "'");
      return illegal(kind,
                     where + "no memory supports layout '" + *layout + "'");
    }
  }
  return legal();
}

LegalityResult checkOwnerSupported(const SearchSpace &space,
                                   const Candidate &candidate,
                                   const SearchConstraint &constraint,
                                   const machine::MachineModel &machine) {
  ConstraintKind kind = ConstraintKind::OwnerSupported;

  // As with layouts, evaluate each referenced owner_mapping parameter so a
  // multi-role declaration is checked in full rather than skipped.
  llvm::SmallVector<const SearchParam *, 2> mappings;
  if (std::optional<std::string> error =
          resolveReferencedParams(space, constraint, "owner_mapping", mappings))
    return illegal(kind, *error);
  if (mappings.empty())
    return legal();

  for (const SearchParam *param : mappings) {
    std::optional<StringRef> mapping = candidate.symbol(param->name);
    if (!mapping)
      return illegal(kind,
                     "candidate does not bind parameter '" + param->name + "'");

    std::string where;
    if (mappings.size() > 1 || !param->role.empty()) {
      std::string label = param->role.empty() ? param->name : param->role;
      where = "(role '" + label + "') ";
    }

    llvm::SmallVector<StringRef, 4> owners;
    mapping->split(owners, '/');
    if (owners.empty())
      return illegal(kind, where + "owner_mapping is empty");

    for (StringRef owner : owners)
      if (!machine.hasOwnerKind(owner))
        return illegal(kind, where + "owner '" + owner +
                                 "' is not modeled by machine '" +
                                 machine.target + "'");

    // Outer comes first: each owner must sit inside the one to its left.
    for (size_t i = 1; i < owners.size(); ++i)
      if (!nestedUnder(machine, owners[i], owners[i - 1]))
        return illegal(kind, where + "owner '" + owners[i] +
                                 "' is not nested under '" + owners[i - 1] +
                                 "'");
  }
  return legal();
}

LegalityResult checkFragmentCompatible(const SearchSpace &space,
                                       const Candidate &candidate,
                                       const machine::MachineModel &machine) {
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
  for (const machine::ComputeNode *engine :
       machine.computesOfKind("matrix_engine")) {
    const std::vector<int64_t> &shape = largestTileShape(*engine);
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
                                            const machine::MachineModel &) {
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
                                      const BindingFacts &facts,
                                      const machine::MachineModel &machine) {
  ConstraintKind kind = ConstraintKind::PipelineLiveTiles;
  const machine::MemoryNode *sram = machine.findMemoryOfKind("sram");
  if (!sram)
    return illegal(kind, "machine '" + machine.target +
                             "' does not model memory 'sram'");

  int64_t stages =
      std::max<int64_t>(1, candidate.integer("pipeline_stages").value_or(1));
  int64_t prefetch = candidate.integer("prefetch_distance").value_or(0);

  // Only a pipelined kernel keeps copies in flight: a single-stage loop
  // completes each movement before issuing the next.
  uint64_t outstanding = stages > 1 ? static_cast<uint64_t>(prefetch) : 0;
  const machine::TransferEngineNode *transfer = machine.primaryTransferEngine();
  uint64_t maxOutstanding = transfer ? transfer->maxOutstanding : 1;
  if (outstanding > maxOutstanding)
    return illegal(kind, llvm::Twine(outstanding) +
                             " outstanding async copies exceed machine '" +
                             machine.target + "' transfer engine limit " +
                             llvm::Twine(maxOutstanding));

  IntResolver resolver{candidate, {}};
  std::optional<int64_t> BM = resolver.get("BM");
  std::optional<int64_t> BN = resolver.get("BN");
  std::optional<int64_t> BK = resolver.get("BK");
  if (!resolver.error.empty())
    return illegal(kind, resolver.error);

  llvm::SmallVector<const WorkloadShape *> shapes = applicableShapes(facts);
  if (shapes.empty())
    return illegal(kind, "cannot be evaluated: the kernel supplies no workload "
                         "shape (no micro.mma and no original-workload "
                         "provenance)");

  uint64_t limit =
      static_cast<uint64_t>(sram->capacityBytes * kSramUtilizationLimit);
  for (const WorkloadShape *shape : shapes) {
    std::optional<uint64_t> tileBytes =
        stagedTileBytes(space, *shape, *BM, *BN, *BK);
    if (!tileBytes)
      return illegal(kind, "input or weight dtype is not a micro dtype");

    uint64_t live = static_cast<uint64_t>(stages) * *tileBytes;
    if (live > limit)
      return illegal(kind, "requires " + llvm::Twine(live) +
                               " live bytes but sram holds " +
                               llvm::Twine(sram->capacityBytes) +
                               " bytes (limit " + llvm::Twine(limit) + ")");
  }
  return legal();
}

} // namespace

LegalityResult checkConstraint(const SearchConstraint &constraint,
                               const SearchSpace &space,
                               const Candidate &candidate,
                               const BindingFacts &facts,
                               const machine::MachineModel &machine) {
  // Resolve every reference to a declared parameter -- by name, or by a role a
  // parameter declares. An unresolved reference is a malformed space and is
  // reported as such rather than attributed to the candidate. Once resolved,
  // the candidate is keyed by the parameter's *name* (Candidate has no role
  // lookup), so a valid role reference is admitted and evaluated instead of
  // being rejected as an unbound parameter. A parameter the candidate never
  // bound is still a rejection: a malformed candidate must not pass by
  // omission.
  for (const std::string &name : constraint.params) {
    const SearchParam *param = space.findParam(name);
    if (!param)
      for (const SearchParam &candidateParam : space.params)
        if (!candidateParam.role.empty() && candidateParam.role == name) {
          param = &candidateParam;
          break;
        }
    if (!param)
      return illegal(constraint.kind, "references '" + name +
                                          "', which the space does not "
                                          "declare as a parameter or role");
    if (!candidate.integer(param->name) && !candidate.symbol(param->name))
      return illegal(constraint.kind,
                     "candidate does not bind parameter '" + param->name + "'");
  }

  switch (constraint.kind) {
  case ConstraintKind::SramCapacity:
    return checkSramCapacity(space, candidate, facts, machine);
  case ConstraintKind::AccCapacity:
    return checkAccCapacity(space, candidate, facts, machine);
  case ConstraintKind::MmaCompatible:
    return checkMmaCompatible(space, candidate, facts, machine);
  case ConstraintKind::MappingExtent:
    return checkMappingExtent(space, candidate, machine);
  case ConstraintKind::TailSupported:
    return checkTailSupported(space, candidate, facts, machine);
  case ConstraintKind::VectorWidthSupported:
    return checkVectorWidthSupported(space, candidate, facts, machine);
  case ConstraintKind::TileHierarchyCompatible:
    return checkTileHierarchyCompatible(space, candidate, machine);
  case ConstraintKind::LayoutSupported:
    return checkLayoutSupported(space, candidate, constraint, machine);
  case ConstraintKind::OwnerSupported:
    return checkOwnerSupported(space, candidate, constraint, machine);
  case ConstraintKind::FragmentCompatible:
    return checkFragmentCompatible(space, candidate, machine);
  case ConstraintKind::PipelineLiveTiles:
    return checkPipelineLiveTiles(space, candidate, facts, machine);
  }
  llvm_unreachable("unhandled constraint kind");
}

LegalityResult checkConstraint(const SearchConstraint &constraint,
                               const SearchSpace &space,
                               const Candidate &candidate,
                               const WorkloadShape &shape,
                               const machine::MachineModel &machine) {
  // The single shape stands in for both facts: the whole workload (tail
  // divisibility) and the one contraction (dtype and fragment requirements).
  return checkConstraint(constraint, space, candidate,
                         BindingFacts{shape, {shape}}, machine);
}

LegalityResult checkLegality(const SearchSpace &space,
                             const Candidate &candidate,
                             const BindingFacts &facts,
                             const machine::MachineModel &machine) {
  for (const SearchConstraint &constraint : space.constraints) {
    LegalityResult result =
        checkConstraint(constraint, space, candidate, facts, machine);
    if (!result.legal)
      return result;
  }
  return legal();
}

LegalityResult checkLegality(const SearchSpace &space,
                             const Candidate &candidate,
                             const WorkloadShape &shape,
                             const machine::MachineModel &machine) {
  return checkLegality(space, candidate, BindingFacts{shape, {shape}}, machine);
}

} // namespace mlir::llk::perf
