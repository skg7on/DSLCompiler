//===- SearchSpace.h - Typed tuning search space for Micro-IR -------------===//
//
// Part of the M12 tuning core (issue #49).
//
// A `micro.search_space` is the auto-scheduler's input: a set of tunable
// parameters, the legality constraints that relate them, and the objective
// candidates are ranked by. This header is the typed C++ mirror of those ops,
// so the tuning core never reasons about MLIR attributes directly.
//
// The dialect is the source of truth for vocabulary. A parameter's `kind`
// (integer, layout, memory_path, owner_mapping, fragment_shape, tail_policy)
// is carried through unchanged, which is what lets a legality rule find "the
// layout parameter" without guessing from its name -- the dialect's `kind` is
// declared precisely so consumers do not infer semantics from names.
//
// loadSearchSpace() is the only entry point that touches MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_PERF_SEARCHSPACE_H
#define LLK_PERF_SEARCHSPACE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace mlir::micro {
class SearchSpaceOp;
} // namespace mlir::micro

namespace mlir::llk::perf {

/// One legal value of a search parameter. Integer choices cover tile sizes,
/// loop bounds, pipeline stages, and vector widths; string choices cover
/// layouts, memory paths, owner mappings, fragment shapes, and policies.
struct SearchChoice {
  std::variant<int64_t, std::string> value;

  SearchChoice(int64_t integer) : value(integer) {}
  SearchChoice(std::string text) : value(std::move(text)) {}
  SearchChoice(const char *text) : value(std::string(text)) {}

  bool isInteger() const { return std::holds_alternative<int64_t>(value); }
  bool isSymbolic() const { return std::holds_alternative<std::string>(value); }

  /// Precondition: isInteger().
  int64_t integer() const { return std::get<int64_t>(value); }
  /// Precondition: isSymbolic().
  llvm::StringRef symbol() const { return std::get<std::string>(value); }
};

/// One tunable parameter and its legal values, in declaration order. The first
/// choice is the preferred value the schedule export anchored on.
struct SearchParam {
  std::string name;
  /// The micro.param domain name: integer, layout, memory_path, owner_mapping,
  /// fragment_shape, or tail_policy.
  std::string kind;
  std::vector<SearchChoice> choices;
};

/// The typed legality rules a `micro.constraint` can name. The enum order and
/// spellings mirror `micro.constraint`'s accepted kinds.
enum class ConstraintKind {
  SramCapacity,
  AccCapacity,
  MmaCompatible,
  MappingExtent,
  TailSupported,
  VectorWidthSupported,
  TileHierarchyCompatible,
  LayoutSupported,
  OwnerSupported,
  FragmentCompatible,
  PipelineLiveTiles
};

llvm::StringRef stringifyConstraintKind(ConstraintKind kind);
std::optional<ConstraintKind> symbolizeConstraintKind(llvm::StringRef text);

/// One legality record: a typed rule plus the parameters it relates. `attrs`
/// is reserved for future `micro.constraint` dictionary attributes; the
/// current dialect op carries none, so it stays empty.
struct SearchConstraint {
  ConstraintKind kind;
  std::vector<std::string> params;
  std::map<std::string, std::string> attrs;
};

enum class ObjectiveDirection { Minimize, Maximize };

llvm::StringRef stringifyObjectiveDirection(ObjectiveDirection direction);
std::optional<ObjectiveDirection>
symbolizeObjectiveDirection(llvm::StringRef text);

/// How candidates are ranked. The primary metric is compared first, then the
/// secondary metrics in the order given, then the candidate id for stability.
struct SearchObjective {
  ObjectiveDirection direction = ObjectiveDirection::Minimize;
  std::string primaryMetric = "latency_cycles";
  std::vector<std::string> secondaryMetrics;
};

/// A loaded search space: parameters, constraints, and objective.
struct SearchSpace {
  std::string name;
  std::string workload;
  std::vector<SearchParam> params;
  std::vector<SearchConstraint> constraints;
  SearchObjective objective;

  const SearchParam *findParam(llvm::StringRef name) const;
  /// The parameter declared with `kind`, or null when the space declares none
  /// or more than one. Used to resolve symbolic roles by domain rather than by
  /// name; a kind shared by several parameters (integer) is not a role.
  const SearchParam *findParamOfKind(llvm::StringRef kind) const;
};

/// The five M buckets the schedule database is keyed by: {1}, [2, 4], [5, 16],
/// [17, 64], and [65, inf). Mirrors `llk::classifyM`, which lives in the LLK
/// transforms layer the perf layer does not depend on; the bucket boundaries
/// are a project convention, so the two must stay in step.
int64_t classifyMBucket(int64_t M);

/// The problem shape a search space is instantiated for. The shape is not part
/// of `micro.search_space` (the op names the workload, not its extents), so the
/// driver supplies it; legality and ranking need the extents and dtypes to
/// decide capacity, divisibility, and dtype support.
struct WorkloadShape {
  int64_t M = 0;
  int64_t N = 0;
  int64_t K = 0;
  std::string inputDType = "bf16";
  std::string weightDType = "bf16";
  std::string accumulatorDType = "f32";
  std::string outputDType = "bf16";
};

/// Loads one `micro.search_space` op. Fails when the op's declarations cannot
/// be represented: an unsupported parameter kind, an empty choice list, a
/// duplicate choice, or an unknown constraint kind.
llvm::Expected<SearchSpace> loadSearchSpace(micro::SearchSpaceOp op);

/// Finds the `micro.search_space` named `symbol` anywhere in `module`, or the
/// only one present when `symbol` is empty. Fails when the name matches
/// nothing or when the choice is ambiguous, mirroring findMicroKernel.
llvm::Expected<SearchSpace> loadSearchSpace(mlir::ModuleOp module,
                                            llvm::StringRef symbol = "");

} // namespace mlir::llk::perf

#endif // LLK_PERF_SEARCHSPACE_H
