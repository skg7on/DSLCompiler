//===- LayoutConstraints.h - LLKMap layout declarations (D3) --------------===//
//
// Part of the target-independent mapping core (epic #67, workstream D3).
//
// LLKMap is a small declarative language for target layout legality. A
// `layout` declaration names a target implementation layout, its parameters,
// the finite domains those parameters range over, the constraints that make it
// legal, and the affine logical-to-physical map it describes.
//
// Layout ids are target-owned strings (`avx2.blocked_2d`). They are results of
// mapping, never new enumerants in the Micro dialect or `#micro.layout`
// (design §13.4) -- which is why this header lives in LLK/Mapping and knows
// nothing about the target it is describing.
//
// The grammar this parser accepts is documented in
// `docs/design/llkmap-layout-grammar.md`.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_LAYOUTCONSTRAINTS_H
#define LLK_MAPPING_LAYOUTCONSTRAINTS_H

#include "LLK/Mapping/LlkMap.h"

#include "LLK/Machine/MachineModel.h"

#include "mlir/IR/AffineMap.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace mlir::llk::mapping {

/// One declared parameter. `symbolic` parameters range over a declared set of
/// names; integer parameters range over integer intervals.
struct LayoutParam {
  std::string name;
  bool symbolic = false;
};

/// The finite set a parameter ranges over, in ascending declaration order.
struct ParamDomain {
  std::vector<LayoutValue> values;
};

/// `map (m, n) -> (m, floordiv(n, VW), mod(n, VW))`.
struct AffineMapSpec {
  std::vector<std::string> dims;
  std::vector<ExprPtr> results;
};

/// Canonical rendering of a map clause: dimensions, then result expressions in
/// order. Shared by layout and rule content hashing.
std::string canonicalAffineMapSpecString(const AffineMapSpec &spec);

/// Source-faithful rendering of a map clause as `(dims) -> (results)`, the
/// inverse of the map grammar. Shared by layout and rule printing.
std::string printAffineMapSpec(const AffineMapSpec &spec);

/// Source-faithful rendering of a declared parameter domain: a contiguous
/// ascending integer run prints as `[lo..hi]`, anything else as `{...}`. Both
/// forms re-parse to the same value list, so the printer is deterministic and
/// faithful without the original spelling being recorded.
std::string printParamDomain(const ParamDomain &domain);

/// Builds the concrete `mlir::AffineMap` a map clause describes, substituting
/// `constants` for integer parameters. `spec.dims` become the map's dimensions
/// in order. Fails when a result references neither a dimension nor a listed
/// constant, or is not affine. Shared by the layout solver and the rule
/// matcher's affine-map predicate, so both interpret a map identically.
llvm::Expected<mlir::AffineMap>
buildAffineMap(const AffineMapSpec &spec,
               const llvm::StringMap<int64_t> &constants,
               mlir::MLIRContext &context);

struct LayoutDef {
  std::string id;
  std::vector<LayoutParam> params;
  std::map<std::string, ParamDomain> domains;
  std::vector<ExprPtr> constraints;
  std::optional<AffineMapSpec> map;

  const LayoutParam *findParam(llvm::StringRef name) const;
  bool isSymbolic(llvm::StringRef name) const;
};

/// Renders one `layout` declaration as LLKMap text that re-parses to an equal
/// `LayoutDef` (design §25.3). Parameters keep their declaration order; domains
/// are emitted in the order the parameters appear, with a parameterless-of-
/// domain parameter still declared by the header list. Deterministic: two calls
/// on the same value produce identical bytes.
std::string printLayout(const LayoutDef &def);

/// Loaded layout declarations, keyed by id.
class LayoutRegistry {
public:
  /// Adds `def`, or fails with a stable message when its id is a duplicate.
  bool add(LayoutDef def, std::string &error);

  const LayoutDef *find(llvm::StringRef id) const;
  llvm::ArrayRef<LayoutDef> all() const { return defs_; }

  /// FNV-1a 64 hash over every declaration's canonical rendering, folded in id
  /// order so it is independent of file or insertion order and stable across
  /// runs and toolchains (design §22.1/§22.2). An empty registry hashes its
  /// empty input, never zero-by-accident.
  uint64_t computeContentHash() const;

private:
  std::vector<LayoutDef> defs_;
};

/// Parses a complete LLKMap file. `sourceName` appears in diagnostics.
llvm::Expected<LayoutRegistry> parseLayoutText(llvm::StringRef text,
                                               llvm::StringRef sourceName);

/// Reads and parses the file at `path`.
llvm::Expected<LayoutRegistry> loadLayoutFile(llvm::StringRef path);

//===----------------------------------------------------------------------===//
// Solving
//===----------------------------------------------------------------------===//

/// One legal instantiation of a layout: the parameter values that satisfy
/// every constraint, and the affine map with those values substituted in.
struct LayoutSolution {
  std::map<std::string, LayoutValue> values;
  mlir::AffineMap map;
};

/// Bounds on a solve. `truncated` in the result reports when any bound ended
/// the search early, so a caller never reads a capped result as complete.
struct SolverLimits {
  uint64_t maxAssignments = 100000;
  uint64_t maxSolutions = 8;
  /// Total domain elements every `forall`/`exists` in one solve may examine
  /// (design §13.3 bounds quantification). A quantifier that would exceed it
  /// stops and the solve is reported truncated -- never a silently accepted
  /// "no solution". A declaration without a quantifier never consumes it.
  ///
  /// This is a bound on *enumeration*: a backend that does not enumerate (a
  /// symbolic solver, say) may ignore it, since it never scans a domain.
  uint64_t maxQuantifierIterations = kDefaultQuantifierIterations;
};

struct LayoutSolveResult {
  std::vector<LayoutSolution> solutions;
  /// True when the search stopped before exhausting the assignment or solution
  /// space (design §13.3). The reported solutions are legal; there may simply
  /// be more.
  bool truncated = false;
  /// True when a quantifier ran out of its budget, so at least one constraint
  /// is *undecided* and the reported solutions may not be legal: an exhausted
  /// quantifier yields 0, and a surrounding `!`/`== 0`/`!= 1` can then read as
  /// satisfied. A caller MUST NOT accept a solution from an undecided solve --
  /// unlike `truncated`, this is a soundness flag, not a completeness one. It
  /// implies `truncated`.
  bool undecided = false;
};

/// The layout-solving backend (design §13.3). The bounded enumerator is one
/// implementation; a future solver -- an SMT backend, say -- implements this
/// same interface without any change to rule files or callers.
class LayoutSolver {
public:
  virtual ~LayoutSolver() = default;

  /// Solves `def`; the contract is `solveLayout`'s below.
  virtual llvm::Expected<LayoutSolveResult>
  solve(const LayoutDef &def, const machine::MachineModel &machine,
        mlir::MLIRContext &context, const LayoutContext &layoutContext,
        const SolverLimits &limits = {}) const = 0;
};

/// The bounded-enumeration backend: deterministic and dependency-free, the only
/// implementation today.
std::unique_ptr<LayoutSolver> makeBoundedLayoutSolver();

/// Solves `def` against `machine` by bounded enumeration over the declared
/// finite domains, in declaration order. Fails on a parameter without a
/// domain, an empty domain, a constraint that cannot be evaluated, or a map
/// clause that is not affine. A thin wrapper over `makeBoundedLayoutSolver()`,
/// kept so existing callers keep working while the interface is adopted.
llvm::Expected<LayoutSolveResult>
solveLayout(const LayoutDef &def, const machine::MachineModel &machine,
            mlir::MLIRContext &context, const LayoutContext &layoutContext,
            const SolverLimits &limits = {});

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_LAYOUTCONSTRAINTS_H
