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

/// A concrete value a layout expression can take: an integer or a symbolic
/// name. Boolean expressions evaluate to integers 0 and 1.
using LayoutValue = std::variant<int64_t, std::string>;

enum class ExprKind {
  IntLit,     ///< integer literal
  StringLit,  ///< quoted string
  Ident,      ///< parameter or builtin (`rank`, `element_type`)
  Call,       ///< `floordiv(a, b)`, `machine.compute("vector_engine")`
  MemberCall, ///< `receiver.lanes(dtype)`, `receiver.count`
  Unary,      ///< `!a`, `-a`
  Binary      ///< `a + b`, `a % b`, `a == b`, `a && b`
};

struct Expr;
using ExprPtr = std::shared_ptr<const Expr>;

struct Expr {
  ExprKind kind = ExprKind::IntLit;
  /// Integer literal value, or a boolean result materialized during parsing.
  int64_t intValue = 0;
  /// Identifier name, call callee, member name, or operator spelling.
  std::string text;
  /// Call/member arguments; for a member call, `operands[0]` is the receiver.
  std::vector<ExprPtr> operands;
};

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

struct LayoutDef {
  std::string id;
  std::vector<LayoutParam> params;
  std::map<std::string, ParamDomain> domains;
  std::vector<ExprPtr> constraints;
  std::optional<AffineMapSpec> map;

  const LayoutParam *findParam(llvm::StringRef name) const;
  bool isSymbolic(llvm::StringRef name) const;
};

/// Loaded layout declarations, keyed by id.
class LayoutRegistry {
public:
  /// Adds `def`, or fails with a stable message when its id is a duplicate.
  bool add(LayoutDef def, std::string &error);

  const LayoutDef *find(llvm::StringRef id) const;
  llvm::ArrayRef<LayoutDef> all() const { return defs_; }

private:
  std::vector<LayoutDef> defs_;
};

/// Parses a complete LLKMap file. `sourceName` appears in diagnostics.
llvm::Expected<LayoutRegistry> parseLayoutText(llvm::StringRef text,
                                               llvm::StringRef sourceName);

/// Reads and parses the file at `path`.
llvm::Expected<LayoutRegistry> loadLayoutFile(llvm::StringRef path);

//===----------------------------------------------------------------------===//
// Expression evaluation
//===----------------------------------------------------------------------===//

/// The values a declaration is instantiated against: the rank and element type
/// of the value being laid out. They back the `rank` and `element_type`
/// builtins.
struct LayoutContext {
  int64_t rank = 0;
  std::string elementType;
};

/// The result of evaluating an expression. `Handle` is the intermediate value
/// a `machine.compute(...)` / `machine.memory(...)` query produces before a
/// member query is applied; it never escapes as a solution value.
struct EvalValue {
  enum class Kind { Int, Str, Handle };
  Kind kind = Kind::Int;
  int64_t intValue = 0;
  std::string text;
};

/// Evaluates a parsed expression. Booleans are integers (0 and 1). Fails on an
/// unknown identifier, an unknown machine fact, a type mismatch between an
/// integer and a string, or division by zero.
llvm::Expected<EvalValue>
evaluateExpr(const Expr &expr, const llvm::StringMap<LayoutValue> &bindings,
             const machine::MachineModel &machine,
             const LayoutContext &context);

//===----------------------------------------------------------------------===//
// Solving
//===----------------------------------------------------------------------===//

/// One legal instantiation of a layout: the parameter values that satisfy
/// every constraint, and the affine map with those values substituted in.
struct LayoutSolution {
  std::map<std::string, LayoutValue> values;
  mlir::AffineMap map;
};

/// Bounds on a solve. `truncated` in the result reports when either bound ended
/// the search early, so a caller never reads a capped result as complete.
struct SolverLimits {
  uint64_t maxAssignments = 100000;
  uint64_t maxSolutions = 8;
};

struct LayoutSolveResult {
  std::vector<LayoutSolution> solutions;
  /// True when the search stopped before exhausting the assignment space.
  bool truncated = false;
};

/// Solves `def` against `machine` by bounded enumeration over the declared
/// finite domains, in declaration order. Fails on a parameter without a
/// domain, an empty domain, a constraint that cannot be evaluated, or a map
/// clause that is not affine.
llvm::Expected<LayoutSolveResult>
solveLayout(const LayoutDef &def, const machine::MachineModel &machine,
            mlir::MLIRContext &context, const LayoutContext &layoutContext,
            const SolverLimits &limits = {});

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_LAYOUTCONSTRAINTS_H
