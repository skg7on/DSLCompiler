//===- LlkMap.h - Shared LLKMap language core (D3/D4) ---------------------===//
//
// LLKMap is one small language with several declaration kinds: layouts (D3)
// and mapping rules (D4). They share a token set, an expression grammar, and
// an evaluator, so those live here rather than being duplicated per
// declaration kind. A declaration parser derives from `LlkMapParser` and adds
// its own statements on top of the shared expression grammar.
//
// Everything in this header is target-independent: it knows about Micro
// vocabulary names and `MachineModel` queries, never about a specific target.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_LLKMAP_H
#define LLK_MAPPING_LLKMAP_H

#include "LLK/Machine/MachineModel.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace mlir::llk::mapping {

/// A concrete value an LLKMap expression can take: an integer or a symbolic
/// name. Boolean expressions evaluate to integers 0 and 1.
using LayoutValue = std::variant<int64_t, std::string>;

enum class ExprKind {
  IntLit,     ///< integer literal
  StringLit,  ///< quoted string
  Ident,      ///< parameter or builtin (`rank`, `element_type`)
  Call,       ///< `floordiv(a, b)`, `machine.compute("vector_engine")`
  MemberCall, ///< `receiver.lanes(dtype)`, `receiver.count`
  Unary,      ///< `!a`, `-a`
  Binary,     ///< `a + b`, `a % b`, `a == b`, `a && b`
  Quantifier  ///< `forall x in domain(N) : p`, `exists e in executors("dma") :
              ///< q`
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

/// The values a declaration is instantiated against: the rank and element type
/// of the value being laid out or matched. They back the `rank` and
/// `element_type` builtins.
struct LayoutContext {
  int64_t rank = 0;
  std::string elementType;
};

/// Canonical, fully-tagged rendering of `expr`: every node carries its kind and
/// its operands follow in order, so two structurally different expressions
/// never render alike. This is a hash input, not a source-faithful printer --
/// its exact spelling may change between releases, but never within one, and
/// it is dependency-free and platform-stable (design §22.1).
std::string canonicalExprString(const Expr &expr);

/// Canonical, type-tagged rendering of a value: integer `8` and string `"8"`
/// render differently, so a declaration that only changes a value's kind still
/// changes its content hash.
std::string canonicalValueString(const LayoutValue &value);

/// The result of evaluating an expression. `Handle` is the intermediate value
/// a `machine.compute(...)` / `machine.memory(...)` query produces before a
/// member query is applied; it never escapes as a solution value.
struct EvalValue {
  enum class Kind { Int, Str, Handle };
  Kind kind = Kind::Int;
  int64_t intValue = 0;
  std::string text;
};

/// A shared, bounded budget for quantified evaluation. Each candidate element a
/// quantifier examines consumes one unit. When the budget is spent before a
/// quantifier has decided, `exhausted` is set and the quantifier yields 0 --
/// *undecided*, never a definite false -- so a caller must surface `exhausted`
/// instead of reading the value as a final answer. The cap is never silently
/// ignored.
struct QuantifierBudget {
  /// Total domain elements one evaluation may examine. The solver sets this
  /// from its `SolverLimits`; a direct evaluator call uses the default.
  uint64_t limit = 65536;
  uint64_t used = 0;
  bool exhausted = false;

  /// Consumes one unit of the budget. Returns false once it is spent (setting
  /// `exhausted`), so a quantifier stops rather than running unbounded.
  bool step() {
    if (used >= limit) {
      exhausted = true;
      return false;
    }
    ++used;
    return true;
  }
};

/// Resolves a `domain(<name>)` quantifier to the finite set a declared
/// parameter ranges over, in declaration order. The evaluator carries no
/// declaration of its own, so the enclosing solver supplies one. A name with no
/// declared domain returns `std::nullopt`, which evaluation reports as an
/// error -- never a silently empty set.
using DomainResolver =
    llvm::function_ref<std::optional<llvm::ArrayRef<LayoutValue>>(
        llvm::StringRef name)>;

/// Extra inputs a quantified expression needs. The four-argument
/// `evaluateExpr` call -- everything without a quantifier -- leaves these
/// defaulted.
struct EvalOptions {
  /// Resolves `domain(<name>)` quantifiers, or empty when the caller has no
  /// declaration to resolve against (in which case such a quantifier errors).
  DomainResolver domainResolver;
  /// Budget shared across one evaluation, or null to use a private default
  /// budget -- in which case exceeding it is an error rather than a value.
  QuantifierBudget *budget = nullptr;
};

/// Evaluates a parsed expression. Booleans are integers (0 and 1). Fails on an
/// unknown identifier, an unknown machine fact, a type mismatch between an
/// integer and a string, division by zero, or -- when `options` carries no
/// budget -- a quantifier that exhausts the private default budget.
llvm::Expected<EvalValue>
evaluateExpr(const Expr &expr, const llvm::StringMap<LayoutValue> &bindings,
             const machine::MachineModel &machine, const LayoutContext &context,
             const EvalOptions &options = {});

/// Resolves the literal subject of a `machine.<callee>("<subject>")` query
/// against a machine. Returns `std::nullopt` when the subject is known, or the
/// diagnostic describing why it is not (an unknown compute kind, memory, ...).
///
/// A declaration file has no machine while it is parsed, so the check is
/// deferred: `validateExpr` accepts the query structurally, and the target
/// loader resolves the subjects against its machine once one exists (design
/// §14.4 -- an unknown capability query is a load-time failure, never a late
/// search failure).
using MachineQueryResolver = llvm::function_ref<std::optional<std::string>(
    llvm::StringRef callee, llvm::StringRef subject)>;

/// Walks `expr`, asking `resolver` about every literal `machine.compute` /
/// `machine.memory` query subject. Returns the first diagnostic, in expression
/// order, or an empty string when every subject resolves. A query whose subject
/// is not a string literal is left to evaluation, which cannot know it either.
std::string resolveMachineQueries(const Expr &expr,
                                  MachineQueryResolver resolver);

//===----------------------------------------------------------------------===//
// Tokens
//===----------------------------------------------------------------------===//

struct LlkMapToken {
  enum class Kind { Identifier, Int, String, Punct, End };
  Kind kind = Kind::End;
  std::string text;
  int64_t intValue = 0;
  unsigned line = 1;
  unsigned column = 1;
};

/// Tokenizes `text`, appending an `End` token. Fails with a message (no source
/// prefix) on an unterminated string or block comment.
llvm::Expected<std::vector<LlkMapToken>> lexLlkMap(llvm::StringRef text,
                                                   llvm::StringRef sourceName);

//===----------------------------------------------------------------------===//
// Shared parser
//===----------------------------------------------------------------------===//

/// A cursor over LLKMap tokens with the shared expression grammar and
/// diagnostics. Declaration parsers derive from it.
class LlkMapParser {
public:
  LlkMapParser(std::vector<LlkMapToken> tokens, llvm::StringRef sourceName)
      : tokens_(std::move(tokens)), source_(sourceName) {}
  virtual ~LlkMapParser() = default;

  // --- cursor -------------------------------------------------------------
  const LlkMapToken &current() const { return tokens_[pos_]; }
  const LlkMapToken &peek(size_t ahead = 1) const {
    size_t index = pos_ + ahead;
    return index < tokens_.size() ? tokens_[index] : tokens_.back();
  }
  bool atEnd() const { return current().kind == LlkMapToken::Kind::End; }
  void advance() {
    if (pos_ + 1 < tokens_.size())
      ++pos_;
  }
  bool isPunct(llvm::StringRef punct) const {
    return current().kind == LlkMapToken::Kind::Punct &&
           current().text == punct;
  }

  // --- diagnostics --------------------------------------------------------
  bool fail(const llvm::Twine &message);
  bool failAt(const LlkMapToken &token, const llvm::Twine &message);
  llvm::Error takeError();
  bool hasError() const { return !error_.empty(); }
  const std::string &errorText() const { return error_; }

  // --- terminals ----------------------------------------------------------
  bool expectPunct(llvm::StringRef punct);
  bool expectIdentifier(llvm::StringRef what, std::string &out);

  /// Parses `[ lo ".." hi ]` or `{ literal ("," literal)* }` into a value list.
  /// Shared by `param` domains in layouts and rules.
  bool parseDomainValues(std::vector<LayoutValue> &out);

  // --- expressions --------------------------------------------------------
  ExprPtr parseExpression();

  /// Validates every identifier, call, and member against the vocabulary the
  /// current declaration allows. A declaration has no machine while it is
  /// parsed, so a `machine.<query>("<subject>")` is accepted structurally here;
  /// its subject is resolved later, against the target's machine, by
  /// `resolveMachineQueries` (design §14.4).
  bool validateExpr(const ExprPtr &expr, const llvm::StringSet<> &allowed);

protected:
  ExprPtr parseOr();
  ExprPtr parseAnd();
  ExprPtr parseEquality();
  ExprPtr parseRelational();
  ExprPtr parseAdditive();
  ExprPtr parseMultiplicative();
  ExprPtr parseUnary();
  ExprPtr parsePostfix();
  ExprPtr parsePrimary();
  ExprPtr parseQuantifier();
  bool parseCallArgs(std::vector<ExprPtr> &out);

  std::vector<LlkMapToken> tokens_;
  llvm::StringRef source_;
  size_t pos_ = 0;
  std::string error_;
};

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_LLKMAP_H
