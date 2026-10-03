//===- LayoutConstraints.cpp - LLKMap layout declarations (D3) ------------===//
//
// Parses `layout` declarations on top of the shared LLKMap expression grammar
// (LlkMap.h) and solves them by bounded enumeration. Diagnostics carry file,
// line, and column.

#include "LLK/Mapping/LayoutConstraints.h"

#include "LLK/Mapping/StableHash.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"

#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace mlir::llk::mapping {

namespace {

/// Parses `layout` declarations, reusing the shared expression grammar.
class Parser : public LlkMapParser {
public:
  Parser(std::vector<LlkMapToken> tokens, llvm::StringRef source)
      : LlkMapParser(std::move(tokens), source) {}

  llvm::Expected<LayoutRegistry> parseFile();

private:
  bool parseLayout(LayoutDef &out);
  bool parseParams(LayoutDef &out);
  bool parseStatement(LayoutDef &out);
  bool parseDomain(LayoutDef &out);
  bool parseRequire(LayoutDef &out);
  bool parseMapClause(LayoutDef &out);
};

llvm::Expected<LayoutRegistry> Parser::parseFile() {
  LayoutRegistry registry;
  while (!atEnd()) {
    if (current().kind != LlkMapToken::Kind::Identifier ||
        current().text != "layout") {
      failAt(current(), "expected 'layout'");
      return takeError();
    }

    LayoutDef def;
    if (!parseLayout(def))
      return takeError();
    std::string addError;
    if (!registry.add(std::move(def), addError)) {
      fail(addError);
      return takeError();
    }
  }
  return registry;
}

bool Parser::parseLayout(LayoutDef &out) {
  advance(); // 'layout'
  if (!expectIdentifier("a layout id", out.id))
    return false;
  if (!expectPunct("("))
    return false;
  if (!parseParams(out))
    return false;
  if (!expectPunct(")"))
    return false;
  if (!expectPunct("{"))
    return false;
  while (!isPunct("}")) {
    if (atEnd())
      return failAt(current(), "expected '}' to close layout '" + out.id + "'");
    if (!parseStatement(out))
      return false;
  }
  advance(); // '}'
  return true;
}

bool Parser::parseParams(LayoutDef &out) {
  if (isPunct(")"))
    return true;
  while (true) {
    LayoutParam param;
    if (current().kind == LlkMapToken::Kind::Identifier &&
        (current().text == "int" || current().text == "sym")) {
      param.symbolic = current().text == "sym";
      advance();
    }
    if (!expectIdentifier("a parameter name", param.name))
      return false;
    for (const LayoutParam &existing : out.params)
      if (existing.name == param.name)
        return failAt(current(), "duplicate parameter '" + param.name + "'");
    out.params.push_back(std::move(param));
    if (isPunct(",")) {
      advance();
      continue;
    }
    return true;
  }
}

bool Parser::parseStatement(LayoutDef &out) {
  if (current().kind != LlkMapToken::Kind::Identifier)
    return failAt(current(), "expected a statement");
  if (current().text == "param")
    return parseDomain(out);
  if (current().text == "require")
    return parseRequire(out);
  if (current().text == "map")
    return parseMapClause(out);
  return failAt(current(), "unknown statement '" + current().text + "'");
}

bool Parser::parseDomain(LayoutDef &out) {
  advance(); // 'param'
  std::string name;
  if (!expectIdentifier("a parameter name", name))
    return false;
  if (!out.findParam(name))
    return failAt(current(), "domain for undeclared parameter '" + name + "'");
  if (out.domains.count(name))
    return failAt(current(), "duplicate domain for parameter '" + name + "'");
  if (current().kind != LlkMapToken::Kind::Identifier || current().text != "in")
    return failAt(current(), "expected 'in'");
  advance();

  std::vector<LayoutValue> values;
  if (!parseDomainValues(values))
    return false;
  if (!expectPunct(";"))
    return false;
  ParamDomain domain;
  domain.values = std::move(values);
  out.domains[name] = std::move(domain);
  return true;
}

bool Parser::parseRequire(LayoutDef &out) {
  advance(); // 'require'
  ExprPtr expr = parseExpression();
  if (!expr)
    return false;
  if (!expectPunct(";"))
    return false;

  llvm::StringSet<> allowed;
  for (const LayoutParam &param : out.params)
    allowed.insert(param.name);
  allowed.insert("rank");
  allowed.insert("element_type");
  if (!validateExpr(expr, allowed))
    return false;
  out.constraints.push_back(std::move(expr));
  return true;
}

bool Parser::parseMapClause(LayoutDef &out) {
  if (out.map)
    return failAt(current(), "duplicate map clause");
  advance(); // 'map'
  if (!expectPunct("("))
    return false;
  AffineMapSpec spec;
  if (!isPunct(")")) {
    while (true) {
      std::string dim;
      if (!expectIdentifier("a map dimension", dim))
        return false;
      spec.dims.push_back(std::move(dim));
      if (isPunct(",")) {
        advance();
        continue;
      }
      break;
    }
  }
  if (!expectPunct(")"))
    return false;
  if (!expectPunct("->"))
    return false;
  if (!expectPunct("("))
    return false;
  while (true) {
    ExprPtr result = parseExpression();
    if (!result)
      return false;
    spec.results.push_back(std::move(result));
    if (isPunct(",")) {
      advance();
      continue;
    }
    break;
  }
  if (!expectPunct(")"))
    return false;
  if (!expectPunct(";"))
    return false;

  llvm::StringSet<> allowed;
  for (const LayoutParam &param : out.params)
    allowed.insert(param.name);
  for (const std::string &dim : spec.dims)
    allowed.insert(dim);
  for (const ExprPtr &result : spec.results)
    if (!validateExpr(result, allowed))
      return false;
  out.map = std::move(spec);
  return true;
}

} // namespace

//===----------------------------------------------------------------------===//
// LayoutDef / LayoutRegistry
//===----------------------------------------------------------------------===//

const LayoutParam *LayoutDef::findParam(llvm::StringRef name) const {
  for (const LayoutParam &param : params)
    if (param.name == name)
      return &param;
  return nullptr;
}

bool LayoutDef::isSymbolic(llvm::StringRef name) const {
  const LayoutParam *param = findParam(name);
  return param && param->symbolic;
}

std::string canonicalAffineMapSpecString(const AffineMapSpec &spec) {
  std::string out;
  for (size_t index = 0; index < spec.dims.size(); ++index) {
    if (index)
      out += ',';
    out += spec.dims[index];
  }
  out += "->";
  for (size_t index = 0; index < spec.results.size(); ++index) {
    if (index)
      out += ',';
    out += spec.results[index] ? canonicalExprString(*spec.results[index])
                               : "<null>";
  }
  return out;
}

//===----------------------------------------------------------------------===//
// Source-faithful printing (design §25.3)
//===----------------------------------------------------------------------===//

std::string printAffineMapSpec(const AffineMapSpec &spec) {
  std::string out = "(";
  for (size_t index = 0; index < spec.dims.size(); ++index) {
    if (index)
      out += ", ";
    out += spec.dims[index];
  }
  out += ") -> (";
  for (size_t index = 0; index < spec.results.size(); ++index) {
    if (index)
      out += ", ";
    out += spec.results[index] ? printExpr(*spec.results[index]) : "<null>";
  }
  out += ")";
  return out;
}

std::string printParamDomain(const ParamDomain &domain) {
  // A non-empty contiguous ascending integer run prints as a range; every other
  // domain (symbolic, sparse, or unordered) prints as an enum. Both re-parse to
  // the identical value list.
  bool isRun = !domain.values.empty();
  const int64_t *previous = nullptr;
  for (const LayoutValue &value : domain.values) {
    const int64_t *integer = std::get_if<int64_t>(&value);
    if (!integer || (previous && *integer != *previous + 1)) {
      isRun = false;
      break;
    }
    previous = integer;
  }
  if (isRun)
    return "[" + std::to_string(std::get<int64_t>(domain.values.front())) +
           ".." + std::to_string(std::get<int64_t>(domain.values.back())) + "]";

  std::string out = "{";
  for (size_t index = 0; index < domain.values.size(); ++index) {
    if (index)
      out += ", ";
    out += printValue(domain.values[index]);
  }
  out += "}";
  return out;
}

std::string printLayout(const LayoutDef &def) {
  std::string out = "layout " + def.id + "(";
  for (size_t index = 0; index < def.params.size(); ++index) {
    if (index)
      out += ", ";
    out += def.params[index].symbolic ? "sym " : "int ";
    out += def.params[index].name;
  }
  out += ") {\n";
  // Domains follow parameter declaration order so `params` round-trips
  // order-for-order (the domain map is canonical by name, so its own order is
  // not semantic). A parameter with no domain is still declared by the header.
  for (const LayoutParam &param : def.params) {
    auto domain = def.domains.find(param.name);
    if (domain == def.domains.end())
      continue;
    out += "  param " + param.name + " in " + printParamDomain(domain->second) +
           ";\n";
  }
  for (const ExprPtr &constraint : def.constraints) {
    out += "  require ";
    out += constraint ? printExpr(*constraint) : "<null>";
    out += ";\n";
  }
  if (def.map)
    out += "  map " + printAffineMapSpec(*def.map) + ";\n";
  out += "}\n";
  return out;
}

namespace {

/// Canonical rendering of one layout declaration: every field that shapes the
/// declaration, with a length-prefixed line per field so no field's bytes can
/// be read as another's.
std::string canonicalLayoutDefString(const LayoutDef &def) {
  std::string out;
  auto field = [&](llvm::StringRef key, llvm::StringRef value) {
    out += key.str();
    out += ':';
    out += std::to_string(value.size());
    out += ':';
    out += value.str();
    out += '\n';
  };
  field("id", def.id);
  for (const LayoutParam &param : def.params)
    field("param", param.name + (param.symbolic ? ":symbolic" : ":integer"));
  for (const auto &entry : def.domains) {
    std::string values;
    for (const LayoutValue &value : entry.second.values) {
      if (!values.empty())
        values += ',';
      values += canonicalValueString(value);
    }
    field("domain", entry.first + "=" + values);
  }
  for (const ExprPtr &constraint : def.constraints)
    field("constraint",
          constraint ? canonicalExprString(*constraint) : "<null>");
  if (def.map)
    field("map", canonicalAffineMapSpecString(*def.map));
  return out;
}

} // namespace

bool LayoutRegistry::add(LayoutDef def, std::string &error) {
  if (find(def.id)) {
    error = "duplicate layout id '" + def.id + "'";
    return false;
  }
  defs_.push_back(std::move(def));
  return true;
}

uint64_t LayoutRegistry::computeContentHash() const {
  // `defs_` is insertion order (`all()` reports it as-is), and declaration
  // order is not part of a layout library's identity, so sort by id before
  // folding.
  std::vector<const LayoutDef *> sorted;
  sorted.reserve(defs_.size());
  for (const LayoutDef &def : defs_)
    sorted.push_back(&def);
  llvm::sort(sorted, [](const LayoutDef *lhs, const LayoutDef *rhs) {
    return lhs->id < rhs->id;
  });

  std::string canonical;
  for (const LayoutDef *def : sorted)
    canonical += canonicalLayoutDefString(*def);
  return stableHash(canonical);
}

const LayoutDef *LayoutRegistry::find(llvm::StringRef id) const {
  for (const LayoutDef &def : defs_)
    if (def.id == id)
      return &def;
  return nullptr;
}

llvm::Expected<LayoutRegistry> parseLayoutText(llvm::StringRef text,
                                               llvm::StringRef sourceName) {
  llvm::Expected<std::vector<LlkMapToken>> tokens = lexLlkMap(text, sourceName);
  if (!tokens)
    return tokens.takeError();
  Parser parser(std::move(*tokens), sourceName);
  return parser.parseFile();
}

llvm::Expected<LayoutRegistry> loadLayoutFile(llvm::StringRef path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "cannot read LLKMap file '" + path + "'");
  return parseLayoutText(buffer.get()->getBuffer(), path);
}

//===----------------------------------------------------------------------===//
// Solving
//===----------------------------------------------------------------------===//

namespace {

using machine::MachineModel;

/// Local error helper: the shared evaluator's is internal to LlkMap.cpp.
llvm::Error evalError(llvm::StringRef message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

/// Translates a `map` clause into affine expressions. Clause dimensions become
/// dims; solved integer parameters are substituted as constants, so a solution
/// carries a fully concrete map.
struct AffineBuilder {
  mlir::MLIRContext &context;
  llvm::StringMap<unsigned> dims;
  llvm::StringMap<int64_t> constants;

  llvm::Expected<mlir::AffineExpr> convert(const Expr &expr);
};

bool isConstant(const mlir::AffineExpr &expr) {
  return mlir::isa<mlir::AffineConstantExpr>(expr);
}

llvm::Expected<mlir::AffineExpr> AffineBuilder::convert(const Expr &expr) {
  switch (expr.kind) {
  case ExprKind::IntLit:
    return mlir::getAffineConstantExpr(expr.intValue, &context);
  case ExprKind::Ident: {
    auto dim = dims.find(expr.text);
    if (dim != dims.end())
      return mlir::getAffineDimExpr(dim->second, &context);
    auto constant = constants.find(expr.text);
    if (constant != constants.end())
      return mlir::getAffineConstantExpr(constant->second, &context);
    return evalError(
        "map references '" + expr.text +
        "', which is neither a dimension nor an integer parameter");
  }
  case ExprKind::Unary: {
    llvm::Expected<mlir::AffineExpr> operand = convert(*expr.operands[0]);
    if (!operand)
      return operand.takeError();
    if (expr.text == "-")
      return mlir::getAffineBinaryOpExpr(
          mlir::AffineExprKind::Mul, mlir::getAffineConstantExpr(-1, &context),
          *operand);
    return evalError("non-affine unary operator '" + expr.text + "' in map");
  }
  case ExprKind::Binary: {
    llvm::Expected<mlir::AffineExpr> lhs = convert(*expr.operands[0]);
    if (!lhs)
      return lhs.takeError();
    llvm::Expected<mlir::AffineExpr> rhs = convert(*expr.operands[1]);
    if (!rhs)
      return rhs.takeError();
    mlir::AffineExprKind kind;
    if (expr.text == "+")
      kind = mlir::AffineExprKind::Add;
    else if (expr.text == "-")
      kind = mlir::AffineExprKind::Add; // lhs + (-1 * rhs), built below
    else if (expr.text == "*") {
      // Affine multiplication is by a constant only.
      if (!isConstant(*lhs) && !isConstant(*rhs))
        return evalError("non-affine multiplication in map");
      kind = mlir::AffineExprKind::Mul;
    } else if (expr.text == "/")
      kind = mlir::AffineExprKind::FloorDiv;
    else if (expr.text == "%")
      kind = mlir::AffineExprKind::Mod;
    else
      return evalError("non-affine operator '" + expr.text + "' in map");
    if (expr.text == "-")
      return mlir::getAffineBinaryOpExpr(
          mlir::AffineExprKind::Add, *lhs,
          mlir::getAffineBinaryOpExpr(mlir::AffineExprKind::Mul,
                                      mlir::getAffineConstantExpr(-1, &context),
                                      *rhs));
    return mlir::getAffineBinaryOpExpr(kind, *lhs, *rhs);
  }
  case ExprKind::Call: {
    if (expr.operands.size() != 2)
      return evalError("'" + expr.text + "' in a map takes two arguments");
    llvm::Expected<mlir::AffineExpr> lhs = convert(*expr.operands[0]);
    if (!lhs)
      return lhs.takeError();
    llvm::Expected<mlir::AffineExpr> rhs = convert(*expr.operands[1]);
    if (!rhs)
      return rhs.takeError();
    if (expr.text == "floordiv")
      return mlir::getAffineBinaryOpExpr(mlir::AffineExprKind::FloorDiv, *lhs,
                                         *rhs);
    if (expr.text == "ceildiv")
      return mlir::getAffineBinaryOpExpr(mlir::AffineExprKind::CeilDiv, *lhs,
                                         *rhs);
    if (expr.text == "mod")
      return mlir::getAffineBinaryOpExpr(mlir::AffineExprKind::Mod, *lhs, *rhs);
    return evalError("non-affine function '" + expr.text + "' in map");
  }
  case ExprKind::StringLit:
  case ExprKind::MemberCall:
  case ExprKind::Quantifier:
    return evalError("non-affine expression in map");
  }
  return evalError("unhandled map expression");
}

} // namespace

llvm::Expected<mlir::AffineMap>
buildAffineMap(const AffineMapSpec &spec,
               const llvm::StringMap<int64_t> &constants,
               mlir::MLIRContext &context) {
  AffineBuilder builder{context, {}, constants};
  for (unsigned index = 0; index < spec.dims.size(); ++index)
    builder.dims[spec.dims[index]] = index;

  llvm::SmallVector<mlir::AffineExpr, 4> results;
  for (const ExprPtr &result : spec.results) {
    llvm::Expected<mlir::AffineExpr> expr = builder.convert(*result);
    if (!expr)
      return expr.takeError();
    results.push_back(*expr);
  }
  return mlir::AffineMap::get(spec.dims.size(), /*numSymbols=*/0, results,
                              &context);
}

namespace {

/// The bounded-enumeration implementation body: deterministic, over the
/// declared finite domains, in declaration order. Design §13.3's first
/// implementation; `BoundedLayoutSolver` wraps it behind the interface.
llvm::Expected<LayoutSolveResult>
solveBounded(const LayoutDef &def, const MachineModel &machine,
             mlir::MLIRContext &context, const LayoutContext &layoutContext,
             const SolverLimits &limits) {
  // Every declared parameter must have a finite domain to enumerate.
  llvm::SmallVector<const LayoutParam *, 4> enumerable;
  uint64_t total = 1;
  for (const LayoutParam &param : def.params) {
    auto domain = def.domains.find(param.name);
    if (domain == def.domains.end())
      return evalError("layout '" + def.id + "': parameter '" + param.name +
                       "' has no declared domain");
    uint64_t size = domain->second.values.size();
    if (size == 0)
      return evalError("layout '" + def.id + "': parameter '" + param.name +
                       "' has an empty domain");
    enumerable.push_back(&param);
    // Saturate rather than overflow when a domain is enormous.
    if (total > std::numeric_limits<uint64_t>::max() / size)
      total = std::numeric_limits<uint64_t>::max();
    else
      total *= size;
  }

  auto domainOf = [&](const LayoutParam &param) -> const ParamDomain & {
    return def.domains.find(param.name)->second;
  };

  // One quantifier budget for the whole solve, and a resolver that lets a
  // `domain(<param>)` quantifier read the declared domains (§13.3). The
  // resolver is a named lvalue: `function_ref` does not own its callable, so it
  // must outlive every `evaluateExpr` that borrows it.
  QuantifierBudget budget;
  budget.limit = limits.maxQuantifierIterations;
  auto resolveDomain =
      [&def](
          llvm::StringRef name) -> std::optional<llvm::ArrayRef<LayoutValue>> {
    auto it = def.domains.find(name.str());
    if (it == def.domains.end())
      return std::nullopt;
    return llvm::ArrayRef<LayoutValue>(it->second.values);
  };
  EvalOptions evalOptions;
  evalOptions.budget = &budget;
  evalOptions.domainResolver = resolveDomain;

  LayoutSolveResult result;
  llvm::SmallVector<size_t, 4> index(enumerable.size(), 0);
  uint64_t assignments = 0;
  bool exhausted = false;
  while (!exhausted) {
    if (assignments >= limits.maxAssignments)
      break;

    llvm::StringMap<LayoutValue> bindings;
    for (size_t i = 0; i < enumerable.size(); ++i)
      bindings[enumerable[i]->name] = domainOf(*enumerable[i]).values[index[i]];
    ++assignments;

    bool satisfied = true;
    for (const ExprPtr &constraint : def.constraints) {
      llvm::Expected<EvalValue> value = evaluateExpr(
          *constraint, bindings, machine, layoutContext, evalOptions);
      if (!value)
        return value.takeError();
      if (value->kind != EvalValue::Kind::Int || value->intValue == 0) {
        satisfied = false;
        break;
      }
    }

    if (satisfied) {
      LayoutSolution solution;
      for (const auto &entry : bindings)
        solution.values[entry.first().str()] = entry.second;
      if (def.map) {
        llvm::StringMap<int64_t> constants;
        for (const auto &entry : bindings)
          if (const auto *value = std::get_if<int64_t>(&entry.second))
            constants[entry.first()] = *value;
        llvm::Expected<mlir::AffineMap> map =
            buildAffineMap(*def.map, constants, context);
        if (!map)
          return map.takeError();
        solution.map = *map;
      }
      result.solutions.push_back(std::move(solution));
      if (result.solutions.size() >= limits.maxSolutions)
        break;
    }

    // Advance the odometer: the last declared parameter varies fastest.
    exhausted = true;
    for (size_t i = enumerable.size(); i-- > 0;) {
      if (++index[i] < domainOf(*enumerable[i]).values.size()) {
        exhausted = false;
        break;
      }
      index[i] = 0;
    }
  }

  // The search is incomplete both when the assignment space was not exhausted
  // and when a quantifier ran out of budget (design §13.3). Exhausting the
  // quantifier budget is stronger: it leaves a constraint *undecided*, so the
  // solutions may not be legal and are withheld -- a consumer must never read
  // an undecided solve as "these layouts are legal, there may be more".
  result.undecided = budget.exhausted;
  if (result.undecided)
    result.solutions.clear();
  result.truncated = assignments < total || budget.exhausted;
  return result;
}

/// The interface wrapper around `solveBounded`.
class BoundedLayoutSolver final : public LayoutSolver {
public:
  llvm::Expected<LayoutSolveResult>
  solve(const LayoutDef &def, const MachineModel &machine,
        mlir::MLIRContext &context, const LayoutContext &layoutContext,
        const SolverLimits &limits) const override {
    return solveBounded(def, machine, context, layoutContext, limits);
  }
};

} // namespace

std::unique_ptr<LayoutSolver> makeBoundedLayoutSolver() {
  return std::make_unique<BoundedLayoutSolver>();
}

llvm::Expected<LayoutSolveResult>
solveLayout(const LayoutDef &def, const MachineModel &machine,
            mlir::MLIRContext &context, const LayoutContext &layoutContext,
            const SolverLimits &limits) {
  // Preserved as a thin wrapper; new clients take the `LayoutSolver` interface.
  return makeBoundedLayoutSolver()->solve(def, machine, context, layoutContext,
                                          limits);
}

} // namespace mlir::llk::mapping
