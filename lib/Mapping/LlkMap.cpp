//===- LlkMap.cpp - Shared LLKMap language core (D3/D4) ------------------===//
//
// The lexer, expression grammar, and evaluator shared by every LLKMap
// declaration kind. Diagnostics carry file, line, and column. Identifier
// validation happens here rather than at solve time, so an unknown parameter,
// call, or machine query is a load-time failure (design 14.4).

#include "LLK/Mapping/LlkMap.h"

#include "LLK/Dialect/Micro/MicroEnums.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <optional>
#include <string>
#include <variant>

namespace mlir::llk::mapping {
namespace {

using machine::MachineModel;

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

bool isIdentifierStart(char c) {
  return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}

bool isIdentifierChar(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.';
}

/// Tokenizes `text`. Returns false with a message in `error` on an unterminated
/// string or block comment.
bool lex(llvm::StringRef text, std::vector<LlkMapToken> &out,
         std::string &error) {
  size_t index = 0;
  unsigned line = 1;
  unsigned column = 1;
  auto advance = [&](size_t count) {
    for (size_t i = 0; i < count; ++i) {
      if (text[index + i] == '\n') {
        ++line;
        column = 1;
      } else {
        ++column;
      }
    }
    index += count;
  };

  while (index < text.size()) {
    char c = text[index];
    if (c == '\n' || c == ' ' || c == '\t' || c == '\r') {
      advance(1);
      continue;
    }
    if (text.substr(index).starts_with("//")) {
      size_t end = text.find('\n', index);
      advance((end == llvm::StringRef::npos ? text.size() : end) - index);
      continue;
    }
    if (text.substr(index).starts_with("/*")) {
      size_t end = text.find("*/", index + 2);
      if (end == llvm::StringRef::npos) {
        error = "unterminated block comment";
        return false;
      }
      advance(end + 2 - index);
      continue;
    }

    LlkMapToken token;
    token.line = line;
    token.column = column;

    if (isIdentifierStart(c)) {
      size_t end = index + 1;
      while (end < text.size() && isIdentifierChar(text[end]))
        ++end;
      token.kind = LlkMapToken::Kind::Identifier;
      token.text = text.substr(index, end - index).str();
      advance(end - index);
      out.push_back(std::move(token));
      continue;
    }

    if (std::isdigit(static_cast<unsigned char>(c))) {
      size_t end = index;
      while (end < text.size() &&
             std::isdigit(static_cast<unsigned char>(text[end])))
        ++end;
      token.kind = LlkMapToken::Kind::Int;
      token.text = text.substr(index, end - index).str();
      token.intValue = std::stoll(token.text);
      advance(end - index);
      out.push_back(std::move(token));
      continue;
    }

    if (c == '"') {
      size_t end = index + 1;
      while (end < text.size() && text[end] != '"')
        ++end;
      if (end == text.size()) {
        error = "unterminated string literal";
        return false;
      }
      token.kind = LlkMapToken::Kind::String;
      token.text = text.substr(index + 1, end - index - 1).str();
      advance(end + 1 - index);
      out.push_back(std::move(token));
      continue;
    }

    // Two-character punctuation first, so `..` and `==` win over `.` and `=`.
    static const char *kTwoChar[] = {
        "..", "->", "==", "!=", "<=", ">=", "&&", "||"};
    bool matched = false;
    for (const char *punct : kTwoChar) {
      if (text.substr(index).starts_with(punct)) {
        token.kind = LlkMapToken::Kind::Punct;
        token.text = punct;
        advance(2);
        out.push_back(std::move(token));
        matched = true;
        break;
      }
    }
    if (matched)
      continue;

    static const std::string kOneChar = "(){}[],;.<>!+-*/%=:";
    if (kOneChar.find(c) == std::string::npos) {
      error = (llvm::Twine("unexpected character '") +
               llvm::Twine(std::string(1, c)) + "'")
                  .str();
      return false;
    }
    token.kind = LlkMapToken::Kind::Punct;
    token.text = std::string(1, c);
    advance(1);
    out.push_back(std::move(token));
  }

  LlkMapToken end;
  end.kind = LlkMapToken::Kind::End;
  end.line = line;
  end.column = column;
  out.push_back(std::move(end));
  return true;
}

//===----------------------------------------------------------------------===//
// Identifier vocabulary
//===----------------------------------------------------------------------===//

/// Call targets the evaluator understands.
bool isKnownCall(llvm::StringRef name) {
  return name == "floordiv" || name == "ceildiv" || name == "mod" ||
         name == "min" || name == "max" || name == "machine.compute" ||
         name == "machine.memory";
}

/// Member queries the evaluator understands.
bool isKnownMember(llvm::StringRef name) {
  return name == "lanes" || name == "count" || name == "capacity_bytes" ||
         name == "alignment_bytes";
}

namespace {
ExprPtr makeExpr(ExprKind kind, std::string text,
                 std::vector<ExprPtr> operands) {
  auto expr = std::make_shared<Expr>();
  expr->kind = kind;
  expr->text = std::move(text);
  expr->operands = std::move(operands);
  return expr;
}
} // namespace

llvm::Error evalError(llvm::StringRef message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
}

EvalValue makeInt(int64_t value) {
  EvalValue result;
  result.kind = EvalValue::Kind::Int;
  result.intValue = value;
  return result;
}

EvalValue makeStr(std::string value) {
  EvalValue result;
  result.kind = EvalValue::Kind::Str;
  result.text = std::move(value);
  return result;
}

EvalValue makeHandle(std::string value) {
  EvalValue result;
  result.kind = EvalValue::Kind::Handle;
  result.text = std::move(value);
  return result;
}

using EvalFn = llvm::function_ref<llvm::Expected<EvalValue>(const Expr &)>;

llvm::Expected<EvalValue> evalBinary(const Expr &expr, EvalFn eval) {
  llvm::Expected<EvalValue> lhs = eval(*expr.operands[0]);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<EvalValue> rhs = eval(*expr.operands[1]);
  if (!rhs)
    return rhs.takeError();

  const std::string &op = expr.text;
  if (op == "&&" || op == "||") {
    if (lhs->kind != EvalValue::Kind::Int || rhs->kind != EvalValue::Kind::Int)
      return evalError("'" + op + "' requires integer operands");
    bool left = lhs->intValue != 0;
    bool right = rhs->intValue != 0;
    return makeInt(op == "&&" ? (left && right) : (left || right));
  }

  if (op == "==" || op == "!=") {
    if (lhs->kind != rhs->kind)
      return evalError("cannot compare an integer with a string");
    bool equal = lhs->kind == EvalValue::Kind::Int
                     ? lhs->intValue == rhs->intValue
                     : lhs->text == rhs->text;
    return makeInt((op == "==") == equal ? 1 : 0);
  }

  if (lhs->kind != EvalValue::Kind::Int || rhs->kind != EvalValue::Kind::Int)
    return evalError("'" + op + "' requires integer operands");
  int64_t left = lhs->intValue;
  int64_t right = rhs->intValue;
  if (op == "<")
    return makeInt(left < right);
  if (op == "<=")
    return makeInt(left <= right);
  if (op == ">")
    return makeInt(left > right);
  if (op == ">=")
    return makeInt(left >= right);
  if (op == "+")
    return makeInt(left + right);
  if (op == "-")
    return makeInt(left - right);
  if (op == "*")
    return makeInt(left * right);
  if (op == "/" || op == "%") {
    if (right == 0)
      return evalError("division by zero");
    return makeInt(op == "/" ? left / right : left % right);
  }
  return evalError("unknown operator '" + op + "'");
}

llvm::Expected<EvalValue> evalCall(const Expr &expr, EvalFn eval) {
  const std::string &callee = expr.text;
  if (callee == "machine.compute" || callee == "machine.memory") {
    if (expr.operands.size() != 1)
      return evalError("'" + callee + "' takes exactly one argument");
    llvm::Expected<EvalValue> argument = eval(*expr.operands[0]);
    if (!argument)
      return argument.takeError();
    if (argument->kind != EvalValue::Kind::Str)
      return evalError("'" + callee + "' needs a string argument");
    return makeHandle(callee + ":" + argument->text);
  }

  if (expr.operands.size() != 2)
    return evalError("'" + callee + "' takes exactly two arguments");
  llvm::Expected<EvalValue> lhs = eval(*expr.operands[0]);
  if (!lhs)
    return lhs.takeError();
  llvm::Expected<EvalValue> rhs = eval(*expr.operands[1]);
  if (!rhs)
    return rhs.takeError();
  if (lhs->kind != EvalValue::Kind::Int || rhs->kind != EvalValue::Kind::Int)
    return evalError("'" + callee + "' requires integer arguments");
  int64_t left = lhs->intValue;
  int64_t right = rhs->intValue;

  if (callee == "floordiv" || callee == "mod") {
    if (right == 0)
      return evalError("division by zero");
    return makeInt(callee == "floordiv" ? left / right : left % right);
  }
  if (callee == "ceildiv") {
    if (right <= 0)
      return evalError("'ceildiv' needs a positive divisor");
    return makeInt((left + right - 1) / right);
  }
  if (callee == "min")
    return makeInt(std::min(left, right));
  if (callee == "max")
    return makeInt(std::max(left, right));
  return evalError("unknown function '" + callee + "'");
}

llvm::Expected<EvalValue> evalMember(const Expr &expr, EvalFn eval,
                                     const MachineModel &machine) {
  llvm::Expected<EvalValue> receiver = eval(*expr.operands[0]);
  if (!receiver)
    return receiver.takeError();
  if (receiver->kind != EvalValue::Kind::Handle)
    return evalError("'" + expr.text + "' applied to a non-machine value");

  // The handle text is "<machine.compute|machine.memory>:<subject>".
  size_t colon = receiver->text.find(':');
  if (colon == std::string::npos)
    return evalError("malformed machine handle");
  bool isCompute = receiver->text.starts_with("machine.compute:");
  llvm::StringRef subject = llvm::StringRef(receiver->text).substr(colon + 1);
  const std::string &member = expr.text;

  if (member == "count") {
    if (!isCompute)
      return evalError("'count' applies to a compute capability");
    int64_t count = 0;
    for (const auto &compute : machine.computes)
      if (compute.kind == subject)
        ++count;
    if (count == 0)
      return evalError("unknown compute kind '" + subject.str() + "'");
    return makeInt(count);
  }

  if (member == "lanes") {
    if (!isCompute)
      return evalError("'lanes' applies to a compute capability");
    if (expr.operands.size() != 2)
      return evalError("'lanes' takes one dtype argument");
    llvm::Expected<EvalValue> dtype = eval(*expr.operands[1]);
    if (!dtype)
      return dtype.takeError();
    if (dtype->kind != EvalValue::Kind::Str)
      return evalError("'lanes' needs a string dtype");
    std::optional<int64_t> lanes = machine.lanesFor(subject, dtype->text);
    if (!lanes)
      return evalError("compute '" + subject.str() +
                       "' does not model dtype '" + dtype->text + "'");
    return makeInt(*lanes);
  }

  if (member == "capacity_bytes" || member == "alignment_bytes") {
    if (isCompute)
      return evalError("'" + member + "' applies to a memory, not a compute");
    const machine::MemoryNode *memory = machine.findMemory(subject);
    if (!memory)
      return evalError("unknown memory '" + subject.str() + "'");
    return makeInt(member == "capacity_bytes"
                       ? static_cast<int64_t>(memory->capacityBytes)
                       : static_cast<int64_t>(memory->alignmentBytes));
  }

  return evalError("unknown machine query '" + member + "'");
}

} // namespace

bool LlkMapParser::fail(const llvm::Twine &message) {
  if (error_.empty())
    error_ = (source_ + ": error: " + message).str();
  return false;
}

bool LlkMapParser::failAt(const LlkMapToken &token,
                          const llvm::Twine &message) {
  if (error_.empty())
    error_ = (source_ + ":" + llvm::Twine(token.line) + ":" +
              llvm::Twine(token.column) + ": error: " + message)
                 .str();
  return false;
}

llvm::Error LlkMapParser::takeError() {
  return llvm::make_error<llvm::StringError>(std::move(error_),
                                             llvm::inconvertibleErrorCode());
}

bool LlkMapParser::expectPunct(llvm::StringRef punct) {
  if (!isPunct(punct))
    return failAt(current(), "expected '" + punct + "'");
  advance();
  return true;
}

bool LlkMapParser::expectIdentifier(llvm::StringRef what, std::string &out) {
  if (current().kind != LlkMapToken::Kind::Identifier)
    return failAt(current(), "expected " + what);
  out = current().text;
  advance();
  return true;
}

bool LlkMapParser::validateExpr(const ExprPtr &expr,
                                const llvm::StringSet<> &allowed) {
  switch (expr->kind) {
  case ExprKind::IntLit:
  case ExprKind::StringLit:
    return true;
  case ExprKind::Ident:
    if (!allowed.contains(expr->text))
      return failAt(current(), "unknown identifier '" + expr->text + "'");
    return true;
  case ExprKind::Call:
    if (!isKnownCall(expr->text))
      return failAt(current(), "unknown function '" + expr->text + "'");
    break;
  case ExprKind::MemberCall:
    if (!isKnownMember(expr->text))
      return failAt(current(), "unknown machine query '" + expr->text + "'");
    break;
  case ExprKind::Unary:
  case ExprKind::Binary:
    break;
  case ExprKind::Quantifier: {
    // operands: [bound-variable Ident, domain, body].
    const Expr &domain = *expr->operands[1];
    if (domain.kind == ExprKind::Ident) {
      if (domain.text != "dimensions")
        return failAt(current(),
                      "unknown quantifier domain '" + domain.text + "'");
    } else if (domain.kind == ExprKind::Call && domain.text == "domain") {
      if (domain.operands.size() != 1 ||
          domain.operands[0]->kind != ExprKind::Ident)
        return failAt(current(), "'domain' needs a declared parameter name");
      if (!allowed.contains(domain.operands[0]->text))
        return failAt(current(), "unknown parameter '" +
                                     domain.operands[0]->text +
                                     "' in quantifier domain");
    } else if (domain.kind == ExprKind::Call && domain.text == "executors") {
      if (domain.operands.size() != 1)
        return failAt(current(), "'executors' takes a kind argument");
      if (!validateExpr(domain.operands[0], allowed))
        return false;
    } else {
      return failAt(current(), "expected a quantifier domain "
                               "('domain(<param>)', 'executors(<kind>)', or "
                               "'dimensions')");
    }
    // The bound variable is local to the body and shadows any outer name.
    llvm::StringSet<> scoped = allowed;
    scoped.insert(expr->operands[0]->text);
    return validateExpr(expr->operands[2], scoped);
  }
  }
  for (const ExprPtr &operand : expr->operands)
    if (!validateExpr(operand, allowed))
      return false;
  return true;
}

ExprPtr LlkMapParser::parseExpression() { return parseOr(); }

ExprPtr LlkMapParser::parseOr() {
  ExprPtr lhs = parseAnd();
  if (!lhs)
    return nullptr;
  while (isPunct("||")) {
    advance();
    ExprPtr rhs = parseAnd();
    if (!rhs)
      return nullptr;
    lhs = makeExpr(ExprKind::Binary, "||", {lhs, rhs});
  }
  return lhs;
}

ExprPtr LlkMapParser::parseAnd() {
  ExprPtr lhs = parseEquality();
  if (!lhs)
    return nullptr;
  while (isPunct("&&")) {
    advance();
    ExprPtr rhs = parseEquality();
    if (!rhs)
      return nullptr;
    lhs = makeExpr(ExprKind::Binary, "&&", {lhs, rhs});
  }
  return lhs;
}

ExprPtr LlkMapParser::parseEquality() {
  ExprPtr lhs = parseRelational();
  if (!lhs)
    return nullptr;
  while (isPunct("==") || isPunct("!=")) {
    std::string op = current().text;
    advance();
    ExprPtr rhs = parseRelational();
    if (!rhs)
      return nullptr;
    lhs = makeExpr(ExprKind::Binary, op, {lhs, rhs});
  }
  return lhs;
}

ExprPtr LlkMapParser::parseRelational() {
  ExprPtr lhs = parseAdditive();
  if (!lhs)
    return nullptr;
  while (isPunct("<") || isPunct("<=") || isPunct(">") || isPunct(">=")) {
    std::string op = current().text;
    advance();
    ExprPtr rhs = parseAdditive();
    if (!rhs)
      return nullptr;
    lhs = makeExpr(ExprKind::Binary, op, {lhs, rhs});
  }
  return lhs;
}

ExprPtr LlkMapParser::parseAdditive() {
  ExprPtr lhs = parseMultiplicative();
  if (!lhs)
    return nullptr;
  while (isPunct("+") || isPunct("-")) {
    std::string op = current().text;
    advance();
    ExprPtr rhs = parseMultiplicative();
    if (!rhs)
      return nullptr;
    lhs = makeExpr(ExprKind::Binary, op, {lhs, rhs});
  }
  return lhs;
}

ExprPtr LlkMapParser::parseMultiplicative() {
  ExprPtr lhs = parseUnary();
  if (!lhs)
    return nullptr;
  while (isPunct("*") || isPunct("/") || isPunct("%")) {
    std::string op = current().text;
    advance();
    ExprPtr rhs = parseUnary();
    if (!rhs)
      return nullptr;
    lhs = makeExpr(ExprKind::Binary, op, {lhs, rhs});
  }
  return lhs;
}

ExprPtr LlkMapParser::parseUnary() {
  if (isPunct("!") || isPunct("-")) {
    std::string op = current().text;
    advance();
    ExprPtr operand = parseUnary();
    if (!operand)
      return nullptr;
    return makeExpr(ExprKind::Unary, op, {operand});
  }
  return parsePostfix();
}

bool LlkMapParser::parseCallArgs(std::vector<ExprPtr> &out) {
  if (!expectPunct("("))
    return false;
  if (isPunct(")")) {
    advance();
    return true;
  }
  while (true) {
    ExprPtr arg = parseExpression();
    if (!arg)
      return false;
    out.push_back(std::move(arg));
    if (isPunct(",")) {
      advance();
      continue;
    }
    break;
  }
  if (!expectPunct(")"))
    return false;
  return true;
}

ExprPtr LlkMapParser::parsePostfix() {
  ExprPtr expr = parsePrimary();
  if (!expr)
    return nullptr;
  while (isPunct(".")) {
    advance();
    std::string member;
    if (!expectIdentifier("a member name", member))
      return nullptr;
    std::vector<ExprPtr> args;
    // A member query may take arguments (`.lanes(dtype)`) or be a property
    // (`.count`, `.capacity_bytes`).
    if (isPunct("("))
      if (!parseCallArgs(args))
        return nullptr;
    args.insert(args.begin(), expr);
    expr = makeExpr(ExprKind::MemberCall, std::move(member), std::move(args));
  }
  return expr;
}

ExprPtr LlkMapParser::parsePrimary() {
  if (current().kind == LlkMapToken::Kind::Int) {
    auto expr = std::make_shared<Expr>();
    expr->kind = ExprKind::IntLit;
    expr->intValue = current().intValue;
    advance();
    return expr;
  }
  if (current().kind == LlkMapToken::Kind::String) {
    auto expr = std::make_shared<Expr>();
    expr->kind = ExprKind::StringLit;
    expr->text = current().text;
    advance();
    return expr;
  }
  if (isPunct("(")) {
    advance();
    ExprPtr inner = parseExpression();
    if (!inner)
      return nullptr;
    if (!expectPunct(")"))
      return nullptr;
    return inner;
  }
  if (current().kind == LlkMapToken::Kind::Identifier) {
    if (current().text == "forall" || current().text == "exists")
      return parseQuantifier();
    std::string name = current().text;
    advance();
    if (isPunct("(")) {
      std::vector<ExprPtr> args;
      if (!parseCallArgs(args))
        return nullptr;
      return makeExpr(ExprKind::Call, std::move(name), std::move(args));
    }
    return makeExpr(ExprKind::Ident, std::move(name), {});
  }
  failAt(current(), "expected an expression");
  return nullptr;
}

ExprPtr LlkMapParser::parseQuantifier() {
  std::string op = current().text; // `forall` or `exists`
  advance();
  std::string variable;
  if (!expectIdentifier("a quantifier variable", variable))
    return nullptr;
  if (current().kind != LlkMapToken::Kind::Identifier ||
      current().text != "in") {
    failAt(current(), "expected 'in'");
    return nullptr;
  }
  advance();

  // The domain is parsed as an ordinary primary-level expression and its shape
  // is checked by `validateExpr`; `domain(...)`, `executors(...)`, and
  // `dimensions` are understood only in this position.
  ExprPtr domain = parseExpression();
  if (!domain)
    return nullptr;
  if (!expectPunct(":"))
    return nullptr;
  // The body extends as far right as possible, so a quantifier that must feed
  // a larger expression needs parentheses.
  ExprPtr body = parseExpression();
  if (!body)
    return nullptr;
  return makeExpr(
      ExprKind::Quantifier, std::move(op),
      {makeExpr(ExprKind::Ident, std::move(variable), {}), domain, body});
}

namespace {

llvm::Expected<EvalValue> evalImpl(const Expr &expr,
                                   const llvm::StringMap<LayoutValue> &bindings,
                                   const MachineModel &machine,
                                   const LayoutContext &context,
                                   const EvalOptions &options);

/// Evaluates a `forall`/`exists` over its finite domain. The domain is either
/// a declared parameter's domain (`domain(<param>)`), a machine-derived set of
/// executor ids (`executors(<kind>)`), or the value's logical dimensions
/// (`dimensions`, the integers `0 .. rank-1`). An empty domain is vacuous:
/// `forall` is 1 and `exists` is 0. Whether it is decisive is bounded by
/// `options.budget`; when the budget runs out the quantifier yields 0 and sets
/// `budget->exhausted`, which the caller must surface rather than accept.
llvm::Expected<EvalValue>
evalQuantifier(const Expr &expr, const llvm::StringMap<LayoutValue> &bindings,
               const MachineModel &machine, const LayoutContext &context,
               const EvalOptions &options) {
  const Expr &variable = *expr.operands[0];
  const Expr &domainNode = *expr.operands[1];
  const Expr &body = *expr.operands[2];
  bool isForall = expr.text == "forall";

  llvm::SmallVector<LayoutValue, 8> domain;
  if (domainNode.kind == ExprKind::Ident && domainNode.text == "dimensions") {
    for (int64_t dimension = 0; dimension < context.rank; ++dimension)
      domain.push_back(dimension);
  } else if (domainNode.kind == ExprKind::Call && domainNode.text == "domain") {
    if (!options.domainResolver)
      return evalError("'domain' quantifier needs a declaration to resolve");
    std::optional<llvm::ArrayRef<LayoutValue>> values =
        options.domainResolver(domainNode.operands[0]->text);
    if (!values)
      return evalError("unknown domain for parameter '" +
                       domainNode.operands[0]->text + "'");
    domain.assign(values->begin(), values->end());
  } else if (domainNode.kind == ExprKind::Call &&
             domainNode.text == "executors") {
    llvm::Expected<EvalValue> kind =
        evalImpl(*domainNode.operands[0], bindings, machine, context, options);
    if (!kind)
      return kind.takeError();
    if (kind->kind != EvalValue::Kind::Str)
      return evalError("'executors' needs a string kind");
    for (const machine::ExecutorNode &executor : machine.executors)
      if (machine.ownerMatches(kind->text, executor.id))
        domain.push_back(executor.id);
    // A kind with no matching executor is an unknown fact, never a silently
    // empty set. Distinguish a kind outside the vocabulary from a valid owner
    // kind the profile simply does not populate: the first is a typo, the
    // second a machine limitation.
    if (domain.empty()) {
      if (!mlir::micro::symbolizeOwner(kind->text))
        return evalError("unknown executor kind '" + kind->text + "'");
      return evalError("the machine offers no executors of kind '" +
                       kind->text + "'");
    }
  } else {
    return evalError("unknown quantifier domain");
  }

  if (!options.budget)
    return evalError("quantifier evaluated without a bound");

  llvm::StringMap<LayoutValue> scoped = bindings;
  for (const LayoutValue &value : domain) {
    if (!options.budget->step())
      return makeInt(0); // undecided; flagged by budget->exhausted
    scoped[variable.text] = value;
    llvm::Expected<EvalValue> bodyValue =
        evalImpl(body, scoped, machine, context, options);
    if (!bodyValue)
      return bodyValue.takeError();
    if (bodyValue->kind != EvalValue::Kind::Int)
      return evalError("a quantifier body must be an integer");
    bool truthy = bodyValue->intValue != 0;
    if (isForall && !truthy)
      return makeInt(0); // counterexample
    if (!isForall && truthy)
      return makeInt(1); // witness
  }
  return makeInt(isForall ? 1 : 0);
}

llvm::Expected<EvalValue> evalImpl(const Expr &expr,
                                   const llvm::StringMap<LayoutValue> &bindings,
                                   const MachineModel &machine,
                                   const LayoutContext &context,
                                   const EvalOptions &options) {
  auto eval = [&](const Expr &node) {
    return evalImpl(node, bindings, machine, context, options);
  };

  switch (expr.kind) {
  case ExprKind::IntLit:
    return makeInt(expr.intValue);
  case ExprKind::StringLit:
    return makeStr(expr.text);
  case ExprKind::Ident: {
    auto it = bindings.find(expr.text);
    if (it != bindings.end()) {
      const LayoutValue &value = it->second;
      if (const auto *integer = std::get_if<int64_t>(&value))
        return makeInt(*integer);
      return makeStr(std::get<std::string>(value));
    }
    if (expr.text == "rank")
      return makeInt(context.rank);
    if (expr.text == "element_type")
      return makeStr(context.elementType);
    return evalError("unknown identifier '" + expr.text + "'");
  }
  case ExprKind::Unary: {
    llvm::Expected<EvalValue> operand = eval(*expr.operands[0]);
    if (!operand)
      return operand.takeError();
    if (operand->kind != EvalValue::Kind::Int)
      return evalError("'" + expr.text + "' requires an integer operand");
    if (expr.text == "!")
      return makeInt(operand->intValue == 0 ? 1 : 0);
    return makeInt(-operand->intValue);
  }
  case ExprKind::Binary:
    return evalBinary(expr, eval);
  case ExprKind::Call:
    return evalCall(expr, eval);
  case ExprKind::MemberCall:
    return evalMember(expr, eval, machine);
  case ExprKind::Quantifier:
    return evalQuantifier(expr, bindings, machine, context, options);
  }
  return evalError("unhandled expression");
}

} // namespace

llvm::Expected<EvalValue>
evaluateExpr(const Expr &expr, const llvm::StringMap<LayoutValue> &bindings,
             const MachineModel &machine, const LayoutContext &context,
             const EvalOptions &options) {
  if (options.budget)
    return evalImpl(expr, bindings, machine, context, options);

  // A caller with no budget of its own still gets a bounded evaluation; an
  // exhausted default budget is an error, not a quietly false value.
  QuantifierBudget local;
  EvalOptions scoped = options;
  scoped.budget = &local;
  llvm::Expected<EvalValue> result =
      evalImpl(expr, bindings, machine, context, scoped);
  if (!result)
    return result.takeError();
  if (local.exhausted)
    return evalError("quantified expression exceeded its evaluation bound; "
                     "the result is undecided");
  return result;
}

std::string resolveMachineQueries(const Expr &expr,
                                  MachineQueryResolver resolver) {
  if (expr.kind == ExprKind::Call &&
      (expr.text == "machine.compute" || expr.text == "machine.memory") &&
      expr.operands.size() == 1 &&
      expr.operands[0]->kind == ExprKind::StringLit) {
    if (std::optional<std::string> message =
            resolver(expr.text, expr.operands[0]->text))
      return *message;
  }
  for (const ExprPtr &operand : expr.operands) {
    std::string message = resolveMachineQueries(*operand, resolver);
    if (!message.empty())
      return message;
  }
  return {};
}

bool LlkMapParser::parseDomainValues(std::vector<LayoutValue> &out) {
  if (isPunct("[")) {
    advance();
    if (current().kind != LlkMapToken::Kind::Int)
      return failAt(current(), "expected an integer lower bound");
    int64_t lower = current().intValue;
    advance();
    if (!expectPunct(".."))
      return false;
    if (current().kind != LlkMapToken::Kind::Int)
      return failAt(current(), "expected an integer upper bound");
    int64_t upper = current().intValue;
    advance();
    if (!expectPunct("]"))
      return false;
    if (upper < lower)
      return failAt(current(), "empty integer range");
    for (int64_t value = lower;; ++value) {
      out.push_back(value);
      // Stop before incrementing the inclusive upper bound: it may be
      // INT64_MAX.
      if (value == upper)
        break;
    }
    return true;
  }
  if (isPunct("{")) {
    advance();
    while (true) {
      if (current().kind == LlkMapToken::Kind::Int) {
        out.push_back(current().intValue);
        advance();
      } else if (current().kind == LlkMapToken::Kind::String) {
        out.push_back(current().text);
        advance();
      } else {
        return failAt(current(), "expected a literal domain member");
      }
      if (isPunct(",")) {
        advance();
        continue;
      }
      break;
    }
    return expectPunct("}");
  }
  return failAt(current(), "expected '[' or '{'");
}

llvm::Expected<std::vector<LlkMapToken>> lexLlkMap(llvm::StringRef text,
                                                   llvm::StringRef sourceName) {
  std::vector<LlkMapToken> tokens;
  std::string error;
  if (!lex(text, tokens, error))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   sourceName + ": error: " + error);
  return tokens;
}

//===----------------------------------------------------------------------===//
// Canonical rendering for content hashing
//===----------------------------------------------------------------------===//

std::string canonicalExprString(const Expr &expr) {
  switch (expr.kind) {
  case ExprKind::IntLit:
    return "int:" + std::to_string(expr.intValue);
  case ExprKind::StringLit:
    // Length-prefixed so a string's own bytes cannot be mistaken for a
    // separator between fields.
    return "str:" + std::to_string(expr.text.size()) + ":" + expr.text;
  case ExprKind::Ident:
    return "id:" + expr.text;
  case ExprKind::Call:
    break;
  case ExprKind::MemberCall:
    break;
  case ExprKind::Unary:
    break;
  case ExprKind::Binary:
    break;
  case ExprKind::Quantifier:
    break;
  }

  llvm::StringRef prefix;
  switch (expr.kind) {
  case ExprKind::Call:
    prefix = "call:";
    break;
  case ExprKind::MemberCall:
    prefix = "member:";
    break;
  case ExprKind::Unary:
    prefix = "unary:";
    break;
  case ExprKind::Binary:
    prefix = "binary:";
    break;
  case ExprKind::Quantifier:
    prefix = "quant:";
    break;
  default:
    break;
  }
  std::string out = prefix.str();
  out += expr.text;
  out += '(';
  for (size_t index = 0; index < expr.operands.size(); ++index) {
    if (index)
      out += ',';
    out += expr.operands[index] ? canonicalExprString(*expr.operands[index])
                                : "<null>";
  }
  out += ')';
  return out;
}

std::string canonicalValueString(const LayoutValue &value) {
  if (const int64_t *integer = std::get_if<int64_t>(&value))
    return "i:" + std::to_string(*integer);
  const std::string &text = std::get<std::string>(value);
  return "s:" + std::to_string(text.size()) + ":" + text;
}

//===----------------------------------------------------------------------===//
// Source-faithful printing
//===----------------------------------------------------------------------===//

std::string printValue(const LayoutValue &value) {
  if (const int64_t *integer = std::get_if<int64_t>(&value))
    return std::to_string(*integer);
  // A symbolic value is quoted so any name re-lexes as a string literal, never
  // as a keyword or a number.
  return "\"" + std::get<std::string>(value) + "\"";
}

namespace {

/// True when `expr` prints as a compound the printer parenthesizes, so a
/// surrounding construct cannot re-associate it on re-parse.
bool isCompoundExpr(const Expr &expr) {
  return expr.kind == ExprKind::Unary || expr.kind == ExprKind::Binary ||
         expr.kind == ExprKind::Quantifier;
}

std::string printExprNode(const ExprPtr &expr) {
  return expr ? printExpr(*expr) : "<null>";
}

} // namespace

std::string printExpr(const Expr &expr) {
  switch (expr.kind) {
  case ExprKind::IntLit:
    return std::to_string(expr.intValue);
  case ExprKind::StringLit:
    return "\"" + expr.text + "\"";
  case ExprKind::Ident:
    return expr.text;
  case ExprKind::Call: {
    std::string out = expr.text + "(";
    for (size_t index = 0; index < expr.operands.size(); ++index) {
      if (index)
        out += ", ";
      out += printExprNode(expr.operands[index]);
    }
    out += ")";
    return out;
  }
  case ExprKind::MemberCall: {
    // `operands[0]` is the receiver; the rest are the member's arguments.
    std::string receiver = expr.operands.empty()
                               ? std::string("<null>")
                               : printExprNode(expr.operands.front());
    if (!expr.operands.empty() && isCompoundExpr(*expr.operands.front()))
      receiver = "(" + receiver + ")";
    std::string out = receiver + "." + expr.text;
    if (expr.operands.size() > 1) {
      out += "(";
      for (size_t index = 1; index < expr.operands.size(); ++index) {
        if (index > 1)
          out += ", ";
        out += printExprNode(expr.operands[index]);
      }
      out += ")";
    }
    return out;
  }
  case ExprKind::Unary:
    return "(" + expr.text + printExprNode(expr.operands[0]) + ")";
  case ExprKind::Binary:
    return "(" + printExprNode(expr.operands[0]) + " " + expr.text + " " +
           printExprNode(expr.operands[1]) + ")";
  case ExprKind::Quantifier:
    // operands: [bound-variable Ident, domain, body]. The body is always
    // compound or a single primary, so the greedy re-parse cannot absorb a
    // following operator.
    return "(" + expr.text + " " + printExprNode(expr.operands[0]) + " in " +
           printExprNode(expr.operands[1]) + " : " +
           printExprNode(expr.operands[2]) + ")";
  }
  return {};
}

} // namespace mlir::llk::mapping
