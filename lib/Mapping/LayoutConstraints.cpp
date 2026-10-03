//===- LayoutConstraints.cpp - LLKMap layout declarations (D3) ------------===//
//
// Hand-written lexer and recursive-descent parser for the LLKMap layout
// subset. Diagnostics carry file, line, and column. Identifier validation
// happens here rather than at solve time, so an unknown parameter, call, or
// machine query is a load-time failure (design §14.4 applied to layouts).
//
//===----------------------------------------------------------------------===//

#include "LLK/Mapping/LayoutConstraints.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cctype>
#include <string>
#include <utility>

namespace mlir::llk::mapping {

namespace {

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

struct Token {
  enum class Kind { Identifier, Int, String, Punct, End };
  Kind kind = Kind::End;
  std::string text;
  int64_t intValue = 0;
  unsigned line = 1;
  unsigned column = 1;
};

bool isIdentifierStart(char c) {
  return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
}

bool isIdentifierChar(char c) {
  return std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.';
}

/// Tokenizes `text`. Returns false with a message in `error` on an unterminated
/// string or block comment.
bool lex(llvm::StringRef text, std::vector<Token> &out, std::string &error) {
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

    Token token;
    token.line = line;
    token.column = column;

    if (isIdentifierStart(c)) {
      size_t end = index + 1;
      while (end < text.size() && isIdentifierChar(text[end]))
        ++end;
      token.kind = Token::Kind::Identifier;
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
      token.kind = Token::Kind::Int;
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
      token.kind = Token::Kind::String;
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
        token.kind = Token::Kind::Punct;
        token.text = punct;
        advance(2);
        out.push_back(std::move(token));
        matched = true;
        break;
      }
    }
    if (matched)
      continue;

    static const std::string kOneChar = "(){}[],;.<>!+-*/%";
    if (kOneChar.find(c) == std::string::npos) {
      error = (llvm::Twine("unexpected character '") +
               llvm::Twine(std::string(1, c)) + "'")
                  .str();
      return false;
    }
    token.kind = Token::Kind::Punct;
    token.text = std::string(1, c);
    advance(1);
    out.push_back(std::move(token));
  }

  Token end;
  end.kind = Token::Kind::End;
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

//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

class Parser {
public:
  Parser(const std::vector<Token> &tokens, llvm::StringRef source)
      : tokens_(tokens), source_(source) {}

  llvm::Expected<LayoutRegistry> parseFile();

private:
  const Token &current() const { return tokens_[pos_]; }
  const Token &peek(size_t ahead = 1) const {
    size_t index = pos_ + ahead;
    return index < tokens_.size() ? tokens_[index] : tokens_.back();
  }
  bool atEnd() const { return current().kind == Token::Kind::End; }
  void advance() {
    if (pos_ + 1 < tokens_.size())
      ++pos_;
  }

  bool fail(const llvm::Twine &message);
  bool failAt(const Token &token, const llvm::Twine &message);
  bool expectPunct(llvm::StringRef punct);
  bool expectIdentifier(llvm::StringRef what, std::string &out);
  llvm::Error takeError();

  bool isPunct(llvm::StringRef punct) const {
    return current().kind == Token::Kind::Punct && current().text == punct;
  }

  bool parseLayout(LayoutDef &out);
  bool parseParams(LayoutDef &out);
  bool parseStatement(LayoutDef &out);
  bool parseDomain(LayoutDef &out);
  bool parseRequire(LayoutDef &out);
  bool parseMapClause(LayoutDef &out);
  ExprPtr parseExpression();
  ExprPtr parseOr();
  ExprPtr parseAnd();
  ExprPtr parseEquality();
  ExprPtr parseRelational();
  ExprPtr parseAdditive();
  ExprPtr parseMultiplicative();
  ExprPtr parseUnary();
  ExprPtr parsePostfix();
  ExprPtr parsePrimary();
  bool parseCallArgs(std::vector<ExprPtr> &out);

  /// Validates every identifier, call, and member against the vocabulary the
  /// current declaration allows.
  bool validateExpr(const ExprPtr &expr, const llvm::StringSet<> &allowed);

  const std::vector<Token> &tokens_;
  llvm::StringRef source_;
  size_t pos_ = 0;
  std::string error_;
};

bool Parser::fail(const llvm::Twine &message) {
  if (error_.empty())
    error_ = (source_ + ": error: " + message).str();
  return false;
}

bool Parser::failAt(const Token &token, const llvm::Twine &message) {
  if (error_.empty())
    error_ = (source_ + ":" + llvm::Twine(token.line) + ":" +
              llvm::Twine(token.column) + ": error: " + message)
                 .str();
  return false;
}

llvm::Error Parser::takeError() {
  return llvm::make_error<llvm::StringError>(std::move(error_),
                                             llvm::inconvertibleErrorCode());
}

bool Parser::expectPunct(llvm::StringRef punct) {
  if (!isPunct(punct))
    return failAt(current(), "expected '" + punct + "'");
  advance();
  return true;
}

bool Parser::expectIdentifier(llvm::StringRef what, std::string &out) {
  if (current().kind != Token::Kind::Identifier)
    return failAt(current(), "expected " + what);
  out = current().text;
  advance();
  return true;
}

llvm::Expected<LayoutRegistry> Parser::parseFile() {
  LayoutRegistry registry;
  while (!atEnd()) {
    if (current().kind != Token::Kind::Identifier ||
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
    if (current().kind == Token::Kind::Identifier &&
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
  if (current().kind != Token::Kind::Identifier)
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
  const LayoutParam *param = out.findParam(name);
  if (!param)
    return failAt(current(), "domain for undeclared parameter '" + name + "'");
  if (out.domains.count(name))
    return failAt(current(), "duplicate domain for parameter '" + name + "'");
  if (current().kind != Token::Kind::Identifier || current().text != "in")
    return failAt(current(), "expected 'in'");
  advance();

  ParamDomain domain;
  if (isPunct("[")) {
    advance();
    if (current().kind != Token::Kind::Int)
      return failAt(current(), "expected an integer lower bound");
    int64_t lower = current().intValue;
    advance();
    if (!expectPunct(".."))
      return false;
    if (current().kind != Token::Kind::Int)
      return failAt(current(), "expected an integer upper bound");
    int64_t upper = current().intValue;
    advance();
    if (!expectPunct("]"))
      return false;
    if (upper < lower)
      return failAt(current(), "empty integer range");
    for (int64_t value = lower; value <= upper; ++value)
      domain.values.push_back(value);
  } else if (isPunct("{")) {
    advance();
    while (true) {
      if (current().kind == Token::Kind::Int) {
        domain.values.push_back(current().intValue);
        advance();
      } else if (current().kind == Token::Kind::String) {
        domain.values.push_back(current().text);
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
    if (!expectPunct("}"))
      return false;
  } else {
    return failAt(current(), "expected '[' or '{'");
  }
  if (!expectPunct(";"))
    return false;
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

bool Parser::validateExpr(const ExprPtr &expr,
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
  }
  for (const ExprPtr &operand : expr->operands)
    if (!validateExpr(operand, allowed))
      return false;
  return true;
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

ExprPtr Parser::parseExpression() { return parseOr(); }

ExprPtr Parser::parseOr() {
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

ExprPtr Parser::parseAnd() {
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

ExprPtr Parser::parseEquality() {
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

ExprPtr Parser::parseRelational() {
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

ExprPtr Parser::parseAdditive() {
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

ExprPtr Parser::parseMultiplicative() {
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

ExprPtr Parser::parseUnary() {
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

bool Parser::parseCallArgs(std::vector<ExprPtr> &out) {
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

ExprPtr Parser::parsePostfix() {
  ExprPtr expr = parsePrimary();
  if (!expr)
    return nullptr;
  while (isPunct(".")) {
    advance();
    std::string member;
    if (!expectIdentifier("a member name", member))
      return nullptr;
    std::vector<ExprPtr> args;
    if (!parseCallArgs(args))
      return nullptr;
    args.insert(args.begin(), expr);
    expr = makeExpr(ExprKind::MemberCall, std::move(member), std::move(args));
  }
  return expr;
}

ExprPtr Parser::parsePrimary() {
  if (current().kind == Token::Kind::Int) {
    auto expr = std::make_shared<Expr>();
    expr->kind = ExprKind::IntLit;
    expr->intValue = current().intValue;
    advance();
    return expr;
  }
  if (current().kind == Token::Kind::String) {
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
  if (current().kind == Token::Kind::Identifier) {
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

bool LayoutRegistry::add(LayoutDef def, std::string &error) {
  if (find(def.id)) {
    error = "duplicate layout id '" + def.id + "'";
    return false;
  }
  defs_.push_back(std::move(def));
  return true;
}

const LayoutDef *LayoutRegistry::find(llvm::StringRef id) const {
  for (const LayoutDef &def : defs_)
    if (def.id == id)
      return &def;
  return nullptr;
}

llvm::Expected<LayoutRegistry> parseLayoutText(llvm::StringRef text,
                                               llvm::StringRef sourceName) {
  std::vector<Token> tokens;
  std::string lexError;
  if (!lex(text, tokens, lexError))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   sourceName + ": error: " + lexError);
  Parser parser(tokens, sourceName);
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

} // namespace mlir::llk::mapping
