//===- MappingRules.cpp - LLKMap mapping rules (D4) -----------------------===//
//
// Parses `rule` declarations on top of the shared LLKMap expression grammar.
// A rule must name one Micro operation from the workload vocabulary, declare
// exactly one bundle and one emitter, and is otherwise free-form: predicates,
// parameters, capability and layout requirements, and ports.
//
// Everything a rule references is checked here or in verifyMappingTarget():
// the operation name against the workload vocabulary, parameters against their
// declarations, and layout ids, capability kinds, and emitter keys against the
// target's other registries.
//
//===----------------------------------------------------------------------===//

#include "LLK/Mapping/MappingRules.h"

#include "LLK/Mapping/WorkloadGraph.h"

#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cctype>
#include <string>
#include <utility>

namespace mlir::llk::mapping {

namespace {

/// `v12` -> 12. Version suffixes are `v` followed by digits.
bool parseVersionToken(llvm::StringRef text, uint64_t &out) {
  if (text.size() < 2 || text[0] != 'v')
    return false;
  for (char c : text.drop_front())
    if (!std::isdigit(static_cast<unsigned char>(c)))
      return false;
  return !text.drop_front().getAsInteger(10, out);
}

class RuleParser : public LlkMapParser {
public:
  RuleParser(std::vector<LlkMapToken> tokens, llvm::StringRef source)
      : LlkMapParser(std::move(tokens), source) {}

  llvm::Expected<RuleRegistry> parseFile();

private:
  bool parseRule(RuleDef &out);
  bool parseMatch(RuleDef &out);
  bool parsePredicate(RulePredicate &out);
  bool parseDomain(RuleDef &out);
  bool parseRequire(RuleDef &out);
  bool parsePort(RuleDef &out);
  bool parseMetadata(RuleDef &out);
};

llvm::Expected<RuleRegistry> RuleParser::parseFile() {
  RuleRegistry registry;
  while (!atEnd()) {
    if (current().kind != LlkMapToken::Kind::Identifier ||
        current().text != "rule") {
      failAt(current(), "expected 'rule'");
      return takeError();
    }
    RuleDef def;
    if (!parseRule(def))
      return takeError();
    std::string addError;
    if (!registry.add(std::move(def), addError)) {
      fail(addError);
      return takeError();
    }
  }
  return registry;
}

bool RuleParser::parseRule(RuleDef &out) {
  advance(); // 'rule'
  if (!expectIdentifier("a rule id", out.id))
    return false;
  if (current().kind == LlkMapToken::Kind::Identifier &&
      current().text.size() > 1 && current().text[0] == 'v') {
    uint64_t version = 0;
    if (parseVersionToken(current().text, version)) {
      out.version = version;
      advance();
    }
  }
  if (!expectPunct("{"))
    return false;

  while (!isPunct("}")) {
    if (atEnd())
      return failAt(current(), "expected '}' to close rule '" + out.id + "'");
    if (current().kind != LlkMapToken::Kind::Identifier)
      return failAt(current(), "expected a statement");
    std::string keyword = current().text;
    bool ok = false;
    if (keyword == "match")
      ok = parseMatch(out);
    else if (keyword == "param")
      ok = parseDomain(out);
    else if (keyword == "require")
      ok = parseRequire(out);
    else if (keyword == "input" || keyword == "output")
      ok = parsePort(out);
    else if (keyword == "bundle" || keyword == "emit" || keyword == "cost")
      ok = parseMetadata(out);
    else
      return failAt(current(), "unknown statement '" + keyword + "'");
    if (!ok)
      return false;
  }
  advance(); // '}'

  if (out.matchOp.empty())
    return failAt(current(), "rule '" + out.id + "' is missing a match clause");
  if (out.bundle.empty())
    return failAt(current(), "rule '" + out.id + "' is missing a bundle");
  if (out.emitter.empty())
    return failAt(current(), "rule '" + out.id + "' is missing an emit");
  return true;
}

bool RuleParser::parseMatch(RuleDef &out) {
  if (!out.matchOp.empty())
    return failAt(current(), "duplicate match clause");
  advance(); // 'match'
  std::string operation;
  if (!expectIdentifier("a Micro operation", operation))
    return false;
  if (!isWorkloadNodeOp(operation))
    return failAt(current(), "unknown Micro operation '" + operation + "'");
  out.matchOp = std::move(operation);

  if (!expectPunct("("))
    return false;
  if (!isPunct(")")) {
    while (true) {
      RulePredicate predicate;
      if (!parsePredicate(predicate))
        return false;
      out.predicates.push_back(std::move(predicate));
      if (isPunct(",")) {
        advance();
        continue;
      }
      break;
    }
  }
  if (!expectPunct(")"))
    return false;
  return expectPunct(";");
}

bool RuleParser::parsePredicate(RulePredicate &out) {
  if (!expectIdentifier("an attribute name", out.attribute))
    return false;
  if (!expectPunct("="))
    return false;
  if (current().kind == LlkMapToken::Kind::Int) {
    out.value = current().intValue;
    advance();
    return true;
  }
  if (current().kind == LlkMapToken::Kind::String) {
    out.value = current().text;
    advance();
    return true;
  }
  // A bare name such as `bf16` is a symbolic value.
  if (current().kind == LlkMapToken::Kind::Identifier) {
    out.value = current().text;
    advance();
    return true;
  }
  return failAt(current(), "expected a predicate value");
}

bool RuleParser::parseDomain(RuleDef &out) {
  advance(); // 'param'
  std::string name;
  if (!expectIdentifier("a parameter name", name))
    return false;
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

  // In a rule a `param` statement both declares and bounds the parameter.
  bool symbolic = false;
  for (const LayoutValue &value : values)
    symbolic |= std::holds_alternative<std::string>(value);
  if (!out.findParam(name)) {
    LayoutParam param;
    param.name = name;
    param.symbolic = symbolic;
    out.params.push_back(std::move(param));
  }
  ParamDomain domain;
  domain.values = std::move(values);
  out.domains[name] = std::move(domain);
  return true;
}

bool RuleParser::parseRequire(RuleDef &out) {
  advance(); // 'require'

  bool isKindRole = current().kind == LlkMapToken::Kind::Identifier &&
                    (current().text == "executor" ||
                     current().text == "compute" || current().text == "memory");
  if (isKindRole && peek().kind == LlkMapToken::Kind::Identifier &&
      peek().text == "kind") {
    KindRequirement requirement;
    requirement.role = current().text;
    advance(); // role
    advance(); // 'kind'
    if (!expectIdentifier("a capability kind", requirement.kind))
      return false;
    if (!expectPunct(";"))
      return false;
    out.kindRequirements.push_back(std::move(requirement));
    return true;
  }

  bool isLayoutRequirement = current().kind == LlkMapToken::Kind::Identifier &&
                             current().text == "layout" &&
                             peek().kind == LlkMapToken::Kind::Identifier &&
                             peek(2).kind == LlkMapToken::Kind::Identifier &&
                             peek(2).text == "satisfies";
  if (isLayoutRequirement) {
    advance(); // 'layout'
    LayoutRequirement requirement;
    if (!expectIdentifier("a port name", requirement.port))
      return false;
    advance(); // 'satisfies'
    if (!expectIdentifier("a layout id", requirement.layoutId))
      return false;
    if (!expectPunct(";"))
      return false;
    out.layoutRequirements.push_back(std::move(requirement));
    return true;
  }

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

bool RuleParser::parsePort(RuleDef &out) {
  RulePort port;
  port.isInput = current().text == "input";
  advance();
  if (current().kind != LlkMapToken::Kind::String)
    return failAt(current(), "expected a quoted port name");
  port.name = current().text;
  advance();
  if (!expectPunct(";"))
    return false;
  for (const RulePort &existing : out.ports)
    if (existing.name == port.name)
      return failAt(current(), "duplicate port '" + port.name + "'");
  out.ports.push_back(std::move(port));
  return true;
}

bool RuleParser::parseMetadata(RuleDef &out) {
  std::string keyword = current().text;
  advance();
  if (keyword == "bundle" || keyword == "emit") {
    std::string &slot = keyword == "bundle" ? out.bundle : out.emitter;
    if (!slot.empty())
      return failAt(current(), "duplicate '" + keyword + "'");
    if (current().kind != LlkMapToken::Kind::String)
      return failAt(current(),
                    "expected a quoted name after '" + keyword + "'");
    slot = current().text;
    advance();
    return expectPunct(";");
  }

  if (out.costLowerBound.has_value())
    return failAt(current(), "duplicate 'cost'");
  if (current().kind != LlkMapToken::Kind::Int)
    return failAt(current(), "expected an integer cost");
  out.costLowerBound = static_cast<uint64_t>(current().intValue);
  advance();
  return expectPunct(";");
}

} // namespace

//===----------------------------------------------------------------------===//
// RuleDef / RuleRegistry
//===----------------------------------------------------------------------===//

const LayoutParam *RuleDef::findParam(llvm::StringRef name) const {
  for (const LayoutParam &param : params)
    if (param.name == name)
      return &param;
  return nullptr;
}

bool RuleRegistry::add(RuleDef def, std::string &error) {
  if (find(def.id)) {
    error = "duplicate rule id '" + def.id + "'";
    return false;
  }
  defs_.push_back(std::move(def));
  return true;
}

const RuleDef *RuleRegistry::find(llvm::StringRef id) const {
  for (const RuleDef &def : defs_)
    if (def.id == id)
      return &def;
  return nullptr;
}

llvm::Expected<RuleRegistry> parseRuleText(llvm::StringRef text,
                                           llvm::StringRef sourceName) {
  llvm::Expected<std::vector<LlkMapToken>> tokens = lexLlkMap(text, sourceName);
  if (!tokens)
    return tokens.takeError();
  RuleParser parser(std::move(*tokens), sourceName);
  return parser.parseFile();
}

llvm::Expected<RuleRegistry> loadRuleFile(llvm::StringRef path) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return llvm::createStringError(buffer.getError(),
                                   "cannot read LLKMap file '" + path + "'");
  return parseRuleText(buffer.get()->getBuffer(), path);
}

} // namespace mlir::llk::mapping
