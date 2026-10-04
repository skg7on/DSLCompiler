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

#include "LLK/Mapping/StableHash.h"
#include "LLK/Mapping/TileFacts.h"
#include "LLK/Mapping/WorkloadGraph.h"

#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <cctype>
#include <optional>
#include <string>
#include <utility>
#include <variant>

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

/// True when `expr` is affine in the map sense: constants, dimensions, `+`,
/// `-`, multiplication by a constant, `floordiv`/`ceildiv`/`mod`, and unary
/// minus. Mirrors the builder in LayoutConstraints.cpp, so an access-map
/// predicate that is not affine is rejected at load time instead of silently
/// never matching.
bool isAffineSpecExpr(const Expr &expr) {
  switch (expr.kind) {
  case ExprKind::IntLit:
  case ExprKind::Ident:
    return true;
  case ExprKind::Unary:
    return expr.text == "-" && isAffineSpecExpr(*expr.operands[0]);
  case ExprKind::Binary: {
    const Expr &lhs = *expr.operands[0];
    const Expr &rhs = *expr.operands[1];
    if (expr.text == "*")
      return (lhs.kind == ExprKind::IntLit || rhs.kind == ExprKind::IntLit) &&
             isAffineSpecExpr(lhs) && isAffineSpecExpr(rhs);
    if (expr.text == "+" || expr.text == "-" || expr.text == "/" ||
        expr.text == "%")
      return isAffineSpecExpr(lhs) && isAffineSpecExpr(rhs);
    return false;
  }
  case ExprKind::Call:
    return (expr.text == "floordiv" || expr.text == "ceildiv" ||
            expr.text == "mod") &&
           expr.operands.size() == 2 && isAffineSpecExpr(*expr.operands[0]) &&
           isAffineSpecExpr(*expr.operands[1]);
  case ExprKind::StringLit:
  case ExprKind::MemberCall:
  case ExprKind::Quantifier:
    return false;
  }
  return false;
}

/// The literal a predicate is allowed to take. Typing the value at parse time
/// keeps a malformed rule (`element_type = 4`) a load-time diagnostic rather
/// than a `std::bad_variant_access` on the first match.
enum class PredicateValueKind { Any, Int, Symbol };

class RuleParser : public LlkMapParser {
public:
  RuleParser(std::vector<LlkMapToken> tokens, llvm::StringRef source)
      : LlkMapParser(std::move(tokens), source) {}

  llvm::Expected<RuleRegistry> parseFile();

private:
  bool parseRule(RuleDef &out);
  bool parseMatch(RuleDef &out);
  bool parsePredicate(RulePredicate &out);
  bool parsePredicateValue(LayoutValue &out, PredicateValueKind expected);
  bool parseAffineMapSpec(AffineMapSpec &out);
  bool parseDomain(RuleDef &out);
  bool parseRequire(RuleDef &out);
  bool parsePort(RuleDef &out);
  bool parseMetadata(RuleDef &out);
  bool parseBundleParameters(RuleDef &out);
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

bool RuleParser::parsePredicateValue(LayoutValue &out,
                                     PredicateValueKind expected) {
  if (!expectPunct("="))
    return false;
  if (expected != PredicateValueKind::Symbol &&
      current().kind == LlkMapToken::Kind::Int) {
    out = current().intValue;
    advance();
    return true;
  }
  // A bare name such as `bf16`, or a quoted name, is a symbolic value.
  if (expected != PredicateValueKind::Int &&
      (current().kind == LlkMapToken::Kind::String ||
       current().kind == LlkMapToken::Kind::Identifier)) {
    out = current().text;
    advance();
    return true;
  }
  if (expected == PredicateValueKind::Int)
    return failAt(current(), "expected an integer predicate value");
  if (expected == PredicateValueKind::Symbol)
    return failAt(current(), "expected a symbolic predicate value");
  return failAt(current(), "expected a predicate value");
}

bool RuleParser::parseAffineMapSpec(AffineMapSpec &out) {
  if (!expectPunct("("))
    return false;
  if (!isPunct(")")) {
    while (true) {
      std::string dim;
      if (!expectIdentifier("a map dimension", dim))
        return false;
      out.dims.push_back(std::move(dim));
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
    out.results.push_back(std::move(result));
    if (isPunct(",")) {
      advance();
      continue;
    }
    break;
  }
  if (!expectPunct(")"))
    return false;

  // A map predicate may reference only its own dimensions, and only affinely.
  llvm::StringSet<> allowed;
  for (const std::string &dim : out.dims)
    allowed.insert(dim);
  for (const ExprPtr &result : out.results) {
    if (!validateExpr(result, allowed))
      return false;
    if (!isAffineSpecExpr(*result))
      return failAt(current(), "access map is not affine");
  }
  return true;
}

bool RuleParser::parsePredicate(RulePredicate &out) {
  // A port subject is `input[i]` or `output[i]`. A bare `input`/`output` stays
  // an attribute name, so `micro.mma(input = bf16)` keeps matching the
  // operation's `input` attribute.
  if ((current().text == "input" || current().text == "output") &&
      peek().kind == LlkMapToken::Kind::Punct && peek().text == "[") {
    out.directionSet = true;
    out.isInput = current().text == "input";
    advance(); // direction
    advance(); // '['
    if (current().kind != LlkMapToken::Kind::Int)
      return failAt(current(), "expected a port index");
    out.portIndex = current().intValue;
    if (out.portIndex < 0)
      return failAt(current(), "expected a non-negative port index");
    advance();
    if (!expectPunct("]"))
      return false;
    if (!expectPunct("."))
      return false;
  }

  std::string name;
  if (!expectIdentifier("a predicate name", name))
    return false;

  if (name == "element_type") {
    out.kind = RulePredicateKind::ElementType;
    out.attribute = std::move(name);
    return parsePredicateValue(out.value, PredicateValueKind::Symbol);
  }
  if (name == "shape") {
    if (!expectPunct("["))
      return false;
    if (current().kind != LlkMapToken::Kind::Int)
      return failAt(current(), "expected a shape dimension");
    out.dimension = current().intValue;
    if (out.dimension < 0)
      return failAt(current(), "expected a non-negative shape dimension");
    advance();
    if (!expectPunct("]"))
      return false;
    out.kind = RulePredicateKind::Shape;
    out.attribute = std::move(name);
    return parsePredicateValue(out.value, PredicateValueKind::Int);
  }
  if (name == "access_map") {
    if (!expectPunct("="))
      return false;
    AffineMapSpec spec;
    if (!parseAffineMapSpec(spec))
      return false;
    out.kind = RulePredicateKind::AccessMap;
    out.attribute = std::move(name);
    out.accessMap = std::move(spec);
    return true;
  }

  if (out.directionSet)
    return failAt(
        current(),
        "port predicates support element_type, shape, and access_map");

  out.kind = RulePredicateKind::Attribute;
  out.attribute = std::move(name);
  return parsePredicateValue(out.value, PredicateValueKind::Any);
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
    const LlkMapToken roleToken = current();
    advance(); // role
    advance(); // 'kind'
    if (!expectIdentifier("a capability kind", requirement.kind))
      return false;
    if (!expectPunct(";"))
      return false;
    // A rule may not require the same (role, kind) twice: placement binds
    // attachments by kind, so a duplicate would collide in the binding map and
    // emit indistinguishable instances. Rejected here, in source order, before
    // any candidate is built.
    for (const KindRequirement &existing : out.kindRequirements)
      if (existing.role == requirement.role &&
          existing.kind == requirement.kind)
        return failAt(roleToken, "duplicate " + requirement.role +
                                     " kind requirement '" + requirement.kind +
                                     "'");
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
    RuleLayoutRequirement requirement;
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
    // Only a bundle may carry typed parameters; `emit` stays a bare key.
    if (keyword == "bundle" && isPunct("{"))
      return parseBundleParameters(out);
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

/// Parses the optional `{ key = value, ... }` block after a bundle name. A
/// value is an integer or a symbolic name (bare or quoted), so the parameter is
/// typed from its spelling alone; the block is small and flat on purpose
/// (design §14.3). The result is sorted by key so declaration order never
/// reaches an id.
bool RuleParser::parseBundleParameters(RuleDef &out) {
  advance(); // '{'
  if (isPunct("}"))
    return failAt(current(), "expected a bundle parameter name");
  while (true) {
    std::string name;
    if (!expectIdentifier("a bundle parameter name", name))
      return false;
    for (const auto &entry : out.bundleParameters)
      if (entry.first == name)
        return failAt(current(), "duplicate bundle parameter '" + name + "'");
    if (!expectPunct("="))
      return false;
    LayoutValue value;
    if (current().kind == LlkMapToken::Kind::Int) {
      value = current().intValue;
      advance();
    } else if (current().kind == LlkMapToken::Kind::String ||
               current().kind == LlkMapToken::Kind::Identifier) {
      value = current().text;
      advance();
    } else {
      return failAt(current(),
                    "expected an integer or symbolic value for '" + name + "'");
    }
    out.bundleParameters.emplace_back(std::move(name), std::move(value));
    if (isPunct(",")) {
      advance();
      continue;
    }
    break;
  }
  if (!expectPunct("}"))
    return false;
  llvm::sort(out.bundleParameters);
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

namespace {

llvm::StringRef predicateKindName(RulePredicateKind kind) {
  switch (kind) {
  case RulePredicateKind::Attribute:
    return "attribute";
  case RulePredicateKind::ElementType:
    return "element_type";
  case RulePredicateKind::Shape:
    return "shape";
  case RulePredicateKind::AccessMap:
    return "access_map";
  }
  return "unknown";
}

/// Canonical rendering of one rule declaration. Declaration-order sequences
/// (predicates, ports, requirements) are rendered in that order -- which is
/// deterministic -- and each scalar is length-prefixed so no field's bytes can
/// be read as another's.
std::string canonicalRuleDefString(const RuleDef &def) {
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
  field("version", std::to_string(def.version));
  field("match", def.matchOp);

  for (const RulePredicate &predicate : def.predicates) {
    std::string text = predicateKindName(predicate.kind).str();
    text += ' ';
    text += predicate.attribute;
    text += predicate.directionSet ? " port=" : " unqualified=";
    if (predicate.directionSet) {
      text += predicate.isInput ? "input[" : "output[";
      text += std::to_string(predicate.portIndex);
      text += ']';
    }
    text += " dim=" + std::to_string(predicate.dimension);
    text += " value=" + canonicalValueString(predicate.value);
    if (predicate.accessMap)
      text += " map=" + canonicalAffineMapSpecString(*predicate.accessMap);
    field("predicate", text);
  }
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
  for (const KindRequirement &requirement : def.kindRequirements)
    field("require", requirement.role + ":" + requirement.kind);
  for (const RuleLayoutRequirement &requirement : def.layoutRequirements)
    field("layout", requirement.port + ":" + requirement.layoutId);
  for (const RulePort &port : def.ports)
    field("port", port.name + (port.isInput ? ":input" : ":output"));
  field("bundle", def.bundle);
  // `bundleParameters` is kept sorted by name, so the order is canonical.
  for (const auto &parameter : def.bundleParameters)
    field("bundle_parameter",
          parameter.first + "=" + canonicalValueString(parameter.second));
  field("emitter", def.emitter);
  if (def.costLowerBound)
    field("cost", std::to_string(*def.costLowerBound));
  return out;
}

} // namespace

//===----------------------------------------------------------------------===//
// Source-faithful printing (design §25.3)
//===----------------------------------------------------------------------===//

namespace {

/// The `input[i].` / `output[i].` subject a port predicate was written with, or
/// empty for an unqualified predicate.
std::string printPredicateSubject(const RulePredicate &predicate) {
  if (!predicate.directionSet)
    return {};
  return (predicate.isInput ? "input[" : "output[") +
         std::to_string(predicate.portIndex) + "].";
}

std::string printPredicate(const RulePredicate &predicate) {
  std::string subject = printPredicateSubject(predicate);
  switch (predicate.kind) {
  case RulePredicateKind::Attribute:
    return predicate.attribute + " = " + printValue(predicate.value);
  case RulePredicateKind::ElementType:
    return subject + "element_type = " + printValue(predicate.value);
  case RulePredicateKind::Shape:
    return subject + "shape[" + std::to_string(predicate.dimension) +
           "] = " + printValue(predicate.value);
  case RulePredicateKind::AccessMap:
    return subject + "access_map = " +
           (predicate.accessMap ? printAffineMapSpec(*predicate.accessMap)
                                : std::string("<null>"));
  }
  return {};
}

} // namespace

std::string printRule(const RuleDef &def) {
  std::string out =
      "rule " + def.id + " v" + std::to_string(def.version) + " {\n";

  out += "  match " + def.matchOp + "(";
  for (size_t index = 0; index < def.predicates.size(); ++index) {
    if (index)
      out += ", ";
    out += printPredicate(def.predicates[index]);
  }
  out += ");\n";

  // A rule declares a parameter through its `param ... in ...` statement, so
  // emit them in `params` order to keep that order on re-parse.
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
  for (const KindRequirement &requirement : def.kindRequirements)
    out +=
        "  require " + requirement.role + " kind " + requirement.kind + ";\n";
  for (const RuleLayoutRequirement &requirement : def.layoutRequirements)
    out += "  require layout " + requirement.port + " satisfies " +
           requirement.layoutId + ";\n";
  for (const RulePort &port : def.ports)
    out += "  " + std::string(port.isInput ? "input" : "output") + " \"" +
           port.name + "\";\n";

  out += "  bundle \"" + def.bundle + "\"";
  if (!def.bundleParameters.empty()) {
    out += " {";
    for (size_t index = 0; index < def.bundleParameters.size(); ++index) {
      if (index)
        out += ",";
      out += " " + def.bundleParameters[index].first + " = " +
             printValue(def.bundleParameters[index].second);
    }
    out += " }";
  }
  out += ";\n";

  out += "  emit \"" + def.emitter + "\";\n";
  if (def.costLowerBound)
    out += "  cost " + std::to_string(*def.costLowerBound) + ";\n";
  out += "}\n";
  return out;
}

bool RuleRegistry::add(RuleDef def, std::string &error) {
  if (find(def.id)) {
    error = "duplicate rule id '" + def.id + "'";
    return false;
  }
  // Insert in id order: `all()` is an output order, so file declaration order
  // must never reach it (design §22.1). `find`'s lower_bound search stays
  // valid.
  std::string id = def.id;
  auto position = llvm::lower_bound(
      defs_, id,
      [](const RuleDef &rule, llvm::StringRef key) { return rule.id < key; });
  defs_.insert(position, std::move(def));
  return true;
}

uint64_t RuleRegistry::computeContentHash() const {
  // `defs_` is stored in id order (see `add`), but sort a view defensively so
  // the hash stays content-derived even if the storage invariant ever changes.
  std::vector<const RuleDef *> sorted;
  sorted.reserve(defs_.size());
  for (const RuleDef &def : defs_)
    sorted.push_back(&def);
  llvm::sort(sorted, [](const RuleDef *lhs, const RuleDef *rhs) {
    return lhs->id < rhs->id;
  });

  std::string canonical;
  for (const RuleDef *def : sorted)
    canonical += canonicalRuleDefString(*def);
  return stableHash(canonical);
}

const RuleDef *RuleRegistry::find(llvm::StringRef id) const {
  auto position = llvm::lower_bound(
      defs_, id,
      [](const RuleDef &rule, llvm::StringRef key) { return rule.id < key; });
  if (position == defs_.end() || position->id != id)
    return nullptr;
  return &*position;
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

//===----------------------------------------------------------------------===//
// One-operation matching
//===----------------------------------------------------------------------===//

namespace {

std::string printedType(mlir::Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return stream.str();
}

// The `!micro.tile` printed-form unwrapping, and the element-type and static
// shape it exposes, live in TileFacts: the same read is what sizes a moving
// value, so it has one implementation rather than one per caller.

/// The ports a predicate reads: exactly the one it names, or -- for an
/// unqualified predicate -- every port in the property's default direction.
/// Element type and access map read inputs; shape reads outputs.
llvm::SmallVector<const WorkloadPort *, 4>
subjectPorts(const RulePredicate &predicate, const WorkloadNode &node) {
  llvm::SmallVector<const WorkloadPort *, 4> selected;
  if (predicate.directionSet) {
    llvm::ArrayRef<WorkloadPort> ports =
        predicate.isInput ? node.inputs : node.outputs;
    if (predicate.portIndex >= 0 &&
        static_cast<size_t>(predicate.portIndex) < ports.size())
      selected.push_back(&ports[predicate.portIndex]);
    return selected;
  }
  bool defaultsToInput = predicate.kind != RulePredicateKind::Shape;
  llvm::ArrayRef<WorkloadPort> ports =
      defaultsToInput ? node.inputs : node.outputs;
  for (const WorkloadPort &port : ports)
    selected.push_back(&port);
  return selected;
}

bool attributeMatches(const RulePredicate &predicate,
                      mlir::DictionaryAttr attributes) {
  if (!attributes)
    return false;
  mlir::Attribute value = attributes.get(predicate.attribute);
  if (!value)
    return false;
  if (const auto *integer = std::get_if<int64_t>(&predicate.value)) {
    if (auto attribute = mlir::dyn_cast<mlir::IntegerAttr>(value))
      return attribute.getInt() == *integer;
    return false;
  }
  const std::string &expected = std::get<std::string>(predicate.value);
  if (auto attribute = mlir::dyn_cast<mlir::StringAttr>(value))
    return attribute.getValue() == expected;
  return false;
}

} // namespace

bool predicateMatches(const RulePredicate &predicate,
                      const WorkloadNode &node) {
  switch (predicate.kind) {
  case RulePredicateKind::Attribute:
    return attributeMatches(predicate, node.attributes);
  case RulePredicateKind::ElementType: {
    const auto *expected = std::get_if<std::string>(&predicate.value);
    if (!expected)
      return false;
    bool exposed = false;
    for (const WorkloadPort *port : subjectPorts(predicate, node)) {
      mlir::Type element = elementTypeOf(port->type);
      if (!element)
        continue;
      if (printedType(element) != *expected)
        return false;
      exposed = true;
    }
    return exposed;
  }
  case RulePredicateKind::Shape: {
    const auto *expected = std::get_if<int64_t>(&predicate.value);
    if (!expected)
      return false;
    bool exposed = false;
    for (const WorkloadPort *port : subjectPorts(predicate, node)) {
      std::optional<llvm::SmallVector<int64_t, 4>> shape =
          staticShapeOf(port->type);
      if (!shape || predicate.dimension < 0 ||
          static_cast<size_t>(predicate.dimension) >= shape->size())
        continue;
      if ((*shape)[predicate.dimension] != *expected)
        return false;
      exposed = true;
    }
    return exposed;
  }
  case RulePredicateKind::AccessMap: {
    if (!predicate.accessMap)
      return false;
    const AffineMapSpec &spec = *predicate.accessMap;
    bool exposed = false;
    for (const WorkloadPort *port : subjectPorts(predicate, node)) {
      if (!port->accessMap)
        continue;
      llvm::Expected<mlir::AffineMap> expected =
          buildAffineMap(spec, {}, *port->accessMap->getContext());
      if (!expected) {
        llvm::consumeError(expected.takeError());
        return false;
      }
      if (mlir::simplifyAffineMap(*expected) !=
          mlir::simplifyAffineMap(*port->accessMap))
        return false;
      exposed = true;
    }
    return exposed;
  }
  }
  return false;
}

std::vector<const RuleDef *> matchRules(const WorkloadNode &node,
                                        const RuleRegistry &rules) {
  std::vector<const RuleDef *> matches;
  for (const RuleDef &rule : rules.all()) {
    if (rule.matchOp != node.opName)
      continue;
    bool matched = true;
    for (const RulePredicate &predicate : rule.predicates) {
      if (!predicateMatches(predicate, node)) {
        matched = false;
        break;
      }
    }
    if (matched)
      matches.push_back(&rule);
  }
  // Canonical candidate order (design §22.1): (covered-node sequence, rule id,
  // candidate id). The covered sequence is this one node for every match, and a
  // rule yields at most one candidate for it, so rule id is the whole key.
  llvm::sort(matches, [](const RuleDef *lhs, const RuleDef *rhs) {
    return lhs->id < rhs->id;
  });
  return matches;
}

namespace {

/// The most assignments a rule's parameter space is enumerated over before the
/// rule is treated as a non-match. A rule's domains are tiny in practice (a
/// vector width's `[4..8]`), so this only stops a pathological rule from
/// stalling a search.
constexpr uint64_t kMaxRuleAssignments = 100000;

/// The rank and element type a rule's `require` constraints see: the facts the
/// node exposes, falling back to `fallback` for anything it does not carry.
/// `element_type` is the first input's element type (else the first output's);
/// `rank` is the first output's static rank (else the first input's). The
/// fallback is the search's `LayoutContext`, which names the value being
/// mapped.
LayoutContext ruleLayoutContext(const WorkloadNode &node,
                                const LayoutContext &fallback) {
  LayoutContext context = fallback;
  bool haveElement = false;
  for (const WorkloadPort &port : node.inputs) {
    if (mlir::Type element = elementTypeOf(port.type)) {
      context.elementType = printedType(element);
      haveElement = true;
      break;
    }
  }
  if (!haveElement)
    for (const WorkloadPort &port : node.outputs) {
      if (mlir::Type element = elementTypeOf(port.type)) {
        context.elementType = printedType(element);
        break;
      }
    }

  bool haveRank = false;
  for (const WorkloadPort &port : node.outputs) {
    if (std::optional<llvm::SmallVector<int64_t, 4>> shape =
            staticShapeOf(port.type)) {
      context.rank = static_cast<int64_t>(shape->size());
      haveRank = true;
      break;
    }
  }
  if (!haveRank)
    for (const WorkloadPort &port : node.inputs) {
      if (std::optional<llvm::SmallVector<int64_t, 4>> shape =
              staticShapeOf(port.type)) {
        context.rank = static_cast<int64_t>(shape->size());
        break;
      }
    }
  return context;
}

/// Adds every identifier an expression references to `out`, so a rule can tell
/// which of its parameters a constraint actually derives. `bound` holds the
/// quantifier variables currently in scope: they shadow a like-named parameter,
/// so a reference to one is not a reference to the parameter.
void collectIdentifiers(const Expr &expr, llvm::StringSet<> &out,
                        const llvm::StringSet<> &bound) {
  if (expr.kind == ExprKind::Ident) {
    if (!bound.contains(expr.text))
      out.insert(expr.text);
    return;
  }
  if (expr.kind == ExprKind::Quantifier) {
    // operands: [bound-variable Ident, domain, body]. `domain(<param>)` names a
    // domain, not a value, so its argument is not a value reference either;
    // `executors(<expr>)` does evaluate its argument, so that one is walked.
    const Expr &domain = *expr.operands[1];
    if (domain.kind == ExprKind::Call && domain.text == "executors")
      for (const ExprPtr &argument : domain.operands)
        collectIdentifiers(*argument, out, bound);
    llvm::StringSet<> scoped = bound;
    scoped.insert(expr.operands[0]->text);
    collectIdentifiers(*expr.operands[2], out, scoped);
    return;
  }
  for (const ExprPtr &operand : expr.operands)
    collectIdentifiers(*operand, out, bound);
}

/// The result of evaluating a rule's `require` constraints against one node:
/// whether some assignment of the rule's declared parameters satisfies every
/// constraint, why not when none does, whether the bounded enumeration was cut
/// off before it could decide, and the satisfying assignment.
struct RuleResolution {
  bool matched = true;
  bool truncated = false;
  std::string reason;
  llvm::StringMap<SearchValue> parameters;
};

/// Enumerates the rule's declared parameter domains in declaration order and
/// returns the first assignment that satisfies every constraint. A rule with
/// no constraints binds nothing. Fails the match -- never the search -- when no
/// assignment satisfies a constraint, or when a constraint cannot be evaluated
/// from the facts the node and context supply.
///
/// A parameter named in `pinned` -- the search-space point the match is
/// evaluated at -- enumerates over the singleton bound value instead of its
/// declared domain, so the binding, not the enumeration order, chooses it. A
/// bound value the declared domain does not contain leaves no admissible
/// assignment: the rule is a non-match, exactly as if a `require` rejected
/// every value. Names not declared by the rule are ignored.
///
/// Only parameters a constraint references are recorded; a declared but
/// unconstrained parameter is not derived, so it never lands in the result.
/// Hitting the assignment cap sets `truncated`: a rule whose space was not
/// exhausted was not proven unsatisfiable.
RuleResolution
resolveRuleConstraints(const RuleDef &rule, const WorkloadNode &node,
                       const machine::MachineModel &machine,
                       const LayoutContext &context,
                       const llvm::StringMap<SearchValue> *pinned) {
  RuleResolution result;
  if (rule.constraints.empty())
    return result;

  // Every declared parameter must have a finite, non-empty domain to enumerate.
  // The domain is restricted to the bound value for a pinned parameter, in the
  // rule's own declaration order, so pinned and unpinned runs enumerate
  // deterministically.
  llvm::SmallVector<const LayoutParam *, 4> enumerable;
  llvm::SmallVector<llvm::SmallVector<LayoutValue, 8>, 4> domains;
  for (const LayoutParam &param : rule.params) {
    auto domain = rule.domains.find(param.name);
    if (domain == rule.domains.end() || domain->second.values.empty()) {
      result.matched = false;
      result.reason = "parameter '" + param.name + "' has no declared domain";
      return result;
    }
    llvm::SmallVector<LayoutValue, 8> values;
    if (pinned) {
      auto bound = pinned->find(param.name);
      if (bound != pinned->end()) {
        if (!llvm::is_contained(domain->second.values, bound->second)) {
          result.matched = false;
          result.reason = "binding pins parameter '" + param.name +
                          "' to a value outside its declared domain";
          return result;
        }
        values.push_back(bound->second);
      }
    }
    if (values.empty())
      values.assign(domain->second.values.begin(), domain->second.values.end());
    enumerable.push_back(&param);
    domains.push_back(std::move(values));
  }

  // The parameters a constraint actually derives: only these are recorded.
  llvm::StringSet<> referenced;
  llvm::StringSet<> bound; // no quantifier is in scope at the top level
  for (const ExprPtr &constraint : rule.constraints)
    collectIdentifiers(*constraint, referenced, bound);

  LayoutContext ruleContext = ruleLayoutContext(node, context);

  // A quantifier in a rule's `require` is bounded like the rule's own
  // enumeration, and `domain(<param>)` reads the rule's declared domains. The
  // resolver is a named lvalue because `function_ref` does not own its
  // callable.
  QuantifierBudget budget;
  budget.limit = kMaxRuleAssignments;
  auto resolveDomain =
      [&rule](
          llvm::StringRef name) -> std::optional<llvm::ArrayRef<LayoutValue>> {
    auto it = rule.domains.find(name.str());
    if (it == rule.domains.end())
      return std::nullopt;
    return llvm::ArrayRef<LayoutValue>(it->second.values);
  };
  EvalOptions evalOptions;
  evalOptions.budget = &budget;
  evalOptions.domainResolver = resolveDomain;

  llvm::SmallVector<size_t, 4> index(enumerable.size(), 0);
  uint64_t assignments = 0;
  std::optional<std::string> failure;
  bool exhausted = false;
  while (!exhausted) {
    if (assignments >= kMaxRuleAssignments) {
      result.matched = false;
      result.truncated = true;
      result.reason = "require constraint search exceeded " +
                      std::to_string(kMaxRuleAssignments) +
                      " assignments without proving the rule unsatisfiable";
      return result;
    }

    llvm::StringMap<LayoutValue> bindings;
    for (size_t i = 0; i < enumerable.size(); ++i)
      bindings[enumerable[i]->name] = domains[i][index[i]];
    ++assignments;

    bool satisfied = true;
    for (const ExprPtr &constraint : rule.constraints) {
      llvm::Expected<EvalValue> value = evaluateExpr(
          *constraint, bindings, machine, ruleContext, evalOptions);
      if (!value) {
        failure = llvm::toString(value.takeError());
        satisfied = false;
        break;
      }
      if (value->kind != EvalValue::Kind::Int || value->intValue == 0) {
        failure = "a require constraint is not satisfied";
        satisfied = false;
        break;
      }
    }

    if (satisfied) {
      // A quantifier that ran out of budget leaves its constraint *undecided*
      // (it yields 0), so under `!`/`==0`/`!=1` the constraint can read as
      // satisfied. Do not accept such an assignment: fail closed and report the
      // search truncated, so the caller never reads it as a rule match.
      if (budget.exhausted) {
        result.matched = false;
        result.truncated = true;
        result.reason =
            "require constraint search exceeded the quantifier bound without "
            "deciding every constraint";
        return result;
      }
      for (const auto &entry : bindings)
        if (referenced.contains(entry.first()))
          result.parameters[entry.first().str()] = entry.second;
      return result;
    }

    // Advance the odometer: the last declared parameter varies fastest.
    exhausted = true;
    for (size_t i = enumerable.size(); i-- > 0;) {
      if (++index[i] < domains[i].size()) {
        exhausted = false;
        break;
      }
      index[i] = 0;
    }
  }

  result.matched = false;
  // A quantifier that ran out of budget leaves the rule undecided, not
  // unsatisfiable; report it so the caller does not read it as a clean miss.
  result.truncated = result.truncated || budget.exhausted;
  result.reason =
      failure.value_or("no assignment satisfies the rule's constraints");
  return result;
}

/// An MLIR context to type a rule's bundle parameters with, taken from the node
/// they matched. A rule that declares parameters only ever matches a node built
/// from real IR, so a context is available; the null return marks a synthetic
/// or context-free node, which the caller turns into a loud failure rather than
/// a silently parameterless bundle.
mlir::MLIRContext *nodeContext(const WorkloadNode &node) {
  if (node.attributes)
    return node.attributes.getContext();
  for (const WorkloadPort &port : node.inputs)
    if (port.type)
      return port.type.getContext();
  for (const WorkloadPort &port : node.outputs)
    if (port.type)
      return port.type.getContext();
  return nullptr;
}

/// Materializes a rule's context-free bundle parameters as a typed
/// `DictionaryAttr`: an integer value becomes an `IntegerAttr` (i64) and a
/// symbolic value a `StringAttr`. Null when the rule declares no parameters.
///
/// A rule that *does* declare parameters must never reach the plan as a
/// parameterless bundle, so a node with no MLIR context (synthetic, or built
/// with neither attributes nor typed ports) is a fatal error naming the rule
/// and node -- not a silent empty result two parameterizations could collide
/// on.
mlir::DictionaryAttr buildBundleParameters(const RuleDef &rule,
                                           const WorkloadNode &node) {
  if (rule.bundleParameters.empty())
    return {};
  mlir::MLIRContext *context = nodeContext(node);
  if (!context) {
    std::string message =
        "rule '" + rule.id + "' declares bundle parameters but workload node " +
        std::to_string(node.id) + " has no MLIR context to type them with";
    llvm::report_fatal_error(llvm::StringRef(message));
  }
  llvm::SmallVector<mlir::NamedAttribute> attributes;
  attributes.reserve(rule.bundleParameters.size());
  for (const auto &entry : rule.bundleParameters) {
    mlir::Attribute value;
    if (const auto *integer = std::get_if<int64_t>(&entry.second))
      value =
          mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), *integer);
    else
      value =
          mlir::StringAttr::get(context, std::get<std::string>(entry.second));
    attributes.emplace_back(mlir::StringAttr::get(context, entry.first), value);
  }
  return mlir::DictionaryAttr::get(context, attributes);
}

} // namespace

std::optional<MappingCandidate>
toMappingCandidate(const RuleDef &rule, const WorkloadNode &node,
                   const machine::MachineModel &machine,
                   const LayoutContext &context, std::string *reason,
                   bool *truncated, const llvm::StringMap<SearchValue> *pinned,
                   const std::string *boundLayout) {
  RuleResolution resolution =
      resolveRuleConstraints(rule, node, machine, context, pinned);
  if (!resolution.matched) {
    if (reason)
      *reason = resolution.reason.empty()
                    ? std::string("require constraints not satisfied")
                    : resolution.reason;
    if (truncated)
      *truncated = resolution.truncated;
    return std::nullopt;
  }

  // A bound layout constrains the rule only where the rule takes on a layout
  // obligation. The value was resolved by `kind == "layout"` at the pass layer
  // (lib/Mapping does not see the search space, and must not depend on
  // LLKPerf); here it is only string-compared against the rule's declared
  // layout ids, which stay target-owned. A rule that declares layout
  // requirements but offers none of them contradicts the binding, so it is a
  // non-match, never an error: a sibling rule that does offer the layout may
  // still match. A rule that declares *no* layout requirement takes on no
  // layout obligation, so it neither offers nor contradicts the bound value and
  // matches unchanged. Both halves are the same ruling (S7): the layout axis
  // vetoes only a rule that takes on an obligation it cannot meet, which is the
  // layout-axis analogue of T2's "a binding name the rule does not declare is
  // ignored". Vetoing a rule with no obligation would make every
  // movement/reduce rule (which the shipped rule files leave layout-agnostic)
  // unmappable under any bound layout.
  if (boundLayout && !rule.layoutRequirements.empty()) {
    bool offered = false;
    for (const RuleLayoutRequirement &requirement : rule.layoutRequirements)
      offered |= requirement.layoutId == *boundLayout;
    if (!offered) {
      if (reason)
        *reason = "binding selects layout '" + *boundLayout +
                  "', which the rule does not offer";
      return std::nullopt;
    }
  }

  MappingCandidate candidate;
  candidate.rule = rule.id;
  candidate.coveredNodes.push_back(node.id);
  candidate.bundle.name = rule.bundle;
  candidate.bundle.emitterKey = rule.emitter;
  candidate.bundle.parameters = buildBundleParameters(rule, node);
  candidate.resolvedParameters = std::move(resolution.parameters);
  if (rule.costLowerBound)
    candidate.lowerBound.latencyCycles =
        static_cast<double>(*rule.costLowerBound);

  size_t inputIndex = 0;
  size_t outputIndex = 0;
  for (const RulePort &port : rule.ports) {
    PortSpec spec;
    spec.name = port.name;
    spec.isInput = port.isInput;
    if (port.isInput) {
      if (inputIndex < node.inputs.size())
        spec.value = node.inputs[inputIndex++].value;
    } else {
      if (outputIndex < node.outputs.size())
        spec.value = node.outputs[outputIndex++].value;
    }
    candidate.ports.push_back(std::move(spec));
  }

  for (const KindRequirement &requirement : rule.kindRequirements) {
    if (requirement.role == "executor") {
      ExecutorRequirement resolved;
      resolved.capability = requirement.kind;
      candidate.executorRequirements.push_back(std::move(resolved));
    } else if (requirement.role == "memory") {
      MemoryRequirement resolved;
      resolved.kind = requirement.kind;
      candidate.memoryRequirements.push_back(std::move(resolved));
    } else if (requirement.role == "compute") {
      ComputeRequirement resolved;
      resolved.kind = requirement.kind;
      candidate.computeRequirements.push_back(std::move(resolved));
    }
  }

  for (const RuleLayoutRequirement &requirement : rule.layoutRequirements) {
    // When a layout is bound, only the requirement(s) naming it are
    // materialized: the binding, not the rule file, chooses which of the rule's
    // declared layouts applies, so a rule that offered several selects the one
    // the search-space point named. Without a bound layout every requirement is
    // kept exactly as before.
    //
    // Known conflation: this treats "two ids on one port" (alternatives) and
    // "ids on two ports" (distinct obligations) alike, so a future multi-port
    // rule would silently drop an obligation -- under-constraint, the opposite
    // direction to the empty-list veto S7 fixed. Every shipped rule declares at
    // most one `require layout`, so the cases do not yet diverge; a per-port
    // selection surface is the place to split them if one appears.
    if (boundLayout && requirement.layoutId != *boundLayout)
      continue;
    LayoutRequirement resolved;
    resolved.layoutClass = requirement.layoutId;
    // A layout applies to the operand the requirement names, so resolve that
    // operand to its own element type and rank: the layout is then solved
    // against the value it constrains rather than the graph's first value,
    // which for a mixed-dtype kernel is a different dtype. Left unset when the
    // port exposes neither, so placement falls back to the caller's context.
    size_t layoutInputIndex = 0;
    size_t layoutOutputIndex = 0;
    for (const RulePort &port : rule.ports) {
      const WorkloadPort *nodePort = nullptr;
      if (port.isInput) {
        if (layoutInputIndex < node.inputs.size())
          nodePort = &node.inputs[layoutInputIndex];
        ++layoutInputIndex;
      } else {
        if (layoutOutputIndex < node.outputs.size())
          nodePort = &node.outputs[layoutOutputIndex];
        ++layoutOutputIndex;
      }
      if (port.name != requirement.port || !nodePort)
        continue;
      if (mlir::Type element = elementTypeOf(nodePort->type))
        resolved.elementType = printedType(element);
      if (std::optional<llvm::SmallVector<int64_t, 4>> shape =
              staticShapeOf(nodePort->type))
        resolved.rank = static_cast<int64_t>(shape->size());
      // The value this port carries, so the solved binding can be attributed to
      // the edge that carries it. A rule that names only an input port
      // therefore does *not* claim a layout for the value its node produces.
      resolved.portValue = static_cast<int64_t>(nodePort->value);
    }
    candidate.layoutRequirements.push_back(std::move(resolved));
  }

  candidate.id = computeCandidateId(candidate);
  return candidate;
}

} // namespace mlir::llk::mapping
