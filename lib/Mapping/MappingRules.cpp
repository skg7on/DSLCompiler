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

#include "LLK/Mapping/Diagnostics.h"
#include "LLK/Mapping/MappingHelpers.h"
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
  bool parseGraphMatch(RuleDef &out);
  bool parsePatternNode(RulePattern &pattern);
  bool parsePatternEdge(RulePattern &pattern);
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
  // A memory requirement's port subject must name a declared port of the same
  // direction, or the requirement could never resolve to an occurrence. Checked
  // after the whole rule body so declaration order does not matter.
  for (const KindRequirement &requirement : out.kindRequirements) {
    if (!requirement.port)
      continue;
    bool declared = false;
    for (const RulePort &port : out.ports)
      if (port.name == requirement.port->name &&
          port.isInput == requirement.port->isInput) {
        declared = true;
        break;
      }
    if (!declared)
      return failAt(current(),
                    "rule '" + out.id +
                        "': memory requirement names undeclared " +
                        (requirement.port->isInput ? "input '" : "output '") +
                        requirement.port->name + "'");
  }
  return true;
}

bool RuleParser::parseMatch(RuleDef &out) {
  if (!out.matchOp.empty())
    return failAt(current(), "duplicate match clause");
  advance(); // 'match'

  // `match graph { ... }` is the fused form. The single-operation syntax is
  // untouched, so every existing rule keeps parsing to exactly what it did.
  if (current().kind == LlkMapToken::Kind::Identifier &&
      current().text == "graph")
    return parseGraphMatch(out);

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

/// `match graph { node X: micro.vector(op = "convert"); edge X.result ->
/// Y.operand0; }`
///
/// The first declared node is the **anchor**: matches are enumerated by trying
/// it at each graph node in canonical order, so the order of matches is a
/// property of the rule and the graph rather than of a traversal.
bool RuleParser::parseGraphMatch(RuleDef &out) {
  advance(); // 'graph'
  if (!expectPunct("{"))
    return false;

  RulePattern pattern;
  while (!isPunct("}")) {
    if (atEnd())
      return failAt(current(), "expected '}' to close the match graph");
    if (current().kind != LlkMapToken::Kind::Identifier)
      return failAt(current(), "expected 'node' or 'edge'");
    std::string keyword = current().text;
    if (keyword == "node") {
      if (!parsePatternNode(pattern))
        return false;
    } else if (keyword == "edge") {
      if (!parsePatternEdge(pattern))
        return false;
    } else {
      return failAt(current(),
                    "expected 'node' or 'edge', found '" + keyword + "'");
    }
  }
  // No trailing `;`: the braces delimit the statement, and requiring one would
  // be a second way to write the same thing.
  advance(); // '}'

  if (pattern.nodes.empty())
    return failAt(current(), "a match graph needs at least one node");

  // Every edge endpoint names a node the graph declared. Checked after the
  // whole graph, so declaration order does not matter -- a forward reference to
  // a node declared below is as valid as one above.
  for (const RulePatternEdge &edge : pattern.edges) {
    if (!pattern.findNode(edge.producer))
      return failAt(current(),
                    "edge names undeclared node '" + edge.producer + "'");
    if (!pattern.findNode(edge.consumer))
      return failAt(current(),
                    "edge names undeclared node '" + edge.consumer + "'");
  }

  out.matchOp = pattern.nodes.front().op;
  out.pattern = std::move(pattern);
  return true;
}

/// `node NAME: micro.vector(op = "convert", ...);`
bool RuleParser::parsePatternNode(RulePattern &pattern) {
  advance(); // 'node'
  std::string name;
  if (!expectIdentifier("a node name", name))
    return false;
  for (const RulePatternNode &existing : pattern.nodes)
    if (existing.name == name)
      return failAt(current(),
                    "duplicate node name '" + name + "' in the match graph");
  if (!expectPunct(":"))
    return false;

  RulePatternNode node;
  node.name = std::move(name);
  if (!expectIdentifier("a Micro operation", node.op))
    return false;
  if (!isWorkloadNodeOp(node.op))
    return failAt(current(), "unknown Micro operation '" + node.op + "'");

  if (!expectPunct("("))
    return false;
  if (!isPunct(")")) {
    while (true) {
      RulePredicate predicate;
      if (!parsePredicate(predicate))
        return false;
      node.predicates.push_back(std::move(predicate));
      if (isPunct(",")) {
        advance();
        continue;
      }
      break;
    }
  }
  if (!expectPunct(")"))
    return false;
  if (!expectPunct(";"))
    return false;

  pattern.nodes.push_back(std::move(node));
  return true;
}

/// `edge PRODUCER.resultK -> CONSUMER.operandK;`
///
/// The index may be omitted: `X.result` is the first result and `Y.operand` the
/// first operand, because most patterns touch exactly one of each and writing
/// the zero every time would be noise.
bool RuleParser::parsePatternEdge(RulePattern &pattern) {
  advance(); // 'edge'

  // `cv.result` is one token: a dot is an identifier character here, because
  // rule and layout ids are dotted. The endpoint is split rather than lexed
  // apart.
  auto parseEndpoint = [&](std::string &name, uint32_t &index,
                           llvm::StringRef prefix) -> bool {
    std::string endpoint;
    if (!expectIdentifier("a node and its occurrence", endpoint))
      return false;
    size_t dot = endpoint.rfind('.');
    if (dot == std::string::npos)
      return failAt(current(), "expected 'NODE." + prefix.str() +
                                   "' or 'NODE." + prefix.str() +
                                   "K', found '" + endpoint + "'");
    llvm::StringRef occurrence = llvm::StringRef(endpoint).drop_front(dot + 1);
    name = endpoint.substr(0, dot);
    if (name.empty() || !occurrence.starts_with(prefix))
      return failAt(current(), "expected 'NODE." + prefix.str() +
                                   "' or 'NODE." + prefix.str() +
                                   "K', found '" + endpoint + "'");
    llvm::StringRef digits = occurrence.drop_front(prefix.size());
    uint64_t value = 0;
    if (!digits.empty() && digits.getAsInteger(10, value))
      return failAt(current(),
                    "expected an occurrence index in '" + endpoint + "'");
    index = static_cast<uint32_t>(value);
    return true;
  };

  RulePatternEdge edge;
  if (!parseEndpoint(edge.producer, edge.resultIndex, "result"))
    return false;
  if (!expectPunct("->"))
    return false;
  if (!parseEndpoint(edge.consumer, edge.operandIndex, "operand"))
    return false;
  if (!expectPunct(";"))
    return false;

  for (const RulePatternEdge &existing : pattern.edges)
    if (existing.producer == edge.producer &&
        existing.resultIndex == edge.resultIndex &&
        existing.consumer == edge.consumer &&
        existing.operandIndex == edge.operandIndex)
      return failAt(current(), "duplicate edge in the match graph");

  pattern.edges.push_back(std::move(edge));
  return true;
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
  if (isKindRole) {
    // Two clause forms share the role prefix: the legacy `kind <k>` and a
    // memory requirement that names the port it governs, `output "x" kind <k>`.
    const bool bareForm =
        peek().kind == LlkMapToken::Kind::Identifier && peek().text == "kind";
    const bool portForm = peek().kind == LlkMapToken::Kind::Identifier &&
                          (peek().text == "input" || peek().text == "output");
    if (bareForm || portForm) {
      KindRequirement requirement;
      requirement.role = current().text;
      const LlkMapToken roleToken = current();
      advance(); // role
      if (portForm) {
        // Only memory is materialized per port; the owner and compute roles are
        // a property of the whole operation, so naming a port there is a
        // load-time error rather than a silently ignored field.
        if (requirement.role != "memory")
          return failAt(roleToken, "only a memory requirement may name a port");
        RulePort port;
        port.isInput = current().text == "input";
        advance(); // direction
        if (current().kind != LlkMapToken::Kind::String)
          return failAt(current(), "expected a quoted port name");
        port.name = current().text;
        advance();
        requirement.port = std::move(port);
        if (current().kind != LlkMapToken::Kind::Identifier ||
            current().text != "kind")
          return failAt(current(), "expected 'kind'");
      }
      advance(); // 'kind'
      if (!expectIdentifier("a capability kind", requirement.kind))
        return false;
      if (!expectPunct(";"))
        return false;
      // Two requirements collide when they share role and kind and their port
      // subjects overlap. Two distinct named ports are distinct requirements;
      // a bare requirement and a named one of the same kind overlap (the bare
      // one would have to govern every port), so that is rejected too.
      auto collides = [](const std::optional<RulePort> &lhs,
                         const std::optional<RulePort> &rhs) {
        if (!lhs || !rhs)
          return true;
        return lhs->name == rhs->name && lhs->isInput == rhs->isInput;
      };
      for (const KindRequirement &existing : out.kindRequirements)
        if (existing.role == requirement.role &&
            existing.kind == requirement.kind &&
            collides(existing.port, requirement.port))
          return failAt(roleToken, "duplicate " + requirement.role +
                                       " kind requirement '" +
                                       requirement.kind + "'");
      out.kindRequirements.push_back(std::move(requirement));
      return true;
    }
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

const RulePatternNode *RulePattern::findNode(llvm::StringRef name) const {
  for (const RulePatternNode &node : nodes)
    if (node.name == name)
      return &node;
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

  auto predicateText = [](const RulePredicate &predicate) {
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
    return text;
  };
  for (const RulePredicate &predicate : def.predicates)
    field("predicate", predicateText(predicate));

  // A graph rule's pattern is part of what the rule *is*, so it is part of the
  // hash: two rule libraries that differ only in a pattern are different
  // libraries, and a report that named one would be wrong about the other.
  if (def.pattern) {
    for (const RulePatternNode &node : def.pattern->nodes) {
      field("pattern.node", node.name);
      field("pattern.op", node.op);
      for (const RulePredicate &predicate : node.predicates)
        field("pattern.predicate", predicateText(predicate));
    }
    for (const RulePatternEdge &edge : def.pattern->edges)
      field("pattern.edge",
            edge.producer + "." + std::to_string(edge.resultIndex) + "->" +
                edge.consumer + "." + std::to_string(edge.operandIndex));
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
  for (const KindRequirement &requirement : def.kindRequirements) {
    std::string text = requirement.role;
    if (requirement.port) {
      // Unit separators keep a port name containing the printable delimiter
      // from being read as the kind.
      text += '\x1f';
      text += requirement.port->isInput ? 'i' : 'o';
      text += '\x1f';
      text += requirement.port->name;
      text += '\x1f';
      text += requirement.kind;
    } else {
      // The bare form renders exactly as before, so a rule that names no port
      // keeps its pre-existing content hash.
      text += ':';
      text += requirement.kind;
    }
    field("require", text);
  }
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

  if (def.pattern) {
    out += "  match graph {\n";
    for (const RulePatternNode &node : def.pattern->nodes) {
      out += "    node " + node.name + ": " + node.op + "(";
      for (size_t index = 0; index < node.predicates.size(); ++index) {
        if (index)
          out += ", ";
        out += printPredicate(node.predicates[index]);
      }
      out += ");\n";
    }
    for (const RulePatternEdge &edge : def.pattern->edges) {
      out += "    edge " + edge.producer + ".result" +
             std::to_string(edge.resultIndex) + " -> " + edge.consumer +
             ".operand" + std::to_string(edge.operandIndex) + ";\n";
    }
    out += "  }\n";
  } else {
    out += "  match " + def.matchOp + "(";
    for (size_t index = 0; index < def.predicates.size(); ++index) {
      if (index)
        out += ", ";
      out += printPredicate(def.predicates[index]);
    }
    out += ");\n";
  }

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
  for (const KindRequirement &requirement : def.kindRequirements) {
    out += "  require " + requirement.role;
    if (requirement.port) {
      out += requirement.port->isInput ? " input \"" : " output \"";
      out += requirement.port->name;
      out += '"';
    }
    out += " kind " + requirement.kind + ";\n";
  }
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
    // A graph rule matches a whole subgraph, never one operation. Offering it
    // here would place it on a single node while its pattern claimed three --
    // the very overlap a covering has to refuse.
    if (rule.pattern)
      continue;
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

//===----------------------------------------------------------------------===//
// Bounded graph-pattern matching
//===----------------------------------------------------------------------===//

namespace {

/// The most graph nodes one pattern may bind, so a pathological rule cannot
/// make matching unbounded.
constexpr size_t kMaxPatternNodes = 64;

/// True when the value at `producer`'s `resultIndex`-th result is the value at
/// `consumer`'s `operandIndex`-th operand.
///
/// The check is on *occurrences*, not on values: two operands that happen to
/// hold the same value are still two different ports, and an edge that named
/// one of them does not name the other.
bool edgeHolds(const WorkloadNode &producer, const WorkloadNode &consumer,
               const RulePatternEdge &edge) {
  if (edge.resultIndex >= producer.outputs.size())
    return false;
  if (edge.operandIndex >= consumer.inputs.size())
    return false;
  return producer.outputs[edge.resultIndex].value ==
         consumer.inputs[edge.operandIndex].value;
}

/// Binds `pattern`'s nodes to graph nodes, in the pattern's declaration order.
///
/// Binding in declaration order is what makes the *order of matches* a property
/// of the rule, and the final all-edges check is what makes it correct even
/// when an edge names a node declared later -- an edge is a statement about the
/// matched subgraph, not about the order the rule happened to write it in.
class PatternMatcher {
public:
  PatternMatcher(const WorkloadGraph &graph, const RulePattern &pattern,
                 uint64_t budget)
      : pattern_(pattern), budget_(budget) {
    nodes_.reserve(graph.getNodes().size());
    for (const WorkloadNode &node : graph.getNodes())
      nodes_.push_back(&node);
    // Canonical node order: matches are enumerated by trying the anchor at each
    // node in this order, so the graph's insertion order never reaches a match.
    llvm::sort(nodes_, [](const WorkloadNode *lhs, const WorkloadNode *rhs) {
      return lhs->id < rhs->id;
    });
  }

  /// Every match, up to the budget. Sets `truncated` when the budget was
  /// reached with more combinations left untried.
  std::vector<RulePatternMatch> run(const RuleDef &rule, bool *truncated) {
    rule_ = &rule;
    bound_.assign(pattern_.nodes.size(), nullptr);
    truncated_ = false;

    const RulePatternNode &anchor = pattern_.nodes.front();
    for (const WorkloadNode *candidate : nodes_) {
      if (truncated_)
        break;
      if (candidate->opName != anchor.op)
        continue;
      if (!predicatesHold(anchor, *candidate))
        continue;
      if (used_.count(candidate->id))
        continue;
      bound_[0] = candidate;
      used_.insert(candidate->id);
      recurse(1);
      used_.erase(candidate->id);
      bound_[0] = nullptr;
    }

    if (truncated)
      *truncated = truncated_;
    return std::move(matches_);
  }

private:
  bool predicatesHold(const RulePatternNode &node,
                      const WorkloadNode &candidate) const {
    for (const RulePredicate &predicate : node.predicates)
      if (!predicateMatches(predicate, candidate))
        return false;
    return true;
  }

  /// The edges between `index` and the nodes already bound must hold for
  /// `candidate` to be a legal binding. Edges to nodes not yet bound are left
  /// to the final check.
  bool partialEdgesHold(size_t index, const WorkloadNode &candidate) const {
    const std::string &name = pattern_.nodes[index].name;
    for (const RulePatternEdge &edge : pattern_.edges) {
      const WorkloadNode *other = nullptr;
      bool candidateIsConsumer = false;
      if (edge.consumer == name) {
        const RulePatternNode *producerNode = pattern_.findNode(edge.producer);
        if (!producerNode)
          continue;
        size_t producerIndex =
            static_cast<size_t>(producerNode - pattern_.nodes.data());
        if (producerIndex >= index || !bound_[producerIndex])
          continue;
        other = bound_[producerIndex];
        candidateIsConsumer = true;
      } else if (edge.producer == name) {
        const RulePatternNode *consumerNode = pattern_.findNode(edge.consumer);
        if (!consumerNode)
          continue;
        size_t consumerIndex =
            static_cast<size_t>(consumerNode - pattern_.nodes.data());
        if (consumerIndex >= index || !bound_[consumerIndex])
          continue;
        other = bound_[consumerIndex];
      } else {
        continue;
      }

      // `edgeHolds` takes (producer, consumer, edge), so the candidate goes
      // first exactly when it is the producer.
      bool holds = candidateIsConsumer ? edgeHolds(*other, candidate, edge)
                                       : edgeHolds(candidate, *other, edge);
      if (!holds)
        return false;
    }
    return true;
  }

  void recurse(size_t index) {
    if (truncated_)
      return;
    if (index == pattern_.nodes.size()) {
      recordMatch();
      return;
    }
    const RulePatternNode &node = pattern_.nodes[index];
    for (const WorkloadNode *candidate : nodes_) {
      if (truncated_)
        return;
      if (candidate->opName != node.op)
        continue;
      if (!predicatesHold(node, *candidate))
        continue;
      if (used_.count(candidate->id))
        continue;
      if (!partialEdgesHold(index, *candidate))
        continue;
      bound_[index] = candidate;
      used_.insert(candidate->id);
      recurse(index + 1);
      used_.erase(candidate->id);
      bound_[index] = nullptr;
    }
  }

  void recordMatch() {
    // Every edge, now that both endpoints are bound.
    for (const RulePatternEdge &edge : pattern_.edges) {
      const RulePatternNode *producerNode = pattern_.findNode(edge.producer);
      const RulePatternNode *consumerNode = pattern_.findNode(edge.consumer);
      if (!producerNode || !consumerNode)
        return;
      size_t producerIndex =
          static_cast<size_t>(producerNode - pattern_.nodes.data());
      size_t consumerIndex =
          static_cast<size_t>(consumerNode - pattern_.nodes.data());
      if (!edgeHolds(*bound_[producerIndex], *bound_[consumerIndex], edge))
        return;
    }

    if (matches_.size() >= budget_) {
      truncated_ = true;
      return;
    }

    RulePatternMatch match;
    match.rule = rule_;
    for (const WorkloadNode *node : bound_)
      match.coveredNodes.push_back(node->id);

    // The boundary: every operand and result occurrence of a matched node that
    // is not an endpoint of an internal edge. Those are what the rest of the
    // program attaches to; an occurrence inside the match is the rule's own.
    for (size_t i = 0; i < pattern_.nodes.size(); ++i) {
      const std::string &name = pattern_.nodes[i].name;
      const WorkloadNode &node = *bound_[i];
      for (uint32_t operand = 0;
           operand < static_cast<uint32_t>(node.inputs.size()); ++operand) {
        bool internal = false;
        for (const RulePatternEdge &edge : pattern_.edges)
          if (edge.consumer == name && edge.operandIndex == operand) {
            internal = true;
            break;
          }
        if (!internal)
          match.boundary.push_back(
              PortRef{node.id, PortDirection::Input, operand});
      }
      for (uint32_t result = 0;
           result < static_cast<uint32_t>(node.outputs.size()); ++result) {
        bool internal = false;
        for (const RulePatternEdge &edge : pattern_.edges)
          if (edge.producer == name && edge.resultIndex == result) {
            internal = true;
            break;
          }
        if (!internal)
          match.boundary.push_back(
              PortRef{node.id, PortDirection::Output, result});
      }
    }
    // Canonical boundary order, so two matches of one subgraph are comparable
    // whatever order the pattern happened to walk them in.
    llvm::sort(match.boundary, [](const PortRef &lhs, const PortRef &rhs) {
      if (lhs.node != rhs.node)
        return lhs.node < rhs.node;
      if (lhs.direction != rhs.direction)
        return lhs.direction < rhs.direction;
      return lhs.index < rhs.index;
    });

    matches_.push_back(std::move(match));
  }

  const RulePattern &pattern_;
  const RuleDef *rule_ = nullptr;
  uint64_t budget_ = 0;
  std::vector<const WorkloadNode *> nodes_;
  std::vector<const WorkloadNode *> bound_;
  llvm::SmallDenseSet<WorkloadNodeId, 16> used_;
  std::vector<RulePatternMatch> matches_;
  bool truncated_ = false;
};

} // namespace

std::vector<RulePatternMatch> matchRulePatterns(const WorkloadGraph &graph,
                                                const RuleRegistry &rules,
                                                uint64_t maxMatches,
                                                bool *truncated) {
  std::vector<RulePatternMatch> matches;
  if (truncated)
    *truncated = false;
  if (maxMatches == 0)
    return matches;

  for (const RuleDef &rule : rules.all()) {
    if (!rule.pattern)
      continue;
    if (rule.pattern->nodes.size() > kMaxPatternNodes)
      continue;
    // One rule's matches are found together so the budget is shared across the
    // rule library: a cap that reset per rule would not cap anything.
    bool ruleTruncated = false;
    PatternMatcher matcher(graph, *rule.pattern, maxMatches - matches.size());
    std::vector<RulePatternMatch> found = matcher.run(rule, &ruleTruncated);
    for (RulePatternMatch &match : found)
      matches.push_back(std::move(match));
    if (ruleTruncated) {
      if (truncated)
        *truncated = true;
      break;
    }
  }
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
mlir::DictionaryAttr
buildBundleParameters(const RuleDef &rule, const WorkloadNode &node,
                      const llvm::StringMap<LayoutValue> &resolvedParameters) {
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
    if (const auto *integer = std::get_if<int64_t>(&entry.second)) {
      value =
          mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64), *integer);
    } else if (auto resolved =
                   resolvedParameters.find(std::get<std::string>(entry.second));
               resolved != resolvedParameters.end() &&
               std::holds_alternative<int64_t>(resolved->second)) {
      value = mlir::IntegerAttr::get(mlir::IntegerType::get(context, 64),
                                     std::get<int64_t>(resolved->second));
    } else {
      value =
          mlir::StringAttr::get(context, std::get<std::string>(entry.second));
    }
    attributes.emplace_back(mlir::StringAttr::get(context, entry.first), value);
  }
  return mlir::DictionaryAttr::get(context, attributes);
}

/// Projects the bound `owner_mapping` and `memory_path` search axes onto the
/// candidate's abstract requirements, validating each against machine data. The
/// projection is explicit: a bound axis changes which executors, computes, and
/// memories placement may bind, rather than being recorded as provenance and
/// ignored. Returns an error message when an axis is malformed, names something
/// the machine does not model, or contradicts the rule's own requirements; on
/// success it appends the projected requirements to `candidate`.
///
/// `owner_mapping` is a `/`-separated chain of owner kinds (`worker/pe`): the
/// outer component is the executor the work must run on, and inner components
/// are nested owners or compute capabilities. A rule's own executor and compute
/// requirements must lie on that chain.
///
/// `memory_path` is a `:`-separated chain of memory kinds (`dram:sram:acc`),
/// the allowed levels. Every memory a rule requires must sit on it. A level the
/// machine does not model, or an owner kind it does not model, is rejected
/// explicitly -- a selected axis is never silently dropped.
///
/// The value for an axis is taken from `boundAxes` (resolved by parameter
/// *kind* by the caller) when set, else from the binding's conventional
/// exported name in `pinned` -- so a name-keyed binding still projects, and a
/// space that named its parameter differently is honoured because the caller
/// resolved it by kind rather than by name.
std::optional<std::string>
projectBoundAxes(const RuleDef &rule, const BoundAxes *boundAxes,
                 const llvm::StringMap<SearchValue> *pinned,
                 const machine::MachineModel &machine,
                 MappingCandidate &candidate) {
  std::string ownerValue;
  if (boundAxes && !boundAxes->ownerMapping.empty())
    ownerValue = boundAxes->ownerMapping;
  else if (pinned)
    if (auto entry = pinned->find("owner_mapping"); entry != pinned->end()) {
      const std::string *value = std::get_if<std::string>(&entry->second);
      if (!value || value->empty())
        return "owner_mapping axis must be a non-empty symbolic value";
      ownerValue = *value;
    }

  if (!ownerValue.empty()) {
    llvm::StringRef value = ownerValue;
    llvm::SmallVector<llvm::StringRef, 4> owners;
    value.split(owners, '/');
    for (llvm::StringRef owner : owners)
      if (owner.empty())
        return "owner_mapping axis has an empty owner";
    // The outer component names the executor the work runs on.
    if (!machine.hasOwnerKind(owners.front()))
      return "owner_mapping names owner '" + owners.front().str() +
             "', which machine '" + machine.target + "' does not model";
    for (const KindRequirement &requirement : rule.kindRequirements)
      if (requirement.role == "executor" && requirement.kind != owners.front())
        return "owner_mapping '" + value.str() +
               "' contradicts executor kind '" + requirement.kind + "'";
    // Inner components are nested owners or attached compute capabilities; each
    // must be modeled as one or the other.
    llvm::SmallVector<std::string, 2> projectedComputes;
    for (size_t index = 1; index < owners.size(); ++index) {
      llvm::StringRef owner = owners[index];
      bool isOwnerKind = machine.hasOwnerKind(owner);
      bool isComputeKind = !machine.computesOfKind(owner).empty();
      if (!isOwnerKind && !isComputeKind)
        return "owner_mapping names owner '" + owner.str() +
               "', which machine '" + machine.target + "' does not model";
      if (isComputeKind)
        projectedComputes.push_back(owner.str());
    }
    // A rule that requires a compute must require one the chain reaches.
    for (const KindRequirement &requirement : rule.kindRequirements) {
      if (requirement.role != "compute")
        continue;
      if (!projectedComputes.empty() &&
          !llvm::is_contained(projectedComputes, requirement.kind))
        return "owner_mapping '" + value.str() +
               "' does not reach compute kind '" + requirement.kind + "'";
    }
    // Project the outer owner as an executor requirement, and each inner
    // compute kind as a compute requirement -- deduplicated, so a rule that
    // already declares them is left unchanged.
    bool hasExecutor = false;
    for (const ExecutorRequirement &existing : candidate.executorRequirements)
      hasExecutor |= existing.capability == owners.front();
    if (!hasExecutor) {
      ExecutorRequirement resolved;
      resolved.capability = owners.front().str();
      candidate.executorRequirements.push_back(std::move(resolved));
    }
    for (const std::string &kind : projectedComputes) {
      bool declared = false;
      for (const ComputeRequirement &existing : candidate.computeRequirements)
        declared |= existing.kind == kind;
      if (!declared) {
        ComputeRequirement resolved;
        resolved.kind = kind;
        candidate.computeRequirements.push_back(std::move(resolved));
      }
    }
  }

  std::string memoryPathValue;
  if (boundAxes && !boundAxes->memoryPath.empty())
    memoryPathValue = boundAxes->memoryPath;
  else if (pinned)
    if (auto entry = pinned->find("memory_path"); entry != pinned->end()) {
      const std::string *value = std::get_if<std::string>(&entry->second);
      if (!value || value->empty())
        return "memory_path axis must be a non-empty symbolic value";
      memoryPathValue = *value;
    }

  if (!memoryPathValue.empty()) {
    llvm::StringRef value = memoryPathValue;
    llvm::SmallVector<llvm::StringRef, 4> path;
    value.split(path, ':');
    for (llvm::StringRef level : path)
      if (level.empty() || !machine.findMemoryOfKind(level))
        return "memory_path names level '" + level.str() +
               "', which machine '" + machine.target + "' does not model";
    for (const MemoryRequirement &requirement : candidate.memoryRequirements)
      if (!llvm::is_contained(path, llvm::StringRef(requirement.kind)))
        return "memory_path '" + value.str() + "' excludes memory kind '" +
               requirement.kind + "'";
  }
  return std::nullopt;
}

} // namespace

std::optional<MappingCandidate>
toMappingCandidate(const RuleDef &rule, const WorkloadNode &node,
                   const machine::MachineModel &machine,
                   const LayoutContext &context, std::string *reason,
                   bool *truncated, const llvm::StringMap<SearchValue> *pinned,
                   const llvm::StringMap<std::string> *boundLayouts,
                   const BoundAxes *boundAxes) {
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
  if (boundLayouts && !rule.layoutRequirements.empty()) {
    // Each bound value is checked against the obligation it belongs to. A
    // value bound for a *role* governs the requirement on that port: a rule
    // that declares no requirement there takes on no obligation and matches
    // unchanged (the same omission rule as the role-less case -- see below). A
    // value bound with no role governs the axis as a whole, so the rule must
    // offer it somewhere.
    for (const auto &entry : *boundLayouts) {
      llvm::StringRef role = entry.first();
      const std::string &value = entry.second;
      bool satisfied = false;
      for (const RuleLayoutRequirement &requirement : rule.layoutRequirements) {
        if (!role.empty() && requirement.port != role)
          continue;
        satisfied |= requirement.layoutId == value;
      }
      // A role the rule does not mention is no obligation -- unless the value
      // is the role-less one, which the rule must offer somewhere.
      if (!satisfied && role.empty()) {
        if (reason)
          *reason = "binding selects layout '" + value +
                    "', which the rule does not offer";
        return std::nullopt;
      }
      if (!satisfied && !role.empty()) {
        bool mentionsRole = false;
        for (const RuleLayoutRequirement &requirement : rule.layoutRequirements)
          mentionsRole |= requirement.port == role;
        if (mentionsRole) {
          if (reason)
            *reason = "binding selects layout '" + value + "' for port '" +
                      role.str() + "', which the rule does not offer there";
          return std::nullopt;
        }
      }
    }
  }

  MappingCandidate candidate;
  candidate.rule = rule.id;
  candidate.coveredNodes.push_back(node.id);
  candidate.bundle.name = rule.bundle;
  candidate.bundle.emitterKey = rule.emitter;
  candidate.bundle.parameters =
      buildBundleParameters(rule, node, resolution.parameters);
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
      if (inputIndex < node.inputs.size()) {
        spec.value = node.inputs[inputIndex].value;
        // The operand occurrence, so two uses of one value stay distinct ports
        // rather than collapsing into one spec.
        spec.port = PortRef{node.id, PortDirection::Input,
                            static_cast<uint32_t>(inputIndex)};
      }
      ++inputIndex;
    } else {
      if (outputIndex < node.outputs.size()) {
        spec.value = node.outputs[outputIndex].value;
        spec.port = PortRef{node.id, PortDirection::Output,
                            static_cast<uint32_t>(outputIndex)};
      }
      ++outputIndex;
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
      // A named port is resolved to the matched node's occurrence the same way
      // a layout requirement is: rule ports wire positionally to the node's
      // inputs then its outputs, so the port's position within its direction is
      // the occurrence index. A requirement whose port the node does not have
      // cannot be attributed, so the rule is a non-match rather than a bare
      // fallback.
      if (requirement.port) {
        resolved.port = portRefForRulePort(rule, node, *requirement.port);
        if (!resolved.port) {
          if (reason)
            *reason = "memory requirement names port '" +
                      requirement.port->name +
                      "', which the matched operation does not expose";
          return std::nullopt;
        }
      }
      candidate.memoryRequirements.push_back(std::move(resolved));
    } else if (requirement.role == "compute") {
      ComputeRequirement resolved;
      resolved.kind = requirement.kind;
      candidate.computeRequirements.push_back(std::move(resolved));
    }
  }

  // Bound search axes are projected onto explicit requirements before the
  // candidate is identified, so placement consumes what the binding selected
  // rather than ignoring it as provenance.
  if (pinned || (boundAxes && !boundAxes->empty())) {
    if (std::optional<std::string> axisError =
            projectBoundAxes(rule, boundAxes, pinned, machine, candidate)) {
      if (reason)
        *reason = *axisError;
      return std::nullopt;
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
    // Keep a requirement only when it is the one the binding selected for its
    // role -- or for the role-less axis, which applies to every role. With no
    // binding, every requirement is kept exactly as before.
    if (boundLayouts) {
      const std::string *byRole = nullptr;
      if (auto it = boundLayouts->find(requirement.port);
          it != boundLayouts->end())
        byRole = &it->second;
      else if (auto it = boundLayouts->find(""); it != boundLayouts->end())
        byRole = &it->second;
      if (byRole && requirement.layoutId != *byRole)
        continue;
    }
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
      PortDirection direction = PortDirection::Input;
      uint32_t position = 0;
      if (port.isInput) {
        position = static_cast<uint32_t>(layoutInputIndex);
        if (layoutInputIndex < node.inputs.size())
          nodePort = &node.inputs[layoutInputIndex];
        ++layoutInputIndex;
      } else {
        direction = PortDirection::Output;
        position = static_cast<uint32_t>(layoutOutputIndex);
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
      // The operand occurrence itself, so two uses of one value resolve their
      // own layout instead of a value-wide lookup that would give up.
      resolved.port = PortRef{node.id, direction, position};
    }
    candidate.layoutRequirements.push_back(std::move(resolved));
  }

  candidate.id = computeCandidateId(candidate);
  return candidate;
}

//===----------------------------------------------------------------------===//
// Re-verifying a recorded selection (design §18.3, phase 2)
//===----------------------------------------------------------------------===//

namespace {

/// Why a recorded parameter assignment is illegal, or nullopt when it is legal.
/// Validates the assignment the plan *recorded* -- it never searches for a
/// different one -- so an unknown name, a value outside its declared domain, a
/// required name left out, and a constraint the recorded values do not satisfy
/// are each reported. The fallback satisfiability test lives in the caller, so
/// this is reached only when an assignment was actually persisted.
std::optional<std::string>
recordedParameterProblem(const RuleDef &rule, const WorkloadNode &node,
                         const machine::MachineModel &machine,
                         const llvm::StringMap<SearchValue> &recorded) {
  for (const auto &entry : recorded) {
    if (!rule.findParam(entry.first()))
      return "records parameter '" + entry.first().str() +
             "', which it does not declare";
    auto domain = rule.domains.find(entry.first().str());
    if (domain == rule.domains.end() ||
        !llvm::is_contained(domain->second.values, entry.second))
      return "records parameter '" + entry.first().str() + " = " +
             canonicalValueString(entry.second) +
             "', outside its declared domain";
  }

  // Every parameter a constraint derives must have been recorded: verification
  // cannot reconstruct the value the plan actually used.
  llvm::StringSet<> referenced;
  llvm::StringSet<> bound; // no quantifier is in scope at the top level
  for (const ExprPtr &constraint : rule.constraints)
    collectIdentifiers(*constraint, referenced, bound);
  for (const LayoutParam &param : rule.params)
    if (referenced.contains(param.name) && !recorded.count(param.name))
      return "does not record the derived parameter '" + param.name + "'";

  llvm::StringMap<LayoutValue> bindings;
  for (const auto &entry : recorded)
    bindings[entry.first()] = entry.second;

  LayoutContext ruleContext = ruleLayoutContext(node, LayoutContext{});
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
  EvalOptions options;
  options.budget = &budget;
  options.domainResolver = resolveDomain;

  for (const ExprPtr &constraint : rule.constraints) {
    llvm::Expected<EvalValue> value =
        evaluateExpr(*constraint, bindings, machine, ruleContext, options);
    if (!value)
      return "does not satisfy its require constraints: " +
             llvm::toString(value.takeError());
    if (value->kind != EvalValue::Kind::Int || value->intValue == 0)
      return "records a parameter assignment its require constraints reject";
  }
  if (budget.exhausted)
    return "require evaluation exceeded the quantifier bound without deciding "
           "every constraint";
  return std::nullopt;
}

} // namespace

bool ruleDerivesParameters(const RuleDef &rule) {
  llvm::StringSet<> referenced;
  llvm::StringSet<> bound; // no quantifier is in scope at the top level
  for (const ExprPtr &constraint : rule.constraints)
    collectIdentifiers(*constraint, referenced, bound);
  for (const LayoutParam &param : rule.params)
    if (referenced.contains(param.name))
      return true;
  return false;
}

llvm::Error verifyRuleSelection(const RuleDef &rule, const WorkloadNode &node,
                                const machine::MachineModel &machine,
                                const RecordedRuleSelection &selection,
                                llvm::StringRef where) {
  auto reject = [&](DiagnosticCode code, std::string detail) {
    std::string message = stringifyDiagnosticCode(code).str();
    message += ": ";
    if (!where.empty()) {
      message += where.str();
      message += ": ";
    }
    message += detail;
    return llvm::createStringError(llvm::inconvertibleErrorCode(), message);
  };

  // 1. The recorded rule must implement this operation, not merely exist.
  if (rule.matchOp != node.opName)
    return reject(DiagnosticCode::NoMatchingRule,
                  "rule '" + rule.id + "' implements '" + rule.matchOp +
                      "', not '" + node.opName + "'");

  // 2. Every predicate must still hold for the operation's endpoints.
  for (const RulePredicate &predicate : rule.predicates)
    if (!predicateMatches(predicate, node))
      return reject(DiagnosticCode::NoMatchingRule,
                    "rule '" + rule.id + "' predicate '" +
                        printPredicate(predicate) +
                        "' does not match the operation");

  // 3. The recorded parameter assignment, or -- when none was recorded --
  // generation's existential check, so a rule that no assignment can satisfy is
  // still rejected.
  if (!selection.parameters.empty()) {
    if (std::optional<std::string> problem =
            recordedParameterProblem(rule, node, machine, selection.parameters))
      return reject(DiagnosticCode::NoMatchingRule,
                    "rule '" + rule.id + "' " + *problem);
  } else if (!rule.constraints.empty()) {
    RuleResolution resolution =
        resolveRuleConstraints(rule, node, machine, LayoutContext{}, nullptr);
    if (!resolution.matched)
      return reject(DiagnosticCode::NoMatchingRule,
                    "rule '" + rule.id + "' " +
                        (resolution.reason.empty()
                             ? std::string("require constraints not satisfied")
                             : resolution.reason));
  }

  // 4. The recorded executor, and the executor *kind* each requirement names.
  if (!selection.executor.empty() && !machine.findExecutor(selection.executor))
    return reject(DiagnosticCode::NoLegalExecutor,
                  "unknown executor '" + selection.executor + "'");
  for (const KindRequirement &requirement : rule.kindRequirements)
    if (requirement.role == "executor" &&
        !machine.ownerMatches(requirement.kind, selection.executor))
      return reject(DiagnosticCode::NoLegalExecutor,
                    "executor '" + selection.executor +
                        "' does not satisfy the rule's required kind '" +
                        requirement.kind + "'");

  // 5. Attached compute capabilities. When the binding records which concrete
  // node it selected, verification checks *that* node -- it must exist, carry
  // the required kind, and be attached to the recorded executor -- and never
  // falls back to the executor's first attached capability (issue #129, task
  // R1). A binding that records none keeps the older existential check, so a
  // pre-v3 plan stays verifiable while a recorded selection is authoritative.
  for (const KindRequirement &requirement : rule.kindRequirements) {
    if (requirement.role != "compute")
      continue;
    auto recorded = selection.computeBindings.find(requirement.kind);
    if (recorded == selection.computeBindings.end()) {
      // A binding that recorded a selection container but no node for a kind
      // the rule requires has had that decision dropped or tampered with. It is
      // rejected rather than passed by the existential check, which would let
      // whichever attached capability the executor lists first stand in for the
      // selected one (issue #129, task R1).
      if (selection.computeBindingsRecorded)
        return reject(DiagnosticCode::UnsupportedComputeFragment,
                      "rule '" + rule.id + "' requires a '" + requirement.kind +
                          "' compute capability, but the mapping records no "
                          "concrete node for it");
      // No container was recorded at all: keep generation's existential check,
      // so a binding that predates compute persistence stays verifiable.
      bool attached = false;
      for (const mlir::llk::machine::ComputeNode *compute :
           machine.computesFor(selection.executor))
        if (compute->kind == requirement.kind) {
          attached = true;
          break;
        }
      if (!attached)
        return reject(DiagnosticCode::UnsupportedComputeFragment,
                      "executor '" + selection.executor +
                          "' has no attached '" + requirement.kind +
                          "' compute capability");
      continue;
    }
    const mlir::llk::machine::ComputeNode *compute =
        machine.findCompute(recorded->second);
    if (!compute)
      return reject(DiagnosticCode::UnsupportedComputeFragment,
                    "unknown compute node '" + recorded->second + "'");
    if (compute->kind != requirement.kind)
      return reject(DiagnosticCode::UnsupportedComputeFragment,
                    "compute node '" + recorded->second + "' has kind '" +
                        compute->kind + "', not the required '" +
                        requirement.kind + "'");
    if (compute->attachedTo != selection.executor)
      return reject(DiagnosticCode::UnsupportedComputeFragment,
                    "compute node '" + recorded->second + "' is attached to '" +
                        compute->attachedTo +
                        "', not to the recorded executor '" +
                        selection.executor + "'");
  }

  // 6. Memory kinds and visibility. A requirement that named a port is checked
  // against the recorded port-to-memory association -- the kind-keyed map
  // cannot tell two same-kind requirements apart -- and a bare requirement
  // against the kind-keyed map, exactly as generation bound each.
  for (const KindRequirement &requirement : rule.kindRequirements) {
    if (requirement.role != "memory")
      continue;
    std::string boundMemory;
    if (requirement.port) {
      std::optional<PortRef> ref =
          portRefForRulePort(rule, node, *requirement.port);
      if (!ref)
        return reject(DiagnosticCode::NoMemoryRoute,
                      "rule '" + rule.id + "' requires a '" + requirement.kind +
                          "' memory on port '" + requirement.port->name +
                          "', which the operation does not expose");
      const PortMemoryBinding *found = nullptr;
      for (const PortMemoryBinding &binding : selection.portMemories)
        if (binding.port == *ref) {
          found = &binding;
          break;
        }
      if (!found)
        return reject(DiagnosticCode::NoMemoryRoute,
                      "rule '" + rule.id + "' requires a '" + requirement.kind +
                          "' memory on port '" + requirement.port->name +
                          "', which the mapping does not bind");
      boundMemory = found->memory;
    } else {
      auto bound = selection.memories.find(requirement.kind);
      if (bound == selection.memories.end())
        return reject(DiagnosticCode::NoMemoryRoute,
                      "rule '" + rule.id + "' requires a '" + requirement.kind +
                          "' memory, which the mapping does not bind");
      boundMemory = bound->second;
    }
    const mlir::llk::machine::MemoryNode *memory =
        machine.findMemory(boundMemory);
    if (!memory)
      return reject(DiagnosticCode::NoMemoryRoute,
                    "unknown memory '" + boundMemory + "'");
    if (memory->kind != requirement.kind)
      return reject(DiagnosticCode::NoMemoryRoute,
                    "memory '" + boundMemory + "' has kind '" + memory->kind +
                        "', not the required '" + requirement.kind + "'");
    if (!machine.isVisible(boundMemory, selection.executor))
      return reject(DiagnosticCode::NoMemoryRoute,
                    "executor '" + selection.executor +
                        "' cannot see memory '" + boundMemory + "'");
  }

  return llvm::Error::success();
}

std::optional<MappingCandidate> toFusedMappingCandidate(
    const RuleDef &rule, const RulePatternMatch &match,
    const WorkloadGraph &graph, const machine::MachineModel &machine,
    const LayoutContext &context, std::string *reason, bool *truncated,
    const llvm::StringMap<SearchValue> *pinned,
    const llvm::StringMap<std::string> *boundLayouts,
    const BoundAxes *boundAxes) {
  if (!rule.pattern || match.coveredNodes.size() != rule.pattern->nodes.size())
    return std::nullopt;

  // The rule's parameters and layout roles are solved against the **anchor**:
  // it is the operation the rule reasons about, and the matcher already checked
  // that every other matched node satisfies its own predicates and edges. What
  // the pattern adds is the coverage, not a second parameter space.
  const WorkloadNode *anchor = graph.findNode(match.coveredNodes.front());
  if (!anchor)
    return std::nullopt;
  std::optional<MappingCandidate> candidate =
      toMappingCandidate(rule, *anchor, machine, context, reason, truncated,
                         pinned, boundLayouts, boundAxes);
  if (!candidate)
    return std::nullopt;

  candidate->coveredNodes.assign(match.coveredNodes.begin(),
                                 match.coveredNodes.end());

  // The boundary replaces the anchor's own ports. A boundary occurrence is
  // named after the pattern node that owns it, so a rule that declares a
  // memory or layout requirement can name `cv.operand0` and mean exactly that
  // occurrence -- one node's `operand0` and another's are different ports, and
  // a name that could not tell them apart would let a requirement resolve to
  // the wrong one.
  candidate->ports.clear();
  for (const PortRef &ref : match.boundary) {
    const WorkloadNode *node = graph.findNode(ref.node);
    if (!node)
      return std::nullopt;

    size_t patternIndex = 0;
    for (size_t i = 0; i < match.coveredNodes.size(); ++i)
      if (match.coveredNodes[i] == ref.node)
        patternIndex = i;
    const std::string &nodeName = rule.pattern->nodes[patternIndex].name;

    PortSpec spec;
    spec.isInput = ref.direction == PortDirection::Input;
    spec.port = ref;
    if (spec.isInput) {
      if (ref.index >= node->inputs.size())
        return std::nullopt;
      spec.value = node->inputs[ref.index].value;
      spec.name = nodeName + ".operand" + std::to_string(ref.index);
    } else {
      if (ref.index >= node->outputs.size())
        return std::nullopt;
      spec.value = node->outputs[ref.index].value;
      spec.name = nodeName + ".result";
      if (ref.index != 0)
        spec.name += std::to_string(ref.index);
    }
    candidate->ports.push_back(std::move(spec));
  }

  // `toMappingCandidate` resolved graph-port layout requirements against the
  // anchor node before the pattern boundary was available. Rebind them to the
  // named boundary occurrence now, including that occurrence's own type and
  // shape; otherwise gate/result requirements inherit the anchor's port index
  // and strict verification tests the wrong endpoint.
  if (candidate->layoutRequirements.size() != rule.layoutRequirements.size()) {
    if (reason)
      *reason = "fused layout requirement count changed while resolving ports";
    return std::nullopt;
  }
  for (size_t index = 0; index < rule.layoutRequirements.size(); ++index) {
    const RuleLayoutRequirement &declared = rule.layoutRequirements[index];
    if (declared.port.empty())
      continue;
    auto port = llvm::find_if(candidate->ports, [&](const PortSpec &spec) {
      return spec.name == declared.port;
    });
    if (port == candidate->ports.end() || !port->port) {
      if (reason)
        *reason = "fused layout requirement names non-boundary port '" +
                  declared.port + "'";
      return std::nullopt;
    }
    const PortRef &ref = *port->port;
    const WorkloadNode *node = graph.findNode(ref.node);
    if (!node) {
      if (reason)
        *reason = "fused layout requirement names missing node " +
                  std::to_string(ref.node);
      return std::nullopt;
    }
    const WorkloadPort *workloadPort = nullptr;
    if (ref.direction == PortDirection::Input &&
        ref.index < node->inputs.size())
      workloadPort = &node->inputs[ref.index];
    else if (ref.direction == PortDirection::Output &&
             ref.index < node->outputs.size())
      workloadPort = &node->outputs[ref.index];
    if (!workloadPort) {
      if (reason)
        *reason = "fused layout requirement resolves to an invalid port '" +
                  declared.port + "'";
      return std::nullopt;
    }
    LayoutRequirement &resolved = candidate->layoutRequirements[index];
    resolved.port = ref;
    resolved.portValue = static_cast<int64_t>(workloadPort->value);
    if (mlir::Type element = elementTypeOf(workloadPort->type))
      resolved.elementType = printedType(element);
    if (std::optional<llvm::SmallVector<int64_t, 4>> shape =
            staticShapeOf(workloadPort->type))
      resolved.rank = static_cast<int64_t>(shape->size());
  }
  return candidate;
}

} // namespace mlir::llk::mapping
