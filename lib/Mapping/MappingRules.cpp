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

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
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

/// A `!micro.tile<shape x element, ...>` read through its printed form. This
/// core deliberately does not link the Micro dialect (see WorkloadGraph.h), so
/// the tile is re-parsed as the `tensor<shape x element>` its head spells. Any
/// other type yields a null type.
mlir::Type tileAsTensor(mlir::Type type) {
  if (!type)
    return {};
  std::string printed = printedType(type);
  llvm::StringRef text(printed);
  if (!text.consume_front("!micro.tile<"))
    return {};
  size_t end = text.find_first_of(",>");
  if (end == llvm::StringRef::npos)
    return {};
  std::string wrapped = ("tensor<" + text.take_front(end) + ">").str();
  return mlir::parseType(wrapped, type.getContext());
}

/// The element type a port type exposes, or a null type when this core cannot
/// read one. A modelled shaped type and a bare float, integer, or index type
/// state theirs; a `!micro.tile` is unwrapped through its printed form.
mlir::Type elementTypeOf(mlir::Type type) {
  if (!type)
    return {};
  if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type))
    return shaped.getElementType();
  if (mlir::isa<mlir::FloatType, mlir::IntegerType, mlir::IndexType>(type))
    return type;
  if (mlir::Type tile = tileAsTensor(type))
    if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(tile))
      return shaped.getElementType();
  return {};
}

/// The static shape a port type exposes, or nullopt for a dynamic, unranked, or
/// opaque type. A `!micro.tile` is unwrapped through its printed form.
std::optional<llvm::SmallVector<int64_t, 4>> shapeOf(mlir::Type type) {
  if (!type)
    return std::nullopt;
  mlir::Type candidate = type;
  if (!mlir::isa<mlir::ShapedType>(candidate))
    candidate = tileAsTensor(type);
  if (!candidate)
    return std::nullopt;
  if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(candidate))
    if (shaped.hasStaticShape())
      return llvm::SmallVector<int64_t, 4>(shaped.getShape());
  return std::nullopt;
}

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
      std::optional<llvm::SmallVector<int64_t, 4>> shape = shapeOf(port->type);
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
            shapeOf(port.type)) {
      context.rank = static_cast<int64_t>(shape->size());
      haveRank = true;
      break;
    }
  }
  if (!haveRank)
    for (const WorkloadPort &port : node.inputs) {
      if (std::optional<llvm::SmallVector<int64_t, 4>> shape =
              shapeOf(port.type)) {
        context.rank = static_cast<int64_t>(shape->size());
        break;
      }
    }
  return context;
}

/// Adds every identifier an expression references to `out`, so a rule can tell
/// which of its parameters a constraint actually derives.
void collectIdentifiers(const Expr &expr, llvm::StringSet<> &out) {
  if (expr.kind == ExprKind::Ident)
    out.insert(expr.text);
  for (const ExprPtr &operand : expr.operands)
    collectIdentifiers(*operand, out);
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
/// Only parameters a constraint references are recorded; a declared but
/// unconstrained parameter is not derived, so it never lands in the result.
/// Hitting the assignment cap sets `truncated`: a rule whose space was not
/// exhausted was not proven unsatisfiable.
RuleResolution resolveRuleConstraints(const RuleDef &rule,
                                      const WorkloadNode &node,
                                      const machine::MachineModel &machine,
                                      const LayoutContext &context) {
  RuleResolution result;
  if (rule.constraints.empty())
    return result;

  // Every declared parameter must have a finite, non-empty domain to enumerate.
  llvm::SmallVector<const LayoutParam *, 4> enumerable;
  for (const LayoutParam &param : rule.params) {
    auto domain = rule.domains.find(param.name);
    if (domain == rule.domains.end() || domain->second.values.empty()) {
      result.matched = false;
      result.reason = "parameter '" + param.name + "' has no declared domain";
      return result;
    }
    enumerable.push_back(&param);
  }

  // The parameters a constraint actually derives: only these are recorded.
  llvm::StringSet<> referenced;
  for (const ExprPtr &constraint : rule.constraints)
    collectIdentifiers(*constraint, referenced);

  auto domainOf = [&](const LayoutParam &param) -> const ParamDomain & {
    return rule.domains.find(param.name)->second;
  };
  LayoutContext ruleContext = ruleLayoutContext(node, context);

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
      bindings[enumerable[i]->name] = domainOf(*enumerable[i]).values[index[i]];
    ++assignments;

    bool satisfied = true;
    for (const ExprPtr &constraint : rule.constraints) {
      llvm::Expected<EvalValue> value =
          evaluateExpr(*constraint, bindings, machine, ruleContext);
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
      for (const auto &entry : bindings)
        if (referenced.contains(entry.first()))
          result.parameters[entry.first().str()] = entry.second;
      return result;
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

  result.matched = false;
  result.reason =
      failure.value_or("no assignment satisfies the rule's constraints");
  return result;
}

/// An MLIR context to type a rule's bundle parameters with, taken from the node
/// they matched. A rule that declares parameters only ever matches a node built
/// from real IR, so a context is available; the null return is the belt-and-
/// braces case of a synthetic node with no attributes and no port types.
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
mlir::DictionaryAttr buildBundleParameters(
    mlir::MLIRContext *context,
    const std::vector<std::pair<std::string, LayoutValue>> &parameters) {
  if (parameters.empty() || !context)
    return {};
  llvm::SmallVector<mlir::NamedAttribute> attributes;
  attributes.reserve(parameters.size());
  for (const auto &entry : parameters) {
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
                   bool *truncated) {
  RuleResolution resolution =
      resolveRuleConstraints(rule, node, machine, context);
  if (!resolution.matched) {
    if (reason)
      *reason = resolution.reason.empty()
                    ? std::string("require constraints not satisfied")
                    : resolution.reason;
    if (truncated)
      *truncated = resolution.truncated;
    return std::nullopt;
  }

  MappingCandidate candidate;
  candidate.rule = rule.id;
  candidate.coveredNodes.push_back(node.id);
  candidate.bundle.name = rule.bundle;
  candidate.bundle.emitterKey = rule.emitter;
  candidate.bundle.parameters =
      buildBundleParameters(nodeContext(node), rule.bundleParameters);
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
    LayoutRequirement resolved;
    resolved.layoutClass = requirement.layoutId;
    candidate.layoutRequirements.push_back(std::move(resolved));
  }

  candidate.id = computeCandidateId(candidate);
  return candidate;
}

} // namespace mlir::llk::mapping
