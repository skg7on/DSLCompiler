//===- MappingRules.h - LLKMap mapping rules (D4) -------------------------===//
//
// Part of the target-independent mapping core (epic #67, workstream D4).
//
// A mapping rule says: when the workload contains this Micro operation, with
// these attribute predicates, and the machine offers this capability, then
// this target bundle implements it. Rules cover exactly one operation
// (design §14.2); fusion is a later, explicitly-listed extension.
//
// A rule never carries target semantics. `bundle` and `emit` are opaque names
// that generic code compares, hashes, and reports -- it does not know what
// `avx2_vector_add` does (design §14.3). Layout ids are referenced, never
// redefined: the layout registry (D3) owns them.
//
// Parsing reuses the shared LLKMap expression grammar, so a rule's `param`,
// `require`, and machine queries behave exactly like a layout's.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_MAPPINGRULES_H
#define LLK_MAPPING_MAPPINGRULES_H

#include "LLK/Mapping/LayoutConstraints.h"
#include "LLK/Mapping/LlkMap.h"
#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/WorkloadGraph.h"

#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mlir::llk::mapping {

/// What a rule predicate constrains.
enum class RulePredicateKind {
  /// `name = value`: an operation attribute.
  Attribute,
  /// `[input[i].]element_type = f32`.
  ElementType,
  /// `[output[i].]shape[d] = 64`.
  Shape,
  /// `[input[i].]access_map = (d0, d1) -> (d1, d0)`.
  AccessMap,
};

/// One predicate on the matched operation: an attribute equality, or a property
/// of one of the operation's boundary ports.
///
/// A port predicate is either qualified (`input[i].`/`output[i].`, naming one
/// port) or unqualified. An unqualified predicate applies to its property's
/// default direction -- element type and access map read inputs, shape reads
/// outputs -- and holds when at least one port in that direction exposes the
/// property and every port that exposes it agrees.
///
/// Port predicates read real port data (`WorkloadPort::type` and `accessMap`).
/// They are deliberately conservative: a port that does not expose a property
/// (an opaque type, a dynamic shape, an absent affine map) never satisfies it,
/// and a predicate with no exposing port does not match.
struct RulePredicate {
  RulePredicateKind kind = RulePredicateKind::Attribute;
  /// Attribute name (`Attribute`) or property name (`element_type`, `shape`,
  /// `access_map`).
  std::string attribute;
  /// True when a port subject (`input[i]`/`output[i]`) was written.
  bool directionSet = false;
  /// The subject port's direction, when `directionSet`.
  bool isInput = true;
  /// The subject port index, when `directionSet`.
  int64_t portIndex = 0;
  /// The shape dimension, for `Shape`.
  int64_t dimension = 0;
  /// The attribute, element-type, or shape value.
  LayoutValue value;
  /// The declared map, for `AccessMap`.
  std::optional<AffineMapSpec> accessMap;
};

/// A named boundary value of the matched operation.
struct RulePort {
  std::string name;
  bool isInput = true;
};

/// An abstract capability requirement: role is `executor`, `compute`, or
/// `memory`, and `kind` is the abstract kind a placement must satisfy.
struct KindRequirement {
  std::string role;
  std::string kind;
};

/// A layout the value on `port` must satisfy, named by layout id.
struct RuleLayoutRequirement {
  std::string port;
  std::string layoutId;
};

struct RuleDef {
  std::string id;
  uint64_t version = 1;
  /// The Micro operation this rule implements, e.g. `micro.vector`.
  std::string matchOp;
  std::vector<RulePredicate> predicates;
  std::vector<LayoutParam> params;
  std::map<std::string, ParamDomain> domains;
  std::vector<ExprPtr> constraints;
  std::vector<KindRequirement> kindRequirements;
  std::vector<RuleLayoutRequirement> layoutRequirements;
  std::vector<RulePort> ports;
  /// Opaque target-owned names; generic code never interprets them.
  std::string bundle;
  std::string emitter;
  /// Static cost lower bound, when the rule declares one.
  std::optional<uint64_t> costLowerBound;

  const LayoutParam *findParam(llvm::StringRef name) const;
};

/// Loaded rules, keyed by id.
class RuleRegistry {
public:
  /// Adds `def`, or fails with a stable message when the id is a duplicate.
  bool add(RuleDef def, std::string &error);

  const RuleDef *find(llvm::StringRef id) const;
  llvm::ArrayRef<RuleDef> all() const { return defs_; }

private:
  std::vector<RuleDef> defs_;
};

/// Parses a complete rule file. `sourceName` appears in diagnostics.
llvm::Expected<RuleRegistry> parseRuleText(llvm::StringRef text,
                                           llvm::StringRef sourceName);

/// Reads and parses the file at `path`.
llvm::Expected<RuleRegistry> loadRuleFile(llvm::StringRef path);

//===----------------------------------------------------------------------===//
// One-operation matching (design §14.2)
//===----------------------------------------------------------------------===//

/// True when `predicate` holds against `node`. An attribute predicate needs an
/// attribute of the same name and value; a port predicate reads the node's port
/// types and access maps. Missing data never matches, so every predicate is
/// conservative.
bool predicateMatches(const RulePredicate &predicate, const WorkloadNode &node);

/// Rules whose match operation and predicates apply to `node`, in registry
/// order. A rule with no predicates matches every operation of its name.
std::vector<const RuleDef *> matchRules(const WorkloadNode &node,
                                        const RuleRegistry &rules);

/// Bridges a matched rule onto the workload node it covers: the result is the
/// unplaced `MappingCandidate` the placement engine consumes. Rule ports are
/// wired positionally to the node's inputs and then its outputs.
///
/// The rule's `require` constraints are evaluated here, against `machine` and
/// the rank and element type the node exposes (falling back to `context` for
/// facts the node does not carry). No assignment of the rule's declared
/// parameters satisfies a constraint, or a constraint cannot be evaluated from
/// the available facts, yields `std::nullopt` -- a non-match, never an error.
/// On a match, the satisfying assignment is recorded in the candidate's
/// `resolvedParameters`. When `reason` is given it receives why the rule did
/// not match.
std::optional<MappingCandidate>
toMappingCandidate(const RuleDef &rule, const WorkloadNode &node,
                   const machine::MachineModel &machine,
                   const LayoutContext &context, std::string *reason = nullptr);

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_MAPPINGRULES_H
