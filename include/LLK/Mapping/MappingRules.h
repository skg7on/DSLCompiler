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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mlir::llk::mapping {

/// One `attribute = value` predicate on the matched operation.
struct RulePredicate {
  std::string attribute;
  LayoutValue value;
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
struct LayoutRequirement {
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
  std::vector<LayoutRequirement> layoutRequirements;
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

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_MAPPINGRULES_H
