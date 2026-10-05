//===- PlanReport.cpp - Versioned JSON plan report (design §22.2) ---------===//
//
// The report is written with `llvm::json::OStream`'s imperative API rather than
// from a `json::Object`: an Object is backed by a `DenseMap`, whose iteration
// order is not a contract, whereas OStream emits attributes in exactly the
// order the code writes them. That gives the fixed key order and the
// byte-for-byte determinism the report needs (§29.12).
//
//===----------------------------------------------------------------------===//

#include "LLK/Mapping/PlanReport.h"

#include "LLK/Mapping/CostModel.h"
#include "LLK/Mapping/Diagnostics.h"
#include "LLK/Mapping/LatencyProvider.h"
#include "LLK/Mapping/MappingMetadata.h"
#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/StableHash.h"
#include "LLK/Version.h"

#include "mlir/AsmParser/AsmParser.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace mlir::llk::mapping {

namespace {

/// The stable spelling of a search mode. Kept local: `SearchMode` has no
/// stringifier because nothing else needs to print it.
llvm::StringRef stringifySearchMode(SearchMode mode) {
  switch (mode) {
  case SearchMode::Deterministic:
    return "deterministic";
  case SearchMode::Beam:
    return "beam";
  case SearchMode::Exact:
    return "exact";
  }
  return "unknown";
}

/// True when `code` names a *rejection* -- something the search refused -- as
/// opposed to a notice. §22.2 asks for "rejected counts", so a notice never
/// inflates the rejection tally.
///
/// Every code is classified explicitly and there is deliberately no `default`:
/// a code added to the enum without a case here makes this switch incomplete
/// (`-Wswitch`), rather than silently defaulting into the rejection bucket.
/// `AssumedValueSize` is the case that motivated it -- its own documentation
/// says it is not an error, so it must land under notices.
bool isRejection(DiagnosticCode code) {
  switch (code) {
  // Notices: a cap, a provider gap, or an advisory assumption. None is a
  // refusal the search made.
  case DiagnosticCode::SearchTruncated:
  case DiagnosticCode::LatencyCacheMiss:
  case DiagnosticCode::AssumedValueSize:
  case DiagnosticCode::ConnectionChoiceUnexplored:
    return false;
  // Rejections: the search refused a rule, a placement, a pair, a layout, a
  // global constraint, a bundle, or a plan.
  case DiagnosticCode::NoMatchingRule:
  case DiagnosticCode::NoLegalLayout:
  case DiagnosticCode::NoLegalExecutor:
  case DiagnosticCode::MemoryCapacityExceeded:
  case DiagnosticCode::UnsupportedComputeFragment:
  case DiagnosticCode::NoMemoryRoute:
  case DiagnosticCode::NoLayoutTransform:
  case DiagnosticCode::GlobalConstraintFailed:
  case DiagnosticCode::TargetBundleInvalid:
  case DiagnosticCode::InvalidMappingMetadata:
  case DiagnosticCode::InvalidGatherDeclaration:
    return true;
  }
  llvm_unreachable("unclassified DiagnosticCode");
}

/// Fixed six-decimal rendering of a double, matching `canonicalCostString`, so
/// a cost component is byte-stable and never printed in exponent form.
/// `snprintf`'s `%f` honours the C locale's decimal separator; the report
/// assumes `LC_NUMERIC=C`, which is the process default and what the rest of
/// the toolchain (including `canonicalCostString`) already relies on.
std::string fixedDouble(double value) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.6f", value);
  return std::string(buffer);
}

/// The printed form of an affine map, for the report's replay state.
std::string printedMapString(mlir::AffineMap map) {
  if (!map)
    return {};
  std::string text;
  llvm::raw_string_ostream stream(text);
  map.print(stream);
  return stream.str();
}

} // namespace

llvm::StringRef compilerVersion() { return LLK_COMPILER_VERSION; }

std::string writePlanReport(const MappingSearchResult &result,
                            const machine::MachineModel &machine,
                            const MappingTarget &target,
                            const MappingSearchOptions &options,
                            uint64_t moduleHash) {
  const CoveringPlan *selected =
      result.plans.empty() ? nullptr : &result.plans.front();

  std::string text;
  llvm::raw_string_ostream stream(text);
  llvm::json::OStream json(stream, /*IndentSize=*/2);

  json.object([&] {
    // --- provenance (fixed key order) -----------------------------------
    json.attribute("version", kPlanReportVersion);
    json.attribute("compilerVersion", compilerVersion());
    json.attribute("costModelVersion", kCostModelVersion);
    json.attribute("target", target.name());
    json.attribute("inputModuleHash", hexId(moduleHash));
    json.attribute("sourceBindingHash",
                   hexId(selected ? selected->sourceBindingHash : 0));
    // The `micro.candidate` symbol the binding was loaded from, so a consumer
    // holding only this report can name the exact `candidate=` a replay must
    // pass: the hash above covers the values alone, so it cannot distinguish
    // two candidates that bind identically. Empty for a binding-free search,
    // where the hash is zero too.
    json.attribute("sourceBindingCandidate",
                   selected ? selected->sourceBindingCandidate : std::string());
    // The source search binding's sorted, type-tagged parameter values -- the
    // point the candidate symbol names, rendered so it is readable without the
    // module. Empty for a binding-free search.
    json.attribute("sourceBinding", selected ? canonicalSearchValueString(
                                                   selected->globalParameters)
                                             : std::string());
    // Computed rather than read off the model: an in-memory model has no
    // cached hash, and the report must never print a stale or zero one.
    json.attribute("machineHash", hexId(machine::computeContentHash(machine)));
    json.attribute("layoutLibraryHash",
                   hexId(target.layouts().computeContentHash()));
    json.attribute("ruleLibraryHash",
                   hexId(target.rules().computeContentHash()));
    // The target's folded content identity (name + machine + layout + rule
    // hashes). A replay must match it, so a report cannot be replayed against a
    // target the plan was not bound for.
    json.attribute("targetHash", hexId(computeTargetContentHash(target)));
    // The canonical, pre-materialization source-graph hash the search ran over.
    // A replay must match the graph it is handed, so a report cannot be
    // replayed against a semantically-changed workload.
    json.attribute("graphHash", hexId(result.workloadHash));

    // --- options --------------------------------------------------------
    json.attributeObject("searchOptions", [&] {
      json.attribute("mode", stringifySearchMode(options.mode));
      json.attribute("beamWidth", static_cast<uint64_t>(options.beamWidth));
      json.attribute("topK", static_cast<uint64_t>(options.topK));
      json.attribute("maxCandidatesPerNode",
                     static_cast<uint64_t>(options.maxCandidatesPerNode));
      json.attribute("maxInstancesPerCandidate",
                     static_cast<uint64_t>(options.maxInstancesPerCandidate));
      json.attribute("maxRoutesPerConnection",
                     static_cast<uint64_t>(options.maxRoutesPerConnection));
      json.attribute("maxConnectionCombinations",
                     static_cast<uint64_t>(options.maxConnectionCombinations));
      json.attribute("memoryBudgetBytes", options.memoryBudgetBytes);
      json.attribute("enableLatencyCache", options.enableLatencyCache);
      json.attribute("enableSymmetryReduction",
                     options.enableSymmetryReduction);
      json.attributeObject("objective", [&] {
        json.attribute("primary",
                       stringifyCostMetric(options.objective.primary));
        json.attribute("minimize", options.objective.minimize);
        json.attributeArray("secondary", [&] {
          for (CostMetric metric : options.objective.secondary)
            json.value(stringifyCostMetric(metric));
        });
      });
    });

    json.attribute("searchTruncated", result.searchTruncated);

    // --- counts ---------------------------------------------------------
    json.attributeObject("counts", [&] {
      json.attribute("candidates", result.candidateCount);
      json.attribute("instances", result.instanceCount);
      json.attribute("routes", result.routeCount);
      json.attribute("plans", result.planCount);
    });

    // --- coded events grouped by stable code ----------------------------
    // `codeCounts` is a `std::map`, so iteration is by code and the grouping is
    // deterministic. §22.2's "rejected counts" are separated from notices: a
    // cap hit or a cache miss is not a rejection, so it never inflates the
    // rejection tally.
    json.attributeArray("rejections", [&] {
      for (const auto &entry : result.frontier.codeCounts) {
        if (!isRejection(entry.first))
          continue;
        json.object([&] {
          json.attribute("code", stringifyDiagnosticCode(entry.first));
          json.attribute("count", entry.second);
        });
      }
    });
    json.attributeArray("notices", [&] {
      for (const auto &entry : result.frontier.codeCounts) {
        if (isRejection(entry.first))
          continue;
        json.object([&] {
          json.attribute("code", stringifyDiagnosticCode(entry.first));
          json.attribute("count", entry.second);
        });
      }
    });

    // --- top-K plans with component costs -------------------------------
    json.attributeArray("plans", [&] {
      for (size_t rank = 0; rank < result.plans.size(); ++rank) {
        const CoveringPlan &plan = result.plans[rank];
        json.object([&] {
          // The plan's content id. It is a snapshot for correlation, *not*
          // something a reader can recompute from this document: the id folds
          // content the summary below does not emit -- since phase-3 T4 the
          // solved layout parameterization of every instance and placement
          // (`CoveringPlan` -> `PlanPlacement::layoutSolutions`), as well as
          // the placements' layouts -- so an id cannot be re-derived from the
          // report's content. (Re-emitting the same search's report does carry
          // every id through verbatim; it just cannot be recomputed from what
          // the document shows.)
          json.attribute("id", hexId(plan.id));
          json.attribute("rank", static_cast<uint64_t>(rank));
          json.attribute("sourceBindingHash", hexId(plan.sourceBindingHash));
          json.attribute("instanceCount",
                         static_cast<uint64_t>(plan.instances.size()));
          json.attribute("placementCount",
                         static_cast<uint64_t>(plan.placements.size()));
          json.attribute("routeCount",
                         static_cast<uint64_t>(plan.connectionPlans.size()));
          // The canonical string is the whole cost vector; the broken-out
          // object is the same numbers per dimension, byte-stable.
          json.attribute("totalCost", canonicalCostString(plan.totalCost));
          json.attributeObject("costComponents", [&] {
            json.attribute("latencyCycles",
                           fixedDouble(plan.totalCost.latencyCycles));
            json.attribute("dramBytes", plan.totalCost.dramBytes);
            json.attribute("localBytes", plan.totalCost.localBytes);
            json.attribute("spillBytes", plan.totalCost.spillBytes);
            json.attribute("computeUtilization",
                           fixedDouble(plan.totalCost.computeUtilization));
            json.attribute("transferUtilization",
                           fixedDouble(plan.totalCost.transferUtilization));
          });
          json.attributeObject("diagnostics", [&] {
            json.attribute("searchTruncated", plan.diagnostics.searchTruncated);
            // `errors`/`warnings` are never written by the search, so these
            // counts are always 0 today; they are emitted so the report shape
            // is stable for a future producer rather than silently omitted.
            json.attribute("errorCount", static_cast<uint64_t>(
                                             plan.diagnostics.errors.size()));
            json.attribute(
                "warningCount",
                static_cast<uint64_t>(plan.diagnostics.warnings.size()));
            // Storage finalization's informational notes: occupancy and
            // analysis-mode reports. They are deliberately separate from the
            // identity-bearing warnings, so a staged note never changes a plan
            // id.
            json.attributeArray("storageNotes", [&] {
              for (const std::string &note : plan.diagnostics.storageNotes)
                json.value(note);
            });
          });
        });
      }
    });

    // --- selected plan state, for versioned replay ----------------------
    // The exact execution-affecting selection: endpoints, routes, resolved
    // layout parameters and concrete maps, resource bindings. `readPlanReport`
    // reconstructs it, so a report alone (plus the target and source graph) can
    // replay the selected plan's data.
    json.attributeObject("selectedState", [&] {
      json.attribute("id", hexId(selected ? selected->id : 0));
      json.attribute("sourceBindingHash",
                     hexId(selected ? selected->sourceBindingHash : 0));
      json.attribute("materialized", selected ? selected->materialized : false);
      json.attributeArray("placements", [&] {
        if (!selected)
          return;
        for (const PlanPlacement &placement : selected->placements) {
          json.object([&] {
            json.attribute("node", static_cast<uint64_t>(placement.node));
            json.attribute("instance", hexId(placement.instance));
            json.attribute("rule", placement.rule);
            json.attribute("bundle", placement.bundle.name);
            json.attribute("emitter", placement.bundle.emitterKey);
            json.attribute("executor", placement.executor);
            // `StringMap` iteration order is not a contract; sort the keys so
            // the report stays byte-identical across runs (§29.12).
            json.attributeObject("memories", [&] {
              std::vector<std::string> keys;
              for (const auto &entry : placement.memories)
                keys.push_back(entry.first().str());
              llvm::sort(keys);
              for (const std::string &key : keys)
                json.attribute(key, placement.memories.lookup(key));
            });
            // The port-to-memory association of every named-port requirement,
            // sorted by occurrence so the report stays byte-identical.
            json.attributeArray("portMemories", [&] {
              std::vector<PortMemoryBinding> sorted =
                  placement.portMemoryBindings;
              llvm::sort(sorted, [](const PortMemoryBinding &lhs,
                                    const PortMemoryBinding &rhs) {
                if (lhs.port.node != rhs.port.node)
                  return lhs.port.node < rhs.port.node;
                if (lhs.port.direction != rhs.port.direction)
                  return lhs.port.direction < rhs.port.direction;
                return lhs.port.index < rhs.port.index;
              });
              for (const PortMemoryBinding &binding : sorted)
                json.object([&] {
                  json.attributeObject("port", [&] {
                    json.attribute("node",
                                   static_cast<uint64_t>(binding.port.node));
                    json.attribute("direction", binding.port.direction ==
                                                        PortDirection::Input
                                                    ? "input"
                                                    : "output");
                    json.attribute("index",
                                   static_cast<uint64_t>(binding.port.index));
                  });
                  json.attribute("memory", binding.memory);
                });
            });
            json.attributeObject("layouts", [&] {
              std::vector<std::string> keys;
              for (const auto &entry : placement.layouts)
                keys.push_back(entry.first().str());
              llvm::sort(keys);
              for (const std::string &key : keys)
                json.attribute(key, placement.layouts.lookup(key));
            });
            json.attributeArray("solutions", [&] {
              std::vector<std::string> keys;
              for (const auto &entry : placement.layoutSolutions)
                keys.push_back(entry.first().str());
              llvm::sort(keys);
              for (const std::string &key : keys) {
                const SolvedLayout &solved =
                    placement.layoutSolutions.lookup(key);
                json.object([&] {
                  json.attribute("key", key);
                  json.attribute("class", solved.layoutClass);
                  json.attribute("family",
                                 placement.layouts.lookup(solved.layoutClass));
                  if (solved.port)
                    json.attributeObject("port", [&] {
                      json.attribute("node",
                                     static_cast<uint64_t>(solved.port->node));
                      json.attribute("direction", solved.port->direction ==
                                                          PortDirection::Input
                                                      ? "input"
                                                      : "output");
                      json.attribute("index",
                                     static_cast<uint64_t>(solved.port->index));
                    });
                  json.attributeObject("parameters", [&] {
                    std::vector<std::string> names;
                    for (const auto &parameter : solved.parameters)
                      names.push_back(parameter.first().str());
                    llvm::sort(names);
                    for (const std::string &parameter : names) {
                      const SearchValue &bound =
                          solved.parameters.lookup(parameter);
                      if (const int64_t *integer = std::get_if<int64_t>(&bound))
                        json.attribute(parameter,
                                       static_cast<int64_t>(*integer));
                      else
                        json.attribute(parameter, std::get<std::string>(bound));
                    }
                  });
                  if (solved.map)
                    json.attribute("map", printedMapString(solved.map));
                });
              }
            });
            // The resolved rule parameters generation solved, so a replay
            // validates the recorded assignment rather than re-deriving one.
            json.attributeObject("ruleParameters", [&] {
              std::vector<std::string> names;
              for (const auto &parameter : placement.resolvedParameters)
                names.push_back(parameter.first().str());
              llvm::sort(names);
              for (const std::string &name : names) {
                const SearchValue &bound =
                    placement.resolvedParameters.lookup(name);
                if (const int64_t *integer = std::get_if<int64_t>(&bound))
                  json.attribute(name, static_cast<int64_t>(*integer));
                else
                  json.attribute(name, std::get<std::string>(bound));
              }
            });
          });
        }
      });
      json.attributeArray("connections", [&] {
        if (!selected)
          return;
        for (const PlanConnection &connection : selected->connectionPlans) {
          json.object([&] {
            json.attribute("id", hexId(connection.id));
            json.attribute("value", static_cast<uint64_t>(connection.value));
            json.attribute("kind",
                           stringifyConnectionKind(connection.kind).str());
            json.attributeArray("route", [&] {
              for (const MemoryNodeId &node : connection.route)
                json.value(node);
            });
            json.attributeArray("engines", [&] {
              for (const ExecutorId &engine : connection.engines)
                json.value(engine);
            });
            json.attributeArray("consumers", [&] {
              for (InstanceId consumer : connection.consumers)
                json.value(hexId(consumer));
            });
            json.attributeArray("storageIds", [&] {
              for (uint64_t id : connection.storageIds)
                json.value(hexId(id));
            });
            if (connection.producerPort)
              json.attributeObject("producerPort", [&] {
                json.attribute("node", static_cast<uint64_t>(
                                           connection.producerPort->node));
                json.attribute("direction",
                               connection.producerPort->direction ==
                                       PortDirection::Input
                                   ? "input"
                                   : "output");
                json.attribute("index", static_cast<uint64_t>(
                                            connection.producerPort->index));
              });
            json.attributeArray("consumerPorts", [&] {
              for (const PortRef &port : connection.consumerPorts)
                json.object([&] {
                  json.attribute("node", static_cast<uint64_t>(port.node));
                  json.attribute("direction",
                                 port.direction == PortDirection::Input
                                     ? "input"
                                     : "output");
                  json.attribute("index", static_cast<uint64_t>(port.index));
                });
            });
            // The explicit combination a reduce performs and the producer
            // occurrences it combines (task B6/B7), so a report states what a
            // gather *means* rather than only that it combines producers.
            if (connection.gatherSemantics)
              json.attribute(
                  "gatherSemantics",
                  stringifyGatherSemantics(*connection.gatherSemantics));
            if (connection.concatAxis)
              json.attribute("concatAxis",
                             static_cast<uint64_t>(*connection.concatAxis));
            if (!connection.producerPorts.empty())
              json.attributeArray("producerPorts", [&] {
                for (const PortRef &port : connection.producerPorts)
                  json.object([&] {
                    json.attribute("node", static_cast<uint64_t>(port.node));
                    json.attribute("direction",
                                   port.direction == PortDirection::Input
                                       ? "input"
                                       : "output");
                    json.attribute("index", static_cast<uint64_t>(port.index));
                  });
              });
            if (connection.transform) {
              json.attributeObject("transform", [&] {
                json.attribute("src", connection.transform->srcLayout);
                json.attribute("dst", connection.transform->dstLayout);
                if (connection.transform->srcMap)
                  json.attribute(
                      "srcMap", printedMapString(connection.transform->srcMap));
                if (connection.transform->dstMap)
                  json.attribute(
                      "dstMap", printedMapString(connection.transform->dstMap));
              });
            }
          });
        }
      });
      // Storage and synchronization state (design §9.6). B1 persists it; B3
      // populates it, so an empty array is legal today.
      json.attributeArray("allocations", [&] {
        if (!selected)
          return;
        for (const StorageAllocation &allocation : selected->allocations) {
          json.object([&] {
            json.attribute("id", hexId(allocation.id));
            json.attribute("value", static_cast<uint64_t>(allocation.value));
            json.attribute("memory", allocation.memory);
            json.attribute("bytes", allocation.bytes);
            if (allocation.aliasOf)
              json.attribute("aliasOf", hexId(*allocation.aliasOf));
            json.attribute("beginStep", allocation.beginStep);
            json.attribute("endStep", allocation.endStep);
          });
        }
      });
      json.attributeArray("synchronization", [&] {
        if (!selected)
          return;
        for (const SynchronizationStep &step : selected->synchronization) {
          json.object([&] {
            json.attribute("id", step.id);
            json.attributeArray("waitsFor", [&] {
              for (ConnectionId connection : step.waitsFor)
                json.value(hexId(connection));
            });
            json.attributeArray("precedes", [&] {
              for (const PortRef &port : step.precedes)
                json.object([&] {
                  json.attribute("node", static_cast<uint64_t>(port.node));
                  json.attribute("direction",
                                 port.direction == PortDirection::Input
                                     ? "input"
                                     : "output");
                  json.attribute("index", static_cast<uint64_t>(port.index));
                });
            });
            json.attribute("requiresBarrier", step.requiresBarrier);
          });
        }
      });
      // The plan-step DAG (design §9.6): every step and the dependency edges
      // between them, so B4-B6 can consume a replayed plan's ordering.
      json.attributeArray("steps", [&] {
        if (!selected)
          return;
        for (const PlanStep &step : selected->steps) {
          json.object([&] {
            json.attribute("id", step.id);
            json.attribute("kind", stringifyPlanStepKind(step.kind));
            json.attribute("node", static_cast<uint64_t>(step.node));
            json.attribute("connection", hexId(step.connection));
          });
        }
      });
      json.attributeArray("stepEdges", [&] {
        if (!selected)
          return;
        for (const PlanStepEdge &edge : selected->stepEdges) {
          json.object([&] {
            json.attribute("from", edge.from);
            json.attribute("to", edge.to);
          });
        }
      });
    });

    json.attribute("selectedPlanId", hexId(selected ? selected->id : 0));
  });

  stream.flush();
  return text;
}

llvm::Error writePlanReportFile(llvm::StringRef path,
                                const MappingSearchResult &result,
                                const machine::MachineModel &machine,
                                const MappingTarget &target,
                                const MappingSearchOptions &options,
                                uint64_t moduleHash) {
  std::string report =
      writePlanReport(result, machine, target, options, moduleHash);
  std::error_code error;
  llvm::raw_fd_ostream out(path, error);
  if (error)
    return llvm::createStringError(
        error, llvm::Twine("cannot write plan report to '") + path + "'");
  out << report;
  out.flush();
  if (out.has_error())
    return llvm::createStringError(
        out.error(), llvm::Twine("cannot write plan report to '") + path + "'");
  return llvm::Error::success();
}

namespace {

llvm::Error reportError(std::string message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 std::move(message));
}

/// Parses the fixed-width lowercase hexadecimal id `hexId` writes. A non-hex
/// string yields 0, which then fails the identity checks that follow.
uint64_t parseHexId(llvm::StringRef text) {
  return std::strtoull(text.str().c_str(), nullptr, 16);
}

/// Reads an endpoint occurrence from a report object with checked types.
llvm::Expected<PortRef> jsonPortRef(const llvm::json::Object *object,
                                    llvm::StringRef where) {
  if (!object)
    return reportError(where.str() + ": expected a port object");
  PortRef ref;
  std::optional<int64_t> node = object->getInteger("node");
  std::optional<int64_t> index = object->getInteger("index");
  std::optional<llvm::StringRef> direction = object->getString("direction");
  if (!node || !index || !direction)
    return reportError(where.str() + ": malformed port reference");
  if (*node < 0 || *index < 0)
    return reportError(where.str() + ": negative port reference");
  ref.node = static_cast<WorkloadNodeId>(*node);
  ref.index = static_cast<uint32_t>(*index);
  if (*direction == "input")
    ref.direction = PortDirection::Input;
  else if (*direction == "output")
    ref.direction = PortDirection::Output;
  else
    return reportError(where.str() + ": unknown port direction");
  return ref;
}

} // namespace

llvm::Expected<CoveringPlan> readPlanReport(llvm::StringRef json,
                                            const MappingTarget &target,
                                            const WorkloadGraph &graph) {
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(json);
  if (!parsed)
    return parsed.takeError();
  const llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return reportError("plan report is not a JSON object");
  std::optional<int64_t> version = root->getInteger("version");
  if (!version || *version != static_cast<int64_t>(kPlanReportVersion))
    return reportError("plan report has an unsupported version");
  std::optional<llvm::StringRef> targetHash = root->getString("targetHash");
  if (!targetHash || *targetHash != hexId(computeTargetContentHash(target)))
    return reportError(
        "plan report target_hash does not match the target; the plan was not "
        "bound for it");
  // The report records the canonical source-graph hash, so a semantic change to
  // the input graph is rejected before any executable binding -- not merely a
  // structural "the placement node still resolves" check.
  std::optional<llvm::StringRef> graphHash = root->getString("graphHash");
  if (!graphHash || *graphHash != hexId(computeSourceGraphHash(graph)))
    return reportError(
        "plan report graph_hash does not match the supplied source graph");
  const llvm::json::Object *state = root->getObject("selectedState");
  if (!state)
    return reportError("plan report has no selectedState");

  CoveringPlan plan;
  plan.schemaVersion = 2;
  if (std::optional<llvm::StringRef> id = state->getString("id"))
    plan.id = parseHexId(*id);
  if (std::optional<llvm::StringRef> binding =
          state->getString("sourceBindingHash"))
    plan.sourceBindingHash = parseHexId(*binding);
  if (std::optional<bool> materialized = state->getBoolean("materialized"))
    plan.materialized = *materialized;
  plan.targetHash = computeTargetContentHash(target);
  plan.machineHash = machine::computeContentHash(target.machine());
  plan.layoutHash = target.layouts().computeContentHash();
  plan.ruleHash = target.rules().computeContentHash();

  if (const llvm::json::Array *placements = state->getArray("placements")) {
    for (const llvm::json::Value &element : *placements) {
      const llvm::json::Object *object = element.getAsObject();
      if (!object)
        return reportError("plan report placement is not an object");
      PlanPlacement placement;
      std::optional<int64_t> node = object->getInteger("node");
      if (!node || *node < 0)
        return reportError("plan report placement has no node");
      placement.node = static_cast<WorkloadNodeId>(*node);
      // A placement must still resolve in the supplied source graph, so a
      // report cannot be replayed against a changed input graph.
      const WorkloadNode *workloadNode = graph.findNode(placement.node);
      if (!workloadNode)
        return reportError("plan report placement node does not resolve in the "
                           "source graph");
      if (std::optional<llvm::StringRef> instance =
              object->getString("instance"))
        placement.instance = static_cast<InstanceId>(parseHexId(*instance));
      if (std::optional<llvm::StringRef> rule = object->getString("rule")) {
        placement.rule = rule->str();
        if (const RuleDef *def = target.rules().find(placement.rule))
          if (def->matchOp != workloadNode->opName)
            return reportError("plan report placement rule does not implement "
                               "the source graph's operation");
      }
      if (std::optional<llvm::StringRef> bundle = object->getString("bundle"))
        placement.bundle.name = bundle->str();
      if (std::optional<llvm::StringRef> emitter = object->getString("emitter"))
        placement.bundle.emitterKey = emitter->str();
      if (std::optional<llvm::StringRef> executor =
              object->getString("executor"))
        placement.executor = executor->str();
      if (const llvm::json::Object *memories = object->getObject("memories"))
        for (const auto &entry : *memories)
          if (std::optional<llvm::StringRef> value = entry.second.getAsString())
            placement.memories[entry.first] = value->str();
      if (const llvm::json::Object *layouts = object->getObject("layouts"))
        for (const auto &entry : *layouts)
          if (std::optional<llvm::StringRef> value = entry.second.getAsString())
            placement.layouts[entry.first] = value->str();
      if (const llvm::json::Array *portMemories =
              object->getArray("portMemories")) {
        for (const llvm::json::Value &element : *portMemories) {
          const llvm::json::Object *entry = element.getAsObject();
          if (!entry)
            return reportError("plan report port memory is not an object");
          const llvm::json::Object *port = entry->getObject("port");
          if (!port)
            return reportError("plan report port memory has no port");
          llvm::Expected<PortRef> ref = jsonPortRef(port, "port memory port");
          if (!ref)
            return ref.takeError();
          if (!lookupPort(graph, *ref))
            return reportError("plan report port memory endpoint does not "
                               "resolve in the source graph");
          std::optional<llvm::StringRef> memory = entry->getString("memory");
          if (!memory)
            return reportError("plan report port memory has no memory");
          placement.portMemoryBindings.push_back(
              PortMemoryBinding{*ref, memory->str()});
        }
      }
      if (const llvm::json::Array *solutions = object->getArray("solutions")) {
        for (const llvm::json::Value &solution : *solutions) {
          const llvm::json::Object *entry = solution.getAsObject();
          if (!entry)
            return reportError("plan report solution is not an object");
          std::optional<llvm::StringRef> key = entry->getString("key");
          if (!key)
            return reportError("plan report solution has no key");
          SolvedLayout solved;
          if (std::optional<llvm::StringRef> klass = entry->getString("class"))
            solved.layoutClass = klass->str();
          if (const llvm::json::Object *port = entry->getObject("port")) {
            llvm::Expected<PortRef> ref = jsonPortRef(port, "solution port");
            if (!ref)
              return ref.takeError();
            if (!lookupPort(graph, *ref))
              return reportError(
                  "plan report solution endpoint does not resolve in the "
                  "source graph");
            solved.port = *ref;
          }
          if (const llvm::json::Object *parameters =
                  entry->getObject("parameters"))
            for (const auto &parameter : *parameters) {
              if (std::optional<int64_t> integer =
                      parameter.second.getAsInteger())
                solved.parameters[parameter.first] = *integer;
              else if (std::optional<llvm::StringRef> text =
                           parameter.second.getAsString())
                solved.parameters[parameter.first] = text->str();
              else
                return reportError("plan report solution parameter is neither "
                                   "an integer nor a string");
            }
          // The concrete map is a pure function of the parameters and the
          // layout declaration, and an `AffineMap` is owned by an
          // `MLIRContext`. The report records the map for verification, but a
          // replay reconstructs it from the parameters (and, when binding,
          // re-validates it against the declaration) rather than returning a
          // context-bound object whose lifetime this reader cannot own.
          placement.layoutSolutions[*key] = std::move(solved);
        }
      }
      if (const llvm::json::Object *parameters =
              object->getObject("ruleParameters"))
        for (const auto &parameter : *parameters) {
          if (std::optional<int64_t> integer = parameter.second.getAsInteger())
            placement.resolvedParameters[parameter.first] = *integer;
          else if (std::optional<llvm::StringRef> text =
                       parameter.second.getAsString())
            placement.resolvedParameters[parameter.first] = text->str();
          else
            return reportError("plan report rule parameter is neither an "
                               "integer nor a string");
        }
      plan.placements.push_back(std::move(placement));
    }
  }

  if (const llvm::json::Array *connections = state->getArray("connections")) {
    for (const llvm::json::Value &element : *connections) {
      const llvm::json::Object *object = element.getAsObject();
      if (!object)
        return reportError("plan report connection is not an object");
      PlanConnection connection;
      if (std::optional<llvm::StringRef> id = object->getString("id"))
        connection.id = static_cast<ConnectionId>(parseHexId(*id));
      if (std::optional<int64_t> value = object->getInteger("value"))
        connection.value = static_cast<WorkloadValueId>(*value);
      std::optional<llvm::StringRef> kind = object->getString("kind");
      if (!kind)
        return reportError("plan report connection has no kind");
      std::optional<ConnectionKind> symbolized = symbolizeConnectionKind(*kind);
      if (!symbolized)
        return reportError("plan report connection has an unknown kind");
      connection.kind = *symbolized;
      if (const llvm::json::Array *route = object->getArray("route"))
        for (const llvm::json::Value &node : *route)
          if (std::optional<llvm::StringRef> text = node.getAsString())
            connection.route.push_back(text->str());
      if (const llvm::json::Array *engines = object->getArray("engines"))
        for (const llvm::json::Value &engine : *engines)
          if (std::optional<llvm::StringRef> text = engine.getAsString())
            connection.engines.push_back(text->str());
      if (const llvm::json::Array *consumers = object->getArray("consumers"))
        for (const llvm::json::Value &consumer : *consumers)
          if (std::optional<llvm::StringRef> id = consumer.getAsString())
            connection.consumers.push_back(
                static_cast<InstanceId>(parseHexId(*id)));
      if (const llvm::json::Array *storageIds = object->getArray("storageIds"))
        for (const llvm::json::Value &id : *storageIds)
          if (std::optional<llvm::StringRef> value = id.getAsString())
            connection.storageIds.push_back(parseHexId(*value));
      if (const llvm::json::Object *producer =
              object->getObject("producerPort")) {
        llvm::Expected<PortRef> ref = jsonPortRef(producer, "producerPort");
        if (!ref)
          return ref.takeError();
        if (!lookupPort(graph, *ref))
          return reportError(
              "plan report connection endpoint does not resolve in the source "
              "graph");
        connection.producerPort = *ref;
      }
      if (const llvm::json::Array *consumerPorts =
              object->getArray("consumerPorts"))
        for (const llvm::json::Value &port : *consumerPorts) {
          llvm::Expected<PortRef> ref =
              jsonPortRef(port.getAsObject(), "consumerPorts");
          if (!ref)
            return ref.takeError();
          if (!lookupPort(graph, *ref))
            return reportError(
                "plan report connection endpoint does not resolve in the "
                "source graph");
          connection.consumerPorts.push_back(*ref);
        }
      if (const llvm::json::Object *transform =
              object->getObject("transform")) {
        LayoutTransform layoutTransform;
        if (std::optional<llvm::StringRef> src = transform->getString("src"))
          layoutTransform.srcLayout = src->str();
        if (std::optional<llvm::StringRef> dst = transform->getString("dst"))
          layoutTransform.dstLayout = dst->str();
        // The transform's maps are context-bound and re-derived by a
        // materializer from the declaration and parameters; the report records
        // them for inspection only.
        connection.transform = layoutTransform;
      }
      plan.connectionPlans.push_back(std::move(connection));
    }
  }

  if (const llvm::json::Array *allocations = state->getArray("allocations")) {
    for (const llvm::json::Value &element : *allocations) {
      const llvm::json::Object *object = element.getAsObject();
      if (!object)
        return reportError("plan report allocation is not an object");
      StorageAllocation allocation;
      if (std::optional<llvm::StringRef> id = object->getString("id"))
        allocation.id = parseHexId(*id);
      if (std::optional<int64_t> value = object->getInteger("value"))
        allocation.value = static_cast<WorkloadValueId>(*value);
      if (std::optional<llvm::StringRef> memory = object->getString("memory"))
        allocation.memory = memory->str();
      if (std::optional<int64_t> bytes = object->getInteger("bytes"))
        allocation.bytes = static_cast<uint64_t>(*bytes);
      if (std::optional<llvm::StringRef> alias = object->getString("aliasOf"))
        allocation.aliasOf = parseHexId(*alias);
      if (std::optional<int64_t> begin = object->getInteger("beginStep"))
        allocation.beginStep = static_cast<uint64_t>(*begin);
      if (std::optional<int64_t> end = object->getInteger("endStep"))
        allocation.endStep = static_cast<uint64_t>(*end);
      plan.allocations.push_back(std::move(allocation));
    }
  }
  if (const llvm::json::Array *synchronization =
          state->getArray("synchronization")) {
    for (const llvm::json::Value &element : *synchronization) {
      const llvm::json::Object *object = element.getAsObject();
      if (!object)
        return reportError("plan report synchronization step is not an object");
      SynchronizationStep step;
      if (std::optional<int64_t> id = object->getInteger("id"))
        step.id = static_cast<uint64_t>(*id);
      if (const llvm::json::Array *waitsFor = object->getArray("waitsFor"))
        for (const llvm::json::Value &id : *waitsFor)
          if (std::optional<llvm::StringRef> text = id.getAsString())
            step.waitsFor.push_back(
                static_cast<ConnectionId>(parseHexId(*text)));
      if (const llvm::json::Array *precedes = object->getArray("precedes"))
        for (const llvm::json::Value &port : *precedes) {
          llvm::Expected<PortRef> ref =
              jsonPortRef(port.getAsObject(), "synchronization precedes");
          if (!ref)
            return ref.takeError();
          step.precedes.push_back(*ref);
        }
      if (std::optional<bool> barrier = object->getBoolean("requiresBarrier"))
        step.requiresBarrier = *barrier;
      plan.synchronization.push_back(std::move(step));
    }
  }
  if (const llvm::json::Array *steps = state->getArray("steps")) {
    for (const llvm::json::Value &element : *steps) {
      const llvm::json::Object *object = element.getAsObject();
      if (!object)
        return reportError("plan report step is not an object");
      PlanStep step;
      if (std::optional<int64_t> id = object->getInteger("id"))
        step.id = static_cast<uint64_t>(*id);
      std::optional<llvm::StringRef> kind = object->getString("kind");
      if (!kind)
        return reportError("plan report step has no kind");
      std::optional<PlanStepKind> symbolized = symbolizePlanStepKind(*kind);
      if (!symbolized)
        return reportError("plan report step has unknown kind");
      step.kind = *symbolized;
      if (std::optional<int64_t> node = object->getInteger("node"))
        step.node = static_cast<WorkloadNodeId>(*node);
      if (std::optional<llvm::StringRef> connection =
              object->getString("connection"))
        step.connection = static_cast<ConnectionId>(parseHexId(*connection));
      plan.steps.push_back(step);
    }
  }
  if (const llvm::json::Array *edges = state->getArray("stepEdges")) {
    for (const llvm::json::Value &element : *edges) {
      const llvm::json::Object *object = element.getAsObject();
      if (!object)
        return reportError("plan report step edge is not an object");
      PlanStepEdge edge;
      if (std::optional<int64_t> from = object->getInteger("from"))
        edge.from = static_cast<uint64_t>(*from);
      if (std::optional<int64_t> to = object->getInteger("to"))
        edge.to = static_cast<uint64_t>(*to);
      plan.stepEdges.push_back(edge);
    }
  }

  for (const PlanPlacement &placement : plan.placements)
    plan.instances.push_back(placement.instance);
  llvm::sort(plan.instances);
  plan.instances.erase(
      std::unique(plan.instances.begin(), plan.instances.end()),
      plan.instances.end());
  for (const PlanConnection &connection : plan.connectionPlans)
    plan.connections.push_back(connection.id);
  llvm::sort(plan.connections);
  return plan;
}

} // namespace mlir::llk::mapping
