//===- MappingMetadata.h - Persisted selected-plan metadata (task B1) -----===//
//
// Part of the target-independent mapping core (epic #67, stage B, task B1).
//
// The binder used to stamp a selected plan onto a kernel as ad hoc metadata
// that could not be round-tripped or re-verified. This header defines the
// *generic*, target-neutral serialization of a selected plan's
// execution-affecting state: explicit endpoint occurrences, resolved rule
// parameters, compute/memory bindings, concrete layout maps, storage ids, the
// canonical source-graph identity and the target content hashes.
//
// Encoding is separated from verification (MappingMetadata.cpp vs
// PlanVerification.cpp) so no target-specific semantics enter generic parsing:
// the encoder only ever compares, hashes and reports target-owned ids.
//
// The metadata schema version (`kMappingMetadataVersion`) is distinct from the
// JSON report schema version (`kPlanReportVersion` in PlanReport.h). A reader
// keys its parser on the `micro.plan` `schema_version`; a schema change that
// alters the meaning of an existing field bumps it.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_MAPPINGMETADATA_H
#define LLK_MAPPING_MAPPINGMETADATA_H

#include "LLK/Mapping/MappingPlan.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/WorkloadGraph.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <string>

namespace mlir::llk::mapping {

/// The `micro.plan` metadata schema version this encoder writes. Version 2 adds
/// explicit endpoint occurrences, resolved rule parameters, compute/memory
/// bindings, concrete layout maps, storage ids, the source-graph hash and the
/// target content hashes; it also requires every mapped operation to record its
/// layout container or an explicit "no layout" marker, so deleting the
/// container can no longer skip layout verification. It is *not*
/// `kPlanReportVersion`: that versions the JSON report, this versions the
/// bound-IR metadata.
inline constexpr uint64_t kMappingMetadataVersion = 2;

/// The largest `micro.plan` schema version this reader understands. A kernel
/// recording a greater version was written by a newer compiler and must not be
/// replayed under these semantics.
inline constexpr uint64_t kSupportedMappingMetadataVersion = 2;

/// Canonical, pre-materialization content hash of `graph`: every semantic
/// (non-binder-movement) node's operation name, its operand occurrences with
/// types and access maps, its semantic attributes (bookkeeping attributes
/// omitted), and the connectivity between those nodes. Binder-emitted movement
/// operations are dropped and their results resolved back to the values they
/// copy, so the hash of a source graph and of the materialized graph built from
/// it agree. Used as the recorded source-graph identity.
uint64_t computeSourceGraphHash(const WorkloadGraph &graph);

/// Canonical content hash of a materialized `micro.kernel`: `graph` is the
/// materialized workload graph and `binding` its SSA correspondence. Movement
/// results and binder-emitted `micro.transform` results are resolved back to
/// the value they read before the source graph is rendered, so a materialized
/// kernel hashes to the identity of the source it was bound from. Fails when
/// the kernel's workload graph cannot be extracted.
llvm::Expected<uint64_t> computeModuleSourceGraphHash(mlir::Operation *kernel);

/// The source (projected) workload graph of a materialized `micro.kernel`,
/// together with the operation each projected — source — node id came from. A
/// recorded endpoint occurrence names a *source* node id, so resolving one in a
/// materialized kernel must go through this view rather than the materialized
/// graph, whose node ids shifted when movement operations were inserted.
struct SourceGraphView {
  WorkloadGraph graph;
  llvm::DenseMap<WorkloadNodeId, mlir::Operation *> nodeOps;
};
llvm::Expected<SourceGraphView> buildSourceGraphView(mlir::Operation *kernel);

/// Content hash of a target: its name folded with the machine, layout and rule
/// library content hashes. Two targets that differ in any of those, or in name,
/// hash differently, so a plan bound for one is rejected against another.
uint64_t computeTargetContentHash(const MappingTarget &target);

/// Persists `plan` as schema-v2 metadata on the single `micro.kernel` in
/// `module`: `micro.plan` on the kernel (schema version, id, binding hash,
/// materialization flag, source-graph and target hashes), `micro.mapping` on
/// each covered operation (rule, instance, bundle, emitter, executor, memories,
/// layouts, resolved parameters and per-port layout entries) and
/// `micro.routes` on the kernel (each connection's endpoints, ordered
/// route/engines, storage ids and transform). Fails when the module does not
/// contain exactly one covered kernel or the plan names a rule the target does
/// not declare. The source-graph hash is computed from the kernel the encoder
/// sees, so call it before the binder materializes movements.
llvm::Error encodeSelectedPlan(mlir::ModuleOp module, const CoveringPlan &plan,
                               const MappingTarget &target);

/// Reads the selected state `encodeSelectedPlan` recorded on `module` back into
/// a `CoveringPlan`, restoring the execution-affecting choices (placements,
/// layout solutions with their concrete maps, endpoints, routes, storage ids
/// and resource bindings). The recorded source-graph and target hashes must
/// still match `module` and `target`, so a frozen plan cannot bypass current
/// verification. A kernel recording a schema version greater than
/// `kSupportedMappingMetadataVersion` is rejected; a legacy (v1) binding is
/// read only when its missing endpoint/resource associations are uniquely
/// recoverable from the graph, and rejected when they are ambiguous.
llvm::Expected<CoveringPlan> decodeSelectedPlan(mlir::ModuleOp module,
                                                const MappingTarget &target);

//===----------------------------------------------------------------------===//
// Shared typed readers for generic mapping metadata
//===----------------------------------------------------------------------===//

/// Reads a required string field from a metadata dictionary with a checked
/// cast. Generic mapping metadata is untrusted (design §25.1): an absent field
/// and a wrongly-typed one are distinct, stable diagnostics rather than a cast
/// that aborts the process.
llvm::Expected<std::string> readMetadataString(mlir::DictionaryAttr dict,
                                               llvm::StringRef name,
                                               llvm::StringRef where);

/// Reads a `key = "value"` string-map attribute, type-checking the container
/// and every entry.
llvm::Expected<llvm::StringMap<std::string>>
readMetadataStringMap(mlir::Attribute raw, llvm::StringRef name,
                      llvm::StringRef where);

/// Reads an array-of-strings attribute, type-checking the container and every
/// element.
llvm::Expected<llvm::SmallVector<std::string, 4>>
readMetadataStringArray(mlir::Attribute raw, llvm::StringRef name,
                        llvm::StringRef where);

/// Reads an endpoint occurrence recorded as `{node, direction, index}` with
/// checked integer width and range and a checked direction string. `index` is
/// validated against the graph by the caller (`lookupPort`); this only rejects
/// malformed containers and out-of-domain integers.
llvm::Expected<PortRef> readMetadataPortRef(mlir::Attribute raw,
                                            llvm::StringRef name,
                                            llvm::StringRef where);

/// The `{node, direction, index}` dictionary a `PortRef` encodes to.
mlir::Attribute metadataPortRefAttr(mlir::MLIRContext *context,
                                    const PortRef &port);

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_MAPPINGMETADATA_H
