//===- resource_regression_fixture.h - Issue #129 shared test fixtures ----===//
//
// Test-support only (issue #129). ONE fixture builds the small mapping
// scenarios the resource/cost repair regresses against, directly from finite
// MachineModel data, LLKMap text and parsed Micro-IR -- never by calling the
// production search or storage planner to compute an expected result. A case's
// expected resources, capacities and schedules are literal values the test
// asserts, so a regression cannot pass by agreeing with the code under test.
//
// The fixture is deliberately extensible: each case is one entry in a table in
// the `.cpp`, and later tasks append their own case (R3 `missing-memory` /
// `named-ports`, R4 `two-hop`, R5 `sequential` / `pipeline-four` /
// `parallel-overlap`, R7 `capacity-topk`, R8 `joint-oracle`) without
// restructuring the constructor or this interface.
//
// It is linked only into integration regression tests, never into a production
// library.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_TEST_RESOURCE_REGRESSION_FIXTURE_H
#define LLK_TEST_RESOURCE_REGRESSION_FIXTURE_H

#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/WorkloadGraph.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>

namespace issue129 {

/// One fixture scenario: the parsed source kernel (and the workload graph
/// extracted from it) plus the loaded target it is mapped against. `context`
/// owns every MLIR object the other members reference, and is declared first so
/// it outlives them.
struct ResourceCase {
  std::unique_ptr<mlir::MLIRContext> context;
  mlir::OwningOpRef<mlir::ModuleOp> source;
  mlir::llk::mapping::WorkloadGraph graph;
  std::unique_ptr<mlir::llk::mapping::MappingTarget> target;
};

/// Builds the named case, or an error naming the ones that exist. The allowed
/// names are owned by task: R1 `two-compute`; R3 `missing-memory`,
/// `named-ports`; R4 `two-hop`; R5 `sequential`, `pipeline-four`,
/// `parallel-overlap`, `padded-layout`; R7 `capacity-topk`; R8 `joint-oracle`.
/// A case's own variants (R4's reduced-L2 intermediate, for instance) are built
/// by the test from the case's literal machine rather than by adding a name.
llvm::Expected<ResourceCase> resourceCase(llvm::StringRef name);

/// Runs the *ordinary* mapping search over the case's extracted source graph
/// with `options`. Once R7 defines the completion callback, this is where the
/// evaluator is supplied; before then a case is searched exactly as a CLI run
/// would search it.
llvm::Expected<mlir::llk::mapping::MappingSearchResult>
searchCase(ResourceCase &c,
           const mlir::llk::mapping::MappingSearchOptions &options);

} // namespace issue129

#endif // LLK_TEST_RESOURCE_REGRESSION_FIXTURE_H
