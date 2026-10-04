//===- SearchBindingLoader.h - micro.candidate -> SearchBinding -----------===//
//
// Part of phase-4 of epic #67: the producer for `mapping::SearchBinding`.
//
// A `micro.candidate` is a complete assignment of its space's parameters, so
// loading one is where the binding-level global legality check happens
// (ruling S2, design §16.5): every declared parameter is bound, every bound
// name is declared, and every value lies inside its parameter's declared
// `micro.param` choices. A candidate outside its domain is a startup
// diagnostic, not a late search failure, so the caller gets an error naming
// the offender rather than a binding the tuner will reject later.
//
// Shape-dependent legality (capacity, MMA compatibility) is *not* checked
// here; that is the perf layer's evaluator, and duplicating it would give the
// rule two homes. This layer only decides whether the point is inside the
// space the IR declares.
//
// Authority for the domain check is split, and deliberately so. On the
// file-to-load path the authority is the *dialect verifier*: `SearchSpaceOp::
// verify` rejects a candidate that leaves a parameter unbound, binds an
// undeclared name, mismatches integer and string, or names a value outside the
// declared choices, so such a candidate never parses. The checks below are
// therefore defence-in-depth, reachable only from programmatic or post-parse
// IR that no verifier has blessed -- which is exactly why this is a public
// API over an arbitrary `ModuleOp` rather than a private helper of a pass that
// always runs on verified input. The rejections are kept (not removed) because
// the loader is the last line before the tuner, and the two error texts are
// kept in step on purpose: both say a value "is not one of the declared
// choices", so the message contract cannot drift between the verifier and the
// loader.
//
// One rejection *is* reachable from valid parsed IR and is the loader's own:
// `micro.candidate` symbols are unique only within their `micro.search_space`,
// so two spaces may both declare `@candidate_17`; the loader is what refuses
// the resulting ambiguity.
//
// The loader lives in the MicroMapping pass library rather than in LLKMapping
// because it reads the space through `llk::perf::loadSearchSpace`, and LLKPerf
// depends on LLKMapping -- the reverse dependency is forbidden.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CONVERSION_MICROMAPPING_SEARCHBINDINGLOADER_H
#define LLK_CONVERSION_MICROMAPPING_SEARCHBINDINGLOADER_H

#include "LLK/Mapping/SearchBinding.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir {
class ModuleOp;
} // namespace mlir

namespace mlir::llk::mapping {

/// Loads the `SearchBinding` for the `micro.candidate` named `candidateSymbol`.
///
/// When `candidateSymbol` is empty the module's only candidate is selected;
/// an empty symbol is an error when the module has no candidate or more than
/// one. The result carries `candidateId` (the candidate's symbol) and the
/// value map, with `stableHash` computed by `makeSearchBinding`.
///
/// Fails when the candidate is unknown, when it does not bind every declared
/// parameter, when it binds a name no parameter declares, or when a value is
/// outside its parameter's declared choices.
llvm::Expected<SearchBinding>
loadSearchBinding(mlir::ModuleOp module, llvm::StringRef candidateSymbol = "");

} // namespace mlir::llk::mapping

#endif // LLK_CONVERSION_MICROMAPPING_SEARCHBINDINGLOADER_H
