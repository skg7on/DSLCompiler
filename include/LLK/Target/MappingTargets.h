//===- MappingTargets.h - Register the mapping target packages
//-------------===//
//
// Part of the target package layer (epic #67).
//
// The mapping core never names a backend. A tool that has to honour
// `--target=<name>` therefore cannot call a package's factory directly, and
// this is the seam that lets it: every package this build ships is registered
// here, once, at the composition root.
//
// It is the same shape MLIR uses for dialects and passes -- the tool says
// "register everything", and what "everything" means is decided where the
// packages can be seen.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_TARGET_MAPPINGTARGETS_H
#define LLK_TARGET_MAPPINGTARGETS_H

namespace mlir::llk::target {

/// Registers every mapping target package this build ships, so a caller can
/// select one by name without naming it in generic code. Registering a name
/// that is already taken replaces it.
void registerAllMappingTargets();

} // namespace mlir::llk::target

#endif // LLK_TARGET_MAPPINGTARGETS_H
