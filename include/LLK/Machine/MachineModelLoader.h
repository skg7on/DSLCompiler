//===- MachineModelLoader.h - Load a v2 machine model from YAML -----------===//
//
// Part of the v2 machine model (issue #82, epic #67).
//
// The loader reads the `schema: llk.machine.v2` shape -- a list of nodes per
// kind -- into the typed `MachineModel`, then runs `verifyMachineModel` so a
// file and a hand-built model face the same rules. It rejects wrong node types,
// missing required fields, unknown kinds, and any schema major other than the
// one this build understands (design §11.6).
//
// Schema minor tolerance (§11.6): the declared minor is preserved on the model
// (`MachineModel::schemaMinor`). When the file's minor is *above*
// `kSupportedSchemaMinor`, unknown keys are ignored -- a newer minor may add
// optional keys this build predates. When the file's minor is at or below the
// supported one, an unknown key is rejected: at the current minor it is a typo,
// and discarding it silently would defeat the diagnostic §11.6 relies on.
//
// The minor is self-declared and unbounded: any file may write a large minor
// (e.g. `llk.machine.v2.99`) and thereby opt out of unknown-key rejection --
// including for a genuine typo. That is deliberate: §11.6 asks the loader not
// to reject a newer-minor file it cannot fully understand, and the schema major
// plus `verifyMachineModel` still guard structure, required fields, and kinds,
// so tolerance never extends past unknown keys.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MACHINE_MACHINEMODELLOADER_H
#define LLK_MACHINE_MACHINEMODELLOADER_H

#include "LLK/Machine/MachineModel.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir::llk::machine {

/// Parses `yamlText`. `sourceName` is used in diagnostics only.
llvm::Expected<MachineModel> parseMachineModel(llvm::StringRef yamlText,
                                               llvm::StringRef sourceName);

/// Reads and parses the file at `path`.
llvm::Expected<MachineModel> loadMachineModel(llvm::StringRef path);

} // namespace mlir::llk::machine

#endif // LLK_MACHINE_MACHINEMODELLOADER_H
