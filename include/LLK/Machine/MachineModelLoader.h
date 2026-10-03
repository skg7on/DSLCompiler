//===- MachineModelLoader.h - Load a v2 machine model from YAML -----------===//
//
// Part of the v2 machine model (issue #82, epic #67).
//
// The loader reads the `schema: llk.machine.v2` shape -- a list of nodes per
// kind -- into the typed `MachineModel`, then runs `verifyMachineModel` so a
// file and a hand-built model face the same rules. It rejects unknown keys,
// wrong node types, missing required fields, and any schema major other than
// the one this build understands (design §11.6).
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
