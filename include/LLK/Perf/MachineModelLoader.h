//===- MachineModelLoader.h - YAML machine model loader -------------------===//
//
// Part of the M10 tile-aware MachineModel (issue #45).
//
// Reads `machines/*.yaml` into a MachineModel using LLVM's YAML parser. The
// loader owns structure and vocabulary; verifyMachineModel() owns values, so a
// hand-built model and a parsed one are held to the same rules.
//
// Every failure returns the first diagnostic found, prefixed with the source
// name and, where the YAML parser knows one, a `line:column` location:
//
//   machines/x86-avx2-cpu.yaml:14:5: error: compute.matrix_engines[0]: \
//       unknown key 'tile_shape'
//
//===----------------------------------------------------------------------===//

#ifndef LLK_PERF_MACHINEMODELLOADER_H
#define LLK_PERF_MACHINEMODELLOADER_H

#include "LLK/Perf/MachineModel.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

namespace mlir::llk::perf {

/// Parses and verifies a machine model from YAML text. `sourceName` names the
/// origin in diagnostics.
llvm::Expected<MachineModel>
parseMachineModel(llvm::StringRef yamlText,
                  llvm::StringRef sourceName = "<memory>");

/// Reads, parses, and verifies a machine model from a YAML file.
llvm::Expected<MachineModel> loadMachineModel(llvm::StringRef path);

} // namespace mlir::llk::perf

#endif // LLK_PERF_MACHINEMODELLOADER_H
