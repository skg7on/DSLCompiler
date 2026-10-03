//===- GenericAcceleratorMappingTarget.h - second target (D7) -------------===//
//
// Part of the target package layer (epic #67, workstream D7).
//
// The second complete mapping target. It exists to prove a claim: a target
// that is not AVX2 -- different owner hierarchy, different compute kinds,
// different layouts, different emitters -- needs no change to generic Micro
// ODS, to `LLK/Mapping`, or to the search. Everything that differs lives in
// this package and in `mapping/generic-ai-accel/`.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_TARGET_GENERICACCELERATOR_MAPPING_GENERICACCELERATORMAPPINGTARGET_H
#define LLK_TARGET_GENERICACCELERATOR_MAPPING_GENERICACCELERATORMAPPINGTARGET_H

#include "LLK/Mapping/MappingTarget.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>

namespace mlir::llk::target::generic_accel {

/// The emitter keys this accelerator plugin implements.
llvm::ArrayRef<llvm::StringRef> emitterKeys();

/// Loads the accelerator mapping target from a configuration root, which must
/// contain `machines/generic-ai-accel-v2.yaml`,
/// `mapping/generic-ai-accel/layouts.llkmap`, and
/// `mapping/generic-ai-accel/rules.llkmap`.
llvm::Expected<std::unique_ptr<mapping::MappingTarget>>
createMappingTarget(llvm::StringRef configurationRoot);

} // namespace mlir::llk::target::generic_accel

#endif // LLK_TARGET_GENERICACCELERATOR_MAPPING_GENERICACCELERATORMAPPINGTARGET_H
