//===- AVX2MappingTarget.h - AVX2 mapping package (D7) --------------------===//
//
// Part of the target package layer (epic #67, workstream D7).
//
// This header is the *only* place that knows the AVX2 mapping configuration by
// name: which machine profile, which LLKMap files, and which emitter keys the
// AVX2 plugin implements. Generic mapping code sees a `MappingTarget` and
// never learns any of it -- that separation is what a second target proves
// (see the generic-accelerator package).
//
// Emitter keys are code, not configuration: they name C++ emitters the plugin
// provides, which is why they live here rather than in rules.llkmap.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_TARGET_X86_MAPPING_AVX2MAPPINGTARGET_H
#define LLK_TARGET_X86_MAPPING_AVX2MAPPINGTARGET_H

#include "LLK/Mapping/MappingTarget.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <memory>

namespace mlir::llk::target::avx2 {

/// The emitter keys the AVX2 plugin implements. A rule whose `emit` is not in
/// this set is rejected when the target loads.
llvm::ArrayRef<llvm::StringLiteral> emitterKeys();

/// Creates the AVX2 emitter for `key`, which must be one `emitterKeys()` names.
/// The emitter carries this plugin's lowering, so a bundle handed to it is
/// lowered as AVX2 rather than rejected as unsupported.
std::unique_ptr<mapping::TargetEmitter> createAVX2Emitter(llvm::StringRef key);

/// Loads the AVX2 mapping target from a configuration root, which must contain
/// `machines/x86-avx2-v2.yaml`, `mapping/x86-avx2/layouts.llkmap`, and
/// `mapping/x86-avx2/rules.llkmap`. The target is verified before it is
/// returned, and its emitters are the AVX2 plugin's own rather than the
/// configuration-only default.
llvm::Expected<std::unique_ptr<mapping::MappingTarget>>
createMappingTarget(llvm::StringRef configurationRoot);

} // namespace mlir::llk::target::avx2

#endif // LLK_TARGET_X86_MAPPING_AVX2MAPPINGTARGET_H
