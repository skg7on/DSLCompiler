//===- MappedObject.h - Target-neutral AOT object emission ---*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLK_RUNTIME_MAPPEDOBJECT_H
#define LLK_RUNTIME_MAPPEDOBJECT_H

#include "LLK/Runtime/PreparedMappedKernel.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace llk {

struct MappedObjectOptions {
  std::string triple;
  std::string cpu;
  std::vector<std::string> features;
  /// LLVM function attributes to attach to the exported `_mlir_ciface_*`
  /// wrapper. Supplied by the target's codegen contract; empty for targets
  /// whose entry needs no ABI annotation.
  std::vector<std::string> invocationAttributes;
};

struct MappedObjectIdentity {
  uint32_t schemaVersion = 2;
  std::string compilerVersion;
  std::string llvmVersion;
  std::string targetTriple;
  std::string cpu;
  std::vector<std::string> targetFeatures;
  std::string entrySymbol;
  std::string wrapperSymbol;
  uint32_t abiVersion = 0;
  uint64_t abiHash = 0;
  uint64_t sourceGraphHash = 0;
  uint64_t planId = 0;
  uint64_t machineHash = 0;
  uint64_t targetHash = 0;
  unsigned selectedGroupsVerified = 0;
  unsigned backendGroupsRealized = 0;
  unsigned referenceGroupsLowered = 0;
  std::optional<int64_t> streamingVectorLengthBits;
  std::vector<std::string> invocationAttributes;
  std::string executionIdentity;
};

/// Attach `attributes` to `wrapperSymbol` in `module`.
///
/// An empty attribute list is a no-op and does not require the wrapper to
/// exist. A nonempty list naming a missing wrapper is an error: silently
/// dropping a required ABI attribute would emit an object whose entry runs in
/// the wrong streaming state. A nonempty list also removes inherited Arm
/// ZA-ownership attributes from the wrapper (`aarch64_new_za`, `aarch64_in_za`,
/// `aarch64_out_za`, `aarch64_inout_za`, `aarch64_preserves_za`): the exported
/// entry declares its own ABI, and a wrapper with ZA ownership would establish
/// a TPIDR2 save block around the kernel.
llvm::Error applyInvocationAttributes(llvm::Module &module,
                                      llvm::StringRef wrapperSymbol,
                                      llvm::ArrayRef<std::string> attributes);

/// Lowers `prepared` through the same LLVM pipeline the JIT uses and writes an
/// object for `options.triple`. Returns the C-interface wrapper symbol.
llvm::Expected<std::string> emitMappedObject(PreparedMappedKernel &prepared,
                                             const MappedObjectOptions &options,
                                             llvm::raw_ostream &objectStream);

llvm::Error writeMappedObjectIdentity(const MappedObjectIdentity &identity,
                                      llvm::raw_ostream &jsonStream);

} // namespace llk
#endif
