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

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llk {

struct MappedObjectOptions {
  std::string triple;
  std::string cpu;
  std::vector<std::string> features;
};

struct MappedObjectIdentity {
  uint32_t schemaVersion = 1;
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
  std::string executionIdentity;
};

/// Lowers `prepared` through the same LLVM pipeline the JIT uses and writes an
/// object for `options.triple`. Returns the C-interface wrapper symbol.
llvm::Expected<std::string> emitMappedObject(PreparedMappedKernel &prepared,
                                             const MappedObjectOptions &options,
                                             llvm::raw_ostream &objectStream);

llvm::Error writeMappedObjectIdentity(const MappedObjectIdentity &identity,
                                      llvm::raw_ostream &jsonStream);

} // namespace llk
#endif
