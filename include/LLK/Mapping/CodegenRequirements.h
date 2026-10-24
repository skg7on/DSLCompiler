//===- CodegenRequirements.h - Host codegen requirements --------*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_CODEGENREQUIREMENTS_H
#define LLK_MAPPING_CODEGENREQUIREMENTS_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace mlir::llk::mapping {
struct TargetCodegenRequirements {
  std::string architecture;
  std::string cpu;
  std::vector<std::string> requiredFeatures;
  /// Exact implemented vector length in bits, when the target's code assumes
  /// one. Nullopt means the target makes no vector-length claim.
  std::optional<int64_t> vectorLengthBits = std::nullopt;
};

/// Check a selected target's ISA policy against the detected host facts.
llvm::Error checkHostRequirements(const TargetCodegenRequirements &requirements,
                                  llvm::StringRef hostArchitecture,
                                  const llvm::StringMap<bool> &hostFeatures,
                                  std::optional<int64_t> hostVectorLengthBits);

/// Check requirements when the caller has no host vector-length information.
inline llvm::Error
checkHostRequirements(const TargetCodegenRequirements &requirements,
                      llvm::StringRef hostArchitecture,
                      const llvm::StringMap<bool> &hostFeatures) {
  return checkHostRequirements(requirements, hostArchitecture, hostFeatures,
                               std::nullopt);
}

/// Convert a `prctl(PR_SVE_GET_VL)` result to a vector length in bits.
///
/// The kernel reports the implemented length in bytes in the low 16 bits
/// (`PR_SVE_VL_LEN_MASK`), so the bits value is that field times eight.
int64_t vectorLengthBitsFromPrctl(long value);

} // namespace mlir::llk::mapping
#endif
