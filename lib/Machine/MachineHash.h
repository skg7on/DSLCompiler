//===- MachineHash.h - Stable content hashing for machine models ----------===//
//
// Internal to LLKMachine. The machine content hash keys the latency cache
// (design §17.4), so it must be stable across toolchain and standard-library
// changes -- FNV-1a 64, fixed by specification, not `llvm::hash_value`.
//
// NOTE: the mapping core (LLK/Mapping) carries an equivalent helper that
// LLKMachine must not depend on. When both land, these belong in a shared
// LLK/Support header; until then the duplication is deliberate.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MACHINE_MACHINEHASH_H
#define LLK_MACHINE_MACHINEHASH_H

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <cstdio>
#include <string>

namespace mlir::llk::machine {

inline constexpr uint64_t kFnvOffsetBasis = 0xcbf29ce484222325ULL;
inline constexpr uint64_t kFnvPrime = 0x100000001b3ULL;

inline uint64_t stableHash(llvm::StringRef data) {
  uint64_t hash = kFnvOffsetBasis;
  for (char raw : data) {
    hash ^= static_cast<unsigned char>(raw);
    hash *= kFnvPrime;
  }
  return hash;
}

/// Formats `value` as exactly 16 lowercase hex digits.
inline std::string hex64(uint64_t value) {
  char buffer[17];
  std::snprintf(buffer, sizeof(buffer), "%016llx",
                static_cast<unsigned long long>(value));
  return std::string(buffer);
}

/// Fixed-format double, so a canonical string is byte-stable.
inline std::string formatDouble(double value) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "%.6f", value);
  return std::string(buffer);
}

} // namespace mlir::llk::machine

#endif // LLK_MACHINE_MACHINEHASH_H
