//===- StableHash.cpp - Toolchain-stable hashing for mapping ids ----------===//

#include "LLK/Mapping/StableHash.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdio>

namespace mlir::llk::mapping {

namespace {
/// FNV-1a 64 parameters, fixed by the specification.
constexpr uint64_t kFnvOffsetBasis = 0xcbf29ce484222325ULL;
constexpr uint64_t kFnvPrime = 0x100000001b3ULL;
} // namespace

uint64_t stableHashBytes(uint64_t seed, llvm::ArrayRef<char> data) {
  uint64_t hash = seed;
  for (char raw : data) {
    hash ^= static_cast<unsigned char>(raw);
    hash *= kFnvPrime;
  }
  return hash;
}

uint64_t stableHash(llvm::StringRef data) {
  return stableHashBytes(kFnvOffsetBasis,
                         llvm::ArrayRef<char>(data.data(), data.size()));
}

uint64_t stableHashCombine(uint64_t seed, uint64_t value) {
  char bytes[8];
  for (unsigned i = 0; i < 8; ++i)
    bytes[i] = static_cast<char>((value >> (8 * i)) & 0xff);
  return stableHashBytes(seed, llvm::ArrayRef<char>(bytes, 8));
}

uint64_t stableHashSortedSet(llvm::ArrayRef<uint64_t> values) {
  llvm::SmallVector<uint64_t> sorted(values);
  llvm::sort(sorted);
  uint64_t hash = kFnvOffsetBasis;
  for (uint64_t value : sorted)
    hash = stableHashCombine(hash, value);
  return hash;
}

std::string hexId(uint64_t value) {
  char buffer[17];
  std::snprintf(buffer, sizeof(buffer), "%016llx",
                static_cast<unsigned long long>(value));
  return std::string(buffer);
}

} // namespace mlir::llk::mapping
