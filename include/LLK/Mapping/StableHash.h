//===- StableHash.h - Toolchain-stable hashing for mapping ids ------------===//
//
// Part of the target-independent mapping core (issue #80, epic #67 D1).
//
// Mapping objects are compared, sorted, and persisted by stable id, so the
// hash that produces those ids must not change when the toolchain or standard
// library changes. FNV-1a 64 is used for exactly that reason: it is fixed by
// specification, deterministic, and dependency-free. `llvm::hash_value` is
// explicitly not used -- its result is not promised to be stable.
//
// Ordering is canonical at the call site: `stableHashSortedSet` sorts before
// folding, so a caller that collects ids in any order still gets one hash.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_STABLEHASH_H
#define LLK_MAPPING_STABLEHASH_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <string>

namespace mlir::llk::mapping {

/// FNV-1a 64-bit hash of `data`.
uint64_t stableHash(llvm::StringRef data);

/// Continues an FNV-1a fold: hashes the raw bytes of `data` starting from
/// `seed`. Pass the result of a previous call to chain fields.
uint64_t stableHashBytes(uint64_t seed, llvm::ArrayRef<char> data);

/// Folds the 8 little-endian bytes of `value` into `seed`. A plain fold, so
/// the order of successive calls matters; use `stableHashSortedSet` when the
/// values form a set.
uint64_t stableHashCombine(uint64_t seed, uint64_t value);

/// Order-independent hash of an integer set: `values` is sorted before being
/// folded, so callers may collect the members in any order. Duplicates are
/// significant -- the sorted multiset is hashed as given.
uint64_t stableHashSortedSet(llvm::ArrayRef<uint64_t> values);

/// Formats `value` as exactly 16 lowercase hexadecimal digits. Ids are printed
/// with fixed width so lexicographic order matches numeric order.
std::string hexId(uint64_t value);

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_STABLEHASH_H
