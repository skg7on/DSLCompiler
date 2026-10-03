//===- SearchBinding.cpp - Immutable point in a search space --------------===//

#include "LLK/Mapping/SearchBinding.h"

#include "LLK/Mapping/StableHash.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <string>

namespace mlir::llk::mapping {

namespace {
/// The map's keys in sorted order; `StringMap` iteration order is not stable.
llvm::SmallVector<llvm::StringRef>
sortedKeys(const llvm::StringMap<SearchValue> &values) {
  llvm::SmallVector<llvm::StringRef> keys;
  keys.reserve(values.size());
  for (const auto &entry : values)
    keys.push_back(entry.first());
  llvm::sort(keys);
  return keys;
}

constexpr uint64_t kIntegerTag = 0;
constexpr uint64_t kSymbolicTag = 1;
} // namespace

uint64_t computeSearchBindingHash(const llvm::StringMap<SearchValue> &values) {
  constexpr uint64_t kSeed = 0xcbf29ce484222325ULL;
  uint64_t hash = kSeed;
  for (llvm::StringRef key : sortedKeys(values)) {
    hash = stableHashCombine(hash, stableHash(key));
    const SearchValue &value = values.find(key)->second;
    if (const auto *integer = std::get_if<int64_t>(&value)) {
      hash = stableHashCombine(hash, kIntegerTag);
      hash = stableHashCombine(hash, static_cast<uint64_t>(*integer));
    } else {
      hash = stableHashCombine(hash, kSymbolicTag);
      hash = stableHashCombine(hash, stableHash(std::get<std::string>(value)));
    }
  }
  return hash;
}

SearchBinding makeSearchBinding(std::string candidateId,
                                llvm::StringMap<SearchValue> values) {
  SearchBinding binding;
  binding.candidateId = std::move(candidateId);
  binding.values = std::move(values);
  binding.stableHash = computeSearchBindingHash(binding.values);
  return binding;
}

std::string canonicalSearchBindingString(const SearchBinding &binding) {
  std::string out;
  for (llvm::StringRef key : sortedKeys(binding.values)) {
    if (!out.empty())
      out += ';';
    out += key.str();
    out += '=';
    const SearchValue &value = binding.values.find(key)->second;
    if (const auto *integer = std::get_if<int64_t>(&value)) {
      out += "i:";
      out += std::to_string(*integer);
    } else {
      out += "s:";
      out += std::get<std::string>(value);
    }
  }
  return out;
}

bool searchBindingLess(const SearchBinding &lhs, const SearchBinding &rhs) {
  std::string left = canonicalSearchBindingString(lhs);
  std::string right = canonicalSearchBindingString(rhs);
  if (left != right)
    return left < right;
  return lhs.candidateId < rhs.candidateId;
}

} // namespace mlir::llk::mapping
