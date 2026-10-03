//===- Candidate.cpp - Candidate bindings and stable ids ------------------===//
//
// Part of the M12 tuning core (issue #49). See Candidate.h.
//
//===----------------------------------------------------------------------===//

#include "LLK/Perf/Candidate.h"

#include <cinttypes>
#include <cstdio>

namespace mlir::llk::perf {

std::optional<int64_t> Candidate::integer(llvm::StringRef name) const {
  auto it = values.find(name.str());
  if (it == values.end())
    return std::nullopt;
  return it->second;
}

std::optional<llvm::StringRef> Candidate::symbol(llvm::StringRef name) const {
  auto it = symbolicValues.find(name.str());
  if (it == symbolicValues.end())
    return std::nullopt;
  return llvm::StringRef(it->second);
}

namespace {

/// FNV-1a over the bytes of `text`, continuing an existing hash.
uint64_t fnv1a(llvm::StringRef text, uint64_t hash) {
  for (char c : text) {
    hash ^= static_cast<unsigned char>(c);
    hash *= 1099511628211ULL;
  }
  return hash;
}

} // namespace

std::string
computeCandidateId(const std::map<std::string, int64_t> &values,
                   const std::map<std::string, std::string> &symbolicValues) {
  // FNV-1a, chosen because it is a fixed, well-specified algorithm: the same
  // bindings produce the same id on any host and any toolchain. std::map
  // iterates in key order, so the byte stream is canonical without sorting.
  uint64_t hash = 14695981039346656037ULL;

  for (const auto &[name, value] : values)
    hash = fnv1a(name + "=" + std::to_string(value) + ";", hash);
  hash = fnv1a("|", hash);
  for (const auto &[name, value] : symbolicValues)
    hash = fnv1a(name + "=" + value + ";", hash);

  char buffer[17];
  std::snprintf(buffer, sizeof(buffer), "%016" PRIx64, hash);
  return std::string("candidate_") + buffer;
}

} // namespace mlir::llk::perf
