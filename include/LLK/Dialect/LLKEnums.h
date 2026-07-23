//===- LLKEnums.h - LLK dialect enums -------------------------------------===//
//
// Manual enum definitions for LLK dialect attributes.
// These are hand-written because the CMake tablegen configuration currently
// does not include -gen-enum-decls/-gen-enum-defs generators (the enum
// stringification/symbolization functions are needed by the generated
// attribute parser in LLKAttributes.cpp.inc).
//
//===----------------------------------------------------------------------===//

#ifndef LLK_DIALECT_LLKENUMS_H
#define LLK_DIALECT_LLKENUMS_H

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"

#include <optional>

namespace mlir::llk {

// Activation function enum
enum Activation : uint32_t {
  silu = 0,
};

// Math approximation mode enum
enum MathMode : uint32_t {
  strict = 0,
  bounded_fast = 1,
  unsafe_fast = 2,
};

// Stringification / symbolization for Activation
inline llvm::StringRef stringifyActivation(Activation val) {
  switch (val) {
  case silu:
    return "silu";
  }
  return "";
}

inline std::optional<Activation> symbolizeActivation(llvm::StringRef str) {
  return llvm::StringSwitch<std::optional<Activation>>(str)
      .Case("silu", silu)
      .Default(std::nullopt);
}

// Stringification / symbolization for MathMode
inline llvm::StringRef stringifyMathMode(MathMode val) {
  switch (val) {
  case strict:
    return "strict";
  case bounded_fast:
    return "bounded_fast";
  case unsafe_fast:
    return "unsafe_fast";
  }
  return "";
}

inline std::optional<MathMode> symbolizeMathMode(llvm::StringRef str) {
  return llvm::StringSwitch<std::optional<MathMode>>(str)
      .Case("strict", strict)
      .Case("bounded_fast", bounded_fast)
      .Case("unsafe_fast", unsafe_fast)
      .Default(std::nullopt);
}

// Softmax algorithm enum
enum SoftmaxMode : uint32_t {
  online = 0,
};

// Stringification / symbolization for SoftmaxMode
inline llvm::StringRef stringifySoftmaxMode(SoftmaxMode val) {
  switch (val) {
  case online:
    return "online";
  }
  return "";
}

inline std::optional<SoftmaxMode> symbolizeSoftmaxMode(llvm::StringRef str) {
  return llvm::StringSwitch<std::optional<SoftmaxMode>>(str)
      .Case("online", online)
      .Default(std::nullopt);
}

} // namespace mlir::llk

#endif // LLK_DIALECT_LLKENUMS_H
