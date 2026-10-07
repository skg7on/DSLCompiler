//===- MicroEnums.h - Micro dialect enums
//----------------------------------===//
//
// Manual enum definitions for Micro dialect attributes.
// These are hand-written because the CMake tablegen configuration currently
// does not include -gen-enum-decls/-gen-enum-defs generators (the enum
// stringification/symbolization functions are needed by the generated
// attribute parser in MicroAttributes.cpp.inc).
//
//===----------------------------------------------------------------------===//

#ifndef LLK_DIALECT_MICRO_MICROENUMS_H
#define LLK_DIALECT_MICRO_MICROENUMS_H

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSwitch.h"

#include <optional>

namespace mlir::micro {

//===----------------------------------------------------------------------===//
// Memory space enum
//===----------------------------------------------------------------------===//

enum MemorySpace : uint32_t {
  dram = 0,
  l2 = 1,
  sram = 2,
  rf = 3,
  acc = 4,
  scratch = 5,
};

inline llvm::StringRef stringifyMemorySpace(MemorySpace val) {
  switch (val) {
  case dram:
    return "dram";
  case l2:
    return "l2";
  case sram:
    return "sram";
  case rf:
    return "rf";
  case acc:
    return "acc";
  case scratch:
    return "scratch";
  }
  return "";
}

inline std::optional<MemorySpace> symbolizeMemorySpace(llvm::StringRef str) {
  return llvm::StringSwitch<std::optional<MemorySpace>>(str)
      .Case("dram", dram)
      .Case("l2", l2)
      .Case("sram", sram)
      .Case("rf", rf)
      .Case("acc", acc)
      .Case("scratch", scratch)
      .Default(std::nullopt);
}

// `#micro.map` has no enumerators of its own. A spatial loop names its axis as
// a symbol (`#micro.map<worker>`), exactly as a tile names its owner, and the
// owning target resolves that spelling through its machine model. See the
// `Micro_MappingTargetAttr` definition in MicroDialect.td.

//===----------------------------------------------------------------------===//
// DType enum
//===----------------------------------------------------------------------===//

enum DType : uint32_t {
  f32 = 0,
  f16 = 1,
  bf16 = 2,
  i32 = 3,
  i8 = 4,
};

inline llvm::StringRef stringifyDType(DType val) {
  switch (val) {
  case f32:
    return "f32";
  case f16:
    return "f16";
  case bf16:
    return "bf16";
  case i32:
    return "i32";
  case i8:
    return "i8";
  }
  return "";
}

inline std::optional<DType> symbolizeDType(llvm::StringRef str) {
  return llvm::StringSwitch<std::optional<DType>>(str)
      .Case("f32", f32)
      .Case("f16", f16)
      .Case("bf16", bf16)
      .Case("i32", i32)
      .Case("i8", i8)
      .Default(std::nullopt);
}

//===----------------------------------------------------------------------===//
// Layout kind enum (for #micro.layout)
//===----------------------------------------------------------------------===//

enum class LayoutKind : uint32_t {
  row_major = 0,
  col_major = 1,
  blocked = 2,
  vectorized = 3,
  swizzled = 4,
};

inline llvm::StringRef stringifyLayoutKind(LayoutKind val) {
  switch (val) {
  case LayoutKind::row_major:
    return "row_major";
  case LayoutKind::col_major:
    return "col_major";
  case LayoutKind::blocked:
    return "blocked";
  case LayoutKind::vectorized:
    return "vectorized";
  case LayoutKind::swizzled:
    return "swizzled";
  }
  return "";
}

inline std::optional<LayoutKind> symbolizeLayoutKind(llvm::StringRef str) {
  return llvm::StringSwitch<std::optional<LayoutKind>>(str)
      .Case("row_major", LayoutKind::row_major)
      .Case("col_major", LayoutKind::col_major)
      .Case("blocked", LayoutKind::blocked)
      .Case("vectorized", LayoutKind::vectorized)
      .Case("swizzled", LayoutKind::swizzled)
      .Default(std::nullopt);
}

//===----------------------------------------------------------------------===//
// Gather semantics (for micro.gather)
//===----------------------------------------------------------------------===//
//
// The explicit combination a multi-producer gather performs. It is never
// inferred from topology: a gather with several inputs and no declared kind is
// not arithmetic, so the op requires one. The string table is hand-written for
// the same reason every other enum here is (the CMake tablegen config does not
// run -gen-enum-decls/-gen-enum-defs).

enum class GatherKind : uint32_t {
  sum = 0,
  max = 1,
  concat = 2,
};

inline llvm::StringRef stringifyGatherKind(GatherKind val) {
  switch (val) {
  case GatherKind::sum:
    return "sum";
  case GatherKind::max:
    return "max";
  case GatherKind::concat:
    return "concat";
  }
  return "";
}

inline std::optional<GatherKind> symbolizeGatherKind(llvm::StringRef str) {
  return llvm::StringSwitch<std::optional<GatherKind>>(str)
      .Case("sum", GatherKind::sum)
      .Case("max", GatherKind::max)
      .Case("concat", GatherKind::concat)
      .Default(std::nullopt);
}

//===----------------------------------------------------------------------===//
// Barrier scope (for micro.barrier)
//===----------------------------------------------------------------------===//
//
// The synchronization domain a barrier covers. `executor_group` is the generic
// executor group the dialect can express; a plan that needs a scope outside it
// has no Micro spelling and must be rejected rather than silently widened.

enum class BarrierScope : uint32_t {
  executor_group = 0,
};

inline llvm::StringRef stringifyBarrierScope(BarrierScope val) {
  switch (val) {
  case BarrierScope::executor_group:
    return "executor_group";
  }
  return "";
}

inline std::optional<BarrierScope> symbolizeBarrierScope(llvm::StringRef str) {
  return llvm::StringSwitch<std::optional<BarrierScope>>(str)
      .Case("executor_group", BarrierScope::executor_group)
      .Default(std::nullopt);
}

//===----------------------------------------------------------------------===//
// Owner enum (for #micro.owner)
//===----------------------------------------------------------------------===//

// The abstract classes a tile's owner belongs to: the whole of the owner
// vocabulary the canonical dialect has. A *concrete* executor spelling --
// whatever a target calls its execution units -- is that target's word for one
// of these, and belongs in that target's machine model, which maps its own
// labels onto them. Enumerating them here is what §5.4 forbids: it would put
// one backend's hierarchy into the IR every backend shares.
//
// Scoped, because several of these names also appear as unscoped enumerators
// elsewhere in this namespace.
enum class Owner : uint32_t {
  group = 0,    // a cooperating group of workers
  worker = 1,   // one worker
  vector = 2,   // the vector engine
  matrix = 3,   // the matrix engine
  transfer = 4, // the data-movement engine
};

inline llvm::StringRef stringifyOwner(Owner val) {
  switch (val) {
  case Owner::group:
    return "group";
  case Owner::worker:
    return "worker";
  case Owner::vector:
    return "vector";
  case Owner::matrix:
    return "matrix";
  case Owner::transfer:
    return "transfer";
  }
  return "";
}

/// The abstract class a spelling names *directly*. A profile's own label is not
/// one of these: it reaches a class through the machine model's alias table,
/// which is where a target's vocabulary lives.
inline std::optional<Owner> symbolizeOwner(llvm::StringRef str) {
  return llvm::StringSwitch<std::optional<Owner>>(str)
      .Case("group", Owner::group)
      .Case("worker", Owner::worker)
      .Case("vector", Owner::vector)
      .Case("matrix", Owner::matrix)
      .Case("transfer", Owner::transfer)
      .Default(std::nullopt);
}

} // namespace mlir::micro

#endif // LLK_DIALECT_MICRO_MICROENUMS_H
