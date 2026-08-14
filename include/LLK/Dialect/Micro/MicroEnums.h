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

//===----------------------------------------------------------------------===//
// Mapping target enum
//===----------------------------------------------------------------------===//

enum MappingTarget : uint32_t {
  cluster_x = 0,
  cluster_y = 1,
  core_x = 2,
  core_y = 3,
  pe_x = 4,
  pe_y = 5,
  lane = 6,
  worker = 7,
  matrix_engine = 8,
  vector_engine = 9,
  dma = 10,
};

inline llvm::StringRef stringifyMappingTarget(MappingTarget val) {
  switch (val) {
  case cluster_x:
    return "cluster_x";
  case cluster_y:
    return "cluster_y";
  case core_x:
    return "core_x";
  case core_y:
    return "core_y";
  case pe_x:
    return "pe_x";
  case pe_y:
    return "pe_y";
  case lane:
    return "lane";
  case worker:
    return "worker";
  case matrix_engine:
    return "matrix_engine";
  case vector_engine:
    return "vector_engine";
  case dma:
    return "dma";
  }
  return "";
}

inline std::optional<MappingTarget>
symbolizeMappingTarget(llvm::StringRef str) {
  return llvm::StringSwitch<std::optional<MappingTarget>>(str)
      .Case("cluster_x", cluster_x)
      .Case("cluster_y", cluster_y)
      .Case("core_x", core_x)
      .Case("core_y", core_y)
      .Case("pe_x", pe_x)
      .Case("pe_y", pe_y)
      .Case("lane", lane)
      .Case("worker", worker)
      .Case("matrix_engine", matrix_engine)
      .Case("vector_engine", vector_engine)
      .Case("dma", dma)
      .Default(std::nullopt);
}

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
// Owner enum (for #micro.owner)
//===----------------------------------------------------------------------===//

// Scoped enum: several owner names (lane, worker, matrix_engine, vector_engine,
// dma) also appear as unscoped MappingTarget enumerators in this namespace.
enum class Owner : uint32_t {
  cluster = 0,
  core = 1,
  warp = 2,
  wave = 3,
  subgroup = 4,
  pe_group = 5,
  pe = 6,
  lane = 7,
  worker = 8,
  matrix_engine = 9,
  vector_engine = 10,
  dma = 11,
};

inline llvm::StringRef stringifyOwner(Owner val) {
  switch (val) {
  case Owner::cluster:
    return "cluster";
  case Owner::core:
    return "core";
  case Owner::warp:
    return "warp";
  case Owner::wave:
    return "wave";
  case Owner::subgroup:
    return "subgroup";
  case Owner::pe_group:
    return "pe_group";
  case Owner::pe:
    return "pe";
  case Owner::lane:
    return "lane";
  case Owner::worker:
    return "worker";
  case Owner::matrix_engine:
    return "matrix_engine";
  case Owner::vector_engine:
    return "vector_engine";
  case Owner::dma:
    return "dma";
  }
  return "";
}

inline std::optional<Owner> symbolizeOwner(llvm::StringRef str) {
  return llvm::StringSwitch<std::optional<Owner>>(str)
      .Case("cluster", Owner::cluster)
      .Case("core", Owner::core)
      .Case("warp", Owner::warp)
      .Case("wave", Owner::wave)
      .Case("subgroup", Owner::subgroup)
      .Case("pe_group", Owner::pe_group)
      .Case("pe", Owner::pe)
      .Case("lane", Owner::lane)
      .Case("worker", Owner::worker)
      .Case("matrix_engine", Owner::matrix_engine)
      .Case("vector_engine", Owner::vector_engine)
      .Case("dma", Owner::dma)
      .Default(std::nullopt);
}

} // namespace mlir::micro

#endif // LLK_DIALECT_MICRO_MICROENUMS_H
