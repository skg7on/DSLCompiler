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

} // namespace mlir::micro

#endif // LLK_DIALECT_MICRO_MICROENUMS_H
