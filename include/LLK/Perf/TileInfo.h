//===- TileInfo.h - Operation-independent tile description ------*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLK_PERF_TILEINFO_H
#define LLK_PERF_TILEINFO_H
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include <cstdint>
#include <string>
namespace mlir::llk::perf {
struct TileInfo {
  bool isTile = false;
  llvm::SmallVector<int64_t, 4> shape;
  std::string dtype;
  std::string layout;
  std::string memory;
  std::string owner;

  /// Bytes of one element of `dtype`, or 0 for a dtype the micro dialect does
  /// not name.
  static unsigned dtypeBytes(llvm::StringRef dtype) {
    return llvm::StringSwitch<unsigned>(dtype)
        .Case("f32", 4)
        .Case("i32", 4)
        .Case("f16", 2)
        .Case("bf16", 2)
        .Case("i8", 1)
        .Default(0);
  }

  /// A dynamic extent makes the element count, and therefore the byte count,
  /// unknown. Concrete micro kernels are expected to have static tiles.
  bool hasDynamicShape() const {
    for (int64_t dim : shape)
      if (ShapedType::isDynamic(dim))
        return true;
    return false;
  }

  uint64_t elements() const {
    if (shape.empty())
      return 0;
    uint64_t count = 1;
    for (int64_t dim : shape) {
      if (ShapedType::isDynamic(dim) || dim <= 0)
        return 0;
      count *= static_cast<uint64_t>(dim);
    }
    return count;
  }

  uint64_t bytes() const {
    unsigned width = dtypeBytes(dtype);
    return width == 0 ? 0 : elements() * width;
  }
};

TileInfo describeType(mlir::Type type);
/// `16x16x32`, with `?` for a dynamic extent.
std::string shapeString(llvm::ArrayRef<int64_t> shape);
} // namespace mlir::llk::perf
#endif
