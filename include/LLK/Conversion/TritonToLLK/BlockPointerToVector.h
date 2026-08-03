//===- BlockPointerToVector.h - block ptr → vector.transfer -------*- C++
//-*-===//
//
// Pass header for lowering Triton block pointers and load/store to
// vector.transfer_read/transfer_write operations.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_CONVERSION_TRITONTOLLK_BLOCKPOINTERTOVECTOR_H
#define LLK_CONVERSION_TRITONTOLLK_BLOCKPOINTERTOVECTOR_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace llk {
std::unique_ptr<Pass> createBlockPointerToVectorPass();
} // namespace llk
} // namespace mlir

#endif // LLK_CONVERSION_TRITONTOLLK_BLOCKPOINTERTOVECTOR_H
