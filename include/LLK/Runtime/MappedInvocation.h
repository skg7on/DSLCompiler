#ifndef LLK_RUNTIME_MAPPEDINVOCATION_H
#define LLK_RUNTIME_MAPPEDINVOCATION_H
#include "LLK/Runtime/JitCache.h"
#include "LLK/Runtime/KernelAbi.h"
#include "llvm/ADT/ArrayRef.h"
namespace llk {
enum class InvocationElementType { F32, BF16, F16, I32, I8 };
struct InvocationBuffer2D {
  MemRef2D descriptor{};
  // Declared by the caller; this tag cannot inspect the backing bytes.
  InvocationElementType elementType = InvocationElementType::F32;
  uint64_t allocationBytes = 0; // bytes from descriptor.allocated
};
llvm::Error
validateMappedInvocation(const KernelAbi &abi,
                         llvm::ArrayRef<InvocationBuffer2D> inputs,
                         llvm::ArrayRef<InvocationBuffer2D> outputs);
} // namespace llk
#endif
