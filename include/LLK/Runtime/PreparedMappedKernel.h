#ifndef LLK_RUNTIME_PREPAREDMAPPEDKERNEL_H
#define LLK_RUNTIME_PREPAREDMAPPEDKERNEL_H

#include "LLK/Runtime/KernelAbi.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#include <string>

namespace llk {
struct PreparedMappedKernel {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  KernelAbi abi;
  std::string entrySymbol;
};
} // namespace llk

#endif // LLK_RUNTIME_PREPAREDMAPPEDKERNEL_H
