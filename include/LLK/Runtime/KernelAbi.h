#ifndef LLK_RUNTIME_KERNELABI_H
#define LLK_RUNTIME_KERNELABI_H
#include "llvm/Support/Error.h"
#include <cstdint>
#include <string>
#include <vector>
namespace llk {
inline constexpr uint32_t kMappedAbiVersion = 1;
inline constexpr unsigned kMappedMaxDescriptors = 12;
// v1: rank-two compact row-major descriptors, offset zero; read-only inputs,
// disjoint caller-owned outputs. Port order is part of the contract.
struct KernelAbi {
  struct Port {
    std::vector<int64_t> shape;
    std::string elementType;
  };
  std::vector<Port> inputs;
  std::vector<Port> outputs;
  uint32_t version = kMappedAbiVersion;
};
llvm::Error validateKernelAbi(const KernelAbi &abi);
uint64_t computeKernelAbiHash(const KernelAbi &abi);
} // namespace llk
#endif
