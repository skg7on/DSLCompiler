//===- KernelAbi.h - Kernel ABI descriptors and validation ------*- C++ -*-===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The callable contract of a compiled kernel.
//
// Version 1 describes a rank-two, compact row-major kernel with positive
// static extents: ordered read-only inputs followed by ordered caller-owned
// outputs. Its encoding, validation and hash are frozen.
//
// Version 2 keeps the same descriptor transport but records the symbolic
// workload contract and the explicit argument/result port bindings. A port
// dimension is then either a nonnegative constant or a named symbol, and a
// signature argument may be a caller-owned output destination rather than a
// read-only input. Dynamic extents are bound per invocation from the
// descriptors, before any code runs; a zero extent is a genuinely empty
// invocation, not a malformed descriptor.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_RUNTIME_KERNELABI_H
#define LLK_RUNTIME_KERNELABI_H
#include "LLK/Shape/ShapeContract.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <map>
#include <string>
#include <vector>
namespace llk {
inline constexpr uint32_t kMappedAbiVersionV1 = 1;
inline constexpr uint32_t kMappedAbiVersionV2 = 2;
/// The version a caller that names no version gets. Version 1 stays readable
/// and keeps its identity, so it remains the default for every existing
/// construction site.
inline constexpr uint32_t kMappedAbiVersion = kMappedAbiVersionV1;
inline constexpr unsigned kMappedMaxDescriptors = 12;

/// Which side of the kernel signature a port describes.
enum class AbiPortRole { Input, Output };

/// One signature argument bound to a contract port, in signature order. The
/// role selects the port list; `port` is an index into that list.
struct KernelAbiArgument {
  AbiPortRole role = AbiPortRole::Input;
  uint32_t port = 0;
};

// v1: rank-two compact row-major descriptors, offset zero; read-only inputs,
// disjoint caller-owned outputs. Port order is part of the contract.
// v2: a port dimension is a nonnegative constant or a declared symbol, an
// output port is bound by a caller-owned destination argument, and the symbol
// table records the declared relationships between ports.
struct KernelAbi {
  struct Port {
    /// Version 1's concrete extents. Version 2 leaves this empty unless the
    /// caller constructed the port positionally, in which case `dimensions`
    /// takes precedence when it is present.
    std::vector<int64_t> shape;
    std::string elementType;
    /// Version 2's symbolic extents: a constant or a named symbol per axis.
    std::vector<::mlir::llk::shape::Dimension> dimensions{};
  };
  /// Ordered input ports. Version 2 orders them by contract declaration, not
  /// by signature position; `arguments` carries the signature order.
  std::vector<Port> inputs;
  /// Ordered output ports, all caller-owned in version 2.
  std::vector<Port> outputs;
  uint32_t version = kMappedAbiVersion;
  /// Version 2: one entry per callable signature argument, in order.
  std::vector<KernelAbiArgument> arguments;
  /// Version 2: one output-port index per yielded result, in order. After
  /// output-destination lowering a callable kernel yields nothing, so this
  /// records the source contract rather than a return value.
  std::vector<uint32_t> results;
  /// Version 2: the declared constraints of every symbol the ports use.
  std::map<std::string, ::mlir::llk::shape::SymbolConstraint> symbols;
};
llvm::Error validateKernelAbi(const KernelAbi &abi);
uint64_t computeKernelAbiHash(const KernelAbi &abi);

/// True when `abi` describes a version 2 artifact. Version 1 keeps its exact
/// validation and hash, so this is the one place the two encodings split.
inline bool isSymbolicKernelAbi(const KernelAbi &abi) {
  return abi.version == kMappedAbiVersionV2;
}
} // namespace llk
#endif
