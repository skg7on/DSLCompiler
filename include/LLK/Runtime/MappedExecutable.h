//===- MappedExecutable.h - Invoke a compiled micro kernel (issue #67, C4)
//-===//
//
// Part of the runtime (epic #67, stage C4).
//
// A lowered `micro.kernel` reaches machine code through the same ORC pipeline
// the legacy kernels use, but its calling convention is its own. Its operands
// are the memref descriptors its signature declared, and its result is a
// caller-owned buffer rather than a returned aggregate -- MLIR expands a memref
// return into a hidden-pointer convention no C caller can declare, and the
// caller has to own that buffer anyway.
//
// `MappedExecutable` owns the compiled code and calls it with descriptors the
// caller owns. It is the ABI boundary the mapping pipeline was missing: a plan
// that selects a target and a schedule is only tested when the program it
// produced is actually run.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_RUNTIME_MAPPEDEXECUTABLE_H
#define LLK_RUNTIME_MAPPEDEXECUTABLE_H

#include "LLK/Runtime/JitCache.h"
#include "LLK/Runtime/MappedInvocation.h"
#include "LLK/Runtime/MappedJitOptions.h"
#include "LLK/Runtime/PreparedMappedKernel.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace llvm::orc {
class LLJIT;
} // namespace llvm::orc

namespace llk {

/// The contract a compiled kernel was built with: one descriptor per operand,
/// in declared order, and one per result, in declared order.
///
/// It is recorded from the MLIR types *before* they are erased into LLVM
/// pointers, because that is the last point at which shapes and element types
/// are knowable. An invocation that disagrees with it is refused rather than
/// reinterpreted: the generated code addresses buffers as if its own types were
/// the truth, so a descriptor of the wrong shape would read and write memory
/// the caller never offered.

class MappedExecutable;
namespace testing {
struct TestAllocatorHooks;
class MappedExecutableTestFactory;
} // namespace testing

/// Compiles `bufferedModule`'s `entrySymbol` into an executable.
///
/// The module must already be bufferized and loop-lowered -- what this adds is
/// the calling convention, not the lowering. Results become caller-owned output
/// parameters, the entry point is given a C-interface wrapper, and the whole
/// thing runs the same LLVM/ORC stages `JitCache` runs.
///
/// The returned executable owns its JIT and its translated LLVM module, so the
/// module handed in may be destroyed the moment this returns: nothing here
/// keeps a borrowed MLIR operation alive.
llvm::Expected<std::unique_ptr<MappedExecutable>>
createMappedExecutable(mlir::ModuleOp bufferedModule,
                       llvm::StringRef entrySymbol);
llvm::Expected<std::unique_ptr<MappedExecutable>>
createMappedExecutable(PreparedMappedKernel prepared,
                       const MappedJitOptions &options = {});

/// A compiled kernel, callable through descriptor pointers.
class MappedExecutable {
public:
  ~MappedExecutable();

  MappedExecutable(const MappedExecutable &) = delete;
  MappedExecutable &operator=(const MappedExecutable &) = delete;

  /// The ABI this executable was compiled for.
  const KernelAbi &abi() const { return abi_; }
  uint64_t abiHash() const { return computeKernelAbiHash(abi_); }
  llvm::StringRef executionIdentity() const { return executionIdentity_; }

  llvm::Error invoke(llvm::ArrayRef<InvocationBuffer2D> inputs,
                     llvm::ArrayRef<InvocationBuffer2D> outputs);

  /// Invokes the kernel with the descriptors its ABI expects, in order.
  ///
  /// The caller owns every buffer: this takes no ownership, frees nothing, and
  /// writes only through the output descriptors it is handed. Descriptors are
  /// checked against the recorded ABI first, so mismatched metadata is rejected
  /// before machine code runs.
  ///
  /// Legacy entry: checks arity, shape and stride only. Untagged pointers
  /// cannot check element type or backing allocation size.
  [[deprecated("use typed InvocationBuffer2D buffers")]]
  llvm::Error invoke(llvm::ArrayRef<MemRef2D *> inputs,
                     llvm::ArrayRef<MemRef2D *> outputs);

private:
  llvm::Error invokeUncheckedLegacy(llvm::ArrayRef<MemRef2D *> inputs,
                                    llvm::ArrayRef<MemRef2D *> outputs);
  friend llvm::Expected<std::unique_ptr<MappedExecutable>>
  createMappedExecutable(mlir::ModuleOp bufferedModule,
                         llvm::StringRef entrySymbol);
  friend llvm::Expected<std::unique_ptr<MappedExecutable>>
  createMappedExecutable(PreparedMappedKernel prepared,
                         const MappedJitOptions &options);
  friend class testing::MappedExecutableTestFactory;

  static llvm::Expected<std::unique_ptr<MappedExecutable>>
  createWithAllocatorHooks(PreparedMappedKernel prepared,
                           const MappedJitOptions &options,
                           testing::TestAllocatorHooks *hooks);

  MappedExecutable(std::unique_ptr<llvm::orc::LLJIT> jit, void *entry,
                   KernelAbi abi, std::string executionIdentity);

  /// The JIT that owns the compiled code. It outlives every call, which is
  /// what lets an invocation be a plain indirect call.
  std::unique_ptr<llvm::orc::LLJIT> jit_;
  void *entry_ = nullptr;
  KernelAbi abi_;
  std::string executionIdentity_;
};

} // namespace llk

#endif // LLK_RUNTIME_MAPPEDEXECUTABLE_H
