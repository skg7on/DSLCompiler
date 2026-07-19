//===- JitCache.h - JIT cache for compiled LLK kernels -------------------===//
//
// Single-level ORC LLJIT cache that compiles MLIR modules on demand.
// Stable C ABI structs for kernel inputs/outputs.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_RUNTIME_JITCACHE_H
#define LLK_RUNTIME_JITCACHE_H

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <shared_mutex>

/// Stable C ABI: 2D tensor descriptor used to construct kernel inputs/outputs.
/// Matches the memory layout expected by the ORC JIT entry point.
struct Tensor2D {
    void* data;
    int64_t dim0;
    int64_t dim1;
    int64_t stride0;
    int64_t stride1;
};

/// MLIR-compatible 2D memref descriptor used as the JIT kernel calling ABI.
///
/// Layout must match the LLVM struct produced by MLIR's LLVM lowering of
/// memref<MxNxT>: {T*, T*, i64, [2 x i64], [2 x i64]}.
/// sizeof(MemRef2D) == 56 bytes on 64-bit systems.
struct MemRef2D {
    void* allocated;
    void* aligned;
    int64_t offset;
    int64_t size0;
    int64_t size1;
    int64_t stride0;
    int64_t stride1;
};

/// Helper: populate a MemRef2D from a Tensor2D that holds row-major data.
inline MemRef2D makeMemRef2D(const Tensor2D& t) {
    return MemRef2D{t.data, t.data, 0, t.dim0, t.dim1, t.stride0, t.stride1};
}

/// Stable C ABI: scratch and context pointer for compiled kernels.
struct KernelContext {
    void* scratch;
    size_t scratch_size;
    // ThreadPool* thread_pool;  // added in Milestone 4
};

namespace llk {

/// Thread-safe JIT cache backed by ORC LLJIT.
///
/// Caches compiled kernel function pointers keyed by a user-supplied
/// cache key (e.g., a hash of the MLIR input + lowering options).
/// On cache miss, lowers the MLIR module to LLVM IR and JIT-compiles it.
class JitCache {
public:
    /// Five MemRef2D arguments: x, wg, wu, init, result.
    /// Matches the MLIR function signature after OneShotBufferize:
    ///   func.func @llk_swiglu(
    ///       memref<MxKxbf16>, memref<KxNxbf16>, memref<KxNxbf16>,
    ///       memref<MxNxbf16>, memref<MxNxbf16>)
    using KernelFn = void(*)(MemRef2D*, MemRef2D*, MemRef2D*, MemRef2D*,
                              MemRef2D*);

    /// Constructs the JIT cache, creating an ORC LLJIT instance.
    explicit JitCache();
    ~JitCache();

    JitCache(const JitCache&) = delete;
    JitCache& operator=(const JitCache&) = delete;
    JitCache(JitCache&&) = delete;
    JitCache& operator=(JitCache&&) = delete;

    /// Look up or compile. Returns cached function if key matches, otherwise
    /// compiles the provided MLIR module and caches the result.
    llvm::Expected<KernelFn> lookupOrCompile(const std::string& cache_key,
                                              mlir::ModuleOp module);

    void clear();
    size_t size() const;

private:
    std::unique_ptr<llvm::orc::LLJIT> jit_;
    std::unordered_map<std::string, KernelFn> cache_;
    mutable std::shared_mutex mutex_;
};

} // namespace llk

#endif // LLK_RUNTIME_JITCACHE_H
