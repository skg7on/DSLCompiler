// The JIT acceptance fixture for the Micro-to-Linalg bridge. The CTest entry
// MicroToLinalgJit lowers this kernel and pipes the result into `llk-compile`,
// which JIT-compiles it; the test passes when that reports "Compilation
// successful".
//
// It is the check that the bridge emits IR the backend pipeline can actually
// finish. That is why the lowering stays in *tensor* land: an already
// bufferized (memref) module leaves `llk-compile` with an unrealized
// `!llvm.array<4 x vector<8xf32>>` cast that LLVM translation rejects, while
// the tensor form tiles, vectorizes, buffers, and translates cleanly.
//
// A kernel has no arguments or results yet, so the compiled function computes
// dead work -- no result is observable until the kernel-ABI slice lands. The
// check is therefore "it compiles", not "it produces the right numbers".

module {
  micro.kernel @add {
    %a = tensor.empty() : tensor<8x8xf32>
    %b = tensor.empty() : tensor<8x8xf32>
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %ta, %tb : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<acc>>
    micro.yield
  }
}
