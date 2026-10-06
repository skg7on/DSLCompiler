// The JIT acceptance fixture for the Micro-to-Linalg bridge. The CTest entry
// MicroToLinalgJit lowers this kernel and pipes the result into `llk-compile`,
// which buffers, vectorizes, and JIT-compiles it; the test passes when that
// reports "Compilation successful".
//
// It is the check that the bridge emits IR the backend pipeline can actually
// finish. That is why the lowering stays in *tensor* land: an already
// bufferized (memref) module leaves `llk-compile` with an unrealized
// `!llvm.array<4 x vector<8xf32>>` cast that LLVM translation rejects, while
// the tensor form tiles, vectorizes, buffers, and translates cleanly.
//
// The kernel exercises the ABI too: its two declared inputs become the
// function's arguments and the yielded value its result, so the JIT-compiled
// symbol takes memref descriptors in and returns one -- the runtime's calling
// convention. The test still asserts "it compiles", not "it produces the right
// numbers": observing a result needs a harness that calls the symbol, which is
// the next step.

module {
  micro.kernel @add(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %ta, %tb : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<acc>>
    micro.yield %r : !micro.tile<8x8xf32, memory = #micro.memory<acc>>
  }
}
