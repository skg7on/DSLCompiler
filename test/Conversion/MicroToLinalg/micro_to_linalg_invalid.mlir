// RUN: llk-opt --verify-diagnostics --micro-to-linalg %s

// An op the bridge cannot lower yet must fail the conversion, not be silently
// left behind: the pass marks the whole micro dialect illegal, so anything
// without a pattern is a hard error. This chunk pins that for `micro.mma`,
// whose Linalg form (`linalg.matmul` over the fragments) is a later slice.
//
// The RUN line above is documentation only; the MicroToLinalgInvalid CTest entry
// supplies the same invocation.

module {
  micro.kernel @unsupported {
    %a = tensor.empty() : tensor<8x8xf32>
    %b = tensor.empty() : tensor<8x8xf32>
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %acc = micro.tile_alloc : !micro.tile<8x8xf32, memory = #micro.memory<acc>>
    // expected-error @+1 {{failed to legalize operation 'micro.mma'}}
    %r = micro.mma %ta, %tb, %acc {shape = array<i64: 8, 8, 8>, input = #micro.dtype<f32>, accumulator = #micro.dtype<f32>} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<acc>> -> !micro.tile<8x8xf32, memory = #micro.memory<acc>>
    micro.yield
  }
}
