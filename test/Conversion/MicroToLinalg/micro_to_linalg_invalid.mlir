// RUN: llk-opt --verify-diagnostics --split-input-file --micro-to-linalg %s

// An op the bridge cannot lower yet must fail the conversion, not be silently
// left behind: the pass marks the whole micro dialect illegal, so anything
// without a pattern is a hard error.
//
// Each chunk is one such refusal, and a chunk gets exactly one because the
// conversion aborts at the first illegal op it meets -- the "failed to
// legalize" error below is the whole conversion for that chunk.
//
// The RUN line above is documentation only; the MicroToLinalgInvalid CTest entry
// supplies the same invocation.

// This chunk pins that for `micro.mma`, whose Linalg form
// (`linalg.matmul` over the fragments) is a later slice.
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

// -----

// A view that starts anywhere other than the origin must not be lowered as its
// source. Equal source/result shapes are not enough: offsets [1, 0] would be
// dropped and the reader would silently see the wrong elements. Slice support
// is a later slice, so this fails loudly until then.
module {
  micro.kernel @offset_window {
    %a = tensor.empty() : tensor<8x8xf32>
    %one = arith.constant 1 : index
    %zero = arith.constant 0 : index
    // expected-error @+1 {{failed to legalize operation 'micro.tile_view'}}
    %ta = micro.tile_view %a[%one, %zero] {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}

// -----

// An offset the pattern cannot fold to a constant is just as unprovable: it
// cannot be assumed zero, so it fails the match rather than reading whatever
// the source happens to hold at an unknown origin.
module {
  func.func private @runtime_offset() -> index
  micro.kernel @dynamic_window {
    %a = tensor.empty() : tensor<8x8xf32>
    %d = func.call @runtime_offset() : () -> index
    // expected-error @+1 {{failed to legalize operation 'micro.tile_view'}}
    %ta = micro.tile_view %a[%d, %d] {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
