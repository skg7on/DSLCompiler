// RUN: llk-opt --verify-diagnostics --split-input-file --micro-to-linalg %s

// A construct the bridge cannot lower must fail the conversion, not be silently
// left behind: the pass marks the whole micro dialect illegal, so anything
// without a pattern is a hard error -- and an op whose pattern refuses is one
// too.
//
// Each chunk is one such refusal, and a chunk gets exactly one because the
// conversion aborts at the first illegal op it meets -- the error below is the
// whole conversion for that chunk.
//
// The RUN line above is documentation only; the MicroToLinalgInvalid CTest entry
// supplies the same invocation.

// `micro.store` is a logical write to an external tensor. The reference bridge
// covers what `--llk-to-micro` and the mapping materializer emit; this is not
// one of them, so it is refused rather than approximated.
module {
  micro.kernel @unsupported(%a: tensor<8x8xf32>) {
    // expected-error @+1 {{failed to legalize operation 'micro.store'}}
    micro.store %a {src_memory = #micro.memory<sram>, dst_memory = #micro.memory<dram>} : tensor<8x8xf32>
    micro.yield
  }
}

// -----

// A window that leaves its source is refused. Lowering it to a slice would read
// past the tensor, and clipping it would silently compute on elements the
// program did not ask for; the offsets [1, 0] with an 8-wide window over an
// 8-wide dimension do exactly that.
module {
  micro.kernel @out_of_bounds_window(%a: tensor<8x8xf32>) {
    %one = arith.constant 1 : index
    %zero = arith.constant 0 : index
    // expected-error @+1 {{failed to legalize operation 'micro.tile_view'}}
    %ta = micro.tile_view %a[%one, %zero] {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}

// -----

// A terminal `micro.tile_store` writes to memory and produces nothing. The
// reference form has no value to put that write in, so it is refused rather
// than dropped -- the written data would be observable to nobody, which is
// exactly what the kernel's result contract exists to prevent.
module {
  micro.kernel @terminal_store(%a: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<acc>>
    // expected-error @+1 {{failed to legalize operation 'micro.tile_store'}}
    micro.tile_store %ta {dst_memory = #micro.memory<dram>} : !micro.tile<8x8xf32, memory = #micro.memory<acc>>
    micro.yield %ta : !micro.tile<8x8xf32, memory = #micro.memory<acc>>
  }
}

// -----

// A kernel with no declared signature has no ABI to lower. Inferring one from
// the entry tensors -- the pre-contract behaviour -- is exactly the ambiguity
// the explicit signature removed, so the pass refuses instead of guessing.
module {
  // expected-error @+1 {{kernel has no explicit signature}}
  micro.kernel @legacy {
    %a = tensor.empty() : tensor<8x8xf32>
    micro.yield
  }
}
