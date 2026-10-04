// RUN: llk-opt --micro-to-linalg %s | FileCheck %s

// The bridge from canonical Micro-IR to the backend pipeline. `micro` was a
// sink: nothing lowered it to anything a backend consumes, so a mapped or
// exported `micro.kernel` could not reach the LLVM/JIT path. After this pass a
// kernel is an ordinary `func.func` over tensors and Linalg, which is exactly
// what `llk-compile` already tiles, vectorizes, buffers, and JITs.
//
// The output stays in *tensor* land on purpose: an already-bufferized (memref)
// module leaves `llk-compile` with an unrealized `!llvm.array<4 x vector<8xf32>>`
// cast it cannot translate, while the tensor form lowers cleanly.
//
// The RUN line above is documentation only: this project has no lit runner, and
// `//` lines are comments. The real invocation is the MicroToLinalg CTest entry
// in CMakeLists.txt.
//
// What the pass drops is stated at the top of the pass: a tile's memory space,
// layout, and owner are placement facts the target owns, so a tile becomes a
// plain tensor of the same shape and element type. A kernel has no arguments
// and no results, so its entry values -- the `tensor.empty` ops it reads --
// stay tensors, and its result values are unused. Wiring real inputs and
// outputs is the kernel-ABI slice, and is what a harness needs to observe a
// result.

// CHECK-LABEL: func.func @elementwise()
module {
  micro.kernel @elementwise {
    // CHECK: %[[A:.*]] = tensor.empty() : tensor<8x8xf32>
    // CHECK: %[[B:.*]] = tensor.empty() : tensor<8x8xf32>
    %a = tensor.empty() : tensor<8x8xf32>
    %b = tensor.empty() : tensor<8x8xf32>
    // A logical view allocates nothing and copies nothing: it is its source.
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    // An elementwise fragment materializes a tensor of its own.
    // CHECK: %[[OUT:.*]] = tensor.empty() : tensor<8x8xf32>
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%[[A]], %[[B]] : tensor<8x8xf32>, tensor<8x8xf32>)
    // CHECK-SAME: outs(%[[OUT]] : tensor<8x8xf32>)
    // CHECK: arith.addf
    %r = micro.vector "add" %ta, %tb : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<acc>>
    micro.yield
  }

  // A movement and a representation change both materialize as copies.
  // CHECK-LABEL: func.func @staged()
  micro.kernel @staged {
    // CHECK: %[[SRC:.*]] = tensor.empty() : tensor<8x8xf32>
    %ext = tensor.empty() : tensor<8x8xf32>
    // The async movement is synchronous once lowered: a copy, with the token
    // and the wait erased.
    // CHECK: %[[MOVED:.*]] = linalg.generic
    // CHECK-SAME: ins(%[[SRC]] : tensor<8x8xf32>)
    // CHECK: linalg.yield
    %t, %tok = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    micro.wait %tok
    // CHECK: %[[CONV:.*]] = linalg.generic
    // CHECK-SAME: ins(%[[MOVED]] : tensor<8x8xf32>)
    %conv = micro.transform %t {src_map = affine_map<(d0, d1) -> (d0, d1)>, dst_map = affine_map<(d0, d1) -> (d0, d1 floordiv 8, d1 mod 8)>} : tensor<8x8xf32> -> tensor<8x8xf32>
    micro.yield
  }
}

// CHECK-NOT: micro.
