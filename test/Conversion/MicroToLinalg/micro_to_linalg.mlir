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
// The kernel ABI is the signature the kernel declares: its block arguments are
// the inputs a caller passes and the values `micro.yield` carries are the
// results it receives. Nothing is inferred from the body, so an internal
// `tensor.empty` can never be mistaken for a caller's buffer. A tile's memory
// space, layout, and owner are placement facts the target owns and are dropped
// here, so a tile becomes a plain tensor of the same shape and element type.

// CHECK-LABEL: func.func @elementwise(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>)
module {
  micro.kernel @elementwise(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>) {
    // A logical view allocates nothing and copies nothing: it is its source.
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    // An elementwise fragment materializes a tensor of its own.
    // CHECK: %[[OUT:.*]] = tensor.empty() : tensor<8x8xf32>
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%arg0, %arg1 : tensor<8x8xf32>, tensor<8x8xf32>)
    // CHECK-SAME: outs(%[[OUT]] : tensor<8x8xf32>)
    // CHECK: arith.addf
    %r = micro.vector "add" %ta, %tb : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<acc>>
    micro.yield
  }

  // A kernel that yields produces a result: the yielded value is what the
  // caller receives, and the declared input is what it passes in.
  // CHECK-LABEL: func.func @staged(%arg0: tensor<8x8xf32>) -> tensor<8x8xf32>
  micro.kernel @staged(%ext: tensor<8x8xf32>) -> tensor<8x8xf32> {
    // The async movement is synchronous once lowered: a copy, with the token
    // and the wait erased.
    // CHECK: %[[MOVED:.*]] = linalg.generic
    // CHECK-SAME: ins(%arg0 : tensor<8x8xf32>)
    %t, %tok = micro.async_copy %ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    micro.wait %tok
    %tv = micro.tile_view %t {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    // A representation change materializes the same way: a copy, since the
    // physical layouts are dropped with the tile's placement facts.
    // CHECK: %[[CONV:.*]] = linalg.generic
    // CHECK-SAME: ins(%[[MOVED]] : tensor<8x8xf32>)
    %conv = micro.transform %tv {src_map = affine_map<(d0, d1) -> (d0, d1)>, dst_map = affine_map<(d0, d1) -> (d0, d1 floordiv 8, d1 mod 8)>} : !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<acc>>
    // The yield becomes the return: the value leaves the kernel here.
    // CHECK: return %{{.*}} : tensor<8x8xf32>
    micro.yield %conv : !micro.tile<8x8xf32, memory = #micro.memory<acc>>
  }

  // An explicit zero offset is still the whole source. The offsets are real SSA
  // values here, not absent -- but each is a constant zero, so the view starts
  // at the origin and stays the identity: no slice is needed or emitted.
  // CHECK-LABEL: func.func @explicit_zero_offsets(%arg0: tensor<8x8xf32>)
  micro.kernel @explicit_zero_offsets(%a: tensor<8x8xf32>) {
    %z = arith.constant 0 : index
    %ta = micro.tile_view %a[%z, %z] {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    // The view is its source: the add reads %arg0 itself twice, not a copy of
    // it. A dropped-but-nonzero offset would read other elements instead.
    // CHECK: linalg.generic
    // CHECK-SAME: ins(%arg0, %arg0 : tensor<8x8xf32>, tensor<8x8xf32>)
    %r = micro.vector "add" %ta, %ta : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<acc>>
    micro.yield
  }
}

// CHECK-NOT: micro.
