// RUN: llk-opt --micro-to-linalg %s | FileCheck %s

// The reference lowering of bounded movement and exact reduction tails. Local
// tiles keep their static capacity; only the boundary rectangle is dynamic.
//
// A bounded load is a clipped `tensor.extract_slice` plus tile-local
// `tensor.pad`, so the whole source is never padded. A clipped store is a
// clipped slice of the source inserted at the destination offsets, so no lane
// outside the rectangle is written. An exact-K mma slices both operands to the
// runtime reduction length and feeds the accumulator as the contraction's
// `outs`, so a BK-sized padded tail is never folded into the result.
//
// The RUN line below is documentation only: the project has no lit runner. The
// real invocation is the MicroToLinalgDynamicBoundary CTest entry.

//===----------------------------------------------------------------------===//
// Bounded load: (validM, validK) = (3, 5) into a fixed 8x32 tile
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @bounded_load(%arg0: tensor<8x32xf32>, %arg1: index, %arg2: index, %arg3: index) -> tensor<8x32xf32>
module {
  micro.kernel @bounded_load(%src: tensor<8x32xf32>, %off: index,
                             %validM: index, %validK: index) -> tensor<8x32xf32> {
    // The copied rectangle is the runtime valid extent, not the static tile.
    // CHECK: %[[SLICE:.*]] = tensor.extract_slice %arg0[%arg1, %arg1] [%arg2, %arg3] [1, 1] : tensor<8x32xf32> to tensor<?x?xf32>
    // The complement is filled inside the tile: tile.shape - valid per axis.
    // CHECK-NEXT: %[[C8:.*]] = arith.constant 8 : index
    // CHECK-NEXT: %[[H0:.*]] = arith.subi %[[C8]], %arg2 : index
    // CHECK-NEXT: %[[C32:.*]] = arith.constant 32 : index
    // CHECK-NEXT: %[[H1:.*]] = arith.subi %[[C32]], %arg3 : index
    // The padding value is the op's own typed constant, not an assumed zero.
    // CHECK-NEXT: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK-NEXT: %[[PAD:.*]] = tensor.pad %[[SLICE]] low[0, 0] high[%[[H0]], %[[H1]]]
    // CHECK: tensor.yield %[[CST]] : f32
    // CHECK: } : tensor<?x?xf32> to tensor<8x32xf32>
    %t, %tok = micro.tile_load %src[%off, %off] valid[%validM, %validK] {
        dst_memory = #micro.memory<sram>, padding = 0.0 : f32
      } : tensor<8x32xf32>
        -> !micro.tile<8x32xf32, memory = #micro.memory<sram>>, !micro.async_token
    micro.wait %tok
    // CHECK: return %[[PAD]] : tensor<8x32xf32>
    micro.yield %t : !micro.tile<8x32xf32, memory = #micro.memory<sram>>
  }
}

//===----------------------------------------------------------------------===//
// Clipped store: write 3x7 out of the 8x32 source tile
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @clipped_store(%arg0: tensor<8x32xf32>, %arg1: tensor<16x64xf32>, %arg2: index, %arg3: index, %arg4: index, %arg5: index) -> tensor<16x64xf32>
module {
  micro.kernel @clipped_store(%src: tensor<8x32xf32>, %out: tensor<16x64xf32>,
                              %offM: index, %offN: index, %validM: index,
                              %validN: index) -> tensor<16x64xf32> {
    %ts = micro.tile_view %src {shape = array<i64: 8, 32>} : tensor<8x32xf32> -> !micro.tile<8x32xf32, memory = #micro.memory<acc>>
    %to = micro.tile_view %out {shape = array<i64: 16, 64>} : tensor<16x64xf32> -> !micro.tile<16x64xf32, memory = #micro.memory<dram>>
    // Only the origin rectangle of the source leaves the tile.
    // CHECK: %[[CLIP:.*]] = tensor.extract_slice %arg0[0, 0] [%arg4, %arg5] [1, 1] : tensor<8x32xf32> to tensor<?x?xf32>
    // The destination keeps its static type: the write updates only the
    // rectangle at the offsets and leaves every other lane untouched.
    // CHECK-NEXT: %[[W:.*]] = tensor.insert_slice %[[CLIP]] into %arg1[%arg2, %arg3] [%arg4, %arg5] [1, 1] : tensor<?x?xf32> into tensor<16x64xf32>
    %w = micro.tile_store %ts into %to[%offM, %offN] valid[%validM, %validN] {
        dst_memory = #micro.memory<dram>
      } : !micro.tile<8x32xf32, memory = #micro.memory<acc>>,
          !micro.tile<16x64xf32, memory = #micro.memory<dram>>
        -> !micro.tile<16x64xf32, memory = #micro.memory<dram>>
    // CHECK: return %[[W]] : tensor<16x64xf32>
    micro.yield %w : !micro.tile<16x64xf32, memory = #micro.memory<dram>>
  }
}

// The same shape with compile-time valid extents: (validM, validK) = (3, 5)
// still slices the source rectangle and pads tile-locally, and the pad amount
// is then the constant remainder of the fixed tile -- 8-3 and 32-5.
// CHECK-LABEL: func.func @bounded_load_constants(%arg0: tensor<8x32xf32>) -> tensor<8x32xf32>
module {
  micro.kernel @bounded_load_constants(%src: tensor<8x32xf32>) -> tensor<8x32xf32> {
    %c0 = arith.constant 0 : index
    %vm = arith.constant 3 : index
    %vk = arith.constant 5 : index
    // CHECK: %[[CLIPPED:.*]] = tensor.extract_slice %arg0[%{{.*}}, %{{.*}}] [%{{.*}}, %{{.*}}] [1, 1] : tensor<8x32xf32> to tensor<?x?xf32>
    // CHECK: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK-NEXT: %[[PADDED:.*]] = tensor.pad %[[CLIPPED]] low[0, 0] high[5, 27]
    // CHECK: return %[[PADDED]] : tensor<8x32xf32>
    %t, %tok = micro.tile_load %src[%c0, %c0] valid[%vm, %vk] {
        dst_memory = #micro.memory<sram>, padding = 0.0 : f32
      } : tensor<8x32xf32>
        -> !micro.tile<8x32xf32, memory = #micro.memory<sram>>, !micro.async_token
    micro.wait %tok
    micro.yield %t : !micro.tile<8x32xf32, memory = #micro.memory<sram>>
  }
}

// A compile-time 3x7 write out of the 8x32 tile: the clipped rectangle is the
// inserted slice, so the destination keeps every element outside it.
// CHECK-LABEL: func.func @clipped_store_constants(%arg0: tensor<8x32xf32>, %arg1: tensor<16x64xf32>) -> tensor<16x64xf32>
module {
  micro.kernel @clipped_store_constants(%src: tensor<8x32xf32>,
                                        %out: tensor<16x64xf32>) -> tensor<16x64xf32> {
    %c0 = arith.constant 0 : index
    %vm = arith.constant 3 : index
    %vn = arith.constant 7 : index
    %ts = micro.tile_view %src {shape = array<i64: 8, 32>} : tensor<8x32xf32> -> !micro.tile<8x32xf32, memory = #micro.memory<acc>>
    %to = micro.tile_view %out {shape = array<i64: 16, 64>} : tensor<16x64xf32> -> !micro.tile<16x64xf32, memory = #micro.memory<dram>>
    // CHECK: %[[CLIP:.*]] = tensor.extract_slice %arg0[0, 0] [%{{.*}}, %{{.*}}] [1, 1] : tensor<8x32xf32> to tensor<?x?xf32>
    // CHECK-NEXT: %[[WRITTEN:.*]] = tensor.insert_slice %[[CLIP]] into %arg1[%{{.*}}, %{{.*}}] [%{{.*}}, %{{.*}}] [1, 1] : tensor<?x?xf32> into tensor<16x64xf32>
    // CHECK: return %[[WRITTEN]] : tensor<16x64xf32>
    %w = micro.tile_store %ts into %to[%c0, %c0] valid[%vm, %vn] {
        dst_memory = #micro.memory<dram>
      } : !micro.tile<8x32xf32, memory = #micro.memory<acc>>,
          !micro.tile<16x64xf32, memory = #micro.memory<dram>>
        -> !micro.tile<16x64xf32, memory = #micro.memory<dram>>
    micro.yield %w : !micro.tile<16x64xf32, memory = #micro.memory<dram>>
  }
}

//===----------------------------------------------------------------------===//
// Exact reduction tail: valid_k = 5 inside a static BK of 32
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @exact_k(%arg0: tensor<16x32xf32>, %arg1: tensor<32x16xf32>, %arg2: tensor<16x16xf32>, %arg3: index) -> tensor<16x16xf32>
module {
  micro.kernel @exact_k(%a: tensor<16x32xf32>, %b: tensor<32x16xf32>,
                        %acc: tensor<16x16xf32>, %k: index) -> tensor<16x16xf32> {
    %ta = micro.tile_view %a {shape = array<i64: 16, 32>} : tensor<16x32xf32> -> !micro.tile<16x32xf32, memory = #micro.memory<acc>>
    %tb = micro.tile_view %b {shape = array<i64: 32, 16>} : tensor<32x16xf32> -> !micro.tile<32x16xf32, memory = #micro.memory<acc>>
    %tc = micro.tile_view %acc {shape = array<i64: 16, 16>} : tensor<16x16xf32> -> !micro.tile<16x16xf32, memory = #micro.memory<acc>>
    // Both operands are reduced over the runtime K only. The M and N extents
    // stay static, so the accumulator fragment keeps its fixed shape.
    // CHECK: %[[LHS:.*]] = tensor.extract_slice %arg0[0, 0] [16, %arg3] [1, 1] : tensor<16x32xf32> to tensor<16x?xf32>
    // CHECK-NEXT: %[[RHS:.*]] = tensor.extract_slice %arg1[0, 0] [%arg3, 16] [1, 1] : tensor<32x16xf32> to tensor<?x16xf32>
    // The accumulator is the contraction's `outs`, so its incoming values are
    // preserved and only the first `valid_k` terms are accumulated.
    // CHECK-NEXT: linalg.matmul {{.*}}ins(%[[LHS]], %[[RHS]] : tensor<16x?xf32>, tensor<?x16xf32>) outs(%arg2 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %r = micro.mma %ta, %tb, %tc valid_k %k {
        shape = array<i64: 16, 16, 32>,
        input = #micro.dtype<f32>,
        accumulator = #micro.dtype<f32>
      } : !micro.tile<16x32xf32, memory = #micro.memory<acc>>,
          !micro.tile<32x16xf32, memory = #micro.memory<acc>>,
          !micro.tile<16x16xf32, memory = #micro.memory<acc>>
        -> !micro.tile<16x16xf32, memory = #micro.memory<acc>>
    // CHECK: return %{{.*}} : tensor<16x16xf32>
    micro.yield %r : !micro.tile<16x16xf32, memory = #micro.memory<acc>>
  }
}

//===----------------------------------------------------------------------===//
// Legacy full-tile mma keeps the full-BK contraction
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @full_k(%arg0: tensor<16x32xf32>, %arg1: tensor<32x16xf32>, %arg2: tensor<16x16xf32>) -> tensor<16x16xf32>
module {
  micro.kernel @full_k(%a: tensor<16x32xf32>, %b: tensor<32x16xf32>,
                       %acc: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %ta = micro.tile_view %a {shape = array<i64: 16, 32>} : tensor<16x32xf32> -> !micro.tile<16x32xf32, memory = #micro.memory<acc>>
    %tb = micro.tile_view %b {shape = array<i64: 32, 16>} : tensor<32x16xf32> -> !micro.tile<32x16xf32, memory = #micro.memory<acc>>
    %tc = micro.tile_view %acc {shape = array<i64: 16, 16>} : tensor<16x16xf32> -> !micro.tile<16x16xf32, memory = #micro.memory<acc>>
    // Without `valid_k` the contraction reads the whole static K capacity.
    // CHECK: linalg.matmul {{.*}}ins(%arg0, %arg1 : tensor<16x32xf32>, tensor<32x16xf32>) outs(%arg2 : tensor<16x16xf32>) -> tensor<16x16xf32>
    %r = micro.mma %ta, %tb, %tc {
        shape = array<i64: 16, 16, 32>,
        input = #micro.dtype<f32>,
        accumulator = #micro.dtype<f32>
      } : !micro.tile<16x32xf32, memory = #micro.memory<acc>>,
          !micro.tile<32x16xf32, memory = #micro.memory<acc>>,
          !micro.tile<16x16xf32, memory = #micro.memory<acc>>
        -> !micro.tile<16x16xf32, memory = #micro.memory<acc>>
    // CHECK: return %{{.*}} : tensor<16x16xf32>
    micro.yield %r : !micro.tile<16x16xf32, memory = #micro.memory<acc>>
  }
}
