module {
  micro.kernel @matmul_f32(%lhs: tensor<4x8xf32>,
      %rhs: tensor<8x8xf32>, %init: tensor<4x8xf32>) -> tensor<4x8xf32> {
    %lhs_tile = micro.tile_view %lhs {shape = array<i64: 4, 8>}
        : tensor<4x8xf32> -> !micro.tile<4x8xf32, memory = #micro.memory<sram>>
    %rhs_tile = micro.tile_view %rhs {shape = array<i64: 8, 8>}
        : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %init_tile = micro.tile_view %init {shape = array<i64: 4, 8>}
        : tensor<4x8xf32> -> !micro.tile<4x8xf32, memory = #micro.memory<acc>>
    %result = micro.mma %lhs_tile, %rhs_tile, %init_tile {
        shape = array<i64: 4, 8, 8>, input = #micro.dtype<f32>,
        accumulator = #micro.dtype<f32>}
        : !micro.tile<4x8xf32, memory = #micro.memory<sram>>,
          !micro.tile<8x8xf32, memory = #micro.memory<sram>>,
          !micro.tile<4x8xf32, memory = #micro.memory<acc>>
       -> !micro.tile<4x8xf32, memory = #micro.memory<acc>>
    micro.yield %result : !micro.tile<4x8xf32, memory = #micro.memory<acc>>
  }
}
