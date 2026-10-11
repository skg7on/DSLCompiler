#identity = affine_map<(d0, d1) -> (d0, d1)>
#transpose = affine_map<(d0, d1) -> (d1, d0)>
module {
  micro.kernel @required_transform(%input: tensor<8x8xf32>,
      %rhs: tensor<8x8xf32>)
      -> tensor<8x8xf32> {
    %tile = micro.tile_view %input {shape = array<i64: 8, 8>}
        : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %transposed = micro.transform %tile {
        src_map = #identity, dst_map = #transpose,
        micro.engine = "worker.0", micro.connection = 7 : i64}
        : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
       -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %rhs_tile = micro.tile_view %rhs {shape = array<i64: 8, 8>}
        : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %sum = micro.vector "add" %transposed, %rhs_tile
        : !micro.tile<8x8xf32, memory = #micro.memory<sram>>,
          !micro.tile<8x8xf32, memory = #micro.memory<sram>>
       -> !micro.tile<8x8xf32, memory = #micro.memory<acc>>
    micro.yield %sum : !micro.tile<8x8xf32, memory = #micro.memory<acc>>
  }
}
