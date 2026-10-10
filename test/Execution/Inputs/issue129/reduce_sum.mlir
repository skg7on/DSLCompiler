module {
  micro.kernel @reduce_sum(%input: tensor<8x8xf32>) -> tensor<8xf32> {
    %tile = micro.tile_view %input {shape = array<i64: 8, 8>}
        : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %sum = micro.reduce "sum" %tile {axis = 1 : i64}
        : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
       -> !micro.tile<8xf32, memory = #micro.memory<acc>>
    micro.yield %sum : !micro.tile<8xf32, memory = #micro.memory<acc>>
  }
}
