module {
  micro.kernel @add_mul_reduce(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>)
      -> tensor<8xf32> {
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>}
        : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b {shape = array<i64: 8, 8>}
        : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %sum = micro.vector "add" %ta, %tb
        : !micro.tile<8x8xf32, memory = #micro.memory<sram>>,
          !micro.tile<8x8xf32, memory = #micro.memory<sram>>
       -> !micro.tile<8x8xf32, memory = #micro.memory<acc>>
    %prod = micro.vector "mul" %sum, %ta
        : !micro.tile<8x8xf32, memory = #micro.memory<acc>>,
          !micro.tile<8x8xf32, memory = #micro.memory<sram>>
       -> !micro.tile<8x8xf32, memory = #micro.memory<acc>>
    %sum_row = micro.reduce "sum" %prod {axis = 1 : i64}
        : !micro.tile<8x8xf32, memory = #micro.memory<acc>>
       -> !micro.tile<8xf32, memory = #micro.memory<acc>>
    micro.yield %sum_row : !micro.tile<8xf32, memory = #micro.memory<acc>>
  }
}
