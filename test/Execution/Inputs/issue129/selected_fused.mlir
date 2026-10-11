module {
  micro.kernel @selected_fused(
      %input: tensor<8x8xf32>, %gate: tensor<8x8xf32>)
      -> tensor<8x8xf32> {
    %tile = micro.tile_view %input {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %gate_tile = micro.tile_view %gate {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %converted = micro.vector "convert" %tile : !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %activated = micro.vector "silu" %converted : !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %result = micro.vector "mul" %activated, %gate_tile : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield %result : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}
