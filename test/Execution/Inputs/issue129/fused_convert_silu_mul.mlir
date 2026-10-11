module {
  micro.kernel @fused_convert_silu_mul {
    %input = tensor.empty() : tensor<8x8xf32>
    %tile = micro.tile_view %input {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %converted = micro.vector "convert" %tile : !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %activated = micro.vector "silu" %converted : !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %result = micro.vector "mul" %activated, %tile : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.tile_store %result {dst_memory = #micro.memory<dram>} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}
