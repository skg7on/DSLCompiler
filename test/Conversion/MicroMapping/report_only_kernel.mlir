module {
  micro.kernel @mapped {
    %0 = tensor.empty() : tensor<8x8xf32>
    %result, %token = micro.async_copy %0 {dst_memory = #micro.memory<sram>, src_memory = #micro.memory<dram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    %1 = micro.tile_view %result {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %2 = micro.vector "add" %1, %1 : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
}

