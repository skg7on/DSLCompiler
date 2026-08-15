// RUN: llk-opt --verify-diagnostics --split-input-file %s

// Invalid tile type, layout/owner attributes, and tile ops must be rejected
// with actionable diagnostics.

// -----

// Invalid layout: block extent must be positive.
func.func @bad_block_layout() attributes {
  // expected-error @+1 {{block extents must be positive}}
  bad = #micro.layout<blocked, block = array<i64: 0, 16>>
} {
  return
}

// -----

// Invalid layout: align must be a power of two.
func.func @bad_align_layout() attributes {
  // expected-error @+1 {{align must be a power of two}}
  bad = #micro.layout<row_major, align = 24>
} {
  return
}

// -----

// Invalid layout: swizzle requires a swizzled layout kind.
func.func @bad_swizzle_layout() attributes {
  // expected-error @+1 {{swizzle requires a swizzled layout}}
  bad = #micro.layout<row_major, swizzle = "xor">
} {
  return
}

// -----

// Invalid layout: blocked requires a block parameter.
func.func @bad_blocked_layout() attributes {
  // expected-error @+1 {{blocked layout requires a block parameter}}
  bad = #micro.layout<blocked>
} {
  return
}

// -----

// Invalid layout: duplicate parameter.
func.func @bad_dup_layout() attributes {
  // expected-error @+1 {{duplicate layout parameter 'vector'}}
  bad = #micro.layout<row_major, vector = 8, vector = 8>
} {
  return
}

// -----

// Invalid tile type: rank-0 (scalar) tile.
func.func @bad_rank0_tile() {
  // expected-error @+1 {{tile must have at least one dimension}}
  %t = micro.tile_alloc : !micro.tile<f32, memory = #micro.memory<acc>>
  return
}

// -----

// tile_alloc result must have a memory space.
func.func @tile_alloc_missing_memory() {
  // expected-error @+1 {{tile_alloc result tile must have a memory space}}
  %acc = micro.tile_alloc : !micro.tile<32x64xf32>
  return
}

// -----

// tile_alloc must not allocate in dram.
func.func @tile_alloc_in_dram() {
  // expected-error @+1 {{tile_alloc memory must not be dram}}
  %acc = micro.tile_alloc : !micro.tile<32x64xf32, memory = #micro.memory<dram>>
  return
}

// -----

// tile_alloc result must have a statically computable byte size.
func.func @tile_alloc_dynamic_shape() {
  // expected-error @+1 {{tile_alloc result must have a statically computable byte size}}
  %acc = micro.tile_alloc : !micro.tile<?x64xf32, memory = #micro.memory<acc>>
  return
}

// -----

// tile_view must preserve the element type.
func.func @tile_view_bad_element_type(%A : tensor<?x?xbf16>) {
  // expected-error @+1 {{source and result element types must match}}
  %t = micro.tile_view %A {shape = array<i64: 32, 64>} : tensor<?x?xbf16> -> !micro.tile<32x64xf32>
  return
}

// -----

// tile_partition fragment shape must divide the parent shape.
func.func @tile_partition_bad_shape(%tile : !micro.tile<32x64xbf16>) {
  // expected-error @+1 {{fragment shape must divide parent shape}}
  %frag = micro.tile_partition %tile {shape = array<i64: 7, 7>} : !micro.tile<32x64xbf16> -> !micro.tile<7x7xbf16>
  return
}

// -----

// tile_partition owner attribute requires the result tile to carry an owner.
func.func @tile_partition_owner_missing(%tile : !micro.tile<32x64xbf16>) {
  // expected-error @+1 {{owner attribute requires the result tile to carry an owner}}
  %frag = micro.tile_partition %tile {shape = array<i64: 16, 16>, owner = #micro.owner<vector_engine>} : !micro.tile<32x64xbf16> -> !micro.tile<16x16xbf16>
  return
}

// -----

// tile_async_copy source and destination memory must differ.
func.func @tile_async_copy_same_memory(%tile : !micro.tile<32x64xbf16, memory = #micro.memory<sram>>) {
  // expected-error @+1 {{source and destination memory must differ}}
  %dst, %tok = micro.tile_async_copy %tile {dst_memory = #micro.memory<sram>} : !micro.tile<32x64xbf16, memory = #micro.memory<sram>> -> !micro.tile<32x64xbf16, memory = #micro.memory<sram>>, !micro.async_token
  return
}

// -----

// tile_async_copy result tile must have a memory space.
func.func @tile_async_copy_missing_memory(%logical : !micro.tile<32x64xbf16>) {
  // expected-error @+1 {{result tile must have a memory space}}
  %dst, %tok = micro.tile_async_copy %logical {dst_memory = #micro.memory<sram>} : !micro.tile<32x64xbf16> -> !micro.tile<32x64xbf16>, !micro.async_token
  return
}

// -----

// tile_async_copy must preserve shape.
func.func @tile_async_copy_shape_mismatch(%logical : !micro.tile<32x64xbf16>) {
  // expected-error @+1 {{result tile shape must match source tile shape}}
  %dst, %tok = micro.tile_async_copy %logical {dst_memory = #micro.memory<sram>} : !micro.tile<32x64xbf16> -> !micro.tile<16x16xbf16, memory = #micro.memory<sram>>, !micro.async_token
  return
}

// -----

// micro.mma fragment shape must have exactly three dimensions.
func.func @mma_bad_shape(%a : !micro.tile<16x32xbf16>, %b : !micro.tile<32x16xbf16>, %c : !micro.tile<16x16xf32>) {
  // expected-error @+1 {{mma shape must have exactly 3 dimensions}}
  %r = micro.mma %a, %b, %c {shape = array<i64: 16, 16>, input = #micro.dtype<bf16>, accumulator = #micro.dtype<f32>} : !micro.tile<16x32xbf16>, !micro.tile<32x16xbf16>, !micro.tile<16x16xf32> -> !micro.tile<16x16xf32>
  return
}

// -----

// micro.mma shape must be geometrically compatible with the tiles.
func.func @mma_bad_geometry(%a : !micro.tile<16x32xbf16>, %b : !micro.tile<32x16xbf16>, %c : !micro.tile<16x16xf32>) {
  // expected-error @+1 {{lhs tile shape must be [M, K]}}
  %r = micro.mma %a, %b, %c {shape = array<i64: 8, 8, 8>, input = #micro.dtype<bf16>, accumulator = #micro.dtype<f32>} : !micro.tile<16x32xbf16>, !micro.tile<32x16xbf16>, !micro.tile<16x16xf32> -> !micro.tile<16x16xf32>
  return
}

// -----

// micro.vector arity must match the operation.
func.func @vector_bad_arity(%x : !micro.tile<32x64xf32>) {
  // expected-error @+1 {{vector operation has wrong number of operands}}
  %r = micro.vector "add" %x : !micro.tile<32x64xf32> -> !micro.tile<32x64xf32>
  return
}

// -----

// micro.reduce result shape must drop the reduced axis.
func.func @reduce_bad_result_shape(%scores : !micro.tile<32x64xf32>) {
  // expected-error @+1 {{reduce result shape must be the input shape with the reduced axis removed}}
  %m = micro.reduce "max" %scores {axis = 1 : i64} : !micro.tile<32x64xf32> -> !micro.tile<16xf32>
  return
}

// -----

// micro.reduce axis must be within the input rank.
func.func @reduce_bad_axis(%scores : !micro.tile<32x64xf32>) {
  // expected-error @+1 {{reduce axis must be within the input rank}}
  %m = micro.reduce "max" %scores {axis = 5 : i64} : !micro.tile<32x64xf32> -> !micro.tile<32xf32>
  return
}
