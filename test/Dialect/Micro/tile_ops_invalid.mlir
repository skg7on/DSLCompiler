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

// tile_partition fragment shape must divide the parent shape.
func.func @tile_partition_bad_shape(%tile : !micro.tile<32x64xbf16>) {
  // expected-error @+1 {{fragment shape must divide parent shape}}
  %frag = micro.tile_partition %tile {shape = array<i64: 7, 7>} : !micro.tile<32x64xbf16> -> !micro.tile<7x7xbf16>
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

// micro.mma fragment shape must have exactly three dimensions.
func.func @mma_bad_shape(%a : !micro.tile<16x32xbf16>, %b : !micro.tile<32x16xbf16>, %c : !micro.tile<16x16xf32>) {
  // expected-error @+1 {{mma shape must have exactly 3 dimensions}}
  %r = micro.mma %a, %b, %c {shape = array<i64: 16, 16>, input = #micro.dtype<bf16>, accumulator = #micro.dtype<f32>} : !micro.tile<16x32xbf16>, !micro.tile<32x16xbf16>, !micro.tile<16x16xf32> -> !micro.tile<16x16xf32>
  return
}

// -----

// micro.reduce axis must be within the input rank.
func.func @reduce_bad_axis(%scores : !micro.tile<32x64xf32>) {
  // expected-error @+1 {{reduce axis must be within the input rank}}
  %m = micro.reduce "max" %scores {axis = 5 : i64} : !micro.tile<32x64xf32> -> !micro.tile<32xf32>
  return
}
