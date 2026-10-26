// RUN: llk-opt --verify-diagnostics --split-input-file %s

// A boundary primitive that cannot be executed as written is a verifier error,
// not a guess. The tile result stays statically sized; a valid rectangle must
// fit inside it and inside a statically known source, and the padding value
// must have the tile's element type.

// -----
// A valid extent is nonnegative. A negative constant would make the clipped
// rectangle empty while still claiming copied lanes.
func.func @negative_valid_size(%src: tensor<8x32xf32>, %vk: index) {
  %neg = arith.constant -1 : index
  %z = arith.constant 0 : index
  // expected-error @+1 {{valid size must be nonnegative}}
  %t, %tok = micro.tile_load %src[%z, %z] valid[%neg, %vk] {
      dst_memory = #micro.memory<sram>, padding = 0.0 : f32
    } : tensor<8x32xf32>
      -> !micro.tile<8x32xf32, memory = #micro.memory<sram>>, !micro.async_token
  return
}

// -----
// A valid extent never exceeds the fixed tile it materializes: the lane is
// either copied or padded, never both.
func.func @valid_size_exceeds_tile(%src: tensor<8x32xf32>, %vk: index) {
  %nine = arith.constant 9 : index
  %z = arith.constant 0 : index
  // expected-error @+1 {{valid size must not exceed the tile extent}}
  %t, %tok = micro.tile_load %src[%z, %z] valid[%nine, %vk] {
      dst_memory = #micro.memory<sram>, padding = 0.0 : f32
    } : tensor<8x32xf32>
      -> !micro.tile<8x32xf32, memory = #micro.memory<sram>>, !micro.async_token
  return
}

// -----
// One offset and one valid extent are required per source dimension; a missing
// one would silently leave an axis unbounded.
func.func @missing_offsets(%src: tensor<8x32xf32>, %vm: index, %vk: index) {
  %z = arith.constant 0 : index
  // expected-error @+1 {{offsets must name one index per source dimension}}
  %t, %tok = micro.tile_load %src[%z] valid[%vm, %vk] {
      dst_memory = #micro.memory<sram>, padding = 0.0 : f32
    } : tensor<8x32xf32>
      -> !micro.tile<8x32xf32, memory = #micro.memory<sram>>, !micro.async_token
  return
}

// -----
// Padding is typed. An i32 zero cannot fill f32 lanes, because the fill would
// have to invent a conversion the program never asked for.
func.func @wrong_padding_dtype(%src: tensor<8x32xf32>, %vm: index, %vk: index) {
  %z = arith.constant 0 : index
  // expected-error @+1 {{padding value type must match the result tile element type}}
  %t, %tok = micro.tile_load %src[%z, %z] valid[%vm, %vk] {
      dst_memory = #micro.memory<sram>, padding = 0 : i32
    } : tensor<8x32xf32>
      -> !micro.tile<8x32xf32, memory = #micro.memory<sram>>, !micro.async_token
  return
}

// -----
// A bounded load moves data out of a ranked tensor or memref. Feeding it an
// already materialized tile would make the padding semantics of an existing
// allocation implicit.
func.func @source_is_a_tile(%t: !micro.tile<8x32xf32, memory = #micro.memory<acc>>,
                            %vm: index, %vk: index) {
  %z = arith.constant 0 : index
  // expected-error @+1 {{source must be a ranked tensor or memref}}
  %r, %tok = micro.tile_load %t[%z, %z] valid[%vm, %vk] {
      dst_memory = #micro.memory<sram>, padding = 0.0 : f32
    } : !micro.tile<8x32xf32, memory = #micro.memory<acc>>
      -> !micro.tile<8x32xf32, memory = #micro.memory<sram>>, !micro.async_token
  return
}

// -----
// The source rank and the tile rank agree; otherwise which source axis a valid
// extent clips is undefined.
func.func @rank_mismatch(%src: tensor<8x32xf32>, %vm: index, %vk: index) {
  %z = arith.constant 0 : index
  // expected-error @+1 {{source rank must match the result tile rank}}
  %t, %tok = micro.tile_load %src[%z, %z] valid[%vm, %vk] {
      dst_memory = #micro.memory<sram>, padding = 0.0 : f32
    } : tensor<8x32xf32>
      -> !micro.tile<8x32x1xf32, memory = #micro.memory<sram>>, !micro.async_token
  return
}

// -----
// A statically known rectangle that leaves its source is refused at compile
// time. Clipping it silently would read elements the program did not ask for.
func.func @rectangle_leaves_source(%src: tensor<8x32xf32>) {
  %two = arith.constant 2 : index
  %seven = arith.constant 7 : index
  %z = arith.constant 0 : index
  %n = arith.constant 32 : index
  // expected-error @+1 {{load rectangle leaves the source}}
  %t, %tok = micro.tile_load %src[%two, %z] valid[%seven, %n] {
      dst_memory = #micro.memory<sram>, padding = 0.0 : f32
    } : tensor<8x32xf32>
      -> !micro.tile<8x32xf32, memory = #micro.memory<sram>>, !micro.async_token
  return
}

// -----
// The destination memory is a property of the tile result, so the attribute and
// the result type cannot disagree.
func.func @destination_memory_mismatch(%src: tensor<8x32xf32>, %vm: index,
                                       %vk: index) {
  %z = arith.constant 0 : index
  // expected-error @+1 {{result tile memory must match destination memory}}
  %t, %tok = micro.tile_load %src[%z, %z] valid[%vm, %vk] {
      dst_memory = #micro.memory<dram>, padding = 0.0 : f32
    } : tensor<8x32xf32>
      -> !micro.tile<8x32xf32, memory = #micro.memory<sram>>, !micro.async_token
  return
}

// -----
// A clipped store's valid extent is nonnegative for the same reason a load's
// is.
func.func @store_negative_valid_size(
    %t: !micro.tile<8x32xf32, memory = #micro.memory<acc>>,
    %d: !micro.tile<16x64xf32, memory = #micro.memory<dram>>,
    %i: index, %j: index, %vn: index) {
  %neg = arith.constant -1 : index
  // expected-error @+1 {{valid size must be nonnegative}}
  %w = micro.tile_store %t into %d[%i, %j] valid[%neg, %vn] {
      dst_memory = #micro.memory<dram>
    } : !micro.tile<8x32xf32, memory = #micro.memory<acc>>,
        !micro.tile<16x64xf32, memory = #micro.memory<dram>>
      -> !micro.tile<16x64xf32, memory = #micro.memory<dram>>
  return
}

// -----
// The clipped rectangle comes out of the source tile, so a valid extent larger
// than the source extent would write lanes the source does not hold.
func.func @store_valid_exceeds_source(
    %t: !micro.tile<8x32xf32, memory = #micro.memory<acc>>,
    %d: !micro.tile<16x64xf32, memory = #micro.memory<dram>>,
    %i: index, %j: index, %vn: index) {
  %nine = arith.constant 9 : index
  // expected-error @+1 {{valid size must not exceed the source extent}}
  %w = micro.tile_store %t into %d[%i, %j] valid[%nine, %vn] {
      dst_memory = #micro.memory<dram>
    } : !micro.tile<8x32xf32, memory = #micro.memory<acc>>,
        !micro.tile<16x64xf32, memory = #micro.memory<dram>>
      -> !micro.tile<16x64xf32, memory = #micro.memory<dram>>
  return
}

// -----
// The reduction length is bounded by the fragment's static K capacity; a larger
// constant would read past the operand tiles.
func.func @valid_k_exceeds_capacity(
    %a: !micro.tile<16x32xbf16, memory = #micro.memory<acc>>,
    %b: !micro.tile<32x16xbf16, memory = #micro.memory<acc>>,
    %c: !micro.tile<16x16xf32, memory = #micro.memory<acc>>) {
  %k = arith.constant 33 : index
  // expected-error @+1 {{valid_k must not exceed the mma K capacity}}
  %r = micro.mma %a, %b, %c valid_k %k {
      shape = array<i64: 16, 16, 32>,
      input = #micro.dtype<bf16>,
      accumulator = #micro.dtype<f32>
    } : !micro.tile<16x32xbf16, memory = #micro.memory<acc>>,
        !micro.tile<32x16xbf16, memory = #micro.memory<acc>>,
        !micro.tile<16x16xf32, memory = #micro.memory<acc>>
      -> !micro.tile<16x16xf32, memory = #micro.memory<acc>>
  return
}

// -----
// A negative reduction length would run zero terms while claiming an exact-K
// reduction; it is rejected rather than clamped.
func.func @negative_valid_k(
    %a: !micro.tile<16x32xbf16, memory = #micro.memory<acc>>,
    %b: !micro.tile<32x16xbf16, memory = #micro.memory<acc>>,
    %c: !micro.tile<16x16xf32, memory = #micro.memory<acc>>) {
  %k = arith.constant -1 : index
  // expected-error @+1 {{valid_k must be nonnegative}}
  %r = micro.mma %a, %b, %c valid_k %k {
      shape = array<i64: 16, 16, 32>,
      input = #micro.dtype<bf16>,
      accumulator = #micro.dtype<f32>
    } : !micro.tile<16x32xbf16, memory = #micro.memory<acc>>,
        !micro.tile<32x16xbf16, memory = #micro.memory<acc>>,
        !micro.tile<16x16xf32, memory = #micro.memory<acc>>
      -> !micro.tile<16x16xf32, memory = #micro.memory<acc>>
  return
}
