// RUN: llk-opt %s | llk-opt | FileCheck %s

// Bounded movement and exact reduction tails. A dynamic contraction keeps its
// local tiles statically sized: the boundary access is clipped to a runtime
// valid rectangle, and the complement of the destination tile is filled with an
// explicit padding value. Each new spelling must round-trip unchanged, and the
// pre-existing full-tile spellings must keep parsing and printing exactly as
// they did -- a legacy store or mma does not become "valid = whole tile".

//===----------------------------------------------------------------------===//
// micro.tile_load — bounded source materialization
//===----------------------------------------------------------------------===//

// A bounded load copies the source rectangle `[offset, offset+valid)` into the
// origin of a fixed 8x32 tile and fills the rest from the typed padding value.
// The source is a ranked tensor (or memref), never a tile operand: a tile is
// already materialized, and an out-of-range view must not be reinterpreted as
// padding.
// CHECK-LABEL: func.func @bounded_load
// CHECK: %[[T:.*]], %[[TOK:.*]] = micro.tile_load %{{.*}}[%{{.*}}, %{{.*}}] valid[%{{.*}}, %{{.*}}] {dst_memory = #micro.memory<sram>, owner = #micro.owner<worker>, padding = 0.000000e+00 : f32} : tensor<8x32xf32> -> !micro.tile<8x32xf32, memory = #micro.memory<sram>, owner = #micro.owner<worker>>, !micro.async_token
// CHECK: micro.wait %[[TOK]]
func.func @bounded_load(%src: tensor<8x32xf32>, %i: index, %j: index,
                        %vm: index, %vk: index) {
  %t, %tok = micro.tile_load %src[%i, %j] valid[%vm, %vk] {
      dst_memory = #micro.memory<sram>,
      owner = #micro.owner<worker>,
      padding = 0.0 : f32
    } : tensor<8x32xf32>
      -> !micro.tile<8x32xf32, memory = #micro.memory<sram>, owner = #micro.owner<worker>>, !micro.async_token
  micro.wait %tok
  return
}

// The same op over a memref source: the boundary form is not tensor-only.
// CHECK-LABEL: func.func @bounded_load_memref
// CHECK: micro.tile_load %{{.*}}[%{{.*}}, %{{.*}}] valid[%{{.*}}, %{{.*}}] {dst_memory = #micro.memory<sram>, padding = 0.000000e+00 : f32} : memref<8x32xf32> -> !micro.tile<8x32xf32, memory = #micro.memory<sram>>, !micro.async_token
func.func @bounded_load_memref(%src: memref<8x32xf32>, %i: index, %j: index,
                               %vm: index, %vk: index) {
  %t, %tok = micro.tile_load %src[%i, %j] valid[%vm, %vk] {
      dst_memory = #micro.memory<sram>,
      padding = 0.0 : f32
    } : memref<8x32xf32>
      -> !micro.tile<8x32xf32, memory = #micro.memory<sram>>, !micro.async_token
  return
}

// Zero is an ordinary valid extent: the whole tile is padding, and no lane of
// the source is read.
// CHECK-LABEL: func.func @bounded_load_empty
// CHECK: micro.tile_load %{{.*}}[%{{.*}}, %{{.*}}] valid[%{{.*}}, %{{.*}}] {dst_memory = #micro.memory<sram>, padding = 0.000000e+00 : f32} : tensor<8x32xf32> -> !micro.tile<8x32xf32, memory = #micro.memory<sram>>, !micro.async_token
func.func @bounded_load_empty(%src: tensor<8x32xf32>, %z: index) {
  %t, %tok = micro.tile_load %src[%z, %z] valid[%z, %z] {
      dst_memory = #micro.memory<sram>,
      padding = 0.0 : f32
    } : tensor<8x32xf32>
      -> !micro.tile<8x32xf32, memory = #micro.memory<sram>>, !micro.async_token
  return
}

//===----------------------------------------------------------------------===//
// micro.tile_store — clipped write-back
//===----------------------------------------------------------------------===//

// With `valid[...]` the store writes only the origin rectangle of the source
// tile at the destination offsets. An 8x32 tile writing a 3x7 rectangle leaves
// every element outside it untouched.
// CHECK-LABEL: func.func @clipped_store
// CHECK: %[[W:.*]] = micro.tile_store %{{.*}} into %{{.*}}[%{{.*}}, %{{.*}}] valid[%{{.*}}, %{{.*}}] {dst_memory = #micro.memory<dram>} : !micro.tile<8x32xf32, memory = #micro.memory<acc>>, !micro.tile<16x64xf32, memory = #micro.memory<dram>> -> !micro.tile<16x64xf32, memory = #micro.memory<dram>>
func.func @clipped_store(%t: !micro.tile<8x32xf32, memory = #micro.memory<acc>>,
                         %d: !micro.tile<16x64xf32, memory = #micro.memory<dram>>,
                         %i: index, %j: index, %vm: index, %vn: index) {
  %w = micro.tile_store %t into %d[%i, %j] valid[%vm, %vn] {
      dst_memory = #micro.memory<dram>
    } : !micro.tile<8x32xf32, memory = #micro.memory<acc>>,
        !micro.tile<16x64xf32, memory = #micro.memory<dram>>
      -> !micro.tile<16x64xf32, memory = #micro.memory<dram>>
  return
}

//===----------------------------------------------------------------------===//
// micro.mma — exact reduction tail
//===----------------------------------------------------------------------===//

// `valid_k` bounds the logical reduction length inside the tile's static K
// capacity. Omission means the full K and keeps the old assembly exactly.
// CHECK-LABEL: func.func @mma_valid_k
// CHECK: micro.mma %{{.*}}, %{{.*}}, %{{.*}} valid_k %{{.*}} {accumulator = #micro.dtype<f32>, input = #micro.dtype<bf16>, shape = array<i64: 16, 16, 32>} : !micro.tile<16x32xbf16, memory = #micro.memory<acc>>, !micro.tile<32x16xbf16, memory = #micro.memory<acc>>, !micro.tile<16x16xf32, memory = #micro.memory<acc>> -> !micro.tile<16x16xf32, memory = #micro.memory<acc>>
func.func @mma_valid_k(%a: !micro.tile<16x32xbf16, memory = #micro.memory<acc>>,
                       %b: !micro.tile<32x16xbf16, memory = #micro.memory<acc>>,
                       %c: !micro.tile<16x16xf32, memory = #micro.memory<acc>>,
                       %k: index) {
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

//===----------------------------------------------------------------------===//
// Legacy full-tile spellings are unchanged
//===----------------------------------------------------------------------===//

// A store without `valid[...]` still writes the whole tile, and a store without
// `into` is still a terminal write.
// CHECK-LABEL: func.func @legacy_store
// CHECK: micro.tile_store %{{.*}} {dst_memory = #micro.memory<dram>} : !micro.tile<32x64xbf16, memory = #micro.memory<acc>>
// CHECK-NOT: valid[
// CHECK: micro.tile_store %{{.*}} into %{{.*}}[%{{.*}}, %{{.*}}] {dst_memory = #micro.memory<dram>} : !micro.tile<8x32xf32, memory = #micro.memory<acc>>, !micro.tile<16x64xf32, memory = #micro.memory<dram>> -> !micro.tile<16x64xf32, memory = #micro.memory<dram>>
func.func @legacy_store(%t: !micro.tile<32x64xbf16, memory = #micro.memory<acc>>,
                        %s: !micro.tile<8x32xf32, memory = #micro.memory<acc>>,
                        %d: !micro.tile<16x64xf32, memory = #micro.memory<dram>>,
                        %i: index, %j: index) {
  micro.tile_store %t {dst_memory = #micro.memory<dram>} : !micro.tile<32x64xbf16, memory = #micro.memory<acc>>
  %w = micro.tile_store %s into %d[%i, %j] {dst_memory = #micro.memory<dram>} : !micro.tile<8x32xf32, memory = #micro.memory<acc>>, !micro.tile<16x64xf32, memory = #micro.memory<dram>> -> !micro.tile<16x64xf32, memory = #micro.memory<dram>>
  return
}

// An mma without `valid_k` reduces the full K capacity, with the same assembly
// as before the operand existed.
// CHECK-LABEL: func.func @legacy_mma
// CHECK: micro.mma %{{.*}}, %{{.*}}, %{{.*}} {accumulator = #micro.dtype<f32>, input = #micro.dtype<bf16>, shape = array<i64: 16, 16, 32>} : !micro.tile<16x32xbf16>, !micro.tile<32x16xbf16>, !micro.tile<16x16xf32> -> !micro.tile<16x16xf32>
// CHECK-NOT: valid_k
func.func @legacy_mma(%a: !micro.tile<16x32xbf16>, %b: !micro.tile<32x16xbf16>,
                      %c: !micro.tile<16x16xf32>) {
  %r = micro.mma %a, %b, %c {
      shape = array<i64: 16, 16, 32>,
      input = #micro.dtype<bf16>,
      accumulator = #micro.dtype<f32>
    } : !micro.tile<16x32xbf16>, !micro.tile<32x16xbf16>, !micro.tile<16x16xf32>
      -> !micro.tile<16x16xf32>
  return
}
