// RUN: llk-opt %s | llk-opt | FileCheck %s

// Tile-centric Micro-IR: M9a (tile type + layout/owner attributes) and
// M9b (tile ops MVP). Verifies that !micro.tile, #micro.layout, #micro.owner,
// and the tile operation family parse, print, and round-trip.

//===----------------------------------------------------------------------===//
// #micro.layout attribute
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_layout
// CHECK: #micro.layout<row_major>
// CHECK: #micro.layout<col_major>
// CHECK: #micro.layout<blocked, block = array<i64: 16, 16>>
// CHECK: #micro.layout<row_major, vector = 8, align = 32>
// CHECK: #micro.layout<swizzled, banks = 32, swizzle = "xor">
func.func @test_layout() attributes {
  layouts = [
    #micro.layout<row_major>,
    #micro.layout<col_major>,
    #micro.layout<blocked, block = array<i64: 16, 16>>,
    #micro.layout<row_major, vector = 8, align = 32>,
    #micro.layout<swizzled, banks = 32, swizzle = "xor">
  ]
} {
  return
}

//===----------------------------------------------------------------------===//
// #micro.owner attribute
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_owner
// CHECK: #micro.owner<cluster>
// CHECK: #micro.owner<worker>
// CHECK: #micro.owner<matrix_engine>
// CHECK: #micro.owner<vector_engine>
func.func @test_owner() attributes {
  owners = [
    #micro.owner<cluster>,
    #micro.owner<worker>,
    #micro.owner<matrix_engine>,
    #micro.owner<vector_engine>
  ]
} {
  return
}

//===----------------------------------------------------------------------===//
// !micro.tile type (minimal and full)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_tile_type
// CHECK: !micro.tile<32x64xbf16, layout = #micro.layout<row_major>>
// CHECK: !micro.tile<32x64xf32, memory = #micro.memory<acc>, owner = #micro.owner<worker>>
func.func @test_tile_type(%a : !micro.tile<32x64xbf16, layout = #micro.layout<row_major>>,
                          %b : !micro.tile<32x64xf32, memory = #micro.memory<acc>, owner = #micro.owner<worker>>) {
  return
}

//===----------------------------------------------------------------------===//
// micro.tile_view — logical zero-cost view
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_tile_view
func.func @test_tile_view(%A : tensor<?x?xbf16>) {
  // CHECK: micro.tile_view %{{.*}} {shape = array<i64: 32, 64>} : tensor<?x?xbf16> -> !micro.tile<32x64xbf16>
  %t = micro.tile_view %A {shape = array<i64: 32, 64>} : tensor<?x?xbf16> -> !micro.tile<32x64xbf16>
  return
}

//===----------------------------------------------------------------------===//
// micro.tile_alloc — materialized memory tile
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_tile_alloc
func.func @test_tile_alloc() {
  // CHECK: micro.tile_alloc : !micro.tile<32x64xf32, memory = #micro.memory<acc>, owner = #micro.owner<worker>>
  %acc = micro.tile_alloc : !micro.tile<32x64xf32, memory = #micro.memory<acc>, owner = #micro.owner<worker>>
  return
}

//===----------------------------------------------------------------------===//
// micro.tile_partition — partition into instruction fragment
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_tile_partition
func.func @test_tile_partition(%tile : !micro.tile<32x64xbf16>) {
  // CHECK: micro.tile_partition %{{.*}} {owner = #micro.owner<vector_engine>, shape = array<i64: 16, 16>} : !micro.tile<32x64xbf16> -> !micro.tile<16x16xbf16>
  %frag = micro.tile_partition %tile {shape = array<i64: 16, 16>, owner = #micro.owner<vector_engine>} : !micro.tile<32x64xbf16> -> !micro.tile<16x16xbf16>
  return
}

//===----------------------------------------------------------------------===//
// micro.tile_async_copy — async movement returning tile + token
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_tile_async_copy
func.func @test_tile_async_copy(%logical : !micro.tile<32x64xbf16>) {
  // CHECK: %[[T:.*]], %[[TOK:.*]] = micro.tile_async_copy %{{.*}} {dst_memory = #micro.memory<sram>, owner = #micro.owner<worker>} : !micro.tile<32x64xbf16> -> !micro.tile<32x64xbf16, memory = #micro.memory<sram>>, !micro.async_token
  %tile, %tok = micro.tile_async_copy %logical {dst_memory = #micro.memory<sram>, owner = #micro.owner<worker>} : !micro.tile<32x64xbf16> -> !micro.tile<32x64xbf16, memory = #micro.memory<sram>>, !micro.async_token
  return
}

//===----------------------------------------------------------------------===//
// micro.tile_store — write back to external memory
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_tile_store
func.func @test_tile_store(%tile : !micro.tile<32x64xbf16, memory = #micro.memory<acc>>) {
  // CHECK: micro.tile_store %{{.*}} {dst_memory = #micro.memory<dram>} : !micro.tile<32x64xbf16, memory = #micro.memory<acc>>
  micro.tile_store %tile {dst_memory = #micro.memory<dram>} : !micro.tile<32x64xbf16, memory = #micro.memory<acc>>
  return
}

//===----------------------------------------------------------------------===//
// micro.mma — tile-typed matrix engine fragment
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_tile_mma
func.func @test_tile_mma(%a : !micro.tile<16x32xbf16>, %b : !micro.tile<32x16xbf16>, %c : !micro.tile<16x16xf32>) {
  // CHECK: micro.mma %{{.*}}, %{{.*}}, %{{.*}} {accumulator = #micro.dtype<f32>, input = #micro.dtype<bf16>, shape = array<i64: 16, 16, 32>} : !micro.tile<16x32xbf16>, !micro.tile<32x16xbf16>, !micro.tile<16x16xf32> -> !micro.tile<16x16xf32>
  %r = micro.mma %a, %b, %c {shape = array<i64: 16, 16, 32>, input = #micro.dtype<bf16>, accumulator = #micro.dtype<f32>} : !micro.tile<16x32xbf16>, !micro.tile<32x16xbf16>, !micro.tile<16x16xf32> -> !micro.tile<16x16xf32>
  return
}

//===----------------------------------------------------------------------===//
// micro.vector — tile-typed elementwise fragment
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_tile_vector
func.func @test_tile_vector(%acc_g : !micro.tile<32x64xf32>) {
  // CHECK: micro.vector "silu" %{{.*}} : !micro.tile<32x64xf32> -> !micro.tile<32x64xf32>
  %gate = micro.vector "silu" %acc_g : !micro.tile<32x64xf32> -> !micro.tile<32x64xf32>
  return
}

//===----------------------------------------------------------------------===//
// micro.reduce — tile-typed reduction fragment
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_tile_reduce
func.func @test_tile_reduce(%scores : !micro.tile<32x64xf32>) {
  // CHECK: micro.reduce "max" %{{.*}} {axis = 1 : i64} : !micro.tile<32x64xf32> -> !micro.tile<32xf32>
  %m = micro.reduce "max" %scores {axis = 1 : i64} : !micro.tile<32x64xf32> -> !micro.tile<32xf32>
  return
}
