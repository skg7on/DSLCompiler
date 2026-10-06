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

// The owner is an *open* symbol: a spelling the dialect does not recognize is
// unresolved analysis data, not a parse error, and it round-trips verbatim.
// Whether `warp` names a class is the owning target's machine model's answer.
// CHECK-LABEL: func.func @test_owner_open_symbol
// CHECK: #micro.owner<warp>
// CHECK: #micro.owner<"worker/lane">
func.func @test_owner_open_symbol() attributes {
  owners = [
    #micro.owner<warp>,
    #micro.owner<"worker/lane">
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
// micro.tile_view — logical zero-cost view (whole and windowed)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_tile_view
func.func @test_tile_view(%A : tensor<?x?xbf16>) {
  // CHECK: micro.tile_view %{{.*}} {shape = array<i64: 32, 64>} : tensor<?x?xbf16> -> !micro.tile<32x64xbf16>
  %t = micro.tile_view %A {shape = array<i64: 32, 64>} : tensor<?x?xbf16> -> !micro.tile<32x64xbf16>
  return
}

// CHECK-LABEL: func.func @test_tile_view_offsets
func.func @test_tile_view_offsets(%A : tensor<?x?xbf16>, %i : index, %j : index) {
  // CHECK: micro.tile_view %{{.*}}[%{{.*}}, %{{.*}}] {shape = array<i64: 32, 64>} : tensor<?x?xbf16> -> !micro.tile<32x64xbf16>
  %t = micro.tile_view %A[%i, %j] {shape = array<i64: 32, 64>} : tensor<?x?xbf16> -> !micro.tile<32x64xbf16>
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
  // CHECK: micro.tile_partition %{{.*}} {owner = #micro.owner<vector_engine>, shape = array<i64: 16, 16>} : !micro.tile<32x64xbf16> -> !micro.tile<16x16xbf16, owner = #micro.owner<vector_engine>>
  %frag = micro.tile_partition %tile {shape = array<i64: 16, 16>, owner = #micro.owner<vector_engine>} : !micro.tile<32x64xbf16> -> !micro.tile<16x16xbf16, owner = #micro.owner<vector_engine>>
  return
}

// CHECK-LABEL: func.func @test_tile_partition_tail
func.func @test_tile_partition_tail(%tile : !micro.tile<32x64xbf16>) {
  // CHECK: micro.tile_partition %{{.*}} {shape = array<i64: 7, 7>, tail = true} : !micro.tile<32x64xbf16> -> !micro.tile<7x7xbf16>
  %frag = micro.tile_partition %tile {shape = array<i64: 7, 7>, tail = true} : !micro.tile<32x64xbf16> -> !micro.tile<7x7xbf16>
  return
}

//===----------------------------------------------------------------------===//
// micro.tile_async_copy — async movement returning tile + token
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_tile_async_copy
func.func @test_tile_async_copy(%logical : !micro.tile<32x64xbf16>) {
  // CHECK: %[[T:.*]], %[[TOK:.*]] = micro.tile_async_copy %{{.*}} {dst_memory = #micro.memory<sram>, owner = #micro.owner<worker>} : !micro.tile<32x64xbf16> -> !micro.tile<32x64xbf16, memory = #micro.memory<sram>, owner = #micro.owner<worker>>, !micro.async_token
  %tile, %tok = micro.tile_async_copy %logical {dst_memory = #micro.memory<sram>, owner = #micro.owner<worker>} : !micro.tile<32x64xbf16> -> !micro.tile<32x64xbf16, memory = #micro.memory<sram>, owner = #micro.owner<worker>>, !micro.async_token
  return
}

// A movement between two *distinct* concrete memories of one abstract space is
// real work. Kind equality is not node identity, so the copy records the two
// concrete node ids it moves between (`micro.src_node`/`micro.dst_node`). The
// abstract memory kind stays `sram` on both the source and the result tile.
// CHECK-LABEL: func.func @test_tile_async_copy_same_kind_distinct_nodes
func.func @test_tile_async_copy_same_kind_distinct_nodes(%tile : !micro.tile<32x64xbf16, memory = #micro.memory<sram>>) {
  // CHECK: micro.tile_async_copy %{{.*}} {dst_memory = #micro.memory<sram>, micro.dst_node = "sram.1", micro.src_node = "sram.0"} : !micro.tile<32x64xbf16, memory = #micro.memory<sram>> -> !micro.tile<32x64xbf16, memory = #micro.memory<sram>>, !micro.async_token
  %dst, %tok = micro.tile_async_copy %tile {dst_memory = #micro.memory<sram>, micro.src_node = "sram.0", micro.dst_node = "sram.1"} : !micro.tile<32x64xbf16, memory = #micro.memory<sram>> -> !micro.tile<32x64xbf16, memory = #micro.memory<sram>>, !micro.async_token
  return
}

//===----------------------------------------------------------------------===//
// micro.async_copy — same-kind movement between distinct concrete nodes
//===----------------------------------------------------------------------===//

// A shaped value carries both spaces as attributes; a same-kind move carries
// the source and destination node identities the same way.
// CHECK-LABEL: func.func @test_async_copy_same_kind_distinct_nodes
func.func @test_async_copy_same_kind_distinct_nodes(%t : tensor<8x8xf32>) {
  // CHECK: micro.async_copy %{{.*}} {dst_memory = #micro.memory<sram>, micro.dst_node = "sram.1", micro.src_node = "sram.0", src_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
  %r, %tok = micro.async_copy %t {src_memory = #micro.memory<sram>, dst_memory = #micro.memory<sram>, micro.src_node = "sram.0", micro.dst_node = "sram.1"} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
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
// micro.transform — re-represent a value under a different layout
//===----------------------------------------------------------------------===//

// A layout transform names the two layouts by their index relation, not by a
// target-owned id, so the operation stays target-neutral. The maps print in
// attribute-name order (dst before src), through MLIR's affine-map aliases
// (`#map`), which the second `llk-opt` pass re-parses -- so the round trip
// checks the maps themselves.
// CHECK-LABEL: func.func @test_tile_transform
func.func @test_tile_transform(%acc : !micro.tile<32x64xf32>) {
  // CHECK: micro.transform %{{.*}} {dst_map = #{{.*}}, src_map = #{{.*}}} : !micro.tile<32x64xf32> -> !micro.tile<32x64xf32>
  %r = micro.transform %acc {src_map = affine_map<(d0, d1) -> (d0, d1 floordiv 8, d1 mod 8)>, dst_map = affine_map<(d0, d1) -> (d0, d1)>} : !micro.tile<32x64xf32> -> !micro.tile<32x64xf32>
  return
}

// A layout that declares no map clause leaves its side to the target: the
// operation carries only what is known, and an empty attribute dictionary is
// valid.
// CHECK-LABEL: func.func @test_tile_transform_no_maps
func.func @test_tile_transform_no_maps(%acc : !micro.tile<32x64xf32>) {
  // CHECK: micro.transform %{{.*}} : !micro.tile<32x64xf32> -> !micro.tile<32x64xf32>
  %r = micro.transform %acc : !micro.tile<32x64xf32> -> !micro.tile<32x64xf32>
  return
}

// A tensor-typed transform is legal too: the bound plan's movement ops are
// emitted over the values the workload graph carries, which may be tensors.
// CHECK-LABEL: func.func @test_tensor_transform
func.func @test_tensor_transform(%t : tensor<32x64xf32>) {
  // CHECK: micro.transform %{{.*}} {dst_map = #{{.*}}} : tensor<32x64xf32> -> tensor<32x64xf32>
  %r = micro.transform %t {dst_map = affine_map<(d0, d1) -> (d0, d1)>} : tensor<32x64xf32> -> tensor<32x64xf32>
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
  // CHECK: micro.vector "silu" %{{.*}} {math_mode = "bounded_fast"} : !micro.tile<32x64xf32> -> !micro.tile<32x64xf32>
  %gate = micro.vector "silu" %acc_g {math_mode = "bounded_fast"} : !micro.tile<32x64xf32> -> !micro.tile<32x64xf32>
  // CHECK: micro.vector "reciprocal" %{{.*}} : !micro.tile<32x64xf32> -> !micro.tile<32x64xf32>
  %rec = micro.vector "reciprocal" %acc_g : !micro.tile<32x64xf32> -> !micro.tile<32x64xf32>
  return
}

//===----------------------------------------------------------------------===//
// micro.reduce — tile-typed reduction fragment
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_tile_reduce
func.func @test_tile_reduce(%scores : !micro.tile<32x64xf32>) {
  // CHECK: micro.reduce "max" %{{.*}} {axis = 1 : i64} : !micro.tile<32x64xf32> -> !micro.tile<32xf32>
  %m = micro.reduce "max" %scores {axis = 1 : i64} : !micro.tile<32x64xf32> -> !micro.tile<32xf32>
  // CHECK: micro.reduce "product" %{{.*}} {axis = 0 : i64} : !micro.tile<32x64xf32> -> !micro.tile<64xf32>
  %p = micro.reduce "product" %scores {axis = 0 : i64} : !micro.tile<32x64xf32> -> !micro.tile<64xf32>
  return
}

//===----------------------------------------------------------------------===//
// micro.gather — combine several producers under explicit semantics
//===----------------------------------------------------------------------===//

// The combination is explicit and round-trips: `sum` and `max` combine
// like-for-like shape/type, and carry no axis.
// CHECK-LABEL: func.func @test_gather_sum
func.func @test_gather_sum(%a : !micro.tile<32x64xf32>, %b : !micro.tile<32x64xf32>) {
  // CHECK: micro.gather %{{.*}}, %{{.*}} kind = "sum" : !micro.tile<32x64xf32>, !micro.tile<32x64xf32> -> !micro.tile<32x64xf32>
  %r = micro.gather %a, %b kind = "sum" : !micro.tile<32x64xf32>, !micro.tile<32x64xf32> -> !micro.tile<32x64xf32>
  return
}

// CHECK-LABEL: func.func @test_gather_max
func.func @test_gather_max(%a : !micro.tile<32x64xf32>, %b : !micro.tile<32x64xf32>) {
  // CHECK: micro.gather %{{.*}}, %{{.*}} kind = "max" : !micro.tile<32x64xf32>, !micro.tile<32x64xf32> -> !micro.tile<32x64xf32>
  %r = micro.gather %a, %b kind = "max" : !micro.tile<32x64xf32>, !micro.tile<32x64xf32> -> !micro.tile<32x64xf32>
  return
}

// A concatenation names the axis and produces exactly the summed extent along
// it, with every other extent matching.
// CHECK-LABEL: func.func @test_gather_concat
func.func @test_gather_concat(%a : !micro.tile<32x64xf32>, %b : !micro.tile<32x32xf32>) {
  // CHECK: micro.gather %{{.*}}, %{{.*}} kind = "concat" axis = 1 : !micro.tile<32x64xf32>, !micro.tile<32x32xf32> -> !micro.tile<32x96xf32>
  %r = micro.gather %a, %b kind = "concat" axis = 1 : !micro.tile<32x64xf32>, !micro.tile<32x32xf32> -> !micro.tile<32x96xf32>
  return
}

//===----------------------------------------------------------------------===//
// micro.barrier — executor-group barrier over dependency tokens
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @test_barrier
func.func @test_barrier(%tok : !micro.async_token) {
  // CHECK: micro.barrier %{{.*}} scope = "executor_group"
  micro.barrier %tok scope = "executor_group"
  return
}

// A collective barrier carries no token: the group itself is the dependency.
// CHECK-LABEL: func.func @test_barrier_collective
func.func @test_barrier_collective() {
  // CHECK: micro.barrier scope = "executor_group"
  micro.barrier scope = "executor_group"
  return
}

