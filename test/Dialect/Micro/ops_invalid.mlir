// RUN: llk-opt --verify-diagnostics --split-input-file %s

// Test that invalid micro dialect constructs produce actionable diagnostics.

// -----

func.func @kernel_empty_workload() {
  // expected-error @+1 {{workload attribute must be non-empty}}
  micro.kernel @bad attributes {workload = ""} {
    micro.yield
  }
  return
}

// -----

// Test that invalid memory space spelling fails with a diagnostic listing
// the valid values.

func.func @bad_memory() attributes {bad = #micro.memory<invalid>} {
  return
}
// expected-error @-3 {{expected ::mlir::micro::MemorySpace to be one of}}
// expected-error @-4 {{failed to parse Micro_MemorySpaceAttr}}

// -----

// Test that invalid mapping target spelling fails.

func.func @bad_map() attributes {bad = #micro.map<unknown>} {
  return
}
// expected-error @-3 {{expected ::mlir::micro::MappingTarget to be one of}}
// expected-error @-4 {{failed to parse Micro_MappingTargetAttr}}

// -----

// Test that invalid dtype spelling fails.

func.func @bad_dtype() attributes {bad = #micro.dtype<x64>} {
  return
}
// expected-error @-3 {{expected ::mlir::micro::DType to be one of}}
// expected-error @-4 {{failed to parse Micro_DTypeAttr}}

// -----

// micro.for step must be positive when statically known.
func.func @for_negative_step(%M : index) {
  %c0 = arith.constant 0 : index
  %cneg = arith.constant -1 : index
  // expected-error @+1 {{step must be positive}}
  micro.for %i = %c0 to %M step %cneg {
  }
  return
}

// -----

// micro.pipeline stages must be at least 1.
func.func @pipeline_zero_stages() {
  // expected-error @+1 {{pipeline stages must be at least 1}}
  micro.pipeline stages = 0 {
  }
  return
}

// -----

// micro.pipeline cannot be nested.
func.func @pipeline_nested() {
  micro.pipeline stages = 2 {
    // expected-error @+1 {{nested pipeline is not allowed}}
    micro.pipeline stages = 1 {
    }
  }
  return
}

// -----

// micro.alloc must not target dram.
func.func @alloc_in_dram() {
  // expected-error @+1 {{alloc memory must not be dram}}
  %x = micro.alloc {memory = #micro.memory<dram>} : tensor<32x64xf32>
  return
}

// -----

// micro.async_copy source and destination memory must differ.
func.func @async_copy_same_memory(%A : tensor<32x64xbf16>) {
  // expected-error @+1 {{source and destination memory must differ}}
  %dst, %tok = micro.async_copy %A {src_memory = #micro.memory<sram>, dst_memory = #micro.memory<sram>} : tensor<32x64xbf16> -> tensor<32x64xbf16>, !micro.async_token
  return
}

// -----

// micro.wait requires at least one token operand.
func.func @wait_no_tokens() {
  // expected-error @+1 {{wait requires at least one token operand}}
  micro.wait
  return
}

// -----

// micro.store source and destination memory must differ.
func.func @store_same_memory(%A : tensor<32x64xbf16>) {
  // expected-error @+1 {{source and destination memory must differ}}
  micro.store %A {src_memory = #micro.memory<sram>, dst_memory = #micro.memory<sram>} : tensor<32x64xbf16>
  return
}

// -----

// micro.kernel: the signature is a contract, so the body must produce exactly
// the declared results. Declaring one result and yielding none is a mismatch.
func.func @kernel_result_count_mismatch() {
  // expected-error @+1 {{body yields 0 value(s) but the signature declares 1 result(s)}}
  micro.kernel @bad(%a: tensor<8x8xf32>) -> tensor<8x8xf32> {
    micro.yield
  }
  return
}

// -----

// A yielded value whose logical shape disagrees with the declared result is
// refused: the caller would receive the wrong extent.
func.func @kernel_result_shape_mismatch() {
  // expected-error @+1 {{which does not match the declared result}}
  micro.kernel @bad(%a: tensor<8x8xf32>) -> tensor<8x4xf32> {
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield %ta : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
  return
}

// -----

// A matching shape is not enough: the element type is part of the contract too.
func.func @kernel_result_dtype_mismatch() {
  // expected-error @+1 {{which does not match the declared result}}
  micro.kernel @bad(%a: tensor<8x8xf32>) -> tensor<8x8xf16> {
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield %ta : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
  return
}

// -----

// `micro.kernel` is isolated from above: its body may only read its own
// declared inputs, never a value captured from the enclosing function. The
// isolation is enforced when the region is parsed, so the outer name is simply
// not in scope.
func.func @kernel_captures_an_outer_value(%x: tensor<8x8xf32>) {
  // expected-error @+2 {{use of undeclared SSA value name}}
  micro.kernel @bad(%a: tensor<8x8xf32>) {
    %ta = micro.tile_view %x {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    micro.yield
  }
  return
}

// -----

// A loop that carries a value must yield one every iteration: the block
// argument the body receives is the previous iteration's yield, so declaring a
// carried value and yielding nothing would leave it undefined.
func.func @loop_carried_value_not_yielded() {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %init = micro.tile_alloc : !micro.tile<8x32xf32, memory = #micro.memory<acc>>
  // expected-error @+1 {{body yields 0 value(s) but the loop carries 1}}
  %acc = micro.for %k = %c0 to %c64 step %c32 iter_args(%carried = %init) -> (!micro.tile<8x32xf32, memory = #micro.memory<acc>>) {
    micro.yield
  }
  return
}

// -----

// The yielded value must have the carried type: the loop's result is the last
// yield, so a different type hands the next iteration the wrong thing.
func.func @loop_yields_the_wrong_type() {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %init = micro.tile_alloc : !micro.tile<8x32xf32, memory = #micro.memory<acc>>
  %other = micro.tile_alloc : !micro.tile<8x64xf32, memory = #micro.memory<acc>>
  // expected-error @+1 {{yielded value #0 has type}}
  %acc = micro.for %k = %c0 to %c64 step %c32 iter_args(%carried = %init) -> (!micro.tile<8x32xf32, memory = #micro.memory<acc>>) {
    micro.yield %other : !micro.tile<8x64xf32, memory = #micro.memory<acc>>
  }
  return
}
