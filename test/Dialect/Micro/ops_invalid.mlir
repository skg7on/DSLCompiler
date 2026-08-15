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
