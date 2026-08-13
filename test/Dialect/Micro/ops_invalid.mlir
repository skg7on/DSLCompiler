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
