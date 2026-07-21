// RUN: llk-opt --verify-diagnostics --scratch-analysis --split-input-file %s
//
// Test: ScratchAnalysis pass audits memref.alloc ops and warns on
// allocations larger than 1 MB (potential full-size intermediates).
//
// Test 1: Large 1024x1024 f32 allocation (4 MB) triggers a warning.
// Test 2: Small 128x128 f32 allocation (64 KB) passes silently.
// Test 3: Dynamic shape allocation (unknown dimension) is skipped.
// Test 4: BF16 large allocation (2 MB) triggers warning.

// -----

// expected-warning@+2 {{Large allocation}}
func.func @large_alloc_f32() {
  %buf = memref.alloc() : memref<1024x1024xf32>
  return
}

// -----

func.func @small_alloc_f32() {
  %buf = memref.alloc() : memref<128x128xf32>
  return
}

// -----

func.func @dynamic_shape(%arg0: index) {
  %buf = memref.alloc(%arg0) : memref<?x128xf32>
  return
}

// -----

// expected-warning@+2 {{Large allocation}}
func.func @large_alloc_bf16() {
  %buf = memref.alloc() : memref<1024x1024xbf16>
  return
}
