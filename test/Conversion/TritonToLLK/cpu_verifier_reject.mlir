// RUN: llk-opt --allow-unregistered-dialect --triton-cpu-verify %s 2>&1 | FileCheck %s

// Unsupported GPU-only Triton ops must be rejected with an actionable error
// before any lowering runs. tt.inline_asm has no CPU lowering.

func.func @unsupported_inline_asm(%str: !tt.ptr) {
  // CHECK: error: Triton op 'tt.inline_asm' is unsupported on CPU
  // CHECK: See the CPU compatibility guide for supported operations
  "tt.inline_asm"(%str) : (!tt.ptr) -> ()
  func.return
}
