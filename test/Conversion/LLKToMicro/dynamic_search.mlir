// RUN: llk-opt --llk-to-micro-search-space="schedule-db=missing.json" %s | FileCheck %s

// A dynamic root still exports a finite search space.
//
// The schedule database is missing, so the export anchors the space on its
// explicit bounded baseline. Unknown extents may never become search choices:
// every parameter below enumerates compile-time constants, and the tail
// policy offers the bounded boundary realization alongside the legacy
// policies.

func.func @dynamic_matmul_root(%a: tensor<?x?xbf16>, %b: tensor<?x?xbf16>,
                               %init: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
  %y = llk.matmul ins(%a, %b : tensor<?x?xbf16>, tensor<?x?xbf16>)
      outs(%init : tensor<?x?xbf16>)
      {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<?x?xbf16>
  return %y : tensor<?x?xbf16>
}

// CHECK-LABEL: func.func @dynamic_matmul_root
// CHECK: llk.matmul

// CHECK-LABEL: micro.search_space @matmul_dynamic attributes {workload = "matmul"} {
// CHECK: micro.param "BM" {choices = [1, 4, 8, 16, 32, 64], kind = "integer"}
// CHECK: micro.param "BN" {choices = [16, 32, 64, 128, 256], kind = "integer"}
// CHECK: micro.param "BK" {choices = [32, 64, 128, 256], kind = "integer"}
// CHECK: micro.param "vector_width" {choices = [8], kind = "integer"}
// CHECK: micro.param "pipeline_stages" {choices = [1, 2], kind = "integer"}
// The bounded baseline is anchored first; the legacy policies stay available
// as explicit alternatives rather than being inferred from an unknown extent.
// CHECK: micro.param "tail_policy" {choices = ["bounded", "none", "pad"], kind = "tail_policy"}
// CHECK: micro.constraint "tail_supported" {params = ["BM", "BN", "BK"]}
// CHECK: micro.objective {direction = "minimize", metric = "latency_cycles", secondary = ["matrix_utilization", "dram_bytes"]}
// CHECK: }

// Search choices remain finite constants: no dynamic sentinel reaches the
// space, and no execution kernel is emitted by this pass.
// CHECK-NOT: -9223372036854775808
// CHECK-NOT: micro.kernel
