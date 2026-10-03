// RUN: llk-opt --llk-to-micro-search-space="schedule-db=missing.json" %s | FileCheck %s

// The llk-tune M-bucket rules are search legality, not generator branches: a
// bucket-0 problem is GEMV-like and only BM = 1 is a legal tile, and the
// large-M buckets reject tiles no larger than 4.
//
// The database is missing on purpose, so each operation falls back to the
// built-in conservative schedule (BM = 8). That is what makes the pruning
// visible: BM = 8 is in the grid, yet bucket 0 must drop it while a large-M
// bucket keeps it.

func.func @gemv(%a: tensor<1x64xbf16>, %b: tensor<64x64xbf16>,
                %init: tensor<1x64xbf16>) -> tensor<1x64xbf16> {
  %y = llk.matmul ins(%a, %b : tensor<1x64xbf16>, tensor<64x64xbf16>)
      outs(%init : tensor<1x64xbf16>)
      {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<1x64xbf16>
  return %y : tensor<1x64xbf16>
}

func.func @large_m(%a: tensor<32x64xbf16>, %b: tensor<64x64xbf16>,
                   %init: tensor<32x64xbf16>) -> tensor<32x64xbf16> {
  %y = llk.matmul ins(%a, %b : tensor<32x64xbf16>, tensor<64x64xbf16>)
      outs(%init : tensor<32x64xbf16>)
      {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
      -> tensor<32x64xbf16>
  return %y : tensor<32x64xbf16>
}

// M = 1 is bucket 0: BM = 1 is the only legal tile, so the scheduled BM = 8 is
// dropped rather than kept.
// CHECK-LABEL: micro.search_space @matmul_M1_N64_K64
// CHECK: micro.param "BM" {choices = [1], kind = "integer"}
// BM = 1 does not divide the problem, so the search space keeps the mask tail
// policy that makes it legal.
// CHECK: micro.param "tail_policy" {choices = ["mask"], kind = "tail_policy"}
// CHECK: micro.constraint "tail_supported" {params = ["BM", "BN", "BK"]}

// M = 32 is bucket 3: tiles of 4 or fewer rows are rejected, and the scheduled
// BM = 8 survives.
// CHECK-LABEL: micro.search_space @matmul_M32_N64_K64
// CHECK: micro.param "BM" {choices = [8, 16, 32, 64], kind = "integer"}
// CHECK: micro.objective {direction = "minimize", metric = "latency_cycles", secondary = ["matrix_utilization", "dram_bytes"]}
