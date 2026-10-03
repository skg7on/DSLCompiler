// The `micro-perf` end-to-end fixture: a concrete tile GEMM, plus the checks
// the CLI test applies to its own report. The CHECK lines describe the report;
// the HELP lines describe the tool's `--help` text, which is checked by a
// second FileCheck run with --check-prefix=HELP. Both live in this file because
// the acceptance criteria for the tool cover both, and neither run sees the
// other's prefixes.
//
// RUN: micro-perf --machine=machines/x86-avx2-v2.yaml --level=1 %s | FileCheck %s
// RUN: micro-perf --help 2>&1 | FileCheck --check-prefix=HELP %s

// A concrete tile GEMM: stage A and B from DRAM through SRAM, run one
// 16x16x32 bf16 MMA fragment into an f32 accumulator, and write the result
// back to DRAM. Every number below is derived from these tile shapes and
// dtypes, so the byte and flop checks do not depend on the machine model.

// HELP: This is a performance simulator, not a functional emulator

module {
  micro.kernel @gemm_tile attributes {workload = "gemm", target = "x86-avx2-cpu"} {
    %a_ext = tensor.empty() : tensor<16x32xbf16>
    %b_ext = tensor.empty() : tensor<32x16xbf16>
    %a_tile, %a_tok = micro.async_copy %a_ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<16x32xbf16> -> tensor<16x32xbf16>, !micro.async_token
    %b_tile, %b_tok = micro.async_copy %b_ext {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<32x16xbf16> -> tensor<32x16xbf16>, !micro.async_token
    micro.wait %a_tok, %b_tok

    %a_frag = micro.tile_view %a_tile {shape = array<i64: 16, 32>} : tensor<16x32xbf16> -> !micro.tile<16x32xbf16, memory = #micro.memory<sram>>
    %b_frag = micro.tile_view %b_tile {shape = array<i64: 32, 16>} : tensor<32x16xbf16> -> !micro.tile<32x16xbf16, memory = #micro.memory<sram>>
    %acc = micro.tile_alloc : !micro.tile<16x16xf32, memory = #micro.memory<acc>>
    %result = micro.mma %a_frag, %b_frag, %acc {shape = array<i64: 16, 16, 32>, input = #micro.dtype<bf16>, accumulator = #micro.dtype<f32>} : !micro.tile<16x32xbf16, memory = #micro.memory<sram>>, !micro.tile<32x16xbf16, memory = #micro.memory<sram>>, !micro.tile<16x16xf32, memory = #micro.memory<acc>> -> !micro.tile<16x16xf32, memory = #micro.memory<acc>>
    micro.tile_store %result {dst_memory = #micro.memory<dram>} : !micro.tile<16x16xf32, memory = #micro.memory<acc>>

    micro.yield
  }
}

// CHECK: schema_version: 1
// CHECK: machine: x86-avx2
// CHECK: kernel: gemm_tile
// CHECK: level: 1
// CHECK: totals:
// CHECK:   flops: 16384
// CHECK:   bytes:
// CHECK:     acc: 1024
// CHECK:     dram: 3072
// CHECK:     sram: 2048
// CHECK: operations:
// CHECK:   async_copy: 2
// CHECK:   mma: 1
// CHECK:   store: 1
// CHECK:   wait: 1
// CHECK: bounds:
// CHECK:   compute_cycles:
// CHECK:   memory_cycles:
// CHECK: predicted_cycles:
// CHECK: predicted_ns:
// CHECK: utilization:
// CHECK:   matrix:
// CHECK:   vector:
// CHECK:   dma:
// CHECK: bandwidth:
// CHECK:   dram:
// CHECK:   sram:
// CHECK: overlap_efficiency:
// CHECK: tiles:
// CHECK:   live_bytes:
// CHECK:     acc: 1024
// CHECK:     sram: 2048
// CHECK:   layouts: {}
// CHECK:   owners: {}
// CHECK: bottleneck: dma
// CHECK: capacity_violations: []
// CHECK: layout_warnings: []
// CHECK: owner_warnings: []
// The shipped machine lists no acc -> dram path, so the store's cost is
// composed from both endpoints and the report says so rather than inventing a
// cost silently.
// CHECK: warnings:
// CHECK:   - "machine 'x86-avx2' declares no copy path 'acc -> dram'; its cost is composed from both endpoints"
