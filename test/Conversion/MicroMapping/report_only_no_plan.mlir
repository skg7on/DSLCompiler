// The negative fixture for `report-only=1`. The kernel's only work node is a
// `micro.vector "sub"`, which no rule in mapping/x86-avx2/rules.llkmap matches,
// so the search produces no complete plan and the pass must fail -- report-only
// included. A report-only run that quietly succeeded here would report nothing
// and hide the failure, which is exactly what this fixture pins.
//
// The kernel body is deliberately the same shape as report_only_kernel.mlir
// with only the op string changed, so the fixture differs in one dimension.
//
// The RUN line below is documentation only: this project has no lit runner, and
// `//` lines are comments. The real invocation is the MicroMappingReportOnly
// CTest entry in CMakeLists.txt, which supplies the same option string with
// absolute paths and checks the failure message and the absent report file.
//
// RUN: not llk-opt %s --micro-map="target=x86-avx2 machine=%p/../../../machines/x86-avx2-v2.yaml layouts=%p/../../../mapping/x86-avx2/layouts.llkmap rules=%p/../../../mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store mode=deterministic report-only=1 report=%t"

module {
  micro.kernel @unmappable {
    %0 = tensor.empty() : tensor<8x8xf32>
    %result, %token = micro.async_copy %0 {dst_memory = #micro.memory<sram>, src_memory = #micro.memory<dram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    %1 = micro.tile_view %result {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %2 = micro.vector "sub" %1, %1 : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}
