// The direct Micro-IR elementwise fixture for Dynamic Shape Task 6.
//
// `test/Execution/dynamic_mapped_acceptance.cpp` parses this file, attaches the
// canonical symbolic contract and executes it in both execution modes. It
// exists to prove that the dynamic shape and boundary machinery is a property
// of the tile IR, not of contraction adapters: the kernel has no M/N/K axes, no
// accumulators and no reduction. It only loads bounded rectangles from two
// runtime-shaped tensors, adds them elementwise, and writes the clipped result
// into a caller-owned destination.
//
// Every tile keeps a static 8x16 capacity; only the boundary rectangles and the
// loop trip counts are runtime values. The destination is an explicit output
// argument (`argument_bindings`/`result_bindings`), so the whole output never
// becomes local storage.
//
// The RUN line is documentation only: the project has no lit runner. The real
// invocation is the MicroToLinalgDynamicElementwise CTest entry, which runs
// `llk-opt --micro-to-linalg` on this file and checks the clipped accesses.

// RUN: llk-opt --micro-to-linalg %s | FileCheck %s

// Both reads are clipped to the runtime valid rectangle of the tile.
// CHECK-LABEL: func.func @elementwise_add_dynamic
// CHECK: tensor.extract_slice %{{.*}}[%{{.*}}, %{{.*}}] [%{{.*}}, %{{.*}}] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
// CHECK: tensor.extract_slice %{{.*}}[%{{.*}}, %{{.*}}] [%{{.*}}, %{{.*}}] [1, 1] : tensor<?x?xf32> to tensor<?x?xf32>
// The write inserts only the valid rectangle into the caller's destination.
// CHECK: tensor.insert_slice {{.*}} into %{{.*}}[%{{.*}}, %{{.*}}] [%{{.*}}, %{{.*}}] [1, 1] : tensor<?x?xf32> into tensor<?x?xf32>
// CHECK: return

module {
  micro.kernel @elementwise_add_dynamic(%a: tensor<?x?xf32>, %b: tensor<?x?xf32>,
                                        %out: tensor<?x?xf32>)
      -> tensor<?x?xf32>
      attributes {
        argument_bindings = [{port = 0 : i64, role = "input"},
                             {port = 1 : i64, role = "input"},
                             {port = 2 : i64, role = "output"}],
        result_bindings = [2]
      } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %m = tensor.dim %a, %c0 : tensor<?x?xf32>
    %n = tensor.dim %a, %c1 : tensor<?x?xf32>

    // The caller's storage is a view, never an allocation.
    %dst = micro.tile_view %out[%c0, %c0]
        {shape = array<i64: -9223372036854775808, -9223372036854775808>}
        : tensor<?x?xf32> -> !micro.tile<?x?xf32, memory = #micro.memory<dram>>

    // ceildiv(extent, tile) = quotient + (remainder != 0), the checked form
    // that cannot overflow by adding tile - 1.
    %mtq = arith.divui %m, %c8 : index
    %mtr = arith.remui %m, %c8 : index
    %mtz = arith.cmpi ne, %mtr, %c0 : index
    %mtb = arith.select %mtz, %c1, %c0 : index
    %mt = arith.addi %mtq, %mtb : index
    %ntq = arith.divui %n, %c16 : index
    %ntr = arith.remui %n, %c16 : index
    %ntz = arith.cmpi ne, %ntr, %c0 : index
    %ntb = arith.select %ntz, %c1, %c0 : index
    %nt = arith.addi %ntq, %ntb : index

    %result = micro.spatial_for %i = %c0 to %mt step %c1 map = #micro.map<worker>
        iter_args(%acc = %dst) -> (!micro.tile<?x?xf32, memory = #micro.memory<dram>>) {
      %ioff = arith.muli %i, %c8 : index
      %mrem = arith.subi %m, %ioff : index
      %vm = arith.minsi %mrem, %c8 : index
      %inner = micro.spatial_for %j = %c0 to %nt step %c1 map = #micro.map<lane>
          iter_args(%acc2 = %acc) -> (!micro.tile<?x?xf32, memory = #micro.memory<dram>>) {
        %joff = arith.muli %j, %c16 : index
        %nrem = arith.subi %n, %joff : index
        %vn = arith.minsi %nrem, %c16 : index
        %ta, %taToken = micro.tile_load %a[%ioff, %joff] valid[%vm, %vn]
            {dst_memory = #micro.memory<sram>, owner = #micro.owner<worker>,
             padding = 0.000000e+00 : f32}
            : tensor<?x?xf32>
           -> !micro.tile<8x16xf32, layout = #micro.layout<row_major, vector = 8>,
                          memory = #micro.memory<sram>, owner = #micro.owner<worker>>,
              !micro.async_token
        %tb, %tbToken = micro.tile_load %b[%ioff, %joff] valid[%vm, %vn]
            {dst_memory = #micro.memory<sram>, owner = #micro.owner<worker>,
             padding = 0.000000e+00 : f32}
            : tensor<?x?xf32>
           -> !micro.tile<8x16xf32, layout = #micro.layout<row_major, vector = 8>,
                          memory = #micro.memory<sram>, owner = #micro.owner<worker>>,
              !micro.async_token
        micro.wait %taToken, %tbToken
        %sum = micro.vector "add" %ta, %tb
            : !micro.tile<8x16xf32, layout = #micro.layout<row_major, vector = 8>,
                          memory = #micro.memory<sram>, owner = #micro.owner<worker>>,
              !micro.tile<8x16xf32, layout = #micro.layout<row_major, vector = 8>,
                          memory = #micro.memory<sram>, owner = #micro.owner<worker>>
           -> !micro.tile<8x16xf32, memory = #micro.memory<acc>,
                          owner = #micro.owner<worker>>
        %written = micro.tile_store %sum into %acc2[%ioff, %joff] valid[%vm, %vn]
            {dst_memory = #micro.memory<dram>}
            : !micro.tile<8x16xf32, memory = #micro.memory<acc>,
                          owner = #micro.owner<worker>>,
              !micro.tile<?x?xf32, memory = #micro.memory<dram>>
           -> !micro.tile<?x?xf32, memory = #micro.memory<dram>>
        micro.yield %written : !micro.tile<?x?xf32, memory = #micro.memory<dram>>
      }
      micro.yield %inner : !micro.tile<?x?xf32, memory = #micro.memory<dram>>
    }
    micro.yield %result : !micro.tile<?x?xf32, memory = #micro.memory<dram>>
  }
}
