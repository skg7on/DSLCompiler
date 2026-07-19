// schedules/x86_avx2/schedule_bf16.mlir
// AVX2 BF16 fused SwiGLU schedule
// Parameters: BM=32, BN=64, BK=64, VM=4, VN=8
//
// This schedule is applied by the --tile-and-vectorize pass.

module attributes {transform.with_named_sequence} {
  transform.named_sequence @schedule_avx2_bf16(
      %root: !transform.any_op {transform.readonly}) {

    // Step 1: Find the fused_swiglu operation
    %swiglu = transform.structured.match ops{["llk.fused_swiglu"]} in %root
      : (!transform.any_op) -> !transform.any_op

    // Step 2: Tile outer MxN parallel loops using scf.forall.
    // Returns (tiled_op, forall_op).
    %tiled, %loops = transform.structured.tile_using_forall %swiglu
        tile_sizes [32, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Step 3: Fuse both matmul producers into the consumer tile.
    // Returns (fused_op, new_containing_op).
    %fused, %new_containing = transform.structured.fuse_into_containing_op
        %tiled into %loops
      : (!transform.any_op, !transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Step 4: Tile the K reduction dimension.
    // Returns (fill_op, split_op, combining_op, for_op).
    %fill, %k_tiled, %combine, %k_loops = transform.structured.tile_reduction_using_for
        %fused by tile_sizes = [64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // Step 5: Vectorize the innermost tiles.
    transform.structured.vectorize %k_tiled
        vector_sizes [4, 8]
      : !transform.any_op

    transform.yield
  }
}
