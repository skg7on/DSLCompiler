// schedules/x86_avx2/schedule_bf16.mlir
// AVX2 BF16 fused SwiGLU schedule
// Parameters: BM=32, BN=64, BK=64, VM=4, VN=8
//
// This schedule is applied by the --tile-and-vectorize pass after
// LLK ops have been lowered to Linalg (linalg.matmul + linalg.generic).

module attributes {transform.with_named_sequence} {
  transform.named_sequence @schedule_avx2_bf16(
      %root: !transform.any_op {transform.readonly}) {

    // --- Matmul ops (gate and up projections) ---
    // These have K as a reduction dimension that must be tiled with scf.for.
    %matmuls = transform.structured.match ops{["linalg.matmul"]}
      in %root
      : (!transform.any_op) -> !transform.any_op

    // Step 1a: Tile the K reduction dimension with scf.for (BEFORE parallel tiling).
    // Returns (fill_op, tiled_op, combining_op, for_op).
    %fill, %k_tiled, %combine, %k_loops = transform.structured.tile_reduction_using_for
        %matmuls by tile_sizes = [64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)

    // Step 1b: Tile MxN parallel dims on the K-tiled matmuls with scf.forall.
    %tiled_matmul, %forall_matmul = transform.structured.tile_using_forall %k_tiled
        tile_sizes [32, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Step 1c: Vectorize the innermost matmul tiles.
    transform.structured.vectorize %tiled_matmul
        vector_sizes [4, 8]
      : !transform.any_op

    // --- Generic op (SiLU activation + elementwise multiply) ---
    // This is a pure parallel op (no reduction dim), so only forall tiling.
    %generics = transform.structured.match ops{["linalg.generic"]}
      in %root
      : (!transform.any_op) -> !transform.any_op

    // Step 2a: Tile MxN parallel dims with scf.forall.
    %tiled_generic, %forall_generic = transform.structured.tile_using_forall %generics
        tile_sizes [32, 64]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Step 2b: Vectorize the innermost generic tiles.
    transform.structured.vectorize %tiled_generic
        vector_sizes [4, 8]
      : !transform.any_op

    transform.yield
  }
}
