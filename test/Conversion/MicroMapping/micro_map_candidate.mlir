// Fixture for micro_map_candidate.sh (phase-4 task 4): `--micro-map` driven by a
// persistent `micro.candidate`.
//
// One single-family kernel -- a staged copy plus one vector add -- so every
// node is on the vector/worker family. A mixed-family kernel (say a `micro.mma`
// in bf16 beside the vector add) could not be covered under one global bound
// layout, because the families declare different layout requirements; that is
// the recorded limitation, not something this fixture pretends away.
//
// The space names its layout-kind parameter `block_shape` on purpose: the pass
// must resolve "the layout parameter" by `kind`, never by name (`tile_layout`
// is the conventional spelling, and the shipped spaces use it). `VW` is an
// integer parameter the add rule also declares, so the candidate pins a rule
// parameter as well as the layout.
//
// The candidates each exercise one branch of the feature:
//   * @candidate_17        -- satisfiable: VW=8 is the module's vector width,
//                             block_shape=blocked is the id the rule declares.
//   * @candidate_pinned_low-- VW=4 contradicts `VW == lanes(f32) == 8`, so every
//                             rule for the vector node is a non-match: a
//                             binding-constrained rejection.
//   * @candidate_foreign_layout -- block_shape=row_major is a legal micro
//                             layout kind but not the id the rule declares, so
//                             the rule is a non-match: the namespace pair to
//                             binding_layouts.llkmap's `blocked`.

module {
  micro.search_space @binding_space attributes {workload = "binding_vector_add"} {
    micro.param "VW" {kind = "integer", choices = [4 : i64, 8 : i64]}
    micro.param "block_shape" {kind = "layout", choices = ["blocked", "row_major"]}
    micro.candidate @candidate_17 {bindings = {VW = 8 : i64, block_shape = "blocked"}}
    micro.candidate @candidate_pinned_low {bindings = {VW = 4 : i64, block_shape = "blocked"}}
    micro.candidate @candidate_foreign_layout {bindings = {VW = 8 : i64, block_shape = "row_major"}}
  }

  micro.kernel @mapped {
    %0 = tensor.empty() : tensor<8x8xf32>
    %result, %token = micro.async_copy %0 {dst_memory = #micro.memory<sram>, src_memory = #micro.memory<dram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    %1 = micro.tile_view %result {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %2 = micro.vector "add" %1, %1 : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}
