// RUN: llk-opt --allow-unregistered-dialect --triton-grid-to-forall --verify-diagnostics %s
// Grid blocks communicate through pointer outputs; values cannot escape forall.
// expected-error @+1 {{triton_grid_result_unsupported}}
func.func @return_references_cloned(%M: index, %BM: index) -> index {
  %pid = "tt.get_program_id"() {axis = 0 : i32} : () -> index
  %c2 = arith.constant 2 : index
  %result = arith.muli %pid, %c2 : index
  func.return %result : index
}
