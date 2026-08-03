// RUN: llk-opt --allow-unregistered-dialect --verify-each=false --triton-grid-to-forall %s | FileCheck %s

// Regression test for Task 6 Finding A: a func.return operand that consumes
// a value produced by an op cloned into the forall body must be remapped to
// the clone before the original op is erased.  Before the fix, erasing the
// original op (which still had a use) aborted in Operation::erase().
//
// --verify-each=false is required: the remapped return references a value
// defined inside the scf.forall body, which is a pre-existing dominance
// limitation of the clone-into-forall design (not fixed by this change).
// The point of this test is to pin the no-crash + remap behavior.

// CHECK-LABEL: return_references_cloned
func.func @return_references_cloned(%M: index, %BM: index) -> index {
  // CHECK: "arith.ceildivsi"
  // CHECK: "scf.forall"
  %pid = "tt.get_program_id"() {axis = 0 : i32} : () -> index
  %c2 = arith.constant 2 : index
  %result = arith.muli %pid, %c2 : index
  // CHECK: "arith.muli"
  // CHECK: "func.return"(%{{.*}})
  // CHECK-NOT: tt.get_program_id
  func.return %result : index
}
