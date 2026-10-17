// RUN: llk-opt %s | llk-opt | FileCheck %s

// A source with no tunable parameter has exactly one schedule. Such a space
// says so explicitly with `singleton = true`: it declares no parameter and
// carries at most one candidate, which binds nothing. The attribute and the
// empty binding round-trip unchanged.

// CHECK-LABEL: func.func @singleton_space
func.func @singleton_space() {
  // CHECK: micro.search_space @elementwise attributes {singleton = true, workload = "elementwise"} {
  micro.search_space @elementwise attributes {singleton = true, workload = "elementwise"} {
    // CHECK: micro.candidate @only {bindings = {}}
    micro.candidate @only {bindings = {}}
  }
  return
}

// A singleton may declare an objective; only parameters are forbidden.
// CHECK-LABEL: func.func @singleton_with_objective
func.func @singleton_with_objective() {
  // CHECK: micro.search_space @direct attributes {singleton = true, workload = "direct"} {
  micro.search_space @direct attributes {singleton = true, workload = "direct"} {
    // CHECK: micro.objective {direction = "minimize", metric = "latency_cycles"}
    micro.objective {direction = "minimize", metric = "latency_cycles"}
  }
  return
}
