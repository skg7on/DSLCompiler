// RUN: llk-opt --verify-diagnostics --split-input-file %s

// A singleton search space is parameter-free and carries at most one candidate
// that binds nothing. An ordinary empty space remains an error.

//===----------------------------------------------------------------------===//
// singleton shape
//===----------------------------------------------------------------------===//

// A singleton declares no tunable parameter: there is nothing to search.

func.func @singleton_with_params() {
  // expected-error @+1 {{singleton search space cannot declare parameters}}
  micro.search_space @bad attributes {singleton = true, workload = "elementwise"} {
    micro.param "T0" {kind = "integer", choices = [4 : i64]}
  }
  return
}

//===----------------------------------------------------------------------===//
// singleton candidates
//===----------------------------------------------------------------------===//

// Its one candidate binds nothing: a singleton has no parameter to name.

func.func @singleton_bound_candidate() {
  micro.search_space @bad attributes {singleton = true, workload = "elementwise"} {
    // expected-error @+1 {{singleton candidate must not bind parameters}}
    micro.candidate @only {bindings = {T0 = 4 : i64}}
  }
  return
}

// At most one explicit candidate is admitted.

func.func @singleton_two_candidates() {
  // expected-error @+1 {{singleton search space admits at most one candidate}}
  micro.search_space @bad attributes {singleton = true, workload = "elementwise"} {
    micro.candidate @first {bindings = {}}
    micro.candidate @second {bindings = {}}
  }
  return
}

//===----------------------------------------------------------------------===//
// ordinary empty spaces
//===----------------------------------------------------------------------===//

// Omitting `singleton` does not turn an empty space into a one-point domain.

func.func @empty_space_is_still_an_error() {
  // expected-error @+1 {{search space requires at least one micro.param}}
  micro.search_space @bad attributes {workload = "elementwise"} {
    micro.candidate @only {bindings = {}}
  }
  return
}

// An explicitly false singleton is not a singleton either.

func.func @singleton_false_is_an_empty_space() {
  // expected-error @+1 {{search space requires at least one micro.param}}
  micro.search_space @bad attributes {singleton = false, workload = "elementwise"} {
  }
  return
}
