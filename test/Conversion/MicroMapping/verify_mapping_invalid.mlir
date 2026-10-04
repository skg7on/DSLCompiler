// RUN: llk-opt --verify-diagnostics --split-input-file %s \
// RUN:   --micro-verify-mapping="target=x86-avx2 machine=%p/../../../machines/x86-avx2-v2.yaml layouts=%p/../../../mapping/x86-avx2/layouts.llkmap rules=%p/../../../mapping/x86-avx2/rules.llkmap emitters=avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store"
//
// The RUN line above is documentation only: this project has no lit runner, and
// `//` lines are comments. The real invocation is the MicroMappingVerifyInvalid
// CTest entry in CMakeLists.txt, which supplies the same option string with
// absolute paths.
//
// Phase 2 (design §18.3) resolves every id a mapped kernel records, and reports
// the first violation as a stable §22.3 code so a caller can group or switch on
// it rather than match prose. Each chunk below isolates one recorded id and
// asserts the code *and* the specific detail (which operation, which id). The
// per-operation walk checks, in order, rule -> executor -> memories -> layouts
// -> emitter, so a chunk that leaves an earlier field valid reaches the later
// one.
//
// `--verify-diagnostics` binds a pass diagnostic emitted on the module to the
// `module {` line, so `expected-error @below` immediately above that line is
// what each chunk uses.

// Phase 2 accepts a fully resolved mapping: this chunk has no expected
// diagnostic, and the run only succeeds if the pass leaves it alone.
module {
  micro.kernel @k {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.vector_add", executor = "worker.0", emitter = "avx2_vector_add"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// An unknown rule: the mapping records an id the target's rule file does not
// declare.
// expected-error @below {{no_matching_rule: mapped op 'micro.vector': unknown rule 'avx2.no_such_rule'}}
module {
  micro.kernel @k {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.no_such_rule", executor = "worker.0", emitter = "avx2_vector_add"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// An unknown executor: the rule resolves, but the machine has no such node.
// expected-error @below {{no_legal_executor: mapped op 'micro.vector': unknown executor 'worker.9'}}
module {
  micro.kernel @k {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.vector_add", executor = "worker.9", emitter = "avx2_vector_add"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// An unknown memory: a rule-bound memory id the machine does not declare.
// expected-error @below {{no_memory_route: mapped op 'micro.vector': unknown memory 'no.such.memory'}}
module {
  micro.kernel @k {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.vector_add", executor = "worker.0", memories = {sram = "no.such.memory"}, emitter = "avx2_vector_add"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// An unknown layout: a layout id the target's layout file does not declare.
// expected-error @below {{no_legal_layout: mapped op 'micro.vector': unknown layout 'no.such.layout'}}
module {
  micro.kernel @k {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.vector_add", executor = "worker.0", layouts = {avx2.blocked_2d = "no.such.layout"}, emitter = "avx2_vector_add"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// An unknown emitter: the target-bundle check, which is why every other field
// is left valid here.
// expected-error @below {{target_bundle_invalid: mapped op 'micro.vector': unknown emitter 'avx2_no_such_emitter'}}
module {
  micro.kernel @k {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.vector_add", executor = "worker.0", emitter = "avx2_no_such_emitter"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// A route naming a memory the machine does not declare.
// expected-error @below {{no_memory_route: route names unknown memory 'no.such.memory'}}
module {
  micro.kernel @k attributes {micro.routes = [{kind = "direct", route = ["no.such.memory"], value = 0 : i64}]} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.vector_add", executor = "worker.0", emitter = "avx2_vector_add"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// A route whose consecutive memories are not joined by a link: both nodes
// resolve, but the machine cannot carry data dram.0 -> acc.0 directly.
// expected-error @below {{no_memory_route: route hop 'dram.0' -> 'acc.0' has no link}}
module {
  micro.kernel @k attributes {micro.routes = [{kind = "direct", route = ["dram.0", "acc.0"], value = 0 : i64}]} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.vector_add", executor = "worker.0", emitter = "avx2_vector_add"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// A module with no `micro.kernel` has nothing to verify; that is a failure, not
// a vacuous success (the same rule `micro-map` follows).
// expected-error @below {{micro-verify-mapping: the module has no micro.kernel}}
module {
  func.func @not_a_kernel() {
    return
  }
}
