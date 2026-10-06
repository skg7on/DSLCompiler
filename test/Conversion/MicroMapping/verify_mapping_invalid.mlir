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
// asserts the code *and* the specific detail (which operation, which id).
//
// A mapped kernel carries `micro.plan`; completeness is checked per kernel, so
// a kernel without it is rejected before any id is resolved. The per-operation
// walk checks, in order, rule -> match-op -> executor -> memories -> layouts ->
// bundle -> emitter, and generic metadata containers are type-checked before
// they are read, so malformed metadata is a diagnostic rather than an aborting
// cast. A chunk that leaves an earlier field valid reaches the later one, so
// every mapping here supplies what its rule requires.
//
// `--verify-diagnostics` binds a pass diagnostic emitted on the module to the
// `module {` line, so `expected-error @below` immediately above that line is
// what each chunk uses.

// Phase 2 accepts a fully resolved mapping: this chunk has no expected
// diagnostic, and the run only succeeds if the pass leaves it alone. A mapped
// kernel carries `micro.plan`; the rule `avx2.vector_add` requires layout
// `avx2.blocked_2d` on `operand0`, so the mapping binds it.
module {
  micro.kernel @k attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.vector_add", executor = "worker.0", layouts = {avx2.blocked_2d = "avx2.blocked_2d"}, layout_parameters = {avx2.blocked_2d = {M = 8 : i64, N = 8 : i64, VW = 8 : i64}}, bundle = "avx2.vector.add.f32", emitter = "avx2_vector_add"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// An unmapped kernel: no `micro.plan` means nothing was ever bound, so there
// is no selection to resolve. Reporting success here would be the whole defect
// the pass exists to catch.
// expected-error @below {{no_matching_rule: kernel 'k' is not mapped: it has no micro.plan}}
module {
  micro.kernel @k {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// A mapped kernel that left a workload operation unannotated is only partly
// mapped: completeness is defined at kernel scope, so every workload op must
// carry its selection.
// expected-error @below {{no_matching_rule: kernel 'k': op 'micro.vector' carries no micro.mapping}}
module {
  micro.kernel @k attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// A compute op that deleted its `micro.mapping` and stamped an arbitrary
// `micro.value` movement attribute onto itself used to evade the completeness
// check: the walk exempted any op carrying the stamp. The exemption now
// requires a verified materialized connection, so the compute op is still
// coverage.
// expected-error @below {{no_matching_rule: kernel 'k': op 'micro.vector' carries no micro.mapping}}
module {
  micro.kernel @k attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.value = 0 : i64} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// A movement stamped with `micro.value` but no connection provenance is not a
// verified materialized connection, so the exemption is refused.
// expected-error @below {{invalid_mapping_metadata: kernel 'k': op 'micro.async_copy': 'micro.connection' is missing}}
module {
  micro.kernel @k attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t, %tok = micro.async_copy %0 {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>, micro.value = 0 : i64, micro.dst_node = "sram.0"} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
  }
}

// -----

// A movement naming a connection the kernel's `micro.routes` does not declare
// cannot be resolved, so it is not the materialized connection it claims.
// expected-error @below {{invalid_mapping_metadata: kernel 'k': op 'micro.async_copy' names connection 99, which micro.routes does not declare}}
module {
  micro.kernel @k attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}, micro.routes = [{id = 7 : i64, kind = "transfer", route = ["dram.0", "sram.0"], engines = ["dma.0"], value = 0 : i64}]} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t, %tok = micro.async_copy %0 {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>, micro.value = 0 : i64, micro.dst_node = "sram.0", micro.connection = 99 : i64, micro.hop = 1 : i64} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
  }
}

// -----

// A resolved hop carried by a transfer engine the machine does not declare is
// not executable: the movement's route must name an engine its link offers.
// expected-error @below {{invalid_mapping_metadata: kernel 'k': op 'micro.async_copy' route names unsupported transfer engine 'no_such_engine'}}
module {
  micro.kernel @k attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}, micro.routes = [{id = 7 : i64, kind = "transfer", route = ["dram.0", "sram.0"], engines = ["no_such_engine"], value = 0 : i64}]} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t, %tok = micro.async_copy %0 {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>, micro.value = 0 : i64, micro.dst_node = "sram.0", micro.connection = 7 : i64, micro.hop = 1 : i64} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
  }
}

// -----

// A movement whose destination stamp is not the memory the resolved hop lands
// in is not the connection it names.
// expected-error @below {{invalid_mapping_metadata: kernel 'k': op 'micro.async_copy' records micro.dst_node 'acc.0', but connection 7 hop 1 lands in 'sram.0'}}
module {
  micro.kernel @k attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}, micro.routes = [{id = 7 : i64, kind = "transfer", route = ["dram.0", "sram.0"], engines = ["dma.0"], value = 0 : i64}]} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t, %tok = micro.async_copy %0 {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>, micro.value = 0 : i64, micro.dst_node = "acc.0", micro.connection = 7 : i64, micro.hop = 1 : i64} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
  }
}

// -----

// A movement whose declared source memory is not the resolved hop's source is
// not the connection it names.
// expected-error @below {{invalid_mapping_metadata: kernel 'k': op 'micro.async_copy' declares memories that do not match connection 7 hop 1 ('dram.0' -> 'sram.0')}}
module {
  micro.kernel @k attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}, micro.routes = [{id = 7 : i64, kind = "transfer", route = ["dram.0", "sram.0"], engines = ["dma.0"], value = 0 : i64}]} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t, %tok = micro.async_copy %0 {src_memory = #micro.memory<l2>, dst_memory = #micro.memory<sram>, micro.value = 0 : i64, micro.dst_node = "sram.0", micro.connection = 7 : i64, micro.hop = 1 : i64} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
  }
}

// -----

// An unknown rule: the mapping records an id the target's rule file does not
// declare.
// expected-error @below {{no_matching_rule: mapped op 'micro.vector': unknown rule 'avx2.no_such_rule'}}
module {
  micro.kernel @k attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.no_such_rule", executor = "worker.0", layouts = {avx2.blocked_2d = "avx2.blocked_2d"}, layout_parameters = {avx2.blocked_2d = {M = 8 : i64, N = 8 : i64, VW = 8 : i64}}, bundle = "avx2.vector.add.f32", emitter = "avx2_vector_add"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// A rule that exists but implements a different operation: an MMA rule on a
// vector op resolved its id, yet cannot implement this op.
// expected-error @below {{no_matching_rule: mapped op 'micro.vector': rule 'avx2.mma_bf16' implements 'micro.mma', not 'micro.vector'}}
module {
  micro.kernel @k attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.mma_bf16", executor = "worker.0", layouts = {avx2.row_major = "avx2.row_major"}, bundle = "avx2.mma.bf16", emitter = "avx2_mma"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// An unknown executor: the rule resolves, but the machine has no such node.
// expected-error @below {{no_legal_executor: mapped op 'micro.vector': unknown executor 'worker.9'}}
module {
  micro.kernel @k attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.vector_add", executor = "worker.9", layouts = {avx2.blocked_2d = "avx2.blocked_2d"}, layout_parameters = {avx2.blocked_2d = {M = 8 : i64, N = 8 : i64, VW = 8 : i64}}, bundle = "avx2.vector.add.f32", emitter = "avx2_vector_add"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// An unknown memory: a rule-bound memory id the machine does not declare.
// expected-error @below {{no_memory_route: mapped op 'micro.vector': unknown memory 'no.such.memory'}}
module {
  micro.kernel @k attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.vector_add", executor = "worker.0", memories = {sram = "no.such.memory"}, layouts = {avx2.blocked_2d = "avx2.blocked_2d"}, layout_parameters = {avx2.blocked_2d = {M = 8 : i64, N = 8 : i64, VW = 8 : i64}}, bundle = "avx2.vector.add.f32", emitter = "avx2_vector_add"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// An unknown layout: a layout id the target's layout file does not declare.
// expected-error @below {{no_legal_layout: mapped op 'micro.vector': unknown layout 'no.such.layout'}}
module {
  micro.kernel @k attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.vector_add", executor = "worker.0", layouts = {avx2.blocked_2d = "no.such.layout"}, bundle = "avx2.vector.add.f32", emitter = "avx2_vector_add"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// A required layout role left unbound: `avx2.vector_add` requires
// `avx2.blocked_2d` on `operand0`, but the mapping records no layouts at all.
// expected-error @below {{no_legal_layout: mapped op 'micro.vector': rule 'avx2.vector_add' requires layout 'avx2.blocked_2d' on port 'operand0', which the mapping does not bind}}
module {
  micro.kernel @k attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.vector_add", executor = "worker.0", bundle = "avx2.vector.add.f32", emitter = "avx2_vector_add"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// A bundle other than the one the selected rule declares: the rule/emitter pair
// is right, but the recorded bundle is not what the rule selected.
// expected-error @below {{target_bundle_invalid: mapped op 'micro.vector': rule 'avx2.vector_add' selects bundle 'avx2.vector.add.f32', but the mapping records 'garbage'}}
module {
  micro.kernel @k attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.vector_add", executor = "worker.0", layouts = {avx2.blocked_2d = "avx2.blocked_2d"}, layout_parameters = {avx2.blocked_2d = {M = 8 : i64, N = 8 : i64, VW = 8 : i64}}, bundle = "garbage", emitter = "avx2_vector_add"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// An emitter that does not match the selected rule: `avx2_mma` is a declared
// emitter, but the vector-add rule selects `avx2_vector_add`.
// expected-error @below {{target_bundle_invalid: mapped op 'micro.vector': rule 'avx2.vector_add' selects emitter 'avx2_vector_add', but the mapping records 'avx2_mma'}}
module {
  micro.kernel @k attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.vector_add", executor = "worker.0", layouts = {avx2.blocked_2d = "avx2.blocked_2d"}, layout_parameters = {avx2.blocked_2d = {M = 8 : i64, N = 8 : i64, VW = 8 : i64}}, bundle = "avx2.vector.add.f32", emitter = "avx2_mma"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// An emitter key the target does not declare at all.
// expected-error @below {{target_bundle_invalid: mapped op 'micro.vector': unknown emitter 'avx2_no_such_emitter'}}
module {
  micro.kernel @k attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.vector_add", executor = "worker.0", layouts = {avx2.blocked_2d = "avx2.blocked_2d"}, layout_parameters = {avx2.blocked_2d = {M = 8 : i64, N = 8 : i64, VW = 8 : i64}}, bundle = "avx2.vector.add.f32", emitter = "avx2_no_such_emitter"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// Malformed generic metadata: a `memories` entry that is not a string. Before
// this check the verifier cast the value unchecked and aborted the process.
// expected-error @below {{invalid_mapping_metadata: mapped op 'micro.vector': 'memories' entry 'operand0' is not a string}}
module {
  micro.kernel @k attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.vector_add", executor = "worker.0", memories = {operand0 = 42 : i64}, layouts = {avx2.blocked_2d = "avx2.blocked_2d"}, layout_parameters = {avx2.blocked_2d = {M = 8 : i64, N = 8 : i64, VW = 8 : i64}}, bundle = "avx2.vector.add.f32", emitter = "avx2_vector_add"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// A `micro.mapping` that is not a dictionary at all: a checked cast reports it
// instead of silently skipping the operation.
// expected-error @below {{invalid_mapping_metadata: mapped op 'micro.vector': micro.mapping is not a dictionary}}
module {
  micro.kernel @k attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = 42 : i64} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// A `micro.routes` entry that is not a dictionary. Before this check the
// verifier cast it unchecked and aborted the process.
// expected-error @below {{invalid_mapping_metadata: micro.routes on 'micro.kernel': a route entry is not a dictionary}}
module {
  micro.kernel @k attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}, micro.routes = [42 : i64]} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.vector_add", executor = "worker.0", layouts = {avx2.blocked_2d = "avx2.blocked_2d"}, layout_parameters = {avx2.blocked_2d = {M = 8 : i64, N = 8 : i64, VW = 8 : i64}}, bundle = "avx2.vector.add.f32", emitter = "avx2_vector_add"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// A route naming a memory the machine does not declare.
// expected-error @below {{no_memory_route: route names unknown memory 'no.such.memory'}}
module {
  micro.kernel @k attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}, micro.routes = [{kind = "direct", route = ["no.such.memory"], value = 0 : i64}]} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.vector_add", executor = "worker.0", layouts = {avx2.blocked_2d = "avx2.blocked_2d"}, layout_parameters = {avx2.blocked_2d = {M = 8 : i64, N = 8 : i64, VW = 8 : i64}}, bundle = "avx2.vector.add.f32", emitter = "avx2_vector_add"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// A route whose consecutive memories are not joined by a link: both nodes
// resolve, but the machine cannot carry data dram.0 -> acc.0 directly.
// expected-error @below {{no_memory_route: route hop 'dram.0' -> 'acc.0' has no link}}
module {
  micro.kernel @k attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}, micro.routes = [{kind = "direct", route = ["dram.0", "acc.0"], value = 0 : i64}]} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.vector_add", executor = "worker.0", layouts = {avx2.blocked_2d = "avx2.blocked_2d"}, layout_parameters = {avx2.blocked_2d = {M = 8 : i64, N = 8 : i64, VW = 8 : i64}}, bundle = "avx2.vector.add.f32", emitter = "avx2_vector_add"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}

// -----

// A route naming a transfer engine the machine does not declare. The route
// itself is legal, so only the engine check can reject it.
// expected-error @below {{no_memory_route: route names unknown transfer engine 'nonexistent'}}
module {
  micro.kernel @k attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}, micro.routes = [{kind = "transfer", route = ["dram.0", "sram.0"], engines = ["nonexistent"], value = 0 : i64}]} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.vector_add", executor = "worker.0", layouts = {avx2.blocked_2d = "avx2.blocked_2d"}, layout_parameters = {avx2.blocked_2d = {M = 8 : i64, N = 8 : i64, VW = 8 : i64}}, bundle = "avx2.vector.add.f32", emitter = "avx2_vector_add"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
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

// -----

// Two kernels and no kernel selector: verifying "the kernel" would silently
// choose the first and ignore the rest, so the module is rejected as ambiguous.
// expected-error @below {{micro-verify-mapping: the module has 2 micro.kernels (@a, @b); mapping one kernel per module is required, since there is no kernel selector}}
module {
  micro.kernel @a attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.vector_add", executor = "worker.0", layouts = {avx2.blocked_2d = "avx2.blocked_2d"}, layout_parameters = {avx2.blocked_2d = {M = 8 : i64, N = 8 : i64, VW = 8 : i64}}, bundle = "avx2.vector.add.f32", emitter = "avx2_vector_add"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
  micro.kernel @b attributes {micro.plan = {id = 0 : i64, binding_hash = 0 : i64, truncated = false}} {
    %0 = tensor.empty() : tensor<8x8xf32>
    %t = micro.tile_view %0 {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %t, %t {micro.mapping = {rule = "avx2.vector_add", executor = "worker.0", layouts = {avx2.blocked_2d = "avx2.blocked_2d"}, layout_parameters = {avx2.blocked_2d = {M = 8 : i64, N = 8 : i64, VW = 8 : i64}}, bundle = "avx2.vector.add.f32", emitter = "avx2_vector_add"}} : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
  }
}
