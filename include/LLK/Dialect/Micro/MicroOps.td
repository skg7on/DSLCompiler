#ifndef MICRO_OPS_TD
#define MICRO_OPS_TD

include "MicroDialect.td"
include "MicroTypes.td"

//===----------------------------------------------------------------------===//
// Base class for all Micro dialect ops
//===----------------------------------------------------------------------===//

class Micro_Op<string mnemonic, list<Trait> traits = []>
    : Op<Micro_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// micro.kernel
//===----------------------------------------------------------------------===//

def Micro_KernelOp
    : Micro_Op<"kernel", [IsolatedFromAbove,
                          SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Top-level container for a concrete execution schedule";
  let description = [{
        Encapsulates a scheduled kernel with optional workload and target
        metadata. The body contains the execution schedule as micro ops.

        A kernel may declare an explicit argument and result contract:

        ```mlir
        micro.kernel @add(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
          ...
          micro.yield %r : !micro.tile<8x8xf32, memory = #micro.memory<sram>>
        }
        ```

        The signature names exactly what the kernel reads and produces, so a
        caller and a lowerer never have to guess which values are external: the
        entry block arguments are the inputs and the values `micro.yield`
        carries are the results. A yielded `!micro.tile` stands for a declared
        tensor result when their logical shape and element type agree -- memory
        space and layout are scheduling facts, not part of the caller's ABI.

        A kernel written without a signature (the legacy argumentless form)
        remains parseable for analysis and round-tripping, but carries no
        contract and cannot be lowered for execution. An inferred signature is
        not an interface, so migrate such a kernel to the explicit form before
        executing it.
    }];

  let arguments = (ins SymbolNameAttr:$sym_name,
      OptionalAttr<TypeAttr>:$function_type, OptionalAttr<StrAttr>:$workload,
      OptionalAttr<StrAttr>:$target, OptionalAttr<StrAttr>:$candidate,
      OptionalAttr<I64Attr>:$schedule_id);

  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// The kernel's declared signature, or a null `FunctionType` when the
    /// kernel carries only the legacy argumentless form. A null result means
    /// "no contract": the kernel is analysis-only and must not be executed.
    ::mlir::FunctionType getKernelFunctionType() {
      if (auto attr = getFunctionTypeAttr())
        return ::llvm::cast<::mlir::FunctionType>(attr.getValue());
      return {};
    }

    /// True when the kernel declares an explicit argument/result contract.
    bool hasExplicitSignature() { return static_cast<bool>(getKernelFunctionType()); }
  }];
}

//===----------------------------------------------------------------------===//
// micro.yield
//===----------------------------------------------------------------------===//

def Micro_YieldOp : Micro_Op<"yield", [Terminator]> {
  let summary = "Terminator for micro.kernel, micro.for, micro.spatial_for, "
                "and micro.pipeline";
  let description = [{
        Terminates a micro dialect region. A `micro.kernel` yields the values it
        produces, which is what lets a lowered kernel return something a caller
        can observe -- the kernel-ABI half of the bridge to the backend
        pipeline. A loop terminator carries none: loop-carried values are a
        later slice.
    }];

  let arguments = (ins Variadic<AnyType>:$results);

  let assemblyFormat = "($results^ `:` type($results))? attr-dict";

  // The implicit-terminator trait on `micro.kernel` and the loop ops builds a
  // terminator with no operands, so that overload has to exist alongside the
  // variadic one.
  let builders = [
    OpBuilder<(ins), [{ /* a bare terminator carries no results */ }]>
  ];
}

//===----------------------------------------------------------------------===//
// micro.tile_view — logical zero-cost view
//===----------------------------------------------------------------------===//

def Micro_TileViewOp : Micro_Op<"tile_view"> {
  let summary = "Logical zero-cost tile view of a tensor or larger tile";
  let description = [{
        Creates a logical tile view over a windowed slice of a source tensor
        or larger tile. The optional offsets index the source; when absent the
        view covers the whole source. A view is zero-cost: it does not allocate
        storage or produce a copy token.
    }];

  let arguments = (ins AnyType:$source, Variadic<Index>:$offsets,
      DenseI64ArrayAttr:$shape, OptionalAttr<Micro_LayoutAttr>:$layout);

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
        $source (`[` $offsets^ `]`)? attr-dict `:` type($source) `->` type($result)
    }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// micro.tile_alloc — materialized memory tile
//===----------------------------------------------------------------------===//

def Micro_TileAllocOp : Micro_Op<"tile_alloc"> {
  let summary = "Allocate a materialized tile in a memory space";
  let description = [{
        Allocates a tile with physical storage in a memory space. The memory
        space and owner are carried by the result tile type.
    }];

  let arguments = (ins);

  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// micro.tile_partition — partition into instruction fragment
//===----------------------------------------------------------------------===//

def Micro_TilePartitionOp : Micro_Op<"tile_partition"> {
  let summary = "Partition a tile into smaller logical or execution fragments";
  let description = [{
        Partitions a tile into smaller fragments. Logical unless followed by
        materialization or compute consumption. The fragment shape must divide
        the parent shape or carry explicit mask metadata.
    }];

  let arguments = (ins AnyType:$source, DenseI64ArrayAttr:$shape,
      OptionalAttr<Micro_OwnerAttr>:$owner, OptionalAttr<BoolAttr>:$tail);

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
        $source attr-dict `:` type($source) `->` type($result)
    }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// micro.tile_async_copy — async movement returning tile + token
//===----------------------------------------------------------------------===//

def Micro_TileAsyncCopyOp : Micro_Op<"tile_async_copy"> {
  let summary = "Asynchronous tile movement between memory spaces";
  let description = [{
        Moves a tile to a destination memory space asynchronously, returning
        a materialized tile and an async token for synchronization.

        A machine may hold several concrete memories of one abstract kind
        (`sram.0`, `sram.1`). Source/destination kind equality is not node
        identity, so a same-kind move records the two concrete node ids it
        crosses in `micro.src_node`/`micro.dst_node`; the structural verifier
        checks those ids are distinct strings, and the machine-aware verifier
        resolves them against the model's links, engines, and transaction and
        alignment facts. A kind-differing move may record them too.
    }];

  let arguments = (ins AnyType:$source, Micro_MemorySpaceAttr:$dst_memory,
      OptionalAttr<Micro_OwnerAttr>:$owner);

  let results = (outs AnyType:$result, AnyType:$token);

  let assemblyFormat = [{
        $source attr-dict
        `:` type($source) `->` type($result) `,` type($token)
    }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// micro.tile_store — write back to external memory
//===----------------------------------------------------------------------===//

def Micro_TileStoreOp : Micro_Op<"tile_store"> {
  let summary = "Store a tile to a destination memory space";
  let description = [{
        Stores a materialized tile to an external or previously allocated
        destination memory space.
    }];

  let arguments = (ins AnyType:$source, Micro_MemorySpaceAttr:$dst_memory);

  let assemblyFormat = [{
        $source attr-dict
        `:` type($source)
    }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// micro.mma — tile-typed matrix engine fragment
//===----------------------------------------------------------------------===//

def Micro_MmaOp : Micro_Op<"mma"> {
  let summary = "Matrix engine compute one level above vendor ISA";
  let description = [{
        Computes a tile-typed matrix fragment: acc += lhs x rhs. The shape
        attribute is interpreted as M, N, K.
    }];

  let arguments = (ins AnyType:$lhs, AnyType:$rhs, AnyType:$acc,
      DenseI64ArrayAttr:$shape, Micro_DTypeAttr:$input,
      Micro_DTypeAttr:$accumulator, OptionalAttr<StrAttr>:$engine);

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
        $lhs `,` $rhs `,` $acc
        attr-dict
        `:` type($lhs) `,` type($rhs) `,` type($acc) `->` type($result)
    }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// micro.vector — tile-typed elementwise fragment
//===----------------------------------------------------------------------===//

def Micro_VectorOp : Micro_Op<"vector"> {
  let summary = "Tile-typed vector elementwise operation";
  let description = [{
        Applies a vector elementwise operation to one or more tile operands.
    }];

  let arguments = (ins StrAttr:$op, Variadic<AnyType>:$inputs,
      OptionalAttr<StrAttr>:$math_mode);

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
        $op $inputs
        attr-dict
        `:` type($inputs) `->` type($result)
    }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// micro.reduce — tile-typed reduction fragment
//===----------------------------------------------------------------------===//

def Micro_ReduceOp : Micro_Op<"reduce"> {
  let summary = "Tile-typed reduction along one axis";
  let description = [{
        Reduces a tile along a single axis using sum, max, min, or prod.
    }];

  let arguments = (ins StrAttr:$op, AnyType:$input, I64Attr:$axis);

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
        $op $input
        attr-dict
        `:` type($input) `->` type($result)
    }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// micro.gather — combine several producers under explicit semantics
//===----------------------------------------------------------------------===//

def Micro_GatherOp : Micro_Op<"gather"> {
  let summary = "Combine several producer values under declared semantics";
  let description = [{
        Gathers two or more producer values into one result under an explicit
        combination semantics, named by `kind`:

          sum     elementwise sum; every input and the result share one shape
                  and element type
          max     elementwise maximum; same shape/type requirement as sum
          concat  concatenation along `axis`; the result's extent along that
                  axis is exactly the sum of the inputs' extents, and every
                  other extent matches

        The semantics are never inferred from how many producers feed the
        value: a gather states what the combination *means*, and a connection
        that does not declare one is not materialized. `axis` is required for
        `concat` and rejected for `sum`/`max`.
    }];

  let arguments = (ins Variadic<AnyType>:$inputs, StrAttr:$kind,
      OptionalAttr<I64Attr>:$axis);

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
        $inputs `kind` `=` $kind (`axis` `=` $axis^)? attr-dict
        `:` type($inputs) `->` type($result)
    }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// micro.barrier — executor-group barrier over dependency tokens
//===----------------------------------------------------------------------===//

def Micro_BarrierOp : Micro_Op<"barrier"> {
  let summary = "Executor-group barrier over dependency tokens";
  let description = [{
        Blocks its scope until every dependency token has completed. The scope
        is generic (`executor_group`): a barrier names the executor group whose
        work must agree, not a concrete target resource. It is emitted only
        where the selected plan's synchronization requires one; two independent
        movements that share no dependency carry no barrier between them.
    }];

  let arguments = (ins Variadic<Micro_AsyncToken>:$tokens, StrAttr:$scope);

  let assemblyFormat = "$tokens `scope` `=` $scope attr-dict";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// micro.for — temporal loop
//===----------------------------------------------------------------------===//

def Micro_ForOp : Micro_Op<"for", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Temporal loop";
  let description = [{
        A temporal loop over an index induction variable with lower bound,
        upper bound, and step, modeled after `scf.for`.

        A loop may carry values across iterations. They are declared with
        `iter_args`, and each iteration's `micro.yield` carries the value the
        next iteration receives:

        ```mlir
        %acc = micro.for %k = %c0 to %K step %BK iter_args(%carried = %init) -> (!micro.tile<8x32xf32>) {
          %next = micro.mma %a, %b, %carried
          micro.yield %next : !micro.tile<8x32xf32>
        }
        ```

        The loop's results are what the final iteration yielded. A loop with no
        `iter_args` carries nothing: it has no results and its terminator is a
        bare `micro.yield`. Deciding *which* values a loop carries is a
        semantics question, not a scheduling one -- a lowerer that invented
        them would be guessing what the program means.
    }];

  let arguments = (ins Index:$lower_bound, Index:$upper_bound, Index:$step,
      Variadic<AnyType>:$initArgs);

  let results = (outs Variadic<AnyType>:$results);

  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
        /// The induction variable is the first block argument; the carried
        /// values follow it, in `iter_args` order.
        BlockArgument getInductionVar() {
          return getBody().front().getArgument(0);
        }
        ::llvm::MutableArrayRef<::mlir::BlockArgument> getRegionIterArgs() {
          return getBody().front().getArguments().drop_front();
        }
    }];
}

//===----------------------------------------------------------------------===//
// micro.spatial_for — spatial loop over machine resources
//===----------------------------------------------------------------------===//

def Micro_SpatialForOp
    : Micro_Op<"spatial_for", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Spatial loop over machine resources";
  let description = [{
        A spatial loop mapping iterations over machine resources via a
        #micro.map target. Modeled after micro.for with an additional map, and
        the same loop-carried value contract:

        ```mlir
        %acc = micro.spatial_for %i = %c0 to %M step %BM map = #micro.map<worker>
            iter_args(%carried = %init) -> (!micro.tile<8x32xf32>) {
          micro.yield %carried : !micro.tile<8x32xf32>
        }
        ```
    }];

  let arguments = (ins Index:$lower_bound, Index:$upper_bound, Index:$step,
      Micro_MappingTargetAttr:$map, Variadic<AnyType>:$initArgs);

  let results = (outs Variadic<AnyType>:$results);

  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
        BlockArgument getInductionVar() {
          return getBody().front().getArgument(0);
        }
        ::llvm::MutableArrayRef<::mlir::BlockArgument> getRegionIterArgs() {
          return getBody().front().getArguments().drop_front();
        }
    }];
}

//===----------------------------------------------------------------------===//
// micro.pipeline — pipeline overlap between movement and compute
//===----------------------------------------------------------------------===//

def Micro_PipelineOp
    : Micro_Op<"pipeline", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Pipeline overlap between movement and compute";
  let description = [{
        Marks pipeline overlap between data movement and compute. The stages
        attribute controls the number of pipeline stages.

        A pipeline may carry values out of its body, on the same terms as a
        loop: `micro.yield` names them and the pipeline's results are those
        values. It needs this because the compute it overlaps with often
        accumulates into something the staged movement produced, and the value
        has to reach whatever comes after the pipeline:

        ```mlir
        %acc = micro.pipeline stages = 2 -> (!micro.tile<8x32xf32, memory = #micro.memory<acc>>) {
          %next = micro.mma %a, %b, %carried
          micro.yield %next : !micro.tile<8x32xf32, memory = #micro.memory<acc>>
        }
        ```

        With no results the pipeline carries nothing and prints without the
        clause, which is the form every existing fixture uses.
    }];

  let arguments = (ins I64Attr:$stages);

  let results = (outs Variadic<AnyType>:$results);

  let regions = (region SizedRegion<1>:$body);

  let hasCustomAssemblyFormat = 1;

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// micro.alloc — allocate logical storage in a memory space
//===----------------------------------------------------------------------===//

def Micro_AllocOp : Micro_Op<"alloc"> {
  let summary = "Allocate logical tile storage in a memory space";
  let description = [{
        Allocates tensor storage in a memory space. DRAM operands represent
        external buffers, so local allocations must not target dram.
    }];

  let arguments = (ins Micro_MemorySpaceAttr:$memory);

  let results = (outs AnyType:$result);

  let assemblyFormat = "attr-dict `:` type($result)";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// micro.async_copy — asynchronous movement between memory spaces
//===----------------------------------------------------------------------===//

def Micro_AsyncCopyOp : Micro_Op<"async_copy"> {
  let summary = "Asynchronous movement between memory spaces";
  let description = [{
        Moves a tensor between two memory spaces asynchronously, returning the
        materialized tensor and an async token for synchronization.

        As with `micro.tile_async_copy`, a same-kind move between two distinct
        concrete memory nodes records `micro.src_node`/`micro.dst_node` so the
        movement is real work rather than collapsed on kind equality; a
        kind-differing move may record them too.
    }];

  let arguments = (ins AnyType:$source, Micro_MemorySpaceAttr:$src_memory,
      Micro_MemorySpaceAttr:$dst_memory);

  let results = (outs AnyType:$result, AnyType:$token);

  let assemblyFormat = [{
        $source attr-dict `:` type($source) `->` type($result) `,` type($token)
    }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// micro.wait — wait on async tokens
//===----------------------------------------------------------------------===//

def Micro_WaitOp : Micro_Op<"wait"> {
  let summary = "Wait on async movement or compute tokens";
  let description = [{
        Blocks until all operand async tokens have completed.
    }];

  let arguments = (ins Variadic<Micro_AsyncToken>:$tokens);

  let assemblyFormat = "$tokens attr-dict";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// micro.transform — re-represent a value under a different layout
//===----------------------------------------------------------------------===//

def Micro_TransformOp : Micro_Op<"transform"> {
  let summary = "Re-represent a shaped value under a different layout";
  let description = [{
        Converts a shaped value from one layout to another in place: the value
        read through `src_map` is written through `dst_map`. Each map is the
        logical-to-physical affine map of the layout on that side, so the
        operation is target-neutral -- it names layouts by their index relation,
        never by a target-owned id (design §13.4). A map is absent when the
        corresponding layout declares no map clause, which leaves that side's
        index relation to the target.

        The operation is deliberately not `Pure`: it performs real work with a
        cost, so it is neither eliminated nor commoned away. A `LayoutTransform`
        connection in a bound plan becomes one of these, and a
        `TransferAndTransform` connection emits its copies followed by one.
    }];

  let arguments = (ins AnyType:$source, OptionalAttr<AffineMapAttr>:$src_map,
      OptionalAttr<AffineMapAttr>:$dst_map);

  let results = (outs AnyType:$result);

  let assemblyFormat = [{
        $source attr-dict `:` type($source) `->` type($result)
    }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// micro.store — store a value to a destination memory space
//===----------------------------------------------------------------------===//

def Micro_StoreOp : Micro_Op<"store"> {
  let summary = "Store a value to a destination memory space";
  let description = [{
        Stores a shaped value to a destination memory space. The source and
        destination memory spaces must differ.
    }];

  let arguments = (ins AnyType:$source, Micro_MemorySpaceAttr:$src_memory,
      Micro_MemorySpaceAttr:$dst_memory);

  let assemblyFormat = [{
        $source attr-dict `:` type($source)
    }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// micro.search_space — container for legal schedule choices
//===----------------------------------------------------------------------===//

def Micro_SearchSpaceOp
    : Micro_Op<"search_space", [IsolatedFromAbove,
                                SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Top-level container for legal schedule choices";
  let description = [{
        Holds the search-space metadata an auto-scheduler consumes: tunable
        parameters, legality constraints, ranking objectives, and candidate
        bindings. Concrete execution ops are not allowed in the body.

        Candidate bindings are named assignments of the declared parameters,
        and are symbols so that later stages can refer to a specific candidate.
    }];

  let arguments = (ins SymbolNameAttr:$sym_name, StrAttr:$workload);

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
        $sym_name attr-dict-with-keyword $body
    }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// micro.param — one tunable search parameter
//===----------------------------------------------------------------------===//

def Micro_ParamOp : Micro_Op<"param"> {
  let summary = "Declare one tunable search parameter";
  let description = [{
        Declares a named parameter and its legal values. The kind names the
        domain so that verifiers do not have to infer semantics from the
        parameter name:

          integer        positive integers, strictly increasing
          layout         a Micro layout kind
          memory_path    colon-separated memory spaces, e.g. dram:sram:acc
          owner_mapping  slash-separated owners, e.g. worker/vector_engine
          fragment_shape MxNxK instruction fragment, e.g. 16x16x32
          tail_policy    mask only in the MVP

        Parameters are lexically scoped to their micro.search_space and are
        referred to by name, not bound to SSA values.

        `role` is optional and names the *thing* a parameter governs, for the
        axes where one space may bind several at once. The layout axis is the
        one that needs it: a rule requires a layout on a named port
        (`operand0`, `lhs`), so a space that binds more than one layout kind
        must say which port each one is for. A parameter with no role governs
        the axis as a whole -- the behaviour before roles existed, and still
        the right spelling for a space that binds one layout.
    }];

  let arguments = (ins StrAttr:$name, StrAttr:$kind, ArrayAttr:$choices,
                       OptionalAttr<StrAttr>:$role);

  let assemblyFormat = "$name attr-dict";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// micro.constraint — one typed legality record
//===----------------------------------------------------------------------===//

def Micro_ConstraintOp : Micro_Op<"constraint"> {
  let summary = "Declare one typed legality constraint";
  let description = [{
        Records that a set of search parameters must satisfy a named legality
        rule. This is a typed record, not a free-form expression language:
        machine-aware evaluation of capacity, MMA compatibility, and the like
        belongs to the MachineModel and tuning stages.

        The kind is one of sram_capacity, acc_capacity, mma_compatible,
        mapping_extent, tail_supported, vector_width_supported,
        tile_hierarchy_compatible, layout_supported, owner_supported,
        fragment_compatible, or pipeline_live_tiles.
    }];

  let arguments = (ins StrAttr:$kind, ArrayAttr:$params);

  let assemblyFormat = "$kind attr-dict";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// micro.objective — ranking objective for candidate search
//===----------------------------------------------------------------------===//

def Micro_ObjectiveOp : Micro_Op<"objective"> {
  let summary = "Ranking objective for candidate search";
  let description = [{
        Names the metric candidates are ranked by, the direction to optimize
        it in, and optional secondary metrics used as tie-breakers in the
        order given.
    }];

  let arguments = (ins StrAttr:$direction, StrAttr:$metric,
      OptionalAttr<ArrayAttr>:$secondary);

  let assemblyFormat = "attr-dict";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// micro.candidate — one parameter binding
//===----------------------------------------------------------------------===//

def Micro_CandidateOp : Micro_Op<"candidate"> {
  let summary = "Bind every search parameter to one selected value";
  let description = [{
        A candidate is a named assignment of every parameter declared in the
        enclosing micro.search_space. Binding a candidate to a concrete
        micro.kernel happens in the tuning stage, not here.
    }];

  let arguments = (ins SymbolNameAttr:$sym_name, DictionaryAttr:$bindings);

  let assemblyFormat = "$sym_name attr-dict";

  let hasVerifier = 1;
}

#endif
