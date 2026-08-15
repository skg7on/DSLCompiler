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

def Micro_KernelOp : Micro_Op<"kernel", [
    IsolatedFromAbove,
    SingleBlockImplicitTerminator<"YieldOp">
]> {
    let summary = "Top-level container for a concrete execution schedule";
    let description = [{
        Encapsulates a scheduled kernel with optional workload and target
        metadata. The body contains the execution schedule as micro ops.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        OptionalAttr<StrAttr>:$workload,
        OptionalAttr<StrAttr>:$target,
        OptionalAttr<StrAttr>:$candidate,
        OptionalAttr<I64Attr>:$schedule_id
    );

    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = [{
        $sym_name
        attr-dict-with-keyword
        $body
    }];

    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// micro.yield
//===----------------------------------------------------------------------===//

def Micro_YieldOp : Micro_Op<"yield", [
    Terminator
]> {
    let summary = "Terminator for micro.kernel, micro.for, micro.spatial_for, and micro.pipeline";
    let description = [{
        Terminates a micro dialect region. No operands in the MVP.
    }];

    let arguments = (ins);

    let assemblyFormat = "attr-dict";
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

    let arguments = (ins
        AnyType:$source,
        Variadic<Index>:$offsets,
        DenseI64ArrayAttr:$shape,
        OptionalAttr<Micro_LayoutAttr>:$layout
    );

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

    let arguments = (ins
        AnyType:$source,
        DenseI64ArrayAttr:$shape,
        OptionalAttr<Micro_OwnerAttr>:$owner,
        OptionalAttr<BoolAttr>:$tail
    );

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
    }];

    let arguments = (ins
        AnyType:$source,
        Micro_MemorySpaceAttr:$dst_memory,
        OptionalAttr<Micro_OwnerAttr>:$owner
    );

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

    let arguments = (ins
        AnyType:$source,
        Micro_MemorySpaceAttr:$dst_memory
    );

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

    let arguments = (ins
        AnyType:$lhs,
        AnyType:$rhs,
        AnyType:$acc,
        DenseI64ArrayAttr:$shape,
        Micro_DTypeAttr:$input,
        Micro_DTypeAttr:$accumulator,
        OptionalAttr<StrAttr>:$engine
    );

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

    let arguments = (ins
        StrAttr:$op,
        Variadic<AnyType>:$inputs,
        OptionalAttr<StrAttr>:$math_mode
    );

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

    let arguments = (ins
        StrAttr:$op,
        AnyType:$input,
        I64Attr:$axis
    );

    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $op $input
        attr-dict
        `:` type($input) `->` type($result)
    }];

    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// micro.for — temporal loop
//===----------------------------------------------------------------------===//

def Micro_ForOp : Micro_Op<"for", [
    SingleBlockImplicitTerminator<"YieldOp">
]> {
    let summary = "Temporal loop";
    let description = [{
        A temporal loop over an index induction variable with lower bound,
        upper bound, and step. Modeled after scf.for; no iter_args in the MVP.
    }];

    let arguments = (ins
        Index:$lower_bound,
        Index:$upper_bound,
        Index:$step
    );

    let regions = (region SizedRegion<1>:$body);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        BlockArgument getInductionVar() {
          return getBody().front().getArgument(0);
        }
    }];
}

//===----------------------------------------------------------------------===//
// micro.spatial_for — spatial loop over machine resources
//===----------------------------------------------------------------------===//

def Micro_SpatialForOp : Micro_Op<"spatial_for", [
    SingleBlockImplicitTerminator<"YieldOp">
]> {
    let summary = "Spatial loop over machine resources";
    let description = [{
        A spatial loop mapping iterations over machine resources via a
        #micro.map target. Modeled after micro.for with an additional map.
    }];

    let arguments = (ins
        Index:$lower_bound,
        Index:$upper_bound,
        Index:$step,
        Micro_MappingTargetAttr:$map
    );

    let regions = (region SizedRegion<1>:$body);

    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        BlockArgument getInductionVar() {
          return getBody().front().getArgument(0);
        }
    }];
}

//===----------------------------------------------------------------------===//
// micro.pipeline — pipeline overlap between movement and compute
//===----------------------------------------------------------------------===//

def Micro_PipelineOp : Micro_Op<"pipeline", [
    SingleBlockImplicitTerminator<"YieldOp">
]> {
    let summary = "Pipeline overlap between movement and compute";
    let description = [{
        Marks pipeline overlap between data movement and compute. The stages
        attribute controls the number of pipeline stages.
    }];

    let arguments = (ins I64Attr:$stages);

    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = "`stages` `=` $stages attr-dict-with-keyword $body";

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
    }];

    let arguments = (ins
        AnyType:$source,
        Micro_MemorySpaceAttr:$src_memory,
        Micro_MemorySpaceAttr:$dst_memory
    );

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
// micro.store — store a value to a destination memory space
//===----------------------------------------------------------------------===//

def Micro_StoreOp : Micro_Op<"store"> {
    let summary = "Store a value to a destination memory space";
    let description = [{
        Stores a shaped value to a destination memory space. The source and
        destination memory spaces must differ.
    }];

    let arguments = (ins
        AnyType:$source,
        Micro_MemorySpaceAttr:$src_memory,
        Micro_MemorySpaceAttr:$dst_memory
    );

    let assemblyFormat = [{
        $source attr-dict `:` type($source)
    }];

    let hasVerifier = 1;
}

#endif
