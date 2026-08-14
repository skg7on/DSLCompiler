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
        Creates a logical tile view over a source tensor or tile. A view is
        zero-cost: it does not allocate storage or produce a copy token.
    }];

    let arguments = (ins
        AnyType:$source,
        DenseI64ArrayAttr:$shape,
        OptionalAttr<Micro_LayoutAttr>:$layout
    );

    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $source attr-dict `:` type($source) `->` type($result)
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
        OptionalAttr<Micro_OwnerAttr>:$owner
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
        Variadic<AnyType>:$inputs
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

#endif
