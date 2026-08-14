#ifndef MICRO_TYPES_TD
#define MICRO_TYPES_TD

include "MicroDialect.td"

//===----------------------------------------------------------------------===//
// Async token type
//===----------------------------------------------------------------------===//

def Micro_AsyncToken : TypeDef<Micro_Dialect, "AsyncToken"> {
    let mnemonic = "async_token";
    let summary = "Token for asynchronous data movement or compute dependency";
    let description = [{
        Represents a synchronization token for pipelined async operations.
        Used by micro.async_copy and micro.wait to model async dependencies.
    }];
    let parameters = (ins);
}

//===----------------------------------------------------------------------===//
// Tile type
//===----------------------------------------------------------------------===//

def Micro_Tile : TypeDef<Micro_Dialect, "Tile"> {
    let mnemonic = "tile";
    let summary = "A tile value: shape + element type + layout + memory + owner";
    let description = [{
        The primary execution value of the micro dialect. A tile carries its
        shape, element type, layout, memory space, and owner scope. Logical
        tiles have no memory space; materialized tiles carry a memory space;
        execution tiles and instruction fragments carry an owner.
    }];

    let parameters = (ins
        ArrayRefParameter<"int64_t">:$shape,
        "Type":$elementType,
        "::mlir::micro::LayoutAttr":$layout,
        "::mlir::micro::MemorySpaceAttr":$memory,
        "::mlir::micro::OwnerAttr":$owner
    );

    let hasCustomAssemblyFormat = 1;
    let genVerifyDecl = 1;
}

#endif
