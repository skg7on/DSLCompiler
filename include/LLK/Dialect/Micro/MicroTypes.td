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

#endif
