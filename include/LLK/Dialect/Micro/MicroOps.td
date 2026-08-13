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

#endif
