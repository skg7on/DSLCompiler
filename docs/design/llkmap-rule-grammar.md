# LLKMap Rule Grammar

For the formal merged-implementation grammar and validation rules, see
[LLKMap formal syntax](llkmap-syntax.md) and [the EBNF](llkmap.ebnf).

The rule subset of LLKMap describes how a Micro operation is implemented on a
target. Rules live beside the layouts they reference:

```text
mapping/
  x86-avx2/
    layouts.llkmap
    rules.llkmap
```

Rules share the lexer, expression grammar, and evaluator with layout
declarations (see `llkmap-layout-grammar.md`). A rule covers **exactly one
Micro operation** (design §14.2); fusion is a later, explicitly-listed
extension.

## Grammar

```text
file      ::= rule*
rule      ::= "rule" id [ version ] "{" stmt* "}"
stmt      ::= match | domain | require | port | bundle | emit | cost
match     ::= "match" micro-op "(" [ predicate ("," predicate)* ] ")" ";"
predicate ::= attr-predicate | port-predicate
attr-predicate ::= ident "=" literal
port-predicate ::= [ subject "." ] property
subject   ::= ( "input" | "output" ) "[" int "]"
property  ::= "element_type" "=" ( ident | string )
            | "shape" "[" int "]" "=" int
            | "access_map" "=" map
map       ::= "(" ident ("," ident)* ")" "->" "(" expr ("," expr)* ")"
domain    ::= "param" ident "in" ( "[" int ".." int "]" | "{" domain-literal ("," domain-literal)* "}" ) ";"
require   ::= "require" expr ";"
            | "require" executor "kind" ident ";"
            | "require" compute  "kind" ident ";"
            | "require" memory   [ ( "input" | "output" ) string ] "kind" ident ";"
            | "require" "layout" ident "satisfies" id ";"
port      ::= ( "input" | "output" ) string ";"
bundle    ::= "bundle" string [ "{" bundle-param ("," bundle-param)* "}" ] ";"
bundle-param ::= ident "=" ( int | ident | string )
emit      ::= "emit" string ";"
cost      ::= "cost" int ";"
micro-op  ::= "micro." ident
version   ::= one identifier token matching v[0-9]+
domain-literal ::= int | string
literal   ::= int | string | ident      // a bare name such as `f32`
```

An expression is the shared LLKMap grammar: `rank`, `element_type`, arithmetic,
comparisons, boolean logic, and `machine.*` queries.

## Predicates

An **attribute predicate** (`kind = "add"`) compares the operation's attribute
of that name, as before. `element_type`, `shape`, and `access_map` are reserved
property names and instead read a boundary value:

| Predicate | Reads |
|---|---|
| `[input[i].]element_type = f32` | the element type of a port's value |
| `[output[i].]shape[d] = 64` | the static dimension `d` of a port's shape |
| `[input[i].]access_map = (d0, d1) -> (d1, d0)` | the affine index map of a port |

A `subject` names exactly one port. Without one, the predicate applies to the
property's **default direction** — element type and access map read inputs,
shape reads outputs — and holds when at least one port in that direction
exposes the property and every port that exposes it agrees.

Port properties are matched against real port data (`WorkloadPort::type` and
`WorkloadPort::accessMap`); a bare `!micro.tile` is unwrapped to its element
type and shape. Matching is **conservative**: a port that does not expose the
property (an opaque type, a dynamic shape, an absent affine map) never
satisfies the predicate, and a predicate with no exposing port does not match.
The `access_map` value is compared as an MLIR `AffineMap` after simplification,
never as text, so two spellings of the same map are equivalent.

A bare `input`/`output` without a `[i]` subject is still an attribute name, so
`micro.mma(input = bf16)` keeps matching the operation's `input` attribute.

## Example

```text
rule avx2.vector_add v1 {
  match micro.vector(op = "add", element_type = f32, shape[0] = 8);
  param VW in [4..8];
  require VW == machine.compute("vector_engine").lanes(element_type);
  require executor kind worker;
  require compute kind vector_engine;
  require layout operand0 satisfies avx2.blocked_2d;
  input "operand0";
  output "result";
  bundle "avx2.vector.add.f32";
  emit "avx2_vector_add";
  cost 4;
}
```

## What each field means

| Field | Meaning |
|---|---|
| `match` | the Micro operation and the attribute and port predicates it must satisfy |
| `param` | a tunable parameter; a `param` statement both declares and bounds it, so it must appear before its first use |
| `require expr` | a constraint over parameters and machine facts |
| `require <role> kind <k>` | an abstract capability requirement; `role` is `executor`, `compute`, or `memory` |
| `require memory <dir> "<port>" kind <k>` | the named port's occurrence is materialized in a memory of kind `<k>` |
| `require layout <port> satisfies <id>` | the value on `<port>` must satisfy that layout |
| `input`/`output` | named boundary values, one declaration per name |
| `bundle` | an opaque target-owned implementation name (exactly one), with optional typed parameters |
| `emit` | an opaque emitter key, validated against the target's declared set (exactly one) |
| `cost` | an optional static cost lower bound |

## Named-port memory requirements

A `memory` requirement may name the port it governs, so an operation with several
outputs can place each in its own memory instead of leaving the output-to-memory
association implicit:

```text
require memory output "small" kind sram;
require memory output "large" kind dram;
```

The subject is resolved through the rule's declared ports (`output "small";`),
then to the matched node's result occurrence by position — the same wiring a
`layout` requirement uses. The requirement is charged to *that* occurrence:
during capacity search, every output the operation writes must have exactly one
selected memory, and the occurrence's own bytes are charged to it. A named-port
association therefore supports a combination the legacy bare form cannot — a
multi-output node with several memory bindings — because the graph now fixes
which output goes where.

The legacy bare `require memory kind sram` remains valid, but only where its role
assignment is unique: it names no port, so placement binds one memory per *kind*
and the search keeps the conservative rule that a node with several outputs and
several bare bindings has no output-to-memory association and is rejected rather
than admitted on an unsafe lower bound. A bare and a named requirement of the
same kind, or one named port and kind twice, overlap ambiguously and are
rejected at load time. A rule that mixes a bare requirement of one kind with a
named requirement of another parses but is rejected at search time as an
ambiguous association, since it leaves some occurrence's memory unfixed.
`executor` and `compute` requirements may not name a port.

## Bound search axes

When a rule is matched at a bound search-space point (`micro.candidate`), the
binding's `owner_mapping` and `memory_path` values are projected onto explicit
requirements rather than recorded as provenance:

- `owner_mapping` (`worker/vector_engine`) is a `/`-separated owner chain; its
  outer component becomes an abstract executor requirement and any inner compute
  kind a compute requirement, so the binding changes which executors and computes
  placement may bind. A component the machine does not model, or one that
  contradicts the rule's own executor/compute requirements, rejects the match.
- `memory_path` (`dram:sram:acc`) is the set of allowed memory levels; a rule
  whose memory requirement sits off the path is rejected, as is a path naming a
  level the machine does not model.

An unsupported bound axis is rejected explicitly (the node ends with no rule in
effect); it is never silently ignored.

## Target bundles (design §14.3)

A `bundle` is opaque to generic mapping code: it may compare, hash, report, and
hand it to a target plugin, but it never reads the name or a parameter as target
semantics. A rule may declare typed parameters on its bundle:

```text
bundle "avx2.vector.add.f32" { tile_m = 8, layout_blocked = blocked_2d };
```

A value is an **integer** or a **symbolic name** (bare or quoted); the spelling
chooses the type, so `tile_m = 8` reaches the plan as an integer attribute and
`layout_blocked = blocked_2d` as a string. Duplicate parameter names are a
load-time error. The parameters travel with the selected bundle from the rule
through the plan (`TargetBundle{name, parameters, emitterKey}`), and are hashed
and printed in sorted, type-tagged order so declaration order never changes an
id. `emit` stays a bare key; only `bundle` accepts a parameter block.

## Load-time validation

Parsing rejects, with a `file:line:column` diagnostic:

- a duplicate rule id;
- a missing `match`, `bundle`, or `emit`, or a duplicate `match`/`bundle`/`emit`/`cost`;
- a Micro operation outside the workload vocabulary;
- a duplicate port name;
- a duplicate bundle parameter name, or a bundle parameter with no value;
- a `shape` predicate without a dimension index, a port subject naming a
  property other than `element_type`/`shape`/`access_map`, or a missing
  non-negative port index;
- a malformed predicate value: `element_type` must be symbolic and `shape`
  must be an integer;
- an `access_map` that references a dimension it did not declare, or that is
  not affine;
- an identifier in a `require` that is neither a declared parameter nor a builtin;
- a memory requirement whose `input`/`output` subject names a port that is not
  declared, or is declared in the other direction, or a port subject on an
  `executor` or `compute` requirement;
- two `memory` requirements that overlap in role, kind, and port (the same named
  port twice, or a bare and a named requirement of the same kind);
- a malformed statement, missing `;`, or unterminated `{`.

Cross-registry validation then rejects a rule that names an **unknown layout
id**, an **unknown capability kind**, or an **unknown emitter key** — see
`verifyMappingTarget` in `LLK/Mapping/MappingTarget.h`.
