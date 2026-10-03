# LLKMap Layout Grammar

LLKMap is the small declarative language that describes target layout legality.
Its layout subset — the part implemented today — lets a target declare the
concrete layouts it supports and the constraints under which each is legal,
**without adding target vocabulary to the `micro` dialect**.

A file lives beside the machine profile it belongs to:

```text
mapping/
  x86-avx2/
    layouts.llkmap
```

Layout ids (`avx2.blocked_2d`) are mapping *results*, not `#micro.layout`
enumerants (design §13.4). Machine facts are read through `MachineModel`
queries only, so a layout file never names a concrete executor.

## Grammar

```text
file       ::= layout*
layout     ::= "layout" id "(" params ")" "{" stmt* "}"
params     ::= [ param ("," param)* ]
param      ::= [ "int" | "sym" ] ident          // a bare name means int
stmt       ::= domain | require | map
domain     ::= "param" ident "in" ( "[" int ".." int "]" | "{" literal ("," literal)* "}" ) ";"
require    ::= "require" expr ";"
map        ::= "map" "(" ident ("," ident)* ")" "->" "(" expr ("," expr)* ")" ";"
expr       ::= or
or         ::= and ("||" and)*
and        ::= equality ("&&" equality)*
equality   ::= relational (("==" | "!=") relational)*
relational ::= additive (("<" | "<=" | ">" | ">=") additive)*
additive   ::= multiplicative (("+" | "-") multiplicative)*
multiplicative ::= unary (("*" | "/" | "%") unary)*
unary      ::= ("!" | "-") unary | postfix
postfix    ::= primary ("." ident "(" args ")")*
primary    ::= int | string | ident | "(" expr ")" | call
call       ::= ident "(" args ")"
id         ::= (letter | "_") (alnum | "_" | ".")*
```

- Comments are `//` to end of line and `/* ... */`.
- Identifiers may contain dots, which is how layout ids and the `machine.`
  query prefix stay single tokens.

## Declarations

A `layout` names its integer and symbolic parameters, then states:

- **domains** — the finite set a parameter ranges over. An integer parameter
  uses `[lo..hi]`; a symbolic parameter uses `{ "a", "b" }`. A parameter
  without a domain is *unbound* and rejected when solved.
- **constraints** — boolean `require` expressions, all of which must hold.
- **map** — the affine logical-to-physical mapping. The `map` clause's
  identifiers are the logical dimensions; referenced parameters become affine
  symbols. Only affine operations are allowed (`+ - * floordiv ceildiv mod`,
  constants, dims, symbols).

### Builtins

| Name | Meaning |
|---|---|
| `rank` | tensor rank of the value being laid out (from the solve context) |
| `element_type` | element type name (from the solve context) |

### Machine queries

| Query | Result |
|---|---|
| `machine.compute(<kind>).lanes(<dtype>)` | elements per instruction for `dtype` |
| `machine.compute(<kind>).count` | number of capabilities of `kind` |
| `machine.memory(<id>).capacity_bytes` | memory capacity |
| `machine.memory(<id>).alignment_bytes` | memory alignment |

## Example

```text
layout avx2.blocked_2d(int M, int N, int VW) {
  param M in [4..16];
  param N in [4..16];
  param VW in [4..8];
  require rank == 2;
  require VW == machine.compute("vector_engine").lanes(element_type);
  require N % VW == 0;
  map (m, n) -> (m, floordiv(n, VW), mod(n, VW));
}
```

## Load-time validation

The parser rejects, with a file/line/column diagnostic:

- duplicate layout ids and duplicate parameter names;
- a `param ... in` clause for an undeclared parameter, or a duplicate domain;
- an identifier in a `require`/`map` that is neither a declared parameter, a
  map dimension, nor a builtin;
- an unknown function (`floordiv`, `ceildiv`, `mod`, `min`, `max`,
  `machine.compute`, `machine.memory`) or member query (`lanes`, `count`,
  `capacity_bytes`, `alignment_bytes`);
- a malformed statement, missing `;`, or unterminated `{`, string, or block
  comment.
