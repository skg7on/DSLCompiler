# LLKMap formal syntax

This reference describes the `.llkmap` language implemented on merged `main`
at commit `31fd9eb440f9cd7fe9946a873237200d57dee5ed`, with the accompanying
fix to inclusive range parsing at the signed 64-bit maximum. It specifies syntax,
name resolution, and the validation stages separately. It does not introduce
new language behavior. The complete token-level grammar is [llkmap.ebnf](llkmap.ebnf).

## File forms and notation

There are two entry points: `layout_file` for `parseLayoutText`/`loadLayoutFile`,
and `rule_file` for `parseRuleText`/`loadRuleFile`. Each consumes the entire file.
Empty files are accepted. A single file cannot mix layout and rule declarations;
the caller selects the loader, not the filename extension. No file-level version,
imports, namespaces, or preprocessing directives are defined.

In the EBNF, quoted text denotes a literal token, commas concatenate,
`|` chooses an alternative, `[...]` is optional, and `{...}` repeats zero or
more times. Parentheses group alternatives. `EOF` denotes end of input.
The grammar operates on tokens after whitespace and comments are removed.
Every statement ends with `;`; declaration closing braces do not take `;`.
Comma-separated lists have no trailing comma.

## Lexical syntax

| Token | Portable spelling | Interpretation |
|---|---|---|
| `IDENT` | `[A-Za-z_][A-Za-z0-9_.]*` | Case-sensitive identifier; dots belong to the token |
| `UNTYPED_PARAM` | Any `IDENT` except exactly `int` or `sym` | Untyped layout header parameter name |
| `UINT` | `[0-9]+` | Decimal non-negative integer, including leading zeros |
| `STRING` | `"` followed by any characters except `"`, followed by `"` | Quoted string; no escape sequences |
| `VERSION` | One `IDENT` matching `v[0-9]+` | Rule version, e.g. `v12`; not `v 12` |
| `EOF` | No spelling | End of input |

The lexer uses C character classification for identifier letters and digits;
the table gives the portable ASCII authoring profile. Identifiers are consumed
greedily: `machine.compute` is one token, while the dot in
`machine.compute("vector_engine").count` is punctuation. Dotted identifiers
need not consist of nonempty segments; repeated or trailing dots are lexically
accepted. Adding spaces inside a dotted identifier changes its tokenization.

Whitespace is space, tab, carriage return, or line feed. `//` comments end at
line feed or EOF. `/* ... */` comments end at the first `*/`, do not nest,
and must terminate. Comments are not recognized inside strings.

Strings can be empty and the current lexer permits literal newlines in them.
Backslashes are ordinary characters: `\n` is not a newline escape and `\"`
does not embed a quote. Prefer single-line strings for portable authoring.

Multi-character punctuation is `.. -> == != <= >= && ||`.
Single-character punctuation is `(){}[],;.<>!+-*/%=`. Other characters are
rejected. There are no floating-point, hexadecimal, or boolean literal tokens.
All language keywords are contextual `IDENT` tokens, not globally reserved.

Ordinary integer tokens are converted to signed 64-bit values; use literals
in `0..9223372036854775807`. Larger literals are outside the supported numeric
profile and are not guaranteed a recoverable diagnostic. Rule versions use
unsigned 64-bit conversion of the digits after `v`. A negative expression is
formed with unary `-`; negative domain members, range bounds, predicate values,
and costs are not accepted.

## Layout declarations

```ebnf
layout = "layout", IDENT, "(", [ layout_param, { ",", layout_param } ],
         ")", "{", { layout_stmt }, "}" ;
layout_param = ( "int" | "sym" ), IDENT | UNTYPED_PARAM ;
layout_stmt = domain_stmt | expr_requirement | map_stmt ;
domain_stmt = "param", IDENT, "in", domain, ";" ;
domain = "[", UINT, "..", UINT, "]"
       | "{", domain_literal, { ",", domain_literal }, "}" ;
domain_literal = UINT | STRING ;
expr_requirement = "require", expr, ";" ;
map_stmt = "map", "(", [ IDENT, { ",", IDENT } ], ")", "->",
           "(", expr, { ",", expr }, ")", ";" ;
```

A bare header parameter defaults to `int`; `sym` records symbolic intent.
At the start of a header parameter, `int` and `sym` always select the typed
alternative and require a following identifier. Thus `layout r(int)` and
`layout r(sym)` are invalid, while `layout r(int int, sym sym)` is valid.
These names remain allowed in other identifier positions.
Ranges are inclusive, require upper bound >= lower bound, and are materialized
as ascending values. Brace domains must contain at least one integer or quoted
string. Bare symbols such as `{f32, bf16}` are invalid domains; use
`{"f32", "bf16"}`. The parser accepts mixed integer/string brace domains;
the header annotation does not impose a load-time homogeneous-domain check.

Layout ids and header parameter names must be unique within their registry
and declaration respectively. Each domain must name a header parameter and
may occur only once. Every parameter needs a domain when the layout is solved;
this is checked after parsing. Requirements may refer to any header parameter,
`rank`, and `element_type`, independently of domain statement order.

A map is optional and may occur at most once. Its dimension list can be empty,
but its result list cannot. Map expressions may refer to dimensions and header
parameters; `rank` and `element_type` are not implicit map bindings.
Dimension-name uniqueness and collisions with parameter names are not rejected
at parse time; authors should use distinct names (dimension lookup takes priority).

The parser initially accepts the shared expression grammar in maps. Conversion
when solving narrows it to affine expressions: integer constants, dimensions,
solved integer parameters, unary `-`, `+`, `-`, multiplication by a constant,
`/`, `%`, and two-argument `floordiv`, `ceildiv`, `mod`. Solved parameters become
constants, so the resulting MLIR map has zero symbols. Strings, machine queries,
logical/comparison operations, and `min`/`max` cannot be converted to affine maps.
Divisors should be positive constants after parameter substitution; early
validation does not fully enforce MLIR affine legality.

```text
layout avx2.blocked_2d(int M, int N, int VW) {
  param M in [4..16];
  param N in [4..16];
  param VW in {4, 8};
  require rank == 2;
  require VW == machine.compute("vector_engine").lanes(element_type);
  require N % VW == 0;
  map (m, n) -> (m, floordiv(n, VW), mod(n, VW));
}
```

## Rule declarations

```ebnf
rule = "rule", IDENT, [ VERSION ], "{", { rule_stmt }, "}" ;
rule_stmt = match_stmt | domain_stmt | expr_requirement | kind_requirement
          | layout_requirement | port_stmt | bundle_stmt | emit_stmt | cost_stmt ;
match_stmt = "match", micro_op, "(", [ predicate, { ",", predicate } ], ")", ";" ;
predicate = IDENT, "=", ( UINT | STRING | IDENT ) ;
kind_requirement = "require", ( "executor" | "compute" | "memory" ),
                   "kind", IDENT, ";" ;
layout_requirement = "require", "layout", IDENT, "satisfies", IDENT, ";" ;
port_stmt = ( "input" | "output" ), STRING, ";" ;
bundle_stmt = "bundle", STRING, ";" ;
emit_stmt = "emit", STRING, ";" ;
cost_stmt = "cost", UINT, ";" ;
```

The accepted `micro_op` vocabulary is exactly `micro.mma`, `micro.vector`,
`micro.reduce`, `micro.async_copy`, `micro.tile_async_copy`, `micro.store`, and
`micro.tile_store`. Other `micro.*` operations cannot be matched by this profile.

Rule ids must be unique. The version defaults to 1; `v0` and leading zeros
are accepted. Each rule needs one match and nonempty bundle and emitter names.
Duplicate match and cost clauses are rejected; duplicate bundle/emit clauses
are rejected once a nonempty name has been stored. An initial empty string can
therefore evade the duplicate check if followed by a nonempty value; authors
must use exactly one nonempty clause for each. Cost is optional.

A rule's `param` statement declares and bounds its parameter. Duplicate domains
are rejected. A parameter must be declared before its use in a requirement;
`rank` and `element_type` are implicit requirement bindings. Predicate bare
identifiers are symbolic string values, not parameter references. Predicates
match operation attributes by name and value; absent attributes do not match.
The parser does not reject repeated predicate attribute names.

Port declarations are optional at parse time and use quoted names. Names must
be unique across inputs and outputs. A layout requirement uses an identifier
port name and a layout id; it does not declare a port. Parsing does not enforce
operation arity or that the referenced port has been declared. Bundle and emitter
names are opaque target-owned strings. Target validation checks layout ids,
capability kinds, and emitter keys against the target registries.

Special `require` alternatives are selected contextually: a role followed by
`kind` selects a capability requirement; `layout IDENT satisfies` selects a
layout requirement. Other `require` statements parse as expressions.

```text
rule avx2.vector_add v1 {
  match micro.vector(op = "add");
  param VW in {4, 8};
  require VW == machine.compute("vector_engine").lanes(element_type);
  require executor kind worker;
  require compute kind vector_engine;
  require layout operand0 satisfies avx2.blocked_2d;
  input "operand0";
  input "operand1";
  output "result";
  bundle "avx2.vector.add.f32";
  emit "avx2_vector_add";
  cost 4;
}
```

## Expressions and evaluation

The complete expression productions are in [llkmap.ebnf](llkmap.ebnf).
Precedence from lowest to highest is:

| Level | Operators / forms | Associativity |
|---|---|---|
| 1 | `||` | Left |
| 2 | `&&` | Left |
| 3 | `== !=` | Left |
| 4 | `< <= > >=` | Left |
| 5 | `+ -` | Left |
| 6 | `* / %` | Left |
| 7 | Prefix `! -` | Right |
| 8 | Postfix `.member` / `.member(arguments)` | Left |
| 9 | Integer, string, identifier, named call, parenthesized expression | Primary |

Calls have zero or more comma-separated expressions syntactically. There is
no unary `+`, indexing, array literal, ternary expression, or quantifier in this
profile. A bare `f32` in a requirement is a variable reference; write
`element_type == "f32"` to compare against a literal string.

Load-time expression validation checks binding names and the allowed function
and member names. It does not completely validate operand types, arity, receivers,
or machine query subjects. Evaluation uses signed 64-bit integers, strings,
and machine handles. Arithmetic and ordering require integers; equality requires
matching types. Logical operations use integer zero/nonzero and return 0 or 1.
Both operands of `&&` and `||` are evaluated; they do not short-circuit.
A satisfied requirement evaluates to a nonzero integer.

| Function / query | Intended arguments and result |
|---|---|
| `floordiv(a, b)` | Two integers; scalar evaluator uses truncating division |
| `ceildiv(a, b)` | Two integers, positive `b`; computes `(a + b - 1) / b` |
| `mod(a, b)` | Two integers; scalar evaluator uses remainder |
| `min(a, b)`, `max(a, b)` | Two integers; integer result |
| `machine.compute(kind).count` | String capability kind; number of matching capability nodes |
| `machine.compute(kind).lanes(dtype)` | String kind and dtype; machine lane query result |
| `machine.memory(id).capacity_bytes` | String concrete memory id; capacity in bytes |
| `machine.memory(id).alignment_bytes` | String concrete memory id; alignment in bytes |

Scalar `/` and `%` use C++ truncation toward zero and remainder, respectively.
Consequently scalar `floordiv`/`mod` are not mathematical floor/modulo for negative
operands. Use non-negative numerators and positive divisors for the intended
scalar floor/ceiling/modulo behavior. In affine maps, `/` and `%` instead become
MLIR floor-division and modulo operations. Numeric overflow is not consistently
checked. A memory kind such as `"sram"` is not interchangeable with a concrete
memory id such as `"sram.0"`.

The postfix grammar is more permissive than the intended query forms above;
parsing a member call does not establish that its receiver and arguments are
valid. The table is the supported authoring contract, not a claim that every
other syntactically expressible combination gets an early diagnostic.

## Syntax boundaries and pending extensions

| Example | Status in this profile |
|---|---|
| `param T in {"f32", "bf16"};` | Valid domain statement |
| `param T in {f32, bf16};` | Invalid: domain symbols must be quoted |
| `param N in [-1..8];` | Invalid: bounds are unsigned tokens |
| `rule r v 1 { ... }` | Invalid: version must be one token `v1` |
| `map () -> (0);` | Valid map syntax |
| `map (i) -> ();` | Invalid: at least one result required |
| `cost -1;` | Invalid: cost is an unsigned token |
| `match micro.vector(op = add);` | Valid: `add` is a symbolic predicate value |

PR [#98](https://github.com/skg7on/DSLCompiler/pull/98), inspected at head
`d8adaa7b9b28f5798b49cb5dcdd6612e1d4f2ac5`, is a separate pending extension
profile. Its port predicates (element type, shape, access map), parameterized
bundles, and quantified constraints are not productions of this merged-baseline
grammar. This reference must be revised against the final merged parser before
those forms are advertised as supported.

## Implementation traceability

| Layer | Source |
|---|---|
| Lexer, expression parser, name validation, scalar evaluation, domain parsing | [LlkMap.cpp](../../lib/Mapping/LlkMap.cpp) |
| Layout parser and affine conversion | [LayoutConstraints.cpp](../../lib/Mapping/LayoutConstraints.cpp) |
| Rule parser, predicates, registry | [MappingRules.cpp](../../lib/Mapping/MappingRules.cpp) |
| Accepted operation vocabulary | [WorkloadGraph.cpp](../../lib/Mapping/WorkloadGraph.cpp) |
| Cross-registry validation | [MappingTarget.cpp](../../lib/Mapping/MappingTarget.cpp) |

The earlier [layout guide](llkmap-layout-grammar.md) and
[rule guide](llkmap-rule-grammar.md) provide introductory context. This formal
reference resolves their shorthand: `v1` is one token, domain symbols must be
quoted, zero map dimensions are allowed, and solved map parameters are constants.
