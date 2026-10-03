# LLKMap Rule Grammar

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
rule      ::= "rule" id [ "v" int ] "{" stmt* "}"
stmt      ::= match | domain | require | port | bundle | emit | cost
match     ::= "match" micro-op "(" [ predicate ("," predicate)* ] ")" ";"
predicate ::= ident "=" literal
domain    ::= "param" ident "in" ( "[" int ".." int "]" | "{" literal ("," literal)* "}" ) ";"
require   ::= "require" expr ";"
            | "require" executor "kind" ident ";"
            | "require" compute  "kind" ident ";"
            | "require" memory   "kind" ident ";"
            | "require" "layout" ident "satisfies" id ";"
port      ::= ( "input" | "output" ) string ";"
bundle    ::= "bundle" string ";"
emit      ::= "emit" string ";"
cost      ::= "cost" int ";"
micro-op  ::= "micro." ident
literal   ::= int | string | ident      // a bare name such as `f32`
```

An expression is the shared LLKMap grammar: `rank`, `element_type`, arithmetic,
comparisons, boolean logic, and `machine.*` queries.

## Example

```text
rule avx2.vector_add v1 {
  match micro.vector(kind = "add", element_type = f32);
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
| `match` | the Micro operation and attribute predicates it must satisfy |
| `param` | a tunable parameter; a `param` statement both declares and bounds it, so it must appear before its first use |
| `require expr` | a constraint over parameters and machine facts |
| `require <role> kind <k>` | an abstract capability requirement; `role` is `executor`, `compute`, or `memory` |
| `require layout <port> satisfies <id>` | the value on `<port>` must satisfy that layout |
| `input`/`output` | named boundary values, one declaration per name |
| `bundle` | an opaque target-owned implementation name (exactly one) |
| `emit` | an opaque emitter key, validated against the target's declared set (exactly one) |
| `cost` | an optional static cost lower bound |

## Load-time validation

Parsing rejects, with a `file:line:column` diagnostic:

- a duplicate rule id;
- a missing `match`, `bundle`, or `emit`, or a duplicate `match`/`bundle`/`emit`/`cost`;
- a Micro operation outside the workload vocabulary;
- a duplicate port name;
- an identifier in a `require` that is neither a declared parameter nor a builtin;
- a malformed statement, missing `;`, or unterminated `{`.

Cross-registry validation then rejects a rule that names an **unknown layout
id**, an **unknown capability kind**, or an **unknown emitter key** — see
`verifyMappingTarget` in `LLK/Mapping/MappingTarget.h`.
