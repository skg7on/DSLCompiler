.. _models-mapping-policy:

Mapping policy (``mapping/*.llkmap``)
=====================================

LLKMap is the declarative language a target uses to state which implementations
of MicroIR operations are legal on it. It has two file forms: **layouts**
(name and constrain a storage layout) and **rules** (match an operation or
bounded subgraph and name the implementation that covers it).

The mapping core reads these files and never branches on a target name. Policy
lives in :source:`mapping/ <mapping>` (one directory per target); the generic
matcher, solver, router and search live in
:source:`lib/Mapping <lib/Mapping>`.

.. _models-policy-package:

Package layout and loading
--------------------------

A target package has four parts:

.. list-table::
   :header-rows: 1
   :widths: 34 66
   :class: wide-table

   * - Part
     - Purpose
   * - ``mapping/<target>/layouts.llkmap``
     - Layout declarations: parameters, finite domains, constraints, affine maps
   * - ``mapping/<target>/rules.llkmap``
     - Rules: operation patterns, capability and layout requirements, bundles
   * - Target plugin under ``lib/Target/<Target>/Mapping/``
     - Declares emitter keys and implements target lowering
   * - ``machines/<target>-v2.yaml``
     - The machine facts the policy queries

``llk-opt`` loads the files explicitly through the mapping passes:

.. code-block:: bash

   EMITTERS="avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store,avx2_fused_convert_silu_mul"
   build/bin/llk-opt \
     --micro-map="target=x86-avx2 machine=machines/x86-avx2-v2.yaml layouts=mapping/x86-avx2/layouts.llkmap rules=mapping/x86-avx2/rules.llkmap emitters=$EMITTERS mode=exact top-k=8" \
     input.micro.mlir

``llk-compile`` loads a registered target package by name and resolves its
files under ``--mapping-root``:

.. code-block:: bash

   build/bin/llk-compile --mapping-target=x86-avx2 --mapping-root=. \
     --mapping-mode=exact docs/manual/source/examples/matmul.mlir

The registered names are ``x86-avx2``, ``generic-ai-accel``, ``arm-sve`` and
``arm-sme``. ``--machine`` overrides the profile a package would otherwise load.

.. list-table::
   :header-rows: 1
   :widths: 26 18 56
   :class: wide-table

   * - Option
     - Pass/tool
     - Meaning
   * - ``target``
     - ``micro-map``
     - Opaque label recorded on the selected plan. It does not select a plugin.
   * - ``machine``
     - ``micro-map``
     - Machine profile path.
   * - ``layouts``, ``rules``
     - ``micro-map``
     - LLKMap file paths.
   * - ``emitters``
     - ``micro-map``
     - Comma-separated emitter keys the caller permits. An availability list,
       not proof of lowering.
   * - ``mode``
     - ``micro-map``
     - ``deterministic``, ``beam`` (default) or ``exact``.
   * - ``top-k``, ``beam-width``
     - ``micro-map``
     - Retained complete plans; beam frontier width.
   * - ``candidate``
     - ``micro-map``
     - Search at one ``micro.candidate`` search-space point.
   * - ``require-executable``
     - ``micro-map``
     - Reject a plan missing an execution-affecting decision.

.. _models-policy-lexical:

File forms and lexical rules
----------------------------

There are two entry points: ``layout_file`` and ``rule_file``. Each consumes the
whole file; empty files are accepted. A single file cannot mix the two forms —
the caller selects the loader, not the filename extension.

.. list-table::
   :header-rows: 1
   :widths: 22 78
   :class: wide-table

   * - Token
     - Rule
   * - Identifier
     - ``[A-Za-z_][A-Za-z0-9_.]*``, case-sensitive. Dots are part of the token,
       so ``avx2.blocked_2d`` and ``machine.compute`` are single identifiers.
   * - Integer
     - Decimal, non-negative, interpreted as signed 64-bit. No hexadecimal or
       floating-point literals.
   * - String
     - Double-quoted, no escape sequences. ``\n`` is a backslash and an ``n``.
   * - Version
     - One token matching ``v[0-9]+``; ``v 1`` is not a version.
   * - Comment
     - ``//`` to end of line; ``/* ... */``, non-nesting, must terminate.
   * - Punctuation
     - ``.. -> == != <= >= && ||`` and ``(){}[],;:.<>!+-*/%=``.

Every statement ends with ``;``. Declaration closing braces do not take one; the
``match graph { ... }`` statement is the one braced form that also omits a
trailing semicolon.

.. note::

   The older EBNF in :source:`docs/design/llkmap.ebnf <docs/design/llkmap.ebnf>`
   predates four implemented forms: graph rules, port predicates, the
   ``implements`` clause, and parameterized bundles. This page and the parser in
   :source:`lib/Mapping/MappingRules.cpp <lib/Mapping/MappingRules.cpp>` and
   :source:`lib/Mapping/LayoutConstraints.cpp <lib/Mapping/LayoutConstraints.cpp>`
   describe the current language.

.. _models-policy-layouts:

Layout declarations
-------------------

A layout declaration names a target-owned layout ID, the parameters it ranges
over, the constraints that make it legal, and the affine logical-to-physical
map it describes.

.. code-block:: text

   layout avx2.blocked_2d(int M, int N, int VW) {
     param M in [4..16];
     param N in [4..16];
     param VW in [4..8];
     require rank == 2;
     require VW == machine.compute("vector_engine").lanes(element_type);
     require N % VW == 0;
     implements blocked;
     map (m, n) -> (m, floordiv(n, VW), mod(n, VW));
   }

Grammar:

.. code-block:: text

   layout      ::= "layout" id "(" [ param { "," param } ] ")" "{" { stmt } "}"
   param       ::= ( "int" | "sym" ) id | id
   stmt        ::= "param" id "in" domain ";"
                 | "require" expr ";"
                 | "implements" id ";"
                 | "map" "(" [ id { "," id } ] ")" "->" "(" expr { "," expr } ")" ";"
   domain      ::= "[" uint ".." uint "]" | "{" literal { "," literal } "}"

* A header parameter without ``int``/``sym`` defaults to ``int``.
* A range is inclusive and must have ``upper >= lower``.
* A brace domain lists integer and/or quoted-string literals. Bare symbols such
  as ``{f32, bf16}`` are invalid; write ``{"f32", "bf16"}``.
* Every parameter needs a domain before the layout can be solved. Duplicate
  domains for one name are rejected.
* ``require`` expressions may reference any header parameter, plus the implicit
  bindings ``rank`` and ``element_type``. Requirements do not depend on
  declaration order.
* ``map`` is optional and may occur once. Its result list must be non-empty. The
  map is narrowed to affine expressions when solved: solved parameters become
  constants, so the resulting MLIR map has no symbols.
* ``implements`` is optional and may occur once. It bridges a Micro layout
  *kind* (``blocked``, ``row_major``, ...) to the target layout ID that provides
  it. A search-space binding names a kind; a rule names a layout ID. The clause
  lets a binding select a target layout instead of only vetoing rules.

Layout IDs and parameter names are target vocabulary. They are **not** new
``#micro.layout`` enumerants: a canonical ``#micro.layout`` still records a kind
and its parameters, and the mapping result states which target layout was
chosen.

.. _models-policy-rules:

Rule declarations
-----------------

A rule matches one Micro operation or a bounded subgraph and names the
implementation that covers it.

.. code-block:: text

   rule avx2.vector_add v1 {
     match micro.vector(op = "add");
     param VW in [4..8];
     require VW == machine.compute("vector_engine").lanes("f32");
     require executor kind worker;
     require compute kind vector_engine;
     require layout operand0 satisfies avx2.blocked_2d;
     input "operand0";
     input "operand1";
     output "result";
     bundle "avx2.vector.add.f32" { tile_m = 8 };
     emit "avx2_vector_add";
     cost 4;
   }

Grammar:

.. code-block:: text

   rule        ::= "rule" id [ version ] "{" { stmt } "}"
   stmt        ::= match | graph-match | domain | require | port
                 | bundle | emit | cost
   match       ::= "match" micro-op "(" [ predicate { "," predicate } ] ")" ";"
   graph-match ::= "match" "graph" "{" { node | edge } "}"
   node        ::= "node" id ":" micro-op "(" [ predicate { "," predicate } ] ")" ";"
   edge        ::= "edge" endpoint "->" endpoint ";"
   endpoint    ::= id "." ( "result" | "result" uint
                          | "operand" | "operand" uint )
   predicate   ::= id "=" literal
                 | [ direction "[" uint "]" "." ] "element_type" "=" symbol
                 | [ direction "[" uint "]" "." ] "shape" "[" uint "]" "=" uint
                 | [ direction "[" uint "]" "." ] "access_map" "=" map
   direction   ::= "input" | "output"
   domain      ::= "param" id "in" ( "[" uint ".." uint "]" | "{" literal { "," literal } "}" ) ";"
   require     ::= "require" expr ";"
                 | "require" ( "executor" | "compute" ) "kind" id ";"
                 | "require" "memory" [ direction string ] "kind" id ";"
                 | "require" "layout" id "satisfies" id ";"
   port        ::= ( "input" | "output" ) string ";"
   bundle      ::= "bundle" string [ "{" bundle-param { "," bundle-param } "}" ] ";"
   bundle-param ::= id "=" literal
   emit        ::= "emit" string ";"
   cost        ::= "cost" uint ";"

The accepted operation vocabulary is exactly the set of operations that become
workload nodes:

``micro.mma``, ``micro.vector``, ``micro.reduce``, ``micro.async_copy``,
``micro.tile_async_copy``, ``micro.store``, ``micro.tile_store``.

``micro.tile_view`` and ``micro.tile_partition`` are logical operations folded
into their consumers; they are not matchable nodes. Other ``micro.*`` operations
cannot be matched.

Statement reference
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 74
   :class: wide-table

   * - Statement
     - Meaning
   * - ``match``
     - Operation and predicates the node must satisfy.
   * - ``param``
     - Declares and bounds a tunable parameter. Must precede its first use.
   * - ``require expr``
     - Constraint over parameters, ``rank``, ``element_type`` and machine
       queries.
   * - ``require executor kind K``
     - Placement may use an executor of abstract owner kind ``K``.
   * - ``require compute kind K``
     - Placement must attach a capability whose ``kind`` is exactly ``K``.
       ``refines`` is not applied to compute requirements.
   * - ``require memory kind K``
     - The node's memory occurrence must be of kind ``K`` (conservative form).
   * - ``require memory input "p" kind K``
     - The named port's occurrence must be materialized in ``K``.
   * - ``require layout p satisfies L``
     - The value on port ``p`` must satisfy target layout ``L``.
   * - ``input "p"`` / ``output "p"``
     - Declares a boundary port. Names are unique across both directions.
   * - ``bundle "..."``
     - The opaque target implementation name, with optional typed parameters.
   * - ``emit "..."``
     - The opaque emitter key, validated against the target's declared set.
   * - ``cost N``
     - Optional static cost lower bound contributed by this rule.

A rule needs exactly one ``match`` and nonempty ``bundle`` and ``emit`` names.
Duplicate ``match``, ``bundle``, ``emit`` and ``cost`` clauses are rejected, as
are duplicate port names and duplicate bundle parameter names.

A bare memory requirement and a named one of the same kind, or the same named
port twice, overlap ambiguously and are rejected at load time. ``executor`` and
``compute`` requirements may not name a port.

Predicates
~~~~~~~~~~

An **attribute predicate** (``op = "add"``) compares an operation attribute by
name. A **port predicate** reads a boundary value instead:

.. list-table::
   :header-rows: 1
   :widths: 34 66
   :class: wide-table

   * - Form
     - Reads
   * - ``[input[i].]element_type = f32``
     - The element type of a port's value.
   * - ``[output[i].]shape[d] = 64``
     - The static extent of dimension ``d``.
   * - ``[input[i].]access_map = (d0, d1) -> (d1, d0)``
     - The affine index map of a port.

Without an explicit subject, the default direction applies: element type and
access map read inputs; shape reads outputs. The predicate holds when at least
one port in that direction exposes the property and every exposing port agrees.
Matching is conservative: a dynamic shape, an opaque type or an absent map never
satisfies a predicate.

An ``access_map`` is compared as a simplified MLIR ``AffineMap``, never as text.
Only affine expressions are accepted, so a malformed map is a load-time
diagnostic rather than a predicate that silently never matches.

.. _models-policy-graph-rules:

Graph rules: matching a subgraph
--------------------------------

A rule may implement a whole subgraph. It declares a pattern instead of a single
operation:

.. code-block:: text

   rule avx2.fused_convert_silu_mul v1 {
     match graph {
       node cv: micro.vector(op = "convert");
       node act: micro.vector(op = "silu");
       node gate: micro.vector(op = "mul");
       edge cv.result -> act.operand0;
       edge act.result -> gate.operand0;
     }
     require executor kind worker;
     require compute kind vector_engine;
     input "cv.operand0";
     input "gate.operand1";
     output "gate.result";
     bundle "avx2.fused.convert_silu_mul";
     emit "avx2_fused_convert_silu_mul";
     cost 3;
   }

* Each ``node`` binds a name and takes the same predicates as a one-operation
  match.
* Each ``edge`` is a constraint: the producer's result value must *be* the
  consumer's operand value. Omitted indices mean the first result or operand.
* The **first declared node is the anchor**. Matches are enumerated by trying the
  anchor at each workload node in canonical order, so the match list is a
  property of the rule and graph, not of traversal order.
* A graph rule's ports are its external boundary: operand/result occurrences of
  matched nodes that are not edge endpoints. Operands are named
  ``<node>.operandK``; results are ``<node>.result`` for the first result and
  ``<node>.resultK`` for a nonzero index.
* Duplicate node names, edges naming undeclared nodes, repeated edges, and empty
  patterns are load-time errors.
* A graph rule produces one candidate covering the whole match. Covering is a
  partition: a node covered by a fused candidate cannot be covered again, and a
  link internal to the match is not turned into a connection.

A rule is either a one-operation rule or a graph rule, never both.

.. _models-policy-how-used:

What the mapping phases read
----------------------------

The policy is consumed stage by stage. Each stage asks a different question:

.. list-table::
   :header-rows: 1
   :widths: 16 30 32 22
   :class: wide-table

   * - Phase
     - Reads from LLKMap
     - Reads from the machine
     - Produces
   * - Extract
     - Nothing (the workload vocabulary defines nodes)
     - Nothing
     - ``WorkloadGraph``
   * - Match
     - Operation, predicates, ports, graph pattern
     - Nothing
     - ``MappingCandidate``
   * - Place
     - ``param`` domains, ``require`` expressions, kind requirements,
       layout requirements
     - Executor/compute/memory kinds, lane widths, capacities
     - ``CandidateInstance``
   * - Connect
     - Layout requirements and port facts
     - Links, transfer engines, alignment, transaction size
     - Direct/transfer/transform alternatives
   * - Cover
     - Rule ``cost``, bundle identity
     - Full cost model
     - ``CoveringPlan``
   * - Bind
     - Bundle name, parameters, emitter key, rule ID
     - Concrete resource IDs
     - Mapped kernel metadata
   * - Verify
     - Rule and layout existence, port wiring
     - Referenced executors, memories, links, engines
     - Diagnostics, or acceptance

A rule that cannot be placed yields no instances for that candidate; it does not
fail the whole search, because another rule may cover the same node.

.. _models-policy-solving:

Layout solving
--------------

A layout requirement is not a string comparison. The solver enumerates the
finite domain of every parameter, evaluates the ``require`` expressions with the
machine's facts, and retains the assignments that hold. A solution carries the
concrete parameter values and the affine map built by substituting them.

.. list-table::
   :header-rows: 1
   :widths: 26 74
   :class: wide-table

   * - Behavior
     - Consequence
   * - Finite domain
     - Solving is bounded enumeration. Wide domains cost search time.
   * - Machine queries in ``require``
     - A layout can follow the machine (``lanes(element_type)``) instead of
       hard-coding a width.
   * - No solution
     - The requirement is unsatisfiable; the candidate is unplaceable.
   * - Solve budget exhausted
     - The result is *undecided*, not "illegal". A caller must not read it as
       either.
   * - Truncation
     - Partial results are reported as truncated, never as exhaustive.

Expression evaluation uses signed 64-bit integers, strings and machine handles.
Both operands of ``&&`` and ``||`` are evaluated; they do not short-circuit.
Numeric overflow is not consistently checked. Use non-negative numerators and
positive divisors when relying on ``floordiv``/``mod``.

Machine queries available to policy:

.. list-table::
   :header-rows: 1
   :widths: 40 60
   :class: wide-table

   * - Query
     - Result
   * - ``machine.compute(kind).count``
     - Number of capability nodes of that kind.
   * - ``machine.compute(kind).lanes(dtype)``
     - Elements per instruction for ``dtype``.
   * - ``machine.memory(id).capacity_bytes``
     - Capacity of a concrete memory ID.
   * - ``machine.memory(id).alignment_bytes``
     - Alignment of a concrete memory ID.

A memory *kind* (``"sram"``) is not interchangeable with a concrete ID
(``"sram.0"``). An unknown capability kind or memory ID is a load-time error.

.. _models-policy-bundles:

Bundles and emitters
--------------------

A bundle is opaque to generic mapping code. It may be compared, hashed,
reported and handed to a target plugin, but its name and parameters are never
read as target semantics.

* A bundle may carry typed parameters: ``{ tile_m = 8, layout_blocked = blocked_2d }``.
  An integer literal arrives as an integer attribute. A bare name that matches
  a solved integer rule parameter is replaced by that parameter's value, also
  as an integer attribute — which is how ``{ VW = VW }`` carries a solved
  width. Any other bare or quoted name arrives as a string. Parameters are
  sorted and type-tagged before hashing, so declaration order never changes an
  identity.
* ``emit`` is a bare key. It is validated against the emitter keys the target
  declares.
* ``TargetEmitter::verify`` checks that a bundle is well-formed and names this
  target's emitter. ``verifyOperations`` adds operation-aware checks before
  ranking. ``hasLowering`` answers whether the target lowers that key at all;
  ``lower`` performs the rewrite.
* Declaring an emitter key is not implementing one. The default ``lower``
  rejects, so a configuration-only target cannot silently impersonate a backend.

Target emitter keys for the shipped AVX2 package are listed in
:source:`AVX2MappingTarget.cpp <lib/Target/X86/Mapping/AVX2MappingTarget.cpp>`.
The ``emitters=`` list on the command line is the caller's permitted subset.

.. _models-policy-validation:

Cross-registry validation
-------------------------

Parsing validates one file. ``verifyMappingTarget`` then validates the files
against each other and against the machine:

* every layout requirement names a declared port and an existing layout ID;
* every port predicate names an index the rule declares a port for;
* every capability kind exists on the machine;
* every emitter key is declared by the target;
* every machine query subject (compute kind or memory ID) resolves;
* the rule's executor/compute/memory requirements are roles the machine offers;
* a rule's declared ports cover every port it references.

The last check is reference-based. The mapping core has no operation-arity table
for variadic Micro ops, so a rule that references no port at all still loads
with an empty boundary.

.. _models-policy-search-binding:

Search-space bindings
---------------------

A ``micro.candidate`` can bind a rule at a chosen search-space point. Two binding
values are projected onto explicit requirements rather than recorded as
provenance:

.. list-table::
   :header-rows: 1
   :widths: 22 78
   :class: wide-table

   * - Binding
     - Projection
   * - ``owner_mapping`` (``worker/vector_engine``)
     - A ``/``-separated owner chain. The outer component becomes an executor
       requirement and each inner component that names a compute capability
       becomes a compute requirement. Inner components that name execution
       scopes are validated as nested owners but are not projected. An unknown
       component, or one contradicting the rule's own requirements, rejects the
       match.
   * - ``memory_path`` (``dram:sram:acc``)
     - The allowed memory levels. A rule whose memory requirement sits off the
       path is rejected.
   * - ``tile_layout`` (a Micro layout kind)
     - Bridges the bound kind to the unique target layout whose ``implements``
       clause names it. If several layouts implement it, the bridge is an
       ambiguity error. If none does, the bare kind is kept and only vetoes
       rules that require a layout implementing a different kind.

An unsupported bound axis is rejected explicitly: the node ends with no rule in
effect rather than silently ignoring the choice.

.. _models-policy-worked-example:

A complete policy, annotated
----------------------------

The shipped AVX2 files are the best worked example:

* :source:`layouts.llkmap <mapping/x86-avx2/layouts.llkmap>` defines
  ``avx2.row_major`` and ``avx2.blocked_2d``. The blocked layout's ``VW``
  parameter is constrained by the machine's vector lane width, so the same
  policy follows a different machine profile without edits.
* :source:`rules.llkmap <mapping/x86-avx2/rules.llkmap>` defines one rule per
  operation variant. ``micro.vector`` is a single op with an ``op`` attribute, so
  ``add``, ``convert``, ``silu`` and ``mul`` are separate rules matched on that
  attribute. There is deliberately no catch-all: an unknown variant remains
  ``no_matching_rule`` rather than binding to an implementation that does not
  exist.
* ``avx2.fused_convert_silu_mul`` is a graph rule covering the SwiGLU epilogue
  as one unit. The intermediate tiles never cross a rule boundary, which is what
  makes the fused implementation a different program rather than a relabelling.
* ``avx2.async_copy`` names its ports' memories because its boundary values are
  plain tensors without tile memory facts. Every other rule leaves memory to the
  tile its operands carry, so the same ``micro.vector "add"`` can run in SRAM in
  one program and in the accumulator in another.

.. _models-policy-authoring:

Authoring checklist
-------------------

#. Keep domains small. The solver enumerates them; a wide domain costs search
   time and rarely buys a better plan.
#. Express target policy as data. A branch on a target name in
   ``lib/Mapping`` or ``lib/Machine`` is a boundary violation.
#. Require kinds, not concrete resource IDs. Executor requirements match an
   abstract owner or a declared refinement; compute and memory requirements
   match the literal kind the profile declares, so write ``vector_engine``
   rather than ``vector``. Concrete IDs belong in the selected plan.
#. Give every referenced port a declaration. A layout requirement or port
   predicate on an undeclared port is rejected at load time.
#. Use one rule per variant. A catch-all that binds an unimplemented variant is
   worse than an explicit ``no_matching_rule``.
#. Keep bundles opaque. Only the target plugin may interpret a parameter.
#. Add a fixture for each new rule. Existing checks live under
   :source:`test/Mapping <test/Mapping>` and :source:`test/Conversion/MicroMapping <test/Conversion/MicroMapping>`.

.. seealso::

   :doc:`/models/machine-model` for the facts these rules query,
   :doc:`/models/schedules` for the tile choices a rule receives,
   :doc:`/tutorials/02-mapping` for a runnable end-to-end example, and
   :source:`docs/design/micro-ir-mapping-workflow.md <docs/design/micro-ir-mapping-workflow.md>`
   for the contributor workflow.
