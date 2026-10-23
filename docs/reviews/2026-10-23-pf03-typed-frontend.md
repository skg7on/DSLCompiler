# PF03 typed frontend contract verification

**Date:** 2026-10-23 (Asia/Shanghai).

**Implementation:** [`b78ee152c2102b4a1e406e77570377018187b4fc`](https://github.com/skg7on/DSLCompiler/commit/b78ee152c2102b4a1e406e77570377018187b4fc).

**Scope:** [issue #214](https://github.com/skg7on/DSLCompiler/issues/214), PF03 in
the [frontend plan](../superpowers/plans/2026-10-18-python-frontend-refinement.md).
Implementation starts from main `29b00ebf7085fadd157991fe4425615c74d8b36d`, after
[PF02 PR #241](https://github.com/skg7on/DSLCompiler/pull/241) merged. Source changes
and generated artifacts use `.claude/worktrees/issue214-typed-frontend` on
`feat/issue214-typed-frontend`.

## Delivered contract

- Frozen records and immutable nested metadata in `llk.frontend`; explicit
  rank-two compact tensor ports with read/write effects, positive three-axis
  launch/block metadata and checked signed 64-bit extents/allocation products.
- Finite opcode schemas with SSA/type/effect checks, exact masks and load padding,
  typed nested counted-loop induction/iter/init/yield regions, no escaping local
  values, and ordered explicit cast sites matching the numerical contract.
- Signature binding with positional/keyword/default arguments, sentinel identity,
  renamed/postponed annotations, immutable constants, source-located failures and
  once-only callable grids. `TensorArgument` retains opaque providers without
  reading their storage; runtime addresses and providers never serialize.
- Canonical UTF-8 JSON, strict decoding and a checked
  [Draft 2020-12 schema](../design/python-frontend-v1.schema.json). The independent
  [literal add fixture](../../test/python/frontend_add_v1.json) pins bytes; object
  key order is canonical while port/operation/region order remains semantic.
- Configuration content snapshots participate in identity. Wire decoding requires
  an explicit object and does not consult the decoder's filesystem.

The [guide](../manual/source/python-frontend.rst) documents the exact loop wire
representation and binding boundary. JSON Schema describes structural grammar;
Python additionally checks dependent types, SSA, effects and overflow. Python
requires integer tokens rather than JSON Schema's mathematical integer semantics.

## Red/green evidence and review

The initial focused command failed collection twice with `ModuleNotFoundError:
No module named 'llk.frontend'`, establishing the absent records/functions.
Implementation then passed the authored fixture and negative cases. Additional
red tests reproduced mismatched program/options numerical policies, unreadable
configuration text and a dropped dtype-named immutable capture; each is covered
by the passing suite.

One fresh independent review found two important issues and one minor issue:

1. On Python 3.14, even `inspect.signature` with annotationlib's string format
   executes deferred annotation expressions. A real source-module regression
   observed a call side effect. Binding now inspects an annotation-free function
   copy and reads deferred annotation syntax from the AST without invoking the
   annotator. The regression requires both rejection and zero calls. A valid
   annotation-only local alias plus positional-only/keyword-only defaults also
   passes on all three tested Python versions.
2. A wire `config_files: null` invoked the Python constructor's local snapshot
   convenience path. A failing test replaced that path with a filesystem-read
   trap. The decoder now rejects null before construction.
3. Empty snapshot paths contradicted the JSON schema. Constructor validation now
   rejects them; a regression covers that case.

All findings were fixed and the affected suites rerun. Postponed annotations on
older Python versions cannot recover an annotation-only enclosing alias that the
runtime did not retain; unresolved symbols fail explicitly with source location.

## Commands and results

From the worktree root, the main development interpreter is:

```sh
export LLK_PYTHON=/Users/skg7on/Workspace/Projects/DSLCompiler/.venv/bin/python
"$LLK_PYTHON" -m pytest test/python/test_frontend_binding.py test/python/test_frontend_schema.py -q
PYTHONDONTWRITEBYTECODE=1 "$LLK_PYTHON" -m pytest -q -p no:cacheprovider
"$LLK_PYTHON" -m pytest test/python/test_package_install.py -q -p no:cacheprovider
```

| Lane | Result |
| --- | --- |
| Python 3.14.8 focused schema/binding, pytest 9.1.1, jsonschema 4.26.0 | 55 passed, zero skips |
| Python 3.14.8 full Python suite, real Torch 2.14.1 | 76 passed, zero skips |
| Python 3.14.8 clean wheel installation/package suite | 8 passed, zero skips |
| Python 3.11.14 focused schema/binding, pytest 9.1.1, jsonschema 4.26.0 | 55 passed, zero skips |
| Python 3.12.12 focused schema/binding, pytest 9.1.1, jsonschema 4.26.0 | 55 passed, zero skips |

The 3.11/3.12 interpreters and test environments are isolated under this
worktree's `build/`. Their commands were respectively:

```sh
build/pf03-py311/bin/python -m pytest test/python/test_frontend_binding.py test/python/test_frontend_schema.py -q
build/pf03-py312/bin/python -m pytest test/python/test_frontend_binding.py test/python/test_frontend_schema.py -q
```

These compiler-independent reference lanes contain pytest/jsonschema, without
Torch. They do not requalify a new Torch pair. The main full suite preserves real
Torch registration coverage. Clean wheel tests import `llk.frontend` outside the
checkout with neither Torch nor jsonschema installed, confirming both remain
optional at runtime. `jsonschema` is a test-extra dependency only.

The suite checks canonical round trips, source/constant/launch/numerical/config
identity changes, provider relocation invariance, deep snapshots, signed-index
values above 2^53, overflow/non-finite/bool rejection, strict fields/versions,
duplicate JSON keys, memory effects, masks, nested loop scope/yield and cast order.
Ruff's import/undefined-name/unused checks and formatting checks pass for the new
frontend and its fixture/tests.

Documentation validation used the project interpreter:

```sh
"$LLK_PYTHON" -m unittest discover -s test/Docs -p 'test_*.py'
"$LLK_PYTHON" -m unittest discover -s docs/manual/source/_ext/tests -p 'test_*.py'
"$LLK_PYTHON" -m sphinx -b html -n -W --keep-going -d build/docs/doctrees docs/manual/source build/docs/html
```

Results: 13 documentation-checker regressions and one source-link regression
passed; the strict HTML build succeeded without warnings. Source license checks
and `git diff --check` also passed.

## Capability boundary

PF03 produces validated metadata and request bytes. It does not capture kernel
bodies, construct MLIR, prove affine whole-domain coverage, adapt or validate real
storage providers, compile kernels or invoke native code. Those remain PF04,
PF05 and subsequent milestones. The accepted opcodes/numerical policies and
backend option values describe the wire contract; they are not executable
capability claims. No numerical/native-execution evidence is claimed here.

No C++/MLIR/runtime code changed, so this milestone's local acceptance consists
of Python and documentation checks. Existing CI still builds/tests native code
and the manual; it does not yet run the Python frontend suite. Required frontend
execution CI integration remains PF20. The parent #203 stays open.
