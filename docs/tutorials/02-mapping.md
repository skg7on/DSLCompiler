# Tutorial 2: map a kernel and replay its plan

**Goal:** turn a concrete Micro kernel into a selected, placed, connected plan, inspect the report, and reproduce the same bound IR. **Prerequisite:** [Tutorial 1](01-first-kernel.md). All commands run from the repository root in the same shell.

## 1. Freeze the input and configuration

```bash
mkdir -p build/tutorial
build/llk-opt --llk-to-micro="schedule-db=build/tutorial/no-schedule.json" \
  docs/examples/matmul.mlir > build/tutorial/matmul.fallback.micro.mlir

EMITTERS="avx2_vector_add,avx2_vector_convert,avx2_vector_silu,avx2_vector_mul,avx2_mma,avx2_reduce,avx2_copy,avx2_tile_copy,avx2_tile_store,avx2_fused_convert_silu_mul"
TARGET_OPTIONS="target=x86-avx2 machine=machines/x86-avx2-v2.yaml layouts=mapping/x86-avx2/layouts.llkmap rules=mapping/x86-avx2/rules.llkmap emitters=$EMITTERS"
MAP_OPTIONS="$TARGET_OPTIONS mode=exact top-k=8"
```

These configuration files serve different purposes:

- [Machine profile](../../machines/x86-avx2-v2.yaml): physical topology, capability, capacity, and cost parameters.
- [Layouts](../../mapping/x86-avx2/layouts.llkmap): declarative layout definitions and constraints.
- [Rules](../../mapping/x86-avx2/rules.llkmap): operation patterns, target bundles, placement requirements, and emitter names.
- `emitters`: implementations this caller permits the mapper to select. This is an availability list; it does not prove each emitter produces the selected ISA faithfully.

`exact` searches the modeled covering alternatives with explicit budgets; it is not a claim that every physical resource constraint or global optimality condition is already implemented. Start with small fixtures and inspect truncation/diagnostics. See [current limits](../features.md).

## 2. Search, bind, and report

```bash
build/llk-opt "--micro-map=$MAP_OPTIONS report=build/tutorial/matmul.plan.json" \
  build/tutorial/matmul.fallback.micro.mlir > build/tutorial/matmul.mapped.mlir
build/llk-opt "--micro-verify-mapping=$TARGET_OPTIONS" build/tutorial/matmul.mapped.mlir \
  > build/tutorial/matmul.verified.mlir
```

The mapped kernel gains `micro.plan`, placement metadata on covered operations, and route/movement metadata where needed. The report includes hashes of the input and target configuration, search options, diagnostic counts, retained plans, and a selected plan ID. The verifier checks the implemented metadata contract; it does not close the complete physical-feasibility gaps tracked in #129.

Inspect the report using Python, without an extra JSON utility:

```bash
python3 -m json.tool build/tutorial/matmul.plan.json
```

Questions to answer before trusting the selection:

1. Did the report retain a complete selected plan?
2. Which rules, executors, layouts, and routes were selected?
3. Is the search truncated? Were candidates rejected, and why?
4. Does storage analysis report a complete footprint or an explicit skipped/incomplete case?
5. Does the plan describe the target behavior you intended to test?

## 3. Replay by plan ID

Extract the reported ID and rerun the same search settings through the binding pass:

```bash
PLAN_ID=$(python3 -c 'import json; print(json.load(open("build/tutorial/matmul.plan.json"))["selectedPlanId"])')
build/llk-opt "--micro-bind-plan=plan-id=$PLAN_ID $MAP_OPTIONS" \
  build/tutorial/matmul.fallback.micro.mlir > build/tutorial/matmul.replayed.mlir
diff -u build/tutorial/matmul.mapped.mlir build/tutorial/matmul.replayed.mlir
```

Expected: `diff` prints nothing and exits successfully. This pass **searches again** using the same input and options, then binds the matching ID. It is different from the compiler's frozen-report replay via `--plan-report`. The latter validates provenance and consumes a recorded selection; see the [compiler manual](../tools/compiler.md).

Record the original input file, machine/layout/rule files, emitter list, compiler revision, mode, beam width if used, and top-K/budgets. A plan ID alone is not enough to reproduce a different input or configuration. Keep both sides of a replay comparison on the same path spelling and configuration.

## 4. Inspect a later compilation stage

```bash
build/llk-compile --mapping-target=x86-avx2 --mapping-root=. \
  --mapping-mode=exact --mapping-stop=lowered \
  docs/examples/matmul.mlir > build/tutorial/matmul.lowered.txt
```

The output starts with mapping/realization status followed by lowered MLIR. This driver independently exports and maps the source using its normal schedule settings; it is not a replay of the fallback export above. Review the status as well as the IR. Selected emitters can delegate tensor realization to a portable reference bridge; successful lowering does not establish selected AVX2 width or instruction fidelity.

To request JIT compilation through the mapped path:

```bash
build/llk-compile --mapping-target=x86-avx2 --mapping-root=. \
  --mapping-mode=exact docs/examples/matmul.mlir
```

Expected: a compilation-success status if this host/build supports the path. The CLI does not accept tensor data or invoke the resulting function. Numerical invocation uses `MappedExecutable` in the C++ runtime and is demonstrated in the [mapped acceptance tests](../../test/Execution/mapped_acceptance.cpp). Repeated invocation ownership, richer ABI validation, and selected-target realization remain work in #129.

## 5. Optional second-target experiment

The [generic accelerator package](../../mapping/generic-ai-accel/) is for mapping/model exploration, not an accelerator runtime. Use the hand-authored [copy/add fixture](../../test/Conversion/MicroMapping/micro_map.mlir):

```bash
build/llk-opt --micro-map="target=generic-ai-accel machine=machines/generic-ai-accel-v2.yaml layouts=mapping/generic-ai-accel/layouts.llkmap rules=mapping/generic-ai-accel/rules.llkmap emitters=accel_vector_add,accel_mxu,accel_copy mode=deterministic report=build/tutorial/accelerator.plan.json" \
  test/Conversion/MicroMapping/micro_map.mlir > build/tutorial/accelerator.mapped.mlir
```

If emitter names or supported rules change, consult the package and [optimizer manual](../tools/optimizer.md). Compare selected placements and routes with AVX2 rather than assuming the same cost model has been calibrated for both machines.

**What you established:** mapping and report generation, metadata verification, and reproducible plan-ID binding. Continue to [Tutorial 3](03-performance.md) to inspect costs and capacity diagnostics independently.
