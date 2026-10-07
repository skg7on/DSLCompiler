# Tutorial 4: explore schedule candidates

**Goal:** run a small Micro search space, inspect persisted predictions, and understand how to expand the search. **Prerequisite:** [getting started](../getting-started.md); [performance basics](03-performance.md) help interpret the output.

## 1. Understand the input

A concrete `micro.kernel` describes one execution program. A `micro.search_space` describes decisions that may produce multiple programs. Open [swiglu.search.mlir](../examples/swiglu.search.mlir):

```mlir
micro.param "BM" {kind = "integer", choices = [8 : i64]}
micro.param "tile_layout" {kind = "layout", choices = ["row_major"]}
micro.constraint "sram_capacity" {params = ["BM", "BN", "BK"]}
micro.objective {direction = "minimize", metric = "latency_cycles",
                 secondary = ["matrix_utilization", "dram_bytes"]}
```

Integer choices control tiling, vector width, threads, and pipeline depth. Symbolic choices describe layout, memory path, owner mapping, fragment shape, and tail policy. Constraints reject combinations the current legality model cannot accept. The objective determines ranking.

This example deliberately offers one candidate, so you can learn the file format without a large search. It describes `M=8,N=64,K=64`. Pass dimensions explicitly: the CLI dimensions are inputs to candidate construction and are not reliably inferred from a symbol name.

## 2. Run the current Micro tuner

```bash
mkdir -p build/tutorial
build/llk-tune --input=docs/examples/swiglu.search.mlir \
  --machine=machines/x86-avx2-v2.yaml -M=8 -N=64 -K=64 \
  --search=grid --max-candidates=1 --top-k=1 --perf-level=1 \
  --output=build/tutorial/swiglu.schedule.yaml
```

The current spelling is `--input`, not `--search-space`. It writes schedule YAML; legacy `-o` selects a different output path in the legacy tuner. Expected output includes the workload and machine identity, hierarchy/layout/memory/owner choices, predicted metrics, and `measurement.measured: false`.

The current session builds a **synthetic concrete kernel** for a candidate and ranks its modeled costs. It does not map the original source through the selected LLKMap target, compile that selected plan, invoke it, or save actual latency measurements. Some emitted objective metrics are incomplete; read the [tuner manual](../tools/tuning.md) before optimizing a metric beyond predicted cycles. #129 supplies the plan for the missing mapping-driven path.

## 3. Export a search space from semantic source

```bash
build/llk-compile --emit=micro-search docs/examples/swiglu.mlir \
  > build/tutorial/swiglu.exported.search.mlir
build/llk-opt build/tutorial/swiglu.exported.search.mlir \
  > build/tutorial/swiglu.exported.search.roundtrip.mlir
```

Inspect the parameters exported for the source and schedule database. The adjacent LLK source is retained, but the current CLI tuning flow uses the search-space description and explicit dimensions rather than compiling that adjacent source. A later mapping-driven implementation must preserve source semantics and provenance; this tutorial does not claim that integration already exists.

## 4. Expand the choices

Copy the small example before editing it:

```bash
cp docs/examples/swiglu.search.mlir build/tutorial/swiglu.experiment.search.mlir
```

In the copy, change `num_threads` choices from `[8 : i64]` to `[1 : i64, 8 : i64]`. Keep workload dimensions and all other choices fixed. Run both candidates:

```bash
build/llk-tune --input=build/tutorial/swiglu.experiment.search.mlir \
  --machine=machines/x86-avx2-v2.yaml -M=8 -N=64 -K=64 \
  --search=grid --max-candidates=2 --top-k=2 \
  --output=build/tutorial/swiglu.experiment.schedule.yaml
```

Compare the hierarchy/owner choices, predicted cycles, and retained records. Thread choices may tie or fail legality depending on the model; a smaller time from this prediction is not proof of faster measured execution. Next, change one tile dimension and observe capacity/legality effects before trying a large Cartesian product.

For bounded random sampling with a reproducible seed:

```bash
build/llk-tune --input=build/tutorial/swiglu.experiment.search.mlir \
  --machine=machines/x86-avx2-v2.yaml -M=8 -N=64 -K=64 \
  --search=random --seed=17 --max-candidates=2 --top-k=2 \
  --output=build/tutorial/swiglu.random.schedule.yaml
```

Use a nonzero `--max-candidates` to bound a random experiment; omitting it or setting it to zero requests the whole finite domain. The seed is meaningful only with the same input, generator, dimensions, compiler, and machine. `--top-k` limits retained records, not the amount of generation work.

## 5. Keep tuning and scheduling formats separate

Micro mode produces schedule YAML records. The legacy shape-grid path writes the JSON schedule database consumed by existing schedule selection. A Micro YAML output is not a drop-in `schedule_db.json`. Binding selected search decisions into mapping, compiling verified executables, and persisting complete measured provenance are separate integration concerns.

The [tuner manual](../tools/tuning.md) covers the legacy grid, supported options/defaults, workload/dtype controls, objectives, and persistence limits. For a meaningful contribution, add a regression that demonstrates one missing contract rather than a new heuristic that obscures it; see the [contributor guide](../contributing.md).

## Verify the tutorial set

```bash
python3 docs/tutorials/run_tutorials.py --build-dir build
```

The runner uses the same checked-in examples, writes its artifacts to `build/tutorial-check/`, and verifies parsing/export, mapped plan replay, work/traffic fields, and predicted schedule records. It does not invoke tensor computations or establish selected ISA fidelity.
