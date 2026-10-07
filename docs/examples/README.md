# Tutorial inputs

These files are small, checked-in inputs for the [developer tutorials](../README.md). Run commands from the repository root and keep generated artifacts in `build/tutorial/`.

| Input | Purpose | Observable result |
|---|---|---|
| [matmul.mlir](matmul.mlir) | Static BF16 matrix multiplication with FP32 accumulation | Concrete MicroIR with copies, MMA, conversion, and output write-back |
| [swiglu.mlir](swiglu.mlir) | Two BF16 projections with a SiLU/multiply epilogue | Two MMA arms and vector epilogue operations |
| [gemm-tile.micro.mlir](gemm-tile.micro.mlir) | One explicit tile MMA, independent of the source exporter | Cost accounting for 16,384 FLOPs and explicit transfer traffic |
| [swiglu.search.mlir](swiglu.search.mlir) | A deliberately small, one-candidate search space | A predicted schedule YAML record from `llk-tune` |

The first two are semantic source programs, not tensor data files. Their destination operands express the LLK interface. The current Micro exporter declares the computation's external input operands and creates its result internally; it does not preserve an arbitrary destination accumulator's initial contents. The tile example is an analysis fixture with `tensor.empty`, not a numerical invocation example. The search example is for the current synthetic candidate binder, not a mapping-driven tuner.

The tutorial commands are exercised by [run_tutorials.py](../tutorials/run_tutorials.py), which checks generated IR, plan replay, prediction reports, and schedule output. It does not claim numerical execution or selected AVX2 instruction verification.
