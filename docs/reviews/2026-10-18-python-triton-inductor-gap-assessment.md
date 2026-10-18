# Python, Triton JIT and PyTorch/Inductor gap assessment

Reviewed on 2026-10-18 at `13cef9c` (`feat: search realized elementwise tile schedules through mapped tuning (B6) (#200)`). Scope: `python/`, the C++ Triton lowering passes, compiler entry points, mapped invocation, and their tests. This is a review; no implementation fixes are included.

The Python package is a prototype for AST capture and IR text generation. It does not currently provide an executable LLK JIT, an upstream Triton compiler backend, or an Inductor extension. The native MicroIR compilation/search/runtime infrastructure is substantially further along, but the Python entry points do not call it.

The approved [July Triton design](../superpowers/specs/2026-07-30-triton-frontend-design.md) supersedes the [M7 stub](../design/m7-python-frontend.md). It describes a separate Triton-compatible DSL feeding the historical CPU pipeline, with eventual reuse of the upstream frontend. It does not specify an Inductor extension. The [October framework design](../superpowers/specs/2026-10-12-generic-microir-dnn-compiler-framework-design.md) makes MicroIR the framework center and treats Python/model frontends as separate capability work. Therefore, missing Inductor integration is a gap against the integration requested in this review, rather than a regression against a delivered project capability.

| Boundary | Current implementation | Missing capability |
|---|---|---|
| `@llk.jit` | Captures source and caches one generated string | Valid typed IR, argument binding, specialization, grid launch, native compilation and invocation |
| `llk.language` | Public `arange`, `program_id`, annotation/dtype sentinels; AST recognizes a few more call names | Supported language contract, type/shape inference, dataflow, control flow and explicit rejection |
| Upstream `@triton.jit` | No backend/compiler/driver adapter in `python/` | Upstream frontend/TTIR ingestion, version compatibility, target selection, launcher/runtime integration |
| `torch.compile(backend="llk")` | Attempts Dynamo registration; backend returns `gm.forward` | Working registration and actual FX/ATen compilation |
| Extended Inductor | No Inductor imports, lowering registrations, scheduler or wrapper integration in the Python package | Defined integration boundary and LLK kernel calls inside an Inductor-managed graph |
| Triton-to-structured C++ | Generic-name rewrite of `tt.dot` to `linalg.matmul`, plus separate memory/grid/atomic passes | Real dialect integration, complete lowering, fusion recognition and verified execution |
| MicroIR backend | Native source adapters, mapping/search, selected/reference compilation, typed invocation | Python binding and frontend-to-source-adapter conversion |

Native support must not be understated: [the registry](../../lib/Conversion/LLKKernel/Registry.cpp) has Micro lowering for matmul/SwiGLU and a bounded `linalg.matmul` compatibility adapter; RoPE/attention are Linalg-only there. The newer [elementwise source adapter](../../lib/Conversion/Workload/ElementwiseSourceAdapter.cpp) handles a constrained rank-two strict-f32 `linalg.generic` add/multiply family and realizes tile choices. These are reusable integration seams, not general Triton or FX importers.

1. **[P1, reproduced API failure] Importing the PyTorch integration uses an incompatible registration call.**

   [Registration](../../python/llk/torch/_dynamo_backend.py), lines 47–50, calls `torch._dynamo.register_backend("llk", LLKBackend(), eager=False)`. The upstream [PyTorch v2.9.0 registry](https://github.com/pytorch/pytorch/blob/v2.9.0/torch/_dynamo/backends/registry.py) takes `compiler_fn`, `name`, and `tags`; the [current upstream source](https://github.com/pytorch/pytorch/blob/main/torch/_dynamo/backends/registry.py), fetched during this review, has the same signature. `eager` is unsupported, and the two positional arguments are reversed. Removing only `eager=False` still supplies a string where a callable is required.

   The review executed the exact upstream `register_backend` function extracted from that version's source, attached it to a minimal torch module, and imported `llk.torch`: `TypeError: register_backend() got an unexpected keyword argument 'eager'`. This was a source-level API probe, not a real PyTorch installation test. The import also invoked `torch._dynamo.reset()` before failing, unnecessarily clearing global compilation state. Registration should use explicit argument names, avoid resetting unrelated state, and have a defined repeated-registration policy.

2. **[P1, reproduced] Python emission is not parseable by this compiler.**

   [The builder](../../python/llk/language/_triton_ir.py), lines 30–37, emits custom-assembly `tt.func`/`tt.return`, untyped bare function arguments, and no SSA argument names. LLK tools do not register the Triton dialect. Even `llk-opt --allow-unregistered-dialect` rejects an empty builder function with `Dialect 'tt' not found for custom op 'tt.func'`. Allowing an unknown dialect permits generic quoted operation syntax; it does not provide a custom assembly parser.

   Further failures remain behind that first error: `load`/`dot` operands contain Python AST representations, symbolic constants appear literally in tensor dimensions, and builder `dot` uses `f32` rather than the shaped accumulator expected by [DotToMatmulPattern](../../lib/Conversion/TritonToLLK/TritonToStructured.cpp), lines 43–48. Repairing the function header alone cannot connect the frontend to the backend. Acceptance needs parser and verifier round trips against the actual consuming compiler, with typed SSA values throughout.

3. **[P1, reproduced semantic loss] The AST visitor drops values, masks and unsupported computations.**

   [The visitor](../../python/llk/language/_ast_compiler.py), lines 45–80, visits assignment values without binding the results to variable names. `load`/`store`/`dot` serialize AST nodes using `str(...)`; they do not resolve operands or recursively compile pointer expressions. Keyword `mask=`, `other=` and `acc=` are ignored. A normal masked copy probe emits no mask at all.

   Unknown calls return `None`, so `z = tl.exp(x); return z` generates the same empty body as `pass`, apart from the function name. Arithmetic, comparisons, indexing, broadcasting and loop-carried values have no SSA implementation. Default AST traversal walks control-flow syntax without creating `scf.if`/`scf.for`, so it cannot preserve branch or loop semantics. `_constexprs` records names but is never populated from invocation values or used to specialize IR. Unsupported syntax must produce a source-located error before any executable can be claimed.

4. **[P1, implementation blocker] There is no executable JIT or Python/native bridge.**

   [JitFunction](../../python/llk/runtime/_jit.py), lines 14–29, always raises `NotImplementedError("JIT execution not yet implemented")` on a call. `_compile` ignores invocation arguments. `kernel[(1,)](...)` raises `TypeError: 'JitFunction' object is not subscriptable`; no launch-grid protocol exists.

   [The FFI module](../../python/llk/runtime/_ffi.py) is a Python descriptor container plus a stride helper. It has no ctypes structure, native-library loading, C++ binding, compile handle, executable invocation or lifetime management. Neither this module nor the decorator calls `compileMappedKernel`, `MappedExecutable` or the C++ JIT. The native [mapped invocation](../../include/LLK/Runtime/MappedInvocation.h) and [ABI](../../include/LLK/Runtime/KernelAbi.h) already provide useful checked contracts, but they are not exposed to Python. A concrete acceptance test must change a caller-owned output buffer through a compiled native kernel and identify the backend that executed it.

5. **[P1, reproduced wrong addressing] Block-pointer lowering multiplies physical strides twice.**

   [BlockPtrLoadPattern](../../lib/Conversion/TritonToLLK/BlockPointerToVector.cpp), lines 94–104, and the store path at lines 208–216 pass block-pointer element strides directly to `memref.subview`. Subview strides are multipliers of the source memref's existing strides. The repository fixture has `memref<128x64xbf16>` and block-pointer strides `[64, 1]`; lowering produces `memref<32x64xbf16, strided<[4096, 1], offset: ?>>`.

   For zero offsets, logical row 1 should begin at element 64; the emitted view begins it at element 4096, and row 2 already starts outside the 8192-element allocation. The generated logical boundary mask does not repair this addressing. This is a semantic/addressing defect in the current lowering, established by inspecting emitted IR; the review did not execute the out-of-bounds program. Base storage and logical tile addressing need one consistent stride representation and numerical load/store coverage.

6. **[P1, verified disconnect] The compiler entry point cannot consume the tested Triton input or route it through the complete lowering stack.**

   [The compiler registry/parser](../../tools/llk-compile/llk-compile.cpp), lines 622–669, neither registers the Triton dialect nor enables unregistered dialects. Running `llk-compile --emit=mlir test/Conversion/TritonToLLK/triton_full_pipeline.mlir` fails during parsing. Consequently, its later `tt.*` detection cannot handle the fixtures used to demonstrate the passes.

   Even after parsing is addressed, [runCompilationPipeline](../../tools/llk-compile/llk-compile.cpp), lines 170–185, adds only structured lowering and grid lowering. It omits CPU compatibility verification, block-pointer, shared-memory and atomic passes. The mapped and `--emit=micro*` branches execute earlier, at lines 671–681, and do not run this Triton preprocessing either. There is no public complete Triton-to-MicroIR compilation path. An isolated acceptable `linalg.matmul` can use the existing compatibility adapter, but that does not import the surrounding pointer/control/dataflow program.

7. **[P2, reproduced invalid IR] Grid lowering lets values escape an `scf.forall` body.**

   [GridToForall](../../lib/Conversion/TritonToLLK/GridToForall.cpp), lines 177–184, rewires a function return to a cloned result inside `scf.forall`, without making it a loop result. With ordinary verification, the existing `grid_to_forall_return.mlir` fails with `operand #0 does not dominate this use`. [CTest](../../test/FileCheck.cmake), lines 485–490, disables verification for this case. The fixture itself documents the limitation.

   Ordinary Triton pointer-output kernels are void, so this is a bounded accepted-IR problem rather than proof that every grid kernel fails. The pass should reject result-returning forms it cannot represent, or explicitly model result propagation. Its launch model also infers dimensions/block sizes from index arguments rather than an explicit Python grid, leaving the frontend/backend launch contract unspecified.

8. **[P2, coverage gap] The CPU whitelist is broader than implemented lowering/runtime support.**

   [TritonCPUVerifier](../../lib/Conversion/TritonToLLK/TritonCPUVerifier.cpp), lines 26–45 and 84–99, validates operation names only. A generic `tt.make_range` passes all six Triton passes and remains `tt.make_range`. Other accepted names include `tt.advance`, `tt.await`, `tt.reduce`, `tt.broadcast` and `tt.trans`, without corresponding rewrites in this pass collection. Conversely, the builder emits `tt.func`/`tt.return`, which are absent from the whitelist. There is no final residual-Triton rejection contract.

   [AtomicLowering](../../lib/Conversion/TritonToLLK/AtomicToLLRT.cpp), lines 50–76, creates declarations/calls such as `llrt.atomic_add_f32`; no implementation or symbol registration for these names was found in `runtime/` or its headers. Its FileCheck test proves call emission, not executable atomics. [Shared-memory lowering](../../lib/Conversion/TritonToLLK/SharedMemToScratch.cpp) emits allocations/copies, without implementing the advertised await operation or demonstrating worker scratch ownership/release. These paths need explicit unsupported diagnostics until executable coverage exists.

9. **[P2, central design gap] Dynamo replacement, Inductor extension and upstream Triton backend are different integration contracts.**

   [LLKBackend.__call__](../../python/llk/torch/_dynamo_backend.py), lines 30–38, increments a counter and returns `gm.forward`. Even with registration repaired, correct numerical outputs would prove only ordinary FX execution. There is no FX/ATen matching, decomposition, partitioning, LLK/MicroIR emission, native compile request or backend execution evidence.

   [PyTorch's custom-backend contract](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_custom_backends.html) places this callable directly after Dynamo; registering a different backend does not extend Inductor. Its AOTAutograd integration is also a separate step for normalized ATen and compiled backward graphs. Current eager graph execution may preserve PyTorch autograd, but no LLK-compiled training path exists.

   The missing architectural decision is where LLK is invoked: an independent graph compiler, selected kernels/subgraphs inside Inductor, or a full Inductor target/codegen backend. [Inductor's documented device integration](https://docs.pytorch.org/docs/main/accelerator/autoload.html) has scheduling/wrapper registration; [PyTorch's Triton integration tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_user_defined_triton_kernel_tutorial.html) describes user-kernel capture and library composition. Neither path is implemented by this package. For the existing CPU ABI, introducing a new device backend is not automatically necessary.

   Likewise, replacing `triton` imports with `llk` imports is not an upstream Triton backend. Upstream [BaseBackend](https://github.com/triton-lang/triton/blob/main/python/triton/backends/compiler.py) defines target support, options, compiler stages and dialect loading; execution also needs runtime/launcher integration. No corresponding LLK adapter exists here. A true upstream integration must pin compatible Triton/LLVM versions; a separate DSL must publish its supported subset. These are missing contracts, not interchangeable names for the current decorator.

10. **[P2, design/runtime gap] Specialization and tensor semantics stop before the Python boundary.**

    [The decorator](../../python/llk/runtime/_jit.py) has one `_compiled` flag, not a specialization cache. Shapes, dtypes, strides, constexpr values, numerical mode, target/rule content and compiler/ABI identity never influence the generated artifact. Native execution identities and replay infrastructure exist; Python has no request object that carries those facts into it.

    The native ABI is explicitly rank-two compact row-major, zero-offset, with read-only inputs and disjoint caller-owned outputs. Arbitrary Triton pointer programs and PyTorch views/mutations do not satisfy that contract automatically. Missing frontend decisions include accepted dtype/rank/device/layout, view handling, output allocation/aliasing, lifetime ownership and behavior on unsupported operations. Symbolic PyTorch dimensions also need either checked specialization or a delivered dynamic ABI; the [dynamic-shape design](../superpowers/specs/2026-10-17-dynamic-shape-llk-microir-design.md) remains a proposal. These limitations should be checked at the frontend boundary.

11. **[P2, validation gap] Passing tests do not demonstrate an integrated frontend.**

    All six `test/python` tests passed. They assert wrapper properties or substrings such as `tt.load`; none parse the emitted IR, exercise masks/constexpr specialization, invoke native code or import the PyTorch backend. The test named `test_jit_call_emits_triton_ir` calls `.ttir()`, not the kernel.

    Thirteen targeted CTest checks passed using the existing local build. `TritonStagedExecution.SwiGLUCorrectness` was disabled. `TritonFullPipeline` contains separate illustrative functions for individual stages, not one end-to-end Python kernel. The grid-return case disables verification. The root CMake/CI configuration does not run `test/python` or a real Torch/Triton compatibility matrix; no Python packaging metadata or supported dependency pins were found. CI passing cannot currently establish either integration requested here.

The dependency order implied by these findings is: define the intended integration boundary and supported tensor/launch contract; repair registration and typed, semantics-preserving frontend emission; expose a checked native compile/invoke API; route supported source adapters into selected MicroIR compilation; then prove real Inductor or upstream Triton composition. Broad operator coverage and tuning should follow a small end-to-end kernel that passes parser, verifier, numerical, specialization and backend-identity checks. Existing MicroIR and source-adapter infrastructure can be reused for that work.

Validation was deliberately targeted: no implementation was changed and no full rebuild or full test-suite run was performed. The available Python environment had neither Torch nor Triton installed. A temporary pytest environment ran the six Python tests; the PyTorch registration failure was tested against extracted upstream source. C++ observations used the existing `build/bin` tools and source inspection. None of these checks establishes hardware execution of a Python-generated kernel.

Representative reproduction commands, from the repository root:

```sh
PYTHONPATH=python python -m pytest -q -p no:cacheprovider test/python
build/bin/llk-opt --allow-unregistered-dialect --triton-block-ptr-to-vector \
  test/Conversion/TritonToLLK/block_ptr_to_vector.mlir
build/bin/llk-opt --allow-unregistered-dialect --triton-grid-to-forall \
  test/Conversion/TritonToLLK/grid_to_forall_return.mlir
build/bin/llk-compile --emit=mlir \
  test/Conversion/TritonToLLK/triton_full_pipeline.mlir
```

The first command requires pytest and is expected to pass; the second exposes the incorrect `[4096, 1]` stride; the third and fourth are expected to fail as described above.
