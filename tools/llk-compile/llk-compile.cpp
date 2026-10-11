//===- llk-compile.cpp - LLK kernel compiler driver -----------------------===//
//
// CLI tool that parses an MLIR module containing LLK dialect ops,
// runs the full progressive lowering pipeline (LLK→Linalg→...→LLVM IR),
// JIT-compiles the result, and reports success.
//
// Usage: llk-compile <input.mlir> [--M=<dim>] [--N=<dim>] [--K=<dim>]
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/LLKToLinalg.h"
#include "LLK/Conversion/LLKToMicro/LLKToMicro.h"
#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Conversion/TritonToLLK/GridToForall.h"
#include "LLK/Conversion/TritonToLLK/TritonToStructured.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/PlanReport.h"
#include "LLK/Mapping/WorkloadGraph.h"
#include "LLK/Runtime/JitCache.h"
#include "LLK/Target/MappingTargets.h"
#include "LLK/Transforms/ForallToLLRT.h"
#include "LLK/Transforms/FuseDoubleContraction.h"
#include "LLK/Transforms/LinearizeForall.h"
#include "LLK/Transforms/PackWeights.h"
#include "LLK/Transforms/ScheduleSelection.h"
#include "LLK/Transforms/ScratchAnalysis.h"
#include "LLK/Transforms/SerialParallelDispatch.h"
#include "LLK/Transforms/ShapeSpecialization.h"
#include "LLK/Transforms/TileAndVectorize.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/TilingInterfaceImpl.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/IR/ValueBoundsOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/SubsetOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/Config/llvm-config.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"

#if LLVM_VERSION_MAJOR >= 21
using BufOpts = mlir::bufferization::OneShotBufferizePassOptions;
#else
using BufOpts = mlir::bufferization::OneShotBufferizationOptions;
#endif

namespace cl = llvm::cl;

static cl::opt<std::string> inputFile(cl::Positional, cl::desc("<input .mlir>"),
                                      cl::Required);

static cl::opt<int64_t> optM("M", cl::desc("M dimension (rows)"), cl::init(0));
static cl::opt<int64_t> optN("N", cl::desc("N dimension (cols)"), cl::init(0));
static cl::opt<int64_t> optK("K", cl::desc("K dimension (hidden)"),
                             cl::init(0));

static cl::opt<std::string> emitMode(
    "emit",
    cl::desc("What to emit: llvm (JIT-compile, the default), mlir (print the "
             "fully lowered module), micro (print concrete Micro-IR), or "
             "micro-search (print the tile search space)"),
    cl::init("llvm"));

// ---------------------------------------------------------------------------
// Mapping (issue #67 stage C5)
// ---------------------------------------------------------------------------
//
// Setting `--mapping-target` takes a different contract from the legacy
// pipeline: the user names the target, the schedule comes from a search over
// the kernel's workload graph, and the code that runs is the target's. Leaving
// it empty leaves the legacy path byte-for-byte what it was.

static cl::opt<std::string> mappingTargetName(
    "mapping-target",
    cl::desc("Map and compile through this target package (for example "
             "x86-avx2). Empty keeps the legacy pipeline."),
    cl::init(""));

static cl::opt<std::string> mappingRoot(
    "mapping-root",
    cl::desc("Configuration root the target package loads its machine profile "
             "and its layout and rule files from"),
    cl::init("."));

static cl::opt<std::string> mappingMachine(
    "machine",
    cl::desc(
        "Machine profile override shared by mapping search and compilation"),
    cl::init(""));

static cl::opt<std::string>
    mappingMode("mapping-mode",
                cl::desc("Search mode: deterministic, beam, or exact"),
                cl::init("beam"));

static cl::opt<std::string> mappingStop(
    "mapping-stop",
    cl::desc("Where the mapping path stops: mapped-micro (the bound plan), "
             "target-lowered (after the target's emitters), lowered (Linalg "
             "and loops), or executable (the default)"),
    cl::init("executable"));

static cl::opt<std::string>
    mappingBackend("mapping-backend",
                   cl::desc("Execution backend: reference or selected-target"),
                   cl::init("reference"));

static cl::opt<std::string> mappingEntry(
    "mapping-entry",
    cl::desc("Kernel symbol to compile. Empty uses the module's single "
             "micro.kernel."),
    cl::init(""));

static cl::opt<std::string> mappingPlanReport(
    "plan-report",
    cl::desc("Replay the frozen selection in this plan report instead of "
             "searching. The report's graph and target hashes must match."),
    cl::init(""));

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

static mlir::LogicalResult runCompilationPipeline(mlir::ModuleOp module) {
  mlir::PassManager pm(module->getContext());

  // Triton IR path (M8b): staged lowering before the existing pipeline.
  // Detect Triton IR by checking for tt.* ops (unregistered dialect, so the
  // check is on the op name prefix rather than a dialect namespace).
  bool hasTritonOps = false;
  module.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef().starts_with("tt."))
      hasTritonOps = true;
  });

  if (hasTritonOps) {
    // Stage 0a: Triton compute ops (tt.dot) → Linalg structured ops.
    pm.addPass(mlir::llk::createTritonToStructuredPass());
    // Stage 0b: Clean up the lowered IR before grid dispatch lowering.
    pm.addPass(mlir::createCanonicalizerPass(mlir::GreedyRewriteConfig()));
    // Stage 0c: Triton grid dispatch (tt.get_program_id) → scf.forall.
    pm.addPass(mlir::llk::createTritonGridToForallPass());
  }

  // Stage 1: Lower LLK dialect ops to Linalg + Arith + Math.
  pm.addPass(mlir::llk::createLLKToLinalgPass());

  // Stage 2: Canonicalize to clean up the lowered IR.
  pm.addPass(mlir::createCanonicalizerPass(mlir::GreedyRewriteConfig()));

  // Stage 3: Shape specialization — classify M bucket and emit dispatch guards.
  pm.addPass(mlir::llk::createShapeSpecializationPass());

  // Stage 4: Select schedule from schedule_db.json based on (op, M, N, K, ISA).
  pm.addPass(mlir::llk::createScheduleSelectionPass());

  // Stage 5: Fuse double contraction (SwiGLU-specific; no-op for
  // RoPE/Attention).
  pm.addPass(mlir::llk::createFuseDoubleContractionPass());

  // Stage 6: Canonicalize after fusion.
  pm.addPass(mlir::createCanonicalizerPass(mlir::GreedyRewriteConfig()));

  // Stage 7: Annotate weights with packing layout.
  pm.addPass(mlir::llk::createPackWeightsPass());

  // Stage 8: Tile and vectorize using Transform dialect schedule.
  pm.addPass(mlir::llk::createTileAndVectorizePass());

  // Stage 9: Canonicalize after tiling/vectorization.
  pm.addPass(mlir::createCanonicalizerPass(mlir::GreedyRewriteConfig()));

  // Stage 10: Linearize 2D scf.forall into 1D (Parallel Decompose).
  pm.addPass(mlir::llk::createLinearizeForallPass());

  // Stage 11: Select serial vs parallel dispatch based on shape thresholds.
  pm.addPass(mlir::llk::createSerialParallelDispatchPass());

  // Stage 12: One-Shot Bufferize (tensor → memref).
  BufOpts bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));

  // Stage 13: Audit scratch allocations post-bufferization.
  pm.addPass(mlir::llk::createScratchAnalysisPass());

  // Stage 14: Lower residual linalg ops on memref to scf.for loops.
  // This handles linalg.generic and linalg.fill ops that survive past
  // TileAndVectorize (which only tiles matmul/contraction ops).
  // Uses the built-in MLIR pass "convert-linalg-to-loops".
  {
    const auto *passInfo = mlir::PassInfo::lookup("convert-linalg-to-loops");
    if (passInfo) {
      if (mlir::failed(
              passInfo->addToPipeline(pm, {}, [](const llvm::Twine &msg) {
                llvm::errs() << "convert-linalg-to-loops: " << msg << "\n";
                return mlir::failure();
              }))) {
        llvm::errs()
            << "ERROR: convert-linalg-to-loops failed to add to pipeline\n";
        return mlir::failure();
      }
    }
  }

  // Stage 15: Lower scf.forall to llrt.parallel_for (ThreadPool runtime).
  pm.addPass(mlir::llk::createForallToLLRTPass());

  // Stage 16: Convert Vector → LLVM (SIMD intrinsics).
  pm.addPass(mlir::createConvertVectorToLLVMPass());

  // Stage 17: Convert SCF → CF.
#if LLVM_VERSION_MAJOR >= 21
  pm.addPass(mlir::createSCFToControlFlowPass());
#else
  pm.addPass(mlir::createConvertSCFToCFPass());
#endif

  // Stage 18: Convert Arith → LLVM.
  pm.addPass(mlir::createArithToLLVMConversionPass());

  // Stage 19: Convert Math → LLVM.
  pm.addPass(mlir::createConvertMathToLLVMPass());

  // Stage 19b: Lower `ub` to LLVM. Vectorization pads out-of-bounds reads with
  // `ub.poison`, which has no LLVM IR translation in this LLVM release, so it
  // must be converted away before translation rather than handed to it.
  pm.addPass(mlir::createUBToLLVMConversionPass());

  // Stage 20: Convert Func → LLVM.
  pm.addPass(mlir::createConvertFuncToLLVMPass());

  // Stage 21: Convert MemRef → LLVM.
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());

  // Stage 22: Reconcile unrealized casts from partial conversions.
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  // NOTE: JitCache::lookupOrCompile runs its own lowering passes
  // inside the cache miss path; these are no-ops when the module
  // is already fully lowered to LLVM dialect.

  return pm.run(module);
}

/// Runs the Micro-IR export on its own.
///
/// The export lowers directly from the LLK root operations, so it does not run
/// the Linalg pipeline: it needs the semantic structure and the selected
/// schedule, not the tiled and bufferized form the JIT path produces.
static mlir::LogicalResult runMicroExport(mlir::ModuleOp module) {
  mlir::PassManager pm(module->getContext());
  pm.addPass(mlir::llk::createLLKToMicroPass());
  return pm.run(module);
}

/// Runs the Micro-IR search-space export on its own, like the concrete export:
/// it lowers straight from the LLK root operations and never reaches the JIT.
static mlir::LogicalResult runMicroSearchSpaceExport(mlir::ModuleOp module) {
  mlir::PassManager pm(module->getContext());
  pm.addPass(mlir::llk::createLLKToMicroSearchSpacePass());
  return pm.run(module);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Mapping path (issue #67 stage C5)
// ---------------------------------------------------------------------------

namespace {

std::string typeToString(mlir::Type type) {
  std::string buffer;
  llvm::raw_string_ostream stream(buffer);
  type.print(stream);
  return buffer;
}

std::optional<mlir::llk::mapping::SearchMode>
parseSearchMode(llvm::StringRef name) {
  if (name == "deterministic")
    return mlir::llk::mapping::SearchMode::Deterministic;
  if (name == "beam")
    return mlir::llk::mapping::SearchMode::Beam;
  if (name == "exact")
    return mlir::llk::mapping::SearchMode::Exact;
  return std::nullopt;
}

std::optional<llk::MappedStop> parseStop(llvm::StringRef name) {
  if (name == "mapped-micro")
    return llk::MappedStop::MappedMicro;
  if (name == "target-lowered")
    return llk::MappedStop::TargetLowered;
  if (name == "lowered")
    return llk::MappedStop::Lowered;
  if (name == "executable")
    return llk::MappedStop::Executable;
  return std::nullopt;
}

std::optional<llk::MappedBackend> parseBackend(llvm::StringRef name) {
  if (name == "reference")
    return llk::MappedBackend::Reference;
  if (name == "selected-target")
    return llk::MappedBackend::SelectedTarget;
  return std::nullopt;
}

} // namespace

/// The mapping path: get a kernel, get a plan for it, compile that plan.
///
/// A plan comes from a search or from a frozen report, and this does not search
/// for a *schedule* either -- turning an LLK/Linalg program into a concrete
/// Micro kernel is the export's job, and it runs first when the input is not
/// already Micro.
static int runMappedCompilation(mlir::ModuleOp module,
                                mlir::MLIRContext &context) {
  mlir::llk::target::registerAllMappingTargets();

  std::optional<llk::MappedStop> stop = parseStop(mappingStop);
  if (!stop) {
    llvm::errs() << "Unsupported --mapping-stop=" << mappingStop
                 << "; expected mapped-micro, target-lowered, lowered, or "
                    "executable\n";
    return 1;
  }
  std::optional<llk::MappedBackend> backend = parseBackend(mappingBackend);
  if (!backend) {
    llvm::errs() << "Unsupported --mapping-backend=" << mappingBackend
                 << "; expected reference or selected-target\n";
    return 1;
  }

  llvm::Expected<std::unique_ptr<mlir::llk::mapping::MappingTarget>> target =
      mlir::llk::mapping::createRegisteredMappingTarget(
          mappingTargetName, mappingRoot, mappingMachine);
  if (!target) {
    llvm::errs() << llvm::toString(target.takeError()) << "\n";
    return 1;
  }

  auto findKernel = [&]() -> mlir::Operation * {
    mlir::Operation *kernel = nullptr;
    module->walk([&](mlir::Operation *op) {
      if (!kernel && op->getName().getStringRef() == "micro.kernel")
        kernel = op;
    });
    return kernel;
  };

  if (!findKernel()) {
    if (mlir::failed(runMicroExport(module))) {
      llvm::errs() << "Mapping needs a micro.kernel, and exporting one from "
                      "this input failed\n";
      return 1;
    }
    // The export leaves the semantic source function beside the kernel it
    // produced: it is the reference the export was made from, and the legacy
    // pipeline lowers it. This path compiles the *kernel*, so keeping the
    // source would compile the program twice -- and its high-level ops have no
    // lowering here.
    llvm::SmallVector<mlir::Operation *> sourceFunctions;
    module->walk([&](mlir::Operation *op) {
      if (!mlir::isa<mlir::func::FuncOp>(op))
        return;
      bool semantic = false;
      op->walk([&](mlir::Operation *inner) {
        if (inner->getName().getStringRef().starts_with("llk."))
          semantic = true;
      });
      if (semantic)
        sourceFunctions.push_back(op);
    });
    for (mlir::Operation *op : sourceFunctions)
      op->erase();
  }
  mlir::Operation *kernel = findKernel();
  if (!kernel) {
    llvm::errs() << "No micro.kernel to map\n";
    return 1;
  }

  llvm::Expected<mlir::llk::mapping::WorkloadGraph> graph =
      mlir::llk::mapping::extractWorkloadGraph(kernel);
  if (!graph) {
    llvm::errs() << llvm::toString(graph.takeError()) << "\n";
    return 1;
  }

  std::optional<mlir::llk::mapping::CoveringPlan> selected;
  if (!mappingPlanReport.empty()) {
    // Replaying a frozen selection: the report is data, not code, so the hash
    // checks below are what tie it to *this* graph and *this* target.
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
        llvm::MemoryBuffer::getFile(mappingPlanReport);
    if (!buffer) {
      llvm::errs() << "Cannot read plan report " << mappingPlanReport << "\n";
      return 1;
    }
    llvm::Expected<mlir::llk::mapping::CoveringPlan> replayed =
        mlir::llk::mapping::readPlanReport(buffer.get()->getBuffer(), **target,
                                           *graph);
    if (!replayed) {
      llvm::errs() << llvm::toString(replayed.takeError()) << "\n";
      return 1;
    }
    selected = std::move(*replayed);
  } else {
    std::optional<mlir::llk::mapping::SearchMode> mode =
        parseSearchMode(mappingMode);
    if (!mode) {
      llvm::errs() << "Unsupported --mapping-mode=" << mappingMode
                   << "; expected deterministic, beam, or exact\n";
      return 1;
    }
    // The layout context comes from the kernel's own operands: solving a layout
    // means reasoning about the element type the tile actually holds.
    mlir::llk::mapping::LayoutContext layoutContext;
    layoutContext.rank = 2;
    layoutContext.elementType = "f32";
    for (mlir::BlockArgument argument : kernel->getRegion(0).getArguments()) {
      if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(argument.getType())) {
        layoutContext.rank = shaped.getRank();
        layoutContext.elementType = typeToString(shaped.getElementType());
        break;
      }
    }

    mlir::llk::mapping::MappingSearchOptions options;
    options.mode = *mode;
    mlir::llk::mapping::CoveringSearch search(*graph, **target, context,
                                              layoutContext, options);
    llvm::Expected<mlir::llk::mapping::MappingSearchResult> result =
        search.search();
    if (!result) {
      llvm::errs() << llvm::toString(result.takeError()) << "\n";
      return 1;
    }
    if (result->plans.empty()) {
      llvm::errs()
          << "The search found no complete plan for kernel '"
          << kernel->getAttrOfType<mlir::StringAttr>("sym_name").getValue()
          << "'\n";
      return 1;
    }
    selected = result->plans.front();
  }

  llk::MappedCompileOptions options;
  options.stop = *stop;
  options.backend = *backend;
  // The kernel this run mapped is the one to compile: `--mapping-entry` names a
  // different one deliberately, and leaving it empty does not have to be an
  // error here because the kernel symbol is right there.
  options.entrySymbol = mappingEntry;
  if (options.entrySymbol.empty())
    options.entrySymbol =
        kernel->getAttrOfType<mlir::StringAttr>("sym_name").getValue().str();
  llvm::Expected<llk::MappedCompilation> compiled =
      llk::compileMappedKernel(module, **target, *selected, options);
  if (!compiled) {
    llvm::errs() << llvm::toString(compiled.takeError()) << "\n";
    return 1;
  }

  // What ran is reported next to what was selected: a run whose operations were
  // all carried by the reference bridge has not exercised the target's code
  // generation, and saying so is the difference between the two claims.
  llvm::outs() << "mapping: target=" << mappingTargetName
               << " backend=" << mappingBackend
               << " stop=" << mappingStop.getValue() << " executable="
               << (compiled->stopped == llk::MappedStop::Executable
                       ? "ready"
                       : "not-built")
               << " invocation=not-run"
               << " target-identity="
               << (compiled->codegenRequirements ? mappingTargetName.getValue()
                                                 : "native-host")
               << " architecture="
               << (compiled->codegenRequirements
                       ? compiled->codegenRequirements->architecture
                       : "native")
               << " cpu="
               << (compiled->codegenRequirements
                       ? compiled->codegenRequirements->cpu
                       : "host")
               << " features=";
  if (compiled->codegenRequirements) {
    for (size_t i = 0;
         i < compiled->codegenRequirements->requiredFeatures.size(); ++i) {
      if (i)
        llvm::outs() << ',';
      llvm::outs() << compiled->codegenRequirements->requiredFeatures[i];
    }
  } else {
    llvm::outs() << "host";
  }
  llvm::outs() << " identity=" << compiled->executionIdentity;
  if (compiled->executable)
    llvm::outs() << " abi-hash=" << compiled->executable->abiHash();
  llvm::outs() << " selected-groups-verified="
               << compiled->selectedGroupsVerified
               << " target-lowered-ops=" << compiled->targetLowered
               << " reference-lowered-ops=" << compiled->referenceLowered
               << " backend-groups-realized=" << compiled->backendGroupsRealized
               << " reference-groups-lowered="
               << compiled->referenceGroupsLowered << "\n";

  if (compiled->stopped != llk::MappedStop::Executable) {
    compiled->module->print(llvm::outs());
    llvm::outs() << "\n";
    return 0;
  }

  llvm::outs() << "Compilation successful\n";
  return 0;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);

  // Register all built-in MLIR passes.
  mlir::registerAllPasses();

  // Register all LLK passes.
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createLLKToLinalgPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createShapeSpecializationPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createScheduleSelectionPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createFuseDoubleContractionPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createPackWeightsPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createTileAndVectorizePass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createLinearizeForallPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createSerialParallelDispatchPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createScratchAnalysisPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createForallToLLRTPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createLLKToMicroPass();
  });
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return mlir::llk::createLLKToMicroSearchSpacePass();
  });

  // Parse command line.
  cl::ParseCommandLineOptions(argc, argv, "LLK kernel compiler\n");

  const std::string emit = emitMode.getValue();
  if (emit != "llvm" && emit != "mlir" && emit != "micro" &&
      emit != "micro-search") {
    llvm::errs() << "Unsupported --emit=" << emit
                 << "; expected llvm, mlir, micro, or micro-search\n";
    return 1;
  }

  // Build dialect registry with all required dialects.
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::scf::SCFDialect,
                  mlir::arith::ArithDialect, mlir::math::MathDialect,
                  mlir::memref::MemRefDialect, mlir::LLVM::LLVMDialect,
                  mlir::ub::UBDialect>();

  // Register TilingInterface external models so Linalg ops can be tiled.
  mlir::linalg::registerTilingInterfaceExternalModels(registry);

  // Register BufferizableOpInterface external models for One-Shot Bufferize.
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);

  // Register SubsetOpInterface external models for bufferization analysis.
  mlir::vector::registerSubsetOpInterfaceExternalModels(registry);

  // Register ValueBoundsOpInterface external models for bufferization
  // analysis (needed when the IR contains scf.for, scf.forall,
  // arith, and linalg ops with dynamic shapes).
  mlir::arith::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::linalg::registerValueBoundsOpInterfaceExternalModels(registry);
  mlir::scf::registerValueBoundsOpInterfaceExternalModels(registry);

  // Register the LLVM IR translations the pipeline can actually reach:
  // builtin (modules and unreachable) and the LLVM dialect. Not the full
  // `registerAllToLLVMIRTranslations`, whose ARM/GPU/SPIRV/... entries this
  // tool does not link -- naming them would turn a complete-looking
  // registration into a link error.
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);

  mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  // Parse the input MLIR file.
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(inputFile, &ctx);
  if (!module) {
    llvm::errs() << "Failed to parse " << inputFile << "\n";
    return 1;
  }

  // A named mapping target takes over before anything else: it has its own
  // stopping points and its own contract, and leaving the option empty keeps
  // the legacy path exactly as it was.
  if (!mappingTargetName.empty())
    return runMappedCompilation(*module, ctx);

  // Micro-IR export runs on its own and never reaches the JIT.
  if (emit == "micro" || emit == "micro-search") {
    mlir::LogicalResult result = emit == "micro"
                                     ? runMicroExport(*module)
                                     : runMicroSearchSpaceExport(*module);
    if (mlir::failed(result)) {
      llvm::errs() << "Micro-IR export failed\n";
      return 1;
    }
    module->print(llvm::outs());
    llvm::outs() << "\n";
    return 0;
  }

  // Run the compilation pipeline.
  if (mlir::failed(runCompilationPipeline(*module))) {
    llvm::errs() << "Compilation pipeline failed\n";
    return 1;
  }

  // --emit=mlir stops after lowering and prints the module instead of
  // executing it.
  if (emit == "mlir") {
    module->print(llvm::outs());
    llvm::outs() << "\n";
    return 0;
  }

  // Derive a cache key from the module content and shape parameters.
  // Hash the module source to get a stable key that varies by operation.
  std::string moduleStr;
  {
    llvm::raw_string_ostream os(moduleStr);
    module->print(os);
  }
  std::string cacheKey = "llk_" +
                         std::to_string(std::hash<std::string>{}(moduleStr)) +
                         "_M" + std::to_string(optM) + "_N" +
                         std::to_string(optN) + "_K" + std::to_string(optK);

  // JIT-compile and look up the kernel entry point.
  llk::JitCache cache;
  auto fnOrErr = cache.lookupOrCompile(cacheKey, *module);
  if (!fnOrErr) {
    llvm::errs() << "JIT compilation failed: "
                 << llvm::toString(fnOrErr.takeError()) << "\n";
    return 1;
  }

  llvm::outs() << "Compilation successful\n";
  return 0;
}
