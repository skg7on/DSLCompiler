//===- JitCache.cpp - ORC LLJIT cache for LLK kernels --------------------===//
//
// Implements on-demand compilation of MLIR modules to native code via ORC.
// The lowering pipeline converts from partially-lowered MLIR dialects
// (SCF, Arith, Math, Func, MemRef) down to the LLVM dialect, then
// translates to LLVM IR, compiles via LLJIT, and returns a function pointer.
//
//===----------------------------------------------------------------------===//

#include "LLK/Runtime/JitCache.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace llk {

// ---------------------------------------------------------------------------
// Lowering pipeline: progressively lower MLIR dialects to LLVM dialect.
// ---------------------------------------------------------------------------
static void addLoweringPasses(mlir::PassManager &pm) {
    // Lower structured control flow (scf) to basic-block control flow (cf).
    pm.addPass(mlir::createConvertSCFToCFPass());
    // Lower control flow to LLVM dialect.
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    // Lower arithmetic ops to LLVM dialect.
    pm.addPass(mlir::createArithToLLVMConversionPass());
    // Lower math ops (exp, erf, etc.) to LLVM dialect.
    pm.addPass(mlir::createConvertMathToLLVMPass());
    // Lower func ops (function boundaries) to LLVM dialect.
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    // Lower memref ops to LLVM dialect (finalizes the MemRef→LLVM conversion).
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    // Reconcile unrealized casts between conversion passes.
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

// ---------------------------------------------------------------------------
// Constructor / destructor
// ---------------------------------------------------------------------------

JitCache::JitCache() {
    // Initialize native target support required by ORC JIT.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();

    // Create an ORC LLJIT instance with default settings.
    auto jitBuilder = llvm::orc::LLJITBuilder();
    auto jitOrErr = jitBuilder.create();
    if (!jitOrErr) {
        llvm::errs() << "Failed to create LLJIT: "
                     << llvm::toString(jitOrErr.takeError()) << "\n";
        return;
    }
    jit_ = std::move(*jitOrErr);
}

JitCache::~JitCache() = default;

// ---------------------------------------------------------------------------
// lookupOrCompile
// ---------------------------------------------------------------------------

llvm::Expected<JitCache::KernelFn>
JitCache::lookupOrCompile(const std::string& cache_key,
                           mlir::ModuleOp module) {
    // Check cache first (read lock).
    {
        std::shared_lock lock(mutex_);
        auto it = cache_.find(cache_key);
        if (it != cache_.end())
            return it->second;
    }

    // Ensure the MLIR context has LLVM IR translation registered.
    // These calls are idempotent.
    mlir::registerBuiltinDialectTranslation(*module->getContext());
    mlir::registerLLVMDialectTranslation(*module->getContext());

    // Lower to LLVM dialect.
    mlir::PassManager pm(module->getContext());
    addLoweringPasses(pm);
    if (mlir::failed(pm.run(module)))
        return llvm::make_error<llvm::StringError>(
            "Lowering passes failed", llvm::inconvertibleErrorCode());

    // Translate to LLVM IR.
    auto llvmCtx = std::make_unique<llvm::LLVMContext>();
    auto llvmModule = mlir::translateModuleToLLVMIR(module, *llvmCtx);
    if (!llvmModule)
        return llvm::make_error<llvm::StringError>(
            "LLVM IR translation failed", llvm::inconvertibleErrorCode());

    // Record the module name for symbol lookup.
    std::string moduleName = llvmModule->getModuleIdentifier();

    // Add the LLVM module to the JIT.
    // ThreadSafeModule takes ownership of both the Module and its LLVMContext.
    if (auto err = jit_->addIRModule(llvm::orc::ThreadSafeModule(
            std::move(llvmModule), std::move(llvmCtx))))
        return std::move(err);

    // Look up the compiled entry-point function.
    auto sym = jit_->lookup("llk_swiglu");
    if (!sym)
        return sym.takeError();

    auto fn = sym->toPtr<KernelFn>();

    // Cache and return (write lock).
    {
        std::unique_lock lock(mutex_);
        cache_[cache_key] = fn;
    }
    return fn;
}

// ---------------------------------------------------------------------------
// Cache management
// ---------------------------------------------------------------------------

void JitCache::clear() {
    std::unique_lock lock(mutex_);
    cache_.clear();
}

size_t JitCache::size() const {
    std::shared_lock lock(mutex_);
    return cache_.size();
}

} // namespace llk
