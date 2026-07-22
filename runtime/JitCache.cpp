//===- JitCache.cpp - ORC LLJIT cache for LLK kernels --------------------===//
//
// Implements on-demand compilation of MLIR modules to native code via ORC.
// The lowering pipeline converts from partially-lowered MLIR dialects
// (SCF, Arith, Math, Func, MemRef) down to the LLVM dialect, then
// translates to LLVM IR, compiles via LLJIT, and returns a function pointer.
//
// M5 upgrade: 4-level cache with LRU eviction.
//
//===----------------------------------------------------------------------===//

#include "LLK/Runtime/JitCache.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cstring>
#include <sstream>

using namespace mlir;

namespace llk {

// ---------------------------------------------------------------------------
// CacheLevel method implementations
// ---------------------------------------------------------------------------

std::string *JitCache::CacheLevel::lookup(const std::string &key) {
  std::unique_lock lock(mutex);
  auto it = index.find(key);
  if (it == index.end()) {
    miss_count++;
    return nullptr;
  }
  hit_count++;
  // Move to front (MRU)
  lru_list.splice(lru_list.begin(), lru_list, it->second);
  return &it->second->value;
}

void JitCache::CacheLevel::insert(const std::string &key, std::string value) {
  std::unique_lock lock(mutex);
  // Update existing entry
  auto it = index.find(key);
  if (it != index.end()) {
    it->second->value = std::move(value);
    lru_list.splice(lru_list.begin(), lru_list, it->second);
    return;
  }
  // Evict LRU if at capacity
  while (lru_list.size() >= max_entries && !lru_list.empty()) {
    const auto &back = lru_list.back();
    index.erase(back.key);
    lru_list.pop_back();
    eviction_count++;
  }
  // Insert at front (MRU)
  lru_list.push_front({key, std::move(value)});
  index[key] = lru_list.begin();
}

void JitCache::CacheLevel::remove(const std::string &key) {
  std::unique_lock lock(mutex);
  auto it = index.find(key);
  if (it != index.end()) {
    lru_list.erase(it->second);
    index.erase(it);
  }
}

void JitCache::CacheLevel::clear() {
  std::unique_lock lock(mutex);
  index.clear();
  lru_list.clear();
}

// ---------------------------------------------------------------------------
// Key serialization helpers
// ---------------------------------------------------------------------------

std::string JitCache::kernelKeyToString(const KernelKey &key) {
  // Pack the 13 fields into a compact binary string for efficient hashing.
  // Layout: op(1) dtype_in(1) dtype_out(1) M_bucket(8) N(8) K(8)
  //         layout(1) isa(1) math(1) threads(2) compiler(8) schedule(8)
  // Total: 48 bytes
  std::string s(48, '\0');
  char *p = s.data();
  p[0] = static_cast<char>(key.operation);
  p[1] = static_cast<char>(key.input_type);
  p[2] = static_cast<char>(key.output_type);
  std::memcpy(p + 3, &key.M_bucket, 8);
  std::memcpy(p + 11, &key.N, 8);
  std::memcpy(p + 19, &key.K, 8);
  p[27] = static_cast<char>(key.weight_layout);
  p[28] = static_cast<char>(key.isa);
  p[29] = static_cast<char>(key.math_mode);
  std::memcpy(p + 30, &key.thread_count, 2);
  std::memcpy(p + 32, &key.compiler_version, 8);
  std::memcpy(p + 40, &key.schedule_version, 8);
  return s;
}

std::string JitCache::weightKeyToString(const WeightKey &key) {
  // Pack: ptr(8) shape_hash(8) BK(8) BN(8) = 32 bytes
  std::string s(32, '\0');
  char *p = s.data();
  uint64_t ptrVal = reinterpret_cast<uint64_t>(key.weight_ptr);
  std::memcpy(p, &ptrVal, 8);
  std::memcpy(p + 8, &key.shape_hash, 8);
  std::memcpy(p + 16, &key.BK, 8);
  std::memcpy(p + 24, &key.BN, 8);
  return s;
}

// ---------------------------------------------------------------------------
// Lowering pipeline: progressively lower MLIR dialects to LLVM dialect.
// ---------------------------------------------------------------------------
static void addLoweringPasses(mlir::PassManager &pm) {
  // Lower vector dialect ops to LLVM dialect (must run before SCF→CF
  // lowering because vector.mask may contain scf ops).
  pm.addPass(mlir::createConvertVectorToLLVMPass());
  // Lower structured control flow (scf) to basic-block control flow (cf).
#if LLVM_VERSION_MAJOR >= 21
  pm.addPass(mlir::createSCFToControlFlowPass());
#else
  pm.addPass(mlir::createConvertSCFToCFPass());
#endif
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
JitCache::lookupOrCompile(const std::string &cache_key, mlir::ModuleOp module) {
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
    return llvm::make_error<llvm::StringError>("Lowering passes failed",
                                               llvm::inconvertibleErrorCode());

  // Translate to LLVM IR.
  auto llvmCtx = std::make_unique<llvm::LLVMContext>();
  auto llvmModule = mlir::translateModuleToLLVMIR(module, *llvmCtx);
  if (!llvmModule)
    return llvm::make_error<llvm::StringError>("LLVM IR translation failed",
                                               llvm::inconvertibleErrorCode());

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
// L3: Object-code cache
// ---------------------------------------------------------------------------

std::optional<JitCache::KernelFn>
JitCache::lookupObjectCode(const KernelKey &key) {
  std::string skey = kernelKeyToString(key);
  std::string *val = object_cache_.lookup(skey);
  if (!val)
    return std::nullopt;

  // Deserialize: the value stores the function pointer as 8 raw bytes.
  if (val->size() < sizeof(KernelFn))
    return std::nullopt;
  KernelFn fn;
  std::memcpy(&fn, val->data(), sizeof(KernelFn));
  return fn;
}

void JitCache::insertObjectCode(const KernelKey &key, KernelFn fn) {
  std::string skey = kernelKeyToString(key);
  std::string value(sizeof(KernelFn), '\0');
  std::memcpy(value.data(), &fn, sizeof(KernelFn));
  object_cache_.insert(skey, std::move(value));
}

// ---------------------------------------------------------------------------
// L2: Optimized-MLIR cache
// ---------------------------------------------------------------------------

std::optional<std::string>
JitCache::lookupOptimizedMLIR(const std::string &key) {
  std::string *val = optimized_cache_.lookup(key);
  if (!val)
    return std::nullopt;
  return *val;
}

void JitCache::insertOptimizedMLIR(const std::string &key,
                                   std::string moduleStr) {
  optimized_cache_.insert(key, std::move(moduleStr));
}

// ---------------------------------------------------------------------------
// L1: IR cache
// ---------------------------------------------------------------------------

std::optional<std::string> JitCache::lookupIR(const std::string &key) {
  std::string *val = ir_cache_.lookup(key);
  if (!val)
    return std::nullopt;
  return *val;
}

void JitCache::insertIR(const std::string &key, std::string moduleStr) {
  ir_cache_.insert(key, std::move(moduleStr));
}

// ---------------------------------------------------------------------------
// L4: Packed-weight cache
// ---------------------------------------------------------------------------

std::optional<std::string> JitCache::lookupPackedWeights(const WeightKey &key) {
  std::string skey = weightKeyToString(key);
  std::string *val = weight_cache_.lookup(skey);
  if (!val)
    return std::nullopt;
  return *val;
}

void JitCache::insertPackedWeights(const WeightKey &key,
                                   std::string packedBlob) {
  std::string skey = weightKeyToString(key);
  weight_cache_.insert(skey, std::move(packedBlob));
}

// ---------------------------------------------------------------------------
// Cache management
// ---------------------------------------------------------------------------

void JitCache::evictLRU(size_t level, size_t max_entries) {
  CacheLevel *cl = nullptr;
  switch (level) {
  case 1:
    cl = &ir_cache_;
    break;
  case 2:
    cl = &optimized_cache_;
    break;
  case 3:
    cl = &object_cache_;
    break;
  case 4:
    cl = &weight_cache_;
    break;
  default:
    return;
  }

  std::unique_lock lock(cl->mutex);
  while (cl->lru_list.size() > max_entries && !cl->lru_list.empty()) {
    const auto &back = cl->lru_list.back();
    cl->index.erase(back.key);
    cl->lru_list.pop_back();
    cl->eviction_count++;
  }
  cl->max_entries = max_entries;
}

void JitCache::setMaxEntries(size_t level, size_t max_entries) {
  evictLRU(level, max_entries);
}

CacheStats JitCache::getStats() const {
  CacheStats s;
  s.hits[0] = ir_cache_.hit_count;
  s.misses[0] = ir_cache_.miss_count;
  s.evictions[0] = ir_cache_.eviction_count;
  s.hits[1] = optimized_cache_.hit_count;
  s.misses[1] = optimized_cache_.miss_count;
  s.evictions[1] = optimized_cache_.eviction_count;
  s.hits[2] = object_cache_.hit_count;
  s.misses[2] = object_cache_.miss_count;
  s.evictions[2] = object_cache_.eviction_count;
  s.hits[3] = weight_cache_.hit_count;
  s.misses[3] = weight_cache_.miss_count;
  s.evictions[3] = weight_cache_.eviction_count;
  return s;
}

void JitCache::clear() {
  {
    std::unique_lock lock(mutex_);
    cache_.clear();
  }
  ir_cache_.clear();
  optimized_cache_.clear();
  object_cache_.clear();
  weight_cache_.clear();
}

size_t JitCache::size() const {
  std::shared_lock lock(mutex_);
  return cache_.size() + ir_cache_.size() + optimized_cache_.size() +
         object_cache_.size() + weight_cache_.size();
}

} // namespace llk
