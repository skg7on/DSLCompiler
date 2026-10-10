//===- MappedExecutable.cpp - Invoke a compiled micro kernel (C4) ---------===//
//
// The calling convention a mapped kernel needs, and the code that calls it.
//
// MLIR expands a memref parameter into its seven descriptor fields and a memref
// *result* into a hidden-pointer convention, so a lowered kernel's own LLVM
// signature is not something a C caller can declare. What this file does is fix
// that once, in the lowering, rather than guess at it at the call site:
//
//   * every result becomes a trailing caller-owned parameter, so the
//     convention is always "N descriptors in, M descriptors out";
//   * the entry point is given MLIR's C-interface wrapper, whose parameters are
//     pointers to those descriptors -- exactly the `MemRef2D` the runtime
//     already uses;
//   * the invocation checks the descriptors against the ABI recorded from the
//     MLIR types, and calls the wrapper at its real arity.
//
// Nothing here casts an expanded LLVM entry point to a descriptor signature. A
// cast like that would be an assertion with no evidence behind it, and the
// first kernel whose arity differed would corrupt its caller's memory.
//
//===----------------------------------------------------------------------===//

#include "LLK/Runtime/MappedExecutable.h"
#include "LLK/Conversion/MappedKernelAbi.h"
#include "LLK/Mapping/CodegenRequirements.h"
#include "mapped_jit_test_factory.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "mlir/ExecutionEngine/CRunnerUtils.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ExecutionEngine/JITSymbol.h"
#include "llvm/ExecutionEngine/Orc/AbsoluteSymbols.h"
#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include <cstdint>
#include <string>
#include <utility>

llvm::Error mlir::llk::mapping::checkHostRequirements(
    const TargetCodegenRequirements &requirements,
    llvm::StringRef hostArchitecture,
    const llvm::StringMap<bool> &hostFeatures) {
  if (!requirements.architecture.empty() &&
      requirements.architecture != hostArchitecture)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "selected target requires architecture '" + requirements.architecture +
            "' but host architecture is '" + hostArchitecture.str() + "'");
  for (const std::string &feature : requirements.requiredFeatures) {
    auto available = hostFeatures.find(feature);
    if (available == hostFeatures.end() || !available->second)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "selected target requires host feature '" +
                                         feature + "'");
  }
  return llvm::Error::success();
}

namespace llk {

namespace {

using llvm::inconvertibleErrorCode;
using llvm::StringError;

llvm::Error abiError(const std::string &message) {
  return llvm::make_error<StringError>(message, inconvertibleErrorCode());
}

/// Renders a type the way the ABI records it, for a diagnostic that names what
/// was expected rather than only that something was wrong.
std::string typeToString(mlir::Type type) {
  std::string buffer;
  llvm::raw_string_ostream stream(buffer);
  type.print(stream);
  return buffer;
}

/// Checks one side's descriptors against the recorded ABI.
///
/// The generated code addresses a buffer as if the type it was compiled for
/// were the truth. A descriptor that disagrees therefore does not merely give a
/// wrong answer -- it reads or writes memory outside what the caller offered,
/// which is why this is checked rather than assumed.
llvm::Error checkDescriptors(llvm::ArrayRef<MemRef2D *> given,
                             llvm::ArrayRef<KernelAbi::Port> ports,
                             llvm::StringRef side) {
  for (size_t index = 0; index < ports.size(); ++index) {
    const MemRef2D *descriptor = given[index];
    std::string where = (side + " descriptor #" + std::to_string(index)).str();
    if (!descriptor)
      return abiError(where + " is null");
    if (!descriptor->aligned)
      return abiError(where + " has no data pointer");
    if (descriptor->offset < 0)
      return abiError(where + " has a negative offset");

    const KernelAbi::Port &port = ports[index];
    if (port.shape.size() != 2)
      continue; // only the rank-2 descriptors the ABI describes are checked
    int64_t rows = port.shape[0], columns = port.shape[1];
    if (descriptor->size0 != rows || descriptor->size1 != columns)
      return abiError(
          where + " declares a " + std::to_string(descriptor->size0) + "x" +
          std::to_string(descriptor->size1) +
          " buffer, but the kernel was compiled for a " + std::to_string(rows) +
          "x" + std::to_string(columns) + " one");
    // The lowered code folds a static shape's strides into its addressing, so a
    // descriptor carrying a different layout would be silently ignored -- and
    // would read the wrong elements.
    if (descriptor->stride1 != 1 || descriptor->stride0 != columns)
      return abiError(where + " has strides [" +
                      std::to_string(descriptor->stride0) + ", " +
                      std::to_string(descriptor->stride1) +
                      "], but the kernel addresses its buffer row-major with "
                      "an inner stride of 1");
  }
  return llvm::Error::success();
}

/// The most descriptors one invocation may pass.
///
/// The C-interface wrapper's arity is real, so calling it with a guessed one is
/// undefined behaviour rather than a convenience. Beyond this bound a kernel is
/// refused instead.
constexpr size_t kMaxDescriptors = 12;

/// Calls `entry` with exactly `sizeof...(Is)` descriptor arguments. Each arity
/// gets its own genuinely-typed function pointer, which is what makes the call
/// an ABI fact rather than an assumption.
template <typename T, size_t> using Repeated = T;

template <size_t... Indices>
void callAtArity(void *entry, llvm::ArrayRef<MemRef2D *> args,
                 std::index_sequence<Indices...>) {
  reinterpret_cast<void (*)(Repeated<MemRef2D *, Indices>...)>(entry)(
      args[Indices]...);
}

} // namespace

llvm::Error
MappedExecutable::invoke(llvm::ArrayRef<InvocationBuffer2D> inputs,
                         llvm::ArrayRef<InvocationBuffer2D> outputs) {
  if (llvm::Error error = validateMappedInvocation(abi_, inputs, outputs))
    return error;

  llvm::SmallVector<MemRef2D, kMaxDescriptors> descriptors;
  for (const InvocationBuffer2D &input : inputs)
    descriptors.push_back(input.descriptor);
  for (const InvocationBuffer2D &output : outputs)
    descriptors.push_back(output.descriptor);
  llvm::SmallVector<MemRef2D *, kMaxDescriptors> args;
  for (MemRef2D &descriptor : descriptors)
    args.push_back(&descriptor);

  switch (args.size()) {
  case 0:
    callAtArity(entry_, args, std::make_index_sequence<0>{});
    break;
  case 1:
    callAtArity(entry_, args, std::make_index_sequence<1>{});
    break;
  case 2:
    callAtArity(entry_, args, std::make_index_sequence<2>{});
    break;
  case 3:
    callAtArity(entry_, args, std::make_index_sequence<3>{});
    break;
  case 4:
    callAtArity(entry_, args, std::make_index_sequence<4>{});
    break;
  case 5:
    callAtArity(entry_, args, std::make_index_sequence<5>{});
    break;
  case 6:
    callAtArity(entry_, args, std::make_index_sequence<6>{});
    break;
  case 7:
    callAtArity(entry_, args, std::make_index_sequence<7>{});
    break;
  case 8:
    callAtArity(entry_, args, std::make_index_sequence<8>{});
    break;
  case 9:
    callAtArity(entry_, args, std::make_index_sequence<9>{});
    break;
  case 10:
    callAtArity(entry_, args, std::make_index_sequence<10>{});
    break;
  case 11:
    callAtArity(entry_, args, std::make_index_sequence<11>{});
    break;
  default:
    callAtArity(entry_, args, std::make_index_sequence<12>{});
    break;
  }
  return llvm::Error::success();
}

llvm::Error MappedExecutable::invoke(llvm::ArrayRef<MemRef2D *> inputs,
                                     llvm::ArrayRef<MemRef2D *> outputs) {
  return invokeUncheckedLegacy(inputs, outputs);
}

llvm::Error
MappedExecutable::invokeUncheckedLegacy(llvm::ArrayRef<MemRef2D *> inputs,
                                        llvm::ArrayRef<MemRef2D *> outputs) {
  if (inputs.size() != abi_.inputs.size())
    return abiError("kernel takes " + std::to_string(abi_.inputs.size()) +
                    " input(s) but " + std::to_string(inputs.size()) +
                    " were given");
  if (outputs.size() != abi_.outputs.size())
    return abiError("kernel produces " + std::to_string(abi_.outputs.size()) +
                    " output(s) but " + std::to_string(outputs.size()) +
                    " were given");
  if (llvm::Error error = checkDescriptors(inputs, abi_.inputs, "input"))
    return error;
  if (llvm::Error error = checkDescriptors(outputs, abi_.outputs, "output"))
    return error;

  llvm::SmallVector<MemRef2D *, kMaxDescriptors> args;
  args.append(inputs.begin(), inputs.end());
  args.append(outputs.begin(), outputs.end());
  if (args.size() > kMaxDescriptors)
    return abiError("kernel takes " + std::to_string(args.size()) +
                    " descriptors, which is more than the " +
                    std::to_string(kMaxDescriptors) + " this ABI supports");

  switch (args.size()) {
  case 0:
    callAtArity(entry_, args, std::make_index_sequence<0>{});
    break;
  case 1:
    callAtArity(entry_, args, std::make_index_sequence<1>{});
    break;
  case 2:
    callAtArity(entry_, args, std::make_index_sequence<2>{});
    break;
  case 3:
    callAtArity(entry_, args, std::make_index_sequence<3>{});
    break;
  case 4:
    callAtArity(entry_, args, std::make_index_sequence<4>{});
    break;
  case 5:
    callAtArity(entry_, args, std::make_index_sequence<5>{});
    break;
  case 6:
    callAtArity(entry_, args, std::make_index_sequence<6>{});
    break;
  case 7:
    callAtArity(entry_, args, std::make_index_sequence<7>{});
    break;
  case 8:
    callAtArity(entry_, args, std::make_index_sequence<8>{});
    break;
  case 9:
    callAtArity(entry_, args, std::make_index_sequence<9>{});
    break;
  case 10:
    callAtArity(entry_, args, std::make_index_sequence<10>{});
    break;
  case 11:
    callAtArity(entry_, args, std::make_index_sequence<11>{});
    break;
  default:
    callAtArity(entry_, args, std::make_index_sequence<12>{});
    break;
  }
  return llvm::Error::success();
}

MappedExecutable::MappedExecutable(std::unique_ptr<llvm::orc::LLJIT> jit,
                                   void *entry, KernelAbi abi,
                                   std::string executionIdentity)
    : jit_(std::move(jit)), entry_(entry), abi_(std::move(abi)),
      executionIdentity_(std::move(executionIdentity)) {}

MappedExecutable::~MappedExecutable() = default;

llvm::Expected<std::unique_ptr<MappedExecutable>>
createMappedExecutable(mlir::ModuleOp module, llvm::StringRef entrySymbol) {
  llvm::Expected<PreparedMappedKernel> prepared =
      prepareMappedKernelForInvocation(module, entrySymbol);
  if (!prepared)
    return prepared.takeError();
  return createMappedExecutable(std::move(*prepared));
}

llvm::Expected<std::unique_ptr<MappedExecutable>>
createMappedExecutable(PreparedMappedKernel prepared,
                       const MappedJitOptions &options) {
  return MappedExecutable::createWithAllocatorHooks(std::move(prepared),
                                                    options, nullptr);
}

llvm::Expected<std::unique_ptr<MappedExecutable>>
MappedExecutable::createWithAllocatorHooks(PreparedMappedKernel prepared,
                                           const MappedJitOptions &options,
                                           testing::TestAllocatorHooks *hooks) {
  if (!prepared.module)
    return abiError("prepared mapped kernel has no module");
  if (llvm::Error error = validateKernelAbi(prepared.abi))
    return std::move(error);
  if (prepared.entrySymbol.empty())
    return abiError("prepared mapped kernel has no entry symbol");

  mlir::ModuleOp module = *prepared.module;
  mlir::func::FuncOp function =
      module.lookupSymbol<mlir::func::FuncOp>(prepared.entrySymbol);
  if (!function)
    return abiError("prepared module has no function named '" +
                    prepared.entrySymbol + "'");
  if (!function.getFunctionType().getResults().empty())
    return abiError("prepared mapped kernel must return void");

  std::string loweredIR;
  llvm::raw_string_ostream loweredIROutput(loweredIR);
  module.print(loweredIROutput);
  loweredIROutput.flush();
  llvm::SHA256 loweredIRHash;
  loweredIRHash.update(loweredIR);
  const auto loweredIRDigest = loweredIRHash.final();
  std::string executionIdentity = options.executionIdentity;
  executionIdentity +=
      "|abi=" + std::to_string(computeKernelAbiHash(prepared.abi));
  executionIdentity +=
      "|lowered-ir-sha256=" +
      llvm::toHex(llvm::ArrayRef<uint8_t>(loweredIRDigest), true);

  size_t argumentCount =
      prepared.abi.inputs.size() + prepared.abi.outputs.size();
  if (function.getFunctionType().getNumInputs() != argumentCount)
    return abiError(
        "prepared mapped kernel argument count does not match its ABI");
  auto validateArguments = [&](llvm::ArrayRef<KernelAbi::Port> ports,
                               size_t offset,
                               llvm::StringRef side) -> llvm::Error {
    for (size_t i = 0; i < ports.size(); ++i) {
      mlir::Type argType = function.getFunctionType().getInput(offset + i);
      auto memref = mlir::dyn_cast<mlir::MemRefType>(argType);
      if (!memref)
        return abiError("prepared " + side.str() + " argument " +
                        std::to_string(i) + " is not a memref");
      if (memref.getShape() != llvm::ArrayRef<int64_t>(ports[i].shape) ||
          typeToString(memref.getElementType()) != ports[i].elementType)
        return abiError("prepared " + side.str() + " argument " +
                        std::to_string(i) + " does not match its ABI port");
    }
    return llvm::Error::success();
  };
  if (llvm::Error error = validateArguments(prepared.abi.inputs, 0, "input"))
    return std::move(error);
  if (llvm::Error error = validateArguments(
          prepared.abi.outputs, prepared.abi.inputs.size(), "output"))
    return std::move(error);

  const auto detectedFeatures = llvm::sys::getHostCPUFeatures();
  llvm::StringMap<bool> hostFeatures;
  for (const auto &feature : detectedFeatures)
    hostFeatures[feature.getKey()] = feature.getValue();
  if (options.selectedTarget) {
    std::string hostTriple = llvm::sys::getDefaultTargetTriple();
    llvm::StringRef hostArchitecture(hostTriple);
    hostArchitecture = hostArchitecture.take_front(hostArchitecture.find('-'));
    if (hostArchitecture == "arm64")
      hostArchitecture = "aarch64";
    else if (hostArchitecture == "amd64")
      hostArchitecture = "x86_64";
    if (llvm::Error error = mlir::llk::mapping::checkHostRequirements(
            *options.selectedTarget, hostArchitecture, hostFeatures))
      return error;
    if (hostFeatures.empty())
      return abiError("cannot detect host CPU features for selected-target "
                      "execution");
  }

  // The C-interface wrapper is what turns descriptor pointers into the expanded
  // form the lowered body expects, so the entry point asks for it explicitly.
  function->setAttr("llvm.emit_c_interface",
                    mlir::UnitAttr::get(module.getContext()));

  mlir::registerBuiltinDialectTranslation(*module.getContext());
  mlir::registerLLVMDialectTranslation(*module.getContext());

  mlir::PassManager pm(module.getContext());
  addKernelToLLVMPasses(pm);
  if (mlir::failed(pm.run(module)))
    return abiError("lowering kernel '" + prepared.entrySymbol +
                    "' to the LLVM dialect failed");

  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(module, *llvmContext);
  if (!llvmModule)
    return abiError("translating kernel '" + prepared.entrySymbol +
                    "' to LLVM IR failed");

  // Checked before the module is handed to the JIT, so a kernel whose wrapper
  // was never generated fails with that fact rather than a lookup error.
  std::string wrapperName = "_mlir_ciface_" + prepared.entrySymbol;
  if (!llvmModule->getFunction(wrapperName))
    return abiError("no C-interface wrapper '" + wrapperName +
                    "' was generated for kernel '" + prepared.entrySymbol +
                    "'");

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  llvm::orc::LLJITBuilder jitBuilder;
  if (options.selectedTarget) {
    llvm::Expected<llvm::orc::JITTargetMachineBuilder> targetMachine =
        llvm::orc::JITTargetMachineBuilder::detectHost();
    if (!targetMachine)
      return targetMachine.takeError();
    if (!options.selectedTarget->cpu.empty())
      targetMachine->setCPU(options.selectedTarget->cpu);
    std::string features;
    for (const std::string &feature :
         options.selectedTarget->requiredFeatures) {
      if (!features.empty())
        features += ',';
      features += '+' + feature;
    }
    if (!features.empty())
      targetMachine->setFeatures(features);
    jitBuilder.setJITTargetMachineBuilder(std::move(*targetMachine));
  }
  llvm::Expected<std::unique_ptr<llvm::orc::LLJIT>> jit = jitBuilder.create();
  if (!jit)
    return jit.takeError();

  // The lowered kernel calls the MLIR C runtime for its `memref.copy`, a call
  // the finalize pass emits rather than inlining. That runtime is linked into
  // this binary, so its address is handed to the JIT directly: relying on the
  // dynamic loader to have kept a symbol nothing else references would work
  // until the linker dropped the library, which is exactly what happened.
  llvm::orc::SymbolMap runtimeSymbols;
  runtimeSymbols[(*jit)->mangleAndIntern("memrefCopy")] =
      llvm::orc::ExecutorSymbolDef(
          llvm::orc::ExecutorAddr::fromPtr(&memrefCopy),
          llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable);
  if (hooks) {
    if (hooks->allocate)
      runtimeSymbols[(*jit)->mangleAndIntern("malloc")] =
          llvm::orc::ExecutorSymbolDef(
              llvm::orc::ExecutorAddr::fromPtr(hooks->allocate),
              llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable);
    if (hooks->alignedAllocate)
      runtimeSymbols[(*jit)->mangleAndIntern("aligned_alloc")] =
          llvm::orc::ExecutorSymbolDef(
              llvm::orc::ExecutorAddr::fromPtr(hooks->alignedAllocate),
              llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable);
    if (hooks->release)
      runtimeSymbols[(*jit)->mangleAndIntern("free")] =
          llvm::orc::ExecutorSymbolDef(
              llvm::orc::ExecutorAddr::fromPtr(hooks->release),
              llvm::JITSymbolFlags::Exported | llvm::JITSymbolFlags::Callable);
  }
  if (llvm::Error error = (*jit)->getMainJITDylib().define(
          llvm::orc::absoluteSymbols(std::move(runtimeSymbols))))
    return std::move(error);

  if (llvm::Error error = (*jit)->addIRModule(llvm::orc::ThreadSafeModule(
          std::move(llvmModule), std::move(llvmContext))))
    return std::move(error);

  llvm::Expected<llvm::orc::ExecutorAddr> symbol = (*jit)->lookup(wrapperName);
  if (!symbol)
    return symbol.takeError();

  return std::unique_ptr<MappedExecutable>(new MappedExecutable(
      std::move(*jit), symbol->toPtr<void *>(), std::move(prepared.abi),
      std::move(executionIdentity)));
}

llvm::Expected<std::unique_ptr<MappedExecutable>>
testing::MappedExecutableTestFactory::create(PreparedMappedKernel prepared,
                                             const MappedJitOptions &options,
                                             TestAllocatorHooks hooks) {
  return MappedExecutable::createWithAllocatorHooks(std::move(prepared),
                                                    options, &hooks);
}

} // namespace llk
