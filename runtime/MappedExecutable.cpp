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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ExecutionEngine/Orc/LLJIT.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>
#include <utility>

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

KernelAbi::Port describe(mlir::MemRefType type) {
  KernelAbi::Port port;
  port.shape.assign(type.getShape().begin(), type.getShape().end());
  port.elementType = typeToString(type.getElementType());
  return port;
}

/// Turns every result into a trailing caller-owned parameter.
///
/// The caller has to own the output buffer, and a returned memref aggregate is
/// a convention no C declaration expresses, so the parameter list is where it
/// belongs. After this the function returns nothing and its convention is the
/// uniform one the C-interface wrapper then exposes.
llvm::Error resultsBecomeOutParams(mlir::func::FuncOp function,
                                   KernelAbi &abi) {
  mlir::FunctionType type = function.getFunctionType();
  mlir::Location loc = function.getLoc();

  for (mlir::Type result : type.getResults()) {
    auto memref = mlir::dyn_cast<mlir::MemRefType>(result);
    if (!memref)
      return abiError("kernel result is '" + typeToString(result) +
                      "', but the mapped ABI describes results as buffers");
    abi.outputs.push_back(describe(memref));
  }
  if (type.getNumResults() == 0)
    return llvm::Error::success();

  mlir::Block &entry = function.getBody().front();
  llvm::SmallVector<mlir::Value> outs;
  for (mlir::Type result : type.getResults())
    outs.push_back(entry.addArgument(result, loc));

  llvm::SmallVector<mlir::Type> inputs(type.getInputs().begin(),
                                       type.getInputs().end());
  inputs.append(type.getResults().begin(), type.getResults().end());
  function.setType(mlir::FunctionType::get(function.getContext(), inputs, {}));

  // The value that used to be returned is written into the caller's buffer,
  // which is the whole point of making it a parameter.
  auto ret = mlir::cast<mlir::func::ReturnOp>(entry.getTerminator());
  mlir::OpBuilder builder(ret);
  for (auto [value, out] : llvm::zip(ret.getOperands(), outs))
    mlir::memref::CopyOp::create(builder, loc, value, out);
  mlir::func::ReturnOp::create(builder, loc);
  ret.erase();
  return llvm::Error::success();
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

llvm::Error MappedExecutable::invoke(llvm::ArrayRef<MemRef2D *> inputs,
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
                                   void *entry, KernelAbi abi)
    : jit_(std::move(jit)), entry_(entry), abi_(std::move(abi)) {}

MappedExecutable::~MappedExecutable() = default;

llvm::Expected<std::unique_ptr<MappedExecutable>>
createMappedExecutable(mlir::ModuleOp module, llvm::StringRef entrySymbol) {
  mlir::func::FuncOp function =
      module.lookupSymbol<mlir::func::FuncOp>(entrySymbol);
  if (!function)
    return abiError("module has no function named '" + entrySymbol.str() + "'");

  KernelAbi abi;
  for (mlir::Type input : function.getFunctionType().getInputs()) {
    auto memref = mlir::dyn_cast<mlir::MemRefType>(input);
    if (!memref)
      return abiError("kernel input is '" + typeToString(input) +
                      "', but the mapped ABI describes inputs as buffers");
    abi.inputs.push_back(describe(memref));
  }
  if (llvm::Error error = resultsBecomeOutParams(function, abi))
    return std::move(error);

  // The C-interface wrapper is what turns descriptor pointers into the expanded
  // form the lowered body expects, so the entry point asks for it explicitly.
  function->setAttr("llvm.emit_c_interface",
                    mlir::UnitAttr::get(module.getContext()));

  mlir::registerBuiltinDialectTranslation(*module.getContext());
  mlir::registerLLVMDialectTranslation(*module.getContext());

  mlir::PassManager pm(module.getContext());
  addKernelToLLVMPasses(pm);
  if (mlir::failed(pm.run(module)))
    return abiError("lowering kernel '" + entrySymbol.str() +
                    "' to the LLVM dialect failed");

  auto llvmContext = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(module, *llvmContext);
  if (!llvmModule)
    return abiError("translating kernel '" + entrySymbol.str() +
                    "' to LLVM IR failed");

  // Checked before the module is handed to the JIT, so a kernel whose wrapper
  // was never generated fails with that fact rather than a lookup error.
  std::string wrapperName = ("_mlir_ciface_" + entrySymbol).str();
  if (!llvmModule->getFunction(wrapperName))
    return abiError("no C-interface wrapper '" + wrapperName +
                    "' was generated for kernel '" + entrySymbol.str() + "'");

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  llvm::Expected<std::unique_ptr<llvm::orc::LLJIT>> jit =
      llvm::orc::LLJITBuilder().create();
  if (!jit)
    return jit.takeError();
  if (llvm::Error error = (*jit)->addIRModule(llvm::orc::ThreadSafeModule(
          std::move(llvmModule), std::move(llvmContext))))
    return std::move(error);

  llvm::Expected<llvm::orc::ExecutorAddr> symbol = (*jit)->lookup(wrapperName);
  if (!symbol)
    return symbol.takeError();

  return std::unique_ptr<MappedExecutable>(new MappedExecutable(
      std::move(*jit), symbol->toPtr<void *>(), std::move(abi)));
}

} // namespace llk
