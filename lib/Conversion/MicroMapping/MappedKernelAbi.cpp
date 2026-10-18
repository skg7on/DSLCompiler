//===- MappedKernelAbi.cpp - Mapped kernel ABI contract -------------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/MappedKernelAbi.h"

#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <utility>
#include <vector>

namespace llk {
namespace {
llvm::Error abiError(std::string message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s",
                                 message.c_str());
}

std::string typeToString(mlir::Type type) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  type.print(stream);
  return text;
}

KernelAbi::Port describe(mlir::MemRefType type) {
  KernelAbi::Port port;
  port.shape.assign(type.getShape().begin(), type.getShape().end());
  port.elementType = typeToString(type.getElementType());
  return port;
}

bool isSupportedAbiElementType(llvm::StringRef type) {
  return type == "f32" || type == "bf16" || type == "f16" || type == "i32" ||
         type == "i8";
}

/// The mapped ABI v1 envelope, stated as an execution capability rather than a
/// malformed-ABI error. Analysis and lowering are rank-neutral; only preparing
/// callable code has to fit these compact descriptors, so the rejection names
/// the limit it hit.
llvm::Error validateMappedAbiEnvelope(const KernelAbi &abi) {
  if (abi.version != kMappedAbiVersion)
    return abiError("runtime_capability: unsupported mapped ABI version " +
                    std::to_string(abi.version));
  if (abi.inputs.size() > kMappedMaxDescriptors ||
      abi.outputs.size() > kMappedMaxDescriptors - abi.inputs.size())
    return abiError("runtime_capability: mapped ABI v1 supports at most 12 "
                    "descriptors");
  auto validatePorts = [](const std::vector<KernelAbi::Port> &ports,
                          llvm::StringRef kind) -> llvm::Error {
    for (size_t i = 0; i < ports.size(); ++i) {
      const auto &port = ports[i];
      if (port.shape.size() != 2)
        return abiError("runtime_capability: mapped ABI v1 requires rank-two "
                        "ports; " +
                        (kind + " " + llvm::Twine(i) + " has rank " +
                         llvm::Twine(port.shape.size()))
                            .str());
      if (port.shape[0] <= 0 || port.shape[1] <= 0)
        return abiError("runtime_capability: mapped ABI v1 requires positive "
                        "static extents; " +
                        (kind + " " + llvm::Twine(i)).str());
      if (!isSupportedAbiElementType(port.elementType))
        return abiError("runtime_capability: mapped ABI v1 does not support "
                        "element type '" +
                        port.elementType + "' on " +
                        (kind + " " + llvm::Twine(i)).str());
    }
    return llvm::Error::success();
  };
  if (llvm::Error error = validatePorts(abi.inputs, "input"))
    return error;
  return validatePorts(abi.outputs, "output");
}

llvm::Error validatePortType(mlir::Type type, const KernelAbi::Port &port,
                             llvm::StringRef kind, size_t index) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memref)
    return abiError((kind + " " + llvm::Twine(index) + " has type '" +
                     typeToString(type) + "', expected a memref")
                        .str());
  KernelAbi::Port actual = describe(memref);
  if (actual.shape != port.shape || actual.elementType != port.elementType)
    return abiError((kind + " " + llvm::Twine(index) +
                     " type does not match the recorded mapped ABI")
                        .str());
  return llvm::Error::success();
}

llvm::Error makeResultsBorrowedOutputs(mlir::func::FuncOp function,
                                       KernelAbi &abi) {
  mlir::FunctionType originalType = function.getFunctionType();
  llvm::SmallVector<mlir::Type> resultTypes(originalType.getResults().begin(),
                                            originalType.getResults().end());
  for (mlir::Type result : resultTypes) {
    auto memref = mlir::dyn_cast<mlir::MemRefType>(result);
    if (!memref)
      return abiError("kernel result is '" + typeToString(result) +
                      "', but mapped results must be memrefs");
    KernelAbi::Port port = describe(memref);
    if (llvm::Error error =
            validatePortType(result, port, "output", abi.outputs.size()))
      return error;
    abi.outputs.push_back(std::move(port));
  }
  if (llvm::Error error = validateMappedAbiEnvelope(abi))
    return error;
  if (resultTypes.empty())
    return llvm::Error::success();

  mlir::Region &body = function.getBody();
  if (body.empty())
    return abiError("cannot prepare an external mapped kernel for invocation");
  mlir::Block &entry = body.front();
  if (entry.getNumArguments() != originalType.getNumInputs())
    return abiError(
        "mapped kernel entry block arguments do not match its type");

  llvm::SmallVector<mlir::Value> outputArguments;
  outputArguments.reserve(resultTypes.size());
  for (mlir::Type type : resultTypes)
    outputArguments.push_back(entry.addArgument(type, function.getLoc()));

  llvm::SmallVector<mlir::Type> newInputs(originalType.getInputs().begin(),
                                          originalType.getInputs().end());
  newInputs.append(resultTypes.begin(), resultTypes.end());
  function.setType(
      mlir::FunctionType::get(function.getContext(), newInputs, {}));

  llvm::SmallVector<mlir::func::ReturnOp> returns;
  function.walk([&](mlir::func::ReturnOp ret) { returns.push_back(ret); });
  if (returns.empty())
    return abiError("mapped kernel with results has no func.return operation");
  for (mlir::func::ReturnOp ret : returns) {
    if (ret.getNumOperands() != resultTypes.size())
      return abiError("mapped kernel return arity changed during preparation");
    mlir::OpBuilder builder(ret);
    for (auto [value, output] : llvm::zip(ret.getOperands(), outputArguments))
      mlir::memref::CopyOp::create(builder, ret.getLoc(), value, output);
    mlir::func::ReturnOp::create(builder, ret.getLoc());
    ret.erase();
  }
  return llvm::Error::success();
}

void registerDeallocationInterfaces(mlir::MLIRContext &context) {
  mlir::DialectRegistry registry;
  mlir::scf::registerBufferDeallocationOpInterfaceExternalModels(registry);
  context.appendDialectRegistry(registry);
}
} // namespace

llvm::Expected<PreparedMappedKernel>
prepareMappedKernelForInvocation(mlir::ModuleOp bufferedModule,
                                 llvm::StringRef entrySymbol) {
  if (!bufferedModule)
    return abiError("cannot prepare a null module for mapped invocation");
  if (entrySymbol.empty())
    return abiError("mapped invocation preparation needs an entry symbol");

  PreparedMappedKernel prepared;
  prepared.module = bufferedModule.clone();
  prepared.entrySymbol = entrySymbol.str();
  mlir::func::FuncOp function =
      prepared.module->lookupSymbol<mlir::func::FuncOp>(entrySymbol);
  if (!function)
    return abiError("module has no function named '" + entrySymbol.str() + "'");
  if (function.isExternal() || function.getBody().empty())
    return abiError("cannot prepare an external mapped kernel for invocation");

  for (mlir::Type input : function.getFunctionType().getInputs()) {
    auto memref = mlir::dyn_cast<mlir::MemRefType>(input);
    if (!memref)
      return abiError("kernel input is '" + typeToString(input) +
                      "', but mapped inputs must be memrefs");
    prepared.abi.inputs.push_back(describe(memref));
  }
  for (size_t i = 0; i < prepared.abi.inputs.size(); ++i)
    if (llvm::Error error =
            validatePortType(function.getFunctionType().getInput(i),
                             prepared.abi.inputs[i], "input", i))
      return std::move(error);

  if (llvm::Error error = makeResultsBorrowedOutputs(function, prepared.abi))
    return std::move(error);

  registerDeallocationInterfaces(*prepared.module->getContext());
  mlir::PassManager pm(prepared.module->getContext());
  mlir::bufferization::buildBufferDeallocationPipeline(pm);
  if (mlir::failed(pm.run(*prepared.module)))
    return abiError("mapped ownership preparation failed");
  return prepared;
}
} // namespace llk
