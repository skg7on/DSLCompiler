//===- MappedKernelAbi.cpp - Mapped kernel ABI contract -------------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/MappedKernelAbi.h"

#include "LLK/Conversion/Workload/WorkloadContract.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferDeallocationOpInterfaceImpl.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace llk {
namespace {
llvm::Error abiError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s",
                                 message.str().c_str());
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

//===----------------------------------------------------------------------===//
// ABI v2: symbolic ports and caller-owned output destinations
//===----------------------------------------------------------------------===//

using mlir::llk::workload::PortDirection;
using mlir::llk::workload::WorkloadContract;

/// The contract port each signature position realizes.
struct PortBindings {
  llvm::SmallVector<llk::AbiPortRole> roles;
  llvm::SmallVector<uint32_t> ports;
};

llvm::Expected<PortBindings> decodeArgumentBindings(mlir::func::FuncOp function,
                                                    unsigned argumentCount) {
  auto bindings = function->getAttrOfType<mlir::ArrayAttr>("argument_bindings");
  if (!bindings)
    return abiError("a symbolic kernel ABI needs argument bindings");
  if (bindings.size() != argumentCount)
    return abiError("argument bindings do not cover the kernel signature");
  PortBindings decoded;
  for (mlir::Attribute entry : bindings) {
    auto dictionary = mlir::dyn_cast<mlir::DictionaryAttr>(entry);
    auto role = dictionary ? dictionary.getAs<mlir::StringAttr>("role")
                           : mlir::StringAttr();
    auto port = dictionary ? dictionary.getAs<mlir::IntegerAttr>("port")
                           : mlir::IntegerAttr();
    if (!role || !port || port.getInt() < 0 ||
        (role.getValue() != "input" && role.getValue() != "output"))
      return abiError("malformed argument binding on the lowered kernel");
    decoded.roles.push_back(role.getValue() == "output"
                                ? llk::AbiPortRole::Output
                                : llk::AbiPortRole::Input);
    decoded.ports.push_back(static_cast<uint32_t>(port.getInt()));
  }
  return decoded;
}

/// A signature argument's memref must realize the port it is bound to. A
/// constant extent is exact; a symbolic extent must stay dynamic so the value
/// is bound per invocation rather than frozen into the artifact.
llvm::Error checkSymbolicArgumentType(mlir::Type type,
                                      const llk::KernelAbi::Port &port,
                                      llvm::StringRef kind, size_t index) {
  auto memref = mlir::dyn_cast<mlir::MemRefType>(type);
  if (!memref)
    return abiError((kind + " " + llvm::Twine(index) + " has type '" +
                     typeToString(type) + "', expected a memref")
                        .str());
  if (memref.getRank() != 2)
    return abiError(
        (kind + " " + llvm::Twine(index) + " must have rank two").str());
  if (!memref.getLayout().isIdentity())
    return abiError((kind + " " + llvm::Twine(index) +
                     " must have an identity layout for the compact "
                     "descriptor transport")
                        .str());
  if (typeToString(memref.getElementType()) != port.elementType)
    return abiError((kind + " " + llvm::Twine(index) +
                     " element type does not match its ABI port")
                        .str());
  if (port.dimensions.size() != 2)
    return abiError(
        (kind + " " + llvm::Twine(index) + " ABI port must have rank two")
            .str());
  for (size_t dim = 0; dim < 2; ++dim) {
    const ::mlir::llk::shape::Dimension &expected = port.dimensions[dim];
    int64_t actual = memref.getShape()[dim];
    if (expected.isConstant()) {
      if (actual != expected.getConstant())
        return abiError((kind + " " + llvm::Twine(index) + " dimension " +
                         llvm::Twine(dim) + " is " + llvm::Twine(actual) +
                         " but its ABI port declares " +
                         llvm::Twine(expected.getConstant()))
                            .str());
      continue;
    }
    if (actual != mlir::ShapedType::kDynamic)
      return abiError((kind + " " + llvm::Twine(index) + " dimension " +
                       llvm::Twine(dim) +
                       " is static but its ABI port is symbolic")
                          .str());
  }
  return llvm::Error::success();
}

/// Follows a returned value back to the caller-owned destination it updates.
///
/// Loop-carried destinations, casts and the tensor/buffer materializations
/// One-Shot Bufferize leaves behind are all this value may pass through. A
/// compiler-owned allocation anywhere on the path is exactly the full-output
/// temporary the caller-owned-output contract forbids, so it is reported as a
/// mismatch rather than accepted.
mlir::Value traceToDestination(mlir::Value value,
                               mlir::BlockArgument destination,
                               unsigned depth) {
  if (depth > 64)
    return {};
  if (value == destination)
    return value;
  if (mlir::isa<mlir::BlockArgument>(value))
    return {};
  auto result = mlir::cast<mlir::OpResult>(value);
  mlir::Operation *defining = value.getDefiningOp();
  unsigned number = result.getResultNumber();
  if (auto loop = mlir::dyn_cast<mlir::scf::ForOp>(defining)) {
    if (number >= loop.getInitArgs().size())
      return {};
    return traceToDestination(loop.getInitArgs()[number], destination,
                              depth + 1);
  }
  if (auto loop = mlir::dyn_cast<mlir::scf::ForallOp>(defining)) {
    if (number >= loop.getOutputs().size())
      return {};
    return traceToDestination(loop.getOutputs()[number], destination,
                              depth + 1);
  }
  if (auto conditional = mlir::dyn_cast<mlir::scf::IfOp>(defining)) {
    mlir::scf::YieldOp thenYield = conditional.thenYield();
    mlir::scf::YieldOp elseYield = conditional.elseYield();
    if (!thenYield || number >= thenYield.getNumOperands())
      return {};
    if (!traceToDestination(thenYield.getOperand(number), destination,
                            depth + 1))
      return {};
    if (elseYield) {
      if (number >= elseYield.getNumOperands())
        return {};
      if (!traceToDestination(elseYield.getOperand(number), destination,
                              depth + 1))
        return {};
    }
    return destination;
  }
  if (auto cast = mlir::dyn_cast<mlir::memref::CastOp>(defining))
    return traceToDestination(cast.getSource(), destination, depth + 1);
  if (auto toBuffer = mlir::dyn_cast<mlir::bufferization::ToBufferOp>(defining))
    return traceToDestination(toBuffer.getTensor(), destination, depth + 1);
  if (auto toTensor = mlir::dyn_cast<mlir::bufferization::ToTensorOp>(defining))
    return traceToDestination(toTensor.getBuffer(), destination, depth + 1);
  return {};
}

/// Builds the version 2 ABI of a bound kernel and prepares its callable form.
///
/// The contract is the ABI's symbolic metadata when the kernel retained one;
/// a kernel without a contract is described by one symbol per dynamic
/// dimension, so the ABI never invents an equality its source did not prove
/// (the kernel's own guards remain the authority on cross-port relationships).
/// Each yielded result must trace to the caller-owned destination bound to its
/// output port; it is then eliminated, leaving the write in the caller's
/// storage rather than copied into a compiler-owned buffer.
llvm::Error prepareSymbolicInvocation(mlir::func::FuncOp function,
                                      KernelAbi &abi) {
  mlir::FunctionType type = function.getFunctionType();
  llvm::Expected<PortBindings> bindings =
      decodeArgumentBindings(function, type.getNumInputs());
  if (!bindings)
    return bindings.takeError();

  std::optional<WorkloadContract> contract;
  if (auto attr = function->getAttrOfType<mlir::DictionaryAttr>(
          "micro.workload_contract")) {
    llvm::Expected<WorkloadContract> decoded =
        mlir::llk::workload::decodeWorkloadContract(attr);
    if (!decoded)
      return decoded.takeError();
    contract = std::move(*decoded);
  }

  abi.version = kMappedAbiVersionV2;
  std::vector<KernelAbi::Port> ports;
  std::vector<llk::AbiPortRole> portRoles;
  if (contract) {
    for (const mlir::llk::workload::WorkloadPort &port : contract->ports) {
      KernelAbi::Port described;
      described.elementType = port.elementType;
      described.dimensions = port.shape;
      ports.push_back(std::move(described));
      portRoles.push_back(port.direction == PortDirection::Input
                              ? llk::AbiPortRole::Input
                              : llk::AbiPortRole::Output);
    }
    abi.symbols = contract->symbols;
  } else {
    // No retained contract: the argument's own position numbers its port and
    // each dynamic extent gets a distinct symbol.
    for (unsigned i = 0; i < type.getNumInputs(); ++i) {
      if ((*bindings).ports[i] != i)
        return abiError("a bound kernel without a workload contract must "
                        "number its ports by signature position");
      auto memref = mlir::dyn_cast<mlir::MemRefType>(type.getInput(i));
      if (!memref)
        return abiError("mapped kernel argument " + llvm::Twine(i) +
                        " is not a memref");
      KernelAbi::Port described;
      described.elementType = typeToString(memref.getElementType());
      for (size_t dim = 0; dim < 2; ++dim) {
        if (memref.isDynamicDim(dim)) {
          std::string name =
              "d" + std::to_string(i) + "_" + std::to_string(dim);
          described.dimensions.push_back(
              ::mlir::llk::shape::Dimension::symbol(name));
          abi.symbols.emplace(name, ::mlir::llk::shape::SymbolConstraint{});
        } else {
          described.dimensions.emplace_back(memref.getShape()[dim]);
        }
      }
      ports.push_back(std::move(described));
      portRoles.push_back((*bindings).roles[i]);
    }
  }

  // Filter contract ports into the caller-facing input and output descriptor
  // lists, remembering where each contract port landed.
  std::vector<uint32_t> inputOfPort(ports.size(), UINT32_MAX);
  std::vector<uint32_t> outputOfPort(ports.size(), UINT32_MAX);
  for (size_t i = 0; i < ports.size(); ++i) {
    if (portRoles[i] == llk::AbiPortRole::Input) {
      inputOfPort[i] = static_cast<uint32_t>(abi.inputs.size());
      abi.inputs.push_back(ports[i]);
    } else {
      outputOfPort[i] = static_cast<uint32_t>(abi.outputs.size());
      abi.outputs.push_back(ports[i]);
    }
  }

  std::vector<bool> boundInputs(abi.inputs.size(), false);
  std::vector<bool> boundOutputs(abi.outputs.size(), false);
  abi.arguments.reserve(type.getNumInputs());
  for (unsigned i = 0; i < type.getNumInputs(); ++i) {
    uint32_t port = (*bindings).ports[i];
    if (port >= ports.size())
      return abiError("argument " + llvm::Twine(i) +
                      " binds an out-of-range contract port");
    llk::AbiPortRole role = (*bindings).roles[i];
    if (role != portRoles[port])
      return abiError("argument " + llvm::Twine(i) +
                      " role disagrees with its contract port");
    uint32_t filtered = role == llk::AbiPortRole::Input ? inputOfPort[port]
                                                        : outputOfPort[port];
    std::vector<bool> &bound =
        role == llk::AbiPortRole::Input ? boundInputs : boundOutputs;
    if (bound[filtered])
      return abiError("contract port " + llvm::Twine(port) +
                      " is bound by more than one argument");
    bound[filtered] = true;
    if (llvm::Error error = checkSymbolicArgumentType(
            type.getInput(i), ports[port],
            role == llk::AbiPortRole::Input ? "input" : "output", i))
      return error;
    abi.arguments.push_back({role, filtered});
  }
  for (size_t i = 0; i < boundInputs.size(); ++i)
    if (!boundInputs[i])
      return abiError("input port " + llvm::Twine(i) +
                      " is not bound by a kernel argument");
  for (size_t i = 0; i < boundOutputs.size(); ++i)
    if (!boundOutputs[i])
      return abiError("output port " + llvm::Twine(i) +
                      " is not bound by a caller-owned destination argument");

  // Results: every yielded value must be the destination bound to its output
  // port. Lowering a destination-bearing store updates that argument in
  // place, so the result itself carries no new information and is dropped.
  mlir::Block &entry = function.getBody().front();
  auto resultsAttr =
      function->getAttrOfType<mlir::ArrayAttr>("result_bindings");
  if (!resultsAttr)
    return abiError("a symbolic kernel ABI needs result bindings");
  if (resultsAttr.size() != type.getNumResults())
    return abiError("result bindings do not cover the kernel signature");

  for (unsigned r = 0; r < type.getNumResults(); ++r) {
    auto portAttr = mlir::dyn_cast<mlir::IntegerAttr>(resultsAttr[r]);
    if (!portAttr || portAttr.getInt() < 0 ||
        static_cast<size_t>(portAttr.getInt()) >= ports.size())
      return abiError("malformed result binding on the lowered kernel");
    uint32_t port = static_cast<uint32_t>(portAttr.getInt());
    if (portRoles[port] != llk::AbiPortRole::Output)
      return abiError("result " + llvm::Twine(r) +
                      " names a port that is not an output");
    unsigned destination = UINT_MAX;
    for (unsigned i = 0; i < (*bindings).ports.size(); ++i)
      if ((*bindings).ports[i] == port &&
          (*bindings).roles[i] == llk::AbiPortRole::Output)
        destination = i;
    if (destination == UINT_MAX)
      return abiError("result " + llvm::Twine(r) +
                      " names an output port with no destination argument");
    mlir::BlockArgument argument = entry.getArgument(destination);

    bool sawReturn = false;
    for (mlir::func::ReturnOp ret : function.getOps<mlir::func::ReturnOp>()) {
      if (ret.getNumOperands() != type.getNumResults())
        return abiError(
            "mapped kernel return arity changed during preparation");
      sawReturn = true;
      if (!traceToDestination(ret.getOperand(r), argument, 0))
        return abiError(
            "output port " + llvm::Twine(port) +
            " must be written through its caller-owned destination; the "
            "result is not that destination, so bufferization materialised a "
            "compiler-owned buffer");
    }
    if (!sawReturn && type.getNumResults() != 0)
      return abiError("mapped kernel with results has no func.return");
    abi.results.push_back(outputOfPort[port]);
  }

  // Drop the verified results: the caller-owned destination is the only exit.
  if (type.getNumResults() != 0) {
    for (mlir::func::ReturnOp ret : function.getOps<mlir::func::ReturnOp>())
      ret.getOperandsMutable().clear();
    function.setType(
        mlir::FunctionType::get(function.getContext(), type.getInputs(), {}));
  }
  return validateKernelAbi(abi);
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

  if (function->hasAttr("argument_bindings")) {
    // A bound kernel keeps the symbolic contract and each argument's port
    // role, so its output destination is already a caller-owned argument.
    if (llvm::Error error = prepareSymbolicInvocation(function, prepared.abi))
      return std::move(error);
  } else {
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
  }

  registerDeallocationInterfaces(*prepared.module->getContext());
  mlir::PassManager pm(prepared.module->getContext());
  mlir::bufferization::buildBufferDeallocationPipeline(pm);
  if (mlir::failed(pm.run(*prepared.module)))
    return abiError("mapped ownership preparation failed");
  return prepared;
}
} // namespace llk
