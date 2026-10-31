//===- ShapeSpecialization.cpp - Build an analysis-only shape sample ------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/MicroMapping/ShapeSpecialization.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"

#include <optional>

namespace mlir::llk::mapping {

namespace {

llvm::Error sampleError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>("shape_sample: " + message.str(),
                                             llvm::inconvertibleErrorCode());
}

/// One signature argument's contract port. `port` is the zero-based index into
/// the contract's ordered port list; `role` is the direction the signature
/// binding declared.
struct ArgumentPort {
  size_t port = 0;
  shape::PortDirection direction = shape::PortDirection::Input;
};

/// Reads the kernel's `argument_bindings` attribute. Absent bindings mean the
/// signature is contract-ordered, which is how a hand-written analysis fixture
/// names its ports without repeating the attribute.
llvm::Expected<SmallVector<ArgumentPort>>
argumentPorts(mlir::Operation *kernel, size_t argumentCount,
              const shape::ShapeContract &contract) {
  SmallVector<ArgumentPort> ports;
  auto bindings = kernel->getAttrOfType<mlir::ArrayAttr>("argument_bindings");
  if (!bindings) {
    if (argumentCount > contract.ports.size())
      return sampleError("the kernel has more arguments than its contract "
                         "ports");
    for (size_t index = 0; index < argumentCount; ++index)
      ports.push_back({index, contract.ports[index].direction});
    return ports;
  }
  if (bindings.size() != argumentCount)
    return sampleError("argument_bindings does not cover every kernel "
                       "argument");
  for (mlir::Attribute entry : bindings) {
    auto dictionary = mlir::dyn_cast<mlir::DictionaryAttr>(entry);
    auto role = dictionary ? dictionary.getAs<mlir::StringAttr>("role")
                           : mlir::StringAttr();
    auto port = dictionary ? dictionary.getAs<mlir::IntegerAttr>("port")
                           : mlir::IntegerAttr();
    if (!role || !port || port.getInt() < 0)
      return sampleError("a kernel argument binding is malformed");
    std::optional<shape::PortDirection> direction =
        shape::symbolizeDirection(role.getValue());
    if (!direction)
      return sampleError("a kernel argument binding has unknown role '" +
                         role.getValue().str() + "'");
    if (static_cast<size_t>(port.getInt()) >= contract.ports.size())
      return sampleError("a kernel argument binding names contract port " +
                         std::to_string(port.getInt()) + ", out of range");
    ports.push_back({static_cast<size_t>(port.getInt()), *direction});
  }
  return ports;
}

/// Replaces every dimension query of `argument` with the contract's concrete
/// extent for the sampled axis.
llvm::Error resolveArgumentDims(mlir::Value argument, const ArgumentPort &bound,
                                const shape::ShapeContract &contract,
                                const shape::ShapeValues &values,
                                mlir::OpBuilder &builder) {
  const shape::ShapePort &port = contract.ports[bound.port];
  if (port.direction != bound.direction)
    return sampleError("argument role disagrees with contract port '" +
                       port.name + "'");

  SmallVector<mlir::Operation *> dims;
  for (mlir::Operation *user : argument.getUsers()) {
    if (mlir::isa<tensor::DimOp>(user) || mlir::isa<memref::DimOp>(user))
      dims.push_back(user);
  }

  for (mlir::Operation *operation : dims) {
    mlir::Value index;
    if (auto dim = mlir::dyn_cast<tensor::DimOp>(operation))
      index = dim.getIndex();
    else
      index = mlir::cast<memref::DimOp>(operation).getIndex();

    std::optional<int64_t> axis = mlir::getConstantIntValue(index);
    if (!axis)
      return sampleError("a dimension query has a non-constant axis");
    if (*axis < 0 || static_cast<size_t>(*axis) >= port.shape.size())
      return sampleError("a dimension query names axis " +
                         std::to_string(*axis) + " of rank-" +
                         std::to_string(port.shape.size()) + " port '" +
                         port.name + "'");

    const shape::Dimension &dimension = port.shape[*axis];
    int64_t extent = 0;
    if (dimension.isConstant()) {
      extent = dimension.getConstant();
    } else {
      auto found = values.find(dimension.getSymbol().str());
      if (found == values.end())
        return sampleError("symbol '" + dimension.getSymbol().str() +
                           "' is not bound");
      extent = found->second;
    }

    builder.setInsertionPoint(operation);
    auto constant =
        arith::ConstantIndexOp::create(builder, operation->getLoc(), extent);
    constant->setAttr("micro.shape_only", builder.getUnitAttr());
    operation->getResult(0).replaceAllUsesWith(constant.getResult());
    operation->erase();
  }
  return llvm::Error::success();
}

} // namespace

shape::ShapeContract
toShapeContract(const workload::WorkloadContract &contract) {
  shape::ShapeContract described;
  described.symbols = contract.symbols;
  described.ports.reserve(contract.ports.size());
  for (const workload::WorkloadPort &port : contract.ports)
    described.ports.push_back({port.name, port.direction, port.shape});
  return described;
}

llvm::Expected<shape::ShapeContract> shapeContractOf(mlir::Operation *kernel) {
  if (!kernel)
    return sampleError("a kernel is required to describe its shape contract");
  auto encoded =
      kernel->getAttrOfType<mlir::DictionaryAttr>("micro.workload_contract");
  if (!encoded)
    return sampleError("kernel has no micro.workload_contract; shape symbols "
                       "cannot be named");
  llvm::Expected<workload::WorkloadContract> contract =
      workload::decodeWorkloadContract(encoded);
  if (!contract)
    return contract.takeError();
  return toShapeContract(*contract);
}

llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>>
materializeShapeSample(mlir::ModuleOp source,
                       const shape::ShapeContract &contract,
                       const shape::ShapeValues &values) {
  if (llvm::Error error = shape::validateShapeValues(contract, values))
    return sampleError(llvm::toString(std::move(error)));

  mlir::OwningOpRef<mlir::ModuleOp> clone(
      mlir::cast<mlir::ModuleOp>(source->clone()));
  mlir::OpBuilder builder(clone->getContext());

  WalkResult walk = clone->walk([&](mlir::Operation *kernel) -> WalkResult {
    if (kernel->getName().getStringRef() != "micro.kernel")
      return WalkResult::advance();
    if (kernel->getNumRegions() == 0 || kernel->getRegion(0).empty())
      return WalkResult::advance();

    mlir::Block &body = kernel->getRegion(0).front();
    llvm::Expected<SmallVector<ArgumentPort>> ports =
        argumentPorts(kernel, body.getNumArguments(), contract);
    if (!ports) {
      kernel->emitError(llvm::toString(ports.takeError()));
      return WalkResult::interrupt();
    }
    for (size_t index = 0; index < body.getNumArguments(); ++index) {
      if (llvm::Error error =
              resolveArgumentDims(body.getArgument(index), (*ports)[index],
                                  contract, values, builder)) {
        kernel->emitError(llvm::toString(std::move(error)));
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  if (walk.wasInterrupted())
    return sampleError("the shape sample could not be applied to every "
                       "micro.kernel");

  // Fold the guards, `ceildiv` trip counts and sampled dimension constants so
  // the shared analysis sees the concrete iteration structure. The clone is
  // analysis-only; the executable source keeps its runtime dimensions.
  mlir::PassManager passes(clone->getContext());
  passes.addPass(mlir::createCanonicalizerPass());
  if (mlir::failed(passes.run(*clone)))
    return sampleError("folding the sampled shape computations failed");
  return std::move(clone);
}

} // namespace mlir::llk::mapping
