//===- MappedInvocation.cpp - Typed mapped-invocation buffers -------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Runtime/MappedInvocation.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <limits>
#include <string>

namespace llk {
namespace {
llvm::Error invocationError(std::string message) {
  return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s",
                                 message.c_str());
}

bool checkedMul(uint64_t lhs, uint64_t rhs, uint64_t &result) {
  if (rhs && lhs > std::numeric_limits<uint64_t>::max() / rhs)
    return false;
  result = lhs * rhs;
  return true;
}

struct TypeInfo {
  const char *name;
  uint64_t bytes;
};

TypeInfo getTypeInfo(InvocationElementType type) {
  switch (type) {
  case InvocationElementType::F32:
    return {"f32", 4};
  case InvocationElementType::BF16:
    return {"bf16", 2};
  case InvocationElementType::F16:
    return {"f16", 2};
  case InvocationElementType::I32:
    return {"i32", 4};
  case InvocationElementType::I8:
    return {"i8", 1};
  }
  return {nullptr, 0};
}

struct Span {
  uintptr_t begin;
  uintptr_t end;
};

bool overlaps(Span lhs, Span rhs) {
  return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

/// Version 1's exact descriptor checks, kept verbatim so older artifacts keep
/// their behavior: positive static extents, a non-null allocation, offset zero
/// and compact row-major strides.
llvm::Expected<Span> validateVersion1Buffer(const KernelAbi::Port &port,
                                            const InvocationBuffer2D &buffer,
                                            llvm::StringRef kind,
                                            size_t index) {
  TypeInfo info = getTypeInfo(buffer.elementType);
  if (!info.name)
    return invocationError(
        (kind + " " + llvm::Twine(index) + " has an unsupported dtype tag")
            .str());
  if (port.elementType != info.name)
    return invocationError((kind + " " + llvm::Twine(index) +
                            " dtype does not match the kernel ABI")
                               .str());

  const MemRef2D &desc = buffer.descriptor;
  if (desc.size0 != port.shape[0] || desc.size1 != port.shape[1])
    return invocationError((kind + " " + llvm::Twine(index) +
                            " extent does not match the kernel ABI")
                               .str());
  if (!desc.allocated || !desc.aligned)
    return invocationError((kind + " " + llvm::Twine(index) +
                            " has a null allocation or aligned pointer")
                               .str());
  if (desc.offset != 0)
    return invocationError(
        (kind + " " + llvm::Twine(index) + " descriptor offset must be zero")
            .str());
  if (desc.stride0 != desc.size1 || desc.stride1 != 1)
    return invocationError(
        (kind + " " + llvm::Twine(index) + " strides must be compact row-major")
            .str());

  uint64_t elements = 0;
  uint64_t requiredBytes = 0;
  if (!checkedMul(static_cast<uint64_t>(desc.size0),
                  static_cast<uint64_t>(desc.size1), elements) ||
      !checkedMul(elements, info.bytes, requiredBytes))
    return invocationError(
        (kind + " " + llvm::Twine(index) + " range size overflow").str());

  uintptr_t allocated = reinterpret_cast<uintptr_t>(desc.allocated);
  uintptr_t aligned = reinterpret_cast<uintptr_t>(desc.aligned);
  if (buffer.allocationBytes >
      std::numeric_limits<uintptr_t>::max() - allocated)
    return invocationError(
        (kind + " " + llvm::Twine(index) + " allocation range overflow").str());
  uintptr_t allocationEnd = allocated + buffer.allocationBytes;
  if (aligned % info.bytes != 0 || aligned < allocated ||
      aligned > allocationEnd)
    return invocationError((kind + " " + llvm::Twine(index) +
                            " aligned pointer is outside its allocation range")
                               .str());
  if (requiredBytes > std::numeric_limits<uintptr_t>::max() - aligned)
    return invocationError(
        (kind + " " + llvm::Twine(index) + " touched range overflow").str());
  uintptr_t end = aligned + requiredBytes;
  if (end > allocationEnd)
    return invocationError((kind + " " + llvm::Twine(index) +
                            " touched range exceeds allocation size")
                               .str());
  return Span{aligned, end};
}

/// The concrete extents a symbolic descriptor supplies, or the field-specific
/// diagnostic that stops the invocation before entry.
llvm::Expected<std::vector<int64_t>>
validateSymbolicExtents(const KernelAbi::Port &port,
                        const InvocationBuffer2D &buffer, llvm::StringRef kind,
                        size_t index) {
  TypeInfo info = getTypeInfo(buffer.elementType);
  if (!info.name)
    return invocationError(
        (kind + " " + llvm::Twine(index) + " has an unsupported dtype tag")
            .str());
  if (port.elementType != info.name)
    return invocationError((kind + " " + llvm::Twine(index) +
                            " dtype does not match the kernel ABI")
                               .str());

  const MemRef2D &desc = buffer.descriptor;
  // Negative extents are rejected as signed values: converting first would
  // wrap them into enormous unsigned sizes and turn a malformed descriptor
  // into a plausible-looking one.
  if (desc.size0 < 0 || desc.size1 < 0)
    return invocationError((kind + " " + llvm::Twine(index) +
                            " extent is negative and is not a valid shape")
                               .str());
  if (desc.offset != 0)
    return invocationError(
        (kind + " " + llvm::Twine(index) + " descriptor offset must be zero")
            .str());
  if (desc.stride0 != desc.size1 || desc.stride1 != 1)
    return invocationError(
        (kind + " " + llvm::Twine(index) + " strides must be compact row-major")
            .str());
  return std::vector<int64_t>{desc.size0, desc.size1};
}

/// Checks the bytes a symbolic descriptor can actually address and returns its
/// touched span. An empty descriptor is genuinely empty: a null pointer with
/// zero bytes is accepted only when the extent touches nothing.
llvm::Expected<Span> validateSymbolicRange(const InvocationBuffer2D &buffer,
                                           llvm::StringRef kind, size_t index,
                                           uint64_t requiredBytes) {
  TypeInfo info = getTypeInfo(buffer.elementType);
  const MemRef2D &desc = buffer.descriptor;
  uintptr_t allocated = reinterpret_cast<uintptr_t>(desc.allocated);
  uintptr_t aligned = reinterpret_cast<uintptr_t>(desc.aligned);

  if (requiredBytes == 0) {
    if (!desc.allocated || !desc.aligned) {
      if (desc.allocated || desc.aligned || buffer.allocationBytes != 0)
        return invocationError((kind + " " + llvm::Twine(index) +
                                " empty extent must use a null pointer with "
                                "zero bytes")
                                   .str());
      return Span{aligned, aligned};
    }
  } else if (!desc.allocated || !desc.aligned) {
    return invocationError((kind + " " + llvm::Twine(index) +
                            " has a null allocation or aligned pointer")
                               .str());
  }

  if (buffer.allocationBytes >
      std::numeric_limits<uintptr_t>::max() - allocated)
    return invocationError(
        (kind + " " + llvm::Twine(index) + " allocation range overflow").str());
  uintptr_t allocationEnd = allocated + buffer.allocationBytes;
  if (aligned % info.bytes != 0 || aligned < allocated ||
      aligned > allocationEnd)
    return invocationError((kind + " " + llvm::Twine(index) +
                            " aligned pointer is outside its allocation range")
                               .str());
  if (requiredBytes > std::numeric_limits<uintptr_t>::max() - aligned)
    return invocationError(
        (kind + " " + llvm::Twine(index) + " touched range overflow").str());
  uintptr_t end = aligned + requiredBytes;
  if (end > allocationEnd)
    return invocationError((kind + " " + llvm::Twine(index) +
                            " touched range exceeds allocation size")
                               .str());
  return Span{aligned, end};
}

llvm::Error validateVersion1(const KernelAbi &abi,
                             llvm::ArrayRef<InvocationBuffer2D> inputs,
                             llvm::ArrayRef<InvocationBuffer2D> outputs) {
  if (inputs.size() != abi.inputs.size())
    return invocationError("input arity does not match the kernel ABI");
  if (outputs.size() != abi.outputs.size())
    return invocationError("output arity does not match the kernel ABI");

  llvm::SmallVector<Span, 12> inputSpans;
  llvm::SmallVector<Span, 12> outputSpans;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto span = validateVersion1Buffer(abi.inputs[i], inputs[i], "input", i);
    if (!span)
      return span.takeError();
    inputSpans.push_back(*span);
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto span = validateVersion1Buffer(abi.outputs[i], outputs[i], "output", i);
    if (!span)
      return span.takeError();
    for (size_t inputIndex = 0; inputIndex < inputSpans.size(); ++inputIndex)
      if (overlaps(inputSpans[inputIndex], *span))
        return invocationError(("output " + llvm::Twine(i) +
                                " overlaps input " + llvm::Twine(inputIndex))
                                   .str());
    for (size_t outputIndex = 0; outputIndex < outputSpans.size();
         ++outputIndex)
      if (overlaps(outputSpans[outputIndex], *span))
        return invocationError(("output " + llvm::Twine(i) +
                                " overlaps output " + llvm::Twine(outputIndex))
                                   .str());
    outputSpans.push_back(*span);
  }
  return llvm::Error::success();
}

/// The shape contract the ABI's version 2 ports describe. Port names are the
/// descriptor labels the diagnostics already use; symbol names and constraints
/// are the artifact's own.
::mlir::llk::shape::ShapeContract buildShapeContract(const KernelAbi &abi) {
  using ::mlir::llk::shape::PortDirection;
  ::mlir::llk::shape::ShapeContract contract;
  contract.symbols = abi.symbols;
  auto append = [&contract](const std::vector<KernelAbi::Port> &ports,
                            PortDirection direction, llvm::StringRef kind) {
    for (size_t i = 0; i < ports.size(); ++i) {
      ::mlir::llk::shape::ShapePort port;
      port.name = (kind + llvm::Twine(i)).str();
      port.direction = direction;
      if (!ports[i].dimensions.empty())
        port.shape = ports[i].dimensions;
      else
        for (int64_t extent : ports[i].shape)
          port.shape.emplace_back(extent);
      contract.ports.push_back(std::move(port));
    }
  };
  append(abi.inputs, PortDirection::Input, "input");
  append(abi.outputs, PortDirection::Output, "output");
  return contract;
}
} // namespace

llvm::Error validateKernelAbi(const KernelAbi &abi) {
  if (abi.version != kMappedAbiVersionV1 && abi.version != kMappedAbiVersionV2)
    return invocationError("unsupported mapped ABI version " +
                           std::to_string(abi.version));
  if (abi.inputs.size() > kMappedMaxDescriptors ||
      abi.outputs.size() > kMappedMaxDescriptors - abi.inputs.size())
    return invocationError("kernel ABI has more than 12 descriptors");

  if (!isSymbolicKernelAbi(abi)) {
    // Version 1 carried no symbolic or binding metadata; its envelope is
    // exactly the static one it has always validated.
    auto validatePorts = [](const std::vector<KernelAbi::Port> &ports,
                            llvm::StringRef kind) -> llvm::Error {
      for (size_t i = 0; i < ports.size(); ++i) {
        const auto &port = ports[i];
        if (port.shape.size() != 2)
          return invocationError(
              (kind + " " + llvm::Twine(i) + " must have rank two").str());
        if (port.shape[0] <= 0 || port.shape[1] <= 0)
          return invocationError((kind + " " + llvm::Twine(i) +
                                  " extents must be positive and static")
                                     .str());
        if (port.elementType != "f32" && port.elementType != "bf16" &&
            port.elementType != "f16" && port.elementType != "i32" &&
            port.elementType != "i8")
          return invocationError((kind + " " + llvm::Twine(i) +
                                  " has unsupported element type '" +
                                  port.elementType + "'")
                                     .str());
      }
      return llvm::Error::success();
    };
    if (llvm::Error error = validatePorts(abi.inputs, "input"))
      return error;
    return validatePorts(abi.outputs, "output");
  }

  // Version 2. Every port is rank two with symbolic or nonnegative constant
  // extents, and the symbol table is validated by the same plain shape layer
  // the workload contract uses.
  auto validatePorts = [](const std::vector<KernelAbi::Port> &ports,
                          llvm::StringRef kind) -> llvm::Error {
    for (size_t i = 0; i < ports.size(); ++i) {
      const auto &port = ports[i];
      bool rankTwo = port.dimensions.empty() ? port.shape.size() == 2
                                             : port.dimensions.size() == 2;
      if (!rankTwo)
        return invocationError(
            (kind + " " + llvm::Twine(i) + " must have rank two").str());
      if (port.elementType != "f32" && port.elementType != "bf16" &&
          port.elementType != "f16" && port.elementType != "i32" &&
          port.elementType != "i8")
        return invocationError((kind + " " + llvm::Twine(i) +
                                " has unsupported element type '" +
                                port.elementType + "'")
                                   .str());
    }
    return llvm::Error::success();
  };
  if (llvm::Error error = validatePorts(abi.inputs, "input"))
    return error;
  if (llvm::Error error = validatePorts(abi.outputs, "output"))
    return error;

  ::mlir::llk::shape::ShapeContract contract = buildShapeContract(abi);
  if (llvm::Error error = ::mlir::llk::shape::validateShapeContract(contract))
    return error;

  if (abi.arguments.empty())
    return invocationError(
        "a symbolic kernel ABI must bind every port to a signature argument");

  // Every port is bound exactly once, and every result names an output port
  // bound by an output argument. Signature order is the callable order.
  std::vector<bool> boundInputs(abi.inputs.size(), false);
  std::vector<bool> boundOutputs(abi.outputs.size(), false);
  for (size_t i = 0; i < abi.arguments.size(); ++i) {
    const KernelAbiArgument &argument = abi.arguments[i];
    std::vector<bool> &bound =
        argument.role == AbiPortRole::Input ? boundInputs : boundOutputs;
    if (argument.port >= bound.size())
      return invocationError(
          ("argument " + llvm::Twine(i) + " binds an out-of-range port").str());
    if (bound[argument.port])
      return invocationError(("argument " + llvm::Twine(i) +
                              " rebinds a port that another argument owns")
                                 .str());
    bound[argument.port] = true;
  }
  for (size_t i = 0; i < boundInputs.size(); ++i)
    if (!boundInputs[i])
      return invocationError(("input port " + llvm::Twine(i) +
                              " is not bound to a signature argument")
                                 .str());
  for (size_t i = 0; i < boundOutputs.size(); ++i)
    if (!boundOutputs[i])
      return invocationError(("output port " + llvm::Twine(i) +
                              " is not bound to a caller-owned output argument")
                                 .str());
  for (uint32_t port : abi.results)
    if (port >= abi.outputs.size() || !boundOutputs[port])
      return invocationError(("a result names output port " +
                              llvm::Twine(port) +
                              ", which no output argument binds")
                                 .str());
  return llvm::Error::success();
}

uint64_t computeKernelAbiHash(const KernelAbi &abi) {
  // Stable FNV-1a over explicit little-endian integer encodings. The fixed
  // contract bytes version the compact layout, zero offset and alias policy.
  uint64_t hash = 14695981039346656037ULL;
  auto byte = [&hash](uint8_t value) {
    hash ^= value;
    hash *= 1099511628211ULL;
  };
  auto integer = [&byte](uint64_t value) {
    for (unsigned i = 0; i < 8; ++i)
      byte(static_cast<uint8_t>(value >> (i * 8)));
  };
  auto text = [&integer, &byte](llvm::StringRef value) {
    integer(value.size());
    for (unsigned char character : value)
      byte(character);
  };
  auto ports = [&integer, &byte](const std::vector<KernelAbi::Port> &values) {
    integer(values.size());
    for (const auto &port : values) {
      integer(port.shape.size());
      for (int64_t extent : port.shape)
        integer(static_cast<uint64_t>(extent));
      integer(port.elementType.size());
      for (unsigned char character : port.elementType)
        byte(character);
    }
  };
  integer(abi.version);
  if (!isSymbolicKernelAbi(abi)) {
    integer(0x434f4d5041435431ULL); // compact row-major, offset zero
    integer(0x524f5f4449534a31ULL); // read-only inputs, disjoint outputs
    ports(abi.inputs);
    ports(abi.outputs);
    return hash;
  }

  // Version 2 tags constants and symbols apart, length-delimits every name and
  // appends the binding and result tables. A v2 artifact can never collide
  // with a v1 artifact that happens to carry the same concrete extents.
  integer(0x4142495f56325f31ULL); // "ABI_V2_1"
  integer(0x53594d424f4c4943ULL); // symbolic extents
  integer(0x4f55544f574e4552ULL); // caller-owned output destinations
  const std::map<std::string, ::mlir::llk::shape::SymbolConstraint> &symbols =
      abi.symbols;
  integer(symbols.size());
  for (const auto &[name, constraint] : symbols) {
    text(name);
    integer(static_cast<uint64_t>(constraint.min));
    integer(constraint.max ? 1 : 0);
    integer(constraint.max ? static_cast<uint64_t>(*constraint.max) : 0);
    integer(static_cast<uint64_t>(constraint.multipleOf));
  }
  auto symbolicPorts = [&](const std::vector<KernelAbi::Port> &values) {
    integer(values.size());
    for (const auto &port : values) {
      text(port.elementType);
      if (port.dimensions.empty()) {
        integer(port.shape.size());
        for (int64_t extent : port.shape)
          integer(static_cast<uint64_t>(extent));
        continue;
      }
      integer(port.dimensions.size());
      for (const ::mlir::llk::shape::Dimension &dimension : port.dimensions) {
        if (dimension.isConstant()) {
          byte(0);
          integer(static_cast<uint64_t>(dimension.getConstant()));
        } else {
          byte(1);
          text(dimension.getSymbol());
        }
      }
    }
  };
  symbolicPorts(abi.inputs);
  symbolicPorts(abi.outputs);
  integer(abi.arguments.size());
  for (const KernelAbiArgument &argument : abi.arguments) {
    byte(argument.role == AbiPortRole::Input ? 0 : 1);
    integer(argument.port);
  }
  integer(abi.results.size());
  for (uint32_t port : abi.results)
    integer(port);
  return hash;
}

llvm::Expected<::mlir::llk::shape::ShapeValues>
resolveMappedShapeValues(const KernelAbi &abi,
                         llvm::ArrayRef<InvocationBuffer2D> inputs,
                         llvm::ArrayRef<InvocationBuffer2D> outputs) {
  if (llvm::Error error = validateKernelAbi(abi))
    return std::move(error);
  if (!isSymbolicKernelAbi(abi)) {
    if (llvm::Error error = validateVersion1(abi, inputs, outputs))
      return std::move(error);
    return ::mlir::llk::shape::ShapeValues{};
  }

  if (inputs.size() != abi.inputs.size())
    return invocationError("input arity does not match the kernel ABI");
  if (outputs.size() != abi.outputs.size())
    return invocationError("output arity does not match the kernel ABI");

  using ::mlir::llk::shape::PortDirection;
  using ::mlir::llk::shape::PortShapeValues;
  llvm::SmallVector<Span, 12> inputSpans;
  llvm::SmallVector<Span, 12> outputSpans;
  llvm::SmallVector<PortShapeValues, 12> extents;

  auto validateSide = [&](llvm::ArrayRef<InvocationBuffer2D> buffers,
                          const std::vector<KernelAbi::Port> &ports,
                          PortDirection direction,
                          llvm::StringRef kind) -> llvm::Error {
    for (size_t i = 0; i < buffers.size(); ++i) {
      auto shape = validateSymbolicExtents(ports[i], buffers[i], kind, i);
      if (!shape)
        return shape.takeError();
      TypeInfo info = getTypeInfo(buffers[i].elementType);
      uint64_t elements = 0;
      uint64_t requiredBytes = 0;
      if (!checkedMul(static_cast<uint64_t>((*shape)[0]),
                      static_cast<uint64_t>((*shape)[1]), elements) ||
          !checkedMul(elements, info.bytes, requiredBytes))
        return invocationError(
            (kind + " " + llvm::Twine(i) + " range size overflow").str());
      auto span = validateSymbolicRange(buffers[i], kind, i, requiredBytes);
      if (!span)
        return span.takeError();
      PortShapeValues entry;
      entry.direction = direction;
      entry.index = static_cast<uint32_t>(i);
      entry.extents = std::move(*shape);
      extents.push_back(std::move(entry));
      (direction == PortDirection::Input ? inputSpans : outputSpans)
          .push_back(*span);
    }
    return llvm::Error::success();
  };
  if (llvm::Error error =
          validateSide(inputs, abi.inputs, PortDirection::Input, "input"))
    return std::move(error);
  if (llvm::Error error =
          validateSide(outputs, abi.outputs, PortDirection::Output, "output"))
    return std::move(error);

  for (size_t i = 0; i < outputSpans.size(); ++i) {
    for (size_t inputIndex = 0; inputIndex < inputSpans.size(); ++inputIndex)
      if (overlaps(inputSpans[inputIndex], outputSpans[i]))
        return invocationError(("output " + llvm::Twine(i) +
                                " overlaps input " + llvm::Twine(inputIndex))
                                   .str());
    for (size_t outputIndex = 0; outputIndex < i; ++outputIndex)
      if (overlaps(outputSpans[outputIndex], outputSpans[i]))
        return invocationError(("output " + llvm::Twine(i) +
                                " overlaps output " + llvm::Twine(outputIndex))
                                   .str());
  }

  // The canonical shape layer binds every symbol, checks repeated uses for
  // equality, and enforces the declared constraints. Its diagnostic names the
  // field that failed; the surrounding descriptor context says which buffer.
  ::mlir::llk::shape::ShapeContract contract = buildShapeContract(abi);
  llvm::Expected<::mlir::llk::shape::ShapeValues> values =
      ::mlir::llk::shape::bindShapeValues(contract, extents);
  if (!values)
    return invocationError("descriptor extents conflict with the kernel ABI: " +
                           llvm::toString(values.takeError()));
  return values;
}

llvm::Error
validateMappedInvocation(const KernelAbi &abi,
                         llvm::ArrayRef<InvocationBuffer2D> inputs,
                         llvm::ArrayRef<InvocationBuffer2D> outputs) {
  llvm::Expected<::mlir::llk::shape::ShapeValues> values =
      resolveMappedShapeValues(abi, inputs, outputs);
  if (!values)
    return values.takeError();
  return llvm::Error::success();
}
} // namespace llk
