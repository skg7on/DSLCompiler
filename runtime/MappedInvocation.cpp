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

llvm::Expected<Span> validateBuffer(const KernelAbi::Port &port,
                                    const InvocationBuffer2D &buffer,
                                    llvm::StringRef kind, size_t index) {
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

bool overlaps(Span lhs, Span rhs) {
  return lhs.begin < rhs.end && rhs.begin < lhs.end;
}
} // namespace

llvm::Error validateKernelAbi(const KernelAbi &abi) {
  if (abi.version != kMappedAbiVersion)
    return invocationError("unsupported mapped ABI version " +
                           std::to_string(abi.version));
  if (abi.inputs.size() > kMappedMaxDescriptors ||
      abi.outputs.size() > kMappedMaxDescriptors - abi.inputs.size())
    return invocationError("kernel ABI has more than 12 descriptors");
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
  integer(0x434f4d5041435431ULL); // compact row-major, offset zero
  integer(0x524f5f4449534a31ULL); // read-only inputs, disjoint outputs
  ports(abi.inputs);
  ports(abi.outputs);
  return hash;
}

llvm::Error
validateMappedInvocation(const KernelAbi &abi,
                         llvm::ArrayRef<InvocationBuffer2D> inputs,
                         llvm::ArrayRef<InvocationBuffer2D> outputs) {
  if (llvm::Error error = validateKernelAbi(abi))
    return error;
  if (inputs.size() != abi.inputs.size())
    return invocationError("input arity does not match the kernel ABI");
  if (outputs.size() != abi.outputs.size())
    return invocationError("output arity does not match the kernel ABI");

  llvm::SmallVector<Span, 12> inputSpans;
  llvm::SmallVector<Span, 12> outputSpans;
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto span = validateBuffer(abi.inputs[i], inputs[i], "input", i);
    if (!span)
      return span.takeError();
    inputSpans.push_back(*span);
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto span = validateBuffer(abi.outputs[i], outputs[i], "output", i);
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
} // namespace llk
