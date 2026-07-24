//===- TargetSVE.cpp - AArch64 SVE feature detection ---------------------===//
//
// Part of the LLK Compiler Project
//
//===----------------------------------------------------------------------===//

#include "LLK/Target/AArch64/TargetSVE.h"

#include <sstream>
#include <string>

namespace llk::aarch64 {

std::string CpuFeatures::toString() const {
  std::ostringstream os;
  os << "AArch64:";
  if (neon) os << " NEON";
  if (sve) os << " SVE";
  if (sve2) os << " SVE2";
  if (bf16) os << " BF16";
  if (i8mm) os << " I8MM";
  if (sve_vector_bits)
    os << " VL=" << sve_vector_bits;
  if (!neon && !sve)
    os << " (no SIMD detected)";
  return os.str();
}

CpuFeatures detectCPUFeatures() {
  CpuFeatures features;

#if defined(__aarch64__)
  // On an actual AArch64 host, read CPU features.
  // NEON is mandatory in ARMv8-A, so always set.
  features.neon = true;

  // Read /proc/cpuinfo for Linux; use sysctl on macOS.
  // For now, return the stub unconditionally.
  //
  // TODO: Parse /proc/cpuinfo Features line for "sve", "sve2",
  //       "bf16", "i8mm". On macOS, use sysctlbyname("hw.optional.arm.FEAT_SVE").
  //
  // Placeholder: assume no SVE for initial skeleton.
  features.sve = false;
  features.sve2 = false;
  features.bf16 = false;
  features.i8mm = false;
#endif

  return features;
}

bool isaSupported(CpuIsa isa) {
  auto features = detectCPUFeatures();

  switch (isa) {
  case CpuIsa::NEON:
    return features.neon;
  case CpuIsa::SVE:
    return features.sve;
  default:
    // Only NEON and SVE are AArch64 ISAs.
    return false;
  }
}

} // namespace llk::aarch64
