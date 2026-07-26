//===- TargetSVE.h - AArch64 SVE feature detection -----------------------===//
//
// CPU feature queries for AArch64 Scalable Vector Extension.
// Stub implementation for post-M6 target expansion.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_TARGET_AARCH64_TARGETSVE_H
#define LLK_TARGET_AARCH64_TARGETSVE_H

#include "LLK/Runtime/KernelKey.h" // for CpuIsa

#include <cstdint>
#include <string>

namespace llk::aarch64 {

/// CPU feature flags for AArch64 targets.
struct CpuFeatures {
  bool neon{false};   // Advanced SIMD (NEON)
  bool sve{false};    // Scalable Vector Extension (SVE)
  bool sve2{false};   // SVE2
  bool bf16{false};   // BF16 support (BFMMLA/BFDOT or SVE BF16)
  bool i8mm{false};   // 8-bit integer matrix multiply
  uint32_t sve_vector_bits{0}; // 0 = unknown/minimum, 128/256/512 = known VL

  /// Summary string for logging.
  std::string toString() const;
};

/// Detect CPU features by reading /proc/cpuinfo or sysctl on macOS.
/// Returns best-effort detection; returns empty features struct if
/// not running on AArch64.
CpuFeatures detectCPUFeatures();

/// Check whether a given ISA is supported by the detected CPU.
/// For cross-compilation scenarios, returns true for NEON if SVE is
/// requested (NEON is mandatory in ARMv8-A).
bool isaSupported(CpuIsa isa);

} // namespace llk::aarch64

#endif // LLK_TARGET_AARCH64_TARGETSVE_H
