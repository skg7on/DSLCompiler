//===- TargetSVE.cpp - AArch64 SVE feature detection ---------------------===//
//
// Part of the LLK Compiler Project
//
//===----------------------------------------------------------------------===//

#include "LLK/Target/AArch64/TargetSVE.h"

#include <fstream>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#include <sstream>
#include <string>

namespace llk::aarch64 {

std::string CpuFeatures::toString() const {
  std::ostringstream os;
  os << "AArch64:";
  if (neon)
    os << " NEON";
  if (sve)
    os << " SVE";
  if (sve2)
    os << " SVE2";
  if (bf16)
    os << " BF16";
  if (i8mm)
    os << " I8MM";
  if (sve_vector_bits)
    os << " VL=" << sve_vector_bits;
  if (!neon && !sve)
    os << " (no SIMD detected)";
  return os.str();
}

CpuFeatures detectCPUFeatures() {
  CpuFeatures features;

#if defined(__aarch64__)
  // NEON is mandatory in ARMv8-A.
  features.neon = true;

#if defined(__linux__)
  // Parse /proc/cpuinfo Features line for SVE/SVE2/BF16/I8MM flags.
  std::ifstream cpuinfo("/proc/cpuinfo");
  std::string line;
  while (std::getline(cpuinfo, line)) {
    if (line.rfind("Features", 0) == 0 || line.rfind("features", 0) == 0) {
      if (line.find(" sve2 ") != std::string::npos ||
          line.find("\tsve2\t") != std::string::npos)
        features.sve2 = true;
      if (line.find(" sve ") != std::string::npos ||
          line.find("\tsve\t") != std::string::npos)
        bool isaSupported(CpuIsa isa) {
          auto features = detectCPUFeatures();

          switch (isa) {
          case CpuIsa::NEON:
            return features.neon;
          case CpuIsa::SVE:
            // SVE implies NEON on AArch64.
            return features.sve;
          case CpuIsa::UNKNOWN:
            return false;
          default:
            // Non-AArch64 ISAs (x86) are not supported on this target.
            return false;
          }
        }
#elif defined(__APPLE__)
  // macOS: NEON is always available on Apple Silicon. SVE is not exposed
  // to userspace on current Apple hardware (M1-M4). Probe via sysctl as a
  // best-effort forward-looking check.
  int val = 0;
  size_t size = sizeof(val);
  if (sysctlbyname("hw.optional.arm.FEAT_SVE", &val, &size, nullptr, 0) == 0 &&
      val)
    features.sve = true;
  val = 0;
  size = sizeof(val);
  if (sysctlbyname("hw.optional.arm.FEAT_SVE2", &val, &size, nullptr, 0) == 0 &&
      val)
    features.sve2 = true;
  // BF16 and I8MM sysctl names are not yet published; leave as false.
#endif

      // SVE2 implies base SVE — ensure consistency.
      if (features.sve2)
        features.sve = true;
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
