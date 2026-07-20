//===- TargetAVX2.cpp - CPU feature detection implementation --------------===//
//
// Part of the LLK Compiler Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Detects x86-64 AVX2+FMA via CPUID leaf 1 / leaf 7, and AArch64 NEON/SVE
/// via /proc/cpuinfo or HWCAP (arm64 macOS or Linux).
///
//===----------------------------------------------------------------------===//

#include "LLK/Target/X86/TargetAVX2.h"

#ifdef __x86_64__
#include <cpuid.h>
#endif

#ifdef __aarch64__
#if defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#elif defined(__APPLE__)
// macOS arm64: NEON is always available (ARMv8 mandatory feature).
// For SVE we could use sysctlbyname("hw.optional.arm.FEAT_SVE", ...),
// but SVE detection is deferred — this stub assumes NEON only.
#include <sys/sysctl.h>
#endif
#endif

namespace llk {

CpuFeatures CpuFeatures::detect() {
  CpuFeatures f;

#ifdef __x86_64__
  unsigned int eax, ebx, ecx, edx;

  // Leaf 1: basic feature flags
  if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
    f.fma = (ecx >> 12) & 1; // FMA is in leaf 1, ECX bit 12
  }

  // Leaf 7 (subleaf 0): extended features
  if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
    f.avx2 = (ebx >> 5) & 1;      // AVX2 is in leaf 7, EBX bit 5
    f.avx512f = (ebx >> 16) & 1;  // AVX-512F is in leaf 7, EBX bit 16
    f.amx_bf16 = (edx >> 22) & 1; // AMX-BF16 is in leaf 7, EDX bit 22
  }

  // Leaf 7 (subleaf 1): AVX-512 BF16
  if (__get_cpuid_count(7, 1, &eax, &ebx, &ecx, &edx)) {
    f.avx512bf16 = (eax >> 5) & 1; // AVX-512 BF16 is in leaf 7.1, EAX bit 5
  }

#elif defined(__aarch64__)
  // NEON is mandatory on ARMv8+; assume available.
  f.neon = true;

#if defined(__linux__)
  unsigned long hwcaps = getauxval(AT_HWCAP);
  if (hwcaps & HWCAP_SVE)
    f.sve = true;
#elif defined(__APPLE__)
  // macOS: check for SVE via sysctl
  int sve_supported = 0;
  size_t sve_size = sizeof(sve_supported);
  if (sysctlbyname("hw.optional.arm.FEAT_SVE", &sve_supported, &sve_size,
                   nullptr, 0) == 0 &&
      sve_supported)
    f.sve = true;
#endif
#endif

  return f;
}

CpuIsa CpuFeatures::bestIsa() const {
  if (amx_bf16)
    return CpuIsa::AMX_BF16;
  if (avx512bf16)
    return CpuIsa::AVX512_BF16;
  if (avx512f)
    return CpuIsa::AVX512_VNNI;
  if (avx2 && fma)
    return CpuIsa::AVX2;
  if (sve)
    return CpuIsa::SVE;
  if (neon)
    return CpuIsa::NEON;
  return CpuIsa::UNKNOWN;
}

std::string CpuFeatures::toString() const {
  std::string s;
  if (avx2)
    s += "avx2 ";
  if (fma)
    s += "fma ";
  if (avx512f)
    s += "avx512f ";
  if (avx512bf16)
    s += "avx512bf16 ";
  if (amx_bf16)
    s += "amx_bf16 ";
  if (neon)
    s += "neon ";
  if (sve)
    s += "sve ";
  return s.empty() ? "none" : s;
}

} // namespace llk
