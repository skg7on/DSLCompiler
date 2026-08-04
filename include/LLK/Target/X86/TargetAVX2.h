//===- TargetAVX2.h - CPU feature detection for x86-64 AVX2 ----*- C++ -*-===//
//
// Part of the LLK Compiler Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// CPU feature detection via CPUID (x86-64) or HWCAP (AArch64).
/// Primary focus: detecting AVX2+FMA for the M2 explicit-vector microkernel.
///
//===----------------------------------------------------------------------===//

#ifndef LLK_TARGET_X86_TARGETAVX2_H
#define LLK_TARGET_X86_TARGETAVX2_H

#include <cstdint>
#include <string>

#include "LLK/Runtime/KernelKey.h"

namespace llk {

struct CpuFeatures {
  bool avx2 = false;
  bool fma = false;
  bool avx512f = false;
  bool avx512bf16 = false;
  bool amx_bf16 = false;
  bool neon = false;
  bool sve = false;

  static CpuFeatures detect();
  CpuIsa bestIsa() const;
  std::string toString() const;
};

} // namespace llk

#endif
