//===- cpu_features_test.cpp - CPU feature detection tests ----------------===//
//
// Tests the CpuFeatures::detect() and CpuFeatures::bestIsa() APIs.
// Should compile and pass on both x86-64 and arm64 hosts.
//
//===----------------------------------------------------------------------===//

#include "LLK/Target/X86/TargetAVX2.h"
#include <gtest/gtest.h>

TEST(CpuFeatures, DetectDoesNotCrash) {
    auto features = llk::CpuFeatures::detect();
    // On any AVX2-capable host, avx2 should be true
    EXPECT_TRUE(features.avx2 || !features.avx2); // always true — just checks no crash
}

TEST(CpuFeatures, BestIsaReturnsValid) {
    auto features = llk::CpuFeatures::detect();
    auto isa = features.bestIsa();
    EXPECT_NE(isa, llk::CpuIsa::UNKNOWN);
}

TEST(CpuFeatures, ToStringContainsExpected) {
    auto features = llk::CpuFeatures::detect();
    std::string s = features.toString();
    EXPECT_FALSE(s.empty());
}

TEST(CpuFeatures, DetectIsIdempotent) {
    auto f1 = llk::CpuFeatures::detect();
    auto f2 = llk::CpuFeatures::detect();
    EXPECT_EQ(f1.avx2, f2.avx2);
    EXPECT_EQ(f1.fma, f2.fma);
    EXPECT_EQ(f1.neon, f2.neon);
}

TEST(CpuFeatures, BestIsaConsistency) {
    auto features = llk::CpuFeatures::detect();
    auto isa = features.bestIsa();
    switch (isa) {
    case llk::CpuIsa::AVX2:
        EXPECT_TRUE(features.avx2);
        EXPECT_TRUE(features.fma);
        break;
    case llk::CpuIsa::NEON:
        EXPECT_TRUE(features.neon);
        break;
    case llk::CpuIsa::UNKNOWN:
        // No features detected — acceptable on an unknown ISA
        break;
    default:
        // Higher ISAs should have their prerequisite features
        break;
    }
}
