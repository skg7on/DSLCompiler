//===- registry.cpp - Tests for the LLK kernel registry -------------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/LLKKernel/LLKKernelLowering.h"
#include "gtest/gtest.h"
using namespace mlir;
namespace {
class Dummy : public llk::LLKKernelLowering {
  std::string name;

public:
  explicit Dummy(std::string name) : name(std::move(name)) {}
  StringRef getOpName() const override { return name; }
  LogicalResult inferContract(Operation *,
                              llk::KernelContract &) const override {
    return failure();
  }
};
TEST(LLKKernelRegistryTest, LookupOrderCapabilitiesAndStableOwnership) {
  llk::LLKKernelRegistry registry;
  EXPECT_EQ(registry.lookup("missing"), nullptr);
  registry.add<Dummy>("first");
  auto *first = registry.lookup("first");
  ASSERT_NE(first, nullptr);
  EXPECT_FALSE(first->hasLinalgLowering());
  EXPECT_FALSE(first->hasMicroLowering());
  for (int i = 0; i < 100; ++i)
    registry.add<Dummy>(std::to_string(i));
  EXPECT_EQ(first, registry.lookup("first"));
  EXPECT_EQ(first, registry.entries().front());
  EXPECT_EQ(registry.entries()[1]->getOpName(), "0");
  EXPECT_EQ(registry.entries().size(), 101u);
  registry.add<Dummy>("first");
  EXPECT_EQ(registry.entries().size(), 101u);
  EXPECT_EQ(registry.lookup("first"), first);
}
} // namespace

TEST(LLKKernelRegistryTest, PopulatedSwiGLU) {
  llk::LLKKernelRegistry registry;
  llk::populateLLKKernelLowerings(registry);
  auto *entry = registry.lookup("llk.fused_swiglu");
  ASSERT_NE(entry, nullptr);
  EXPECT_TRUE(entry->hasLinalgLowering());
}

TEST(LLKKernelRegistryTest, PopulatedLinalgOnlyKernels) {
  llk::LLKKernelRegistry registry;
  llk::populateLLKKernelLowerings(registry);
  for (auto name : {"llk.rope", "llk.attention"}) {
    auto *entry = registry.lookup(name);
    ASSERT_NE(entry, nullptr);
    EXPECT_TRUE(entry->hasLinalgLowering());
    EXPECT_FALSE(entry->hasMicroLowering());
  }
}
TEST(LLKKernelRegistryTest, MicroCapabilitiesAndCompatibility) {
  llk::LLKKernelRegistry r;
  llk::populateLLKKernelLowerings(r);
  for (auto name : {"llk.fused_swiglu", "llk.matmul", "linalg.matmul"}) {
    auto *entry = r.lookup(name);
    ASSERT_NE(entry, nullptr);
    EXPECT_TRUE(entry->hasMicroLowering());
  }
  EXPECT_FALSE(r.lookup("linalg.matmul")->hasLinalgLowering());
}
