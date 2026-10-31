//===- dynamic_shape_replay.cpp - Replay identity and variant selection ---===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dynamic-shape execution identity has two levels (design §7):
//
//   * a compiled variant is the reusable code -- source content, symbolic
//     contract and version, numerical contract, guard domain, ABI version and
//     hash, schedule, target/backend and compiler versions;
//   * a measurement is that variant plus the concrete shape observation (or
//     the profile that selected it) and the measurement protocol.
//
// The baseline variant accepts the whole declared domain, so different
// invocations reuse its code. They must never reuse each other's latency.
// Optional guarded variants are ordered: the first matching guard wins, with
// the baseline as the final choice, all bounded by an explicit budget.

#include "LLK/Perf/ShapeProfile.h"
#include "LLK/Runtime/CompiledVariantIdentity.h"
#include "LLK/Runtime/JitCache.h"
#include "LLK/Runtime/KernelAbi.h"
#include "LLK/Shape/ShapeContract.h"

#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

#include <string>
#include <vector>

namespace {

namespace shape = ::mlir::llk::shape;
namespace perf = ::mlir::llk::perf;

llk::CompiledVariantIdentityInputs baselineInputs() {
  llk::CompiledVariantIdentityInputs inputs;
  inputs.sourceContent = "module { func.func @matmul_dynamic_mnk }";
  inputs.symbolicContract = "shape-contract{M,N,K}";
  inputs.symbolicContractVersion = 2;
  inputs.numericalContract = "math-mode=strict";
  inputs.abiVersion = llk::kMappedAbiVersionV2;
  inputs.abiHash = 0x1234;
  inputs.staticSchedule = "plan=1|binding=2";
  inputs.targetIdentity = "x86-avx2|machine=target-hash";
  inputs.backendIdentity = "compiler=test|backend=selected-target";
  inputs.executionMode = "selected-target";
  inputs.compilerVersion = "test-compiler";
  return inputs;
}

shape::ShapeValues values(int64_t M, int64_t N, int64_t K) {
  return {{"M", M}, {"N", N}, {"K", K}};
}

shape::ShapePredicate divisible(llvm::StringRef symbol, int64_t value) {
  return {shape::ShapePredicate::Kind::DivisibleBy, symbol.str(), value};
}

shape::ShapePredicate atLeast(llvm::StringRef symbol, int64_t value) {
  return {shape::ShapePredicate::Kind::AtLeast, symbol.str(), value};
}

shape::ShapePredicate equalTo(llvm::StringRef symbol, int64_t value) {
  return {shape::ShapePredicate::Kind::EqualTo, symbol.str(), value};
}

void noopKernel(MemRef2D *, MemRef2D *, MemRef2D *, MemRef2D *, MemRef2D *) {}

llk::GuardedVariant baselineVariant() {
  return {"baseline",
          llk::makeCompiledVariantIdentity(baselineInputs()),
          {},
          /*baseline=*/true};
}

llk::GuardedVariant guardedVariant(llvm::StringRef id,
                                   std::vector<shape::ShapePredicate> guards) {
  llk::CompiledVariantIdentityInputs inputs = baselineInputs();
  inputs.guards = guards;
  return {id.str(), llk::makeCompiledVariantIdentity(inputs), std::move(guards),
          /*baseline=*/false};
}

} // namespace

// The baseline covers every valid shape. Two unrelated invocations must select
// the same compiled variant and share its cached code.
TEST(DynamicShapeReplay, DifferentShapesReuseBaseline) {
  llk::GuardedVariant baseline = baselineVariant();
  llvm::Expected<llk::DynamicVariantSet> set =
      llk::DynamicVariantSet::create({baseline}, /*maxVariants=*/1);
  ASSERT_TRUE(bool(set)) << llvm::toString(set.takeError());

  const llk::GuardedVariant &small = set->select(values(1, 1, 1));
  const llk::GuardedVariant &large = set->select(values(17, 65, 67));
  EXPECT_EQ(small.id, "baseline");
  EXPECT_EQ(large.id, "baseline");
  EXPECT_EQ(small.identity.canonical, large.identity.canonical);
  EXPECT_FALSE(small.identity.contentHash.empty());

  // Exact shapes are not part of the baseline executable cache key: inserting
  // the variant once serves both invocations.
  llk::JitCache cache;
  EXPECT_FALSE(cache.lookupVariantObjectCode(small.identity).has_value());
  cache.insertVariantObjectCode(small.identity, &noopKernel);
  EXPECT_TRUE(cache.lookupVariantObjectCode(small.identity).has_value());
  EXPECT_TRUE(cache.lookupVariantObjectCode(large.identity).has_value());
}

// Reusing code never means reusing a measured latency. The measurement identity
// binds the same compiled variant to the concrete shape or profile.
TEST(DynamicShapeReplay, DifferentShapesDoNotShareMeasurements) {
  llk::CompiledVariantIdentity variant =
      llk::makeCompiledVariantIdentity(baselineInputs());

  const std::string small =
      llk::canonicalMeasurementIdentity(variant.canonical, values(1, 1, 1));
  const std::string large =
      llk::canonicalMeasurementIdentity(variant.canonical, values(17, 65, 67));
  EXPECT_NE(small, large);
  EXPECT_EQ(small, llk::canonicalMeasurementIdentity(variant.canonical,
                                                     values(1, 1, 1)));
  EXPECT_EQ(large, llk::canonicalMeasurementIdentity(variant.canonical,
                                                     values(17, 65, 67)));

  perf::ShapeProfile first;
  first.samples.push_back({values(7, 13, 5), 1.0});
  perf::ShapeProfile second = first;
  second.samples.push_back({values(8, 32, 32), 1.0});
  const std::string firstProfile = perf::canonicalShapeProfile(first);
  const std::string secondProfile = perf::canonicalShapeProfile(second);
  EXPECT_NE(
      llk::canonicalMeasurementIdentity(variant.canonical, {}, firstProfile),
      llk::canonicalMeasurementIdentity(variant.canonical, {}, secondProfile));
}

// The guard domain is part of compiled-variant identity: adding, removing or
// changing a guard changes the code's identity even when the source is the
// same.
TEST(DynamicShapeReplay, GuardChangesInvalidate) {
  llk::CompiledVariantIdentityInputs plain = baselineInputs();
  llk::CompiledVariantIdentityInputs guarded = baselineInputs();
  guarded.guards = {divisible("M", 8)};

  llk::CompiledVariantIdentity plainIdentity =
      llk::makeCompiledVariantIdentity(plain);
  llk::CompiledVariantIdentity guardedIdentity =
      llk::makeCompiledVariantIdentity(guarded);
  EXPECT_NE(plainIdentity.canonical, guardedIdentity.canonical);
  EXPECT_NE(plainIdentity.contentHash, guardedIdentity.contentHash);

  guarded.guards[0].value = 16;
  EXPECT_NE(guardedIdentity.canonical,
            llk::makeCompiledVariantIdentity(guarded).canonical);

  guarded.guards.push_back(atLeast("N", 32));
  EXPECT_NE(llk::makeCompiledVariantIdentity(guarded).canonical,
            guardedIdentity.canonical);
}

// A changed profile invalidates the stored measurement/selection, while the
// compiled variant itself is unchanged.
TEST(DynamicShapeReplay, ProfileChangesInvalidateSelection) {
  llk::CompiledVariantIdentity variant =
      llk::makeCompiledVariantIdentity(baselineInputs());

  perf::ShapeProfile profile;
  profile.samples.push_back({values(7, 13, 5), 1.0});
  perf::ShapeProfile changed = profile;
  changed.samples[0].weight = 2.0;
  ASSERT_NE(perf::canonicalShapeProfile(profile),
            perf::canonicalShapeProfile(changed));

  const std::string original = llk::canonicalMeasurementIdentity(
      variant.canonical, {}, perf::canonicalShapeProfile(profile));
  const std::string updated = llk::canonicalMeasurementIdentity(
      variant.canonical, {}, perf::canonicalShapeProfile(changed));
  EXPECT_NE(original, updated);

  // Only the observation changed; the reusable code identity did not.
  EXPECT_EQ(variant.canonical,
            llk::makeCompiledVariantIdentity(baselineInputs()).canonical);
}

// ABI versions and hashes are part of compiled-variant identity. Version 1
// artifacts stay readable and keep a distinct identity from version 2.
TEST(DynamicShapeReplay, AbiVersionsDiffer) {
  llk::CompiledVariantIdentityInputs v1 = baselineInputs();
  v1.abiVersion = llk::kMappedAbiVersionV1;
  v1.abiHash = 0x11;
  llk::CompiledVariantIdentityInputs v2 = baselineInputs();
  v2.abiVersion = llk::kMappedAbiVersionV2;
  v2.abiHash = 0x22;

  const std::string v1Identity = llk::makeCompiledVariantIdentity(v1).canonical;
  const std::string v2Identity = llk::makeCompiledVariantIdentity(v2).canonical;
  EXPECT_NE(v1Identity, v2Identity);

  v2.abiHash = v1.abiHash;
  EXPECT_NE(v1Identity, llk::makeCompiledVariantIdentity(v2).canonical);
}

// The packed-weight cache binds actual K/N as well as the existing content and
// lifetime identity. A mismatched shape never hits a packed buffer.
TEST(DynamicShapeReplay, PackedShapeMismatch) {
  std::vector<float> storage(64 * 128, 1.0f);
  const Tensor2D weight{storage.data(), 64, 128, 128, 1};
  const llk::WeightKey key = makeWeightKey(weight, /*BK=*/64);
  EXPECT_EQ(key.K, 64);
  EXPECT_EQ(key.N, 128);

  llk::JitCache cache;
  cache.insertPackedWeights(key, "packed_blob");
  EXPECT_TRUE(cache.lookupPackedWeights(key).has_value());

  // Same allocation, different actual extents: a different packed buffer.
  const Tensor2D narrower{storage.data(), 32, 128, 128, 1};
  const llk::WeightKey mismatch = makeWeightKey(narrower, /*BK=*/64);
  EXPECT_NE(key, mismatch);
  EXPECT_FALSE(cache.lookupPackedWeights(mismatch).has_value());

  // Different lifetime (allocation) with the same extents is also distinct.
  std::vector<float> other(64 * 128, 1.0f);
  const Tensor2D otherAllocation{other.data(), 64, 128, 128, 1};
  EXPECT_FALSE(
      cache.lookupPackedWeights(makeWeightKey(otherAllocation, /*BK=*/64))
          .has_value());
}

// Frozen replay keeps the declared variant order: the first matching guard
// wins, the baseline is the final choice, and the canonical form is stable.
TEST(DynamicShapeReplay, FrozenOrderStable) {
  llk::GuardedVariant fast = guardedVariant("fast", {divisible("M", 8)});
  // Overlaps `fast` at M=16, but does not match the uncovered shape below.
  llk::GuardedVariant alsoFast =
      guardedVariant("also-fast", {equalTo("M", 16)});
  llk::GuardedVariant baseline = baselineVariant();

  llvm::Expected<llk::DynamicVariantSet> first =
      llk::DynamicVariantSet::create({fast, alsoFast, baseline}, 3);
  ASSERT_TRUE(bool(first)) << llvm::toString(first.takeError());
  llvm::Expected<llk::DynamicVariantSet> second =
      llk::DynamicVariantSet::create({fast, alsoFast, baseline}, 3);
  ASSERT_TRUE(bool(second)) << llvm::toString(second.takeError());
  EXPECT_EQ(first->canonical(), second->canonical());
  EXPECT_EQ(first->contentHash(), second->contentHash());

  // Overlapping guards: declaration order decides, deterministically.
  EXPECT_EQ(first->select(values(16, 32, 32)).id, "fast");
  EXPECT_EQ(first->select(values(17, 65, 67)).id, "baseline");
  EXPECT_EQ(first->select(values(8, 32, 32)).id, "fast");

  // Reordering the same variants changes both the frozen form and selection;
  // order is semantic, not incidental.
  llvm::Expected<llk::DynamicVariantSet> reordered =
      llk::DynamicVariantSet::create({alsoFast, fast, baseline}, 3);
  ASSERT_TRUE(bool(reordered)) << llvm::toString(reordered.takeError());
  EXPECT_NE(first->canonical(), reordered->canonical());
  EXPECT_NE(first->contentHash(), reordered->contentHash());
  EXPECT_EQ(reordered->select(values(16, 32, 32)).id, "also-fast");
}

// The declared budget bounds optional specialization. A set may not exceed it,
// and it must contain exactly one full-domain baseline in final position.
TEST(DynamicShapeReplay, VariantBudgetBounded) {
  llk::GuardedVariant baseline = baselineVariant();
  llk::GuardedVariant fast = guardedVariant("fast", {divisible("M", 8)});

  EXPECT_FALSE(
      bool(llk::DynamicVariantSet::create({baseline}, /*maxVariants=*/0)));
  EXPECT_TRUE(
      bool(llk::DynamicVariantSet::create({baseline}, /*maxVariants=*/1)));

  llvm::Expected<llk::DynamicVariantSet> overBudget =
      llk::DynamicVariantSet::create({fast, baseline}, /*maxVariants=*/1);
  ASSERT_FALSE(bool(overBudget));
  EXPECT_NE(llvm::toString(overBudget.takeError()).find("budget"),
            std::string::npos);

  // Exactly the budget is accepted.
  EXPECT_TRUE(bool(llk::DynamicVariantSet::create({fast, baseline},
                                                  /*maxVariants=*/2)));

  // Structural invariants: one baseline, and it is the final choice.
  EXPECT_FALSE(bool(llk::DynamicVariantSet::create({baseline, baseline}, 2)));
  EXPECT_FALSE(bool(llk::DynamicVariantSet::create({baseline, fast}, 2)));
  llk::GuardedVariant unguarded = baselineVariant();
  unguarded.id = "unguarded";
  unguarded.baseline = false;
  EXPECT_FALSE(bool(llk::DynamicVariantSet::create({unguarded, baseline}, 2)));
}
