//===- arm_sve_backend.cpp - Selected SVE lowering -----------------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Framework E10 (issue #184): the `arm-sve` target is executable for exactly
// one subset -- f32 `micro.vector "add"` and `"mul"`. These tests prove the
// three properties that make that claim meaningful:
//
//  * the two arithmetic emitters declare a lowering and the target publishes
//    the AArch64/SVE code-generation contract, while every other key (and the
//    contraction signature) keeps the explicit rejection;
//  * a selected add compiles with no reference-lowered group and reaches
//    scalable SVE Vector IR;
//  * a selected contraction still refuses to compile, so "executable subset"
//    does not silently widen to the whole target;
//  * an unimplemented dtype and an emitter-only operation are refused by the
//    backend's own accounting, and a capability receipt reports that boundary
//    from the artifacts rather than from the rule table.

#include "LLK/Conversion/FrameworkCapabilities.h"
#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Conversion/MappedKernelAbi.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Mapping/MappingLowering.h"
#include "LLK/Target/AArch64/Mapping/SVEMappingTarget.h"

#include "arm_target_fixture.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace {

constexpr const char *kVectorAdd =
    "test/Conversion/MicroMapping/Inputs/arm/vector_add_f32.mlir";
constexpr const char *kVectorMulFull =
    "test/Conversion/MicroMapping/Inputs/arm/vector_mul_f32_full.mlir";
constexpr const char *kVectorBf16Add =
    "test/Conversion/MicroMapping/Inputs/arm/vector_add_bf16.mlir";
constexpr const char *kVectorI8Mul =
    "test/Conversion/MicroMapping/Inputs/arm/vector_mul_i8.mlir";
constexpr const char *kVectorSiluF32 =
    "test/Conversion/MicroMapping/Inputs/arm/vector_silu_f32.mlir";
constexpr const char *kContraction =
    "test/Conversion/MicroMapping/Inputs/arm/contraction_f32.mlir";

TEST(ArmSVEBackend, OnlyArithmeticKeysLower) {
  auto fixture = arm_test::mapFixture("arm-sve", kVectorAdd);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());
  EXPECT_TRUE(fixture->target->createEmitter("sve_vector_add")->hasLowering());
  EXPECT_TRUE(fixture->target->createEmitter("sve_vector_mul")->hasLowering());
  EXPECT_FALSE(fixture->target->createEmitter("sve_mma")->hasLowering());
  EXPECT_FALSE(fixture->target->createEmitter("sve_reduce")->hasLowering());
  auto requirements = fixture->target->codegenRequirements();
  ASSERT_TRUE(requirements.has_value());
  EXPECT_EQ(requirements->architecture, "aarch64");
  EXPECT_EQ(requirements->requiredFeatures, std::vector<std::string>{"sve"});
}

TEST(ArmSVEBackend, SelectedVectorAddVectorizesToScalableSVE) {
  auto fixture = arm_test::mapFixture("arm-sve", kVectorAdd);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());
  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::cast<mlir::ModuleOp>(fixture->source->clone());
  llk::MappedCompileOptions options;
  options.entrySymbol = "arm_vector_add_f32";
  options.backend = llk::MappedBackend::SelectedTarget;
  options.stop = llk::MappedStop::Lowered;
  llvm::Expected<llk::MappedCompilation> compiled = llk::compileMappedKernel(
      *source, *fixture->target, fixture->plan, options);
  ASSERT_TRUE(static_cast<bool>(compiled))
      << llvm::toString(compiled.takeError());
  EXPECT_GT(compiled->backendGroupsRealized, 0u);
  EXPECT_EQ(compiled->referenceGroupsLowered, 0u);
  std::string text;
  llvm::raw_string_ostream stream(text);
  compiled->module->print(stream);
  EXPECT_NE(text.find("vector<8x[4]xf32>"), std::string::npos) << text;
}

// A row of an 8x8 f32 tile is half of the modeled 512-bit vector, so the
// scalable form needs a mask. A 16x16 row is exactly one 512-bit vector: it is
// the full-width shape the executable subset must also realize, and it lowers
// to the same `vector<16x[4]xf32>` form without any fallback group.
TEST(ArmSVEBackend, SelectedFullWidthVectorMulVectorizesToScalableSVE) {
  auto fixture = arm_test::mapFixture("arm-sve", kVectorMulFull);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());
  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::cast<mlir::ModuleOp>(fixture->source->clone());
  llk::MappedCompileOptions options;
  options.entrySymbol = "arm_vector_mul_f32_full";
  options.backend = llk::MappedBackend::SelectedTarget;
  options.stop = llk::MappedStop::Lowered;
  llvm::Expected<llk::MappedCompilation> compiled = llk::compileMappedKernel(
      *source, *fixture->target, fixture->plan, options);
  ASSERT_TRUE(static_cast<bool>(compiled))
      << llvm::toString(compiled.takeError());
  EXPECT_GT(compiled->backendGroupsRealized, 0u);
  EXPECT_EQ(compiled->referenceGroupsLowered, 0u);
  std::string text;
  llvm::raw_string_ostream stream(text);
  compiled->module->print(stream);
  EXPECT_NE(text.find("vector<16x[4]xf32>"), std::string::npos) << text;
}

TEST(ArmSVEBackend, ContractionStillHasNoSelectedLowering) {
  auto fixture = arm_test::mapFixture("arm-sve", kContraction);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());
  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::cast<mlir::ModuleOp>(fixture->source->clone());
  llk::MappedCompileOptions options;
  options.entrySymbol = "arm_contraction";
  options.backend = llk::MappedBackend::SelectedTarget;
  options.stop = llk::MappedStop::Lowered;
  llvm::Expected<llk::MappedCompilation> compiled = llk::compileMappedKernel(
      *source, *fixture->target, fixture->plan, options);
  EXPECT_FALSE(static_cast<bool>(compiled));
}

// The executable subset is a dtype boundary, not a key boundary. A reference
// compilation is the portable path and must still lower the bf16 and i8 rules
// the same keys cover; the per-group capability query keeps them on the
// reference bridge instead of letting the f32-only emitter fail the run.
TEST(ArmSVEBackend, ReferenceBackendKeepsUnimplementedDtypesPortable) {
  struct Case {
    const char *path;
    const char *entry;
  };
  for (const Case &testCase : {Case{kVectorBf16Add, "arm_vector_add_bf16"},
                               Case{kVectorI8Mul, "arm_vector_mul_i8"}}) {
    SCOPED_TRACE(testCase.path);
    auto fixture = arm_test::mapFixture("arm-sve", testCase.path);
    ASSERT_TRUE(static_cast<bool>(fixture))
        << llvm::toString(fixture.takeError());
    mlir::OwningOpRef<mlir::ModuleOp> source =
        mlir::cast<mlir::ModuleOp>(fixture->source->clone());
    llk::MappedCompileOptions options;
    options.entrySymbol = testCase.entry;
    options.backend = llk::MappedBackend::Reference;
    options.stop = llk::MappedStop::Lowered;
    llvm::Expected<llk::MappedCompilation> compiled = llk::compileMappedKernel(
        *source, *fixture->target, fixture->plan, options);
    ASSERT_TRUE(static_cast<bool>(compiled))
        << llvm::toString(compiled.takeError());
    // The honest accounting: a group the target cannot lower is carried by the
    // reference bridge, not quietly claimed as target code.
    EXPECT_GT(compiled->referenceGroupsLowered, 0u);
    EXPECT_EQ(compiled->targetLowered, 0u);
    EXPECT_EQ(compiled->backendGroupsRealized, 0u);
  }
}

// The same bf16 signature selected for the target is refused: the shared
// contract reports that no selected compute group was realized rather than
// silently falling back to reference lowering and calling it target execution.
TEST(ArmSVEBackend, SelectedBackendRefusesUnimplementedDtypes) {
  auto fixture = arm_test::mapFixture("arm-sve", kVectorBf16Add);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());
  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::cast<mlir::ModuleOp>(fixture->source->clone());
  llk::MappedCompileOptions options;
  options.entrySymbol = "arm_vector_add_bf16";
  options.backend = llk::MappedBackend::SelectedTarget;
  options.stop = llk::MappedStop::Lowered;
  llvm::Expected<llk::MappedCompilation> compiled = llk::compileMappedKernel(
      *source, *fixture->target, fixture->plan, options);
  ASSERT_FALSE(static_cast<bool>(compiled));
  const std::string message = llvm::toString(compiled.takeError());
  EXPECT_NE(message.find("realized no selected compute groups"),
            std::string::npos)
      << message;
}

// The executable subset is narrower than the target's rule coverage in two
// independent directions: a dtype the arithmetic emitters do not implement
// (bf16 add) and a f32 operation whose rule names an emitter that only
// verifies its bundle (SiLU). Both are refused by the selected backend's own
// group accounting -- not by falling back to the portable reference bridge and
// reporting success, which would call reference code target execution.
TEST(ArmSVEBackend, UnsupportedDtypeAndOperationAreRefused) {
  struct Case {
    const char *path;
    const char *entry;
  };
  for (const Case &testCase : {Case{kVectorBf16Add, "arm_vector_add_bf16"},
                               Case{kVectorSiluF32, "arm_vector_silu_f32"}}) {
    SCOPED_TRACE(testCase.path);
    auto fixture = arm_test::mapFixture("arm-sve", testCase.path);
    ASSERT_TRUE(static_cast<bool>(fixture))
        << llvm::toString(fixture.takeError());
    mlir::OwningOpRef<mlir::ModuleOp> source =
        mlir::cast<mlir::ModuleOp>(fixture->source->clone());
    llk::MappedCompileOptions options;
    options.entrySymbol = testCase.entry;
    options.backend = llk::MappedBackend::SelectedTarget;
    options.stop = llk::MappedStop::Lowered;
    llvm::Expected<llk::MappedCompilation> compiled = llk::compileMappedKernel(
        *source, *fixture->target, fixture->plan, options);
    ASSERT_FALSE(static_cast<bool>(compiled));
    const std::string message = llvm::toString(compiled.takeError());
    EXPECT_NE(message.find("selected-target backend realized no selected "
                           "compute groups"),
              std::string::npos)
        << message;
    EXPECT_EQ(message.find("reference"), std::string::npos)
        << "the refusal is reported as a reference fallback: " << message;
  }
}

// A capability receipt states what the artifacts establish, not what the
// target's rule table promises. Before any selected compilation runs, the f32
// add plan leaves selected lowering `not_checked` -- even though both of its
// emitter keys implement a lowering. The SiLU plan is different: its only
// compute emitter verifies bundles and never lowers, so the receipt can report
// `unsupported` statically, without running a backend. After the f32 add
// compilation, selected lowering is credited and the unexercised stages stay
// `not_checked`.
TEST(ArmSVEBackend, CapabilityReceiptDrawsTheExecutableBoundary) {
  using mlir::llk::framework::CapabilityReceipt;
  using mlir::llk::framework::CapabilityStatus;

  auto fixture = arm_test::mapFixture("arm-sve", kVectorAdd);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());
  mlir::Operation *kernel = nullptr;
  fixture->source->walk([&](mlir::Operation *op) {
    if (!kernel && op->getName().getStringRef() == "micro.kernel")
      kernel = op;
  });
  ASSERT_NE(kernel, nullptr);
  // A direct MicroIR fixture carries no workload contract; the ABI stages stay
  // `not_checked` rather than being guessed from the signature.
  mlir::llk::workload::WorkloadContract contract;
  CapabilityReceipt planned = mlir::llk::framework::describeCapabilities(
      kernel, contract, *fixture->target, &fixture->analysis, &fixture->plan);
  EXPECT_EQ(planned.selectedLowering, CapabilityStatus::NotChecked);
  EXPECT_EQ(planned.referenceLowering, CapabilityStatus::NotChecked);
  EXPECT_EQ(planned.invocation, CapabilityStatus::NotChecked);

  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::cast<mlir::ModuleOp>(fixture->source->clone());
  llk::MappedCompileOptions options;
  options.entrySymbol = "arm_vector_add_f32";
  options.backend = llk::MappedBackend::SelectedTarget;
  options.stop = llk::MappedStop::Lowered;
  llvm::Expected<llk::MappedCompilation> compiled = llk::compileMappedKernel(
      *source, *fixture->target, fixture->plan, options);
  ASSERT_TRUE(static_cast<bool>(compiled))
      << llvm::toString(compiled.takeError());
  CapabilityReceipt realized = mlir::llk::framework::describeCapabilities(
      kernel, contract, *fixture->target, &fixture->analysis, &fixture->plan,
      &*compiled);
  EXPECT_EQ(realized.selectedLowering, CapabilityStatus::Supported);
  EXPECT_EQ(realized.referenceLowering, CapabilityStatus::NotChecked);
  EXPECT_EQ(realized.invocation, CapabilityStatus::NotChecked);

  auto silu = arm_test::mapFixture("arm-sve", kVectorSiluF32);
  ASSERT_TRUE(static_cast<bool>(silu)) << llvm::toString(silu.takeError());
  mlir::Operation *siluKernel = nullptr;
  silu->source->walk([&](mlir::Operation *op) {
    if (!siluKernel && op->getName().getStringRef() == "micro.kernel")
      siluKernel = op;
  });
  ASSERT_NE(siluKernel, nullptr);
  CapabilityReceipt refused = mlir::llk::framework::describeCapabilities(
      siluKernel, contract, *silu->target, &silu->analysis, &silu->plan);
  EXPECT_EQ(refused.selectedLowering, CapabilityStatus::Unsupported);
  const std::string reasons = llvm::join(refused.reasons, "\n");
  EXPECT_NE(reasons.find("sve_vector_silu"), std::string::npos) << reasons;
}

// The prepared artifact is what the AOT object and the JIT both consume.
// Ownership-based buffer deallocation runs there, and it inserts a constant
// into every region it traverses -- which a `vector.mask` region (exactly one
// masked operation) cannot hold. The selected pass dissolves the mask and
// rank-reduces the scalable transfers, so preparation must succeed and the
// whole-tile scalable arithmetic must survive.
TEST(ArmSVEBackend, SelectedVectorAddPreparesForInvocation) {
  auto fixture = arm_test::mapFixture("arm-sve", kVectorAdd);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());
  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::cast<mlir::ModuleOp>(fixture->source->clone());
  llk::MappedCompileOptions options;
  options.entrySymbol = "arm_vector_add_f32";
  options.backend = llk::MappedBackend::SelectedTarget;
  options.stop = llk::MappedStop::Lowered;
  llvm::Expected<llk::MappedCompilation> compiled = llk::compileMappedKernel(
      *source, *fixture->target, fixture->plan, options);
  ASSERT_TRUE(static_cast<bool>(compiled))
      << llvm::toString(compiled.takeError());

  llvm::Expected<llk::PreparedMappedKernel> prepared =
      llk::prepareMappedKernelForInvocation(*compiled->module,
                                            "arm_vector_add_f32");
  ASSERT_TRUE(static_cast<bool>(prepared))
      << llvm::toString(prepared.takeError());

  bool hasMask = false;
  prepared->module->walk([&](mlir::Operation *op) {
    hasMask |= op->getName().getStringRef() == "vector.mask";
  });
  EXPECT_FALSE(hasMask) << "a vector.mask region survives kernel preparation";
  std::string text;
  llvm::raw_string_ostream stream(text);
  prepared->module->print(stream);
  EXPECT_NE(text.find("vector<8x[4]xf32>"), std::string::npos) << text;
}

// The emitter's pre-write contract: a covered group is validated as a whole, so
// a later unsupported operation cannot leave earlier operations marked as
// handed off. The per-group capability query answers the same question without
// mutating anything, which is what keeps a portable reference run off the
// f32-only emitter.
TEST(ArmSVEBackend, EmitterValidatesTheWholeGroupBeforeMutating) {
  auto fixture = arm_test::mapFixture("arm-sve", kVectorAdd);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());

  mlir::Operation *vector = nullptr;
  mlir::Operation *other = nullptr;
  fixture->bound.module->walk([&](mlir::Operation *op) {
    llvm::StringRef name = op->getName().getStringRef();
    if (!vector && name == "micro.vector")
      vector = op;
    if (!other && name == "micro.yield")
      other = op;
  });
  ASSERT_NE(vector, nullptr);
  ASSERT_NE(other, nullptr);

  auto emitter = fixture->target->createEmitter("sve_vector_add");
  ASSERT_NE(emitter, nullptr);
  mlir::IRRewriter rewriter(fixture->context.get());
  mlir::llk::mapping::TargetLoweringContext lowering{
      fixture->target->machine(), {}, {}};
  mlir::llk::mapping::TargetBundle bundle{
      "sve.vector_add_f32", {}, "sve_vector_add"};

  EXPECT_TRUE(emitter->hasLoweringFor({vector}));
  EXPECT_FALSE(emitter->hasLoweringFor({vector, other}));
  // The capability query agrees with `lower`, which rejects an empty group: a
  // query that said "lowerable" would invite a caller to lower nothing.
  EXPECT_FALSE(emitter->hasLoweringFor({}));

  llvm::Error error =
      emitter->lower({vector, other}, bundle, lowering, rewriter);
  EXPECT_TRUE(static_cast<bool>(error));
  EXPECT_FALSE(vector->hasAttr("llk.sve.selected_handoff"))
      << "a rejected group left an earlier operation mutated";

  // The valid group lowers and is marked only after the whole group passed.
  EXPECT_FALSE(
      static_cast<bool>(emitter->lower({vector}, bundle, lowering, rewriter)));
  EXPECT_TRUE(vector->hasAttr("llk.sve.selected_handoff"));
}

// The emitter's static-shape requirement is the pass's too: a dynamic
// rank-two tile must be declined by both, so a mixed module cannot have one
// group handed off by the emitter and skipped as unrealizable by the pass.
TEST(ArmSVEBackend, EmitterDeclinesDynamicGroups) {
  auto fixture = arm_test::mapFixture("arm-sve", kVectorAdd);
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());

  mlir::MLIRContext context;
  context.loadDialect<mlir::micro::MicroDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
module {
  micro.kernel @dynamic(%a: tensor<?x8xf32>, %b: tensor<?x8xf32>)
      -> tensor<?x8xf32> {
    %ta = micro.tile_view %a
        {shape = array<i64: -9223372036854775808, 8>}
        : tensor<?x8xf32> -> !micro.tile<?x8xf32, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b
        {shape = array<i64: -9223372036854775808, 8>}
        : tensor<?x8xf32> -> !micro.tile<?x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %ta, %tb
        : !micro.tile<?x8xf32, memory = #micro.memory<sram>>,
          !micro.tile<?x8xf32, memory = #micro.memory<sram>>
        -> !micro.tile<?x8xf32, memory = #micro.memory<sram>>
    micro.yield %r : !micro.tile<?x8xf32, memory = #micro.memory<sram>>
  }
}
)mlir",
                                              &context);
  ASSERT_TRUE(module) << "dynamic-tile fixture did not parse";
  mlir::Operation *dynamicVector = nullptr;
  module->walk([&](mlir::Operation *op) {
    if (!dynamicVector && op->getName().getStringRef() == "micro.vector")
      dynamicVector = op;
  });
  ASSERT_NE(dynamicVector, nullptr);

  auto emitter = fixture->target->createEmitter("sve_vector_add");
  ASSERT_NE(emitter, nullptr);
  EXPECT_FALSE(emitter->hasLoweringFor({dynamicVector}))
      << "the emitter accepted a dynamic tile the backend pass will skip";

  mlir::IRRewriter rewriter(&context);
  mlir::llk::mapping::TargetLoweringContext lowering{
      fixture->target->machine(), {}, {}};
  mlir::llk::mapping::TargetBundle bundle{
      "sve.vector_add_f32", {}, "sve_vector_add"};
  EXPECT_TRUE(static_cast<bool>(
      emitter->lower({dynamicVector}, bundle, lowering, rewriter)));
}

// A selected group with no SSA result (the memory-out Linalg form) must be
// declined before any `getResult(0)`. The pass leaves the realized count at
// zero instead of asserting on the empty result list.
TEST(ArmSVEBackend, ZeroResultSelectedGroupIsNotRealized) {
  mlir::MLIRContext context;
  context
      .loadDialect<mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                   mlir::tensor::TensorDialect, mlir::memref::MemRefDialect>();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @zero_result(%in: memref<2x2xf32>, %out: memref<2x2xf32>) {
  linalg.generic {indexing_maps = [#map, #map],
                  iterator_types = ["parallel", "parallel"],
                  micro.op = "add",
                  micro.mapping = {emitter = "sve_vector_add"}}
      ins(%in : memref<2x2xf32>) outs(%out : memref<2x2xf32>) {
  ^bb0(%value: f32, %yielded: f32):
    linalg.yield %value : f32
  }
  return
}
)mlir",
                                              &context);
  ASSERT_TRUE(module) << "zero-result fixture did not parse";

  mlir::PassManager pm(&context);
  mlir::llk::target::sve::buildSVESelectedBackendPipeline(pm, 16);
  EXPECT_TRUE(mlir::succeeded(pm.run(*module)));
  auto realized = (*module)->getAttrOfType<mlir::IntegerAttr>(
      "llk.backend_groups_realized");
  ASSERT_TRUE(realized != nullptr);
  EXPECT_EQ(realized.getInt(), 0)
      << "a zero-result selected group was counted as realized";
}

} // namespace
