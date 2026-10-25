//===- backend_pipeline_hooks.cpp - Target pipeline ordering -------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Target/AArch64/Mapping/SVEMappingTarget.h"

#include "arm_target_fixture.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include <gtest/gtest.h>

#include <memory>

namespace {

struct ProbeState {
  bool ran = false;
  bool sawMemref = false;
  bool sawLinalg = false;
};

class ProbePass
    : public mlir::PassWrapper<ProbePass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ProbePass)
  explicit ProbePass(ProbeState *state) : state_(state) {}
  mlir::StringRef getArgument() const final { return "llk-test-hook-probe"; }

  void runOnOperation() override {
    state_->ran = true;
    getOperation().walk([&](mlir::Operation *op) {
      if (mlir::isa<mlir::linalg::LinalgOp>(op))
        state_->sawLinalg = true;
      for (mlir::Value value : op->getOperands())
        state_->sawMemref |= mlir::isa<mlir::MemRefType>(value.getType());
      for (mlir::Value value : op->getResults())
        state_->sawMemref |= mlir::isa<mlir::MemRefType>(value.getType());
    });
  }

private:
  ProbeState *state_;
};

/// Leaves one benign tensor `linalg.generic` in the module so the probe has
/// open Linalg work to observe after bufferization. The arm-sve selected
/// backend consumes its own Linalg group before bufferization, so without this
/// marker `convert-linalg-to-loops` would have nothing to act on and the hook's
/// position relative to it would be unobservable. The marker is dead code: it
/// is not part of any selected group and does not affect the realized count.
class MarkerLinalgPass
    : public mlir::PassWrapper<MarkerLinalgPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MarkerLinalgPass)
  mlir::StringRef getArgument() const final { return "llk-test-linalg-marker"; }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect>();
  }

  void runOnOperation() override {
    mlir::func::FuncOp function;
    getOperation().walk([&](mlir::func::FuncOp candidate) {
      if (!function)
        function = candidate;
    });
    if (!function)
      return;
    mlir::OpBuilder builder(function.getBody());
    builder.setInsertionPointToStart(&function.getBody().front());
    auto type = mlir::RankedTensorType::get({4}, builder.getF32Type());
    mlir::Value empty = mlir::tensor::EmptyOp::create(
        builder, function.getLoc(), type.getShape(), type.getElementType());
    llvm::SmallVector<mlir::AffineMap> maps{builder.getMultiDimIdentityMap(1),
                                            builder.getMultiDimIdentityMap(1)};
    llvm::SmallVector<mlir::utils::IteratorType> iterators{
        mlir::utils::IteratorType::parallel};
    mlir::linalg::GenericOp::create(
        builder, function.getLoc(), mlir::TypeRange{type},
        mlir::ValueRange{empty}, mlir::ValueRange{empty}, maps, iterators,
        [](mlir::OpBuilder &nested, mlir::Location loc, mlir::ValueRange args) {
          mlir::linalg::YieldOp::create(nested, loc, args[0]);
        });
  }
};

/// Delegates to the real arm-sve target and records where the second hook runs.
class HookProbeTarget : public mlir::llk::mapping::MappingTarget {
public:
  HookProbeTarget(std::unique_ptr<mlir::llk::mapping::MappingTarget> inner,
                  ProbeState *state)
      : inner_(std::move(inner)), state_(state) {}

  mlir::StringRef name() const override { return inner_->name(); }
  const mlir::llk::machine::MachineModel &machine() const override {
    return inner_->machine();
  }
  const mlir::llk::mapping::LayoutRegistry &layouts() const override {
    return inner_->layouts();
  }
  const mlir::llk::mapping::RuleRegistry &rules() const override {
    return inner_->rules();
  }
  bool isKnownEmitter(mlir::StringRef key) const override {
    return inner_->isKnownEmitter(key);
  }
  std::optional<mlir::llk::mapping::TargetCodegenRequirements>
  codegenRequirements() const override {
    return inner_->codegenRequirements();
  }
  const mlir::llk::mapping::LatencyProvider *latencyProvider() const override {
    return inner_->latencyProvider();
  }
  std::unique_ptr<mlir::llk::mapping::TargetEmitter>
  createEmitter(mlir::StringRef key) const override {
    return inner_->createEmitter(key);
  }
  std::unique_ptr<mlir::llk::mapping::TargetEmitter>
  createEmitter() const override {
    return inner_->createEmitter();
  }
  llvm::Error
  buildSelectedBackendPipeline(mlir::OpPassManager &pm) const override {
    if (llvm::Error error = inner_->buildSelectedBackendPipeline(pm))
      return error;
    pm.addPass(std::make_unique<MarkerLinalgPass>());
    return llvm::Error::success();
  }
  llvm::Error buildPostBufferizationBackendPipeline(
      mlir::OpPassManager &pm) const override {
    pm.addPass(std::make_unique<ProbePass>(state_));
    return llvm::Error::success();
  }

private:
  std::unique_ptr<mlir::llk::mapping::MappingTarget> inner_;
  ProbeState *state_;
};

TEST(BackendPipelineHooks, PostBufferizationHookSeesMemrefsAndOpenLinalg) {
  auto fixture = arm_test::mapFixture(
      "arm-sve", "test/Conversion/MicroMapping/Inputs/arm/vector_add_f32.mlir");
  ASSERT_TRUE(static_cast<bool>(fixture))
      << llvm::toString(fixture.takeError());

  ProbeState state;
  HookProbeTarget probe(std::move(fixture->target), &state);
  mlir::OwningOpRef<mlir::ModuleOp> source =
      mlir::cast<mlir::ModuleOp>(fixture->source->clone());
  llk::MappedCompileOptions options;
  options.entrySymbol = "arm_vector_add_f32";
  options.backend = llk::MappedBackend::SelectedTarget;
  options.stop = llk::MappedStop::Lowered;
  llvm::Expected<llk::MappedCompilation> compiled =
      llk::compileMappedKernel(*source, probe, fixture->plan, options);
  ASSERT_TRUE(static_cast<bool>(compiled))
      << llvm::toString(compiled.takeError());
  EXPECT_TRUE(state.ran) << "the post-bufferization hook never ran";
  EXPECT_TRUE(state.sawMemref) << "the hook ran before bufferization";
  EXPECT_TRUE(state.sawLinalg) << "the hook ran after convert-linalg-to-loops";
}

} // namespace
