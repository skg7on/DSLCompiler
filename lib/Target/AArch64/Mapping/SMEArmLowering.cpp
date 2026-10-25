//===- SMEArmLowering.cpp - Selected ArmSME conversion -------------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The post-bufferization half of the selected Arm SME backend. It applies the
// Vector lowering patterns (permutation maps, unit-dim transfers,
// contract-to-outerproduct, masks) now that the transfers are memrefs, then
// runs the production MLIR ArmSME sequence with caller-managed streaming and a
// new ZA array per call. The sequence deliberately omits
// `arm-sme-outer-product-fusion`: fusion emits 2-way/4-way widening operations
// that are outside this SME1 acceptance.

#include "LLK/Target/AArch64/Mapping/SMEMappingTarget.h"

#include "mlir/Conversion/ArithToArmSME/ArithToArmSME.h"
#include "mlir/Conversion/ArmSMEToLLVM/ArmSMEToLLVM.h"
#include "mlir/Conversion/ArmSMEToSCF/ArmSMEToSCF.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToArmSME/VectorToArmSME.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ArmSME/IR/ArmSME.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/ArmSVE/IR/ArmSVEDialect.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Error.h"

#include <memory>

namespace mlir::llk::target::sme {
namespace {

class SMEArmLoweringPass
    : public PassWrapper<SMEArmLoweringPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SMEArmLoweringPass)
  StringRef getArgument() const final { return "llk-sme-arm-lowering"; }
  StringRef getDescription() const final {
    return "Convert selected SME vector IR to ArmSME and LLVM intrinsics";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    // A pipeline run from inside a pass does not get its passes' dependent
    // dialects loaded the way a top-level PassManager does. Every dialect the
    // nested ArmSME sequence can create must be declared here, or the created
    // operations are unregistered and pattern-created ops such as `cf.br` come
    // back as null.
    registry.insert<affine::AffineDialect, arm_sme::ArmSMEDialect,
                    arm_sve::ArmSVEDialect, arith::ArithDialect,
                    cf::ControlFlowDialect, func::FuncDialect,
                    LLVM::LLVMDialect, memref::MemRefDialect, scf::SCFDialect,
                    tensor::TensorDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto selected =
        module->getAttrOfType<IntegerAttr>("llk.sme.selected_groups");
    if (!selected || selected.getInt() == 0)
      return; // nothing was selected; the shared contract reports that

    // The pre-bufferization pass left `vector.contract` and tensor transfers.
    // Bufferization has made the transfers memrefs, so the reference ArmSME
    // pattern order applies here.
    RewritePatternSet patterns(&getContext());
    vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
    vector::populateVectorTransferDropUnitDimsPatterns(patterns);
    vector::populateVectorContractLoweringPatterns(
        patterns, vector::VectorContractLowering::OuterProduct,
        /*benefit=*/1, /*disableOuterProductLowering=*/true);
    vector::populateVectorMaskLoweringPatternsForSideEffectingOps(patterns);
    vector::populateVectorMaskOpLoweringPatterns(patterns);
    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      module.emitError(
          "SME backend could not lower the vector contraction to outer "
          "products");
      signalPassFailure();
      return;
    }

    // Convert through the point where ArmSME tile operations are live. This
    // pass deliberately stops before SCF-to-CF: the shared ownership
    // preparation that turns the module into a callable kernel only accepts
    // structured loops, while ArmSME tile allocation needs the flattened form.
    // `SMELateLoweringPass` does that conversion after preparation.
    OpPassManager toArmSME(ModuleOp::getOperationName());
    toArmSME.addPass(arm_sme::createVectorLegalizationPass());
    toArmSME.addPass(createCanonicalizerPass());
    toArmSME.addPass(createCSEPass());
    toArmSME.addPass(createArithToArmSMEConversionPass());
    toArmSME.addPass(createConvertVectorToArmSMEPass());
    toArmSME.addPass(createConvertArmSMEToSCFPass());
    toArmSME.addPass(createConvertVectorToSCFPass(
        VectorTransferToSCFOptions().enableFullUnroll()));
    toArmSME.addNestedPass<func::FuncOp>(arm_sme::createEnableArmStreamingPass(
        arm_sme::ArmStreamingMode::Streaming, arm_sme::ArmZaMode::NewZA,
        /*ifRequiredByOps=*/true));
    if (failed(runPipeline(toArmSME, module))) {
      module.emitError("SME ArmSME conversion pipeline failed");
      signalPassFailure();
      return;
    }

    // The late pass erases the high-level tile operations, so the evidence
    // that the selected contraction became SME code is checked here.
    unsigned armSMEOps = 0;
    module.walk([&](Operation *op) {
      if (op->getName().getStringRef().starts_with("arm_sme."))
        ++armSMEOps;
    });
    if (armSMEOps == 0) {
      module.emitError(
          "selected SME lowering produced no ArmSME tile operations");
      signalPassFailure();
      return;
    }

    module->setAttr("llk.backend_groups_realized",
                    IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                     selected.getInt()));
  }
};

/// The late half of the selected Arm SME backend.
///
/// The shared ABI/ownership preparation runs between the two halves: it clones
/// the lowered module, makes results borrowed outputs and runs ownership-based
/// buffer deallocation, which refuses any function with a backedge ("Only
/// structured control-flow loops are supported"). ArmSME tile allocation, on
/// the other hand, is only correct after `convert-scf-to-cf` has flattened the
/// loops. This pass therefore owns the flattening and the ArmSME-to-LLVM
/// conversion, and runs after preparation has finished with the structured
/// form.
class SMELateLoweringPass
    : public PassWrapper<SMELateLoweringPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SMELateLoweringPass)
  StringRef getArgument() const final { return "llk-sme-late-lowering"; }
  StringRef getDescription() const final {
    return "Flatten selected SME control flow and convert ArmSME to LLVM";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    // As in the post-bufferization pass, the nested pipeline is run from
    // inside a pass, so every dialect it can create must be declared here or
    // the operations are created unregistered.
    registry
        .insert<affine::AffineDialect, arm_sme::ArmSMEDialect,
                arith::ArithDialect, cf::ControlFlowDialect, func::FuncDialect,
                LLVM::LLVMDialect, memref::MemRefDialect, scf::SCFDialect,
                tensor::TensorDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto selected =
        module->getAttrOfType<IntegerAttr>("llk.sme.selected_groups");
    if (!selected || selected.getInt() == 0)
      return; // nothing selected; the shared contract already reported it

    OpPassManager toLLVM(ModuleOp::getOperationName());
#if LLVM_VERSION_MAJOR >= 21
    toLLVM.addPass(createSCFToControlFlowPass());
#else
    toLLVM.addPass(createConvertSCFToCFPass());
#endif
    toLLVM.addNestedPass<func::FuncOp>(createConvertArmSMEToLLVMPass());
    toLLVM.addPass(createCanonicalizerPass());
    toLLVM.addPass(createCSEPass());
    if (failed(runPipeline(toLLVM, module))) {
      module.emitError("SME late ArmSME conversion pipeline failed");
      signalPassFailure();
      return;
    }
  }
};

} // namespace

void buildSMEPostBufferizationPipeline(mlir::OpPassManager &pm) {
  pm.addPass(std::make_unique<SMEArmLoweringPass>());
}

void buildSMELateBackendPipeline(mlir::OpPassManager &pm) {
  pm.addPass(std::make_unique<SMELateLoweringPass>());
}

} // namespace mlir::llk::target::sme
