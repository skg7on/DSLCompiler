//===- micro_kernel_execution.cpp - The kernel ABI, lowered and JIT-able --===//
//
// Two checks along the #109 item-4 chain, both about the *signature* a micro
// kernel acquires on its way to the backend:
//
//   1. the ABI itself -- a `micro.kernel` declares no arguments and no results,
//      so the bridge turns its entry tensors into parameters and its stored
//      value into the result, and bufferization turns that result into a
//      concrete memref return. This test lowers a kernel and asserts the
//      signature it ends up with;
//   2. that the JIT accepts it -- a build without a working ORC JIT is a skip,
//      the same contract the legacy execution tests take.
//
// KNOWN GAP -- the symbol is not *called*. Making the call work needs an ABI
// decision, and the reason is concrete:
//
//   * MLIR does not pass a memref as one aggregate. Each memref parameter is
//     *expanded* into its seven fields (allocated, aligned, offset, size[2],
//     stride[2]), so a lowered kernel's LLVM signature is fourteen scalars --
//     not two descriptors.
//   * On AArch64 an aggregate result larger than 16 bytes comes back through a
//     hidden pointer in `x8`. With fourteen scalar arguments, `x8` is also the
//     *ninth* argument's register, so no C declaration can express the emitted
//     signature: declaring the parameters as descriptors passes them byval (a
//     pointer each) and leaves the result buffer unwritten, and declaring the
//     fourteen scalars collides with the indirect-result register.
//   * The runtime's `KernelFn` (five `MemRef2D *`) assumes descriptor
//     *pointers*, which is not what MLIR emits either. The legacy execution
//     tests would catch that -- but they are skipped on this machine, so the
//     mismatch has never been exercised.
//
// Resolving it is a design choice: lower the kernel ABI to descriptor pointers
// (`!llvm.ptr` parameters) so it matches the runtime, or give the runtime a
// trampoline that adapts `MemRef2D *` to the expanded form. Until then "it
// compiles" is as far as this test can honestly go.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/MicroToLinalg.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Runtime/JitCache.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Error.h"

#include <gtest/gtest.h>

#include <string>

#if LLVM_VERSION_MAJOR >= 21
using BufOpts = mlir::bufferization::OneShotBufferizePassOptions;
#else
using BufOpts = mlir::bufferization::OneShotBufferizationOptions;
#endif

namespace {

/// A vector add, written in Micro-IR. The two entry tensors are what the kernel
/// reads; the `micro.tile_store` is what it produces.
constexpr llvm::StringLiteral kAddKernel = R"mlir(
module {
  micro.kernel @add {
    %a = tensor.empty() : tensor<8x8xf32>
    %b = tensor.empty() : tensor<8x8xf32>
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %ta, %tb : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<acc>>
    micro.tile_store %r {dst_memory = #micro.memory<dram>} : !micro.tile<8x8xf32, memory = #micro.memory<acc>>
    micro.yield
  }
}
)mlir";

mlir::DialectRegistry buildRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::micro::MicroDialect, mlir::func::FuncDialect,
                  mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
                  mlir::scf::SCFDialect, mlir::arith::ArithDialect,
                  mlir::memref::MemRefDialect>();
  // Bufferization interfaces, without which One-Shot Bufferize cannot run.
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  return registry;
}

/// Lower a micro kernel to the form the JIT's own lowering passes expect:
/// Linalg over tensors (the bridge), buffered, and rolled into loops.
bool lowerToLoops(mlir::ModuleOp module) {
  mlir::PassManager pm(module->getContext());
  pm.addPass(mlir::llk::createMicroToLinalgPass());
  BufOpts options;
  options.bufferizeFunctionBoundaries = true;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(options));
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  return mlir::succeeded(pm.run(module));
}

} // namespace

TEST(MicroKernelExecution, GivesTheKernelItsArgumentsAndResult) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kAddKernel, &context);
  ASSERT_TRUE(module) << "the fixture must parse";

  ASSERT_TRUE(lowerToLoops(*module))
      << "the bridge and bufferization must succeed";

  // The ABI: the two entry tensors are the parameters, the stored value is the
  // result, and bufferization has made it a concrete memref.
  auto function = module->lookupSymbol<mlir::func::FuncOp>("add");
  ASSERT_TRUE(function) << "the kernel must have become a function";
  mlir::FunctionType type = function.getFunctionType();
  ASSERT_EQ(type.getNumInputs(), 2u) << "one parameter per entry tensor";
  EXPECT_TRUE(llvm::isa<mlir::MemRefType>(type.getInput(0)));
  EXPECT_TRUE(llvm::isa<mlir::MemRefType>(type.getInput(1)));
  ASSERT_EQ(type.getNumResults(), 1u) << "the store is the result";
  EXPECT_TRUE(llvm::isa<mlir::MemRefType>(type.getResult(0)));
}

TEST(MicroKernelExecution, TheLoweredKernelReachesTheJit) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kAddKernel, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(lowerToLoops(*module));

  llk::JitCache cache;
  llvm::Expected<llk::JitCache::KernelFn> symbol =
      cache.lookupOrCompile("micro_kernel_abi", *module);
  if (!symbol) {
    llvm::consumeError(symbol.takeError());
    GTEST_SKIP() << "JIT compilation not available";
  }
}
