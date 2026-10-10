//===- micro_kernel_execution.cpp - The kernel ABI, lowered and JIT-able --===//
//
// Two checks along the #109 item-4 chain, both about the *signature* a micro
// kernel acquires on its way to the backend:
//
//   1. the ABI itself -- a `micro.kernel` declares its inputs and results in
//      its signature; the entry block arguments are the parameters a caller
//      passes and the yielded value is the result, which bufferization turns
//      into a concrete memref return. This test lowers a kernel and asserts
//      the signature it ends up with;
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
#include "LLK/Runtime/MappedExecutable.h"

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

#include <algorithm>
#include <string>
#include <vector>

#if LLVM_VERSION_MAJOR >= 21
using BufOpts = mlir::bufferization::OneShotBufferizePassOptions;
#else
using BufOpts = mlir::bufferization::OneShotBufferizationOptions;
#endif

namespace {

/// A vector add, written in Micro-IR. Its signature names the two inputs the
/// caller passes and the result it receives; nothing about the ABI is inferred
/// from the body. The `micro.yield` carries the logical result, which the
/// interface accepts as the declared tensor because shape and element type
/// agree -- memory space is a placement fact, not part of the caller's
/// contract.
constexpr llvm::StringLiteral kAddKernel = R"mlir(
module {
  micro.kernel @add(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %ta, %tb : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<acc>>
    micro.yield %r : !micro.tile<8x8xf32, memory = #micro.memory<acc>>
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

  // The ABI: the declared inputs are the parameters, the yielded value is the
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

//===----------------------------------------------------------------------===//
// C4: the mapped kernel is *called*
//===----------------------------------------------------------------------===//
//
// Everything above stops at "it compiles". These tests run the lowered kernel
// through descriptor pointers and check what it computed -- which is the only
// thing that turns the calling convention into a claim rather than a hope. A
// compilation or invocation failure is a test failure here, never a skip: an
// ABI that is only exercised when it happens to work proves nothing.

namespace {

/// Lowers `kAddKernel` and compiles it for invocation. The MLIR context belongs
/// to the caller; the executable owns its own JIT and translated module, so
/// nothing here keeps a borrowed MLIR operation alive.
std::unique_ptr<llk::MappedExecutable> compileAdd(mlir::MLIRContext &context) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kAddKernel, &context);
  EXPECT_TRUE(module);
  if (!module)
    return nullptr;
  EXPECT_TRUE(lowerToLoops(*module));

  llvm::Expected<std::unique_ptr<llk::MappedExecutable>> executable =
      llk::createMappedExecutable(*module, "add");
  if (!executable) {
    ADD_FAILURE() << llvm::toString(executable.takeError());
    return nullptr;
  }
  return std::move(*executable);
}

/// A[i] = i and B[i] = 1 give out[i] = i + 1 exactly, so every element of the
/// result is checkable. The B storage can start at an offset, so a caller can
/// hand over a window into a larger buffer instead of a fresh allocation.
void fillInputs(std::vector<float> &a, std::vector<float> &bStorage,
                size_t bBase) {
  a.resize(64);
  for (size_t i = 0; i < a.size(); ++i)
    a[i] = float(i);
  bStorage.assign(bBase + 64, -1.0f);
  std::fill(bStorage.begin() + bBase, bStorage.end(), 1.0f);
}

/// Runs the add kernel over freshly filled inputs and returns what it wrote.
std::vector<float> runAdd(llk::MappedExecutable &executable,
                          std::vector<float> &aStorage,
                          std::vector<float> &bStorage) {
  std::vector<float> out(64, -1.0f);
  llk::InvocationBuffer2D da{{aStorage.data(), aStorage.data(), 0, 8, 8, 8, 1},
                             llk::InvocationElementType::F32,
                             aStorage.size() * sizeof(float)};
  llk::InvocationBuffer2D db{{bStorage.data(), bStorage.data(), 0, 8, 8, 8, 1},
                             llk::InvocationElementType::F32,
                             bStorage.size() * sizeof(float)};
  llk::InvocationBuffer2D dout{{out.data(), out.data(), 0, 8, 8, 8, 1},
                               llk::InvocationElementType::F32,
                               out.size() * sizeof(float)};
  llvm::Error error = executable.invoke({da, db}, {dout});
  if (error) {
    ADD_FAILURE() << llvm::toString(std::move(error));
    return {};
  }
  return out;
}

} // namespace

TEST(MicroKernelExecution, InvokesDescriptorPointersAndWritesEveryResult) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  std::unique_ptr<llk::MappedExecutable> executable = compileAdd(context);
  ASSERT_TRUE(executable);

  // The ABI is what the kernel declared, recorded before its types were erased
  // into LLVM pointers.
  ASSERT_EQ(executable->abi().inputs.size(), 2u);
  ASSERT_EQ(executable->abi().outputs.size(), 1u);
  EXPECT_EQ(executable->abi().inputs[0].shape, (std::vector<int64_t>{8, 8}));
  EXPECT_EQ(executable->abi().inputs[0].elementType, "f32");
  EXPECT_EQ(executable->abi().outputs[0].shape, (std::vector<int64_t>{8, 8}));

  std::vector<float> a, b;
  fillInputs(a, b, /*bBase=*/0);
  std::vector<float> out = runAdd(*executable, a, b);
  ASSERT_EQ(out.size(), 64u);
  for (size_t i = 0; i < out.size(); ++i)
    EXPECT_EQ(out[i], float(i + 1)) << "element " << i;
}

TEST(MicroKernelExecution, InvokesTheSameKernelRepeatedly) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  std::unique_ptr<llk::MappedExecutable> executable = compileAdd(context);
  ASSERT_TRUE(executable);

  std::vector<float> a, b;
  fillInputs(a, b, /*bBase=*/0);
  // The second invocation reuses the compiled code, so it also checks that the
  // first call consumed nothing the executable still needs.
  for (int round = 0; round < 2; ++round) {
    std::vector<float> out = runAdd(*executable, a, b);
    ASSERT_EQ(out.size(), 64u);
    for (size_t i = 0; i < out.size(); ++i)
      EXPECT_EQ(out[i], float(i + 1)) << "round " << round << " element " << i;
  }
}

TEST(MicroKernelExecution, AcceptsABufferThatStartsInsideALargerAllocation) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  std::unique_ptr<llk::MappedExecutable> executable = compileAdd(context);
  ASSERT_TRUE(executable);

  // Writing into the middle of a buffer the caller owns is the normal case in a
  // real program, and the descriptor's `aligned` pointer is how it is said.
  std::vector<float> a;
  std::vector<float> bStorage;
  fillInputs(a, bStorage, /*bBase=*/8);

  const size_t outBase = 16;
  std::vector<float> outStorage(64 + 32, -2.0f);
  llk::InvocationBuffer2D da{{a.data(), a.data(), 0, 8, 8, 8, 1},
                             llk::InvocationElementType::F32,
                             a.size() * sizeof(float)};
  llk::InvocationBuffer2D db{
      {bStorage.data(), bStorage.data() + 8, 0, 8, 8, 8, 1},
      llk::InvocationElementType::F32,
      bStorage.size() * sizeof(float)};
  llk::InvocationBuffer2D dout{
      {outStorage.data(), outStorage.data() + outBase, 0, 8, 8, 8, 1},
      llk::InvocationElementType::F32,
      outStorage.size() * sizeof(float)};

  llvm::Error error = executable->invoke({da, db}, {dout});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));
  for (size_t i = 0; i < 64; ++i)
    EXPECT_EQ(outStorage[outBase + i], float(i + 1)) << "element " << i;
  // Nothing outside the window was touched.
  EXPECT_EQ(outStorage[outBase - 1], -2.0f);
  EXPECT_EQ(outStorage[outBase + 64], -2.0f);
}

TEST(MicroKernelExecution, RefusesDescriptorsThatDisagreeWithTheAbi) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  std::unique_ptr<llk::MappedExecutable> executable = compileAdd(context);
  ASSERT_TRUE(executable);

  std::vector<float> a, b, out(64, -1.0f);
  fillInputs(a, b, /*bBase=*/0);
  llk::InvocationBuffer2D da{{a.data(), a.data(), 0, 8, 8, 8, 1},
                             llk::InvocationElementType::F32,
                             a.size() * sizeof(float)};
  llk::InvocationBuffer2D db{{b.data(), b.data(), 0, 8, 8, 8, 1},
                             llk::InvocationElementType::F32,
                             b.size() * sizeof(float)};
  llk::InvocationBuffer2D dout{{out.data(), out.data(), 0, 8, 8, 8, 1},
                               llk::InvocationElementType::F32,
                               out.size() * sizeof(float)};

  // Too few inputs: the kernel expects two, and calling it with one would pass
  // whatever happened to be in the next register.
  llvm::Error arity = executable->invoke({da}, {dout});
  ASSERT_TRUE(static_cast<bool>(arity));
  EXPECT_NE(llvm::toString(std::move(arity)).find("input"), std::string::npos);

  // A descriptor of the wrong shape would make the kernel address memory the
  // caller never offered.
  auto wrongShape = dout;
  wrongShape.descriptor.size0 = 4;
  wrongShape.descriptor.size1 = 4;
  wrongShape.descriptor.stride0 = 4;
  llvm::Error shape = executable->invoke({da, db}, {wrongShape});
  ASSERT_TRUE(static_cast<bool>(shape));
  EXPECT_NE(llvm::toString(std::move(shape)).find("extent"), std::string::npos);

  // A row-major kernel cannot honour a different stride, so it is refused
  // rather than silently reading the wrong elements.
  auto wrongStride = dout;
  wrongStride.descriptor.stride0 = 16;
  wrongStride.descriptor.stride1 = 2;
  llvm::Error stride = executable->invoke({da, db}, {wrongStride});
  ASSERT_TRUE(static_cast<bool>(stride));
  EXPECT_NE(llvm::toString(std::move(stride)).find("stride"),
            std::string::npos);

  // A null descriptor is a caller mistake, not undefined behaviour.
  auto nullBuffer = db;
  nullBuffer.descriptor.allocated = nullptr;
  llvm::Error null = executable->invoke({da, nullBuffer}, {dout});
  ASSERT_TRUE(static_cast<bool>(null));
  EXPECT_NE(llvm::toString(std::move(null)).find("null"), std::string::npos);
}

TEST(MicroKernelExecution, RefusesAModuleWithoutTheEntrySymbol) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kAddKernel, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(lowerToLoops(*module));

  llvm::Expected<std::unique_ptr<llk::MappedExecutable>> executable =
      llk::createMappedExecutable(*module, "not_a_kernel");
  ASSERT_FALSE(static_cast<bool>(executable));
  EXPECT_NE(llvm::toString(executable.takeError()).find("not_a_kernel"),
            std::string::npos);
}

TEST(MicroKernelExecution, KeepsTheExecutableAliveAfterTheModuleIsGone) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(kAddKernel, &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(lowerToLoops(*module));

  llvm::Expected<std::unique_ptr<llk::MappedExecutable>> executable =
      llk::createMappedExecutable(*module, "add");
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());

  // The executable owns its JIT and its translated LLVM module, so destroying
  // the source module must not disturb it -- that is what "no borrowed
  // operation" means in practice.
  module = nullptr;

  std::vector<float> a, b;
  fillInputs(a, b, /*bBase=*/0);
  std::vector<float> out = runAdd(**executable, a, b);
  ASSERT_EQ(out.size(), 64u);
  for (size_t i = 0; i < out.size(); ++i)
    EXPECT_EQ(out[i], float(i + 1)) << "element " << i;
}
