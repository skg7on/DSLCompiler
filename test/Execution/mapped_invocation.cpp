#include "LLK/Runtime/MappedExecutable.h"
#include "LLK/Runtime/MappedInvocation.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Parser/Parser.h"
#include <gtest/gtest.h>
#include <limits>

namespace {
llk::InvocationBuffer2D buffer(std::vector<float> &data) {
  return {{data.data(), data.data(), 0, 2, 2, 2, 1},
          llk::InvocationElementType::F32,
          data.size() * sizeof(float)};
}
void rejects(const llk::KernelAbi &abi,
             llvm::ArrayRef<llk::InvocationBuffer2D> in,
             llvm::ArrayRef<llk::InvocationBuffer2D> out,
             llvm::StringRef reason) {
  auto error = llk::validateMappedInvocation(abi, in, out);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error)).find(reason.str()),
            std::string::npos);
}
} // namespace

TEST(MappedInvocation, ChecksTypedRangesAndLayout) {
  llk::KernelAbi abi;
  abi.inputs.push_back({{2, 2}, "f32"});
  abi.outputs.push_back({{2, 2}, "f32"});
  std::vector<float> a(8), b(4);
  auto in = buffer(a), out = buffer(b);
  EXPECT_FALSE(llk::validateMappedInvocation(abi, {in}, {out}));
  auto bad = in;
  bad.elementType = llk::InvocationElementType::BF16;
  rejects(abi, {bad}, {out}, "dtype");
  bad = in;
  bad.descriptor.allocated = nullptr;
  rejects(abi, {bad}, {out}, "null");
  bad = in;
  bad.descriptor.aligned = nullptr;
  rejects(abi, {bad}, {out}, "null");
  bad = in;
  bad.descriptor.size0 = 1;
  rejects(abi, {bad}, {out}, "extent");
  bad = in;
  bad.descriptor.offset = 1;
  rejects(abi, {bad}, {out}, "offset");
  bad = in;
  bad.descriptor.offset = -1;
  rejects(abi, {bad}, {out}, "offset");
  for (auto strides : {std::pair{3, 1}, std::pair{2, 2}, std::pair{-2, 1}}) {
    bad = in;
    bad.descriptor.stride0 = strides.first;
    bad.descriptor.stride1 = strides.second;
    rejects(abi, {bad}, {out}, "stride");
  }
  bad = in;
  bad.allocationBytes = 15;
  rejects(abi, {bad}, {out}, "range");
  bad = in;
  bad.descriptor.aligned = a.data() + 5;
  rejects(abi, {bad}, {out}, "range");
  bad = in;
  bad.descriptor.allocated = a.data() + 1;
  rejects(abi, {bad}, {out}, "range");
  bad = in;
  bad.descriptor.aligned = a.data() + 4;
  EXPECT_FALSE(llk::validateMappedInvocation(abi, {bad}, {out}));
  bad.allocationBytes = std::numeric_limits<uint64_t>::max();
  rejects(abi, {bad}, {out}, "overflow");
  rejects(abi, {}, {out}, "input");
  rejects(abi, {in}, {}, "output");
}

TEST(MappedInvocation, RejectsOutputAliasingButAllowsOverlappingInputs) {
  llk::KernelAbi abi;
  abi.inputs = {{{2, 2}, "f32"}, {{2, 2}, "f32"}};
  abi.outputs = {{{2, 2}, "f32"}, {{2, 2}, "f32"}};
  std::vector<float> a(4), b(8);
  auto in = buffer(a), out = buffer(b), second = out;
  second.descriptor.aligned = b.data() + 4;
  EXPECT_FALSE(llk::validateMappedInvocation(abi, {in, in}, {out, second}));
  rejects(abi, {in, in}, {out, out}, "overlap");
  second.descriptor.aligned = b.data() + 3;
  rejects(abi, {in, in}, {out, second}, "overlap");
  rejects(abi, {in, in}, {out, in}, "overlap");
}

TEST(MappedInvocation, HashDistinguishesAbiContracts) {
  llk::KernelAbi abi;
  abi.inputs = {{{2, 2}, "f32"}, {{2, 4}, "i8"}};
  abi.outputs = {{{2, 2}, "f32"}};
  auto hash = llk::computeKernelAbiHash(abi);
  EXPECT_EQ(hash, llk::computeKernelAbiHash(abi));
  auto changed = abi;
  changed.version++;
  EXPECT_NE(hash, llk::computeKernelAbiHash(changed));
  changed = abi;
  changed.inputs[0].shape[0]++;
  EXPECT_NE(hash, llk::computeKernelAbiHash(changed));
  changed = abi;
  changed.outputs[0].elementType = "i32";
  EXPECT_NE(hash, llk::computeKernelAbiHash(changed));
  changed = abi;
  std::swap(changed.inputs[0], changed.inputs[1]);
  EXPECT_NE(hash, llk::computeKernelAbiHash(changed));
}

TEST(MappedInvocation, RejectsUnsupportedSignaturesBeforeJit) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                      mlir::arith::ArithDialect>();
  for (const char *type : {"memref<2x2x2xf32>", "memref<?x2xf32>",
                           "memref<2x2xf64>", "memref<0x2xf32>"}) {
    auto module = mlir::parseSourceString<mlir::ModuleOp>(
        std::string("module { func.func @kernel(%a: ") + type +
            ") { return } }",
        &context);
    ASSERT_TRUE(module);
    auto executable = llk::createMappedExecutable(*module, "kernel");
    EXPECT_FALSE(static_cast<bool>(executable)) << type;
    if (!executable)
      llvm::consumeError(executable.takeError());
  }
}

TEST(MappedInvocation, RejectsThirteenDescriptorsAtConstruction) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
  std::string source = "module { func.func @kernel(";
  for (unsigned i = 0; i < 13; ++i) {
    if (i)
      source += ", ";
    source += "%a" + std::to_string(i) + ": memref<2x2xf32>";
  }
  source += ") { return } }";
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
  ASSERT_TRUE(module);
  auto executable = llk::createMappedExecutable(*module, "kernel");
  EXPECT_FALSE(static_cast<bool>(executable));
  if (!executable)
    llvm::consumeError(executable.takeError());
}

TEST(MappedInvocation, CheckedInvokeUsesAlignedSubviewAndRejectsBeforeWrite) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      "module { func.func @kernel(%arg: memref<2x2xf32>) "
      "-> memref<2x2xf32> { return %arg : memref<2x2xf32> } }",
      &context);
  ASSERT_TRUE(module);
  auto executable = llk::createMappedExecutable(*module, "kernel");
  ASSERT_TRUE(static_cast<bool>(executable));

  std::vector<float> input{90, 91, 1, 2, 3, 4};
  std::vector<float> output(4, -7);
  auto in = buffer(input);
  in.descriptor.aligned = input.data() + 2;
  auto out = buffer(output);
  auto bad = in;
  bad.elementType = llk::InvocationElementType::BF16;
  auto error = (*executable)->invoke({bad}, {out});
  ASSERT_TRUE(static_cast<bool>(error));
  llvm::consumeError(std::move(error));
  EXPECT_EQ(output, (std::vector<float>(4, -7)));

  error = (*executable)->invoke({in}, {out});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));
  EXPECT_EQ(output, (std::vector<float>{1, 2, 3, 4}));
}

TEST(MappedInvocation, CopiesMultipleBorrowedResultsToDistinctOutputs) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect, mlir::memref::MemRefDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      "module { func.func @kernel(%a: memref<2x2xf32>, "
      "%b: memref<2x2xf32>) -> (memref<2x2xf32>, memref<2x2xf32>) { "
      "return %a, %b : memref<2x2xf32>, memref<2x2xf32> } }",
      &context);
  ASSERT_TRUE(module);
  auto executable = llk::createMappedExecutable(*module, "kernel");
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());
  ASSERT_EQ((*executable)->abi().outputs.size(), 2u);

  std::vector<float> first{1, 2, 3, 4};
  std::vector<float> second{10, 20, 30, 40};
  std::vector<float> outputA(4, -1), outputB(4, -2);
  auto a = buffer(first), b = buffer(second);
  auto outA = buffer(outputA), outB = buffer(outputB);
  llvm::Error error = (*executable)->invoke({a, b}, {outA, outB});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));
  EXPECT_EQ(outputA, first);
  EXPECT_EQ(outputB, second);
}
