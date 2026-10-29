//===- dynamic_mapped_acceptance.cpp - Compile-once dynamic acceptance ---===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dynamic Shape Task 6 (issue #197). Tasks 1-5 proved the contracts: a
// symbolic workload, fixed-capacity tiles, bounded boundary movement, exact K
// tails, plan certification, ABI v2 and caller-owned destinations. None of
// them ran one compiled kernel against unrelated shapes.
//
// These tests do exactly that. One fully dynamic module per operator is
// exported, mapped and compiled *once* per execution mode, and the resulting
// executable is invoked with unrelated M/N/K triples, with each empty
// dimension, and with mixed static/dynamic forms. Every element of every
// result is checked against the source's numerical contract, the caller's
// input buffers are checked unchanged, and the output canaries are checked
// untouched. The compiled artifact is never rebuilt for a new shape; that is
// what "fully dynamic" means here, and `successfulInvocations()` rising while
// the compilation count stays at one is the evidence for it.
//
// The chains are host-portable: the exported program is Linalg, SCF and
// memref, and the JIT compiles it for the machine running the test. The
// selected-target binary makes a missing AVX2 host a failure rather than a
// skip, so selected evidence cannot silently come from reference execution.

#include "LLK/Conversion/LLKToMicro/LLKToMicro.h"
#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Conversion/MappedKernelAbi.h"
#include "LLK/Conversion/Workload/SourceAdapter.h"
#include "LLK/Conversion/Workload/WorkloadContract.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Machine/MachineModel.h"
#include "LLK/Mapping/CodegenRequirements.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/WorkloadGraph.h"
#include "LLK/Runtime/MappedExecutable.h"
#include "LLK/Runtime/MappedInvocation.h"
#include "LLK/Runtime/PreparedMappedKernel.h"
#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include "mapped_jit_test_factory.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/TargetParser/Host.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifndef LLK_SOURCE_DIR
#error "LLK_SOURCE_DIR must name the repository root"
#endif

namespace {

namespace shape = ::mlir::llk::shape;
namespace workload = ::mlir::llk::workload;

using mlir::llk::mapping::LayoutContext;
using mlir::llk::mapping::MappingSearchOptions;
using mlir::llk::mapping::MappingSearchResult;
using mlir::llk::mapping::MappingTarget;
using mlir::llk::mapping::SearchMode;

/// Store the high 16 bits as the BF16 representation used by the ABI fixtures.
uint16_t toBf16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7fffu + ((bits >> 16) & 1u); // round to nearest, ties to even
  return static_cast<uint16_t>(bits >> 16);
}

float fromBf16(uint16_t value) {
  uint32_t bits = static_cast<uint32_t>(value) << 16;
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

float nextSignedValue(uint32_t &state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return static_cast<float>(state & 0xffffu) / 65535.0f - 0.5f;
}

#ifdef LLK_REQUIRE_SELECTED_TARGET
bool avx2DisabledByTestOverride() {
  const char *disabled = std::getenv("LLK_TEST_DISABLE_AVX2");
  return disabled && std::strcmp(disabled, "1") == 0;
}

/// Whether this host can run the AVX2 target's code. The gate is the same
/// `checkHostRequirements` the runtime applies to the compiled artifact, fed
/// with the requirements the target declares, so a host the runtime would
/// refuse can never be reported as selected execution.
bool selectedAvx2Available() {
  if (avx2DisabledByTestOverride())
    return false;
  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      mlir::llk::target::avx2::createMappingTarget(LLK_SOURCE_DIR);
  if (!target) {
    llvm::consumeError(target.takeError());
    return false;
  }
  std::optional<mlir::llk::mapping::TargetCodegenRequirements> requirements =
      (*target)->codegenRequirements();
  if (!requirements)
    return false;
  const std::string triple = llvm::sys::getDefaultTargetTriple();
  const llvm::StringRef architecture = llvm::StringRef(triple).split('-').first;
  llvm::Error error = mlir::llk::mapping::checkHostRequirements(
      *requirements, architecture, llvm::sys::getHostCPUFeatures());
  if (error) {
    llvm::consumeError(std::move(error));
    return false;
  }
  return true;
}
#endif

#ifdef LLK_REQUIRE_SELECTED_TARGET
constexpr llk::MappedBackend kDefaultBackend =
    llk::MappedBackend::SelectedTarget;
#else
constexpr llk::MappedBackend kDefaultBackend = llk::MappedBackend::Reference;
#endif

/// The fully dynamic matmul the acceptance chain exports. Every port is
/// `?x?`: M, N and K are all runtime values, including K.
constexpr llvm::StringLiteral kDynamicMatmulSource = R"mlir(
module {
  func.func @matmul_dynamic_root(%a: tensor<?x?xbf16>, %b: tensor<?x?xbf16>,
                                 %init: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
    %y = llk.matmul ins(%a, %b : tensor<?x?xbf16>, tensor<?x?xbf16>)
        outs(%init : tensor<?x?xbf16>)
        {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
        -> tensor<?x?xbf16>
    return %y : tensor<?x?xbf16>
  }
}
)mlir";

/// The fully dynamic two-arm SwiGLU. X is [M,K] and both weights are [K,N];
/// M, N and K are runtime values.
constexpr llvm::StringLiteral kDynamicSwigluSource = R"mlir(
module {
  func.func @swiglu_dynamic_root(%x: tensor<?x?xbf16>, %wg: tensor<?x?xbf16>,
                                 %wu: tensor<?x?xbf16>, %init: tensor<?x?xbf16>)
      -> tensor<?x?xbf16> {
    %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<?x?xbf16>,
                              tensor<?x?xbf16>, tensor<?x?xbf16>)
        outs(%init : tensor<?x?xbf16>)
        {accumulator_type = f32, activation = #llk.activation<silu>,
         math_mode = #llk.math_mode<bounded_fast>}
        -> tensor<?x?xbf16>
    return %y : tensor<?x?xbf16>
  }
}
)mlir";

/// The same two-arm SwiGLU with a static N. The destination values are not
/// read by SwiGLU, so its port is the caller-owned output, not a fourth read
/// input.
constexpr llvm::StringLiteral kMixedSwigluSource = R"mlir(
module {
  func.func @swiglu_mixed_root(%x: tensor<?x?xbf16>, %wg: tensor<?x32xbf16>,
                               %wu: tensor<?x32xbf16>, %init: tensor<?x32xbf16>)
      -> tensor<?x32xbf16> {
    %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<?x?xbf16>,
                              tensor<?x32xbf16>, tensor<?x32xbf16>)
        outs(%init : tensor<?x32xbf16>)
        {accumulator_type = f32, activation = #llk.activation<silu>,
         math_mode = #llk.math_mode<bounded_fast>}
        -> tensor<?x32xbf16>
    return %y : tensor<?x32xbf16>
  }
}
)mlir";

/// A mixed static/dynamic contraction: only N is fixed, M and K are runtime
/// values. A blocked layout proven for N = 32 is still legal for the full
/// symbolic M/K domain, so the same artifact serves unrelated M/K pairs.
/// The strict-math counterpart of the dynamic matmul. Its numerical contract
/// makes reduction order and IEEE special values observable, so a padded K
/// tail cannot be substituted for the exact valid reduction.
constexpr llvm::StringLiteral kStrictMatmulSource = R"mlir(
module {
  func.func @matmul_strict_root(%a: tensor<?x?xbf16>, %b: tensor<?x?xbf16>,
                                %init: tensor<?x?xbf16>) -> tensor<?x?xbf16> {
    %y = llk.matmul ins(%a, %b : tensor<?x?xbf16>, tensor<?x?xbf16>)
        outs(%init : tensor<?x?xbf16>)
        {accumulator_type = f32, math_mode = #llk.math_mode<strict>}
        -> tensor<?x?xbf16>
    return %y : tensor<?x?xbf16>
  }
}
)mlir";

constexpr llvm::StringLiteral kMixedMatmulSource = R"mlir(
module {
  func.func @matmul_mixed_root(%a: tensor<?x?xbf16>, %b: tensor<?x32xbf16>,
                               %init: tensor<?x32xbf16>) -> tensor<?x32xbf16> {
    %y = llk.matmul ins(%a, %b : tensor<?x?xbf16>, tensor<?x32xbf16>)
        outs(%init : tensor<?x32xbf16>)
        {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
        -> tensor<?x32xbf16>
    return %y : tensor<?x32xbf16>
  }
}
)mlir";

mlir::DialectRegistry buildRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::scf::SCFDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                  mlir::cf::ControlFlowDialect>();
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  return registry;
}

/// What one compile-once chain produced, plus the evidence the selected
/// acceptance needs to distinguish realization from reference fallback.
struct DynamicArtifact {
  std::unique_ptr<llk::MappedExecutable> executable;
  workload::WorkloadContract contract;
  std::string entrySymbol;
  std::string llvmDialect;
  std::string llvmIR;
  uint64_t planId = 0;
  unsigned targetLowered = 0;
  unsigned referenceLowered = 0;
  unsigned selectedGroupsVerified = 0;
  unsigned backendGroupsRealized = 0;
  unsigned referenceGroupsLowered = 0;
  std::optional<mlir::llk::mapping::TargetCodegenRequirements>
      codegenRequirements;
  unsigned compilations = 0;
};

/// Symbolic contract of the exported source, taken from the adapter that
/// already knows the root's ports and axes. Reusing it is deliberate: a test
/// that hand-writes M/N/K could pass while the export itself never described
/// them.
llvm::Expected<workload::WorkloadContract>
describeSourceContract(mlir::ModuleOp module, llvm::StringRef sourceSymbol,
                       mlir::Operation **describedRoot) {
  workload::SourceAdapterRegistry adapters =
      workload::makeDefaultSourceAdapters();
  llvm::Expected<mlir::Operation *> root =
      workload::selectSourceRoot(module, {sourceSymbol.str(), 0}, adapters);
  if (!root)
    return root.takeError();
  llvm::Expected<const workload::SourceAdapter *> adapter =
      adapters.resolve(*root);
  if (!adapter)
    return adapter.takeError();
  llvm::Expected<workload::WorkloadContract> contract =
      (*adapter)->describe(*root);
  if (!contract)
    return contract.takeError();
  if (describedRoot)
    *describedRoot = *root;
  return contract;
}

mlir::Operation *findMicroKernel(mlir::ModuleOp module) {
  mlir::Operation *kernel = nullptr;
  module.walk([&](mlir::Operation *op) {
    if (!kernel && op->getName().getStringRef() == "micro.kernel")
      kernel = op;
  });
  return kernel;
}

/// Searches the shipped AVX2 target for the module's kernel and compiles the
/// first complete plan to `stop`. `sink` observes the JIT input when the
/// compilation reaches an executable.
llvm::Expected<llk::MappedCompilation>
searchAndCompile(mlir::MLIRContext &context, mlir::ModuleOp module,
                 const workload::WorkloadContract &contract,
                 llvm::StringRef entrySymbol, llk::MappedBackend backend,
                 llk::MappedStop stop, std::unique_ptr<MappingTarget> target,
                 llk::MappedJitEvidenceSink sink = {}) {
  mlir::Operation *kernel = findMicroKernel(module);
  if (!kernel)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the module contains no micro.kernel");
  if (!target) {
    llvm::Expected<std::unique_ptr<MappingTarget>> created =
        mlir::llk::target::avx2::createMappingTarget(LLK_SOURCE_DIR);
    if (!created)
      return created.takeError();
    target = std::move(*created);
  }
  llvm::Expected<mlir::llk::mapping::WorkloadGraph> graph =
      mlir::llk::mapping::extractWorkloadGraph(kernel);
  if (!graph)
    return graph.takeError();

  LayoutContext layout;
  layout.rank = 2;
  layout.elementType = contract.ports.front().elementType;
  MappingSearchOptions searchOptions;
  searchOptions.mode = SearchMode::Exact;
  mlir::llk::mapping::CoveringSearch search(*graph, *target, context, layout,
                                            searchOptions);
  llvm::Expected<MappingSearchResult> result = search.search();
  if (!result)
    return result.takeError();
  if (result->plans.empty()) {
    std::string detail = "the dynamic source has no complete plan";
    for (const auto &diagnostic : result->frontier.diagnostics)
      detail += "; " + diagnostic.message;
    return llvm::createStringError(llvm::inconvertibleErrorCode(), detail);
  }

  llk::MappedCompileOptions options;
  options.entrySymbol = entrySymbol.str();
  options.backend = backend;
  options.stop = stop;
  options.evidenceSink = std::move(sink);
  return llk::compileMappedKernel(module, *target, result->plans.front(),
                                  options);
}

/// Maps and compiles a module that already holds one `micro.kernel` carrying
/// its canonical contract. The returned artifact holds the single executable
/// every shape is invoked through.
llvm::Expected<DynamicArtifact>
compileMappedModule(mlir::MLIRContext &context, mlir::ModuleOp module,
                    workload::WorkloadContract contract,
                    llk::MappedBackend backend,
                    std::unique_ptr<MappingTarget> target = nullptr) {
  mlir::Operation *kernel = findMicroKernel(module);
  if (!kernel)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the module contains no micro.kernel");
  auto symbol = kernel->getAttrOfType<mlir::StringAttr>("sym_name");
  if (!symbol)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the exported kernel has no symbol");

  DynamicArtifact artifact;
  artifact.contract = std::move(contract);
  artifact.entrySymbol = symbol.getValue().str();

  llk::MappedJitEvidenceSink sink =
      [&artifact](const llk::MappedJitEvidence &evidence) -> llvm::Error {
    ++artifact.compilations;
    artifact.llvmDialect = evidence.llvmDialect;
    artifact.llvmIR = evidence.llvmIR;
    artifact.planId = evidence.planId;
    artifact.selectedGroupsVerified = evidence.selectedGroupsVerified;
    artifact.backendGroupsRealized = evidence.backendGroupsRealized;
    artifact.referenceGroupsLowered = evidence.referenceGroupsLowered;
    artifact.codegenRequirements = evidence.selectedTarget;
    return llvm::Error::success();
  };
  llvm::Expected<llk::MappedCompilation> compiled = searchAndCompile(
      context, module, artifact.contract, artifact.entrySymbol, backend,
      llk::MappedStop::Executable, std::move(target), std::move(sink));
  if (!compiled)
    return compiled.takeError();
  if (!compiled->executable)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the dynamic compilation produced no "
                                   "executable");
  artifact.targetLowered = compiled->targetLowered;
  artifact.referenceLowered = compiled->referenceLowered;
  artifact.executable = std::move(compiled->executable);
  return artifact;
}

/// Exports one LLK source to Micro-IR and attaches the canonical contract the
/// source adapter already described. The returned module is the compile input;
/// `contract` is what the adapter said the source implements.
llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>>
exportDynamicSource(mlir::MLIRContext &context, llvm::StringRef sourceText,
                    llvm::StringRef sourceSymbol,
                    workload::WorkloadContract &contract) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &context);
  if (!module)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the LLK source did not parse");

  llvm::Expected<workload::WorkloadContract> described =
      describeSourceContract(*module, sourceSymbol, /*describedRoot=*/nullptr);
  if (!described)
    return described.takeError();
  contract = std::move(*described);

  mlir::PassManager pm(&context);
  pm.addPass(mlir::llk::createLLKToMicroPass());
  if (mlir::failed(pm.run(*module)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the LLK-to-Micro export failed");

  mlir::Operation *kernel = findMicroKernel(*module);
  if (!kernel)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the export produced no micro.kernel");
  auto encoded = workload::encodeWorkloadContract(&context, contract);
  if (!encoded)
    return encoded.takeError();
  kernel->setAttr("micro.workload_contract", *encoded);
  return module;
}

/// Exports one fully dynamic LLK source and compiles it once.
llvm::Expected<DynamicArtifact>
compileDynamicSource(mlir::MLIRContext &context, llvm::StringRef sourceText,
                     llvm::StringRef sourceSymbol,
                     llk::MappedBackend backend = kDefaultBackend,
                     std::unique_ptr<MappingTarget> target = nullptr) {
  workload::WorkloadContract contract;
  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> module =
      exportDynamicSource(context, sourceText, sourceSymbol, contract);
  if (!module)
    return module.takeError();
  return compileMappedModule(context, **module, std::move(contract), backend,
                             std::move(target));
}

/// The shipped AVX2 target on a machine whose local SRAM is smaller than the
/// external outputs these tests feed it. Only the profile's SRAM capacity
/// changes: the policy, layouts, rules and emitter set stay the shipped ones,
/// so this is a genuine capacity-bound profile rather than a loosened target.
///
/// `createMappingTarget` already accepts an alternate machine path, so the
/// test writes one edit of the checked-in profile to a temporary file instead
/// of growing the target package's API for a test.
llvm::Expected<std::unique_ptr<MappingTarget>>
makeSmallSramTarget(uint64_t sramBytes) {
  const std::string root = LLK_SOURCE_DIR;
  const std::string machinePath = root + "/machines/x86-avx2-v2.yaml";
  std::ifstream input(machinePath);
  if (!input)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot read the AVX2 machine profile");
  std::stringstream buffer;
  buffer << input.rdbuf();
  std::string profile = buffer.str();

  const size_t sram = profile.find("id: sram.0");
  const size_t capacity = sram == std::string::npos
                              ? std::string::npos
                              : profile.find("capacity_bytes:", sram);
  if (capacity == std::string::npos)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the AVX2 machine profile has no SRAM "
                                   "capacity");
  const size_t number = profile.find(':', capacity) + 1;
  const size_t end = profile.find('\n', number);
  profile.replace(number, end - number, " " + std::to_string(sramBytes));

  const std::filesystem::path directory =
      std::filesystem::temp_directory_path() /
      ("llk-small-sram-" + std::to_string(sramBytes));
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error)
    return llvm::createStringError(error,
                                   "cannot create the profile directory");
  const std::filesystem::path path = directory / "machine.yaml";
  {
    std::ofstream output(path, std::ios::trunc);
    output << profile;
    if (!output)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "cannot write the profile");
  }
  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      mlir::llk::target::avx2::createMappingTarget(root, path.string());
  std::filesystem::remove_all(directory, error);
  return target;
}

/// One BF16 buffer of `rows` x `columns` with caller-owned storage.
struct Bf16Buffer {
  std::vector<uint16_t> storage;
  int64_t rows = 0;
  int64_t columns = 0;

  Bf16Buffer() = default;
  Bf16Buffer(int64_t rowsValue, int64_t columnsValue, float fill)
      : storage(static_cast<size_t>(rowsValue * columnsValue), toBf16(fill)),
        rows(rowsValue), columns(columnsValue) {}

  llk::InvocationBuffer2D checked() {
    const uint64_t bytes = storage.size() * sizeof(uint16_t);
    uint16_t *data = storage.empty() ? nullptr : storage.data();
    return {{data, data, 0, rows, columns, columns, 1},
            llk::InvocationElementType::BF16,
            bytes};
  }
};

void fillSignedBf16(Bf16Buffer &buffer, uint32_t &state) {
  for (uint16_t &value : buffer.storage)
    value = toBf16(nextSignedValue(state));
}

float storedBf16(const Bf16Buffer &buffer, size_t index) {
  return fromBf16(buffer.storage[index]);
}

/// Invokes one compiled dynamic matmul for `m` x `n` x `k` and checks every
/// element, the canaries around the destination and the read-only inputs.
void expectMatmulShape(llk::MappedExecutable &executable, int64_t m, int64_t n,
                       int64_t k) {
  const llk::KernelAbi &abi = executable.abi();
  ASSERT_EQ(abi.inputs.size(), 3u);
  ASSERT_EQ(abi.outputs.size(), 1u);
  ASSERT_EQ(abi.arguments.size(), 4u);

  Bf16Buffer a(m, k, 0.0f);
  Bf16Buffer b(k, n, 0.0f);
  Bf16Buffer init(m, n, 0.0f);
  uint32_t state = 0x51a3u;
  fillSignedBf16(a, state);
  fillSignedBf16(b, state);
  fillSignedBf16(init, state);
  const auto originalA = a.storage;
  const auto originalB = b.storage;
  const auto originalInit = init.storage;

  // The destination sits inside a larger caller allocation with sentinels on
  // both sides. A kernel that wrote a full tile past the valid rectangle would
  // change them.
  const size_t elements = static_cast<size_t>(m * n);
  std::vector<uint16_t> guarded(elements + 2, toBf16(-77.0f));
  uint16_t *destination = guarded.data() + 1;
  if (elements == 0)
    destination = nullptr;
  llk::InvocationBuffer2D output{
      {destination, destination, 0, m, n, n, 1},
      llk::InvocationElementType::BF16,
      elements == 0 ? 0 : guarded.size() * sizeof(uint16_t)};

  llvm::Error error =
      executable.invoke({a.checked(), b.checked(), init.checked()}, {output});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t column = 0; column < n; ++column) {
      const size_t index = static_cast<size_t>(row * n + column);
      float expected = storedBf16(init, index);
      for (int64_t inner = 0; inner < k; ++inner) {
        expected += storedBf16(a, static_cast<size_t>(row * k + inner)) *
                    storedBf16(b, static_cast<size_t>(inner * n + column));
      }
      // The source accumulates in f32 and stores BF16, so the comparison is
      // against the exact-K f32 reference plus BF16 output rounding (half an
      // ULP is 2^-9 of the magnitude) and f32 summation order. Narrowing to
      // BF16 is part of the source's numerical contract; dropping K terms or
      // accumulating in BF16 would exceed this by an order of magnitude.
      EXPECT_NEAR(fromBf16(guarded[index + 1]), expected,
                  2e-4f + 4e-3f * std::abs(expected))
          << "shape " << m << "x" << n << "x" << k << " at " << row << ","
          << column;
    }
  }
  EXPECT_EQ(fromBf16(guarded.front()), -77.0f);
  EXPECT_EQ(fromBf16(guarded.back()), -77.0f);
  EXPECT_EQ(a.storage, originalA);
  EXPECT_EQ(b.storage, originalB);
  EXPECT_EQ(init.storage, originalInit);
}

/// Invokes one compiled dynamic SwiGLU for `m` x `n` x `k` and checks every
/// element against the source's bounded-fast activation contract.
void expectSwigluShape(llk::MappedExecutable &executable, int64_t m, int64_t n,
                       int64_t k) {
  const llk::KernelAbi &abi = executable.abi();
  ASSERT_EQ(abi.inputs.size(), 3u);
  ASSERT_EQ(abi.outputs.size(), 1u);
  ASSERT_EQ(abi.arguments.size(), 4u);

  Bf16Buffer x(m, k, 0.0f);
  Bf16Buffer wg(k, n, 0.0f);
  Bf16Buffer wu(k, n, 0.0f);
  uint32_t state = 0x31d5u;
  fillSignedBf16(x, state);
  fillSignedBf16(wg, state);
  fillSignedBf16(wu, state);
  const auto originalX = x.storage;
  const auto originalWg = wg.storage;
  const auto originalWu = wu.storage;

  const size_t elements = static_cast<size_t>(m * n);
  std::vector<uint16_t> guarded(elements + 2, toBf16(-77.0f));
  uint16_t *destination = guarded.data() + 1;
  if (elements == 0)
    destination = nullptr;
  llk::InvocationBuffer2D output{
      {destination, destination, 0, m, n, n, 1},
      llk::InvocationElementType::BF16,
      elements == 0 ? 0 : guarded.size() * sizeof(uint16_t)};

  llvm::Error error =
      executable.invoke({x.checked(), wg.checked(), wu.checked()}, {output});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));

  for (int64_t row = 0; row < m; ++row) {
    for (int64_t column = 0; column < n; ++column) {
      float gate = 0.0f;
      float up = 0.0f;
      for (int64_t inner = 0; inner < k; ++inner) {
        const float value = storedBf16(x, static_cast<size_t>(row * k + inner));
        gate += value * storedBf16(wg, static_cast<size_t>(inner * n + column));
        up += value * storedBf16(wu, static_cast<size_t>(inner * n + column));
      }
      const float expected = gate / (1.0f + std::exp(-gate)) * up;
      const size_t index = static_cast<size_t>(row * n + column);
      const float actual = fromBf16(guarded[index + 1]);
      // bounded_fast evaluates silu with a cubic on its reduced interval; the
      // bound covers that approximation over the fixture's range plus BF16
      // output rounding. This is the same tolerance the static SwiGLU
      // acceptance uses, so a dynamic shape cannot buy a looser check.
      EXPECT_NEAR(actual, expected, 0.04f + 0.02f * std::abs(expected))
          << "shape " << m << "x" << n << "x" << k << " at " << row << ","
          << column;
    }
  }
  EXPECT_EQ(fromBf16(guarded.front()), -77.0f);
  EXPECT_EQ(fromBf16(guarded.back()), -77.0f);
  EXPECT_EQ(x.storage, originalX);
  EXPECT_EQ(wg.storage, originalWg);
  EXPECT_EQ(wu.storage, originalWu);
}

/// Invokes a 1 x 1 x K matmul with caller-supplied contents and returns the
/// raw BF16 bits of its single result.
uint16_t invokeSingleMatmul(llk::MappedExecutable &executable, Bf16Buffer &a,
                            Bf16Buffer &b, Bf16Buffer &init) {
  std::vector<uint16_t> guarded(3, toBf16(-77.0f));
  llk::InvocationBuffer2D output{
      {guarded.data() + 1, guarded.data() + 1, 0, 1, 1, 1, 1},
      llk::InvocationElementType::BF16,
      guarded.size() * sizeof(uint16_t)};
  llvm::Error error =
      executable.invoke({a.checked(), b.checked(), init.checked()}, {output});
  EXPECT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));
  EXPECT_EQ(fromBf16(guarded.front()), -77.0f);
  EXPECT_EQ(fromBf16(guarded.back()), -77.0f);
  return guarded[1];
}

//===----------------------------------------------------------------------===//
// Scratch ownership
//===----------------------------------------------------------------------===//

/// Counts allocations made by generated code so a repeated invocation can be
/// shown to release its local scratch and never adopt a caller pointer.
class AllocationTracker {
public:
  void *allocate(size_t size) {
    void *pointer = std::malloc(size);
    record(pointer, size);
    return pointer;
  }

  void *alignedAllocate(size_t alignment, size_t size) {
    void *pointer = std::aligned_alloc(alignment, size);
    record(pointer, size);
    return pointer;
  }

  void release(void *pointer) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto found = live_.find(pointer);
    if (found == live_.end()) {
      ++doubleFreeCount_;
      return;
    }
    live_.erase(found);
    freed_.insert(pointer);
    ++releaseCount_;
    std::free(pointer);
  }

  std::set<void *> outstandingPointers() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::set<void *> result;
    for (const auto &[pointer, size] : live_) {
      (void)size;
      result.insert(pointer);
    }
    return result;
  }

  bool freedAny(llvm::ArrayRef<void *> pointers) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (void *pointer : pointers)
      if (freed_.contains(pointer))
        return true;
    return false;
  }

  uint64_t doubleFreeCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return doubleFreeCount_;
  }
  uint64_t allocateCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocateCount_;
  }
  uint64_t releaseCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return releaseCount_;
  }

private:
  void record(void *pointer, size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pointer) {
      ++doubleFreeCount_;
      return;
    }
    live_[pointer] = size;
    ++allocateCount_;
  }

  mutable std::mutex mutex_;
  std::map<void *, size_t> live_;
  std::set<void *> freed_;
  uint64_t allocateCount_ = 0;
  uint64_t releaseCount_ = 0;
  uint64_t doubleFreeCount_ = 0;
};

thread_local AllocationTracker *activeTracker = nullptr;

void *allocateHook(size_t size) { return activeTracker->allocate(size); }
void *alignedAllocateHook(size_t alignment, size_t size) {
  return activeTracker->alignedAllocate(alignment, size);
}
void releaseHook(void *pointer) { activeTracker->release(pointer); }

class ActiveTracker {
public:
  explicit ActiveTracker(AllocationTracker &tracker) {
    EXPECT_EQ(activeTracker, nullptr);
    activeTracker = &tracker;
  }
  ~ActiveTracker() { activeTracker = nullptr; }
};

/// Fails unless `error` carries `needle`.
void expectInvocationError(llvm::Error error, llvm::StringRef needle) {
  ASSERT_TRUE(static_cast<bool>(error));
  const std::string message = llvm::toString(std::move(error));
  EXPECT_NE(message.find(needle.str()), std::string::npos) << message;
}

//===----------------------------------------------------------------------===//
// Compile once, run unrelated shapes
//===----------------------------------------------------------------------===//

TEST(DynamicMappedAcceptance, CompilesOnceAndExecutesUnrelatedMatmulShapes) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::Expected<DynamicArtifact> artifact = compileDynamicSource(
      context, kDynamicMatmulSource, "matmul_dynamic_root");
  ASSERT_TRUE(static_cast<bool>(artifact))
      << llvm::toString(artifact.takeError());

  // One compilation produced this artifact, and the ABI is symbolic: M, N and
  // K are bound by the descriptors, never frozen into the compiled code.
  EXPECT_EQ(artifact->compilations, 1u);
  const llk::KernelAbi &abi = artifact->executable->abi();
  EXPECT_EQ(abi.version, llk::kMappedAbiVersionV2);
  for (llvm::StringRef name : {"M", "N", "K"})
    EXPECT_TRUE(abi.symbols.count(name.str())) << name.str();
  ASSERT_EQ(abi.outputs[0].dimensions.size(), 2u);
  EXPECT_TRUE(abi.outputs[0].dimensions[0].isSymbol());
  EXPECT_TRUE(abi.outputs[0].dimensions[1].isSymbol());

  const std::array<std::array<int64_t, 3>, 5> shapes{
      std::array<int64_t, 3>{1, 1, 1},
      {7, 13, 5},
      {8, 32, 32},
      {9, 33, 35},
      {17, 65, 67}};
  for (const auto &shapeValues : shapes)
    expectMatmulShape(*artifact->executable, shapeValues[0], shapeValues[1],
                      shapeValues[2]);

  EXPECT_EQ(artifact->executable->successfulInvocations(), shapes.size());
  // The same compiled artifact served every triple; no exact-shape
  // recompilation happened.
  EXPECT_EQ(artifact->compilations, 1u);
}

TEST(DynamicMappedAcceptance, ServesUnrelatedShapesFromOneMixedStaticArtifact) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::Expected<DynamicArtifact> artifact =
      compileDynamicSource(context, kMixedMatmulSource, "matmul_mixed_root");
  ASSERT_TRUE(static_cast<bool>(artifact))
      << llvm::toString(artifact.takeError());

  // N was declared static, so the ABI records a constant there and symbols
  // only for the axes the source left open.
  EXPECT_EQ(artifact->compilations, 1u);
  const llk::KernelAbi &abi = artifact->executable->abi();
  EXPECT_EQ(abi.version, llk::kMappedAbiVersionV2);
  EXPECT_TRUE(abi.symbols.count("M"));
  EXPECT_TRUE(abi.symbols.count("K"));
  EXPECT_FALSE(abi.symbols.count("N"));
  ASSERT_EQ(abi.outputs[0].dimensions.size(), 2u);
  EXPECT_TRUE(abi.outputs[0].dimensions[0].isSymbol());
  ASSERT_TRUE(abi.outputs[0].dimensions[1].isConstant());
  EXPECT_EQ(abi.outputs[0].dimensions[1].getConstant(), 32);

  expectMatmulShape(*artifact->executable, 3, 32, 7);
  expectMatmulShape(*artifact->executable, 5, 32, 70);
  expectMatmulShape(*artifact->executable, 0, 32, 11);
  EXPECT_EQ(artifact->executable->successfulInvocations(), 3u);
  EXPECT_EQ(artifact->compilations, 1u);
}

TEST(DynamicMappedAcceptance, EmptyDimensionsDoNoWorkAndPreserveTheResult) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::Expected<DynamicArtifact> artifact = compileDynamicSource(
      context, kDynamicMatmulSource, "matmul_dynamic_root");
  ASSERT_TRUE(static_cast<bool>(artifact))
      << llvm::toString(artifact.takeError());

  // M = 0 and N = 0 are valid invocations with no touched output element.
  // The empty ports arrive as a null pointer with zero bytes: if the kernel
  // executed any movement or compute for them it would fault, so a clean
  // return is evidence that the empty extent skipped the work rather than
  // masking it.
  expectMatmulShape(*artifact->executable, 0, 5, 13);
  expectMatmulShape(*artifact->executable, 7, 0, 13);
  expectMatmulShape(*artifact->executable, 0, 0, 13);

  // K = 0 keeps the contraction's initialization: with no reduction terms the
  // destination is exactly the initializer, byte for byte.
  const int64_t m = 7, n = 5;
  Bf16Buffer a(m, 0, 0.0f);
  Bf16Buffer b(0, n, 0.0f);
  Bf16Buffer init(m, n, 0.0f);
  uint32_t state = 0x77a1u;
  fillSignedBf16(init, state);
  std::vector<uint16_t> guarded(static_cast<size_t>(m * n) + 2, toBf16(-77.0f));
  llk::InvocationBuffer2D output{
      {guarded.data() + 1, guarded.data() + 1, 0, m, n, n, 1},
      llk::InvocationElementType::BF16,
      guarded.size() * sizeof(uint16_t)};
  llvm::Error error = artifact->executable->invoke(
      {a.checked(), b.checked(), init.checked()}, {output});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));
  for (int64_t index = 0; index < m * n; ++index)
    EXPECT_EQ(guarded[static_cast<size_t>(index) + 1],
              init.storage[static_cast<size_t>(index)])
        << "element " << index;
  EXPECT_EQ(fromBf16(guarded.front()), -77.0f);
  EXPECT_EQ(fromBf16(guarded.back()), -77.0f);

  EXPECT_EQ(artifact->compilations, 1u);
}

TEST(DynamicMappedAcceptance,
     RepeatedInvocationsReleaseScratchAndBorrowCallerBuffers) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  workload::WorkloadContract contract;
  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> module =
      exportDynamicSource(context, kDynamicMatmulSource, "matmul_dynamic_root",
                          contract);
  ASSERT_TRUE(static_cast<bool>(module)) << llvm::toString(module.takeError());
  auto entry = findMicroKernel(**module)
                   ->getAttrOfType<mlir::StringAttr>("sym_name")
                   .getValue()
                   .str();

  llvm::Expected<llk::MappedCompilation> lowered =
      searchAndCompile(context, **module, contract, entry, kDefaultBackend,
                       llk::MappedStop::Lowered, nullptr);
  ASSERT_TRUE(static_cast<bool>(lowered))
      << llvm::toString(lowered.takeError());
  ASSERT_TRUE(lowered->module);

  // No output-sized temporary: every allocation the lowered kernel keeps is a
  // static tile. The caller's dynamic destination is storage it already owns,
  // so the generated code must never allocate an M x N buffer.
  bool dynamicAllocation = false;
  lowered->module->walk([&](mlir::memref::AllocOp alloc) {
    dynamicAllocation |= !alloc.getType().hasStaticShape();
  });
  EXPECT_FALSE(dynamicAllocation)
      << "a whole-output temporary was materialized";

  llvm::Expected<llk::PreparedMappedKernel> prepared =
      llk::prepareMappedKernelForInvocation(*lowered->module, entry);
  ASSERT_TRUE(static_cast<bool>(prepared))
      << llvm::toString(prepared.takeError());
  EXPECT_EQ(prepared->abi.version, llk::kMappedAbiVersionV2);

  AllocationTracker tracker;
  ActiveTracker active(tracker);
  llk::testing::TestAllocatorHooks hooks{allocateHook, alignedAllocateHook,
                                         releaseHook};
  llvm::Expected<std::unique_ptr<llk::MappedExecutable>> executable =
      llk::testing::createMappedExecutableForTest(std::move(*prepared), {},
                                                  hooks);
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());

  const std::set<void *> empty;
  const std::array<std::array<int64_t, 3>, 3> shapes{
      std::array<int64_t, 3>{1, 1, 1}, {7, 13, 5}, {17, 65, 67}};
  for (const auto &shape : shapes) {
    expectMatmulShape(**executable, shape[0], shape[1], shape[2]);
    // Each invocation allocated its local tiles and released them again.
    EXPECT_EQ(tracker.outstandingPointers(), empty);
  }
  EXPECT_GT(tracker.allocateCount(), 0u);
  EXPECT_EQ(tracker.allocateCount(), tracker.releaseCount());
  EXPECT_EQ(tracker.doubleFreeCount(), 0u);
}

/// One F32 buffer with caller-owned storage.
struct F32Buffer {
  std::vector<float> storage;
  int64_t rows = 0;
  int64_t columns = 0;

  F32Buffer(int64_t rowsValue, int64_t columnsValue)
      : storage(static_cast<size_t>(rowsValue * columnsValue), 0.0f),
        rows(rowsValue), columns(columnsValue) {}

  llk::InvocationBuffer2D checked() {
    const uint64_t bytes = storage.size() * sizeof(float);
    float *data = storage.empty() ? nullptr : storage.data();
    return {{data, data, 0, rows, columns, columns, 1},
            llk::InvocationElementType::F32,
            bytes};
  }
};

/// Invokes the direct elementwise Micro-IR kernel for one runtime shape.
void expectElementwiseShape(llk::MappedExecutable &executable, int64_t m,
                            int64_t n) {
  F32Buffer a(m, n);
  F32Buffer b(m, n);
  for (size_t index = 0; index < a.storage.size(); ++index) {
    a.storage[index] = static_cast<float>(index) * 0.5f + 0.25f;
    b.storage[index] = -static_cast<float>(index) * 1.5f;
  }
  const auto originalA = a.storage;
  const auto originalB = b.storage;

  const size_t elements = static_cast<size_t>(m * n);
  std::vector<float> guarded(elements + 2, -77.0f);
  float *destination = guarded.data() + 1;
  if (elements == 0)
    destination = nullptr;
  llk::InvocationBuffer2D output{
      {destination, destination, 0, m, n, n, 1},
      llk::InvocationElementType::F32,
      elements == 0 ? 0 : guarded.size() * sizeof(float)};

  llvm::Error error = executable.invoke({a.checked(), b.checked()}, {output});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));

  for (size_t index = 0; index < elements; ++index)
    EXPECT_FLOAT_EQ(guarded[index + 1], originalA[index] + originalB[index])
        << "shape " << m << "x" << n << " at " << index;
  EXPECT_EQ(guarded.front(), -77.0f);
  EXPECT_EQ(guarded.back(), -77.0f);
  EXPECT_EQ(a.storage, originalA);
  EXPECT_EQ(b.storage, originalB);
}

TEST(DynamicMappedAcceptance, ExecutesADirectDynamicElementwiseMicroKernel) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> source =
      llvm::MemoryBuffer::getFile(
          std::string(LLK_SOURCE_DIR) +
          "/test/Conversion/MicroMapping/dynamic_elementwise.mlir");
  ASSERT_TRUE(static_cast<bool>(source)) << source.getError().message();
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>((*source)->getBuffer(), &context);
  ASSERT_TRUE(module) << "the dynamic elementwise fixture did not parse";

  // The contract the fixture implements: two read-only M x N inputs and a
  // caller-owned M x N output. It has no M/N/K contraction axes.
  const shape::Dimension m = shape::Dimension::symbol("M");
  const shape::Dimension n = shape::Dimension::symbol("N");
  workload::WorkloadContract contract;
  contract.version = 2;
  contract.ports = {
      {"a",
       workload::PortDirection::Input,
       workload::PortEffect::Read,
       {m, n},
       "f32"},
      {"b",
       workload::PortDirection::Input,
       workload::PortEffect::Read,
       {m, n},
       "f32"},
      {"out",
       workload::PortDirection::Output,
       workload::PortEffect::Write,
       {m, n},
       "f32"},
  };
  contract.symbols = {{"M", {}}, {"N", {}}};
  llvm::Expected<mlir::DictionaryAttr> encoded =
      workload::encodeWorkloadContract(&context, contract);
  ASSERT_TRUE(static_cast<bool>(encoded))
      << llvm::toString(encoded.takeError());
  mlir::Operation *kernel = findMicroKernel(*module);
  ASSERT_TRUE(kernel);
  kernel->setAttr("micro.workload_contract", *encoded);

  llvm::Expected<DynamicArtifact> artifact =
      compileMappedModule(context, *module, contract, kDefaultBackend);
  ASSERT_TRUE(static_cast<bool>(artifact))
      << llvm::toString(artifact.takeError());
  EXPECT_EQ(artifact->compilations, 1u);

  const llk::KernelAbi &abi = artifact->executable->abi();
  EXPECT_EQ(abi.version, llk::kMappedAbiVersionV2);
  EXPECT_EQ(abi.inputs.size(), 2u);
  EXPECT_EQ(abi.outputs.size(), 1u);
  EXPECT_TRUE(abi.symbols.count("M"));
  EXPECT_TRUE(abi.symbols.count("N"));

  expectElementwiseShape(*artifact->executable, 1, 1);
  expectElementwiseShape(*artifact->executable, 7, 13);
  expectElementwiseShape(*artifact->executable, 8, 16);
  expectElementwiseShape(*artifact->executable, 9, 33);
  expectElementwiseShape(*artifact->executable, 17, 65);
  expectElementwiseShape(*artifact->executable, 0, 13);
  expectElementwiseShape(*artifact->executable, 7, 0);
  EXPECT_EQ(artifact->executable->successfulInvocations(), 7u);
  EXPECT_EQ(artifact->compilations, 1u);
}

//===----------------------------------------------------------------------===//
// Selected emission
//===----------------------------------------------------------------------===//

/// The block argument bound to the kernel's output port, or null.
mlir::BlockArgument outputArgument(mlir::func::FuncOp function) {
  auto bindings = function->getAttrOfType<mlir::ArrayAttr>("argument_bindings");
  if (!bindings || bindings.size() != function.getNumArguments())
    return {};
  for (size_t index = 0; index < bindings.size(); ++index) {
    auto entry = mlir::dyn_cast<mlir::DictionaryAttr>(bindings[index]);
    auto role =
        entry ? entry.getAs<mlir::StringAttr>("role") : mlir::StringAttr();
    if (role && role.getValue() == "output")
      return function.getArgument(static_cast<unsigned>(index));
  }
  return {};
}

bool isRuntimeBound(mlir::Value value) {
  return !value.getDefiningOp<mlir::arith::ConstantOp>() &&
         !value.getDefiningOp<mlir::arith::ConstantIndexOp>();
}

bool isRuntimeSize(mlir::OpFoldResult size) {
  if (isa<mlir::Attribute>(size))
    return false;
  return isRuntimeBound(cast<mlir::Value>(size));
}

/// True when `value` reaches `destination` through loop-carried state or a
/// view. Structured loops thread the caller's destination as an iter_arg, so
/// the store's operand is the loop's argument, not the function argument.
bool aliasesDestination(mlir::Value value, mlir::BlockArgument destination,
                        unsigned depth = 0) {
  if (value == destination || depth > 16)
    return value == destination;
  if (auto argument = mlir::dyn_cast<mlir::BlockArgument>(value)) {
    auto loop = mlir::dyn_cast_or_null<mlir::scf::ForOp>(
        argument.getOwner()->getParentOp());
    if (!loop)
      return false;
    unsigned index = argument.getArgNumber() - loop.getNumInductionVars();
    return index < loop.getInitArgs().size() &&
           aliasesDestination(loop.getInitArgs()[index], destination,
                              depth + 1);
  }
  if (auto view = value.getDefiningOp<mlir::memref::SubViewOp>())
    return aliasesDestination(view.getSource(), destination, depth + 1);
  if (auto cast = value.getDefiningOp<mlir::memref::CastOp>())
    return aliasesDestination(cast.getSource(), destination, depth + 1);
  return false;
}

/// Compiles the dynamic matmul for the selected target and stops before the
/// JIT, so the code the selected pipeline produced is the artifact under test.
llvm::Expected<llk::MappedCompilation>
compileSelectedDynamicLowered(mlir::MLIRContext &context,
                              workload::WorkloadContract &contract,
                              std::string &entrySymbol) {
  llvm::Expected<mlir::OwningOpRef<mlir::ModuleOp>> module =
      exportDynamicSource(context, kDynamicMatmulSource, "matmul_dynamic_root",
                          contract);
  if (!module)
    return module.takeError();
  mlir::Operation *kernel = findMicroKernel(**module);
  if (!kernel)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the export produced no micro.kernel");
  entrySymbol =
      kernel->getAttrOfType<mlir::StringAttr>("sym_name").getValue().str();
  return searchAndCompile(context, **module, contract, entrySymbol,
                          llk::MappedBackend::SelectedTarget,
                          llk::MappedStop::Lowered, nullptr);
}

/// The selected pipeline must keep the dynamic boundary realizations: no
/// whole-output temporary, runtime loop bounds, and clipped slices of the
/// caller's destination. A static padded lowering would satisfy the numbers
/// but not these facts, which is why they are asserted on the emitted IR.
TEST(DynamicMappedSelected, SelectedLoweringKeepsBoundedBoundariesAndRuntimeK) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  workload::WorkloadContract contract;
  std::string entrySymbol;
  llvm::Expected<llk::MappedCompilation> lowered =
      compileSelectedDynamicLowered(context, contract, entrySymbol);
  ASSERT_TRUE(static_cast<bool>(lowered))
      << llvm::toString(lowered.takeError());
  ASSERT_TRUE(lowered->module);
  EXPECT_EQ(lowered->backend, llk::MappedBackend::SelectedTarget);

  // 1. No compiler-owned whole-output buffer. Every allocation that remains is
  //    a fixed-capacity tile.
  bool dynamicAllocation = false;
  unsigned staticAllocations = 0;
  lowered->module->walk([&](mlir::memref::AllocOp alloc) {
    if (!alloc.getType().hasStaticShape())
      dynamicAllocation = true;
    else
      ++staticAllocations;
  });
  EXPECT_FALSE(dynamicAllocation)
      << "the selected lowering materialised a whole-output temporary";
  EXPECT_GT(staticAllocations, 0u);

  auto function =
      lowered->module->lookupSymbol<mlir::func::FuncOp>(entrySymbol);
  ASSERT_TRUE(function);

  // 2. Runtime extents survive to the selected signature.
  EXPECT_EQ(function.getNumArguments(), 4u);
  for (mlir::BlockArgument argument : function.getArguments()) {
    auto memref = mlir::dyn_cast<mlir::MemRefType>(argument.getType());
    ASSERT_TRUE(memref);
    EXPECT_TRUE(memref.isDynamicDim(0));
    EXPECT_TRUE(memref.isDynamicDim(1));
  }

  // 3. Runtime K and runtime tile counts: at least one loop per dynamic axis
  //    whose bound is not a compile-time constant.
  unsigned runtimeLoops = 0;
  function.walk([&](mlir::scf::ForOp loop) {
    if (isRuntimeBound(loop.getUpperBound()))
      ++runtimeLoops;
  });
  EXPECT_GE(runtimeLoops, 3u)
      << "the selected lowering froze a runtime tile count or K";

  // 4. Clipped writes: the caller's destination is reached through a
  //    dynamically sized subview, so only the valid rectangle is written.
  mlir::BlockArgument destination = outputArgument(function);
  ASSERT_TRUE(destination);
  unsigned dynamicDestinationViews = 0;
  function.walk([&](mlir::memref::SubViewOp view) {
    if (!aliasesDestination(view.getSource(), destination))
      return;
    for (mlir::OpFoldResult size : view.getMixedSizes())
      if (isRuntimeSize(size))
        ++dynamicDestinationViews;
  });
  EXPECT_GT(dynamicDestinationViews, 0u)
      << "the selected lowering wrote a full tile instead of the valid "
         "rectangle";

  // 5. Bounded source accesses: dynamically sized subviews of an input.
  unsigned dynamicInputViews = 0;
  function.walk([&](mlir::memref::SubViewOp view) {
    bool fromInput = false;
    for (mlir::BlockArgument argument : function.getArguments())
      fromInput |= view.getSource() == argument && argument != destination;
    if (!fromInput)
      return;
    for (mlir::OpFoldResult size : view.getMixedSizes())
      if (isRuntimeSize(size))
        ++dynamicInputViews;
  });
  EXPECT_GT(dynamicInputViews, 0u)
      << "the selected lowering padded a whole source instead of bounding it";
}

#ifdef LLK_REQUIRE_SELECTED_TARGET
TEST(DynamicMappedSelected, SelectedAvx2ExecutesCompileOnceDynamicAcceptance) {
  if (!selectedAvx2Available()) {
    if (avx2DisabledByTestOverride())
      FAIL() << "test override disabled required selected AVX2 execution";
    else
      FAIL() << "required selected AVX2 execution needs an x86_64 AVX2 host";
  }

  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::Expected<DynamicArtifact> artifact =
      compileDynamicSource(context, kDynamicMatmulSource, "matmul_dynamic_root",
                           llk::MappedBackend::SelectedTarget);
  ASSERT_TRUE(static_cast<bool>(artifact))
      << llvm::toString(artifact.takeError());
  EXPECT_EQ(artifact->compilations, 1u);
  EXPECT_EQ(artifact->executable->executionIdentity().contains(
                "backend=selected-target"),
            true);

  // The artifact carries the ISA contract the runtime enforces, and this host
  // satisfies it.
  ASSERT_TRUE(artifact->codegenRequirements.has_value());
  const std::string triple = llvm::sys::getDefaultTargetTriple();
  const llvm::StringRef architecture = llvm::StringRef(triple).split('-').first;
  EXPECT_FALSE(static_cast<bool>(mlir::llk::mapping::checkHostRequirements(
      *artifact->codegenRequirements, architecture,
      llvm::sys::getHostCPUFeatures())));

  for (const auto &shapeValues :
       std::array<std::array<int64_t, 3>, 5>{std::array<int64_t, 3>{1, 1, 1},
                                             {7, 13, 5},
                                             {8, 32, 32},
                                             {9, 33, 35},
                                             {17, 65, 67}})
    expectMatmulShape(*artifact->executable, shapeValues[0], shapeValues[1],
                      shapeValues[2]);
  expectMatmulShape(*artifact->executable, 0, 5, 13);
  EXPECT_EQ(artifact->executable->successfulInvocations(), 6u);
  EXPECT_EQ(artifact->compilations, 1u);
  // The selected bundles were verified against the plan even though the
  // movement/contraction keys hand their operations to the shared lowering:
  // the handoff is accounted for, not silently treated as reference output.
  EXPECT_GT(artifact->selectedGroupsVerified, 0u);
}
#endif

TEST(DynamicMappedAcceptance, ExecutesExternalOutputsLargerThanLocalSram) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  // Local SRAM is 8 KiB. The output below is 65 x 129 BF16 = 16 KiB, so the
  // plan is capacity-legal only if the caller's destination is external
  // storage rather than local scratch.
  const uint64_t sramBytes = 8192;
  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      makeSmallSramTarget(sramBytes);
  ASSERT_TRUE(static_cast<bool>(target)) << llvm::toString(target.takeError());
  // The profile edit took effect: this is the source of the capacity bound,
  // not a comment about it.
  bool foundSram = false;
  for (const mlir::llk::machine::MemoryNode &memory :
       (*target)->machine().memories)
    if (memory.kind == "sram") {
      EXPECT_EQ(memory.capacityBytes, sramBytes);
      foundSram = true;
    }
  ASSERT_TRUE(foundSram);

  llvm::Expected<DynamicArtifact> artifact =
      compileDynamicSource(context, kDynamicMatmulSource, "matmul_dynamic_root",
                           kDefaultBackend, std::move(*target));
  ASSERT_TRUE(static_cast<bool>(artifact))
      << llvm::toString(artifact.takeError());
  EXPECT_EQ(artifact->compilations, 1u);

  const int64_t m = 65, n = 129, k = 33;
  EXPECT_GT(static_cast<uint64_t>(m * n) * sizeof(uint16_t), sramBytes);

  expectMatmulShape(*artifact->executable, m, n, k);
  expectMatmulShape(*artifact->executable, 9, 33, 35);
  EXPECT_EQ(artifact->executable->successfulInvocations(), 2u);
  EXPECT_EQ(artifact->compilations, 1u);
}

TEST(DynamicMappedAcceptance, RejectsWuDimensionDisagreementBeforeEntry) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::Expected<DynamicArtifact> artifact = compileDynamicSource(
      context, kDynamicSwigluSource, "swiglu_dynamic_root");
  ASSERT_TRUE(static_cast<bool>(artifact))
      << llvm::toString(artifact.takeError());
  llk::MappedExecutable &executable = *artifact->executable;

  const int64_t m = 7, n = 13, k = 5;
  Bf16Buffer x(m, k, 0.0f);
  Bf16Buffer wg(k, n, 0.0f);
  Bf16Buffer wu(k, n, 0.0f);
  uint32_t state = 0x20b9u;
  fillSignedBf16(x, state);
  fillSignedBf16(wg, state);
  fillSignedBf16(wu, state);
  // Establish that this exact invocation is admissible, so the refusals below
  // are about the mutated descriptor and not about the fixture.
  expectSwigluShape(executable, m, n, k);
  ASSERT_EQ(executable.successfulInvocations(), 1u);

  std::vector<uint16_t> guarded(static_cast<size_t>(m * n) + 2, toBf16(-77.0f));
  llk::InvocationBuffer2D output{
      {guarded.data() + 1, guarded.data() + 1, 0, m, n, n, 1},
      llk::InvocationElementType::BF16,
      guarded.size() * sizeof(uint16_t)};

  // Wu must reduce the same K and produce the same N: both are shared symbols,
  // so either disagreement is a contract violation, not a different shape.
  Bf16Buffer wrongN(k, 12, 0.0f);
  expectInvocationError(
      executable.invoke({x.checked(), wg.checked(), wrongN.checked()},
                        {output}),
      "conflict");

  Bf16Buffer wrongK(4, n, 0.0f);
  expectInvocationError(
      executable.invoke({x.checked(), wg.checked(), wrongK.checked()},
                        {output}),
      "conflict");

  EXPECT_EQ(executable.successfulInvocations(), 1u);
  for (uint16_t value : guarded)
    EXPECT_EQ(fromBf16(value), -77.0f)
        << "a rejected invocation wrote through the destination";
}

TEST(DynamicMappedAcceptance, RejectsMalformedOutputDescriptorsBeforeEntry) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::Expected<DynamicArtifact> artifact = compileDynamicSource(
      context, kDynamicMatmulSource, "matmul_dynamic_root");
  ASSERT_TRUE(static_cast<bool>(artifact))
      << llvm::toString(artifact.takeError());
  llk::MappedExecutable &executable = *artifact->executable;

  const int64_t m = 7, n = 7, k = 7;
  Bf16Buffer a(m, k, 1.0f);
  Bf16Buffer b(k, n, 1.0f);
  Bf16Buffer init(m, n, 0.0f);
  std::vector<uint16_t> guarded(static_cast<size_t>(m * n) + 2, toBf16(-77.0f));
  llk::InvocationBuffer2D good{
      {guarded.data() + 1, guarded.data() + 1, 0, m, n, n, 1},
      llk::InvocationElementType::BF16,
      guarded.size() * sizeof(uint16_t)};

  // The admissible call first: every element is the K=7 reduction of ones.
  ASSERT_FALSE(static_cast<bool>(
      executable.invoke({a.checked(), b.checked(), init.checked()}, {good})));
  for (int64_t index = 0; index < m * n; ++index)
    EXPECT_EQ(fromBf16(guarded[static_cast<size_t>(index) + 1]), 7.0f);
  ASSERT_EQ(executable.successfulInvocations(), 1u);

  llk::InvocationBuffer2D malformed = good;
  malformed.descriptor.aligned = nullptr;
  malformed.allocationBytes = 0;
  expectInvocationError(
      executable.invoke({a.checked(), b.checked(), init.checked()},
                        {malformed}),
      "null allocation");

  malformed = good;
  malformed.elementType = llk::InvocationElementType::F32;
  expectInvocationError(
      executable.invoke({a.checked(), b.checked(), init.checked()},
                        {malformed}),
      "dtype");

  malformed = good;
  malformed.descriptor.stride0 = n + 1;
  expectInvocationError(
      executable.invoke({a.checked(), b.checked(), init.checked()},
                        {malformed}),
      "compact");

  malformed = good;
  malformed.descriptor.offset = 1;
  expectInvocationError(
      executable.invoke({a.checked(), b.checked(), init.checked()},
                        {malformed}),
      "offset");

  malformed = good;
  // One byte short of the touched M x N span, so the canary slack cannot hide
  // the undersized allocation.
  malformed.allocationBytes =
      static_cast<uint64_t>(m * n) * sizeof(uint16_t) - 1;
  expectInvocationError(
      executable.invoke({a.checked(), b.checked(), init.checked()},
                        {malformed}),
      "allocation");

  malformed = good;
  malformed.descriptor.size1 = -1;
  expectInvocationError(
      executable.invoke({a.checked(), b.checked(), init.checked()},
                        {malformed}),
      "negative");

  // The caller-owned destination must not alias a read-only input.
  llk::InvocationBuffer2D aliased{
      {a.storage.data(), a.storage.data(), 0, m, n, n, 1},
      llk::InvocationElementType::BF16,
      a.storage.size() * sizeof(uint16_t)};
  expectInvocationError(
      executable.invoke({a.checked(), b.checked(), init.checked()}, {aliased}),
      "overlap");

  // Every refusal happened before entry: the generated code ran once, no
  // output element changed, and the overlapping destination was not written.
  EXPECT_EQ(executable.successfulInvocations(), 1u);
  for (int64_t index = 0; index < m * n; ++index)
    EXPECT_EQ(fromBf16(guarded[static_cast<size_t>(index) + 1]), 7.0f);
  EXPECT_EQ(guarded.front(), toBf16(-77.0f));
  EXPECT_EQ(guarded.back(), toBf16(-77.0f));
  for (uint16_t value : a.storage)
    EXPECT_EQ(fromBf16(value), 1.0f);
}

TEST(DynamicMappedAcceptance, ExactKTailsPreserveSignedZeroAndNonfiniteValues) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::Expected<DynamicArtifact> artifact =
      compileDynamicSource(context, kStrictMatmulSource, "matmul_strict_root");
  ASSERT_TRUE(static_cast<bool>(artifact))
      << llvm::toString(artifact.takeError());

  // K = 65 leaves a one-element tail beyond the BK = 64 capacity.
  const int64_t k = 65;
  Bf16Buffer a(1, k, 0.0f);
  Bf16Buffer b(k, 1, 0.0f);
  Bf16Buffer init(1, 1, 0.0f);

  // Signed zero: every valid product is -0.0 and so is the initializer, so
  // the exact reduction is -0.0. A full-capacity padded tail would add +0.0
  // lanes and turn the result into +0.0, which is a different value under the
  // strict contract.
  for (int64_t inner = 0; inner < k; ++inner) {
    a.storage[static_cast<size_t>(inner)] = toBf16(-0.0f);
    b.storage[static_cast<size_t>(inner)] = toBf16(0.0f);
  }
  init.storage[0] = toBf16(-0.0f);
  uint16_t sum = invokeSingleMatmul(*artifact->executable, a, b, init);
  EXPECT_TRUE(std::signbit(fromBf16(sum)))
      << "a padded K tail added +0.0 and lost the signed zero";

  // A nonfinite value in the tail must be reduced, not dropped.
  std::fill(a.storage.begin(), a.storage.end(), toBf16(0.0f));
  std::fill(b.storage.begin(), b.storage.end(), toBf16(0.0f));
  init.storage[0] = toBf16(0.0f);
  a.storage[static_cast<size_t>(k - 1)] =
      toBf16(std::numeric_limits<float>::infinity());
  b.storage[static_cast<size_t>(k - 1)] = toBf16(1.0f);
  uint16_t infinity = invokeSingleMatmul(*artifact->executable, a, b, init);
  EXPECT_TRUE(std::isinf(fromBf16(infinity)));
  EXPECT_GT(fromBf16(infinity), 0.0f);

  b.storage[static_cast<size_t>(k - 1)] =
      toBf16(std::numeric_limits<float>::quiet_NaN());
  uint16_t notANumber = invokeSingleMatmul(*artifact->executable, a, b, init);
  EXPECT_TRUE(std::isnan(fromBf16(notANumber)));

  // Ordinary strict values on both sides of a capacity boundary still match
  // the exact-K f32 reference.
  expectMatmulShape(*artifact->executable, 9, 33, 35);
  expectMatmulShape(*artifact->executable, 17, 65, 67);
  EXPECT_EQ(artifact->executable->successfulInvocations(), 5u);
  EXPECT_EQ(artifact->compilations, 1u);
}

TEST(DynamicMappedAcceptance, CompilesOnceAndExecutesUnrelatedSwiGLUShapes) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::Expected<DynamicArtifact> artifact = compileDynamicSource(
      context, kDynamicSwigluSource, "swiglu_dynamic_root");
  ASSERT_TRUE(static_cast<bool>(artifact))
      << llvm::toString(artifact.takeError());

  EXPECT_EQ(artifact->compilations, 1u);
  const llk::KernelAbi &abi = artifact->executable->abi();
  EXPECT_EQ(abi.version, llk::kMappedAbiVersionV2);
  for (llvm::StringRef name : {"M", "N", "K"})
    EXPECT_TRUE(abi.symbols.count(name.str())) << name.str();
  // SwiGLU's destination is written, never read: it is the output port, and
  // the three read inputs are X, Wg and Wu.
  EXPECT_EQ(abi.inputs.size(), 3u);
  EXPECT_EQ(abi.outputs.size(), 1u);

  const std::array<std::array<int64_t, 3>, 5> shapes{
      std::array<int64_t, 3>{1, 1, 1},
      {7, 13, 5},
      {8, 32, 32},
      {9, 33, 35},
      {17, 65, 67}};
  for (const auto &shapeValues : shapes)
    expectSwigluShape(*artifact->executable, shapeValues[0], shapeValues[1],
                      shapeValues[2]);

  // Empty M or N touches no output element; empty K reduces to the epilogue
  // evaluated at zero, silu(0) * 0 = 0.
  expectSwigluShape(*artifact->executable, 0, 5, 13);
  expectSwigluShape(*artifact->executable, 7, 0, 13);
  expectSwigluShape(*artifact->executable, 7, 5, 0);

  EXPECT_EQ(artifact->executable->successfulInvocations(), shapes.size() + 3);
  EXPECT_EQ(artifact->compilations, 1u);
}

TEST(DynamicMappedAcceptance, ServesUnrelatedShapesFromOneMixedStaticSwiGLU) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::Expected<DynamicArtifact> artifact =
      compileDynamicSource(context, kMixedSwigluSource, "swiglu_mixed_root");
  ASSERT_TRUE(static_cast<bool>(artifact))
      << llvm::toString(artifact.takeError());

  EXPECT_EQ(artifact->compilations, 1u);
  const llk::KernelAbi &abi = artifact->executable->abi();
  EXPECT_TRUE(abi.symbols.count("M"));
  EXPECT_TRUE(abi.symbols.count("K"));
  EXPECT_FALSE(abi.symbols.count("N"));
  ASSERT_EQ(abi.outputs[0].dimensions.size(), 2u);
  ASSERT_TRUE(abi.outputs[0].dimensions[1].isConstant());
  EXPECT_EQ(abi.outputs[0].dimensions[1].getConstant(), 32);

  expectSwigluShape(*artifact->executable, 3, 32, 7);
  expectSwigluShape(*artifact->executable, 5, 32, 70);
  expectSwigluShape(*artifact->executable, 0, 32, 11);
  EXPECT_EQ(artifact->executable->successfulInvocations(), 3u);
  EXPECT_EQ(artifact->compilations, 1u);
}

} // namespace

int main(int argc, char **argv) {
  bool requireSelectedTarget = false;
  for (int index = 1; index < argc;) {
    if (std::strcmp(argv[index], "--require-selected-target") == 0) {
      requireSelectedTarget = true;
      for (int move = index; move + 1 < argc; ++move)
        argv[move] = argv[move + 1];
      --argc;
      continue;
    }
    ++index;
  }

#ifdef LLK_REQUIRE_SELECTED_TARGET
  if (!requireSelectedTarget) {
    std::fprintf(stderr, "DynamicMappedAVX2Acceptance requires "
                         "--require-selected-target\n");
    return 2;
  }
  // A required selected run on a host that cannot execute AVX2 is a failure,
  // never a skip: otherwise reference success could be reported as selected
  // evidence. Listing tests runs nothing, so it stays host-independent and the
  // command-line contract can be checked anywhere.
  bool listingTests = false;
  for (int index = 1; index < argc; ++index)
    listingTests |= std::strcmp(argv[index], "--gtest_list_tests") == 0;
  if (!listingTests && !selectedAvx2Available()) {
    if (avx2DisabledByTestOverride())
      std::fprintf(stderr,
                   "test override disabled required selected AVX2 execution\n");
    else
      std::fprintf(
          stderr,
          "required selected AVX2 execution needs an x86_64 AVX2 host\n");
    return 1;
  }
#else
  if (requireSelectedTarget) {
    std::fprintf(stderr, "--require-selected-target is only valid for the "
                         "selected-target executable\n");
    return 2;
  }
#endif

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
