//===- mapped_acceptance.cpp - Numeric acceptance for mapped kernels ------===//
//
// Stage C8's numeric chains. Everything before this proves that IR traverses,
// that a plan binds, that an ABI is declared; none of it runs a *compiler-
// generated* kernel and checks what it computed. These tests do exactly that,
// and only that: the LLK source below is the same source the IR fixtures use,
// so the kernel under test is produced live by the export rather than
// hand-authored, and every element of every result is checked.
//
// The chains are host-portable on purpose. The exported program is Linalg, SCF
// and memref, and the JIT compiles it for whatever machine is running, so the
// numbers are the same on arm64 and on x86. What cannot be checked here is that
// AVX2 *instructions* were emitted -- that needs an AVX2 runner, and this file
// does not pretend otherwise.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/LLKToMicro/LLKToMicro.h"
#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Machine/MachineModelLoader.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/LayoutConstraints.h"
#include "LLK/Mapping/MappingRules.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/WorkloadGraph.h"
#include "LLK/Runtime/MappedExecutable.h"
#include "LLK/Target/X86/Mapping/AVX2MappingTarget.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
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

#include "llvm/Config/llvm-config.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/TargetParser/Host.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifndef LLK_SOURCE_DIR
#error "LLK_SOURCE_DIR must name the repository root"
#endif

namespace {

using mlir::llk::mapping::CoveringPlan;
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

bool avx2DisabledByTestOverride() {
  const char *disabled = std::getenv("LLK_TEST_DISABLE_AVX2");
  return disabled && std::strcmp(disabled, "1") == 0;
}

bool selectedAvx2Available() {
  if (avx2DisabledByTestOverride())
    return false;
  const std::string triple = llvm::sys::getDefaultTargetTriple();
  if (triple.rfind("x86_64-", 0) != 0)
    return false;
  const auto features = llvm::sys::getHostCPUFeatures();
  auto avx2 = features.find("avx2");
  return avx2 != features.end() && avx2->second;
}

/// The LLK source of `test/Conversion/MicroMapping/matmul_e2e.mlir`. The
/// fixture's whole point is that the kernel is *exported* rather than written
/// by hand, so the acceptance chain starts from the same place the IR fixture
/// does.
constexpr llvm::StringLiteral kMatmulSource = R"mlir(
module {
  func.func @matmul(%a: tensor<16x64xbf16>, %b: tensor<64x64xbf16>,
                    %init: tensor<16x64xbf16>) -> tensor<16x64xbf16> {
    %y = llk.matmul ins(%a, %b : tensor<16x64xbf16>, tensor<64x64xbf16>)
        outs(%init : tensor<16x64xbf16>)
        {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>}
        -> tensor<16x64xbf16>
    return %y : tensor<16x64xbf16>
  }
}
)mlir";

/// The LLK source of `test/Conversion/MicroMapping/swiglu_e2e.mlir`.
constexpr llvm::StringLiteral kSwigluSource = R"mlir(
module {
  func.func @swiglu(%x: tensor<16x64xbf16>, %wg: tensor<64x64xbf16>,
                    %wu: tensor<64x64xbf16>, %init: tensor<16x64xbf16>)
      -> tensor<16x64xbf16> {
    %y = llk.fused_swiglu ins(%x, %wg, %wu : tensor<16x64xbf16>,
                              tensor<64x64xbf16>, tensor<64x64xbf16>)
        outs(%init : tensor<16x64xbf16>)
        {accumulator_type = f32, activation = #llk.activation<silu>,
         math_mode = #llk.math_mode<bounded_fast>}
        -> tensor<16x64xbf16>
    return %y : tensor<16x64xbf16>
  }
}
)mlir";

mlir::DialectRegistry buildRegistry() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::llk::LLKDialect, mlir::micro::MicroDialect,
                  mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
                  mlir::tensor::TensorDialect, mlir::scf::SCFDialect,
                  mlir::arith::ArithDialect, mlir::memref::MemRefDialect>();
  mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
      registry);
  return registry;
}

llk::MappedJitEvidenceSink
acceptanceEvidenceSinkForDirectory(std::filesystem::path outputDirectory) {
  return [outputDirectory](
             const llk::MappedJitEvidence &evidence) -> llvm::Error {
    std::string stem;
    for (unsigned char character : evidence.entrySymbol)
      stem.push_back(std::isalnum(character) || character == '_' ? character
                                                                 : '_');
    stem += "-" + std::to_string(evidence.planId);
    const std::string dialectName = stem + ".llvm-dialect.mlir";
    const std::string irName = stem + ".ll";
    const std::string manifestName = stem + ".manifest.json";

    std::error_code error;
    std::filesystem::create_directories(outputDirectory, error);
    if (error)
      return llvm::createStringError(error, "cannot create evidence directory");

    auto writeFile = [&](const std::string &name,
                         llvm::StringRef contents) -> llvm::Error {
      std::ofstream output(outputDirectory / name,
                           std::ios::out | std::ios::binary | std::ios::trunc);
      if (!output)
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       "cannot write selected evidence file '" +
                                           name + "'");
      output.write(contents.data(),
                   static_cast<std::streamsize>(contents.size()));
      output.flush();
      if (!output)
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(),
            "failed writing selected evidence file '" + name + "'");
      return llvm::Error::success();
    };

    if (llvm::Error writeError = writeFile(dialectName, evidence.llvmDialect))
      return writeError;
    if (llvm::Error writeError = writeFile(irName, evidence.llvmIR))
      return writeError;

    llvm::json::Array features;
    if (evidence.selectedTarget)
      for (const std::string &feature :
           evidence.selectedTarget->requiredFeatures)
        features.emplace_back(feature);
    llvm::json::Object manifest;
    manifest["schema_version"] = 1;
    manifest["entry_symbol"] = evidence.entrySymbol;
    manifest["backend"] =
        evidence.selectedTarget ? "selected-target" : "reference";
    manifest["target"] = evidence.targetName;
    manifest["plan_id"] = std::to_string(evidence.planId);
    manifest["machine_hash"] = std::to_string(evidence.machineHash);
    manifest["execution_identity"] = evidence.executionIdentity;
    manifest["architecture"] = evidence.selectedTarget
                                   ? evidence.selectedTarget->architecture
                                   : std::string();
    manifest["cpu"] =
        evidence.selectedTarget ? evidence.selectedTarget->cpu : std::string();
    manifest["required_features"] = std::move(features);
    manifest["selected_groups_verified"] =
        static_cast<int64_t>(evidence.selectedGroupsVerified);
    manifest["backend_groups_realized"] =
        static_cast<int64_t>(evidence.backendGroupsRealized);
    manifest["reference_groups_lowered"] =
        static_cast<int64_t>(evidence.referenceGroupsLowered);
    manifest["abi_hash"] = std::to_string(evidence.abiHash);
    manifest["llvm_dialect_file"] = dialectName;
    manifest["llvm_ir_file"] = irName;
    std::string manifestText =
        llvm::formatv("{0:2}", llvm::json::Value(std::move(manifest))).str();
    manifestText.push_back('\n');
    return writeFile(manifestName, manifestText);
  };
}

llk::MappedJitEvidenceSink acceptanceEvidenceSink() {
  const char *directory = std::getenv("LLK_ACCEPTANCE_ARTIFACT_DIR");
  if (!directory || directory[0] == '\0')
    return {};
  return acceptanceEvidenceSinkForDirectory(directory);
}

/// Runs the whole chain on `source` and returns the executable it produced:
/// export to concrete Micro, search the shipped AVX2 target, and compile the
/// plan the search selected.
llvm::Expected<std::unique_ptr<llk::MappedExecutable>>
compileChain(mlir::MLIRContext &context, llvm::StringRef sourceText,
             llvm::StringRef entrySymbol,
#ifdef LLK_REQUIRE_SELECTED_TARGET
             llk::MappedBackend backend = llk::MappedBackend::SelectedTarget,
#else
             llk::MappedBackend backend = llk::MappedBackend::Reference,
#endif
             llvm::StringRef scheduleDb = {},
             llk::MappedJitEvidenceSink evidenceSink = {}) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &context);
  if (!module)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the LLK source did not parse");

  // The kernel is exported here, not written by hand.
  mlir::PassManager pm(&context);
  if (scheduleDb.empty())
    pm.addPass(mlir::llk::createLLKToMicroPass());
  else
    pm.addPass(mlir::llk::createLLKToMicroPass(scheduleDb));
  if (mlir::failed(pm.run(*module)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the LLK-to-Micro export failed");

  mlir::Operation *kernel = nullptr;
  module->walk([&](mlir::Operation *op) {
    if (!kernel && op->getName().getStringRef() == "micro.kernel")
      kernel = op;
  });
  if (!kernel)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the export produced no micro.kernel");

  llvm::Expected<std::unique_ptr<MappingTarget>> target =
      mlir::llk::target::avx2::createMappingTarget(LLK_SOURCE_DIR);
  if (!target)
    return target.takeError();

  llvm::Expected<mlir::llk::mapping::WorkloadGraph> graph =
      mlir::llk::mapping::extractWorkloadGraph(kernel);
  if (!graph)
    return graph.takeError();

  mlir::llk::mapping::LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "bf16";
  MappingSearchOptions options;
  options.mode = SearchMode::Exact;
  mlir::llk::mapping::CoveringSearch search(*graph, **target, context,
                                            layoutContext, options);
  llvm::Expected<MappingSearchResult> result = search.search();
  if (!result)
    return result.takeError();
  if (result->plans.empty()) {
    std::string detail =
        "the search found no complete plan (nodesWithoutRules=" +
        std::to_string(result->frontier.nodesWithoutRules) +
        ", candidatesWithoutPlacement=" +
        std::to_string(result->frontier.candidatesWithoutPlacement) +
        ", incompatibleInstancePairs=" +
        std::to_string(result->frontier.incompatibleInstancePairs) + ")";
    for (const auto &diagnostic : result->frontier.diagnostics)
      detail += "; " + diagnostic.message;
    return llvm::createStringError(llvm::inconvertibleErrorCode(), detail);
  }

  llk::MappedCompileOptions compileOptions;
  compileOptions.entrySymbol = entrySymbol.str();
  compileOptions.backend = backend;
  compileOptions.evidenceSink = std::move(evidenceSink);
  if (!compileOptions.evidenceSink &&
      backend == llk::MappedBackend::SelectedTarget)
    compileOptions.evidenceSink = acceptanceEvidenceSink();
  if (!compileOptions.evidenceSink &&
      backend == llk::MappedBackend::SelectedTarget)
    compileOptions.evidenceSink = acceptanceEvidenceSink();
  llvm::Expected<llk::MappedCompilation> compiled = llk::compileMappedKernel(
      *module, **target, result->plans.front(), compileOptions);
  if (!compiled)
    return compiled.takeError();
  if (!compiled->executable)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the compilation produced no executable");
  return std::move(compiled->executable);
}

llvm::Expected<std::unique_ptr<llk::MappedExecutable>>
compileMicroKernel(mlir::MLIRContext &context, llvm::StringRef sourceText,
                   llvm::StringRef entrySymbol, bool requireTransform = false,
                   bool *transformMaterialized = nullptr,
                   int64_t vectorWidth = 8, bool noTransformControl = false) {
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(sourceText, &context);
  if (!module)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the Micro source did not parse");
  mlir::Operation *kernel = nullptr;
  module->walk([&](mlir::Operation *op) {
    if (!kernel && op->getName().getStringRef() == "micro.kernel")
      kernel = op;
  });
  if (!kernel)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the module contains no micro.kernel");
  llvm::Expected<std::unique_ptr<MappingTarget>> target = [&]() {
    if (requireTransform) {
      const std::string root = LLK_SOURCE_DIR;
      return mlir::llk::mapping::loadMappingTarget(
          "x86-avx2", root + "/machines/x86-avx2-v2.yaml",
          root + "/test/Conversion/MicroMapping/Inputs/issue129/"
                 "numeric_transform_layouts.llkmap",
          root + "/test/Conversion/MicroMapping/Inputs/issue129/"
                 "required_transform.llkmap",
          {"avx2_vector_add", "avx2_vector_mul", "avx2_copy"});
    }
    if (vectorWidth == 4) {
      const std::string root = LLK_SOURCE_DIR;
      llvm::Expected<mlir::llk::machine::MachineModel> machine =
          mlir::llk::machine::loadMachineModel(root +
                                               "/machines/x86-avx2-v2.yaml");
      if (!machine)
        return llvm::Expected<std::unique_ptr<MappingTarget>>(
            machine.takeError());
      bool changedWidth = false;
      for (auto &compute : machine->computes) {
        if (compute.kind == "vector_engine") {
          compute.lanes["f32"] = vectorWidth;
          changedWidth = true;
        }
      }
      if (!changedWidth)
        return llvm::Expected<std::unique_ptr<MappingTarget>>(
            llvm::createStringError(llvm::inconvertibleErrorCode(),
                                    "AVX2 machine has no vector engine"));
      llvm::Expected<mlir::llk::mapping::LayoutRegistry> layouts =
          mlir::llk::mapping::loadLayoutFile(
              root + "/mapping/x86-avx2/layouts.llkmap");
      if (!layouts)
        return llvm::Expected<std::unique_ptr<MappingTarget>>(
            layouts.takeError());
      llvm::Expected<mlir::llk::mapping::RuleRegistry> rules =
          mlir::llk::mapping::loadRuleFile(root +
                                           "/mapping/x86-avx2/rules.llkmap");
      if (!rules)
        return llvm::Expected<std::unique_ptr<MappingTarget>>(
            rules.takeError());
      auto custom = std::make_unique<mlir::llk::mapping::FileMappingTarget>(
          "x86-avx2-vw4", std::move(*machine), std::move(*layouts),
          std::move(*rules),
          std::vector<std::string>{
              "avx2_vector_add", "avx2_vector_convert", "avx2_vector_silu",
              "avx2_vector_mul", "avx2_fused_convert_silu_mul", "avx2_mma",
              "avx2_reduce", "avx2_copy", "avx2_tile_copy", "avx2_tile_store"});
      if (llvm::Error error = mlir::llk::mapping::verifyMappingTarget(*custom))
        return llvm::Expected<std::unique_ptr<MappingTarget>>(std::move(error));
      return llvm::Expected<std::unique_ptr<MappingTarget>>(
          std::unique_ptr<MappingTarget>(std::move(custom)));
    }
    return mlir::llk::target::avx2::createMappingTarget(LLK_SOURCE_DIR);
  }();
  if (!target)
    return target.takeError();
  llvm::Expected<mlir::llk::mapping::WorkloadGraph> graph =
      mlir::llk::mapping::extractWorkloadGraph(kernel);
  if (!graph)
    return graph.takeError();
  mlir::llk::mapping::LayoutContext layoutContext;
  layoutContext.rank = 2;
  layoutContext.elementType = "f32";
  MappingSearchOptions searchOptions;
  searchOptions.mode = SearchMode::Exact;
  mlir::llk::mapping::CoveringSearch search(*graph, **target, context,
                                            layoutContext, searchOptions);
  llvm::Expected<MappingSearchResult> result = search.search();
  if (!result)
    return result.takeError();
  if (result->plans.empty()) {
    std::string detail = "the Micro graph has no complete mapping";
    for (const auto &diagnostic : result->frontier.diagnostics)
      detail += "; " + diagnostic.message;
    return llvm::createStringError(llvm::inconvertibleErrorCode(), detail);
  }
  if (noTransformControl) {
    bool foundTransform = false;
    for (auto &connection : result->plans.front().connectionPlans) {
      if (!connection.transform)
        continue;
      connection.transform.reset();
      connection.kind = mlir::llk::mapping::ConnectionKind::Direct;
      foundTransform = true;
    }
    if (!foundTransform)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "the selected plan has no transform");
    result->plans.front().id =
        mlir::llk::mapping::computePlanId(result->plans.front());
  }
  llk::MappedCompileOptions options;
  options.entrySymbol = entrySymbol.str();
#ifdef LLK_REQUIRE_SELECTED_TARGET
  options.backend = (requireTransform || vectorWidth == 4)
                        ? llk::MappedBackend::Reference
                        : llk::MappedBackend::SelectedTarget;
#else
  options.backend = llk::MappedBackend::Reference;
#endif
  if (options.backend == llk::MappedBackend::SelectedTarget)
    options.evidenceSink = acceptanceEvidenceSink();
  if (requireTransform) {
    llk::MappedCompileOptions boundOptions = options;
    boundOptions.stop = llk::MappedStop::MappedMicro;
    llvm::Expected<llk::MappedCompilation> bound = llk::compileMappedKernel(
        *module, **target, result->plans.front(), boundOptions);
    if (!bound)
      return bound.takeError();
    bool foundTransform = false;
    bound->module->walk([&](mlir::Operation *op) {
      foundTransform |= op->getName().getStringRef() == "micro.transform";
    });
    if (transformMaterialized)
      *transformMaterialized = foundTransform;
  }
  llvm::Expected<llk::MappedCompilation> compiled = llk::compileMappedKernel(
      *module, **target, result->plans.front(), options);
  if (!compiled)
    return compiled.takeError();
  if (!compiled->executable)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "the mapped Micro kernel did not compile");
  return std::move(compiled->executable);
}

constexpr llvm::StringLiteral kTwoOutputVectorKernel = R"mlir(
module {
  micro.kernel @two_outputs(%a: tensor<8x8xf32>, %b: tensor<8x8xf32>)
      -> (tensor<8x8xf32>, tensor<8x8xf32>) {
    %ta = micro.tile_view %a {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %tb = micro.tile_view %b {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %sum = micro.vector "add" %ta, %tb : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<acc>>
    %product = micro.vector "mul" %ta, %tb : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<acc>>
    micro.yield %sum, %product : !micro.tile<8x8xf32, memory = #micro.memory<acc>>, !micro.tile<8x8xf32, memory = #micro.memory<acc>>
  }
}
)mlir";

constexpr llvm::StringLiteral kRequiredTransformNumericKernel = R"mlir(
module {
  micro.kernel @required_transform(%x: tensor<8x8xf32>) -> tensor<8x8xf32> {
    %t, %tok = micro.async_copy %x {src_memory = #micro.memory<dram>, dst_memory = #micro.memory<sram>} : tensor<8x8xf32> -> tensor<8x8xf32>, !micro.async_token
    %a = micro.tile_view %t {shape = array<i64: 8, 8>} : tensor<8x8xf32> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %r = micro.vector "add" %a, %a : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<sram>>
    %product = micro.vector "mul" %r, %a : !micro.tile<8x8xf32, memory = #micro.memory<sram>>, !micro.tile<8x8xf32, memory = #micro.memory<sram>> -> !micro.tile<8x8xf32, memory = #micro.memory<dram>>
    micro.yield %product : !micro.tile<8x8xf32, memory = #micro.memory<dram>>
  }
}
)mlir";

/// One bf16 buffer of `rows` x `columns`.
struct Buffer {
  std::vector<uint16_t> storage;
  MemRef2D descriptor;

  Buffer(int64_t rows, int64_t columns, float fill)
      : storage(static_cast<size_t>(rows * columns), toBf16(fill)) {
    descriptor =
        MemRef2D{storage.data(), storage.data(), 0, rows, columns, columns, 1};
  }

  llk::InvocationBuffer2D checked() const {
    return {descriptor, llk::InvocationElementType::BF16,
            storage.size() * sizeof(uint16_t)};
  }
};

std::string matmulSource(int64_t M, int64_t N, int64_t K) {
  const std::string lhs =
      "tensor<" + std::to_string(M) + "x" + std::to_string(K) + "xbf16>";
  const std::string rhs =
      "tensor<" + std::to_string(K) + "x" + std::to_string(N) + "xbf16>";
  const std::string output =
      "tensor<" + std::to_string(M) + "x" + std::to_string(N) + "xf32>";
  return "module {\n  func.func @matmul(%a: " + lhs + ", %b: " + rhs +
         ", %init: " + output + ") -> " + output +
         " {\n"
         "    %y = llk.matmul ins(%a, %b : " +
         lhs + ", " + rhs + ") outs(%init : " + output +
         ") {accumulator_type = f32, math_mode = #llk.math_mode<bounded_fast>} "
         "-> " +
         output +
         "\n"
         "    return %y : " +
         output + "\n  }\n}\n";
}

void fillSignedBf16(Buffer &buffer, uint32_t &state) {
  for (uint16_t &value : buffer.storage)
    value = toBf16(nextSignedValue(state));
}

float storedBf16(const Buffer &buffer, size_t index) {
  return fromBf16(buffer.storage[index]);
}

struct FloatBuffer {
  std::vector<float> storage;
  MemRef2D descriptor;

  FloatBuffer(int64_t rows, int64_t columns, float fill)
      : storage(static_cast<size_t>(rows * columns), fill) {
    descriptor =
        MemRef2D{storage.data(), storage.data(), 0, rows, columns, columns, 1};
  }

  llk::InvocationBuffer2D checked() const {
    return {descriptor, llk::InvocationElementType::F32,
            storage.size() * sizeof(float)};
  }
};

void fillSigned(FloatBuffer &buffer, uint32_t &state) {
  for (float &value : buffer.storage)
    value = nextSignedValue(state);
}

void expectPaddedMatmul(mlir::MLIRContext &context, llk::MappedBackend backend,
                        int64_t M, int64_t N, int64_t K) {
  const std::string source = matmulSource(M, N, K);
  const std::string schedule =
      std::string(LLK_SOURCE_DIR) +
      "/test/Conversion/MicroMapping/tail_acceptance_schedule.json";
  const std::string entry = "matmul_M" + std::to_string(M) + "_N" +
                            std::to_string(N) + "_K" + std::to_string(K);
  llvm::Expected<std::unique_ptr<llk::MappedExecutable>> executable =
      compileChain(context, source, entry, backend, schedule);
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());
  const llk::KernelAbi &abi = (*executable)->abi();
  ASSERT_EQ(abi.inputs.size(), 3u);
  ASSERT_EQ(abi.outputs.size(), 1u);
  EXPECT_EQ(abi.inputs[0].shape, (std::vector<int64_t>{M, K}));
  EXPECT_EQ(abi.inputs[1].shape, (std::vector<int64_t>{K, N}));
  EXPECT_EQ(abi.inputs[2].shape, (std::vector<int64_t>{M, N}));
  EXPECT_EQ(abi.outputs[0].shape, (std::vector<int64_t>{M, N}));

  Buffer a(M, K, 0.0f);
  Buffer b(K, N, 0.0f);
  uint32_t state = 0x12967u;
  fillSignedBf16(a, state);
  fillSignedBf16(b, state);
  FloatBuffer init(M, N, 0.0f);
  fillSigned(init, state);
  const auto originalA = a.storage;
  const auto originalB = b.storage;
  const auto originalInit = init.storage;

  std::vector<float> guarded(static_cast<size_t>(M * N) + 2, -77.0f);
  MemRef2D descriptor{guarded.data(), guarded.data() + 1, 0, M, N, N, 1};
  llk::InvocationBuffer2D checkedOutput{descriptor,
                                        llk::InvocationElementType::F32,
                                        guarded.size() * sizeof(float)};
  llvm::Error error =
      (*executable)
          ->invoke({a.checked(), b.checked(), init.checked()}, {checkedOutput});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));

  for (int64_t m = 0; m < M; ++m) {
    for (int64_t n = 0; n < N; ++n) {
      size_t index = static_cast<size_t>(m * N + n);
      float expected = init.storage[index];
      for (int64_t k = 0; k < K; ++k) {
        size_t aIndex = static_cast<size_t>(m * K + k);
        size_t bIndex = static_cast<size_t>(k * N + n);
        expected += storedBf16(a, aIndex) * storedBf16(b, bIndex);
      }
      EXPECT_NEAR(guarded[index + 1], expected,
                  2e-4f + 2e-4f * std::abs(expected))
          << "shape " << M << "x" << N << "x" << K << " at " << m << "," << n;
    }
  }
  EXPECT_EQ(guarded.front(), -77.0f);
  EXPECT_EQ(guarded.back(), -77.0f);
  EXPECT_EQ(a.storage, originalA);
  EXPECT_EQ(b.storage, originalB);
  EXPECT_EQ(init.storage, originalInit);
}

} // namespace

TEST(MappedAcceptance, InvokesTheCompilerGeneratedMatmulAndChecksEveryElement) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::Expected<std::unique_ptr<llk::MappedExecutable>> executable =
      compileChain(context, kMatmulSource, "matmul_M16_N64_K64");
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());
  EXPECT_NE((*executable)->executionIdentity().find("backend=reference"),
            llvm::StringRef::npos);
  EXPECT_NE((*executable)->executionIdentity().find("abi="),
            llvm::StringRef::npos);
  EXPECT_NE((*executable)->executionIdentity().find("lowered-ir-sha256="),
            llvm::StringRef::npos);

  // The exported contract: both operands, the caller's initialized output,
  // and the output tile.
  const llk::KernelAbi &abi = (*executable)->abi();
  ASSERT_EQ(abi.inputs.size(), 3u);
  ASSERT_EQ(abi.outputs.size(), 1u);
  EXPECT_EQ(abi.inputs[0].shape, (std::vector<int64_t>{16, 64}));
  EXPECT_EQ(abi.inputs[1].shape, (std::vector<int64_t>{64, 64}));
  EXPECT_EQ(abi.inputs[2].shape, (std::vector<int64_t>{16, 64}));
  EXPECT_EQ(abi.outputs[0].shape, (std::vector<int64_t>{16, 64}));

  Buffer a(16, 64, 1.0f);
  Buffer b(64, 64, 1.0f);
  Buffer init(16, 64, 7.0f);
  Buffer out(16, 64, -1.0f);

  llvm::Error error =
      (*executable)
          ->invoke({a.checked(), b.checked(), init.checked()}, {out.checked()});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));

  // Every output is a full K=64 reduction of ones, so every one of the 1024
  // elements is exactly 71, including the nonzero output initializer, --
  // which is only true if the staged copies, the
  // contraction, the accumulator that the K loop threads, the narrowing
  // epilogue and the write-back all happened, in that order.
  for (size_t i = 0; i < out.storage.size(); ++i)
    EXPECT_EQ(fromBf16(out.storage[i]), 71.0f) << "element " << i;
}

TEST(MappedAcceptance, JitEvidenceSinkReceivesLlvmAndSelectedPlanMetadata) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  bool received = false;
  llk::MappedJitEvidence captured;
  auto sink = [&](const llk::MappedJitEvidence &evidence) -> llvm::Error {
    captured = evidence;
    received = true;
    return llvm::Error::success();
  };
  llvm::Expected<std::unique_ptr<llk::MappedExecutable>> executable =
      compileChain(context, kMatmulSource, "matmul_M16_N64_K64",
                   llk::MappedBackend::Reference, {}, sink);
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());
  ASSERT_TRUE(received);
  EXPECT_EQ(captured.entrySymbol, "matmul_M16_N64_K64");
  EXPECT_NE(captured.executionIdentity.find("backend=reference"),
            std::string::npos);
  EXPECT_NE(captured.executionIdentity.find("|abi="), std::string::npos);
  EXPECT_NE(captured.executionIdentity.find("|lowered-ir-sha256="),
            std::string::npos);
  EXPECT_NE(captured.llvmDialect.find("llvm.func"), std::string::npos);
  EXPECT_NE(captured.llvmIR.find("define "), std::string::npos);
  EXPECT_NE(captured.planId, 0u);
  EXPECT_NE(captured.machineHash, 0u);
  EXPECT_NE(captured.abiHash, 0u);
  EXPECT_GT(captured.selectedGroupsVerified, 0u);
}

TEST(MappedAcceptance, EvidenceSinkWritesLlvmAndManifestFiles) {
  struct TemporaryDirectory {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("llk-mapped-evidence-" +
         std::to_string(
             std::chrono::steady_clock::now().time_since_epoch().count()));
    ~TemporaryDirectory() {
      std::error_code ignored;
      std::filesystem::remove_all(path, ignored);
    }
  } temporary;

  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  llvm::Expected<std::unique_ptr<llk::MappedExecutable>> executable =
      compileChain(context, kMatmulSource, "matmul_M16_N64_K64",
                   llk::MappedBackend::Reference, {},
                   acceptanceEvidenceSinkForDirectory(temporary.path));
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());

  std::filesystem::path manifestPath;
  std::filesystem::path dialectPath;
  std::filesystem::path llvmIRPath;
  for (const auto &entry :
       std::filesystem::directory_iterator(temporary.path)) {
    const std::string name = entry.path().filename().string();
    if (name.ends_with(".manifest.json"))
      manifestPath = entry.path();
    else if (name.ends_with(".llvm-dialect.mlir"))
      dialectPath = entry.path();
    else if (name.ends_with(".ll"))
      llvmIRPath = entry.path();
  }
  ASSERT_FALSE(manifestPath.empty());
  ASSERT_FALSE(dialectPath.empty());
  ASSERT_FALSE(llvmIRPath.empty());
  std::ifstream manifestFile(manifestPath);
  std::stringstream manifestText;
  manifestText << manifestFile.rdbuf();
  EXPECT_NE(manifestText.str().find("\"execution_identity\""),
            std::string::npos);
  EXPECT_NE(manifestText.str().find("\"abi_hash\""), std::string::npos);
  EXPECT_NE(manifestText.str().find("\"plan_id\""), std::string::npos);
  std::ifstream dialectFile(dialectPath);
  std::stringstream dialectText;
  dialectText << dialectFile.rdbuf();
  EXPECT_NE(dialectText.str().find("llvm.func"), std::string::npos);
  std::ifstream llvmIRFile(llvmIRPath);
  std::stringstream llvmIRText;
  llvmIRText << llvmIRFile.rdbuf();
  EXPECT_NE(llvmIRText.str().find("define "), std::string::npos);
}

TEST(MappedAcceptance, EvidenceSinkFailureStopsJitCompilation) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto sink = [](const llk::MappedJitEvidence &) -> llvm::Error {
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "evidence sink failed");
  };
  llvm::Expected<std::unique_ptr<llk::MappedExecutable>> executable =
      compileChain(context, kMatmulSource, "matmul_M16_N64_K64",
                   llk::MappedBackend::Reference, {}, sink);
  ASSERT_FALSE(static_cast<bool>(executable));
  EXPECT_NE(llvm::toString(executable.takeError()).find("evidence sink failed"),
            std::string::npos);
}

TEST(MappedAcceptance, InvokesTheCompilerGeneratedSwiGLUAndChecksEveryElement) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::Expected<std::unique_ptr<llk::MappedExecutable>> executable =
      compileChain(context, kSwigluSource, "fused_swiglu_M16_N64_K64");
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());

  const llk::KernelAbi &abi = (*executable)->abi();
  ASSERT_EQ(abi.inputs.size(), 3u);
  ASSERT_EQ(abi.outputs.size(), 1u);
  EXPECT_EQ(abi.outputs[0].shape, (std::vector<int64_t>{16, 64}));

  Buffer x(16, 64, 1.0f);
  Buffer wg(64, 64, 1.0f);
  Buffer wu(64, 64, 1.0f);
  Buffer out(16, 64, -1.0f);

  llvm::Error error =
      (*executable)
          ->invoke({x.checked(), wg.checked(), wu.checked()}, {out.checked()});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));

  // The gate and the up projection each reduce K=64 ones to 64, `silu(64)` is
  // 64 to the precision of the activation, and the gating multiply gives 4096.
  // That is a number only a fused kernel that ran both projections *and* the
  // activation can produce.
  for (size_t i = 0; i < out.storage.size(); ++i)
    EXPECT_EQ(fromBf16(out.storage[i]), 4096.0f) << "element " << i;

  uint32_t state = 0x12967u;
  fillSignedBf16(x, state);
  fillSignedBf16(wg, state);
  fillSignedBf16(wu, state);
  const auto originalX = x.storage;
  const auto originalWg = wg.storage;
  const auto originalWu = wu.storage;
  std::fill(out.storage.begin(), out.storage.end(), toBf16(-7.0f));
  error =
      (*executable)
          ->invoke({x.checked(), wg.checked(), wu.checked()}, {out.checked()});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));

  // bounded_fast replaces exp with a cubic on its reduced interval. Inputs are
  // in [-0.5, 0.5], so each K=64 projection is bounded by 16 in magnitude;
  // this absolute-plus-relative tolerance includes BF16 output rounding and
  // the approximation error over that declared range.
  for (int64_t m = 0; m < 16; ++m) {
    for (int64_t n = 0; n < 64; ++n) {
      float gate = 0.0f;
      float up = 0.0f;
      for (int64_t k = 0; k < 64; ++k) {
        const float value = storedBf16(x, static_cast<size_t>(m * 64 + k));
        gate += value * storedBf16(wg, static_cast<size_t>(k * 64 + n));
        up += value * storedBf16(wu, static_cast<size_t>(k * 64 + n));
      }
      const float expected = gate / (1.0f + std::exp(-gate)) * up;
      const float actual =
          fromBf16(out.storage[static_cast<size_t>(m * 64 + n)]);
      EXPECT_NEAR(actual, expected, 0.04f + 0.02f * std::abs(expected))
          << "element " << m << "," << n << " gate=" << gate << " up=" << up;
    }
  }
  EXPECT_EQ(x.storage, originalX);
  EXPECT_EQ(wg.storage, originalWg);
  EXPECT_EQ(wu.storage, originalWu);
}

TEST(MappedAcceptance, MapsAndInvokesTwoIndependentVectorOutputs) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  llvm::Expected<std::unique_ptr<llk::MappedExecutable>> executable =
      compileMicroKernel(context, kTwoOutputVectorKernel, "two_outputs");
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());
  ASSERT_EQ((*executable)->abi().inputs.size(), 2u);
  ASSERT_EQ((*executable)->abi().outputs.size(), 2u);

  FloatBuffer a(8, 8, 0.0f);
  FloatBuffer b(8, 8, 0.0f);
  uint32_t state = 0x12967u;
  fillSigned(a, state);
  fillSigned(b, state);
  const auto originalA = a.storage;
  const auto originalB = b.storage;
  std::vector<float> first(66, -77.0f);
  std::vector<float> second(66, -79.0f);
  llk::InvocationBuffer2D out0{{first.data(), first.data() + 1, 0, 8, 8, 8, 1},
                               llk::InvocationElementType::F32,
                               first.size() * sizeof(float)};
  llk::InvocationBuffer2D out1{
      {second.data(), second.data() + 1, 0, 8, 8, 8, 1},
      llk::InvocationElementType::F32,
      second.size() * sizeof(float)};
  llvm::Error error =
      (*executable)->invoke({a.checked(), b.checked()}, {out0, out1});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));
  for (size_t i = 0; i < a.storage.size(); ++i) {
    EXPECT_FLOAT_EQ(first[i + 1], a.storage[i] + b.storage[i]);
    EXPECT_FLOAT_EQ(second[i + 1], a.storage[i] * b.storage[i]);
  }
  EXPECT_EQ(first.front(), -77.0f);
  EXPECT_EQ(first.back(), -77.0f);
  EXPECT_EQ(second.front(), -79.0f);
  EXPECT_EQ(second.back(), -79.0f);
  EXPECT_EQ(a.storage, originalA);
  EXPECT_EQ(b.storage, originalB);

  state = 0x12968u;
  fillSigned(a, state);
  fillSigned(b, state);
  error = (*executable)->invoke({a.checked(), b.checked()}, {out0, out1});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));
  for (size_t i = 0; i < a.storage.size(); ++i) {
    EXPECT_FLOAT_EQ(first[i + 1], a.storage[i] + b.storage[i]);
    EXPECT_FLOAT_EQ(second[i + 1], a.storage[i] * b.storage[i]);
  }
}

TEST(MappedAcceptance, ExecutesWidthFourVectorAdditionNumerically) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  llvm::Expected<std::unique_ptr<llk::MappedExecutable>> executable =
      compileMicroKernel(context, kTwoOutputVectorKernel, "two_outputs",
                         /*requireTransform=*/false,
                         /*transformMaterialized=*/nullptr, /*vectorWidth=*/4);
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());

  FloatBuffer a(8, 8, 0.0f);
  FloatBuffer b(8, 8, 0.0f);
  uint32_t state = 0x12967u;
  fillSigned(a, state);
  fillSigned(b, state);
  const auto originalA = a.storage;
  const auto originalB = b.storage;
  std::vector<float> output(66, -91.0f);
  std::vector<float> product(66, -93.0f);
  llk::InvocationBuffer2D out{{output.data(), output.data() + 1, 0, 8, 8, 8, 1},
                              llk::InvocationElementType::F32,
                              output.size() * sizeof(float)};
  llk::InvocationBuffer2D productOut{
      {product.data(), product.data() + 1, 0, 8, 8, 8, 1},
      llk::InvocationElementType::F32,
      product.size() * sizeof(float)};
  llvm::Error error =
      (*executable)->invoke({a.checked(), b.checked()}, {out, productOut});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));
  for (size_t i = 0; i < a.storage.size(); ++i)
    EXPECT_FLOAT_EQ(output[i + 1], a.storage[i] + b.storage[i])
        << "element " << i;
  for (size_t i = 0; i < a.storage.size(); ++i)
    EXPECT_FLOAT_EQ(product[i + 1], a.storage[i] * b.storage[i])
        << "product element " << i;
  EXPECT_EQ(output.front(), -91.0f);
  EXPECT_EQ(output.back(), -91.0f);
  EXPECT_EQ(product.front(), -93.0f);
  EXPECT_EQ(product.back(), -93.0f);
  EXPECT_EQ(a.storage, originalA);
  EXPECT_EQ(b.storage, originalB);
}

TEST(MappedAcceptance, ExecutesTheRequiredLayoutTransformNumerically) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  bool transformMaterialized = false;
  llvm::Expected<std::unique_ptr<llk::MappedExecutable>> executable =
      compileMicroKernel(context, kRequiredTransformNumericKernel,
                         "required_transform", /*requireTransform=*/true,
                         &transformMaterialized);
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());
  ASSERT_TRUE(transformMaterialized)
      << "incompatible selected maps did not materialize micro.transform";
  ASSERT_EQ((*executable)->abi().inputs.size(), 1u);
  ASSERT_EQ((*executable)->abi().outputs.size(), 1u);

  FloatBuffer input(8, 8, 0.0f);
  uint32_t state = 0x12967u;
  fillSigned(input, state);
  const auto original = input.storage;
  FloatBuffer output(8, 8, -7.0f);
  llvm::Error error =
      (*executable)->invoke({input.checked()}, {output.checked()});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));
  for (int64_t row = 0; row < 8; ++row) {
    for (int64_t column = 0; column < 8; ++column) {
      const size_t index = static_cast<size_t>(row * 8 + column);
      const size_t transposed = static_cast<size_t>(column * 8 + row);
      const float expected =
          2.0f * input.storage[transposed] * input.storage[index];
      EXPECT_NEAR(output.storage[index], expected, 1e-6f)
          << "element " << row << "," << column;
    }
  }
  EXPECT_EQ(input.storage, original);
}

TEST(MappedAcceptance, NoTransformControlChangesTheNumericResult) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  bool transformMaterialized = true;
  llvm::Expected<std::unique_ptr<llk::MappedExecutable>> executable =
      compileMicroKernel(context, kRequiredTransformNumericKernel,
                         "required_transform", /*requireTransform=*/true,
                         &transformMaterialized, /*vectorWidth=*/8,
                         /*noTransformControl=*/true);
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());
  ASSERT_FALSE(transformMaterialized)
      << "the no-transform plan unexpectedly materialized micro.transform";

  FloatBuffer input(8, 8, 0.0f);
  uint32_t state = 0x12967u;
  fillSigned(input, state);
  const auto original = input.storage;
  FloatBuffer output(8, 8, -7.0f);
  llvm::Error error =
      (*executable)->invoke({input.checked()}, {output.checked()});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));

  size_t mismatches = 0;
  for (int64_t row = 0; row < 8; ++row) {
    for (int64_t column = 0; column < 8; ++column) {
      const size_t index = static_cast<size_t>(row * 8 + column);
      const size_t transposed = static_cast<size_t>(column * 8 + row);
      const float transformedExpected =
          2.0f * input.storage[transposed] * input.storage[index];
      const float noTransformExpected =
          2.0f * input.storage[index] * input.storage[index];
      EXPECT_NEAR(output.storage[index], noTransformExpected, 1e-6f)
          << "element " << row << "," << column;
      mismatches +=
          std::abs(output.storage[index] - transformedExpected) > 1e-6f;
    }
  }
  EXPECT_GT(mismatches, 0u)
      << "omitting the selected layout transform preserved its result";
  EXPECT_EQ(input.storage, original);
}

TEST(MappedAcceptance, LeavesTheCallersBuffersOwnedByTheCaller) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::Expected<std::unique_ptr<llk::MappedExecutable>> executable =
      compileChain(context, kMatmulSource, "matmul_M16_N64_K64");
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());

  // The output buffer is handed over inside a larger allocation with sentinels
  // on both sides: the kernel writes the region it was given and nothing else,
  // and it does not adopt the pointer.
  const int64_t rows = 16, columns = 64;
  std::vector<uint16_t> storage(static_cast<size_t>(rows * columns) + 32,
                                toBf16(-7.0f));
  const size_t lead = 16;
  MemRef2D out{
      storage.data(), storage.data() + lead, 0, rows, columns, columns, 1};

  Buffer a(rows, columns, 1.0f);
  Buffer b(64, 64, 1.0f);
  Buffer init(rows, columns, 0.0f);

  llk::InvocationBuffer2D checkedOut{out, llk::InvocationElementType::BF16,
                                     storage.size() * sizeof(uint16_t)};
  llvm::Error error =
      (*executable)
          ->invoke({a.checked(), b.checked(), init.checked()}, {checkedOut});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));

  for (int64_t i = 0; i < rows * columns; ++i)
    EXPECT_EQ(fromBf16(storage[lead + static_cast<size_t>(i)]), 64.0f)
        << "element " << i;
  EXPECT_EQ(fromBf16(storage[lead - 1]), -7.0f);
  EXPECT_EQ(fromBf16(storage[lead + static_cast<size_t>(rows * columns)]),
            -7.0f);
}

TEST(MappedAcceptance, ExecutesPaddedTailsWithNonzeroAccumulators) {
  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  for (const std::array<int64_t, 3> shape : {std::array<int64_t, 3>{16, 64, 64},
                                             {5, 9, 7},
                                             {1, 17, 3},
                                             {17, 65, 63}})
    expectPaddedMatmul(context, llk::MappedBackend::Reference, shape[0],
                       shape[1], shape[2]);
}

TEST(MappedAcceptance, SelectedAvx2ExecutesThePaddedTailFixtures) {
  if (!selectedAvx2Available()) {
#ifdef LLK_REQUIRE_SELECTED_TARGET
    if (avx2DisabledByTestOverride())
      FAIL() << "test override disabled required selected AVX2 execution";
    else
      FAIL() << "required selected AVX2 execution needs an x86_64 AVX2 host";
#else
    GTEST_SKIP() << "selected AVX2 invocation requires an x86_64 host";
#endif
  }

  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  for (const std::array<int64_t, 3> shape : {std::array<int64_t, 3>{16, 64, 64},
                                             {5, 9, 7},
                                             {1, 17, 3},
                                             {17, 65, 63}})
    expectPaddedMatmul(context, llk::MappedBackend::SelectedTarget, shape[0],
                       shape[1], shape[2]);
}

TEST(MappedAcceptance, SelectedAvx2BackendExecutesNumerically) {
  if (!selectedAvx2Available()) {
#ifdef LLK_REQUIRE_SELECTED_TARGET
    if (avx2DisabledByTestOverride())
      FAIL() << "test override disabled required selected AVX2 execution";
    else
      FAIL() << "required selected AVX2 execution needs an x86_64 AVX2 host";
#else
    GTEST_SKIP() << "selected AVX2 invocation requires an x86_64 host";
#endif
  }

  mlir::DialectRegistry registry = buildRegistry();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  llvm::Expected<std::unique_ptr<llk::MappedExecutable>> executable =
      compileChain(context, kMatmulSource, "matmul_M16_N64_K64",
                   llk::MappedBackend::SelectedTarget);
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());
  EXPECT_NE((*executable)->executionIdentity().find("backend=selected-target"),
            llvm::StringRef::npos);
  EXPECT_NE((*executable)->executionIdentity().find("features=avx2"),
            llvm::StringRef::npos);

  const int64_t rows = 16, columns = 64;
  std::vector<uint16_t> output(static_cast<size_t>(rows * columns), 0);
  MemRef2D out{output.data(), output.data(), 0, rows, columns, columns, 1};
  Buffer a(rows, columns, 1.0f);
  Buffer b(64, 64, 1.0f);
  Buffer init(rows, columns, 0.0f);
  llk::InvocationBuffer2D checkedOut{out, llk::InvocationElementType::BF16,
                                     output.size() * sizeof(uint16_t)};
  llvm::Error error =
      (*executable)
          ->invoke({a.checked(), b.checked(), init.checked()}, {checkedOut});
  ASSERT_FALSE(static_cast<bool>(error)) << llvm::toString(std::move(error));
  for (uint16_t value : output)
    EXPECT_EQ(fromBf16(value), 64.0f);
}

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
    std::fprintf(stderr,
                 "MappedAVX2Acceptance requires --require-selected-target\n");
    return 2;
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
