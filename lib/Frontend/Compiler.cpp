//===- Compiler.cpp - Shared mapped compilation for frontend requests ----===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "LLK/Frontend/Compiler.h"
#include "LLK/Conversion/MicroMapping/MappedTuningSession.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Frontend/Importer.h"
#include "LLK/Target/MappingTargets.h"
#include "LLK/Version.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/TargetParser/Host.h"
#include <filesystem>
#include <limits>
#include <mutex>
#include <new>
#include <set>

namespace llk::frontend {
namespace {
namespace native = mlir::llk;
std::string print(mlir::Operation *operation) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  operation->print(stream);
  return text;
}
std::string hex(uint64_t value) { return "0x" + llvm::utohexstr(value, true); }
llvm::Error configurationError(llvm::Twine message) {
  return frontendError("invalid_frontend_configuration", message, {},
                       "compile");
}
/// Existing target factories load files. Materialize only the immutable request
/// snapshot in a private directory, then destroy it after loading; no live
/// config_root contents can alter the target that the request identifies.
class ConfigurationSnapshot {
public:
  llvm::SmallString<128> root;
  ~ConfigurationSnapshot() {
    if (!root.empty())
      (void)llvm::sys::fs::remove_directories(root);
  }
  static llvm::Expected<std::unique_ptr<ConfigurationSnapshot>>
  create(const llvm::json::Object &files) {
    if (files.empty())
      return configurationError("configuration snapshot is empty");
    auto snapshot = std::make_unique<ConfigurationSnapshot>();
    if (auto ec = llvm::sys::fs::createUniqueDirectory("llk-frontend",
                                                       snapshot->root))
      return configurationError(ec.message());
    std::set<std::string> paths;
    for (const auto &file : files) {
      auto path = std::filesystem::path(file.first.str()).lexically_normal();
      auto relative = path.generic_string();
      if (relative.empty() || relative == "." || path.is_absolute() ||
          !paths.insert(relative).second)
        return configurationError("ambiguous configuration snapshot path");
      for (const auto &part : path)
        if (part == "..")
          return configurationError("configuration path escapes snapshot");
      llvm::SmallString<256> destination(snapshot->root);
      llvm::sys::path::append(destination, relative);
      auto parent = llvm::sys::path::parent_path(destination);
      if (auto ec = llvm::sys::fs::create_directories(parent))
        return configurationError(ec.message());
      std::error_code ec;
      llvm::raw_fd_ostream output(destination, ec);
      if (ec)
        return configurationError(ec.message());
      auto content = file.second.getAsString();
      if (!content)
        return configurationError("snapshot contents must be text");
      output << *content;
      output.close();
      if (output.has_error()) {
        auto failure = output.error();
        output.clear_error();
        return configurationError(failure.message());
      }
    }
    return snapshot;
  }
};
} // namespace
struct FrontendCompilation::State {
  std::unique_ptr<MappedExecutable> executable;
  std::string reportText, frontendIR, llkIR, microIR;
  std::unique_ptr<float[]> destination;
  uint64_t destinationBytes = 0;
  std::vector<unsigned> inputMap;
  unsigned destinationPort = 2;
  std::mutex invocation;
};
FrontendCompilation::FrontendCompilation(std::unique_ptr<State> state)
    : state(std::move(state)) {}
FrontendCompilation::~FrontendCompilation() = default;
FrontendCompilation::FrontendCompilation(FrontendCompilation &&) noexcept =
    default;
FrontendCompilation &
FrontendCompilation::operator=(FrontendCompilation &&) noexcept = default;
llvm::StringRef FrontendCompilation::report() const {
  return state->reportText;
}
const KernelAbi &FrontendCompilation::nativeABI() const {
  return state->executable->abi();
}
llvm::Expected<llvm::StringRef>
FrontendCompilation::inspect(llvm::StringRef stage) const {
  if (stage == "frontend")
    return llvm::StringRef(state->frontendIR);
  if (stage == "llk")
    return llvm::StringRef(state->llkIR);
  if (stage == "micro")
    return llvm::StringRef(state->microIR);
  return frontendError("inspection_unavailable",
                       "inspection stage is unavailable: " + stage, {},
                       "inspect");
}
llvm::Error
FrontendCompilation::invoke(llvm::ArrayRef<InvocationBuffer2D> inputs,
                            llvm::ArrayRef<InvocationBuffer2D> outputs) {
  std::lock_guard<std::mutex> lock(state->invocation);
  if (inputs.size() != state->inputMap.size() || outputs.size() != 1)
    return frontendError("invalid_frontend_buffer",
                         "public input/output arity mismatch", {}, "invoke");
  std::vector<InvocationBuffer2D> nativeInputs(
      state->executable->abi().inputs.size());
  for (size_t i = 0; i != inputs.size(); ++i)
    nativeInputs[state->inputMap[i]] = inputs[i];
  const auto &shape =
      state->executable->abi().inputs[state->destinationPort].shape;
  nativeInputs[state->destinationPort] = {{state->destination.get(),
                                           state->destination.get(), 0,
                                           shape[0], shape[1], shape[1], 1},
                                          InvocationElementType::F32,
                                          state->destinationBytes};
  // MappedExecutable validates every native descriptor and disjointness before
  // entering code. The private no-read destination follows the same ABI rules.
  if (auto error = state->executable->invoke(nativeInputs, outputs))
    return frontendError("invalid_frontend_buffer",
                         llvm::toString(std::move(error)), {}, "invoke");
  return llvm::Error::success();
}
llvm::Expected<FrontendCompilation>
compileFrontend(const FrontendRequest &request) {
  const auto &options = request.options();
  if (options.topK != 1)
    return frontendError("unsupported_frontend_options",
                         "initial frontend service requires top_k=1", {},
                         "compile");
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::micro::MicroDialect, native::LLKDialect>();
  mlir::MLIRContext context(registry);
  auto imported = importProgram(request, context);
  if (!imported)
    return imported.takeError();
  auto snapshot = ConfigurationSnapshot::create(options.configFiles);
  if (!snapshot)
    return snapshot.takeError();
  static std::once_flag targets;
  std::call_once(targets, [] { native::target::registerAllMappingTargets(); });
  auto target = native::mapping::createRegisteredMappingTarget(
      options.target, (*snapshot)->root);
  if (!target)
    return configurationError(llvm::toString(target.takeError()));
  const auto backend = options.backend == "reference"
                           ? MappedBackend::Reference
                           : MappedBackend::SelectedTarget;
  if (backend == MappedBackend::SelectedTarget) {
    auto requirements = (*target)->codegenRequirements();
    if (!requirements)
      return frontendError("native_backend_unavailable",
                           "target has no selected backend", {}, "compile");
    auto triple = llvm::sys::getDefaultTargetTriple();
    if (auto error = native::mapping::checkHostRequirements(
            *requirements, llvm::StringRef(triple).split('-').first,
            llvm::sys::getHostCPUFeatures()))
      return frontendError("native_backend_unavailable",
                           llvm::toString(std::move(error)), {}, "compile");
  }
  native::tuning::MappedTuningOptions tuning;
  tuning.source.sourceSymbol = imported->selection.symbol;
  tuning.source.sourceRootOrdinal = imported->selection.rootOrdinal;
  tuning.source.adapterKey = imported->adapterKey;
  tuning.backend = backend;
  tuning.executable = true;
  tuning.topK = 1;
  tuning.measureTop = 0;
  tuning.generator.mode = native::perf::SearchMode::Grid;
  tuning.generator.maxCandidates = static_cast<uint64_t>(options.maxCandidates);
  tuning.mapping.mode = native::mapping::SearchMode::Deterministic;
  tuning.mapping.topK = 1;
  auto searched = native::tuning::runMappedTuningSession(
      *imported->module, imported->space, imported->contract, **target, tuning);
  if (!searched)
    return frontendError("frontend_compilation_failed",
                         llvm::toString(searched.takeError()), {}, "compile");
  if (searched->ranked.empty()) {
    std::string reason = "no executable frontend candidate";
    for (const auto &rejection : searched->rejected)
      reason += "; " + rejection.rejectionReason;
    return frontendError("frontend_compilation_failed", reason, {}, "compile");
  }
  const auto &selected = searched->ranked.front();
  auto compiled = native::tuning::compileMappedTuningSelection(
      *imported->module, imported->space, imported->contract, **target,
      selected, tuning);
  if (!compiled)
    return frontendError("frontend_compilation_failed",
                         llvm::toString(compiled.takeError()), {}, "compile");
  if (!compiled->executable)
    return frontendError("frontend_compilation_failed",
                         "native compiler returned no executable", {},
                         "compile");
  auto state = std::make_unique<FrontendCompilation::State>();
  state->frontendIR = request.canonicalJSON().str();
  state->llkIR = print(imported->module->getOperation());
  state->microIR = compiled->mappedMicroIR;
  const auto &abi = compiled->executable->abi();
  if (auto error = validateKernelAbi(abi))
    return frontendError("frontend_compilation_failed",
                         llvm::toString(std::move(error)), {}, "compile");
  if (abi.inputs.size() != 3 || abi.outputs.size() != 1 ||
      imported->internalDestinationPort != 2)
    return frontendError("frontend_compilation_failed",
                         "unsupported destination adaptation ABI", {},
                         "compile");
  state->destinationPort = imported->internalDestinationPort;
  llvm::json::Array publicPorts;
  for (const auto &map : imported->publicToNativePorts) {
    const auto &port = request.program().tensors[map.publicPort];
    if (port.direction == "input")
      state->inputMap.push_back(map.nativePort);
    publicPorts.push_back(llvm::json::Object{{"public", map.publicPort},
                                             {"native", map.nativePort}});
  }
  const auto &shape = abi.inputs[state->destinationPort].shape;
  uint64_t elements =
      static_cast<uint64_t>(shape[0]) * static_cast<uint64_t>(shape[1]);
  if (elements > std::numeric_limits<size_t>::max() / sizeof(float))
    return frontendError("frontend_compilation_failed",
                         "destination allocation exceeds host range", {},
                         "compile");
  state->destinationBytes = elements * sizeof(float);
  state->destination.reset(
      new (std::nothrow) float[static_cast<size_t>(elements)]);
  if (!state->destination)
    return frontendError("frontend_compilation_failed",
                         "destination allocation failed", {}, "compile");
  auto bound =
      mlir::parseSourceString<mlir::ModuleOp>(state->microIR, &context);
  if (!bound)
    return frontendError("frontend_compilation_failed",
                         "bound inspection did not parse", {}, "compile");
  mlir::Operation *kernel = nullptr;
  bound->walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() == "micro.kernel")
      kernel = op;
  });
  if (!kernel)
    return frontendError("frontend_compilation_failed", "bound kernel missing",
                         {}, "compile");
  auto capabilities = native::framework::describeCapabilities(
      kernel, imported->contract, **target, nullptr, &selected.plan,
      &*compiled);
  auto abiHash = compiled->executable->abiHash();
  llvm::json::Object sourceBindings;
  for (const auto &binding : selected.ranking.candidate.values)
    sourceBindings[binding.first] = binding.second;
  for (const auto &binding : selected.ranking.candidate.symbolicValues)
    sourceBindings[binding.first] = binding.second;
  llvm::json::Array sourceReceipt, mappingReceipt;
  for (const auto &parameter : selected.receipt.source)
    sourceReceipt.push_back(parameter);
  for (const auto &parameter : selected.receipt.mapping)
    mappingReceipt.push_back(parameter);
  llvm::json::Object nativeIdentity{
      {"request", request.canonicalJSON()},
      {"compiler", LLK_COMPILER_VERSION},
      {"target", hex(compiled->targetHash)},
      {"plan", native::mapping::canonicalPlanString(selected.plan)},
      {"abi", hex(abiHash)},
      {"execution", compiled->executable->executionIdentity()}};
  llvm::json::Object report{
      {"backend", options.backend},
      {"source_adapter", imported->adapterKey},
      {"native_request_identity",
       llvm::formatv("{0}", llvm::json::Value(std::move(nativeIdentity)))
           .str()},
      {"frontend_identity", request.canonicalJSON()},
      {"compiler_identity", LLK_COMPILER_VERSION},
      {"workload_identity",
       native::workload::canonicalWorkloadContract(imported->contract)},
      {"source_adapter_identity", imported->adapterKey},
      {"source_graph_hash", hex(selected.sourceGraphHash)},
      {"source_bindings", std::move(sourceBindings)},
      {"binding_receipt",
       llvm::json::Object{{"source", std::move(sourceReceipt)},
                          {"mapping", std::move(mappingReceipt)}}},
      {"binding_hash", hex(selected.bindingHash)},
      {"target_hash", hex(compiled->targetHash)},
      {"machine_hash", hex(compiled->machineHash)},
      {"rules_hash", hex(selected.plan.ruleHash)},
      {"plan_id", hex(compiled->planId)},
      {"abi_hash", hex(abiHash)},
      {"selected_group_count", compiled->backendGroupsRealized},
      {"reference_group_count", compiled->referenceGroupsLowered},
      {"selected_groups_verified", compiled->selectedGroupsVerified},
      {"target_lowered", compiled->targetLowered},
      {"reference_lowered", compiled->referenceLowered},
      {"execution_identity", compiled->executable->executionIdentity()},
      {"stages",
       llvm::json::Array{"frontend", "llk", "micro", "lowered", "executable"}},
      {"capabilities",
       native::framework::capabilityReceiptToJson(capabilities)},
      {"plan_report", selected.planReport},
      {"adaptation",
       llvm::json::Object{{"internal_destination_port", state->destinationPort},
                          {"allocation_bytes", state->destinationBytes},
                          {"public_to_native", std::move(publicPorts)}}},
      {"search",
       llvm::json::Object{
           {"generated", searched->generated},
           {"retained", searched->ranked.size()},
           {"generator_truncated", searched->generatorTruncated},
           {"mapping_truncated", searched->mappingTruncated},
           {"mapping_candidates", searched->mappingCandidateCount},
           {"mapping_routes", searched->mappingRouteCount},
           {"mapping_plans", searched->mappingPlanCount},
           {"rejected_source", searched->rejectedBySource},
           {"rejected_legality", searched->rejectedByLegality},
           {"rejected_mapping", searched->rejectedByMapping},
           {"rejected_compile", searched->rejectedByCompile},
           {"failed_compile_attempts", searched->failedCompileAttempts},
           {"max_candidates", options.maxCandidates},
           {"mode", "deterministic"},
           {"top_k", 1},
           {"max_candidates_per_node", tuning.mapping.maxCandidatesPerNode},
           {"max_instances_per_candidate",
            tuning.mapping.maxInstancesPerCandidate},
           {"max_routes_per_connection", tuning.mapping.maxRoutesPerConnection},
           {"max_connection_combinations",
            tuning.mapping.maxConnectionCombinations},
           {"max_fused_matches", tuning.mapping.maxFusedMatches},
           {"memory_budget_bytes", tuning.mapping.memoryBudgetBytes},
           {"measurement_attempts", searched->measurementAttempts}}}};
  state->reportText =
      llvm::formatv("{0}", llvm::json::Value(std::move(report))).str();
  state->executable = std::move(compiled->executable);
  return FrontendCompilation(std::move(state));
}
} // namespace llk::frontend
