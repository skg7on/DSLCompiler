//===- llk-tune.cpp - LLK schedule tuning driver --------------------------===//
//
// Two modes, selected by whether a Micro search space is given:
//
//   * Micro mode (--input=<space.mlir>): load one `micro.search_space` and a
//     machine::MachineModel, generate candidates, reject the illegal ones, bind
//     the rest to concrete kernels, cost them with the L0/L1 models, and write
//     the ranked top-K as schedule YAML. This is the M12 tuning flow (#50).
//   * Legacy mode (no --input): the pre-Micro grid over BM/BN/BK/VM/VN,
//     num_threads, and grain_size, filtered by an L1 footprint estimate and
//     written as a JSON schedule_db entry to -o. Preserved so the existing
//     schedule-consumption pipeline keeps working.
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/MicroMapping/CandidateInstantiation.h"
#include "LLK/Conversion/MicroMapping/MappedTuningSession.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Machine/MachineModelLoader.h"
#include "LLK/Mapping/CoveringSearch.h"
#include "LLK/Mapping/MappingTarget.h"
#include "LLK/Mapping/PlanReport.h"
#include "LLK/Perf/ScheduleRecord.h"
#include "LLK/Perf/TuningSession.h"
#include "LLK/Target/MappingTargets.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace cl = llvm::cl;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Options
//===----------------------------------------------------------------------===//

static cl::opt<int64_t> tuneM("M", cl::desc("M dimension (rows)"),
                              cl::init(128));
static cl::opt<int64_t> tuneN("N", cl::desc("N dimension (hidden)"),
                              cl::init(4096));
static cl::opt<int64_t> tuneK("K", cl::desc("K dimension (input)"),
                              cl::init(4096));

static cl::opt<std::string> tuneInput(
    "input",
    cl::desc("Micro search-space file to tune; omit for the legacy grid"),
    cl::init(""));
static cl::opt<std::string> tuneWorkload(
    "workload",
    cl::desc("Workload name override for the search space (default: the "
             "space's own workload attribute)"),
    cl::init(""));
static cl::opt<std::string>
    tuneMachine("machine",
                cl::desc("Machine model to evaluate candidates against"),
                cl::init("machines/x86-avx2-v2.yaml"));
static cl::opt<std::string>
    tuneSearch("search", cl::desc("Candidate search: grid or random"),
               cl::init("grid"));
static cl::opt<uint64_t>
    tuneSeed("seed", cl::desc("RNG seed for random search"), cl::init(0));
static cl::opt<uint64_t> tuneMaxCandidates(
    "max-candidates",
    cl::desc("Stop after this many candidates (0 = the whole space)"),
    cl::init(0));
static cl::opt<unsigned>
    tunePerfLevel("perf-level",
                  cl::desc("Prediction depth: 0 static bound, 1 resource"),
                  cl::init(1));
static cl::opt<uint64_t> tuneTopK("top-k", cl::desc("Schedule records to keep"),
                                  cl::init(10));
static cl::opt<std::string>
    tuneOutput("output", cl::desc("Schedule YAML output file (Micro mode)"),
               cl::init("tune_result.yaml"));

static cl::opt<std::string>
    tuneMappingTarget("mapping-target",
                      cl::desc("Use a registered target-aware mapping tuner"),
                      cl::init(""));
static cl::opt<std::string>
    tuneMappingRoot("mapping-root",
                    cl::desc("Root containing target mapping configuration"),
                    cl::init("."));
static cl::opt<std::string>
    tuneMappingMode("mapping-mode",
                    cl::desc("Mapping search mode: deterministic, beam, exact"),
                    cl::init("exact"));
static cl::opt<std::string> tuneMappingBackend(
    "mapping-backend",
    cl::desc("Mapped compile backend: reference or selected-target"),
    cl::init("reference"));
static cl::opt<std::string>
    tuneMappingReport("mapping-report",
                      cl::desc("Optional mapped tuning JSON report"),
                      cl::init(""));
static cl::opt<std::string> tuneCandidateArtifacts(
    "candidate-artifacts",
    cl::desc("Directory for replayable candidate sources and plans"),
    cl::init(""));
static cl::opt<std::string> tuneCandidateSource(
    "candidate-source",
    cl::desc("Candidate source mode: semantic, concrete-micro, synthetic"),
    cl::init("semantic"));
static cl::opt<std::string>
    tuneSourceSymbol("source-symbol",
                     cl::desc("Semantic source function to instantiate"),
                     cl::init(""));
static cl::opt<unsigned>
    tuneSourceRoot("source-root",
                   cl::desc("Zero-based semantic contraction root ordinal"),
                   cl::init(0));
static cl::opt<uint64_t> tuneMeasureTop(
    "measure-top",
    cl::desc("Mapped candidates to measure (0 disables measurement)"),
    cl::init(0));
static cl::opt<std::string>
    tuneMeasurementInputs("measurement-inputs",
                          cl::desc("Versioned JSON typed input/output fixture"),
                          cl::init(""));

static cl::opt<std::string> tuneInputDType("input-dtype",
                                           cl::desc("Input element dtype"),
                                           cl::init("bf16"));
static cl::opt<std::string> tuneWeightDType("weight-dtype",
                                            cl::desc("Weight element dtype"),
                                            cl::init("bf16"));
static cl::opt<std::string> tuneAccumulatorDType("accumulator-dtype",
                                                 cl::desc("Accumulator dtype"),
                                                 cl::init("f32"));
static cl::opt<std::string> tuneOutputDType("output-dtype",
                                            cl::desc("Output element dtype"),
                                            cl::init("bf16"));

// Legacy grid-search options.
static cl::opt<std::string>
    tuneLegacyOutput("o", cl::desc("Legacy JSON output file"),
                     cl::init("tune_result.json"));
static cl::opt<bool> tuneDryRun("dry-run",
                                cl::desc("Generate configs without measuring"),
                                cl::init(false));

namespace {

int reportError(const llvm::Twine &message) {
  llvm::errs() << "llk-tune: error: " << message << "\n";
  return 1;
}

//===----------------------------------------------------------------------===//
// Legacy grid search
//===----------------------------------------------------------------------===//

/// A single tuning configuration with tile sizes and parallelism knobs.
struct TuningConfig {
  int64_t BM, BN, BK;
  int64_t VM, VN;
  int64_t vector_width;
  int64_t num_threads;
  int64_t grain_size;
  std::string parallel_axis;
  double gflops = 0.0; // measured (0 = not measured yet)

  /// Estimated L1 footprint of the working set for this config (bytes).
  /// X tile: BM * BK * 2 (BF16), 2 weight tiles: 2 * BK * BN * 2 (BF16).
  size_t l1Footprint() const {
    return (BM * BK + 2 * BK * BN) * 2; // bytes
  }
};

/// M bucket classifier (matches ShapeSpecialization pass).
static int classifyM(int64_t M) {
  if (M == 1)
    return 0;
  if (M <= 4)
    return 1;
  if (M <= 16)
    return 2;
  if (M <= 64)
    return 3;
  return 4;
}

/// Generate candidate configs for a given shape regime.
/// Filters by L1 cache capacity (80% of 32KB for the working set).
static std::vector<TuningConfig> generateConfigs(int64_t M, int64_t N,
                                                 int64_t K) {
  std::vector<TuningConfig> configs;
  constexpr size_t l1Limit = static_cast<size_t>(32 * 1024 * 0.8); // 25.6 KB

  int M_bucket = classifyM(M);

  for (int64_t BM : {1, 4, 8, 16, 32, 64}) {
    // GEMV-like: only BM=1
    if (M_bucket == 0 && BM != 1)
      continue;
    // Large batch: skip BM=1 (too small)
    if (M_bucket >= 3 && BM <= 4)
      continue;

    for (int64_t BN : {16, 32, 64, 128, 256}) {
      for (int64_t BK : {32, 64, 128, 256}) {
        for (int64_t VM : {1, 2, 4}) {
          for (int64_t VN : {4, 8}) {
            // Vector width currently fixed at 8 (AVX2 BF16 → 8 x 16-bit)
            int64_t VW = 8;

            TuningConfig cfg;
            cfg.BM = BM;
            cfg.BN = BN;
            cfg.BK = BK;
            cfg.VM = VM;
            cfg.VN = VN;
            cfg.vector_width = VW;

            // L1 capacity check
            if (cfg.l1Footprint() > l1Limit)
              continue;

            // Thread count variants
            for (int64_t nt : {1, 2, 4, 8}) {
              cfg.num_threads = nt;

              // Total tiles in M dimension
              int64_t mTiles = (M + BM - 1) / BM;
              int64_t nTiles = (N + BN - 1) / BN;

              for (int64_t gs : {1, 2, 4}) {
                // Grain must not exceed total tiles
                int64_t totalTiles =
                    (cfg.parallel_axis == "m") ? mTiles : nTiles;
                if (gs > totalTiles)
                  continue;

                cfg.grain_size = gs;
                cfg.parallel_axis = (M_bucket == 0) ? "n" : "m";
                cfg.gflops = 0.0;

                configs.push_back(cfg);
              }
            }
          }
        }
      }
    }
  }

  return configs;
}

/// Write results as a JSON schedule_db entry.
static int writeLegacyResults(const std::string &path,
                              const std::vector<TuningConfig> &configs,
                              int64_t M, int64_t N, int64_t K) {
  int M_bucket = classifyM(M);

  // Sort by GFLOPS descending
  auto sorted = configs;
  std::sort(sorted.begin(), sorted.end(),
            [](const TuningConfig &a, const TuningConfig &b) {
              return a.gflops > b.gflops;
            });

  // Write top-3 results
  llvm::json::Array entries;
  size_t count = std::min(sorted.size(), size_t(3));
  for (size_t i = 0; i < count; i++) {
    const auto &c = sorted[i];

    llvm::json::Object entry;
    entry["operation"] = "fused_swiglu";
    entry["target"] = "x86-avx2";
    entry["dtype"] = "bf16";
    entry["math_mode"] = "bounded_fast";

    llvm::json::Object shape;
    shape["M_bucket"] = M_bucket;
    shape["N"] = N;
    shape["K"] = K;
    entry["shape"] = std::move(shape);

    llvm::json::Object schedule;
    schedule["BM"] = c.BM;
    schedule["BN"] = c.BN;
    schedule["BK"] = c.BK;
    schedule["VM"] = c.VM;
    schedule["VN"] = c.VN;
    schedule["vector_width"] = c.vector_width;
    schedule["num_threads"] = c.num_threads;
    schedule["parallel_axis"] = c.parallel_axis;
    schedule["grain_size"] = c.grain_size;
    entry["schedule"] = std::move(schedule);

    // Record measured gflops as metadata
    entry["measured_gflops"] = c.gflops;

    entries.push_back(std::move(entry));
  }

  llvm::json::Object root;
  root["version"] = 1;
  root["entries"] = std::move(entries);

  std::ofstream ofs(path);
  if (!ofs)
    return reportError("cannot open output file: " + path);

  std::string jsonStr;
  llvm::raw_string_ostream rss(jsonStr);
  rss << llvm::json::Value(std::move(root));
  ofs << jsonStr;
  if (!ofs)
    return reportError("failed to write output file: " + path);

  llvm::outs() << "Wrote top-" << count << " configs to " << path << "\n";
  return 0;
}

/// The pre-Micro grid search, preserved unchanged for existing consumers.
int runLegacyGrid() {
  int M_bucket = classifyM(tuneM);
  auto configs = generateConfigs(tuneM, tuneN, tuneK);

  llvm::outs() << "M=" << tuneM << " (bucket " << M_bucket << ") N=" << tuneN
               << " K=" << tuneK << "\n";
  llvm::outs() << "Generated " << configs.size()
               << " candidate configs (L1-filtered)\n";

  if (tuneDryRun) {
    llvm::outs() << "\nTop candidate configs (estimated):\n";
    size_t n = std::min(configs.size(), size_t(5));
    for (size_t i = 0; i < n; i++) {
      const auto &c = configs[i];
      llvm::outs() << "  [" << i << "] BM=" << c.BM << " BN=" << c.BN
                   << " BK=" << c.BK << " VM=" << c.VM << " VN=" << c.VN
                   << " VW=" << c.vector_width << " threads=" << c.num_threads
                   << " grain=" << c.grain_size << " axis=" << c.parallel_axis
                   << " L1=" << c.l1Footprint() << "B\n";
    }
  }

  return writeLegacyResults(tuneLegacyOutput, configs, tuneM, tuneN, tuneK);
}

//===----------------------------------------------------------------------===//
// Micro search-space tuning
//===----------------------------------------------------------------------===//

bool parseSearchMode(mlir::llk::perf::SearchMode &mode,
                     const std::string &name) {
  if (name == "grid") {
    mode = mlir::llk::perf::SearchMode::Grid;
    return true;
  }
  if (name == "random") {
    mode = mlir::llk::perf::SearchMode::Random;
    return true;
  }
  return false;
}

bool parseMappingMode(mlir::llk::mapping::SearchMode &mode,
                      llvm::StringRef name) {
  if (name == "deterministic")
    mode = mlir::llk::mapping::SearchMode::Deterministic;
  else if (name == "beam")
    mode = mlir::llk::mapping::SearchMode::Beam;
  else if (name == "exact")
    mode = mlir::llk::mapping::SearchMode::Exact;
  else
    return false;
  return true;
}

struct FixturePort {
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<double> data;
  double absoluteTolerance = 0.0;
  double relativeTolerance = 0.0;
};

struct MeasurementFixture {
  uint64_t warmup = 0;
  uint64_t repeat = 1;
  std::vector<FixturePort> inputs;
  std::vector<FixturePort> outputs;
};

llvm::Error fixtureError(const llvm::Twine &message) {
  std::string text = message.str();
  return llvm::createStringError(llvm::inconvertibleErrorCode(), "%s",
                                 text.c_str());
}

llvm::Expected<FixturePort> parseFixturePort(const llvm::json::Value &value,
                                             bool output) {
  const llvm::json::Object *object = value.getAsObject();
  if (!object)
    return fixtureError("measurement port must be an object");
  FixturePort port;
  auto dtype = object->getString("dtype");
  auto shape = object->getArray("shape");
  const llvm::json::Value *data = object->get("data");
  if (!dtype || !shape || !data)
    return fixtureError("measurement port needs dtype, shape, and data");
  port.dtype = dtype->str();
  if (port.dtype != "f32" && port.dtype != "bf16" && port.dtype != "f16" &&
      port.dtype != "i32" && port.dtype != "i8")
    return fixtureError("unsupported measurement dtype '" + port.dtype + "'");
  for (const llvm::json::Value &extent : *shape) {
    auto integer = extent.getAsInteger();
    if (!integer || *integer <= 0)
      return fixtureError(
          "measurement shape extents must be positive integers");
    port.shape.push_back(*integer);
  }
  if (port.shape.size() != 2)
    return fixtureError("measurement ports must have rank-two shapes");
  uint64_t rows = static_cast<uint64_t>(port.shape[0]);
  uint64_t columns = static_cast<uint64_t>(port.shape[1]);
  if (columns && rows > std::numeric_limits<uint64_t>::max() / columns)
    return fixtureError("measurement tensor element count overflows");
  uint64_t count = rows * columns;
  if (count > std::numeric_limits<size_t>::max())
    return fixtureError("measurement tensor is too large");
  if (count > port.data.max_size())
    return fixtureError(
        "measurement tensor exceeds the supported element count");
  if (auto scalar = data->getAsNumber()) {
    port.data.assign(static_cast<size_t>(count), *scalar);
  } else if (const llvm::json::Array *array = data->getAsArray()) {
    if (array->size() != count)
      return fixtureError("measurement data count does not match shape");
    for (const llvm::json::Value &element : *array) {
      auto number = element.getAsNumber();
      if (!number || !std::isfinite(*number))
        return fixtureError("measurement data values must be finite numbers");
      port.data.push_back(*number);
    }
  } else {
    return fixtureError("measurement data must be a number or numeric array");
  }
  if (llvm::any_of(port.data, [](double n) { return !std::isfinite(n); }))
    return fixtureError("measurement data values must be finite numbers");
  double maximumFloat =
      port.dtype == "f16"
          ? 65504.0
          : static_cast<double>(std::numeric_limits<float>::max());
  if ((port.dtype == "f16" || port.dtype == "bf16" || port.dtype == "f32") &&
      llvm::any_of(port.data,
                   [&](double n) { return std::fabs(n) > maximumFloat; }))
    return fixtureError("floating measurement data exceeds the dtype range");
  if (port.dtype == "i32" || port.dtype == "i8") {
    double minimum =
        port.dtype == "i32"
            ? static_cast<double>(std::numeric_limits<int32_t>::min())
            : static_cast<double>(std::numeric_limits<int8_t>::min());
    double maximum =
        port.dtype == "i32"
            ? static_cast<double>(std::numeric_limits<int32_t>::max())
            : static_cast<double>(std::numeric_limits<int8_t>::max());
    for (double number : port.data)
      if (std::trunc(number) != number || number < minimum || number > maximum)
        return fixtureError(
            "integer measurement data is outside its dtype range");
  }
  if (output) {
    if (auto tolerance = object->getNumber("absolute_tolerance"))
      port.absoluteTolerance = *tolerance;
    if (auto tolerance = object->getNumber("relative_tolerance"))
      port.relativeTolerance = *tolerance;
    if (!std::isfinite(port.absoluteTolerance) ||
        !std::isfinite(port.relativeTolerance) || port.absoluteTolerance < 0 ||
        port.relativeTolerance < 0)
      return fixtureError(
          "measurement tolerances must be finite and nonnegative");
  }
  return port;
}

llvm::Expected<MeasurementFixture>
loadMeasurementFixture(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return fixtureError("cannot read measurement fixture '" + path +
                        "': " + buffer.getError().message());
  auto json = llvm::json::parse((*buffer)->getBuffer());
  if (!json)
    return json.takeError();
  const llvm::json::Object *object = json->getAsObject();
  if (!object || object->getInteger("version") != 1)
    return fixtureError("measurement fixture must be a version 1 JSON object");
  MeasurementFixture fixture;
  auto warmup = object->getInteger("warmup");
  auto repeat = object->getInteger("repeat");
  auto inputs = object->getArray("inputs");
  auto outputs = object->getArray("outputs");
  constexpr int64_t kMaxInvocations = 100000;
  if (!warmup || *warmup < 0 || *warmup > kMaxInvocations || !repeat ||
      *repeat <= 0 || *repeat > kMaxInvocations || !inputs || !outputs ||
      outputs->empty())
    return fixtureError("measurement fixture needs warmup/repeat in [0, "
                        "100000], inputs, and outputs");
  fixture.warmup = static_cast<uint64_t>(*warmup);
  fixture.repeat = static_cast<uint64_t>(*repeat);
  for (const llvm::json::Value &value : *inputs) {
    auto port = parseFixturePort(value, false);
    if (!port)
      return port.takeError();
    fixture.inputs.push_back(std::move(*port));
  }
  for (const llvm::json::Value &value : *outputs) {
    auto port = parseFixturePort(value, true);
    if (!port)
      return port.takeError();
    fixture.outputs.push_back(std::move(*port));
  }
  return fixture;
}

std::optional<::llk::InvocationElementType>
parseInvocationType(llvm::StringRef type) {
  if (type == "f32")
    return ::llk::InvocationElementType::F32;
  if (type == "bf16")
    return ::llk::InvocationElementType::BF16;
  if (type == "f16")
    return ::llk::InvocationElementType::F16;
  if (type == "i32")
    return ::llk::InvocationElementType::I32;
  if (type == "i8")
    return ::llk::InvocationElementType::I8;
  return std::nullopt;
}

size_t invocationElementBytes(llvm::StringRef type) {
  return type == "f32" || type == "i32" ? 4 : type == "i8" ? 1 : 2;
}

std::optional<mlir::llk::mapping::CostMetric>
mapObjectiveMetric(llvm::StringRef name) {
  using mlir::llk::mapping::CostMetric;
  if (name == "latency_cycles")
    return CostMetric::LatencyCycles;
  if (name == "dram_bytes")
    return CostMetric::DramBytes;
  if (name == "sram_bytes" || name == "local_bytes")
    return CostMetric::LocalBytes;
  if (name == "spill_bytes")
    return CostMetric::SpillBytes;
  if (name == "matrix_utilization" || name == "compute_utilization")
    return CostMetric::ComputeUtilization;
  if (name == "dma_utilization" || name == "transfer_utilization")
    return CostMetric::TransferUtilization;
  return std::nullopt;
}

void storeFixtureNumber(uint8_t *destination, llvm::StringRef type,
                        double number) {
  if (type == "f32") {
    float value = static_cast<float>(number);
    std::memcpy(destination, &value, sizeof(value));
  } else if (type == "bf16" || type == "f16") {
    llvm::APFloat value(number);
    const llvm::fltSemantics &semantics =
        type == "bf16" ? llvm::APFloat::BFloat() : llvm::APFloat::IEEEhalf();
    bool losesInfo = false;
    value.convert(semantics, llvm::APFloat::rmNearestTiesToEven, &losesInfo);
    uint16_t bits =
        static_cast<uint16_t>(value.bitcastToAPInt().getZExtValue());
    std::memcpy(destination, &bits, sizeof(bits));
  } else if (type == "i32") {
    int32_t value = static_cast<int32_t>(number);
    std::memcpy(destination, &value, sizeof(value));
  } else {
    int8_t value = static_cast<int8_t>(number);
    std::memcpy(destination, &value, sizeof(value));
  }
}

double loadFixtureNumber(const uint8_t *source, llvm::StringRef type) {
  if (type == "f32") {
    float value;
    std::memcpy(&value, source, sizeof(value));
    return value;
  }
  if (type == "bf16" || type == "f16") {
    uint16_t bits;
    std::memcpy(&bits, source, sizeof(bits));
    const llvm::fltSemantics &semantics =
        type == "bf16" ? llvm::APFloat::BFloat() : llvm::APFloat::IEEEhalf();
    return llvm::APFloat(semantics, llvm::APInt(16, bits)).convertToDouble();
  }
  if (type == "i32") {
    int32_t value;
    std::memcpy(&value, source, sizeof(value));
    return value;
  }
  int8_t value;
  std::memcpy(&value, source, sizeof(value));
  return value;
}

llvm::Expected<mlir::llk::tuning::OwnedInvocationBuffers>
makeFixtureBuffers(const MeasurementFixture &fixture,
                   const ::llk::KernelAbi &abi) {
  using namespace mlir::llk::tuning;
  if (fixture.inputs.size() != abi.inputs.size() ||
      fixture.outputs.size() != abi.outputs.size())
    return fixtureError(
        "measurement fixture port counts do not match the kernel ABI");
  OwnedInvocationBuffers buffers;
  auto append =
      [&](const FixturePort &fixturePort, const ::llk::KernelAbi::Port &abiPort,
          std::vector<::llk::InvocationBuffer2D> &ports) -> llvm::Error {
    if (fixturePort.dtype != abiPort.elementType ||
        fixturePort.shape != abiPort.shape)
      return fixtureError("measurement fixture port dtype or shape does not "
                          "match the kernel ABI");
    auto type = parseInvocationType(fixturePort.dtype);
    if (!type)
      return fixtureError("unsupported kernel ABI dtype");
    size_t elementBytes = invocationElementBytes(fixturePort.dtype);
    if (fixturePort.data.size() >
        std::numeric_limits<size_t>::max() / elementBytes)
      return fixtureError("measurement buffer byte size overflows");
    size_t bytes = fixturePort.data.size() * elementBytes;
    buffers.storage.emplace_back(bytes);
    std::vector<uint8_t> &storage = buffers.storage.back();
    for (size_t i = 0; i < fixturePort.data.size(); ++i)
      storeFixtureNumber(storage.data() +
                             i * invocationElementBytes(fixturePort.dtype),
                         fixturePort.dtype, fixturePort.data[i]);
    auto *data = storage.data();
    ports.push_back({{data, data, 0, fixturePort.shape[0], fixturePort.shape[1],
                      fixturePort.shape[1], 1},
                     *type,
                     bytes});
    return llvm::Error::success();
  };
  for (size_t i = 0; i < fixture.inputs.size(); ++i)
    if (llvm::Error error =
            append(fixture.inputs[i], abi.inputs[i], buffers.inputs))
      return std::move(error);
  for (size_t i = 0; i < fixture.outputs.size(); ++i)
    if (llvm::Error error =
            append(fixture.outputs[i], abi.outputs[i], buffers.outputs))
      return std::move(error);
  return buffers;
}

int runMappedSearch(ModuleOp source, const mlir::llk::perf::SearchSpace &space,
                    const mlir::llk::perf::WorkloadShape &shape) {
  namespace perf = mlir::llk::perf;
  namespace mapping = mlir::llk::mapping;
  namespace tuning = mlir::llk::tuning;
  if (tuneMappingTarget.empty())
    return reportError("--mapping-target is required for mapped tuning");
  if (tuneMeasureTop && tuneMeasurementInputs.empty())
    return reportError("--measure-top requires --measurement-inputs");
  if (!tuneMeasureTop && !tuneMeasurementInputs.empty())
    return reportError("--measurement-inputs requires --measure-top > 0");
  std::optional<MeasurementFixture> fixture;
  if (!tuneMeasurementInputs.empty()) {
    auto loadedFixture = loadMeasurementFixture(tuneMeasurementInputs);
    if (!loadedFixture)
      return reportError(llvm::toString(loadedFixture.takeError()));
    fixture = std::move(*loadedFixture);
  }
  mapping::SearchMode mappingMode;
  if (!parseMappingMode(mappingMode, tuneMappingMode))
    return reportError("unsupported --mapping-mode=" + tuneMappingMode);
  if (tuneCandidateSource != "semantic" &&
      tuneCandidateSource != "concrete-micro" &&
      tuneCandidateSource != "synthetic")
    return reportError("unsupported --candidate-source=" + tuneCandidateSource);
  mlir::llk::target::registerAllMappingTargets();
  const bool explicitMachine = tuneMachine.getNumOccurrences() != 0;
  auto target = mapping::createRegisteredMappingTarget(
      tuneMappingTarget, tuneMappingRoot,
      explicitMachine ? llvm::StringRef(tuneMachine) : llvm::StringRef());
  if (!target)
    return reportError(llvm::toString(target.takeError()));
  std::string machinePath = tuneMachine;
  if (!explicitMachine) {
    llvm::StringRef profile = tuneMappingTarget == "x86-avx2"
                                  ? "machines/x86-avx2-v2.yaml"
                                  : "machines/generic-ai-accel-v2.yaml";
    machinePath = (llvm::Twine(tuneMappingRoot) + "/" + profile).str();
  }
  tuning::MappedTuningOptions options;
  perf::SearchMode candidateMode;
  if (!parseSearchMode(candidateMode, tuneSearch))
    return reportError("unsupported --search=" + tuneSearch);
  options.generator.mode = candidateMode;
  options.generator.seed = tuneSeed;
  if (tuneMaxCandidates)
    options.generator.maxCandidates = tuneMaxCandidates;
  options.topK = tuneTopK;
  options.measureTop = tuneMeasureTop;
  options.backend = tuneMappingBackend == "reference"
                        ? ::llk::MappedBackend::Reference
                        : ::llk::MappedBackend::SelectedTarget;
  if (tuneMappingBackend != "reference" &&
      tuneMappingBackend != "selected-target")
    return reportError("unsupported --mapping-backend=" + tuneMappingBackend);
  options.mapping.mode = mappingMode;
  options.mapping.topK = 8;
  auto &mappingObjective = options.mapping.objective;
  bool measuredObjective = space.objective.primaryMetric == "measured_ns" ||
                           space.objective.primaryMetric == "measured_gflops";
  mappingObjective.minimize =
      measuredObjective ||
      space.objective.direction == perf::ObjectiveDirection::Minimize;
  if (auto metric = mapObjectiveMetric(space.objective.primaryMetric))
    mappingObjective.primary = *metric;
  else if (space.objective.primaryMetric != "measured_ns" &&
           space.objective.primaryMetric != "measured_gflops")
    return reportError("objective metric '" + space.objective.primaryMetric +
                       "' cannot rank mapped costs");
  for (const std::string &secondary : space.objective.secondaryMetrics)
    if (auto metric = mapObjectiveMetric(secondary))
      mappingObjective.secondary.push_back(*metric);
  options.source.machine = &(*target)->machine();
  options.source.sourceSymbol = tuneSourceSymbol;
  options.source.sourceRootOrdinal = tuneSourceRoot;
  options.source.sourceMode = tuneCandidateSource == "semantic"
                                  ? tuning::SourceMode::SemanticSource
                              : tuneCandidateSource == "concrete-micro"
                                  ? tuning::SourceMode::ConcreteMicro
                                  : tuning::SourceMode::Synthetic;
  options.executable = true;
  if (fixture) {
    options.measurement.inputs = [fixture =
                                      *fixture](const ::llk::KernelAbi &abi)
        -> llvm::Expected<tuning::OwnedInvocationBuffers> {
      return makeFixtureBuffers(fixture, abi);
    };
    options.measurement.measure =
        [warmup = fixture->warmup, repeat = fixture->repeat](
            const tuning::MappedMeasurementRequest &request)
        -> llvm::Expected<std::optional<perf::CandidateMetrics>> {
      for (uint64_t i = 0; i < warmup; ++i)
        if (llvm::Error error =
                request.executable.invoke(request.inputs, request.outputs))
          return std::move(error);
      std::vector<double> samples;
      samples.reserve(repeat);
      for (uint64_t i = 0; i < repeat; ++i) {
        auto begin = std::chrono::steady_clock::now();
        if (llvm::Error error =
                request.executable.invoke(request.inputs, request.outputs))
          return std::move(error);
        auto end = std::chrono::steady_clock::now();
        samples.push_back(
            std::chrono::duration<double, std::nano>(end - begin).count());
      }
      llvm::sort(samples);
      double median = samples[samples.size() / 2];
      if (samples.size() % 2 == 0)
        median = (samples[samples.size() / 2 - 1] + median) / 2.0;
      perf::CandidateMetrics metrics;
      metrics.measuredNs = median;
      return std::optional<perf::CandidateMetrics>(std::move(metrics));
    };
    options.measurement.verify =
        [fixture = *fixture](
            const tuning::MappedTuningCandidate &,
            llvm::ArrayRef<::llk::InvocationBuffer2D>,
            llvm::ArrayRef<::llk::InvocationBuffer2D> outputs) -> llvm::Error {
      if (outputs.size() != fixture.outputs.size())
        return fixtureError(
            "measurement output count changed after ABI validation");
      for (size_t i = 0; i < outputs.size(); ++i) {
        const FixturePort &expected = fixture.outputs[i];
        const auto *data =
            static_cast<const uint8_t *>(outputs[i].descriptor.aligned);
        for (size_t element = 0; element < expected.data.size(); ++element) {
          double actual = loadFixtureNumber(
              data + element * invocationElementBytes(expected.dtype),
              expected.dtype);
          double wanted = expected.data[element];
          double tolerance = expected.absoluteTolerance +
                             expected.relativeTolerance * std::fabs(wanted);
          if (!std::isfinite(actual) || std::fabs(actual - wanted) > tolerance)
            return fixtureError("measurement output " + std::to_string(i) +
                                " element " + std::to_string(element) +
                                " differs from its expected value");
        }
      }
      return llvm::Error::success();
    };
  }
  auto report =
      tuning::runMappedTuningSession(source, space, shape, **target, options);
  if (!report)
    return reportError(llvm::toString(report.takeError()));
  if (report->ranked.empty()) {
    std::string why =
        "mapped search produced no candidates (source=" +
        std::to_string(report->rejectedBySource) +
        ", legality=" + std::to_string(report->rejectedByLegality) +
        ", mapping=" + std::to_string(report->rejectedByMapping) +
        ", compile=" + std::to_string(report->rejectedByCompile) + ")";
    if (!report->rejected.empty())
      why += ": " + report->rejected.front().rejectionReason;
    return reportError(why);
  }

  llvm::json::Array rows;
  llvm::json::Array measurements;
  std::vector<perf::ScheduleRecord> records;
  for (const auto &candidate : report->ranked) {
    llvm::json::Object row;
    row["candidate_id"] = candidate.ranking.candidate.id;
    llvm::json::Object bindings;
    for (const auto &v : candidate.ranking.candidate.values)
      bindings[v.first] = v.second;
    for (const auto &v : candidate.ranking.candidate.symbolicValues)
      bindings[v.first] = v.second;
    row["bindings"] = std::move(bindings);
    row["source_mode"] = tuneCandidateSource;
    row["plan_id"] = llvm::formatv("{0:x}", candidate.plan.id).str();
    row["binding_hash"] = llvm::formatv("{0:x}", candidate.bindingHash).str();
    row["plan_report"] = candidate.planReport;
    row["predicted_cycles"] = candidate.ranking.metrics.predictedCycles;
    row["predicted_ns"] = candidate.ranking.metrics.predictedNs;
    row["matrix_utilization"] = candidate.ranking.metrics.matrixUtilization;
    row["dma_utilization"] = candidate.ranking.metrics.dmaUtilization;
    row["dram_bytes"] = candidate.ranking.metrics.dramBytes;
    row["sram_bytes"] = candidate.ranking.metrics.sramBytes;
    row["bottleneck"] = candidate.ranking.metrics.bottleneck;
    row["predicted_cycles_origin"] = "static-model";
    row["predicted_ns_origin"] = "static-model";
    llvm::json::Object metricOrigins;
    metricOrigins["predicted_cycles"] = "static-model";
    metricOrigins["predicted_ns"] = "static-model";
    metricOrigins["matrix_utilization"] = "static-model";
    metricOrigins["dma_utilization"] = "static-model";
    metricOrigins["dram_bytes"] = "static-model";
    metricOrigins["sram_bytes"] = "static-model";
    row["metric_origins"] = std::move(metricOrigins);
    if (candidate.measurementIdentity) {
      const auto &identity = *candidate.measurementIdentity;
      llvm::json::Object id;
      id["canonical"] = identity.canonical;
      id["content_hash"] = identity.contentHash;
      id["plan_id"] = llvm::formatv("{0:x}", identity.planId).str();
      id["source_graph_hash"] =
          llvm::formatv("{0:x}", identity.originalGraphHash).str();
      id["instantiated_graph_hash"] =
          llvm::formatv("{0:x}", identity.instantiatedGraphHash).str();
      id["binding_hash"] = llvm::formatv("{0:x}", identity.bindingHash).str();
      id["abi_hash"] = llvm::formatv("{0:x}", identity.abiHash).str();
      id["target"] = identity.targetIdentity;
      id["backend"] = identity.backendIdentity;
      llvm::json::Array operationKeys;
      for (const auto &key : identity.operationKeys)
        operationKeys.push_back(key);
      id["operation_keys"] = std::move(operationKeys);
      llvm::json::Array connectionKeys;
      for (const auto &key : identity.connectionKeys)
        connectionKeys.push_back(key);
      id["connection_keys"] = std::move(connectionKeys);
      llvm::json::Object measured;
      measured["candidate_id"] = candidate.ranking.candidate.id;
      measured["measured_ns"] = *candidate.ranking.metrics.measuredNs;
      measured["measured_ns_origin"] = "runtime-observation";
      measured["warmup"] = fixture->warmup;
      measured["repeat"] = fixture->repeat;
      measured["identity"] = std::move(id);
      measurements.push_back(std::move(measured));
    }
    perf::ScheduleRecord record;
    record.workload = space.workload;
    record.shape = shape;
    record.target = (*target)->machine().target;
    record.machine = machinePath;
    record.candidate = candidate.ranking.candidate;
    record.metrics = candidate.ranking.metrics;
    record.mappingProvenanceVersion = 1;
    record.mappingPlanId = llvm::formatv("{0:x}", candidate.plan.id).str();
    record.mappingBindingHash =
        llvm::formatv("{0:x}", candidate.bindingHash).str();
    if (candidate.ranking.metrics.measuredNs) {
      record.measured = true;
      record.warmup = fixture ? static_cast<unsigned>(fixture->warmup) : 0;
      record.repeat = fixture ? static_cast<unsigned>(fixture->repeat) : 0;
      record.medianNs = *candidate.ranking.metrics.measuredNs;
      record.measuredGflops =
          candidate.ranking.metrics.measuredGflops.value_or(0.0);
    }
    if (!tuneCandidateArtifacts.empty()) {
      std::string dir = tuneCandidateArtifacts + "/" + record.candidate.id;
      if (std::error_code ec = llvm::sys::fs::create_directories(dir))
        return reportError("cannot create candidate artifact directory: " +
                           ec.message());
      std::string sourcePath = dir + "/source.mlir",
                  planPath = dir + "/plan.json";
      tuning::CandidateInstantiationOptions sourceOptions = options.source;
      sourceOptions.machine = &(*target)->machine();
      auto instance = tuning::instantiateCandidate(
          source, space, record.candidate, shape, sourceOptions);
      if (!instance)
        return reportError(llvm::toString(instance.takeError()));
      std::error_code ec;
      llvm::raw_fd_ostream sourceOut(sourcePath, ec);
      if (ec)
        return reportError("cannot write " + sourcePath);
      instance->module->print(sourceOut);
      sourceOut.close();
      std::ofstream planOut(planPath);
      planOut << candidate.planReport;
      if (!planOut)
        return reportError("cannot write " + planPath);
      record.mappingSourceArtifact = sourcePath;
      record.mappingPlanReport = planPath;
      row["source_artifact"] = sourcePath;
    }
    records.push_back(std::move(record));
    rows.push_back(std::move(row));
  }
  if (llvm::Error e = perf::writeScheduleYamlFile(tuneOutput, records))
    return reportError(llvm::toString(std::move(e)));
  if (!tuneMappingReport.empty()) {
    llvm::json::Object out;
    out["schema_version"] = 1;
    out["target"] = tuneMappingTarget;
    out["machine"] = machinePath;
    out["execution_mode"] = "lowering-validation";
    out["measurement"] = "optional-sidecar";
    out["backend_requested"] = tuneMappingBackend;
    out["objective"] = space.objective.primaryMetric;
    out["objective_direction"] =
        perf::stringifyObjectiveDirection(space.objective.direction).str();
    out["generated"] = report->generated;
    out["mapping_truncated"] = report->mappingTruncated;
    out["generator_truncated"] = report->generatorTruncated;
    out["mapping_candidate_count"] = report->mappingCandidateCount;
    out["mapping_route_count"] = report->mappingRouteCount;
    out["mapping_plan_count"] = report->mappingPlanCount;
    out["rejected_by_source"] = report->rejectedBySource;
    out["rejected_by_legality"] = report->rejectedByLegality;
    out["rejected_by_mapping"] = report->rejectedByMapping;
    out["rejected_by_compile"] = report->rejectedByCompile;
    out["candidates"] = std::move(rows);
    std::error_code ec;
    llvm::raw_fd_ostream os(tuneMappingReport, ec);
    if (ec)
      return reportError("cannot open mapping report: " + ec.message());
    os << llvm::json::Value(std::move(out)) << "\n";
  }
  if (!measurements.empty()) {
    std::string measurementPath =
        (tuneMappingReport.empty() ? tuneOutput.getValue()
                                   : tuneMappingReport.getValue()) +
        ".measurements.json";
    llvm::json::Object observed;
    observed["schema_version"] = 1;
    observed["target"] = tuneMappingTarget;
    observed["warmup"] = fixture->warmup;
    observed["repeat"] = fixture->repeat;
    observed["setup_policy"] = "fixture-values-before-timing";
    observed["timing_policy"] =
        "steady_clock-per-invocation; median of repeat samples";
    observed["records"] = std::move(measurements);
    std::error_code ec;
    llvm::raw_fd_ostream os(measurementPath, ec);
    if (ec)
      return reportError("cannot open measurement sidecar: " + ec.message());
    os << llvm::json::Value(std::move(observed)) << "\n";
  }
  return 0;
}

int runMicroSearch() {
  namespace perf = mlir::llk::perf;

  // Check values rather than getNumOccurrences(): LLVM command-line option
  // aliases/normalization can leave occurrence counts at zero even when the
  // user supplied a target-aware option. Silently falling through to the
  // legacy tuner would otherwise discard those options.
  const bool mappedOptionsUsed =
      (tuneMappingRoot != "." && !tuneMappingRoot.empty()) ||
      (tuneMappingMode != "exact" && !tuneMappingMode.empty()) ||
      (tuneMappingBackend != "reference" && !tuneMappingBackend.empty()) ||
      !tuneMappingReport.empty() || !tuneCandidateArtifacts.empty() ||
      tuneCandidateSource != "semantic" || !tuneSourceSymbol.empty() ||
      tuneSourceRoot != 0 || tuneMeasureTop != 0 ||
      !tuneMeasurementInputs.empty();
  if (tuneMappingTarget.empty() && mappedOptionsUsed)
    return reportError("target-aware options require --mapping-target");

  if (tunePerfLevel > 1)
    return reportError("unsupported --perf-level=" +
                       llvm::Twine(tunePerfLevel) + "; expected 0 or 1");

  perf::SearchMode mode;
  if (!parseSearchMode(mode, tuneSearch))
    return reportError("unsupported --search=" + llvm::Twine(tuneSearch) +
                       "; expected grid or random (staged is not implemented)");

  DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::micro::MicroDialect>();
  // A search space file exported from LLK keeps the source operations, so the
  // LLK dialect is registered to read that output directly.
  registry.insert<mlir::llk::LLKDialect>();

  MLIRContext context(registry);
  ParserConfig parserConfig(&context);
  auto module =
      mlir::parseSourceFile<ModuleOp>(tuneInput.getValue(), parserConfig);
  if (!module)
    return reportError("cannot parse " + llvm::Twine(tuneInput.getValue()));

  auto loaded = perf::loadSearchSpace(module.get());
  if (!loaded)
    return reportError(llvm::toString(loaded.takeError()));

  perf::SearchSpace space = std::move(*loaded);
  if (!tuneWorkload.empty())
    space.workload = tuneWorkload;

  perf::WorkloadShape shape;
  shape.M = tuneM;
  shape.N = tuneN;
  shape.K = tuneK;
  shape.inputDType = tuneInputDType;
  shape.weightDType = tuneWeightDType;
  shape.accumulatorDType = tuneAccumulatorDType;
  shape.outputDType = tuneOutputDType;

  if (!tuneMappingTarget.empty())
    return runMappedSearch(module.get(), space, shape);

  auto machine = ::mlir::llk::machine::loadMachineModel(tuneMachine);
  if (!machine)
    return reportError(llvm::toString(machine.takeError()));

  perf::TuningSessionOptions options;
  options.generator.mode = mode;
  options.generator.seed = tuneSeed;
  if (tuneMaxCandidates > 0)
    options.generator.maxCandidates = tuneMaxCandidates;
  options.perfLevel = tunePerfLevel;
  options.topK = tuneTopK;
  options.machinePath = tuneMachine;

  auto report =
      perf::runTuningSession(context, space, shape, *machine, options);
  if (!report)
    return reportError(llvm::toString(report.takeError()));

  std::vector<perf::ScheduleRecord> records =
      perf::buildScheduleRecords(*report, space, shape);
  if (llvm::Error error = perf::writeScheduleYamlFile(tuneOutput, records)) {
    return reportError(llvm::toString(std::move(error)));
  }

  llvm::outs() << "workload " << space.workload << " M=" << tuneM
               << " N=" << tuneN << " K=" << tuneK << " on " << machine->target
               << "\n";
  llvm::outs() << "Generated " << report->generated
               << " candidates: " << report->ranked.size() << " ranked, "
               << report->rejected.size() << " rejected\n";
  // What "best" meant, next to the report schema that defines it: a stored
  // ranking is only interpretable with the objective it was ordered by, and a
  // record read later has to know which schema wrote it.
  llvm::outs() << "objective " << report->objective.primaryMetric << " ("
               << perf::stringifyObjectiveDirection(report->objective.direction)
               << ")";
  for (const std::string &name : report->objective.secondaryMetrics)
    llvm::outs() << " then " << name;
  llvm::outs() << ", report schema v" << report->schemaVersion << "\n";
  for (size_t i = 0; i < records.size(); ++i) {
    const perf::ScheduleRecord &record = records[i];
    llvm::outs() << "  [" << i << "] " << record.candidate.id << "  "
                 << record.metrics.predictedCycles << " cycles  "
                 << record.metrics.bottleneck << "\n";
  }
  llvm::outs() << "Wrote " << records.size() << " schedule records to "
               << tuneOutput << "\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(
      argc, argv,
      "LLK schedule tuner\n"
      "\n"
      "With --input=<micro.search_space.mlir> this runs the Micro-IR tuning\n"
      "flow: generate candidates, check legality against a "
      "machine::MachineModel, bind\n"
      "them to concrete micro kernels, rank by predicted cycles, and write "
      "the\n"
      "top-K as schedule YAML.\n"
      "\n"
      "Without --input it keeps the pre-Micro grid search and writes a JSON\n"
      "schedule_db entry to -o.\n");

  if (tuneInput.empty())
    return runLegacyGrid();
  return runMicroSearch();
}
