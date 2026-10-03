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

#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Machine/MachineModelLoader.h"
#include "LLK/Perf/ScheduleRecord.h"
#include "LLK/Perf/TuningSession.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <fstream>
#include <iostream>
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

int runMicroSearch() {
  namespace perf = mlir::llk::perf;

  if (tunePerfLevel > 1)
    return reportError("unsupported --perf-level=" +
                       llvm::Twine(tunePerfLevel) + "; expected 0 or 1");

  perf::SearchMode mode;
  if (!parseSearchMode(mode, tuneSearch))
    return reportError("unsupported --search=" + llvm::Twine(tuneSearch) +
                       "; expected grid or random (staged is not implemented)");

  auto machine = ::mlir::llk::machine::loadMachineModel(tuneMachine);
  if (!machine)
    return reportError(llvm::toString(machine.takeError()));

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
