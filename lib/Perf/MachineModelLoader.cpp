//===- MachineModelLoader.cpp - YAML machine model loader -----------------===//
//
// Walks the YAML document with LLVM's low-level parser rather than
// MappingTraits, because the schema has string-keyed maps (`owners`, `memory`,
// `lanes`) and because the original key has to survive into the diagnostic:
// "unknown key 'tile_shape'" is only useful if it names what the author typed.
//
// The walk is depth first and each value is read before the iterator advances.
// The underlying scanner is single pass, so a node is only valid until its
// parent entry has been skipped; consuming values in place is what makes that
// safe without the buffering `yaml::Input` provides.
//
// This file owns structure and vocabulary. Values are checked by
// verifyMachineModel(), so a parsed model and a hand-built one face the same
// rules -- including the vocabulary checks, which live there.
//
//===----------------------------------------------------------------------===//

#include "LLK/Perf/MachineModelLoader.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"

#include <array>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;
using namespace llvm::yaml;

namespace mlir::llk::perf {
namespace {

/// Source name plus whatever the YAML parser reported, if anything.
struct DiagCapture {
  StringRef source;
  std::string text;
};

void captureDiagnostic(const SMDiagnostic &diagnostic, void *context) {
  auto *capture = static_cast<DiagCapture *>(context);
  StringRef file = diagnostic.getFilename();
  if (file.empty())
    file = capture->source;
  capture->text =
      (file + ":" + Twine(diagnostic.getLineNo()) + ":" +
       Twine(diagnostic.getColumnNo()) + ": error: " + diagnostic.getMessage())
          .str();
}

class Loader {
public:
  explicit Loader(StringRef source) : source_(source) {}

  Expected<MachineModel> load(StringRef yamlText);

private:
  // --- diagnostics --------------------------------------------------------
  bool failed() const { return !error_.empty(); }
  bool fail(const Twine &message);
  bool failAt(const Node *node, const Twine &message);
  Error takeError();

  // --- scalar readers -----------------------------------------------------
  bool readText(Node *node, const Twine &path, std::string &out);
  bool readUInt(Node *node, const Twine &path, uint64_t &out);
  bool readUInt32(Node *node, const Twine &path, uint32_t &out);
  bool readInt64(Node *node, const Twine &path, int64_t &out);
  bool readDouble(Node *node, const Twine &path, double &out);
  bool readStringList(Node *node, const Twine &path,
                      std::vector<std::string> &out);
  bool readTileShapes(Node *node, const Twine &path,
                      std::vector<std::array<int64_t, 3>> &out);
  bool readLanes(Node *node, const Twine &path,
                 std::map<std::string, int64_t> &out);

  // --- structure ----------------------------------------------------------
  using EntryHandler =
      function_ref<bool(StringRef key, Node *value, Node *keyNode)>;
  bool forEachEntry(Node *node, const Twine &path, EntryHandler handler);

  // --- sections -----------------------------------------------------------
  bool parseRoot(Node *root, MachineModel &model);
  bool parseCompute(Node *node, MachineModel &model);
  bool parseOwners(Node *node, MachineModel &model);
  bool parseWorkerThreads(Node *node, MachineModel &model);
  bool parseMatrixEngines(Node *node, MachineModel &model);
  bool parseVectorEngines(Node *node, MachineModel &model);
  bool parseMemory(Node *node, MachineModel &model);
  bool parseMemoryLevel(Node *node, const Twine &path, MemoryLevelModel &level);
  bool parseDma(Node *node, MachineModel &model);
  bool parseCopyPath(Node *node, const Twine &path, CopyPathModel &copyPath);
  bool parseSync(Node *node, MachineModel &model);

  bool requireKey(bool present, const Twine &path, StringRef key);

  StringRef source_;
  StringRef text_;
  std::string error_;
};

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

bool Loader::fail(const Twine &message) {
  if (failed())
    return false;
  error_ = (source_ + ": error: " + message).str();
  return false;
}

bool Loader::failAt(const Node *node, const Twine &message) {
  if (failed())
    return false;

  std::string location = source_.str();
  if (node) {
    SMRange range = node->getSourceRange();
    const char *begin = range.Start.getPointer();
    if (begin && begin >= text_.data() &&
        begin <= text_.data() + text_.size()) {
      unsigned line = 1;
      unsigned column = 1;
      for (const char *cursor = text_.data(); cursor < begin; ++cursor) {
        if (*cursor == '\n') {
          ++line;
          column = 1;
        } else {
          ++column;
        }
      }
      location += (":" + Twine(line) + ":" + Twine(column)).str();
    }
  }

  error_ = (location + ": error: " + message).str();
  return false;
}

Error Loader::takeError() {
  return make_error<StringError>(std::move(error_), inconvertibleErrorCode());
}

//===----------------------------------------------------------------------===//
// Scalar readers
//===----------------------------------------------------------------------===//

bool Loader::readText(Node *node, const Twine &path, std::string &out) {
  if (auto *scalar = dyn_cast_or_null<ScalarNode>(node)) {
    SmallVector<char, 32> storage;
    out = scalar->getValue(storage).str();
    return true;
  }
  // `>-`/`|` descriptions arrive as block scalars, not plain scalars.
  if (auto *block = dyn_cast_or_null<BlockScalarNode>(node)) {
    out = block->getValue().str();
    return true;
  }
  return failAt(node, path + ": expected a scalar value");
}

bool Loader::readUInt(Node *node, const Twine &path, uint64_t &out) {
  std::string text;
  if (!readText(node, path, text))
    return false;
  if (!StringRef(text).getAsInteger(10, out))
    return true;
  return failAt(node,
                path + ": expected a non-negative integer, got '" + text + "'");
}

bool Loader::readUInt32(Node *node, const Twine &path, uint32_t &out) {
  uint64_t value = 0;
  if (!readUInt(node, path, value))
    return false;
  if (value > std::numeric_limits<uint32_t>::max())
    return failAt(node, path + ": value " + Twine(value) +
                            " does not fit in 32 bits");
  out = static_cast<uint32_t>(value);
  return true;
}

bool Loader::readInt64(Node *node, const Twine &path, int64_t &out) {
  std::string text;
  if (!readText(node, path, text))
    return false;
  if (StringRef(text).getAsInteger(10, out))
    return failAt(node, path + ": expected an integer, got '" + text + "'");
  return true;
}

bool Loader::readDouble(Node *node, const Twine &path, double &out) {
  std::string text;
  if (!readText(node, path, text))
    return false;
  // getAsDouble only accepts a decimal point or an exponent, so `16` -- the
  // natural way to write a whole-number bandwidth -- has to go through the
  // integer reader first.
  uint64_t integer = 0;
  if (!StringRef(text).getAsInteger(10, integer)) {
    out = static_cast<double>(integer);
    return true;
  }
  if (!StringRef(text).getAsDouble(out))
    return true;
  return failAt(node, path + ": expected a number, got '" + text + "'");
}

bool Loader::readStringList(Node *node, const Twine &path,
                            std::vector<std::string> &out) {
  auto *sequence = dyn_cast_or_null<SequenceNode>(node);
  if (!sequence)
    return failAt(node, path + ": expected a sequence");

  size_t index = 0;
  for (auto &element : *sequence) {
    std::string text;
    if (!readText(&element, path + "[" + Twine(index++) + "]", text))
      return false;
    out.push_back(std::move(text));
  }
  return true;
}

bool Loader::readTileShapes(Node *node, const Twine &path,
                            std::vector<std::array<int64_t, 3>> &out) {
  auto *sequence = dyn_cast_or_null<SequenceNode>(node);
  if (!sequence)
    return failAt(node, path + ": expected a sequence");

  size_t index = 0;
  for (auto &element : *sequence) {
    std::string shapePath = (path + "[" + Twine(index++) + "]").str();
    auto *shape = dyn_cast_or_null<SequenceNode>(&element);
    if (!shape)
      return failAt(&element, Twine(shapePath + ": expected a sequence of 3 "
                                                "integers"));

    std::array<int64_t, 3> dimensions{0, 0, 0};
    size_t found = 0;
    for (auto &value : *shape) {
      // Count every dimension so an over-long shape is reported with its real
      // width, but only read the first three.
      if (found < dimensions.size() &&
          !readInt64(&value, shapePath + "[" + Twine(found) + "]",
                     dimensions[found]))
        return false;
      ++found;
    }
    if (found != dimensions.size())
      return failAt(&element,
                    Twine(shapePath + " must have exactly 3 dimensions, got ") +
                        Twine(found));
    out.push_back(dimensions);
  }
  return true;
}

bool Loader::readLanes(Node *node, const Twine &path,
                       std::map<std::string, int64_t> &out) {
  return forEachEntry(node, path,
                      [&](StringRef key, Node *value, Node *) -> bool {
                        int64_t lanes = 0;
                        if (!readInt64(value, path + "['" + key + "']", lanes))
                          return false;
                        out[key.str()] = lanes;
                        return true;
                      });
}

//===----------------------------------------------------------------------===//
// Structure
//===----------------------------------------------------------------------===//

bool Loader::forEachEntry(Node *node, const Twine &path, EntryHandler handler) {
  auto *mapping = dyn_cast_or_null<MappingNode>(node);
  if (!mapping)
    return failAt(node, path + ": expected a mapping");

  StringSet<> seen;
  for (auto &entry : *mapping) {
    Node *keyNode = entry.getKey();
    auto *keyScalar = dyn_cast_or_null<ScalarNode>(keyNode);
    if (!keyScalar)
      return failAt(keyNode, path + ": expected a scalar key");

    SmallVector<char, 24> storage;
    std::string key = keyScalar->getValue(storage).str();
    if (!seen.insert(key).second)
      return failAt(keyNode, path + ": duplicate key '" + key + "'");

    if (!handler(key, entry.getValue(), keyNode))
      return false;
  }
  return true;
}

bool Loader::requireKey(bool present, const Twine &path, StringRef key) {
  if (present)
    return true;
  if (path.str().empty())
    return fail("missing required key '" + key + "'");
  return fail(path + ": missing required key '" + key + "'");
}

//===----------------------------------------------------------------------===//
// Top level
//===----------------------------------------------------------------------===//

bool Loader::parseRoot(Node *root, MachineModel &model) {
  bool sawSchema = false;
  bool sawName = false;
  bool sawClock = false;
  bool sawCompute = false;
  bool sawMemory = false;
  bool sawDma = false;
  bool sawSync = false;

  bool ok = forEachEntry(
      root, "document", [&](StringRef key, Node *value, Node *keyNode) -> bool {
        if (key == "schema_version") {
          sawSchema = true;
          return readUInt32(value, "schema_version", model.schemaVersion);
        }
        if (key == "name") {
          sawName = true;
          return readText(value, "name", model.name);
        }
        if (key == "description")
          return readText(value, "description", model.description);
        if (key == "clock_hz") {
          sawClock = true;
          return readUInt(value, "clock_hz", model.clockHz);
        }
        if (key == "compute") {
          sawCompute = true;
          return parseCompute(value, model);
        }
        if (key == "memory") {
          sawMemory = true;
          return parseMemory(value, model);
        }
        if (key == "dma") {
          sawDma = true;
          return parseDma(value, model);
        }
        if (key == "sync") {
          sawSync = true;
          return parseSync(value, model);
        }
        return failAt(keyNode, "document: unknown key '" + key + "'");
      });
  if (!ok)
    return false;

  return requireKey(sawSchema, "", "schema_version") &&
         requireKey(sawName, "", "name") &&
         requireKey(sawClock, "", "clock_hz") &&
         requireKey(sawCompute, "", "compute") &&
         requireKey(sawMemory, "", "memory") && requireKey(sawDma, "", "dma") &&
         requireKey(sawSync, "", "sync");
}

//===----------------------------------------------------------------------===//
// compute
//===----------------------------------------------------------------------===//

bool Loader::parseCompute(Node *node, MachineModel &model) {
  bool sawOwners = false;
  bool sawMatrix = false;
  bool sawVector = false;

  bool ok = forEachEntry(
      node, "compute", [&](StringRef key, Node *value, Node *keyNode) -> bool {
        if (key == "owners") {
          sawOwners = true;
          return parseOwners(value, model);
        }
        if (key == "worker_threads")
          return parseWorkerThreads(value, model);
        if (key == "matrix_engines") {
          sawMatrix = true;
          return parseMatrixEngines(value, model);
        }
        if (key == "vector_engines") {
          sawVector = true;
          return parseVectorEngines(value, model);
        }
        return failAt(keyNode, "compute: unknown key '" + key + "'");
      });
  if (!ok)
    return false;

  // worker_threads is optional: MachineModel defaults it to 1.
  return requireKey(sawOwners, "compute", "owners") &&
         requireKey(sawMatrix, "compute", "matrix_engines") &&
         requireKey(sawVector, "compute", "vector_engines");
}

bool Loader::parseOwners(Node *node, MachineModel &model) {
  bool ok = forEachEntry(
      node, "compute.owners", [&](StringRef name, Node *value, Node *) -> bool {
        std::string path = ("compute.owners['" + name + "']").str();
        OwnerModel owner;
        owner.name = name.str();
        bool sawCount = false;

        bool entryOk = forEachEntry(
            value, path, [&](StringRef key, Node *field, Node *) -> bool {
              if (key == "count") {
                sawCount = true;
                return readUInt32(field, path + ".count", owner.count);
              }
              if (key == "maps_to")
                return readText(field, path + ".maps_to",
                                owner.mapsTo.emplace());
              if (key == "parent")
                return readText(field, path + ".parent",
                                owner.parent.emplace());
              return failAt(field, Twine(path + ": unknown key '") + key + "'");
            });
        if (!entryOk)
          return false;

        if (!requireKey(sawCount, path, "count"))
          return false;
        model.owners.push_back(std::move(owner));
        return true;
      });
  return ok;
}

bool Loader::parseWorkerThreads(Node *node, MachineModel &model) {
  bool sawCount = false;
  bool ok = forEachEntry(
      node, "compute.worker_threads",
      [&](StringRef key, Node *value, Node *keyNode) -> bool {
        if (key == "count") {
          sawCount = true;
          return readUInt32(value, "compute.worker_threads.count",
                            model.workerThreads);
        }
        return failAt(keyNode,
                      "compute.worker_threads: unknown key '" + key + "'");
      });
  if (!ok)
    return false;
  return requireKey(sawCount, "compute.worker_threads", "count");
}

bool Loader::parseMatrixEngines(Node *node, MachineModel &model) {
  auto *sequence = dyn_cast_or_null<SequenceNode>(node);
  if (!sequence)
    return failAt(node, "compute.matrix_engines: expected a sequence");

  size_t index = 0;
  for (auto &element : *sequence) {
    std::string path = ("compute.matrix_engines[" + Twine(index++) + "]").str();
    MatrixEngineModel engine;
    bool sawName = false, sawCount = false, sawShapes = false;
    bool sawInputs = false, sawAccumulators = false, sawLayouts = false;
    bool sawIssue = false, sawLatency = false;

    bool ok = forEachEntry(
        &element, path, [&](StringRef key, Node *value, Node *) -> bool {
          if (key == "name") {
            sawName = true;
            return readText(value, path + ".name", engine.name);
          }
          if (key == "count") {
            sawCount = true;
            return readUInt32(value, path + ".count", engine.count);
          }
          if (key == "owner")
            return readText(value, path + ".owner", engine.owner.emplace());
          if (key == "issue_cycles") {
            sawIssue = true;
            return readUInt(value, path + ".issue_cycles", engine.issueCycles);
          }
          if (key == "latency_cycles") {
            sawLatency = true;
            return readUInt(value, path + ".latency_cycles",
                            engine.latencyCycles);
          }
          if (key == "flops_per_cycle") {
            engine.flopsPerCycle.emplace();
            return readDouble(value, path + ".flops_per_cycle",
                              *engine.flopsPerCycle);
          }
          if (key == "supported_layouts") {
            sawLayouts = true;
            return readStringList(value, path + ".supported_layouts",
                                  engine.supportedLayouts);
          }
          if (key == "input_dtypes") {
            sawInputs = true;
            return readStringList(value, path + ".input_dtypes",
                                  engine.inputDTypes);
          }
          if (key == "accumulator_dtypes") {
            sawAccumulators = true;
            return readStringList(value, path + ".accumulator_dtypes",
                                  engine.accumulatorDTypes);
          }
          if (key == "tile_shapes") {
            sawShapes = true;
            return readTileShapes(value, path + ".tile_shapes",
                                  engine.tileShapes);
          }
          return failAt(value, Twine(path + ": unknown key '") + key + "'");
        });
    if (!ok)
      return false;

    if (!requireKey(sawName, path, "name") ||
        !requireKey(sawCount, path, "count") ||
        !requireKey(sawShapes, path, "tile_shapes") ||
        !requireKey(sawInputs, path, "input_dtypes") ||
        !requireKey(sawAccumulators, path, "accumulator_dtypes") ||
        !requireKey(sawIssue, path, "issue_cycles") ||
        !requireKey(sawLatency, path, "latency_cycles") ||
        !requireKey(sawLayouts, path, "supported_layouts"))
      return false;
    model.matrixEngines.push_back(std::move(engine));
  }
  return true;
}

bool Loader::parseVectorEngines(Node *node, MachineModel &model) {
  auto *sequence = dyn_cast_or_null<SequenceNode>(node);
  if (!sequence)
    return failAt(node, "compute.vector_engines: expected a sequence");

  size_t index = 0;
  for (auto &element : *sequence) {
    std::string path = ("compute.vector_engines[" + Twine(index++) + "]").str();
    VectorEngineModel engine;
    bool sawName = false, sawCount = false, sawLanes = false;
    bool sawLayouts = false, sawIssue = false, sawLatency = false;

    bool ok = forEachEntry(
        &element, path, [&](StringRef key, Node *value, Node *) -> bool {
          if (key == "name") {
            sawName = true;
            return readText(value, path + ".name", engine.name);
          }
          if (key == "count") {
            sawCount = true;
            return readUInt32(value, path + ".count", engine.count);
          }
          if (key == "owner")
            return readText(value, path + ".owner", engine.owner.emplace());
          if (key == "issue_cycles") {
            sawIssue = true;
            return readUInt(value, path + ".issue_cycles", engine.issueCycles);
          }
          if (key == "latency_cycles") {
            sawLatency = true;
            return readUInt(value, path + ".latency_cycles",
                            engine.latencyCycles);
          }
          if (key == "supported_layouts") {
            sawLayouts = true;
            return readStringList(value, path + ".supported_layouts",
                                  engine.supportedLayouts);
          }
          if (key == "lanes") {
            sawLanes = true;
            return readLanes(value, path + ".lanes", engine.lanes);
          }
          return failAt(value, Twine(path + ": unknown key '") + key + "'");
        });
    if (!ok)
      return false;

    if (!requireKey(sawName, path, "name") ||
        !requireKey(sawCount, path, "count") ||
        !requireKey(sawLanes, path, "lanes") ||
        !requireKey(sawIssue, path, "issue_cycles") ||
        !requireKey(sawLatency, path, "latency_cycles") ||
        !requireKey(sawLayouts, path, "supported_layouts"))
      return false;
    model.vectorEngines.push_back(std::move(engine));
  }
  return true;
}

//===----------------------------------------------------------------------===//
// memory
//===----------------------------------------------------------------------===//

bool Loader::parseMemory(Node *node, MachineModel &model) {
  return forEachEntry(node, "memory",
                      [&](StringRef name, Node *value, Node *) -> bool {
                        MemoryLevelModel level;
                        level.name = name.str();
                        std::string path = ("memory['" + name + "']").str();
                        if (!parseMemoryLevel(value, path, level))
                          return false;
                        model.memory[level.name] = std::move(level);
                        return true;
                      });
}

bool Loader::parseMemoryLevel(Node *node, const Twine &path,
                              MemoryLevelModel &level) {
  bool sawCapacity = false, sawBandwidth = false, sawLatency = false;
  bool sawLayouts = false;

  bool ok =
      forEachEntry(node, path, [&](StringRef key, Node *value, Node *) -> bool {
        if (key == "alias")
          return readText(value, path + ".alias", level.alias.emplace());
        if (key == "capacity_bytes") {
          sawCapacity = true;
          return readUInt(value, path + ".capacity_bytes", level.capacityBytes);
        }
        if (key == "bandwidth_bytes_per_cycle") {
          sawBandwidth = true;
          return readDouble(value, path + ".bandwidth_bytes_per_cycle",
                            level.bandwidthBytesPerCycle);
        }
        if (key == "latency_cycles") {
          sawLatency = true;
          return readUInt(value, path + ".latency_cycles", level.latencyCycles);
        }
        if (key == "banks") {
          level.banks.emplace();
          return readUInt32(value, path + ".banks", *level.banks);
        }
        if (key == "supported_layouts") {
          sawLayouts = true;
          return readStringList(value, path + ".supported_layouts",
                                level.supportedLayouts);
        }
        return failAt(value, Twine(path + ": unknown key '") + key + "'");
      });
  if (!ok)
    return false;

  return requireKey(sawCapacity, path, "capacity_bytes") &&
         requireKey(sawBandwidth, path, "bandwidth_bytes_per_cycle") &&
         requireKey(sawLatency, path, "latency_cycles") &&
         requireKey(sawLayouts, path, "supported_layouts");
}

//===----------------------------------------------------------------------===//
// dma and sync
//===----------------------------------------------------------------------===//

bool Loader::parseDma(Node *node, MachineModel &model) {
  bool sawEngines = false, sawOutstanding = false, sawPaths = false;

  bool ok = forEachEntry(
      node, "dma", [&](StringRef key, Node *value, Node *) -> bool {
        if (key == "engines") {
          sawEngines = true;
          return readUInt32(value, "dma.engines", model.dma.engines);
        }
        if (key == "max_outstanding") {
          sawOutstanding = true;
          return readUInt32(value, "dma.max_outstanding",
                            model.dma.maxOutstanding);
        }
        if (key == "setup_cycles")
          return readUInt(value, "dma.setup_cycles", model.dma.setupCycles);
        if (key == "maps_to")
          return readText(value, "dma.maps_to", model.dma.mapsTo.emplace());
        if (key == "owner")
          return readText(value, "dma.owner", model.dma.owner.emplace());
        if (key == "paths") {
          sawPaths = true;
          auto *sequence = dyn_cast_or_null<SequenceNode>(value);
          if (!sequence)
            return failAt(value, "dma.paths: expected a sequence");
          size_t index = 0;
          for (auto &element : *sequence) {
            std::string path = ("dma.paths[" + Twine(index++) + "]").str();
            CopyPathModel copyPath;
            if (!parseCopyPath(&element, path, copyPath))
              return false;
            model.dma.paths.push_back(std::move(copyPath));
          }
          return true;
        }
        return failAt(value, "dma: unknown key '" + key + "'");
      });
  if (!ok)
    return false;

  return requireKey(sawEngines, "dma", "engines") &&
         requireKey(sawOutstanding, "dma", "max_outstanding") &&
         requireKey(sawPaths, "dma", "paths");
}

bool Loader::parseCopyPath(Node *node, const Twine &path,
                           CopyPathModel &copyPath) {
  bool sawSrc = false, sawDst = false;

  bool ok =
      forEachEntry(node, path, [&](StringRef key, Node *value, Node *) -> bool {
        if (key == "src") {
          sawSrc = true;
          return readText(value, path + ".src", copyPath.source);
        }
        if (key == "dst") {
          sawDst = true;
          return readText(value, path + ".dst", copyPath.destination);
        }
        if (key == "bandwidth_bytes_per_cycle") {
          copyPath.bandwidthBytesPerCycle.emplace();
          return readDouble(value, path + ".bandwidth_bytes_per_cycle",
                            *copyPath.bandwidthBytesPerCycle);
        }
        if (key == "latency_cycles") {
          copyPath.latencyCycles.emplace();
          return readUInt(value, path + ".latency_cycles",
                          *copyPath.latencyCycles);
        }
        return failAt(value, Twine(path + ": unknown key '") + key + "'");
      });
  if (!ok)
    return false;

  return requireKey(sawSrc, path, "src") && requireKey(sawDst, path, "dst");
}

bool Loader::parseSync(Node *node, MachineModel &model) {
  bool sawBarrier = false, sawWait = false;

  bool ok = forEachEntry(
      node, "sync", [&](StringRef key, Node *value, Node *) -> bool {
        if (key == "barrier_cycles") {
          sawBarrier = true;
          return readUInt(value, "sync.barrier_cycles",
                          model.sync.barrierCycles);
        }
        if (key == "wait_cycles") {
          sawWait = true;
          return readUInt(value, "sync.wait_cycles", model.sync.waitCycles);
        }
        return failAt(value, "sync: unknown key '" + key + "'");
      });
  if (!ok)
    return false;

  return requireKey(sawBarrier, "sync", "barrier_cycles") &&
         requireKey(sawWait, "sync", "wait_cycles");
}

//===----------------------------------------------------------------------===//
// Entry point
//===----------------------------------------------------------------------===//

Expected<MachineModel> Loader::load(StringRef yamlText) {
  text_ = yamlText;

  SourceMgr sourceMgr;
  DiagCapture capture{source_, {}};
  sourceMgr.setDiagHandler(captureDiagnostic, &capture);
  yaml::Stream stream(yamlText, sourceMgr, /*ShowColors=*/false);

  auto document = stream.begin();
  if (document == stream.end() || !capture.text.empty()) {
    if (!capture.text.empty())
      error_ = capture.text;
    else
      error_ = (source_ + ": error: expected a YAML document").str();
    return takeError();
  }

  Node *root = document->getRoot();
  if (!root || !capture.text.empty()) {
    if (!capture.text.empty())
      error_ = capture.text;
    else
      error_ = (source_ + ": error: expected a YAML document").str();
    return takeError();
  }

  MachineModel model;
  if (!parseRoot(root, model)) {
    // A syntax error is what actually stopped the walk; the type error the
    // walker saw (a sequence where a scalar was expected) is a symptom.
    if (!capture.text.empty())
      error_ = capture.text;
    return takeError();
  }

  // The scanner is lazy: a syntax error anywhere in the document can surface
  // during the walk rather than at the first token.
  if (!capture.text.empty()) {
    error_ = capture.text;
    return takeError();
  }

  if (Error error = verifyMachineModel(model)) {
    std::string message = toString(std::move(error));
    error_ = (source_ + ": error: " + message).str();
    return takeError();
  }

  return model;
}

} // namespace

Expected<MachineModel> parseMachineModel(StringRef yamlText,
                                         StringRef sourceName) {
  return Loader(sourceName).load(yamlText);
}

Expected<MachineModel> loadMachineModel(StringRef path) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> buffer = MemoryBuffer::getFile(path);
  if (!buffer)
    return make_error<StringError>(path +
                                       ": error: cannot open machine model: " +
                                       buffer.getError().message(),
                                   inconvertibleErrorCode());
  return parseMachineModel((*buffer)->getBuffer(), path);
}

} // namespace mlir::llk::perf
