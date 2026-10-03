//===- MachineModelLoader.cpp - Load a v2 machine model from YAML ---------===//
//
// Walks the document with LLVM's low-level YAML parser rather than
// MappingTraits, for the same reasons the v1 loader does: the original key has
// to survive into the diagnostic ("unknown key 'warp_size'" is only useful if
// it names what the author typed), and the scanner is single pass, so values
// are read in place before the iterator advances.
//
// This file owns structure and vocabulary. Semantic rules -- unique ids,
// acyclic containment, known kinds, resolvable references -- live in
// verifyMachineModel(), so a parsed model and a hand-built one face the same
// checks.
//
//===----------------------------------------------------------------------===//

#include "LLK/Machine/MachineModelLoader.h"

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/YAMLParser.h"

#include <cctype>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;
using namespace llvm::yaml;

namespace mlir::llk::machine {
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

/// Accepts `llk.machine.v2` and `llk.machine.v2.<minor>`, rejecting any other
/// major or a malformed minor. Returns the major and minor on success; a bare
/// version is minor 0.
bool parseSchema(StringRef text, uint32_t &major, uint32_t &minor) {
  constexpr StringLiteral prefix = "llk.machine.v";
  if (!text.consume_front(prefix))
    return false;
  size_t digits = 0;
  while (digits < text.size() &&
         std::isdigit(static_cast<unsigned char>(text[digits])))
    ++digits;
  if (digits == 0)
    return false;
  if (text.substr(0, digits).getAsInteger(10, major))
    return false;
  StringRef rest = text.substr(digits);
  if (rest.empty()) {
    minor = 0;
    return true;
  }
  if (!rest.consume_front(".") || rest.empty())
    return false;
  return !rest.getAsInteger(10, minor);
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
  bool readIntList(Node *node, const Twine &path, std::vector<int64_t> &out);
  bool readLanes(Node *node, const Twine &path,
                 std::map<std::string, int64_t> &out);
  bool readShapes(Node *node, const Twine &path,
                  std::vector<std::vector<int64_t>> &out);

  // --- structure ----------------------------------------------------------
  using EntryHandler =
      function_ref<bool(StringRef key, Node *value, Node *keyNode)>;
  using ElementHandler = function_ref<bool(Node *element, size_t index)>;
  bool forEachEntry(Node *node, const Twine &path, EntryHandler handler);
  bool forEachElement(Node *node, const Twine &path, ElementHandler handler);

  // --- sections -----------------------------------------------------------
  bool parseRoot(Node *root, MachineModel &model);
  bool parseSync(Node *node, MachineModel &model);
  bool parseExecutor(Node *node, size_t index, ExecutorNode &out);
  bool parseMemory(Node *node, size_t index, MemoryNode &out);
  bool parseCompute(Node *node, size_t index, ComputeNode &out);
  bool parseTransferEngine(Node *node, size_t index, TransferEngineNode &out);
  bool parseLink(Node *node, size_t index, LinkEdge &out);

  bool requireKey(bool present, const Twine &path, StringRef key);

  /// Records an unrecognized key. The verdict is deferred: whether the key is
  /// a typo or a forward-compatible addition depends on this file's schema
  /// minor, which may be declared after the key, and a YAML mapping is a
  /// single-pass stream, so resolveUnknownKeys() decides once the walk ends.
  bool unknownKey(const Node *node, const Twine &path, StringRef key);
  bool resolveUnknownKeys();

  StringRef source_;
  StringRef text_;
  std::string error_;

  /// True once the root `schema` entry declares a minor above the one this
  /// build understands (design §11.6).
  bool tolerantMinor_ = false;
  /// The first unrecognized key seen, held until the schema minor is known.
  const Node *firstUnknownNode_ = nullptr;
  std::string firstUnknownPath_;
  std::string firstUnknownKey_;
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
  // getAsDouble only accepts a decimal point or an exponent, so `32` -- the
  // natural way to write a whole-number bandwidth -- goes through the integer
  // reader first.
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
  return forEachElement(node, path, [&](Node *element, size_t index) -> bool {
    std::string text;
    if (!readText(element, path + "[" + Twine(index) + "]", text))
      return false;
    out.push_back(std::move(text));
    return true;
  });
}

bool Loader::readIntList(Node *node, const Twine &path,
                         std::vector<int64_t> &out) {
  return forEachElement(node, path, [&](Node *element, size_t index) -> bool {
    int64_t value = 0;
    if (!readInt64(element, path + "[" + Twine(index) + "]", value))
      return false;
    out.push_back(value);
    return true;
  });
}

bool Loader::readShapes(Node *node, const Twine &path,
                        std::vector<std::vector<int64_t>> &out) {
  return forEachElement(node, path, [&](Node *element, size_t index) -> bool {
    std::vector<int64_t> shape;
    if (!readIntList(element, path + "[" + Twine(index) + "]", shape))
      return false;
    out.push_back(std::move(shape));
    return true;
  });
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

bool Loader::forEachElement(Node *node, const Twine &path,
                            ElementHandler handler) {
  auto *sequence = dyn_cast_or_null<SequenceNode>(node);
  if (!sequence)
    return failAt(node, path + ": expected a sequence");

  size_t index = 0;
  for (auto &element : *sequence)
    if (!handler(&element, index++))
      return false;
  return true;
}

bool Loader::requireKey(bool present, const Twine &path, StringRef key) {
  if (present)
    return true;
  return fail(path + ": missing required key '" + key + "'");
}

bool Loader::unknownKey(const Node *node, const Twine &path, StringRef key) {
  // A file at a newer minor may carry keys added after this build; §11.6
  // requires those additions to be optional or defaulted, so they are ignored.
  if (tolerantMinor_)
    return true;
  if (!firstUnknownNode_) {
    firstUnknownPath_ = path.str();
    firstUnknownKey_ = key.str();
    firstUnknownNode_ = node;
  }
  return true;
}

bool Loader::resolveUnknownKeys() {
  if (tolerantMinor_ || !firstUnknownNode_)
    return true;
  return failAt(firstUnknownNode_,
                firstUnknownPath_ + ": unknown key '" + firstUnknownKey_ + "'");
}

//===----------------------------------------------------------------------===//
// Sections
//===----------------------------------------------------------------------===//

bool Loader::parseExecutor(Node *node, size_t index, ExecutorNode &out) {
  std::string path = "executors[" + std::to_string(index) + "]";
  bool sawId = false;
  bool sawKind = false;
  return forEachEntry(node, path,
                      [&](StringRef key, Node *value, Node *keyNode) -> bool {
                        if (key == "id") {
                          sawId = true;
                          return readText(value, path + ".id", out.id);
                        }
                        if (key == "kind") {
                          sawKind = true;
                          return readText(value, path + ".kind", out.kind);
                        }
                        if (key == "parent") {
                          std::string parent;
                          if (!readText(value, path + ".parent", parent))
                            return false;
                          out.parent = std::move(parent);
                          return true;
                        }
                        if (key == "coordinates")
                          return readIntList(value, path + ".coordinates",
                                             out.coordinates);
                        if (key == "concurrency")
                          return readUInt32(value, path + ".concurrency",
                                            out.concurrency);
                        if (key == "refines")
                          return readStringList(value, path + ".refines",
                                                out.refines);
                        return unknownKey(keyNode, path, key);
                      }) &&
         requireKey(sawId, path, "id") && requireKey(sawKind, path, "kind");
}

bool Loader::parseMemory(Node *node, size_t index, MemoryNode &out) {
  std::string path = "memories[" + std::to_string(index) + "]";
  bool sawId = false;
  bool sawKind = false;
  bool sawVisibleFrom = false;
  bool sawCapacity = false;
  bool parsed = forEachEntry(
      node, path, [&](StringRef key, Node *value, Node *keyNode) -> bool {
        if (key == "id") {
          sawId = true;
          return readText(value, path + ".id", out.id);
        }
        if (key == "kind") {
          sawKind = true;
          return readText(value, path + ".kind", out.kind);
        }
        if (key == "visible_from") {
          sawVisibleFrom = true;
          return readText(value, path + ".visible_from", out.visibleFrom);
        }
        if (key == "capacity_bytes") {
          sawCapacity = true;
          return readUInt(value, path + ".capacity_bytes", out.capacityBytes);
        }
        if (key == "alignment_bytes")
          return readUInt(value, path + ".alignment_bytes", out.alignmentBytes);
        if (key == "supported_layouts")
          return readStringList(value, path + ".supported_layouts",
                                out.supportedLayouts);
        if (key == "bandwidth_bytes_per_cycle")
          return readDouble(value, path + ".bandwidth_bytes_per_cycle",
                            out.bandwidthBytesPerCycle);
        if (key == "latency_cycles")
          return readUInt(value, path + ".latency_cycles", out.latencyCycles);
        if (key == "banks") {
          uint32_t banks = 0;
          if (!readUInt32(value, path + ".banks", banks))
            return false;
          out.banks = banks;
          return true;
        }
        return unknownKey(keyNode, path, key);
      });
  return parsed && requireKey(sawId, path, "id") &&
         requireKey(sawKind, path, "kind") &&
         requireKey(sawVisibleFrom, path, "visible_from") &&
         requireKey(sawCapacity, path, "capacity_bytes");
}

bool Loader::parseCompute(Node *node, size_t index, ComputeNode &out) {
  std::string path = "compute[" + std::to_string(index) + "]";
  bool sawId = false;
  bool sawKind = false;
  bool sawAttachedTo = false;
  bool sawShapes = false;
  bool parsed = forEachEntry(
      node, path, [&](StringRef key, Node *value, Node *keyNode) -> bool {
        if (key == "id") {
          sawId = true;
          return readText(value, path + ".id", out.id);
        }
        if (key == "kind") {
          sawKind = true;
          return readText(value, path + ".kind", out.kind);
        }
        if (key == "attached_to") {
          sawAttachedTo = true;
          return readText(value, path + ".attached_to", out.attachedTo);
        }
        if (key == "element_types")
          return readStringList(value, path + ".element_types",
                                out.elementTypes);
        if (key == "accumulator_dtypes")
          return readStringList(value, path + ".accumulator_dtypes",
                                out.accumulatorDTypes);
        if (key == "supported_layouts")
          return readStringList(value, path + ".supported_layouts",
                                out.supportedLayouts);
        if (key == "shapes") {
          sawShapes = true;
          return readShapes(value, path + ".shapes", out.shapes);
        }
        if (key == "lanes")
          return readLanes(value, path + ".lanes", out.lanes);
        if (key == "issue_cycles")
          return readUInt(value, path + ".issue_cycles", out.issueCycles);
        if (key == "latency_cycles")
          return readUInt(value, path + ".latency_cycles", out.latencyCycles);
        if (key == "throughput_per_cycle") {
          double throughput = 0.0;
          if (!readDouble(value, path + ".throughput_per_cycle", throughput))
            return false;
          out.throughputPerCycle = throughput;
          return true;
        }
        if (key == "concurrency")
          return readUInt32(value, path + ".concurrency", out.concurrency);
        return unknownKey(keyNode, path, key);
      });
  return parsed && requireKey(sawId, path, "id") &&
         requireKey(sawKind, path, "kind") &&
         requireKey(sawAttachedTo, path, "attached_to") &&
         requireKey(sawShapes, path, "shapes");
}

bool Loader::parseTransferEngine(Node *node, size_t index,
                                 TransferEngineNode &out) {
  std::string path = "transfer_engines[" + std::to_string(index) + "]";
  bool sawId = false;
  bool sawKind = false;
  bool sawAttachedTo = false;
  bool parsed = forEachEntry(
      node, path, [&](StringRef key, Node *value, Node *keyNode) -> bool {
        if (key == "id") {
          sawId = true;
          return readText(value, path + ".id", out.id);
        }
        if (key == "kind") {
          sawKind = true;
          return readText(value, path + ".kind", out.kind);
        }
        if (key == "attached_to") {
          sawAttachedTo = true;
          return readText(value, path + ".attached_to", out.attachedTo);
        }
        if (key == "count")
          return readUInt32(value, path + ".count", out.count);
        if (key == "max_outstanding")
          return readUInt32(value, path + ".max_outstanding",
                            out.maxOutstanding);
        if (key == "setup_cycles")
          return readUInt(value, path + ".setup_cycles", out.setupCycles);
        return unknownKey(keyNode, path, key);
      });
  return parsed && requireKey(sawId, path, "id") &&
         requireKey(sawKind, path, "kind") &&
         requireKey(sawAttachedTo, path, "attached_to");
}

bool Loader::parseLink(Node *node, size_t index, LinkEdge &out) {
  std::string path = "links[" + std::to_string(index) + "]";
  bool sawId = false;
  bool sawSource = false;
  bool sawDestination = false;
  bool sawBandwidth = false;
  bool parsed = forEachEntry(
      node, path, [&](StringRef key, Node *value, Node *keyNode) -> bool {
        if (key == "id") {
          sawId = true;
          return readText(value, path + ".id", out.id);
        }
        if (key == "source") {
          sawSource = true;
          return readText(value, path + ".source", out.source);
        }
        if (key == "destination") {
          sawDestination = true;
          return readText(value, path + ".destination", out.destination);
        }
        if (key == "bandwidth_bytes_per_cycle") {
          sawBandwidth = true;
          return readDouble(value, path + ".bandwidth_bytes_per_cycle",
                            out.bandwidthBytesPerCycle);
        }
        if (key == "latency_cycles")
          return readUInt(value, path + ".latency_cycles", out.latencyCycles);
        if (key == "transaction_bytes")
          return readUInt(value, path + ".transaction_bytes",
                          out.transactionBytes);
        if (key == "transfer_engines")
          return readStringList(value, path + ".transfer_engines",
                                out.transferEngines);
        if (key == "concurrency")
          return readUInt32(value, path + ".concurrency", out.concurrency);
        return unknownKey(keyNode, path, key);
      });
  return parsed && requireKey(sawId, path, "id") &&
         requireKey(sawSource, path, "source") &&
         requireKey(sawDestination, path, "destination") &&
         requireKey(sawBandwidth, path, "bandwidth_bytes_per_cycle");
}

bool Loader::parseSync(Node *node, MachineModel &model) {
  return forEachEntry(
      node, "sync", [&](StringRef key, Node *value, Node *keyNode) -> bool {
        if (key == "barrier_cycles")
          return readUInt(value, "sync.barrier_cycles",
                          model.sync.barrierCycles);
        if (key == "wait_cycles")
          return readUInt(value, "sync.wait_cycles", model.sync.waitCycles);
        return unknownKey(keyNode, "sync", key);
      });
}

bool Loader::parseRoot(Node *root, MachineModel &model) {
  bool sawSchema = false;
  bool sawTarget = false;
  bool ok = forEachEntry(
      root, "document", [&](StringRef key, Node *value, Node *keyNode) -> bool {
        if (key == "schema") {
          sawSchema = true;
          std::string schema;
          if (!readText(value, "schema", schema))
            return false;
          uint32_t major = 0;
          uint32_t minor = 0;
          if (!parseSchema(schema, major, minor))
            return failAt(value, "schema: expected 'llk.machine.v2', got '" +
                                     schema + "'");
          model.schemaMajor = major;
          model.schemaMinor = minor;
          // A newer minor may add optional keys this build does not know
          // (design §11.6); at or below the current minor any unknown key is
          // an error.
          tolerantMinor_ = minor > kSupportedSchemaMinor;
          return true;
        }
        if (key == "target") {
          sawTarget = true;
          return readText(value, "target", model.target);
        }
        if (key == "description")
          return readText(value, "description", model.description);
        if (key == "clock_hz") {
          uint64_t clock = 0;
          if (!readUInt(value, "clock_hz", clock))
            return false;
          model.clockHz = clock;
          return true;
        }
        if (key == "worker_threads")
          return readUInt32(value, "worker_threads", model.workerThreads);
        if (key == "sync")
          return parseSync(value, model);
        if (key == "executors")
          return forEachElement(
              value, "executors", [&](Node *element, size_t index) -> bool {
                ExecutorNode executor;
                if (!parseExecutor(element, index, executor))
                  return false;
                model.executors.push_back(std::move(executor));
                return true;
              });
        if (key == "memories")
          return forEachElement(value, "memories",
                                [&](Node *element, size_t index) -> bool {
                                  MemoryNode memory;
                                  if (!parseMemory(element, index, memory))
                                    return false;
                                  model.memories.push_back(std::move(memory));
                                  return true;
                                });
        if (key == "compute")
          return forEachElement(value, "compute",
                                [&](Node *element, size_t index) -> bool {
                                  ComputeNode compute;
                                  if (!parseCompute(element, index, compute))
                                    return false;
                                  model.computes.push_back(std::move(compute));
                                  return true;
                                });
        if (key == "transfer_engines")
          return forEachElement(
              value, "transfer_engines",
              [&](Node *element, size_t index) -> bool {
                TransferEngineNode engine;
                if (!parseTransferEngine(element, index, engine))
                  return false;
                model.transferEngines.push_back(std::move(engine));
                return true;
              });
        if (key == "links")
          return forEachElement(value, "links",
                                [&](Node *element, size_t index) -> bool {
                                  LinkEdge link;
                                  if (!parseLink(element, index, link))
                                    return false;
                                  model.links.push_back(std::move(link));
                                  return true;
                                });
        return unknownKey(keyNode, "document", key);
      });
  return ok && resolveUnknownKeys() && requireKey(sawSchema, "", "schema") &&
         requireKey(sawTarget, "", "target");
}

//===----------------------------------------------------------------------===//
// Top level
//===----------------------------------------------------------------------===//

Expected<MachineModel> Loader::load(StringRef yamlText) {
  text_ = yamlText;

  SourceMgr sourceMgr;
  DiagCapture capture{source_, {}};
  sourceMgr.setDiagHandler(captureDiagnostic, &capture);
  yaml::Stream stream(yamlText, sourceMgr, /*ShowColors=*/false);

  auto document = stream.begin();
  if (document == stream.end() || !capture.text.empty()) {
    error_ = capture.text.empty()
                 ? (source_ + ": error: expected a YAML document").str()
                 : capture.text;
    return takeError();
  }

  Node *root = document->getRoot();
  if (!root || !capture.text.empty()) {
    error_ = capture.text.empty()
                 ? (source_ + ": error: expected a YAML document").str()
                 : capture.text;
    return takeError();
  }

  MachineModel model;
  if (!parseRoot(root, model)) {
    // A syntax error is what actually stopped the walk; the type error the
    // walker saw is a symptom of it.
    if (!capture.text.empty())
      error_ = capture.text;
    return takeError();
  }

  // The scanner is lazy: a syntax error later in the document can surface
  // during the walk. Advancing the iterator finishes the document and sweeps
  // up anything left after the last key the walker read.
  ++document;
  if (!capture.text.empty()) {
    error_ = capture.text;
    return takeError();
  }
  if (document != stream.end()) {
    error_ = (source_ +
              ": error: expected a single YAML document, found a second one")
                 .str();
    return takeError();
  }

  if (Error error = verifyMachineModel(model)) {
    error_ = (source_ + ": error: " + toString(std::move(error))).str();
    return takeError();
  }

  model.contentHash = computeContentHash(model);
  return model;
}

} // namespace

Expected<MachineModel> parseMachineModel(StringRef yamlText,
                                         StringRef sourceName) {
  Loader loader(sourceName);
  return loader.load(yamlText);
}

Expected<MachineModel> loadMachineModel(StringRef path) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> buffer = MemoryBuffer::getFile(path);
  if (!buffer)
    return createStringError(buffer.getError(),
                             "cannot read machine model '" + path + "'");
  return parseMachineModel(buffer.get()->getBuffer(), path);
}

} // namespace mlir::llk::machine
