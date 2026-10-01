//===- MachineModel.cpp - Typed target model for Micro-IR perf ------------===//
//
// Lookup helpers plus verifyMachineModel(), the single place that decides
// whether a machine model is usable. The YAML loader calls it, and so can a
// caller that builds a model in code, so both are held to the same rules.
//
// Diagnostics report exactly one violation, the first in schema order, so an
// invalid file always produces the same message. Message paths mirror the
// YAML schema (`compute.matrix_engines[0].count`) rather than the C++ member
// names, because the person reading them is looking at a .yaml file.
//
//===----------------------------------------------------------------------===//

#include "LLK/Perf/MachineModel.h"

#include "LLK/Dialect/Micro/MicroEnums.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

#include <string>

using namespace llvm;

namespace mlir::llk::perf {
namespace {

Error invalid(const Twine &message) {
  return make_error<StringError>(message.str(), inconvertibleErrorCode());
}

/// Tracks every named resource in the model so that one name means one thing.
class NameTable {
public:
  Error claim(StringRef name, const Twine &path) {
    auto [it, inserted] = names_.try_emplace(name, path.str());
    if (!inserted)
      return invalid("duplicate resource name '" + name +
                     "' (already declared at " + it->second + ")");
    return Error::success();
  }

private:
  StringMap<std::string> names_;
};

/// Checks an optional owner reference against the `micro` owner vocabulary and
/// the owner scopes the model actually declares.
Error verifyOwnerRef(const std::optional<std::string> &ownerRef,
                     const StringMap<const OwnerModel *> &declared,
                     const Twine &path) {
  if (!ownerRef)
    return Error::success();
  if (!micro::symbolizeOwner(*ownerRef))
    return invalid(path + ": unknown owner '" + *ownerRef + "'");
  if (!declared.count(*ownerRef))
    return invalid(path + ": owner '" + *ownerRef +
                   "' is not declared in compute.owners");
  return Error::success();
}

Error verifyDTypeList(ArrayRef<std::string> dtypes, const Twine &path) {
  if (dtypes.empty())
    return invalid(path + " must not be empty");
  for (size_t i = 0; i < dtypes.size(); ++i)
    if (!micro::symbolizeDType(dtypes[i]))
      return invalid(path + "[" + Twine(i) + "]: unknown dtype '" + dtypes[i] +
                     "'");
  return Error::success();
}

Error verifyLayoutList(ArrayRef<std::string> layouts, const Twine &path) {
  if (layouts.empty())
    return invalid(path + " must not be empty");
  for (size_t i = 0; i < layouts.size(); ++i)
    if (!micro::symbolizeLayoutKind(layouts[i]))
      return invalid(path + "[" + Twine(i) + "]: unknown layout '" +
                     layouts[i] + "'");
  return Error::success();
}

/// Verifies the parts shared by both engine kinds: identity, uniqueness, and
/// the cycle costs every engine carries.
Error verifyEngineHeader(StringRef name, uint32_t count, uint64_t issueCycles,
                         uint64_t latencyCycles, NameTable &names,
                         const Twine &path) {
  if (name.empty())
    return invalid(path + ".name must not be empty");
  if (Error err = names.claim(name, path))
    return err;
  if (count == 0)
    return invalid(path + ".count must be positive");
  if (issueCycles == 0)
    return invalid(path + ".issue_cycles must be positive");
  if (latencyCycles == 0)
    return invalid(path + ".latency_cycles must be positive");
  return Error::success();
}

} // namespace

//===----------------------------------------------------------------------===//
// Lookup helpers
//===----------------------------------------------------------------------===//

const OwnerModel *MachineModel::findOwner(StringRef name) const {
  for (const OwnerModel &owner : owners)
    if (owner.name == name)
      return &owner;
  return nullptr;
}

const MatrixEngineModel *MachineModel::findMatrixEngine(StringRef name) const {
  for (const MatrixEngineModel &engine : matrixEngines)
    if (engine.name == name)
      return &engine;
  return nullptr;
}

const VectorEngineModel *MachineModel::findVectorEngine(StringRef name) const {
  for (const VectorEngineModel &engine : vectorEngines)
    if (engine.name == name)
      return &engine;
  return nullptr;
}

const MemoryLevelModel *MachineModel::findMemory(StringRef name) const {
  auto it = memory.find(name);
  return it == memory.end() ? nullptr : &it->second;
}

const CopyPathModel *MachineModel::findCopyPath(StringRef source,
                                                StringRef destination) const {
  for (const CopyPathModel &path : dma.paths)
    if (path.source == source && path.destination == destination)
      return &path;
  return nullptr;
}

uint32_t MachineModel::getOwnerCount(StringRef name) const {
  const OwnerModel *owner = findOwner(name);
  return owner ? owner->count : 0;
}

//===----------------------------------------------------------------------===//
// Verification
//===----------------------------------------------------------------------===//

Error verifyMachineModel(const MachineModel &model) {
  if (model.schemaVersion != kSupportedSchemaVersion)
    return invalid("unsupported schema_version " + Twine(model.schemaVersion) +
                   " (expected " + Twine(kSupportedSchemaVersion) + ")");
  if (model.name.empty())
    return invalid("name must not be empty");
  if (model.clockHz == 0)
    return invalid("clock_hz must be positive");
  if (model.workerThreads == 0)
    return invalid("compute.worker_threads.count must be positive");

  NameTable names;

  // --- Owners -------------------------------------------------------------
  if (model.owners.empty())
    return invalid("compute.owners must declare at least one owner");

  StringMap<const OwnerModel *> declaredOwners;
  for (const OwnerModel &owner : model.owners) {
    if (!micro::symbolizeOwner(owner.name))
      return invalid("compute.owners: unknown owner '" + owner.name + "'");
    if (owner.count == 0)
      return invalid("compute.owners['" + owner.name +
                     "'].count must be positive");
    if (declaredOwners.count(owner.name))
      return invalid("compute.owners: duplicate owner '" + owner.name + "'");
    declaredOwners[owner.name] = &owner;
  }

  for (const OwnerModel &owner : model.owners) {
    if (!owner.parent)
      continue;
    std::string path = "compute.owners['" + owner.name + "']";
    if (!declaredOwners.count(*owner.parent))
      return invalid(path + ".parent: unknown parent '" + *owner.parent + "'");
    // Walk to the root: revisiting a scope means the hierarchy loops.
    SmallPtrSet<const OwnerModel *, 16> seen;
    for (const OwnerModel *scope = &owner; scope;
         scope = scope->parent ? declaredOwners.lookup(*scope->parent)
                               : nullptr) {
      if (!seen.insert(scope).second)
        return invalid(path + ": owner hierarchy contains a cycle");
    }
  }

  // --- Compute engines ----------------------------------------------------
  for (size_t i = 0; i < model.matrixEngines.size(); ++i) {
    const MatrixEngineModel &engine = model.matrixEngines[i];
    std::string path = ("compute.matrix_engines[" + Twine(i) + "]").str();
    if (Error err =
            verifyEngineHeader(engine.name, engine.count, engine.issueCycles,
                               engine.latencyCycles, names, path))
      return err;
    if (engine.tileShapes.empty())
      return invalid(path + ".tile_shapes must not be empty");
    for (size_t j = 0; j < engine.tileShapes.size(); ++j) {
      std::string shapePath = (path + ".tile_shapes[" + Twine(j) + "]").str();
      const std::array<int64_t, 3> &shape = engine.tileShapes[j];
      for (size_t d = 0; d < shape.size(); ++d)
        if (shape[d] <= 0)
          return invalid(shapePath + "[" + Twine(d) +
                         "]: dimensions must be positive");
    }
    if (Error err = verifyDTypeList(engine.inputDTypes, path + ".input_dtypes"))
      return err;
    if (Error err = verifyDTypeList(engine.accumulatorDTypes,
                                    path + ".accumulator_dtypes"))
      return err;
    if (engine.flopsPerCycle && *engine.flopsPerCycle <= 0)
      return invalid(path + ".flops_per_cycle must be positive");
    if (Error err = verifyLayoutList(engine.supportedLayouts,
                                     path + ".supported_layouts"))
      return err;
    if (Error err =
            verifyOwnerRef(engine.owner, declaredOwners, path + ".owner"))
      return err;
  }

  for (size_t i = 0; i < model.vectorEngines.size(); ++i) {
    const VectorEngineModel &engine = model.vectorEngines[i];
    std::string path = ("compute.vector_engines[" + Twine(i) + "]").str();
    if (Error err =
            verifyEngineHeader(engine.name, engine.count, engine.issueCycles,
                               engine.latencyCycles, names, path))
      return err;
    if (engine.lanes.empty())
      return invalid(path + ".lanes must not be empty");
    for (const auto &[dtype, laneCount] : engine.lanes) {
      if (!micro::symbolizeDType(dtype))
        return invalid(path + ".lanes: unknown dtype '" + dtype + "'");
      if (laneCount <= 0)
        return invalid(path + ".lanes['" + dtype + "'] must be positive, got " +
                       Twine(laneCount));
    }
    if (Error err = verifyLayoutList(engine.supportedLayouts,
                                     path + ".supported_layouts"))
      return err;
    if (Error err =
            verifyOwnerRef(engine.owner, declaredOwners, path + ".owner"))
      return err;
  }

  if (model.matrixEngines.empty() && model.vectorEngines.empty())
    return invalid("compute has no compute engines: declare at least one "
                   "matrix_engines or vector_engines entry");

  // --- Memory hierarchy ---------------------------------------------------
  if (model.memory.empty())
    return invalid("memory must declare at least one memory level");

  StringMap<std::string> aliases;
  for (const auto &entry : model.memory) {
    StringRef levelName = entry.first();
    const MemoryLevelModel &level = entry.second;
    if (!micro::symbolizeMemorySpace(levelName))
      return invalid("memory: unknown memory space '" + levelName + "'");
    std::string path = ("memory['" + levelName + "']").str();
    if (Error err = names.claim(levelName, path))
      return err;
    if (level.capacityBytes == 0)
      return invalid(path + ".capacity_bytes must be positive");
    if (level.bandwidthBytesPerCycle <= 0)
      return invalid(path + ".bandwidth_bytes_per_cycle must be positive");
    if (level.latencyCycles == 0)
      return invalid(path + ".latency_cycles must be positive");
    if (level.banks && *level.banks == 0)
      return invalid(path + ".banks must be positive");
    if (Error err = verifyLayoutList(level.supportedLayouts,
                                     path + ".supported_layouts"))
      return err;
    if (level.alias) {
      auto [it, inserted] = aliases.try_emplace(*level.alias, levelName.str());
      if (!inserted)
        return invalid(path + ".alias '" + *level.alias +
                       "' is already used by memory level '" + it->second +
                       "'");
    }
  }

  for (const auto &entry : aliases) {
    StringRef alias = entry.first();
    const std::string &levelName = entry.second;
    if (model.memory.count(alias))
      return invalid("memory['" + levelName + "'].alias '" + alias +
                     "' conflicts with memory level '" + alias + "'");
  }

  // --- DMA ----------------------------------------------------------------
  if (model.dma.engines == 0)
    return invalid("dma.engines must be positive");
  if (model.dma.maxOutstanding == 0)
    return invalid("dma.max_outstanding must be positive");
  if (Error err = verifyOwnerRef(model.dma.owner, declaredOwners, "dma.owner"))
    return err;
  if (model.dma.paths.empty())
    return invalid("dma.paths must declare at least one copy path");

  StringSet<> seenPaths;
  for (size_t i = 0; i < model.dma.paths.size(); ++i) {
    const CopyPathModel &path = model.dma.paths[i];
    std::string location = ("dma.paths[" + Twine(i) + "]").str();
    if (!micro::symbolizeMemorySpace(path.source) ||
        !model.memory.count(path.source))
      return invalid(location + ".src: unknown memory space '" + path.source +
                     "'");
    if (!micro::symbolizeMemorySpace(path.destination) ||
        !model.memory.count(path.destination))
      return invalid(location + ".dst: unknown memory space '" +
                     path.destination + "'");
    if (path.source == path.destination)
      return invalid(location + ": source and destination must differ, both '" +
                     path.source + "'");
    if (path.bandwidthBytesPerCycle && *path.bandwidthBytesPerCycle <= 0)
      return invalid(location + ".bandwidth_bytes_per_cycle must be positive");
    if (path.latencyCycles && *path.latencyCycles == 0)
      return invalid(location + ".latency_cycles must be positive");
    if (!seenPaths.insert(path.source + " -> " + path.destination).second)
      return invalid(location + ": duplicate copy path '" + path.source +
                     " -> " + path.destination + "'");
  }

  return Error::success();
}

} // namespace mlir::llk::perf
