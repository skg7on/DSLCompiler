//===- MachineModel.h - Typed target model for Micro-IR perf --------------===//
//
// Part of the M10 tile-aware MachineModel (issue #45).
//
// A MachineModel answers two questions for a tile dataflow, tile mapping, and
// tile schedule: is it legal on this target, and what does it cost? It is
// authored as YAML (see MachineModelLoader.h), validated into these typed
// structs, and consumed by the L0/L1 performance models.
//
// Names are not free-form strings: owner names, dtype names, layout names, and
// memory space names are the `micro` dialect vocabulary declared in
// LLK/Dialect/Micro/MicroEnums.h, so a machine model and a concrete
// `micro.kernel` always speak about the same things. verifyMachineModel()
// enforces that.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_PERF_MACHINEMODEL_H
#define LLK_PERF_MACHINEMODEL_H

#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"

#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace mlir::llk::perf {

/// The only schema version this build understands. Bump it when a change
/// makes older machine files mean something different rather than merely
/// adding an optional key.
inline constexpr uint32_t kSupportedSchemaVersion = 1;

/// A resource scope that can own an execution tile. `parent` nests the scope
/// inside another one (`pe` inside `pe_group` inside `core`); it is absent for
/// top-level scopes such as a CPU's `worker` threads.
struct OwnerModel {
  std::string name;
  uint32_t count = 0;
  std::optional<std::string> mapsTo;
  std::optional<std::string> parent;
};

/// A matrix/MMA-style engine. `tileShapes` are `[m, n, k]` fragments the
/// engine can issue in one instruction; `inputDTypes`/`accumulatorDTypes` are
/// the dtype combinations it accepts.
struct MatrixEngineModel {
  std::string name;
  uint32_t count = 0;
  std::vector<std::array<int64_t, 3>> tileShapes;
  std::vector<std::string> inputDTypes;
  std::vector<std::string> accumulatorDTypes;
  uint64_t issueCycles = 0;
  uint64_t latencyCycles = 0;
  std::optional<double> flopsPerCycle;
  std::vector<std::string> supportedLayouts;
  std::optional<std::string> owner;
};

/// A vector/SIMD engine. `lanes` maps a dtype to the number of elements the
/// engine consumes per instruction, which is its fragment shape.
struct VectorEngineModel {
  std::string name;
  uint32_t count = 0;
  std::map<std::string, int64_t> lanes;
  uint64_t issueCycles = 0;
  uint64_t latencyCycles = 0;
  std::vector<std::string> supportedLayouts;
  std::optional<std::string> owner;
};

/// One level of the memory hierarchy. `name` mirrors the key it is stored
/// under; `alias` is the hardware name it stands for (`sram` aliasing `l1`).
struct MemoryLevelModel {
  std::string name;
  std::optional<std::string> alias;
  uint64_t capacityBytes = 0;
  double bandwidthBytesPerCycle = 0;
  uint64_t latencyCycles = 0;
  std::optional<uint32_t> banks;
  std::vector<std::string> supportedLayouts;
};

/// A legal memory-to-memory copy path. The optional costs override the
/// endpoint memory levels when a target's copy engine differs from both.
struct CopyPathModel {
  std::string source;
  std::string destination;
  std::optional<double> bandwidthBytesPerCycle;
  std::optional<uint64_t> latencyCycles;
};

struct DmaModel {
  uint32_t engines = 0;
  uint32_t maxOutstanding = 0;
  uint64_t setupCycles = 0;
  std::optional<std::string> mapsTo;
  std::optional<std::string> owner;
  std::vector<CopyPathModel> paths;
};

struct SyncModel {
  uint64_t barrierCycles = 0;
  uint64_t waitCycles = 0;
};

struct MachineModel {
  uint32_t schemaVersion = 0;
  std::string name;
  std::string description;
  uint64_t clockHz = 0;

  /// Host worker threads available to the target. Defaults to 1: a machine
  /// that does not model thread-level parallelism still executes somewhere.
  uint32_t workerThreads = 1;

  std::vector<OwnerModel> owners;
  std::vector<MatrixEngineModel> matrixEngines;
  std::vector<VectorEngineModel> vectorEngines;
  llvm::StringMap<MemoryLevelModel> memory;
  DmaModel dma;
  SyncModel sync;

  const OwnerModel *findOwner(llvm::StringRef name) const;
  const MatrixEngineModel *findMatrixEngine(llvm::StringRef name) const;
  const VectorEngineModel *findVectorEngine(llvm::StringRef name) const;
  const MemoryLevelModel *findMemory(llvm::StringRef name) const;
  const CopyPathModel *findCopyPath(llvm::StringRef source,
                                    llvm::StringRef destination) const;

  /// Owner resource count, or 0 when the owner is not modeled.
  uint32_t getOwnerCount(llvm::StringRef name) const;
};

/// Checks the invariants a loaded model must satisfy: known micro vocabulary,
/// positive capacities/latencies/bandwidths, well-formed fragment shapes,
/// declared and acyclic owners, unique resource names, and copy paths between
/// modeled memory spaces. Returns the first violation found, so diagnostics
/// are deterministic; message paths mirror the YAML schema
/// (`compute.matrix_engines[0].count`).
llvm::Error verifyMachineModel(const MachineModel &model);

} // namespace mlir::llk::perf

#endif // LLK_PERF_MACHINEMODEL_H
