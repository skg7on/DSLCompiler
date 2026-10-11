//===- OpDAGBuilder.h - Injectable leaf-operation DAG extraction
//-----------===//
#ifndef LLK_PERF_OPDAGBUILDER_H
#define LLK_PERF_OPDAGBUILDER_H

#include "LLK/Perf/MicroDAG.h"
#include "LLK/Perf/TileInfo.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"
#include <functional>
#include <utility>

namespace mlir::llk::perf {
/// Non-copyable facade valid only for the duration of one handler invocation.
/// Graph insertion, traversal state, producers and allocation accounting remain
/// core-owned. Handlers must not retain this context or mutate the IR.
class OpDAGBuildContext {
public:
  virtual ~OpDAGBuildContext() = default;
  OpDAGBuildContext(const OpDAGBuildContext &) = delete;
  OpDAGBuildContext &operator=(const OpDAGBuildContext &) = delete;
  virtual mlir::Operation &operation() const = 0;
  virtual const machine::MachineModel &machineModel() const = 0;
  virtual llvm::StringRef kernelName() const = 0;
  /// Explicit IDs in event.deps and extraDeps must name previously emitted
  /// events in this build. Use the IDs returned here; never retain IDs between
  /// builds. Value inputs are resolved through the core producer map.
  virtual uint32_t emitEvent(MicroEvent event, mlir::ValueRange inputs,
                             mlir::ValueRange outputs,
                             llvm::ArrayRef<uint32_t> extraDeps = {}) = 0;
  virtual void aliasValue(mlir::Value result, mlir::Value source) = 0;
  virtual std::string memoryOf(mlir::Value value) const = 0;
  virtual void accountStorage(llvm::StringRef memory, uint64_t bytes) = 0;
  virtual void noteUnsizable(const TileInfo &info) = 0;
  virtual void noteWarning(std::string message) = 0;
  virtual void noteLayoutUsage(llvm::StringRef engine,
                               const std::vector<std::string> &supported,
                               llvm::StringRef layout) = 0;
  virtual llvm::Error buildLogicalTile(mlir::Value source, mlir::Value result,
                                       const std::optional<std::string> &owner,
                                       EventKind kind) = 0;
  /// Resolve the current operation's route and bind outputs to its final hop.
  virtual llvm::Error buildMovement(llvm::StringRef src, llvm::StringRef dst,
                                    const TileInfo &source,
                                    const TileInfo &result, EventKind kind,
                                    llvm::ArrayRef<mlir::Value> outputs) = 0;
  virtual llvm::Error noteAllocation(const TileInfo &info,
                                     llvm::StringRef memory) = 0;
  /// Select the requested capability, honoring the current mapped placement.
  virtual const machine::ComputeNode *
  pickMatrixEngine(llvm::StringRef requested, std::string &reason) const = 0;
  virtual const machine::ComputeNode *
  pickVectorEngine(std::string &reason) const = 0;
  /// Select the conversion capability from the current transform executor.
  virtual const machine::ComputeNode *
  pickTransformEngine(std::string &reason) const = 0;
  virtual uint64_t vectorCycles(const machine::ComputeNode *engine,
                                llvm::StringRef dtype,
                                uint64_t elements) const = 0;
  virtual uint64_t mmaCycles(const machine::ComputeNode &engine,
                             llvm::ArrayRef<int64_t> shape,
                             uint64_t flops) const = 0;

protected:
  OpDAGBuildContext() = default;
};

/// Owns callbacks and their value captures. Configure before building and keep
/// immutable throughout a build. No implicit global/default registrations.
class OpDAGBuilderRegistry {
public:
  using Callback =
      std::function<llvm::Error(mlir::Operation &, OpDAGBuildContext &)>;
  llvm::Error registerBuilder(llvm::StringRef name, Callback callback);
  llvm::Error replaceBuilder(llvm::StringRef name, Callback callback);
  const Callback *lookup(llvm::StringRef name) const;
  template <typename OpT, typename Fn> llvm::Error registerBuilder(Fn fn) {
    return registerBuilder(OpT::getOperationName(), typed<OpT>(std::move(fn)));
  }
  template <typename OpT, typename Fn> llvm::Error replaceBuilder(Fn fn) {
    return replaceBuilder(OpT::getOperationName(), typed<OpT>(std::move(fn)));
  }

private:
  template <typename OpT, typename Fn> static Callback typed(Fn fn) {
    std::function<llvm::Error(OpT, OpDAGBuildContext &)> callback =
        std::move(fn);
    if (!callback)
      return {};
    return [fn = std::move(callback)](
               mlir::Operation &op, OpDAGBuildContext &context) -> llvm::Error {
      auto typedOp = llvm::dyn_cast<OpT>(&op);
      if (!typedOp)
        return llvm::make_error<llvm::StringError>(
            "operation builder type mismatch for '" +
                OpT::getOperationName().str() + "'",
            llvm::inconvertibleErrorCode());
      return fn(typedOp, context);
    };
  }
  llvm::StringMap<Callback> builders;
};
OpDAGBuilderRegistry makeDefaultOpDAGBuilders();
} // namespace mlir::llk::perf
#endif
