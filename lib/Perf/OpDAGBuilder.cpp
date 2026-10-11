#include "LLK/Perf/OpDAGBuilder.h"

namespace mlir::llk::perf {
namespace {
llvm::Error invalid(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}
llvm::Error validate(llvm::StringRef name,
                     const OpDAGBuilderRegistry::Callback &callback) {
  if (name.empty())
    return invalid("operation builder name is empty");
  if (name == "micro.for" || name == "micro.spatial_for" ||
      name == "micro.pipeline" || name == "micro.yield")
    return invalid("cannot register structural operation '" + name + "'");
  if (!callback)
    return invalid("operation builder callback is empty for '" + name + "'");
  return llvm::Error::success();
}
} // namespace
llvm::Error OpDAGBuilderRegistry::registerBuilder(llvm::StringRef name,
                                                  Callback callback) {
  if (auto error = validate(name, callback))
    return error;
  if (builders.count(name))
    return invalid("duplicate operation builder for '" + name + "'");
  builders.try_emplace(name, std::move(callback));
  return llvm::Error::success();
}
llvm::Error OpDAGBuilderRegistry::replaceBuilder(llvm::StringRef name,
                                                 Callback callback) {
  if (auto error = validate(name, callback))
    return error;
  auto found = builders.find(name);
  if (found == builders.end())
    return invalid("no operation builder to replace for '" + name + "'");
  found->second = std::move(callback);
  return llvm::Error::success();
}
const OpDAGBuilderRegistry::Callback *
OpDAGBuilderRegistry::lookup(llvm::StringRef name) const {
  auto found = builders.find(name);
  return found == builders.end() ? nullptr : &found->second;
}
} // namespace mlir::llk::perf
