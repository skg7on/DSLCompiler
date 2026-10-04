//===- SearchBindingLoader.cpp - micro.candidate -> SearchBinding ---------===//
//
// Part of phase-4 of epic #67. See SearchBindingLoader.h.
//
// The dialect verifier already rejects a malformed candidate when it runs, but
// the loader must stand on its own: the binding it returns is the point the
// tuner searches around, so the domain check has to hold at load time even if
// no verifier has run. The checks here therefore mirror the dialect's and are
// deliberately duplicated -- they are the startup diagnostic the tuning stage
// relies on (ruling S2).
//
//===----------------------------------------------------------------------===//

#include "LLK/Conversion/MicroMapping/SearchBindingLoader.h"

#include "LLK/Perf/SearchSpace.h"

// MicroDialect/MicroEnums/MicroHelpers are prerequisites for the generated
// attribute, type, and op declarations below, not direct uses here.
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Dialect/Micro/MicroEnums.h"
#include "LLK/Dialect/Micro/MicroHelpers.h"

#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

// Micro attribute, type, and op declarations.
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/Micro/MicroAttributes.h.inc"
#define GET_TYPEDEF_CLASSES
#include "LLK/Dialect/Micro/MicroTypes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/Micro/MicroOps.h.inc"

namespace mlir::llk::mapping {

using llvm::StringRef;

namespace {

llvm::Error error(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>(message.str(),
                                             llvm::inconvertibleErrorCode());
}

/// Finds the module's candidates, in walk order.
llvm::SmallVector<micro::CandidateOp, 4> findCandidates(mlir::ModuleOp module) {
  llvm::SmallVector<micro::CandidateOp, 4> candidates;
  module.walk(
      [&](micro::CandidateOp candidate) { candidates.push_back(candidate); });
  return candidates;
}

/// Reads one binding attribute as a `SearchValue`. The dialect verifier only
/// admits integers and strings, but the loader does not assume it ran.
llvm::Expected<SearchValue> readValue(micro::CandidateOp candidate,
                                      StringRef name, mlir::Attribute value) {
  if (auto integer = dyn_cast<mlir::IntegerAttr>(value))
    return SearchValue(int64_t{integer.getInt()});
  if (auto text = dyn_cast<mlir::StringAttr>(value))
    return SearchValue(text.getValue().str());
  return error("micro.candidate @" + candidate.getSymName() + " binding for '" +
               name + "' must be an integer or a string");
}

} // namespace

llvm::Expected<SearchBinding> loadSearchBinding(mlir::ModuleOp module,
                                                StringRef candidateSymbol) {
  llvm::SmallVector<micro::CandidateOp, 4> candidates = findCandidates(module);

  micro::CandidateOp candidate;
  if (candidateSymbol.empty()) {
    if (candidates.empty())
      return error("module has no micro.candidate");
    if (candidates.size() > 1)
      return error("module has more than one micro.candidate; name the one to "
                   "load");
    candidate = candidates.front();
  } else {
    for (micro::CandidateOp found : candidates)
      if (found.getSymName() == candidateSymbol) {
        if (candidate)
          return error("more than one micro.candidate named '" +
                       candidateSymbol + "'");
        candidate = found;
      }
    if (!candidate)
      return error("no micro.candidate named '" + candidateSymbol + "'");
  }

  // The parameters are declared by the enclosing space, so read them through
  // the typed loader rather than re-parsing the ops here.
  auto space = dyn_cast<micro::SearchSpaceOp>(candidate->getParentOp());
  if (!space)
    return error("micro.candidate @" + candidate.getSymName() +
                 " is not nested in a micro.search_space");

  llvm::Expected<perf::SearchSpace> loaded = perf::loadSearchSpace(space);
  if (!loaded)
    return loaded.takeError();
  const perf::SearchSpace &typed = *loaded;

  llvm::StringMap<SearchValue> values;
  for (mlir::NamedAttribute binding : candidate.getBindings()) {
    StringRef name = binding.getName().strref();
    const perf::SearchParam *param = typed.findParam(name);
    if (!param)
      return error("micro.candidate @" + candidate.getSymName() +
                   " binds unknown parameter '" + name + "'");

    auto value = readValue(candidate, name, binding.getValue());
    if (!value)
      return value.takeError();

    bool inDomain = false;
    for (const perf::SearchChoice &choice : param->choices)
      if (choice.value == *value) {
        inDomain = true;
        break;
      }
    if (!inDomain)
      // Wording matches SearchSpaceOp::verify's "is not one of the declared
      // choices" so the verifier/loader message contract cannot drift.
      return error("micro.candidate @" + candidate.getSymName() +
                   " value for '" + name +
                   "' is not one of the declared choices of micro.param '" +
                   param->name + "'");

    values[name] = std::move(*value);
  }

  // A candidate is a *complete* assignment: an unbound parameter is as fatal
  // as an out-of-domain one.
  for (const perf::SearchParam &param : typed.params)
    if (values.find(param.name) == values.end())
      return error("micro.candidate @" + candidate.getSymName() +
                   " does not bind parameter '" + param.name + "'");

  return makeSearchBinding(candidate.getSymName().str(), std::move(values));
}

} // namespace mlir::llk::mapping
