//===- ShapeProfile.cpp - Validated concrete samples for dynamic analysis -===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Perf/ShapeProfile.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cmath>

namespace mlir::llk::perf {

namespace {

llvm::Error profileError(const llvm::Twine &message) {
  return llvm::make_error<llvm::StringError>("shape_profile: " + message.str(),
                                             llvm::inconvertibleErrorCode());
}

/// Writes one value with a length delimiter so a symbol name containing the
/// separator can never collide with a different profile.
void appendDelimited(std::string &out, llvm::StringRef text) {
  out += std::to_string(text.size());
  out += ':';
  out.append(text.begin(), text.end());
}

/// Checks that `values` binds exactly the symbols `contract` declares.
llvm::Error checkSampleSymbols(const shape::ShapeContract &contract,
                               const shape::ShapeValues &values,
                               size_t sampleIndex) {
  for (const auto &entry : contract.symbols)
    if (!values.count(entry.first))
      return profileError("sample " + llvm::Twine(sampleIndex) +
                          " does not bind symbol '" + entry.first + "'");
  for (const auto &entry : values)
    if (!contract.symbols.count(entry.first))
      return profileError("sample " + llvm::Twine(sampleIndex) +
                          " binds unknown symbol '" + entry.first + "'");
  return llvm::Error::success();
}

/// Rebuilds the per-port extents from bound symbol values and lets the plain
/// shape layer check rank, constants, repeated symbols and the declared
/// constraints. This is the same check an invocation performs, so a sample is
/// only accepted when it is a shape the contract can actually bind.
llvm::Error checkSampleValues(const shape::ShapeContract &contract,
                              const shape::ShapeValues &values,
                              size_t sampleIndex) {
  std::vector<shape::PortShapeValues> ports;
  ports.reserve(contract.ports.size());
  std::array<uint32_t, 2> next = {0, 0};
  for (const shape::ShapePort &port : contract.ports) {
    shape::PortShapeValues entry;
    entry.direction = port.direction;
    entry.index = next[port.direction == shape::PortDirection::Input ? 0 : 1]++;
    entry.extents.reserve(port.shape.size());
    for (const shape::Dimension &dimension : port.shape) {
      if (dimension.isConstant()) {
        entry.extents.push_back(dimension.getConstant());
        continue;
      }
      auto found = values.find(dimension.getSymbol().str());
      if (found == values.end())
        return profileError("sample " + llvm::Twine(sampleIndex) +
                            " does not bind symbol '" +
                            dimension.getSymbol().str() + "'");
      entry.extents.push_back(found->second);
    }
    ports.push_back(std::move(entry));
  }

  llvm::Expected<shape::ShapeValues> bound =
      shape::bindShapeValues(contract, ports);
  if (!bound)
    return profileError("sample " + llvm::Twine(sampleIndex) +
                        " is invalid: " + llvm::toString(bound.takeError()));
  return llvm::Error::success();
}

/// Rejects any field outside the version-1 schema so a typo cannot silently
/// fall back to a default.
llvm::Error checkFields(const llvm::json::Object &object,
                        llvm::ArrayRef<llvm::StringRef> allowed,
                        llvm::StringRef context) {
  for (const auto &entry : object) {
    llvm::StringRef key = entry.first;
    if (llvm::is_contained(allowed, key))
      continue;
    return profileError("unknown field '" + key.str() + "' in " +
                        context.str());
  }
  return llvm::Error::success();
}

std::optional<double> numberOf(const llvm::json::Value &value) {
  if (std::optional<double> number = value.getAsNumber())
    return *number;
  return std::nullopt;
}

/// `llvm::json` represents an object as a map, so a duplicate key is silently
/// collapsed before semantic validation can see it. A profile is a durable
/// observation record, and accepting one of two conflicting values would make
/// its meaning depend on parser order; reject duplicate keys before parsing.
llvm::Error rejectDuplicateKeys(llvm::StringRef json) {
  llvm::SmallVector<llvm::StringSet<>, 8> objects;
  bool inString = false;
  bool escaped = false;
  std::string current;
  for (size_t index = 0; index < json.size(); ++index) {
    char character = json[index];
    if (inString) {
      if (escaped) {
        current += character;
        escaped = false;
      } else if (character == '\\') {
        escaped = true;
      } else if (character == '"') {
        inString = false;
        // A string followed by ':' is an object key. A string in an array (an
        // aggregation name, for instance) is followed by ',' or ']'.
        size_t next = index + 1;
        while (next < json.size() && (json[next] == ' ' || json[next] == '\t' ||
                                      json[next] == '\n' || json[next] == '\r'))
          ++next;
        if (next < json.size() && json[next] == ':') {
          if (objects.empty())
            return profileError("a key occurs outside an object");
          if (!objects.back().insert(current).second)
            return profileError("duplicate key '" + current + "'");
        }
      } else {
        current += character;
      }
      continue;
    }
    if (character == '"') {
      inString = true;
      escaped = false;
      current.clear();
    } else if (character == '{') {
      objects.emplace_back();
    } else if (character == '}') {
      if (objects.empty())
        return profileError("unbalanced '}' in the profile");
      objects.pop_back();
    }
  }
  if (inString)
    return profileError("unterminated string in the profile");
  return llvm::Error::success();
}

} // namespace

double sampleWeight(const ShapeSample &sample) { return sample.weight; }

llvm::Error validateShapeProfile(const shape::ShapeContract &contract,
                                 const ShapeProfile &profile) {
  if (profile.version != 1)
    return profileError("unsupported version " + llvm::Twine(profile.version) +
                        "; expected 1");
  if (profile.aggregation != "weighted_mean")
    return profileError("unsupported aggregation '" + profile.aggregation +
                        "'; expected weighted_mean");
  if (profile.maxVariants == 0)
    return profileError("max_variants must be positive");
  if (profile.samples.empty())
    return profileError("the profile has no samples; a dynamic plan cannot be "
                        "ranked without concrete observations");
  if (llvm::Error error = shape::validateShapeContract(contract))
    return profileError("invalid shape contract: " +
                        llvm::toString(std::move(error)));

  for (size_t index = 0; index < profile.samples.size(); ++index) {
    const ShapeSample &sample = profile.samples[index];
    if (!std::isfinite(sample.weight) || sample.weight <= 0.0)
      return profileError("sample " + llvm::Twine(index) + " has weight " +
                          std::to_string(sample.weight) +
                          "; weights must be finite and positive");
    if (llvm::Error error = checkSampleSymbols(contract, sample.values, index))
      return error;
    if (llvm::Error error = checkSampleValues(contract, sample.values, index))
      return error;
  }
  return llvm::Error::success();
}

llvm::Expected<ShapeProfile> parseShapeProfile(llvm::StringRef json) {
  if (llvm::Error duplicate = rejectDuplicateKeys(json))
    return std::move(duplicate);
  llvm::Expected<llvm::json::Value> parsed = llvm::json::parse(json);
  if (!parsed)
    return profileError("invalid JSON: " + llvm::toString(parsed.takeError()));
  llvm::json::Object *root = parsed->getAsObject();
  if (!root)
    return profileError("the profile must be a JSON object");
  if (llvm::Error error = checkFields(
          *root, {"version", "samples", "aggregation", "max_variants"},
          "the profile"))
    return std::move(error);

  ShapeProfile profile;

  if (const llvm::json::Value *version = root->get("version")) {
    std::optional<int64_t> value = version->getAsInteger();
    if (!value || *value < 0 || *value > UINT32_MAX)
      return profileError("version must be a nonnegative integer");
    profile.version = static_cast<uint32_t>(*value);
  } else {
    return profileError("version is required");
  }

  const llvm::json::Value *samples = root->get("samples");
  if (!samples)
    return profileError("samples is required");
  const llvm::json::Array *sampleArray = samples->getAsArray();
  if (!sampleArray)
    return profileError("samples must be an array");

  for (size_t index = 0; index < sampleArray->size(); ++index) {
    const llvm::json::Object *object = (*sampleArray)[index].getAsObject();
    if (!object)
      return profileError("sample " + llvm::Twine(index) +
                          " must be an object");
    if (llvm::Error error = checkFields(*object, {"values", "weight"},
                                        "sample " + llvm::Twine(index).str()))
      return std::move(error);

    const llvm::json::Value *values = object->get("values");
    if (!values)
      return profileError("sample " + llvm::Twine(index) + " requires values");
    const llvm::json::Object *valueObject = values->getAsObject();
    if (!valueObject)
      return profileError("sample " + llvm::Twine(index) +
                          " values must be an object");

    ShapeSample sample;
    llvm::StringSet<> seen;
    for (const auto &entry : *valueObject) {
      llvm::StringRef key = entry.first;
      if (!seen.insert(key).second)
        return profileError("sample " + llvm::Twine(index) + " binds symbol '" +
                            key.str() + "' more than once");
      std::optional<int64_t> value = entry.second.getAsInteger();
      if (!value)
        return profileError("sample " + llvm::Twine(index) + " symbol '" +
                            key.str() + "' must be an integer");
      sample.values.emplace(key.str(), *value);
    }

    if (const llvm::json::Value *weight = object->get("weight")) {
      if (std::optional<double> number = numberOf(*weight))
        sample.weight = *number;
      else
        return profileError("sample " + llvm::Twine(index) +
                            " weight must be a number");
    }
    profile.samples.push_back(std::move(sample));
  }

  if (const llvm::json::Value *aggregation = root->get("aggregation")) {
    if (std::optional<llvm::StringRef> text = aggregation->getAsString())
      profile.aggregation = text->str();
    else
      return profileError("aggregation must be a string");
  }

  if (const llvm::json::Value *variants = root->get("max_variants")) {
    std::optional<int64_t> value = variants->getAsInteger();
    if (!value || *value <= 0 || *value > UINT32_MAX)
      return profileError("max_variants must be a positive integer");
    profile.maxVariants = static_cast<uint32_t>(*value);
  }

  return profile;
}

std::string canonicalShapeProfile(const ShapeProfile &profile) {
  std::string out;
  appendDelimited(out, "shape-profile");
  appendDelimited(out, std::to_string(profile.version));
  appendDelimited(out, profile.aggregation);
  appendDelimited(out, std::to_string(profile.maxVariants));
  appendDelimited(out, std::to_string(profile.samples.size()));
  for (const ShapeSample &sample : profile.samples) {
    appendDelimited(out, std::to_string(sample.values.size()));
    for (const auto &entry : sample.values) {
      appendDelimited(out, entry.first);
      appendDelimited(out, std::to_string(entry.second));
    }
    llvm::SmallString<32> weight;
    llvm::raw_svector_ostream weightStream(weight);
    weightStream << llvm::format("%.17g", sample.weight);
    appendDelimited(out, weight);
  }
  return out;
}

} // namespace mlir::llk::perf
