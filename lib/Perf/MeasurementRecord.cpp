//===- MeasurementRecord.cpp - Verified kernel measurement records --------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Perf/MeasurementRecord.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mlir::llk::perf {
namespace {

/// Version 1 of the record schema and of the measurement protocol.
constexpr uint32_t kMeasurementRecordVersion = 1;
constexpr uint32_t kMeasurementProtocolVersion = 1;

llvm::Error recordError(const llvm::Twine &message) {
  return llvm::createStringError(llvm::Twine("measurement_record: ") + message);
}

llvm::Error missingField(llvm::StringRef field) {
  return recordError(llvm::Twine("missing field '") + field + "'");
}

llvm::Error fieldType(llvm::StringRef field, llvm::StringRef expected) {
  return recordError(llvm::Twine("field '") + field + "' must be " + expected);
}

llvm::Error invalidField(const llvm::Twine &field, const llvm::Twine &reason) {
  return recordError(llvm::Twine("field '") + field + "' " + reason);
}

llvm::Error requireNonEmpty(llvm::StringRef value, llvm::StringRef field) {
  if (value.empty())
    return invalidField(field, "must not be empty");
  return llvm::Error::success();
}

/// The median of a copy, leaving the recorded sample order untouched. An odd
/// count takes the middle value; an even count takes the mean of the two
/// middle values.
double sampleMedian(llvm::ArrayRef<double> samples) {
  std::vector<double> sorted(samples.begin(), samples.end());
  std::sort(sorted.begin(), sorted.end());
  const size_t count = sorted.size();
  if (count % 2 == 1)
    return sorted[count / 2];
  return (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0;
}

bool isPositiveFinite(double value) {
  return std::isfinite(value) && value > 0.0;
}

void appendField(std::string &out, llvm::StringRef value) {
  out += std::to_string(value.size());
  out += ':';
  out += value;
}

//===----------------------------------------------------------------------===//
// JSON field readers. Each distinguishes a missing key from a present value of
// the wrong type so a hand-edited record cannot silently take a default.
//===----------------------------------------------------------------------===//

llvm::StringRef displayName(llvm::StringRef field, llvm::StringRef display) {
  return display.empty() ? field : display;
}

llvm::Error readStringField(const llvm::json::Object &object,
                            llvm::StringRef field, std::string &out,
                            llvm::StringRef display = {}) {
  const llvm::json::Value *value = object.get(field);
  const llvm::StringRef name = displayName(field, display);
  if (!value)
    return missingField(name);
  std::optional<llvm::StringRef> text = value->getAsString();
  if (!text)
    return fieldType(name, "a string");
  out = text->str();
  return llvm::Error::success();
}

llvm::Error
readUnsignedField(const llvm::json::Object &object, llvm::StringRef field,
                  uint64_t &out, llvm::StringRef display = {},
                  uint64_t maximum = std::numeric_limits<uint64_t>::max()) {
  const llvm::json::Value *value = object.get(field);
  const llvm::StringRef name = displayName(field, display);
  if (!value)
    return missingField(name);
  std::optional<int64_t> number = value->getAsInteger();
  if (!number || *number < 0 || static_cast<uint64_t>(*number) > maximum)
    return fieldType(name, "a nonnegative integer");
  out = static_cast<uint64_t>(*number);
  return llvm::Error::success();
}

llvm::Error readNumberField(const llvm::json::Object &object,
                            llvm::StringRef field, double &out,
                            llvm::StringRef display = {}) {
  const llvm::json::Value *value = object.get(field);
  const llvm::StringRef name = displayName(field, display);
  if (!value)
    return missingField(name);
  std::optional<double> number = value->getAsNumber();
  if (!number)
    return fieldType(name, "a number");
  out = *number;
  return llvm::Error::success();
}

llvm::Error readBoolField(const llvm::json::Object &object,
                          llvm::StringRef field, bool &out,
                          llvm::StringRef display = {}) {
  const llvm::json::Value *value = object.get(field);
  const llvm::StringRef name = displayName(field, display);
  if (!value)
    return missingField(name);
  std::optional<bool> flag = value->getAsBoolean();
  if (!flag)
    return fieldType(name, "a boolean");
  out = *flag;
  return llvm::Error::success();
}

llvm::Error readObjectField(const llvm::json::Object &object,
                            llvm::StringRef field,
                            const llvm::json::Object *&out) {
  const llvm::json::Value *value = object.get(field);
  if (!value)
    return missingField(field);
  out = value->getAsObject();
  if (!out)
    return fieldType(field, "an object");
  return llvm::Error::success();
}

// Prediction members are optional so a static estimate that did not fill every
// secondary metric round-trips. A present member of the wrong type is still an
// error.
llvm::Error readOptionalNumber(const llvm::json::Object &object,
                               llvm::StringRef field, double &out) {
  const llvm::json::Value *value = object.get(field);
  if (!value)
    return llvm::Error::success();
  std::optional<double> number = value->getAsNumber();
  if (!number)
    return fieldType(field, "a number");
  out = *number;
  return llvm::Error::success();
}

llvm::Error readOptionalUnsigned(const llvm::json::Object &object,
                                 llvm::StringRef field, uint64_t &out) {
  const llvm::json::Value *value = object.get(field);
  if (!value)
    return llvm::Error::success();
  std::optional<int64_t> number = value->getAsInteger();
  if (!number || *number < 0)
    return fieldType(field, "a nonnegative integer");
  out = static_cast<uint64_t>(*number);
  return llvm::Error::success();
}

llvm::Error readOptionalString(const llvm::json::Object &object,
                               llvm::StringRef field, std::string &out) {
  const llvm::json::Value *value = object.get(field);
  if (!value)
    return llvm::Error::success();
  std::optional<llvm::StringRef> text = value->getAsString();
  if (!text)
    return fieldType(field, "a string");
  out = text->str();
  return llvm::Error::success();
}

llvm::json::Object
environmentToJson(const MeasurementEnvironment &environment) {
  llvm::json::Object object;
  object["architecture"] = environment.architecture;
  object["cpu_identity"] = environment.cpuIdentity;
  object["isa"] = environment.isa;
  object["threads"] = environment.threads;
  object["affinity"] = environment.affinity;
  object["clock_policy"] = environment.clockPolicy;
  return object;
}

llvm::json::Object protocolToJson(const MeasurementProtocol &protocol) {
  llvm::json::Object object;
  object["version"] = static_cast<uint64_t>(protocol.version);
  object["warmup"] = protocol.warmup;
  object["repeat"] = protocol.repeat;
  object["timer"] = protocol.timer;
  object["statistic"] = protocol.statistic;
  object["scope"] = protocol.scope;
  return object;
}

llvm::json::Object
verificationToJson(const VerificationIdentity &verification) {
  llvm::json::Object object;
  object["fixture_content"] = verification.fixtureContent;
  object["verifier_version"] = verification.verifierVersion;
  object["numerical_contract"] = verification.numericalContract;
  return object;
}

llvm::json::Object predictionToJson(const CandidateMetrics &prediction) {
  llvm::json::Object object;
  object["predicted_cycles"] = prediction.predictedCycles;
  object["predicted_ns"] = prediction.predictedNs;
  object["matrix_utilization"] = prediction.matrixUtilization;
  object["dma_utilization"] = prediction.dmaUtilization;
  object["dram_bytes"] = prediction.dramBytes;
  object["sram_bytes"] = prediction.sramBytes;
  object["bottleneck"] = prediction.bottleneck;
  // measured_ns/measured_gflops are not part of a prediction and are never
  // emitted here.
  return object;
}

} // namespace

llvm::Error validateMeasurementSamples(const MeasurementProtocol &protocol,
                                       llvm::ArrayRef<double> samplesNs,
                                       double medianNs) {
  if (protocol.version != kMeasurementProtocolVersion)
    return invalidField("protocol.version", "must be 1");
  if (protocol.timer != "steady_clock")
    return invalidField("protocol.timer", "must be \"steady_clock\"");
  if (protocol.statistic != "median")
    return invalidField("protocol.statistic", "must be \"median\"");
  if (protocol.scope != "per_invocation")
    return invalidField("protocol.scope", "must be \"per_invocation\"");
  if (samplesNs.empty())
    return invalidField("samples_ns", "must not be empty");
  for (size_t index = 0; index < samplesNs.size(); ++index) {
    if (!isPositiveFinite(samplesNs[index]))
      return invalidField(llvm::Twine("samples_ns[") + llvm::Twine(index) + "]",
                          "must be finite and positive");
  }
  if (protocol.repeat != samplesNs.size())
    return invalidField("protocol.repeat",
                        "must equal the number of samples_ns");
  if (!isPositiveFinite(medianNs))
    return invalidField("median_ns", "must be finite and positive");
  const double expectedMedian = sampleMedian(samplesNs);
  const double scale = std::max(1.0, std::abs(expectedMedian));
  if (std::abs(medianNs - expectedMedian) > 1e-9 * scale)
    return invalidField("median_ns", "does not match the sample median");

  return llvm::Error::success();
}

llvm::Error validateMeasurementRecord(const MeasurementRecord &record) {
  if (record.version != kMeasurementRecordVersion)
    return invalidField("version", "must be 1");

  if (llvm::Error error =
          requireNonEmpty(record.executableIdentity, "executable_identity"))
    return error;
  if (llvm::Error error =
          requireNonEmpty(record.executableHash, "executable_hash"))
    return error;
  if (record.executableHash !=
      measurementContentHash(record.executableIdentity))
    return invalidField("executable_hash",
                        "does not match executable_identity");
  if (llvm::Error error =
          requireNonEmpty(record.targetIdentity, "target_identity"))
    return error;
  if (llvm::Error error =
          requireNonEmpty(record.backendIdentity, "backend_identity"))
    return error;

  if (llvm::Error error = requireNonEmpty(record.environment.architecture,
                                          "environment.architecture"))
    return error;
  if (llvm::Error error = requireNonEmpty(record.environment.cpuIdentity,
                                          "environment.cpu_identity"))
    return error;
  if (llvm::Error error =
          requireNonEmpty(record.environment.isa, "environment.isa"))
    return error;
  if (record.environment.threads == 0)
    return invalidField("environment.threads", "must be positive");
  if (llvm::Error error =
          requireNonEmpty(record.environment.affinity, "environment.affinity"))
    return error;
  if (llvm::Error error = requireNonEmpty(record.environment.clockPolicy,
                                          "environment.clock_policy"))
    return error;

  if (llvm::Error error = requireNonEmpty(record.verification.fixtureContent,
                                          "verification.fixture_content"))
    return error;
  if (llvm::Error error = requireNonEmpty(record.verification.verifierVersion,
                                          "verification.verifier_version"))
    return error;
  if (llvm::Error error = requireNonEmpty(record.verification.numericalContract,
                                          "verification.numerical_contract"))
    return error;

  if (llvm::Error error =
          requireNonEmpty(record.sourceAdapter, "source_adapter"))
    return error;
  if (llvm::Error error =
          requireNonEmpty(record.workloadContract, "workload_contract"))
    return error;
  if (llvm::Error error =
          requireNonEmpty(record.candidateBindings, "candidate_bindings"))
    return error;
  if (llvm::Error error =
          requireNonEmpty(record.compilerVersion, "compiler_version"))
    return error;

  if (!record.verified)
    return invalidField("verified", "must be true");

  if (auto error = validateMeasurementSamples(record.protocol, record.samplesNs,
                                              record.medianNs))
    return error;

  if (!std::isfinite(record.prediction.predictedNs) ||
      record.prediction.predictedNs < 0.0)
    return invalidField("prediction.predicted_ns",
                        "must be finite and nonnegative");

  return llvm::Error::success();
}

llvm::json::Object measurementRecordToJson(const MeasurementRecord &record) {
  llvm::json::Object object;
  object["version"] = static_cast<uint64_t>(record.version);
  object["executable_identity"] = record.executableIdentity;
  object["executable_hash"] = record.executableHash;
  object["target_identity"] = record.targetIdentity;
  object["backend_identity"] = record.backendIdentity;
  object["environment"] = environmentToJson(record.environment);
  object["protocol"] = protocolToJson(record.protocol);
  object["verification"] = verificationToJson(record.verification);
  object["verified"] = record.verified;

  llvm::json::Array samples;
  samples.reserve(record.samplesNs.size());
  for (double sample : record.samplesNs)
    samples.push_back(sample);
  object["samples_ns"] = std::move(samples);
  object["median_ns"] = record.medianNs;
  object["prediction"] = predictionToJson(record.prediction);

  object["source_adapter"] = record.sourceAdapter;
  object["workload_contract"] = record.workloadContract;
  object["candidate_bindings"] = record.candidateBindings;
  object["bottleneck"] = record.bottleneck;
  object["timestamp_utc"] = record.timestampUtc;
  object["compiler_version"] = record.compilerVersion;
  return object;
}

llvm::Expected<MeasurementRecord>
measurementRecordFromJson(const llvm::json::Value &value) {
  const llvm::json::Object *root = value.getAsObject();
  if (!root)
    return recordError("record must be a JSON object");

  MeasurementRecord record;

  auto qualify = [](llvm::StringRef prefix, llvm::StringRef field) {
    return (llvm::Twine(prefix) + "." + field).str();
  };

  uint64_t number = 0;
  if (llvm::Error error = readUnsignedField(
          *root, "version", number, {}, std::numeric_limits<uint32_t>::max()))
    return std::move(error);
  record.version = static_cast<uint32_t>(number);

  if (llvm::Error error = readStringField(*root, "executable_identity",
                                          record.executableIdentity))
    return std::move(error);
  if (llvm::Error error =
          readStringField(*root, "executable_hash", record.executableHash))
    return std::move(error);
  if (llvm::Error error =
          readStringField(*root, "target_identity", record.targetIdentity))
    return std::move(error);
  if (llvm::Error error =
          readStringField(*root, "backend_identity", record.backendIdentity))
    return std::move(error);

  const llvm::json::Object *environment = nullptr;
  if (llvm::Error error = readObjectField(*root, "environment", environment))
    return std::move(error);
  if (llvm::Error error = readStringField(
          *environment, "architecture", record.environment.architecture,
          qualify("environment", "architecture")))
    return std::move(error);
  if (llvm::Error error = readStringField(
          *environment, "cpu_identity", record.environment.cpuIdentity,
          qualify("environment", "cpu_identity")))
    return std::move(error);
  if (llvm::Error error =
          readStringField(*environment, "isa", record.environment.isa,
                          qualify("environment", "isa")))
    return std::move(error);
  if (llvm::Error error =
          readUnsignedField(*environment, "threads", record.environment.threads,
                            qualify("environment", "threads")))
    return std::move(error);
  if (llvm::Error error =
          readStringField(*environment, "affinity", record.environment.affinity,
                          qualify("environment", "affinity")))
    return std::move(error);
  if (llvm::Error error = readStringField(
          *environment, "clock_policy", record.environment.clockPolicy,
          qualify("environment", "clock_policy")))
    return std::move(error);

  const llvm::json::Object *protocol = nullptr;
  if (llvm::Error error = readObjectField(*root, "protocol", protocol))
    return std::move(error);
  if (llvm::Error error = readUnsignedField(
          *protocol, "version", number, qualify("protocol", "version"),
          std::numeric_limits<uint32_t>::max()))
    return std::move(error);
  record.protocol.version = static_cast<uint32_t>(number);
  if (llvm::Error error =
          readUnsignedField(*protocol, "warmup", record.protocol.warmup,
                            qualify("protocol", "warmup")))
    return std::move(error);
  if (llvm::Error error =
          readUnsignedField(*protocol, "repeat", record.protocol.repeat,
                            qualify("protocol", "repeat")))
    return std::move(error);
  if (llvm::Error error =
          readStringField(*protocol, "timer", record.protocol.timer,
                          qualify("protocol", "timer")))
    return std::move(error);
  if (llvm::Error error =
          readStringField(*protocol, "statistic", record.protocol.statistic,
                          qualify("protocol", "statistic")))
    return std::move(error);
  if (llvm::Error error =
          readStringField(*protocol, "scope", record.protocol.scope,
                          qualify("protocol", "scope")))
    return std::move(error);

  const llvm::json::Object *verification = nullptr;
  if (llvm::Error error = readObjectField(*root, "verification", verification))
    return std::move(error);
  if (llvm::Error error = readStringField(
          *verification, "fixture_content", record.verification.fixtureContent,
          qualify("verification", "fixture_content")))
    return std::move(error);
  if (llvm::Error error =
          readStringField(*verification, "verifier_version",
                          record.verification.verifierVersion,
                          qualify("verification", "verifier_version")))
    return std::move(error);
  if (llvm::Error error =
          readStringField(*verification, "numerical_contract",
                          record.verification.numericalContract,
                          qualify("verification", "numerical_contract")))
    return std::move(error);

  if (llvm::Error error = readBoolField(*root, "verified", record.verified))
    return std::move(error);

  const llvm::json::Value *samplesValue = root->get("samples_ns");
  if (!samplesValue)
    return missingField("samples_ns");
  const llvm::json::Array *samples = samplesValue->getAsArray();
  if (!samples)
    return fieldType("samples_ns", "an array of numbers");
  for (const llvm::json::Value &sample : *samples) {
    std::optional<double> numberValue = sample.getAsNumber();
    if (!numberValue)
      return fieldType("samples_ns", "an array of numbers");
    record.samplesNs.push_back(*numberValue);
  }

  if (llvm::Error error = readNumberField(*root, "median_ns", record.medianNs))
    return std::move(error);

  const llvm::json::Object *prediction = nullptr;
  if (llvm::Error error = readObjectField(*root, "prediction", prediction))
    return std::move(error);
  if (llvm::Error error = readOptionalUnsigned(
          *prediction, "predicted_cycles", record.prediction.predictedCycles))
    return std::move(error);
  if (llvm::Error error = readOptionalNumber(*prediction, "predicted_ns",
                                             record.prediction.predictedNs))
    return std::move(error);
  if (llvm::Error error =
          readOptionalNumber(*prediction, "matrix_utilization",
                             record.prediction.matrixUtilization))
    return std::move(error);
  if (llvm::Error error = readOptionalNumber(*prediction, "dma_utilization",
                                             record.prediction.dmaUtilization))
    return std::move(error);
  if (llvm::Error error = readOptionalUnsigned(*prediction, "dram_bytes",
                                               record.prediction.dramBytes))
    return std::move(error);
  if (llvm::Error error = readOptionalUnsigned(*prediction, "sram_bytes",
                                               record.prediction.sramBytes))
    return std::move(error);
  if (llvm::Error error = readOptionalString(*prediction, "bottleneck",
                                             record.prediction.bottleneck))
    return std::move(error);

  if (llvm::Error error =
          readStringField(*root, "source_adapter", record.sourceAdapter))
    return std::move(error);
  if (llvm::Error error =
          readStringField(*root, "workload_contract", record.workloadContract))
    return std::move(error);
  if (llvm::Error error = readStringField(*root, "candidate_bindings",
                                          record.candidateBindings))
    return std::move(error);
  if (llvm::Error error =
          readOptionalString(*root, "bottleneck", record.bottleneck))
    return std::move(error);
  if (llvm::Error error =
          readStringField(*root, "timestamp_utc", record.timestampUtc))
    return std::move(error);
  if (llvm::Error error =
          readStringField(*root, "compiler_version", record.compilerVersion))
    return std::move(error);

  if (llvm::Error error = validateMeasurementRecord(record))
    return std::move(error);
  return record;
}

std::string canonicalMeasurementKey(const MeasurementRecord &record) {
  std::string key;
  appendField(key, "measurement-record-v1");
  appendField(key, std::to_string(record.version));
  appendField(key, record.executableIdentity);
  appendField(key, record.targetIdentity);
  appendField(key, record.backendIdentity);
  appendField(key, record.environment.architecture);
  appendField(key, record.environment.cpuIdentity);
  appendField(key, record.environment.isa);
  appendField(key, std::to_string(record.environment.threads));
  appendField(key, record.environment.affinity);
  appendField(key, record.environment.clockPolicy);
  appendField(key, std::to_string(record.protocol.version));
  appendField(key, std::to_string(record.protocol.warmup));
  appendField(key, std::to_string(record.protocol.repeat));
  appendField(key, record.protocol.timer);
  appendField(key, record.protocol.statistic);
  appendField(key, record.protocol.scope);
  appendField(key, record.verification.fixtureContent);
  appendField(key, record.verification.verifierVersion);
  appendField(key, record.verification.numericalContract);
  appendField(key, record.sourceAdapter);
  appendField(key, record.workloadContract);
  appendField(key, record.candidateBindings);
  appendField(key, record.compilerVersion);
  return key;
}

std::string measurementContentHash(llvm::StringRef value) {
  uint64_t hash = 14695981039346656037ull;
  for (unsigned char byte : value.bytes()) {
    hash ^= byte;
    hash *= 1099511628211ull;
  }
  return std::to_string(hash);
}

} // namespace mlir::llk::perf
