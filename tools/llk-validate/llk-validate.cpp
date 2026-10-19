//===- llk-validate.cpp - Held-out prediction validation ----------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Perf/PredictionValidation.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>
#include <set>

namespace {
namespace perf = mlir::llk::perf;
namespace cl = llvm::cl;
cl::list<std::string> inputs(cl::Positional, cl::ZeroOrMore,
                             cl::desc("record.json or record-directory ..."));
cl::opt<std::string> format("format", cl::init("json"),
                            cl::desc("json or text"));
cl::opt<std::string> output("output", cl::init("-"),
                            cl::desc("Report path or - for stdout"));
cl::opt<std::string>
    holdout("holdout-manifest", cl::init(""),
            cl::desc("Version-1 manifest of full executable identities"));
cl::list<std::string>
    training("training-manifest",
             cl::desc("Training identity manifest; repeatable"));
cl::list<std::string>
    calibration("calibration-manifest",
                cl::desc("Calibration identity manifest; repeatable"));

llvm::Error fail(const llvm::Twine &message) {
  return llvm::createStringError("llk-validate: " + message);
}

llvm::Expected<llvm::json::Value> readJson(llvm::StringRef path) {
  auto buffer = llvm::MemoryBuffer::getFile(path);
  if (!buffer)
    return fail("cannot read '" + path + "': " + buffer.getError().message());
  auto parsed = llvm::json::parse((*buffer)->getBuffer());
  if (!parsed)
    return fail("invalid JSON in '" + path +
                "': " + llvm::toString(parsed.takeError()));
  return parsed;
}

llvm::Expected<std::set<std::string>> readManifest(llvm::StringRef path) {
  auto json = readJson(path);
  if (!json)
    return json.takeError();
  auto *object = json->getAsObject();
  if (!object || object->getInteger("version") != 1 ||
      !object->getArray("identities"))
    return fail("manifest '" + path +
                "' requires version 1 and identities array");
  std::set<std::string> identities;
  for (const auto &value : *object->getArray("identities")) {
    auto identity = value.getAsString();
    if (!identity || identity->empty())
      return fail("manifest '" + path +
                  "' requires nonempty full identity strings");
    if (!identities.insert(identity->str()).second)
      return fail("duplicate identity in manifest '" + path + "'");
  }
  return identities;
}

llvm::Expected<std::vector<std::string>> recordFiles() {
  if (inputs.empty())
    return fail("supply explicit record files or directories");
  std::vector<std::string> paths;
  std::set<std::string> canonicalPaths;
  for (const auto &input : inputs) {
    std::vector<std::string> discovered;
    if (llvm::sys::fs::is_directory(input)) {
      std::error_code error;
      for (llvm::sys::fs::directory_iterator it(input, error), end;
           it != end && !error; it.increment(error)) {
        if (llvm::StringRef(it->path()).ends_with(".json") &&
            llvm::sys::fs::is_regular_file(it->path()))
          discovered.push_back(it->path());
      }
      if (error)
        return fail("cannot list '" + input + "': " + error.message());
      std::sort(discovered.begin(), discovered.end());
    } else {
      discovered.push_back(input);
    }
    for (const auto &path : discovered) {
      llvm::SmallString<256> canonical;
      if (auto error = llvm::sys::fs::real_path(path, canonical))
        return fail("cannot resolve '" + path + "': " + error.message());
      if (!canonicalPaths.insert(canonical.str().str()).second)
        return fail("duplicate input record path '" + path + "'");
      paths.push_back(canonical.str().str());
    }
  }
  return paths;
}

llvm::Expected<llvm::json::Object> buildReport() {
  if (format != "json" && format != "text")
    return fail("--format must be json or text");
  if (holdout.empty() && (!training.empty() || !calibration.empty()))
    return fail(
        "training/calibration manifests require an explicit holdout manifest");
  auto files = recordFiles();
  if (!files)
    return files.takeError();
  std::set<std::string> heldOut, fitted;
  if (!holdout.empty()) {
    auto identities = readManifest(holdout);
    if (!identities)
      return identities.takeError();
    heldOut = std::move(*identities);
  }
  // Reject all overlap before selecting any rows; a missing input record does
  // not make a training/holdout identity intersection acceptable.
  for (const auto *manifests : {&training, &calibration}) {
    for (const auto &path : *manifests) {
      auto identities = readManifest(path);
      if (!identities)
        return identities.takeError();
      for (const auto &identity : *identities) {
        if (heldOut.count(identity))
          return fail("holdout/training or calibration identity overlap in '" +
                      path + "'");
        if (!fitted.insert(identity).second)
          return fail("duplicate training/calibration identity in '" + path +
                      "'");
      }
    }
  }

  // Parse and validate every supplied record, including non-holdout data. A
  // corrupt training record must not silently disappear during selection.
  std::map<std::string, std::vector<perf::MeasurementRecord>> groups;
  std::set<std::string> keys, observed;
  uint64_t selectedCount = 0, notHeldOut = 0;
  for (const auto &path : *files) {
    auto json = readJson(path);
    if (!json)
      return json.takeError();
    auto record = perf::measurementRecordFromJson(*json);
    if (!record)
      return fail("invalid record '" + path +
                  "': " + llvm::toString(record.takeError()));
    if (!keys.insert(perf::canonicalMeasurementKey(*record)).second)
      return fail("duplicate measurement identity in '" + path + "'");
    observed.insert(record->executableIdentity);
    if (!holdout.empty() && !heldOut.count(record->executableIdentity)) {
      notHeldOut++;
      continue;
    }
    std::string regime = perf::predictionValidationRegimeKey(*record);
    groups[regime].push_back(std::move(*record));
    selectedCount++;
  }
  for (const auto &identity : heldOut)
    if (!observed.count(identity))
      return fail("holdout identity missing from supplied records: '" +
                  identity + "'");

  llvm::json::Array reports;
  uint64_t compared = 0, excluded = 0, samples = 0;
  for (const auto &group : groups) {
    auto validation = perf::validatePredictions(group.second);
    if (!validation)
      return validation.takeError();
    compared += validation->rows.size();
    excluded += validation->excluded.size();
    samples += validation->sampleCount;
    reports.push_back(perf::predictionValidationToJson(*validation));
  }
  llvm::json::Array paths;
  for (const auto &path : *files)
    paths.push_back(path);
  return llvm::json::Object{
      {"version", 1},
      {"calibrated", false},
      {"held_out", !holdout.empty()},
      {"status", compared ? "ok" : "no_comparable_samples"},
      {"record_count", selectedCount},
      {"input_record_count", files->size()},
      {"not_in_holdout_count", notHeldOut},
      {"compared_record_count", compared},
      {"excluded_record_count", excluded},
      {"sample_count", samples},
      {"holdout_identity_count", heldOut.size()},
      {"training_calibration_identity_count", fitted.size()},
      {"input_files", std::move(paths)},
      {"groups", std::move(reports)}};
}

void printText(const llvm::json::Object &report, llvm::raw_ostream &out) {
  out << "uncalibrated prediction validation; held_out="
      << (*report.getBoolean("held_out") ? "true" : "false")
      << " status=" << *report.getString("status") << '\n';
  for (const char *counter :
       {"input_record_count", "record_count", "compared_record_count",
        "excluded_record_count", "not_in_holdout_count", "sample_count",
        "holdout_identity_count", "training_calibration_identity_count"})
    out << counter << '=' << *report.getInteger(counter) << '\n';
  size_t index = 0;
  for (const auto &value : *report.getArray("groups")) {
    const auto &group = *value.getAsObject();
    out << "group " << index++ << " status=" << *group.getString("status")
        << '\n';
    for (const char *counter : {"record_count", "compared_record_count",
                                "excluded_record_count", "sample_count"})
      out << counter << '=' << *group.getInteger(counter) << '\n';
    out << "regime=" << *group.get("regime") << '\n';
    for (const char *metric :
         {"median_absolute_relative_error", "p90_absolute_relative_error"})
      out << metric << '=' << *group.get(metric) << '\n';
    out << "rows=" << *group.get("rows") << '\n';
    out << "excluded=" << *group.get("excluded") << '\n';
  }
}

llvm::Error run() {
  auto report = buildReport();
  if (!report)
    return report.takeError();
  // A report is a new artifact, never a rewrite of an input record/manifest.
  if (output != "-") {
    llvm::SmallString<256> canonicalOutput;
    if (!llvm::sys::fs::real_path(output, canonicalOutput)) {
      for (const auto &path : *report->getArray("input_files"))
        if (llvm::sys::fs::equivalent(*path.getAsString(), output))
          return fail("report output would overwrite an input record");
      std::vector<std::string> manifests(training.begin(), training.end());
      manifests.insert(manifests.end(), calibration.begin(), calibration.end());
      if (!holdout.empty())
        manifests.push_back(holdout);
      for (const auto &path : manifests) {
        if (llvm::sys::fs::equivalent(path, output))
          return fail("report output would overwrite an identity manifest");
      }
    }
  }
  std::error_code error;
  llvm::raw_fd_ostream out(output, error);
  if (error)
    return fail("cannot open output '" + output + "': " + error.message());
  if (format == "json")
    out << llvm::formatv("{0:2}", llvm::json::Value(std::move(*report)))
        << '\n';
  else
    printText(*report, out);
  out.flush();
  if (out.has_error()) {
    auto message = out.error().message();
    out.clear_error();
    return fail("cannot write report: " + message);
  }
  return llvm::Error::success();
}
} // namespace

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv,
                              "Uncalibrated held-out prediction validation\n");
  if (auto error = run()) {
    llvm::errs() << llvm::toString(std::move(error)) << '\n';
    return 1;
  }
  return 0;
}
