//===- micro-perf.cpp - Micro-IR performance simulator driver -------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Reads concrete tile-centric Micro IR and a machine model, and reports what
// the kernel would cost on that machine.
//
// The MVP is a performance simulator, not a functional emulator: it never
// executes tensor math and never produces a tensor. Writing "emulator" anywhere
// in this tool would invite exactly that confusion, so the help text says what
// this is and nothing else in the file uses the word.
//
// Exit codes:
//   0  the kernel was analyzed
//   1  the input, machine model, or kernel selection could not be read
//   2  --fail-on-capacity-violation was given and the kernel does not fit

#include "LLK/Conversion/MicroMapping/ShapeSpecialization.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Machine/MachineModelLoader.h"
#include "LLK/Perf/MicroPerfReport.h"
#include "LLK/Shape/ShapeContract.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace cl = llvm::cl;

static cl::opt<std::string> machinePath(
    "machine",
    cl::desc("Machine model to evaluate the kernel against (machines/*.yaml)"),
    cl::value_desc("path"), cl::Required);

static cl::opt<unsigned> analysisLevel(
    "level",
    cl::desc("Analysis depth: 0 static bound only, 1 also schedule resources"),
    cl::init(1));

static cl::opt<std::string>
    outputFormat("format", cl::desc("Output format: yaml or text"),
                 cl::init("yaml"));

static cl::opt<std::string> kernelSymbol(
    "kernel",
    cl::desc("Name of the micro.kernel to analyze; required when the input "
             "holds more than one"),
    cl::init(""));

static cl::opt<bool> failOnCapacityViolation(
    "fail-on-capacity-violation",
    cl::desc("Exit with status 2 when live tiles exceed machine capacity"),
    cl::init(false));

static cl::opt<std::string> shapeValuesOption(
    "shape-values",
    cl::desc("Concrete symbol values for a dynamic kernel, e.g. "
             "M=7,N=13,K=5; symbol names come from the kernel's workload "
             "contract"),
    cl::init(""), cl::value_desc("SYM=V,..."));

static cl::opt<std::string>
    inputFile(cl::Positional, cl::desc("<input micro IR>"), cl::Required);

namespace {

constexpr const char *kDescription =
    "micro-perf - estimate what a concrete micro kernel costs on a machine.\n"
    "\n"
    "This is a performance simulator, not a functional emulator: it reads\n"
    "concrete tile-centric micro IR and reports cycles, resource utilization,\n"
    "bandwidth pressure, and bottleneck. It never executes tensor math.\n";

int reportError(const Twine &message) {
  errs() << "micro-perf: error: " << message << "\n";
  return 1;
}

llvm::Expected<mlir::llk::shape::ShapeValues>
parseShapeValues(llvm::StringRef text) {
  mlir::llk::shape::ShapeValues values;
  llvm::SmallVector<llvm::StringRef> entries;
  text.split(entries, ',');
  for (llvm::StringRef entry : entries) {
    auto [name, value] = entry.split('=');
    if (name.empty() || value.empty())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "malformed --shape-values entry '" +
                                         entry.str() +
                                         "'; expected SYMBOL=VALUE");
    int64_t parsed = 0;
    if (value.getAsInteger(10, parsed))
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "shape value for '" + name.str() +
                                         "' is not an integer");
    if (!values.emplace(name.str(), parsed).second)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "shape symbol '" + name.str() +
                                         "' is bound more than once");
  }
  return values;
}

} // namespace

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, kDescription);

  if (analysisLevel.getValue() > 1)
    return reportError(
        "unsupported --level=" + Twine(analysisLevel.getValue()) +
        "; micro-perf knows 0 (static bound) and 1 (resource "
        "schedule)");
  const std::string format = outputFormat.getValue();
  if (format != "yaml" && format != "text")
    return reportError("unsupported --format=" + format +
                       "; expected yaml or text");

  auto model = ::mlir::llk::machine::loadMachineModel(machinePath.getValue());
  if (!model)
    return reportError(toString(model.takeError()));

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::micro::MicroDialect>();
  // The compiler emits a micro.kernel alongside the LLK operations it came
  // from, so the simulator registers the LLK dialect to read that output
  // directly.
  registry.insert<mlir::llk::LLKDialect>();

  mlir::MLIRContext context(registry);
  mlir::ParserConfig parserConfig(&context);
  auto module =
      mlir::parseSourceFile<mlir::ModuleOp>(inputFile.getValue(), parserConfig);
  if (!module)
    return 1;

  auto kernel =
      mlir::llk::perf::findMicroKernel(module.get(), kernelSymbol.getValue());
  if (!kernel)
    return reportError(toString(kernel.takeError()));

  auto print = [&](const mlir::llk::perf::MicroPerfReport &report) {
    if (format == "yaml")
      mlir::llk::perf::printMicroPerfYaml(outs(), report);
    else
      mlir::llk::perf::printMicroPerfText(outs(), report);
  };

  if (shapeValuesOption.empty()) {
    auto report = mlir::llk::perf::analyzeKernel(*kernel, *model,
                                                 analysisLevel.getValue());
    if (!report)
      return reportError(toString(report.takeError()));
    print(*report);
    if (failOnCapacityViolation && !report->capacityViolations.empty())
      return 2;
    return 0;
  }

  // A sampled analysis needs the kernel's symbolic contract: the symbol names
  // the caller supplies are the names that contract declares, never a
  // positional guess from the signature.
  mlir::llk::shape::ShapeContract contract;
  {
    auto described = mlir::llk::mapping::shapeContractOf(*kernel);
    if (!described)
      return reportError(toString(described.takeError()));
    contract = std::move(*described);
  }

  auto requested = parseShapeValues(shapeValuesOption);
  if (!requested)
    return reportError(toString(requested.takeError()));

  mlir::llk::perf::MicroPerfReport sampled;
  sampled.machine = model->target;
  sampled.kernel =
      kernelSymbol.empty() ? std::string() : kernelSymbol.getValue();
  if (auto name = (*kernel)->getAttrOfType<mlir::StringAttr>("sym_name"))
    sampled.kernel = name.getValue().str();
  sampled.level = analysisLevel.getValue();
  sampled.clockHz = model->clockHz ? *model->clockHz : 0;
  sampled.sampled = true;
  sampled.shapeValues = *requested;

  // Missing or inconsistent bindings are reported as an incomplete analysis,
  // never as a zero-cost one-iteration estimate. The output deliberately has
  // no score block at all: a zero would be a rankable-looking number.
  if (llvm::Error error =
          mlir::llk::shape::validateShapeValues(contract, *requested)) {
    const std::string reason = "shape_values: " + toString(std::move(error));
    if (format == "yaml") {
      outs() << "schema_version: 1\n"
             << "machine: " << sampled.machine << "\n"
             << "kernel: " << sampled.kernel << "\n"
             << "level: " << sampled.level << "\n"
             << "complete: false\n"
             << "incomplete_reasons:\n"
             << "  - \"" << reason << "\"\n"
             << "sampled: true\n"
             << "shape_values:\n";
      for (const auto &entry : *requested)
        outs() << "  " << entry.first << ": " << entry.second << "\n";
    } else {
      outs() << "kernel: " << sampled.kernel << "\n"
             << "machine: " << sampled.machine << "\n"
             << "complete: false\n"
             << "incomplete reason: " << reason << "\n"
             << "shape sample:";
      for (const auto &entry : *requested)
        outs() << " " << entry.first << "=" << entry.second;
      outs() << "\n";
    }
    return 0;
  }

  auto clone = mlir::llk::mapping::materializeShapeSample(module.get(),
                                                          contract, *requested);
  if (!clone)
    return reportError(toString(clone.takeError()));
  auto sampledKernel =
      mlir::llk::perf::findMicroKernel(clone->get(), kernelSymbol.getValue());
  if (!sampledKernel)
    return reportError(toString(sampledKernel.takeError()));

  auto report = mlir::llk::perf::analyzeKernel(*sampledKernel, *model,
                                               analysisLevel.getValue());
  if (!report)
    return reportError(toString(report.takeError()));
  report->sampled = true;
  report->shapeValues = *requested;
  print(*report);

  if (failOnCapacityViolation && !report->capacityViolations.empty())
    return 2;
  return 0;
}
