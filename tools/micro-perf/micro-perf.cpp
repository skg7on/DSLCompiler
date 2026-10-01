//===- micro-perf.cpp - Micro-IR performance simulator driver -------------===//
//
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
//
//===----------------------------------------------------------------------===//

#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Perf/MachineModelLoader.h"
#include "LLK/Perf/MicroPerfReport.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
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

  auto model = mlir::llk::perf::loadMachineModel(machinePath.getValue());
  if (!model)
    return reportError(toString(model.takeError()));

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::micro::MicroDialect>();

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

  auto report =
      mlir::llk::perf::analyzeKernel(*kernel, *model, analysisLevel.getValue());
  if (!report)
    return reportError(toString(report.takeError()));

  if (format == "yaml")
    mlir::llk::perf::printMicroPerfYaml(outs(), *report);
  else
    mlir::llk::perf::printMicroPerfText(outs(), *report);

  if (failOnCapacityViolation && !report->capacityViolations.empty())
    return 2;
  return 0;
}
