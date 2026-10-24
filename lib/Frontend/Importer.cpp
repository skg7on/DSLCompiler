//===- Importer.cpp - Proven structured recovery of frontend programs -----===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "LLK/Frontend/Importer.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include <algorithm>
#include <array>
#include <limits>
#include <map>
#include <optional>

namespace llk::frontend {
namespace {
using Shape = std::vector<int64_t>;
/// Variables are program_id[0..2] followed by the value's two lane axes.
struct Affine {
  std::array<int64_t, 5> coefficients{};
  int64_t constant = 0;
  bool operator==(const Affine &) const = default;
};
struct Address {
  size_t port = 0;
  Affine rows;
  Affine columns;
};
struct Facts {
  FrontendValue value;
  std::optional<Affine> index;
  std::optional<bool> mask;
  std::optional<Address> address;
  bool load = false;
  std::string binary;
  std::vector<std::string> operands;
};
struct Recovery {
  std::array<size_t, 2> inputs{};
  size_t output = 0;
  std::array<unsigned, 2> scalarInputs{};
  bool multiply = false;
  SourceSpan span;
};
bool add(int64_t a, int64_t b, int64_t &result) {
  return !__builtin_add_overflow(a, b, &result);
}
bool multiply(int64_t a, int64_t b, int64_t &result) {
  return !__builtin_mul_overflow(a, b, &result);
}
std::optional<Affine> scale(Affine expression, int64_t factor) {
  if (!multiply(expression.constant, factor, expression.constant))
    return {};
  for (auto &coefficient : expression.coefficients)
    if (!multiply(coefficient, factor, coefficient))
      return {};
  return expression;
}
std::optional<Affine> plus(Affine a, const Affine &b) {
  if (!add(a.constant, b.constant, a.constant))
    return {};
  for (size_t i = 0; i != 5; ++i)
    if (!add(a.coefficients[i], b.coefficients[i], a.coefficients[i]))
      return {};
  return a;
}
bool uniform(const Affine &value) {
  return llvm::all_of(value.coefficients, [](int64_t n) { return n == 0; });
}
class Proof {
public:
  explicit Proof(const FrontendProgram &program) : program(program) {}
  llvm::Expected<Recovery> run() {
    const auto &n = program.numericalContract;
    if (n.storageDtype != "f32" || n.computeDtype != "f32" ||
        n.accumulatorDtype != "f32" || !n.castSites.empty() ||
        n.rounding != "rtne" || n.activationExpression != "none" ||
        n.activationApproximation != "none" || n.reductionOrder != "source" ||
        n.reassociation || n.fma || n.subnormal != "preserve")
      return frontendError(
          "unsupported_numerical_contract",
          "elementwise import requires the complete strict-f32 policy", {},
          "import");
    Recovery recovered;
    size_t inputs = 0, outputs = 0;
    for (size_t i = 0; i < program.tensors.size(); ++i) {
      const auto &t = program.tensors[i];
      Facts fact;
      fact.value = {t.name, "ptr<" + t.dtype + ">", {}};
      fact.address = Address{i, {}, {}};
      values[t.name] = std::move(fact);
      if (t.direction == "input") {
        if (inputs < 2)
          recovered.inputs[inputs] = i;
        ++inputs;
      } else {
        recovered.output = i;
        ++outputs;
      }
    }
    if (inputs != 2 || outputs != 1)
      return unsupported(
          "exactly two read inputs and one write output are supported");
    const auto &output = program.tensors[recovered.output];
    for (const auto &tensor : program.tensors)
      if (tensor.shape != output.shape || tensor.dtype != "f32")
        return unsupported("all ports must have the same rank-two f32 shape");
    if (program.launch.grid[2] != 1 || program.launch.block[2] != 1 ||
        !productEquals(program.launch.grid[0], program.launch.block[0],
                       output.shape[0]) ||
        !productEquals(program.launch.grid[1], program.launch.block[1],
                       output.shape[1]))
      return coverage("grid/block does not cover the declared domain exactly");
    unsigned stores = 0;
    for (const auto &op : program.operations) {
      const auto &code = op.opcode;
      if (code == "return")
        continue;
      std::vector<const Facts *> operands;
      for (const auto &name : op.operands)
        operands.push_back(&values.at(name));
      if (code == "store") {
        if (++stores != 1)
          return coverage("multiple output writers", op.span);
        const auto &pointer = *operands[0];
        const auto &written = *operands[1];
        if (!pointer.address || pointer.address->port != recovered.output ||
            !active(*operands[2]) ||
            !rectangular(*pointer.address, pointer.value.shape))
          return coverage(
              "output address/mask is not an exact rectangular bijection",
              op.span);
        if (written.binary != "add" && written.binary != "mul")
          return unsupported("output must be exactly one add/multiply",
                             op.span);
        for (unsigned i = 0; i != 2; ++i) {
          const auto &loaded = values.at(written.operands[i]);
          if (!loaded.load || !loaded.address ||
              loaded.value.shape != pointer.value.shape ||
              loaded.address->rows != pointer.address->rows ||
              loaded.address->columns != pointer.address->columns)
            return coverage("read coordinates do not match the output element",
                            op.span);
          auto port = loaded.address->port;
          if (port == recovered.inputs[0])
            recovered.scalarInputs[i] = 0;
          else if (port == recovered.inputs[1])
            recovered.scalarInputs[i] = 1;
          else
            return unsupported("arithmetic must read the declared inputs",
                               op.span);
        }
        if (recovered.scalarInputs[0] == recovered.scalarInputs[1])
          return unsupported("arithmetic must use both declared read inputs",
                             op.span);
        recovered.multiply = written.binary == "mul";
        recovered.span = op.span;
        continue;
      }
      Facts result;
      if (op.results.empty())
        return unsupported(
            "region/control-flow opcode is outside this importer", op.span);
      result.value = op.results.front();
      if (code == "constant") {
        if (result.value.dtype == "index")
          result.index =
              Affine{{}, op.attributes.getInteger("value").value_or(0)};
        else if (result.value.dtype == "i1")
          result.mask = op.attributes.getBoolean("value");
      } else if (code == "program_id") {
        Affine index;
        index.coefficients[op.attributes.getInteger("axis").value_or(0)] = 1;
        result.index = index;
      } else if (code == "range") {
        Affine index;
        index.constant = op.attributes.getInteger("start").value_or(0);
        index.coefficients[3] = 1;
        result.index = index;
      } else if (code == "expand_dims" || code == "broadcast") {
        result.mask = operands[0]->mask;
        if (operands[0]->index) {
          auto index = *operands[0]->index;
          index.coefficients[3] = index.coefficients[4] = 0;
          for (size_t axis = 0; axis < operands[0]->value.shape.size();
               ++axis) {
            int64_t mapped =
                code == "expand_dims"
                    ? int64_t(axis) +
                          (axis >=
                           uint64_t(
                               op.attributes.getInteger("axis").value_or(0)))
                    : (*op.attributes.getArray("dimensions"))[axis]
                          .getAsInteger()
                          .value_or(0);
            index.coefficients[3 + mapped] =
                operands[0]->index->coefficients[3 + axis];
          }
          result.index = index;
        } else if (result.value.dtype != "i1")
          return unsupported("only index/mask views are supported", op.span);
      } else if (code == "add" || code == "sub" || code == "mul" ||
                 code == "div" || code == "rem" || code == "ceildiv") {
        if (result.value.dtype == "index") {
          if (!operands[0]->index || !operands[1]->index)
            return coverage("non-affine index operand", op.span);
          auto a = promote(*operands[0]->index, operands[0]->value.shape,
                           result.value.shape);
          auto b = promote(*operands[1]->index, operands[1]->value.shape,
                           result.value.shape);
          if (code == "add")
            result.index = plus(a, b);
          else if (code == "sub") {
            auto negated = scale(b, -1);
            if (negated)
              result.index = plus(a, *negated);
          } else if (code == "mul") {
            if (uniform(a))
              result.index = scale(b, a.constant);
            else if (uniform(b))
              result.index = scale(a, b.constant);
          } else if (uniform(b) && b.constant > 0) {
            if (code == "ceildiv") {
              auto bounds = interval(a, result.value.shape);
              if (bounds && bounds->first >= 0) {
                auto shifted = plus(a, Affine{{}, b.constant - 1});
                if (shifted)
                  result.index =
                      divide(*shifted, b.constant, result.value.shape, false);
              }
            } else
              result.index =
                  divide(a, b.constant, result.value.shape, code == "rem");
          }
          if (!result.index)
            return coverage(
                "index expression is not a proven affine map or overflows",
                op.span);
        } else if (result.value.dtype == "f32" &&
                   (code == "add" || code == "mul")) {
          result.binary = code;
          result.operands = op.operands;
        } else
          return unsupported("unsupported scalar computation", op.span);
      } else if (code == "compare") {
        if (!operands[0]->index || !operands[1]->index)
          return unsupported("only affine-index comparisons are supported",
                             op.span);
        auto a = promote(*operands[0]->index, operands[0]->value.shape,
                         result.value.shape);
        auto b = promote(*operands[1]->index, operands[1]->value.shape,
                         result.value.shape);
        auto negative = scale(b, -1);
        auto difference = negative ? plus(a, *negative) : std::nullopt;
        auto bounds = difference ? interval(*difference, result.value.shape)
                                 : std::nullopt;
        if (!bounds)
          return coverage("comparison bounds overflow", op.span);
        auto [lo, hi] = *bounds;
        auto predicate = op.attributes.getString("predicate").value_or("");
        if (predicate == "lt") {
          if (hi < 0)
            result.mask = true;
          else if (lo >= 0)
            result.mask = false;
        }
        if (predicate == "le") {
          if (hi <= 0)
            result.mask = true;
          else if (lo > 0)
            result.mask = false;
        }
        if (predicate == "gt") {
          if (lo > 0)
            result.mask = true;
          else if (hi <= 0)
            result.mask = false;
        }
        if (predicate == "ge") {
          if (lo >= 0)
            result.mask = true;
          else if (hi < 0)
            result.mask = false;
        }
        if (predicate == "eq") {
          if (lo == 0 && hi == 0)
            result.mask = true;
          else if (hi < 0 || lo > 0)
            result.mask = false;
        }
        if (predicate == "ne") {
          if (lo == 0 && hi == 0)
            result.mask = false;
          else if (hi < 0 || lo > 0)
            result.mask = true;
        }
      } else if (code == "select") {
        if (result.value.dtype != "i1")
          return unsupported(
              "data selects are outside the add/multiply importer", op.span);
        if (operands[0]->mask)
          result.mask = operands[*operands[0]->mask ? 1 : 2]->mask;
        else if (operands[1]->mask == operands[2]->mask)
          result.mask = operands[1]->mask;
      } else if (code == "address") {
        if (!operands[0]->address || !operands[1]->index || !operands[2]->index)
          return coverage("address is not affine", op.span);
        result.address =
            Address{operands[0]->address->port,
                    promote(*operands[1]->index, operands[1]->value.shape,
                            result.value.shape),
                    promote(*operands[2]->index, operands[2]->value.shape,
                            result.value.shape)};
      } else if (code == "load") {
        const auto &pointer = *operands[0];
        if (!pointer.address || !active(*operands[1]) ||
            !inBounds(*pointer.address, pointer.value.shape))
          return coverage("load is not proven masked/in-bounds", op.span);
        result.load = true;
        result.address = pointer.address;
      } else
        return unsupported(
            "opcode is outside the strict add/multiply importer: " + code,
            op.span);
      if (result.index) {
        *result.index = normalize(*result.index, result.value.shape);
        if (!interval(*result.index, result.value.shape))
          return coverage("index intermediate overflows signed 64-bit range",
                          op.span);
      }
      values.emplace(result.value.id, std::move(result));
    }
    if (stores != 1)
      return coverage("one complete output write is required");
    return recovered;
  }

private:
  const FrontendProgram &program;
  std::map<std::string, Facts> values;
  static bool productEquals(int64_t a, int64_t b, int64_t expected) {
    int64_t result;
    return multiply(a, b, result) && result == expected;
  }
  llvm::Error coverage(llvm::Twine message,
                       std::optional<SourceSpan> span = {}) {
    return frontendError("invalid_launch_coverage", message, span, "import");
  }
  llvm::Error unsupported(llvm::Twine message,
                          std::optional<SourceSpan> span = {}) {
    return frontendError("unsupported_frontend_program", message, span,
                         "import");
  }
  static bool active(const Facts &fact) { return fact.mask.value_or(false); }
  Affine normalize(Affine expression, const Shape &shape) const {
    for (size_t i = 0; i != 3; ++i)
      if (program.launch.grid[i] == 1)
        expression.coefficients[i] = 0;
    for (size_t i = 0; i != 2; ++i)
      if (i >= shape.size() || shape[i] == 1)
        expression.coefficients[3 + i] = 0;
    return expression;
  }
  Affine promote(Affine expression, const Shape &from, const Shape &to) const {
    Affine result = expression;
    result.coefficients[3] = result.coefficients[4] = 0;
    size_t shift = to.size() - from.size();
    for (size_t i = 0; i < from.size(); ++i)
      result.coefficients[3 + shift + i] = expression.coefficients[3 + i];
    return normalize(result, to);
  }
  std::optional<std::pair<int64_t, int64_t>>
  interval(const Affine &expression, const Shape &shape) const {
    int64_t lo = expression.constant, hi = expression.constant;
    for (size_t i = 0; i != 5; ++i) {
      int64_t extent = i < 3                  ? program.launch.grid[i]
                       : i - 3 < shape.size() ? shape[i - 3]
                                              : 1;
      int64_t term;
      if (!multiply(expression.coefficients[i], extent - 1, term))
        return {};
      if (term < 0) {
        if (!add(lo, term, lo))
          return {};
      } else if (!add(hi, term, hi))
        return {};
    }
    return std::pair<int64_t, int64_t>{lo, hi};
  }
  std::optional<Affine> divide(const Affine &input, int64_t divisor,
                               const Shape &shape, bool remainder) const {
    Affine quotient, residue;
    quotient.constant = input.constant / divisor;
    residue.constant = input.constant % divisor;
    for (size_t i = 0; i != 5; ++i) {
      quotient.coefficients[i] = input.coefficients[i] / divisor;
      residue.coefficients[i] = input.coefficients[i] % divisor;
    }
    auto bounds = interval(residue, shape);
    if (!bounds || bounds->first < 0 || bounds->second >= divisor)
      return {};
    return normalize(remainder ? residue : quotient, shape);
  }
  bool inBounds(const Address &address, const Shape &shape) const {
    auto rows = interval(address.rows, shape),
         cols = interval(address.columns, shape);
    const auto &tensor = program.tensors[address.port];
    return rows && cols && rows->first >= 0 && cols->first >= 0 &&
           rows->second < tensor.shape[0] && cols->second < tensor.shape[1];
  }
  bool rectangular(const Address &address, const Shape &shape) const {
    if (shape != Shape{program.launch.block[0], program.launch.block[1]} ||
        !inBounds(address, shape))
      return false;
    Affine rows, cols;
    rows.coefficients[0] = shape[0];
    rows.coefficients[3] = 1;
    cols.coefficients[1] = shape[1];
    cols.coefficients[4] = 1;
    return address.rows == normalize(rows, shape) &&
           address.columns == normalize(cols, shape);
  }
};
} // namespace

llvm::Expected<ImportedProgram> importProgram(const FrontendRequest &request,
                                              mlir::MLIRContext &context) {
  auto proof = Proof(request.program()).run();
  if (!proof)
    return proof.takeError();
  if (proof->span.line > UINT32_MAX || proof->span.column > UINT32_MAX)
    return frontendError("invalid_frontend_schema",
                         "source location exceeds MLIR's coordinate range",
                         proof->span, "import");
  using namespace mlir;
  context.loadDialect<func::FuncDialect, arith::ArithDialect,
                      tensor::TensorDialect, linalg::LinalgDialect>();
  auto loc = FileLineColLoc::get(&context, proof->span.file,
                                 unsigned(proof->span.line),
                                 unsigned(proof->span.column));
  ImportedProgram imported;
  imported.module = ModuleOp::create(loc);
  OpBuilder builder(&context);
  builder.setInsertionPointToStart(imported.module->getBody());
  const auto &program = request.program();
  const auto &shape = program.tensors[proof->output].shape;
  auto type = RankedTensorType::get(shape, builder.getF32Type());
  auto function =
      func::FuncOp::create(builder, loc, program.entry,
                           builder.getFunctionType({type, type}, {type}));
  Block *body = function.addEntryBlock();
  builder.setInsertionPointToStart(body);
  auto destination =
      tensor::EmptyOp::create(builder, loc, shape, builder.getF32Type());
  auto identity = builder.getMultiDimIdentityMap(2);
  auto generic = linalg::GenericOp::create(
      builder, loc, TypeRange{type},
      ValueRange{body->getArgument(0), body->getArgument(1)},
      ValueRange{destination.getResult()},
      ArrayRef<AffineMap>{identity, identity, identity},
      ArrayRef<utils::IteratorType>{utils::IteratorType::parallel,
                                    utils::IteratorType::parallel},
      [&](OpBuilder &nested, Location nestedLoc, ValueRange args) {
        Value a = args[proof->scalarInputs[0]],
              b = args[proof->scalarInputs[1]];
        Value result =
            proof->multiply
                ? arith::MulFOp::create(nested, nestedLoc, a, b).getResult()
                : arith::AddFOp::create(nested, nestedLoc, a, b).getResult();
        linalg::YieldOp::create(nested, nestedLoc, result);
      });
  func::ReturnOp::create(builder, loc, generic.getResults());
  (*imported.module)
      ->setAttr("llk.frontend.request",
                builder.getStringAttr(request.canonicalJSON()));
  if (failed(verify(*imported.module)))
    return frontendError("invalid_frontend_schema",
                         "structured import failed MLIR verification",
                         proof->span, "import");
  imported.selection = {program.entry, 0};
  imported.adapterKey = "linalg-elementwise-v1";
  auto registry = mlir::llk::workload::makeDefaultSourceAdapters();
  const auto *adapter = registry.findByKey(imported.adapterKey);
  if (!adapter)
    return frontendError("unsupported_frontend_program",
                         "elementwise source adapter is not registered",
                         proof->span, "import");
  auto described = adapter->describe(generic.getOperation());
  if (!described)
    return frontendError("unsupported_frontend_program",
                         llvm::toString(described.takeError()), proof->span,
                         "import");
  imported.contract = std::move(*described);
  imported.space.name = "frontend_blocks";
  imported.space.workload = program.entry;
  for (unsigned axis = 0; axis != 2; ++axis) {
    mlir::llk::perf::SearchParam parameter;
    parameter.name = "frontend_block_" + std::to_string(axis);
    parameter.kind = "integer";
    parameter.role = "axis:d" + std::to_string(axis);
    parameter.choices.emplace_back(program.launch.block[axis]);
    imported.space.params.push_back(std::move(parameter));
  }
  for (size_t i = 0; i < program.tensors.size(); ++i) {
    unsigned nativePort = i == proof->output      ? 3
                          : i == proof->inputs[0] ? 0
                                                  : 1;
    imported.publicToNativePorts.push_back({unsigned(i), nativePort});
  }
  return imported;
}
} // namespace llk::frontend
