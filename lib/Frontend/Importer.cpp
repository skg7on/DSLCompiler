//===- Importer.cpp - Proven structured recovery of frontend programs -----===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "LLK/Frontend/Importer.h"
#include "LLK/Conversion/LLKToMicro/LLKToMicro.h"
#include "LLK/Conversion/Workload/ContractionSourceAdapter.h"
#include "LLK/Dialect/LLKDialect.h"
#include "LLK/Dialect/Micro/MicroDialect.h"
#include "LLK/Perf/SearchSpace.h"
#include "LLK/Transforms/Common/ScheduleLoader.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Verifier.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSet.h"
#include <array>
#include <functional>
#include <limits>
#include <map>
#include <optional>

// Generated LLK attribute and op class declarations. The importer builds the
// recovered root as a real `llk` operation, so it needs the op definitions.
#include "LLK/Dialect/LLKEnums.h" // IWYU pragma: keep
#define GET_ATTRDEF_CLASSES
#include "LLK/Dialect/LLKAttributes.h.inc"
#define GET_OP_CLASSES
#include "LLK/Dialect/LLKOps.h.inc"

namespace llk::frontend {
namespace {
using Shape = std::vector<int64_t>;
/// Variables are program_id[0..2] followed by the value's two lane axes.
struct Affine {
  std::array<int64_t, 5> coefficients{};
  int64_t constant = 0;
  bool operator==(const Affine &) const = default;
};
/// A linear residue that the caller's mask must bound: `expression < divisor`.
/// A `div`/`rem` whose residue can leave the divisor range is only the true
/// quotient/remainder on lanes a matching upper-bound selects. The requirement
/// travels with the index facts and is discharged at the load or store that
/// consumes the address.
struct ResidueRequirement {
  Affine expression;
  int64_t divisor = 0;
};
struct Address {
  size_t port = 0;
  Affine rows;
  Affine columns;
  llvm::SmallVector<ResidueRequirement, 2> requiresMaskBound{};
};
/// One in-bounds lane term: `program_id[axis] * block + lane < bound`.
struct LaneBound {
  Affine expression;
  int64_t bound = 0;
  bool operator==(const LaneBound &) const = default;
};
/// What is known about an i1 value. Constant facts are the whole-tile masks the
/// first increment accepted; `Bounds` is a conjunction of lane bounds, which is
/// exactly the tail predicate a rounded launch has to declare. Anything else is
/// unknown and never proves memory safety.
struct MaskFacts {
  enum class Kind { Unknown, Constant, Bounds };
  Kind kind = Kind::Unknown;
  bool constant = false;
  llvm::SmallVector<LaneBound, 2> bounds;
  bool operator==(const MaskFacts &) const = default;
};
struct Facts {
  FrontendValue value;
  std::optional<Affine> index;
  MaskFacts mask;
  std::optional<ResidueRequirement> requiresMaskBound;
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
  bool tail = false;
  int64_t block0 = 0;
  int64_t block1 = 0;
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
MaskFacts unknownMask() { return MaskFacts{}; }
MaskFacts constantMask(bool value) {
  MaskFacts mask;
  mask.kind = MaskFacts::Kind::Constant;
  mask.constant = value;
  return mask;
}
MaskFacts boundsMask(llvm::ArrayRef<LaneBound> bounds) {
  MaskFacts mask;
  mask.kind = MaskFacts::Kind::Bounds;
  mask.bounds.assign(bounds.begin(), bounds.end());
  return mask;
}
bool containsBound(llvm::ArrayRef<LaneBound> bounds, const LaneBound &term) {
  return llvm::any_of(bounds,
                      [&](const LaneBound &entry) { return entry == term; });
}
/// `a && b`. A false constant dominates, a true constant is neutral, and two
/// conjunctions conjoin. An unknown operand leaves the result unknown unless
/// the other side is already false.
MaskFacts andMasks(const MaskFacts &a, const MaskFacts &b) {
  auto isConstant = [](const MaskFacts &mask, bool value) {
    return mask.kind == MaskFacts::Kind::Constant && mask.constant == value;
  };
  if (isConstant(a, false) || isConstant(b, false))
    return constantMask(false);
  if (isConstant(a, true))
    return b;
  if (isConstant(b, true))
    return a;
  if (a.kind != MaskFacts::Kind::Bounds || b.kind != MaskFacts::Kind::Bounds)
    return unknownMask();
  MaskFacts result = a;
  for (const LaneBound &term : b.bounds)
    if (!containsBound(result.bounds, term))
      result.bounds.push_back(term);
  return result;
}
/// `a || b`. Only the cases the AST's `|` lowering can produce are exact: a
/// true constant dominates, a false constant is neutral, and a conjunction
/// already contained in the other is redundant. Anything else is unknown,
/// because a union of two different lane bounds is not a conjunction.
MaskFacts orMasks(const MaskFacts &a, const MaskFacts &b) {
  auto isConstant = [](const MaskFacts &mask, bool value) {
    return mask.kind == MaskFacts::Kind::Constant && mask.constant == value;
  };
  if (isConstant(a, true) || isConstant(b, true))
    return constantMask(true);
  if (isConstant(a, false))
    return b;
  if (isConstant(b, false))
    return a;
  if (a.kind != MaskFacts::Kind::Bounds || b.kind != MaskFacts::Kind::Bounds)
    return unknownMask();
  auto subset = [](const MaskFacts &small, const MaskFacts &large) {
    return llvm::all_of(small.bounds, [&](const LaneBound &term) {
      return containsBound(large.bounds, term);
    });
  };
  if (subset(a, b))
    return b;
  if (subset(b, a))
    return a;
  return unknownMask();
}
/// The condition-selected mask of `cond ? x : y`.
MaskFacts selectMask(const MaskFacts &condition, const MaskFacts &x,
                     const MaskFacts &y) {
  if (condition.kind == MaskFacts::Kind::Constant)
    return condition.constant ? x : y;
  const auto isConstant = [](const MaskFacts &mask, bool value) {
    return mask.kind == MaskFacts::Kind::Constant && mask.constant == value;
  };
  // `a & b` lowers to select(a, b, false); `a | b` to select(a, true, b).
  if (isConstant(y, false))
    return andMasks(condition, x);
  if (isConstant(x, true))
    return orMasks(condition, y);
  if (x.kind != MaskFacts::Kind::Unknown && x == y)
    return x;
  return unknownMask();
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
      fact.address = Address{i, {}, {}, {}};
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
    if (program.launch.grid[2] != 1 || program.launch.block[2] != 1)
      return coverage("grid z and block z must both be one");
    // The grid must be exactly ceil(extent / block) on each axis. A divisible
    // axis is the original exact-coverage contract; a rounded axis needs a
    // declared tail mask, which the load/store proofs below discharge.
    for (unsigned axis = 0; axis != 2; ++axis) {
      const int64_t extent = output.shape[axis];
      const int64_t block = program.launch.block[axis];
      if (block <= 0)
        return coverage("block extents must be positive");
      const int64_t whole = extent / block + (extent % block != 0 ? 1 : 0);
      int64_t span = 0;
      if (program.launch.grid[axis] != whole ||
          !multiply(program.launch.grid[axis], block, span) || span < extent)
        return coverage("grid/block does not cover the declared domain with "
                        "whole blocks");
      if (span != extent)
        tailRequired = true;
    }
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
            !rectangular(*pointer.address, pointer.value.shape))
          return coverage(
              "output address is not an exact rectangular bijection", op.span);
        if (!maskMatchesExpected(operands[2]->mask))
          return tailRequired
                     ? tailUnsupported(
                           "the store mask does not prove the rounded domain's "
                           "in-bounds lanes",
                           op.span)
                     : coverage("the store mask must be true on every lane",
                                op.span);
        if (!requirementsSatisfied(*pointer.address, operands[2]->mask))
          return tailRequired
                     ? tailUnsupported(
                           "the store mask does not bound the decomposed "
                           "row-major index",
                           op.span)
                     : coverage("the store address is not proven in bounds",
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
          result.mask =
              constantMask(op.attributes.getBoolean("value").value_or(false));
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
            } else {
              bool provisional = false;
              std::optional<Affine> residue;
              result.index = divide(a, b.constant, result.value.shape,
                                    code == "rem", &provisional, &residue);
              if (result.index && provisional)
                result.requiresMaskBound =
                    ResidueRequirement{*residue, b.constant};
            }
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
        std::optional<bool> constant;
        if (predicate == "lt") {
          if (hi < 0)
            constant = true;
          else if (lo >= 0)
            constant = false;
        } else if (predicate == "le") {
          if (hi <= 0)
            constant = true;
          else if (lo > 0)
            constant = false;
        } else if (predicate == "gt") {
          if (lo > 0)
            constant = true;
          else if (hi <= 0)
            constant = false;
        } else if (predicate == "ge") {
          if (lo >= 0)
            constant = true;
          else if (hi < 0)
            constant = false;
        } else if (predicate == "eq") {
          if (lo == 0 && hi == 0)
            constant = true;
          else if (hi < 0 || lo > 0)
            constant = false;
        } else if (predicate == "ne") {
          if (lo == 0 && hi == 0)
            constant = false;
          else if (hi < 0 || lo > 0)
            constant = true;
        }
        if (constant)
          result.mask = constantMask(*constant);
        else if (auto term =
                     laneBound(*difference, result.value.shape, predicate))
          result.mask = boundsMask({*term});
      } else if (code == "select") {
        if (result.value.dtype != "i1")
          return unsupported(
              "data selects are outside the add/multiply importer", op.span);
        result.mask =
            selectMask(operands[0]->mask, operands[1]->mask, operands[2]->mask);
      } else if (code == "address") {
        if (!operands[0]->address || !operands[1]->index || !operands[2]->index)
          return coverage("address is not affine", op.span);
        Address address{operands[0]->address->port,
                        promote(*operands[1]->index, operands[1]->value.shape,
                                result.value.shape),
                        promote(*operands[2]->index, operands[2]->value.shape,
                                result.value.shape),
                        {}};
        for (const Facts *operand : {operands[1], operands[2]})
          if (operand->requiresMaskBound)
            address.requiresMaskBound.push_back(
                {promote(operand->requiresMaskBound->expression,
                         operand->value.shape, result.value.shape),
                 operand->requiresMaskBound->divisor});
        result.address = std::move(address);
      } else if (code == "load") {
        const auto &pointer = *operands[0];
        if (!pointer.address)
          return coverage("load has no affine address", op.span);
        if (!rectangular(*pointer.address, pointer.value.shape))
          return coverage("load address is not an exact rectangular "
                          "bijection",
                          op.span);
        if (!maskMatchesExpected(operands[1]->mask))
          return tailRequired
                     ? tailUnsupported(
                           "the load mask does not prove the rounded "
                           "domain's in-bounds lanes",
                           op.span)
                     : coverage("the load mask must be true on every lane",
                                op.span);
        if (!requirementsSatisfied(*pointer.address, operands[1]->mask))
          return tailRequired
                     ? tailUnsupported(
                           "the load mask does not bound the decomposed "
                           "row-major index",
                           op.span)
                     : coverage("the load address is not proven in bounds",
                                op.span);
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
    recovered.tail = tailRequired;
    recovered.block0 = program.launch.block[0];
    recovered.block1 = program.launch.block[1];
    return recovered;
  }

private:
  const FrontendProgram &program;
  std::map<std::string, Facts> values;
  /// True when at least one axis rounds the domain up, so a declared mask is
  /// required. Set before the operation loop; the load/store proofs read it.
  bool tailRequired = false;
  llvm::Error coverage(llvm::Twine message,
                       std::optional<SourceSpan> span = {}) {
    return frontendError("invalid_launch_coverage", message, span, "import");
  }
  llvm::Error unsupported(llvm::Twine message,
                          std::optional<SourceSpan> span = {}) {
    return frontendError("unsupported_frontend_program", message, span,
                         "import");
  }
  llvm::Error tailUnsupported(llvm::Twine message,
                              std::optional<SourceSpan> span = {}) {
    return frontendError("unsupported_tail_policy", message, span, "import");
  }
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
  /// Recognize `difference < 0` (or `<= 0`) as "one lane expression is below a
  /// positive bound". The expression must be exactly
  /// `program_id[axis] * block + lane` on its own axis, so the term means the
  /// same thing as the expected in-bounds predicate -- not merely something
  /// that happens to be true for this program.
  std::optional<LaneBound> laneBound(const Affine &difference,
                                     const Shape &shape,
                                     llvm::StringRef predicate) const {
    int64_t bound = 0;
    if (predicate == "lt") {
      if (difference.constant >= 0 ||
          difference.constant == std::numeric_limits<int64_t>::min())
        return {};
      bound = -difference.constant;
    } else if (predicate == "le") {
      if (difference.constant == std::numeric_limits<int64_t>::min())
        return {};
      bound = 1 - difference.constant;
    } else {
      return {};
    }
    if (bound <= 0)
      return {};
    // The lane coefficient identifies the axis. The program-id coefficient was
    // normalized to zero for a one-block axis before this point, so it is
    // checked against the form normalization leaves rather than the raw block.
    int axis = -1;
    for (unsigned lane = 0; lane != 2; ++lane) {
      if (difference.coefficients[3 + lane] == 0)
        continue;
      if (axis >= 0 || difference.coefficients[3 + lane] != 1)
        return {};
      axis = static_cast<int>(lane);
    }
    if (axis < 0)
      return {};
    for (unsigned program = 0; program != 3; ++program) {
      if (program == static_cast<unsigned>(axis))
        continue;
      if (difference.coefficients[program] != 0)
        return {};
    }
    const int64_t expectedProgram =
        program.launch.grid[axis] == 1 ? 0 : program.launch.block[axis];
    if (difference.coefficients[axis] != expectedProgram)
      return {};
    Affine expression = difference;
    expression.constant = 0;
    return LaneBound{normalize(expression, shape), bound};
  }
  std::optional<Affine>
  divide(const Affine &input, int64_t divisor, const Shape &shape,
         bool remainder, bool *provisional = nullptr,
         std::optional<Affine> *residueExpression = nullptr) const {
    if (provisional)
      *provisional = false;
    if (divisor <= 0)
      return {};
    Affine quotient, residue;
    quotient.constant = input.constant / divisor;
    residue.constant = input.constant % divisor;
    for (size_t i = 0; i != 5; ++i) {
      quotient.coefficients[i] = input.coefficients[i] / divisor;
      residue.coefficients[i] = input.coefficients[i] % divisor;
    }
    auto bounds = interval(residue, shape);
    if (!bounds || bounds->first < 0)
      return {};
    // A residue that can leave [0, divisor) is the true remainder only on
    // lanes an upper-bound mask selects. Keep the coefficient-wise answer and
    // record the bound the consuming load/store must prove. The mask terms the
    // tail contract accepts are exactly such bounds.
    if (bounds->second >= divisor) {
      if (residueExpression)
        *residueExpression = normalize(residue, shape);
      if (provisional)
        *provisional = true;
    }
    return normalize(remainder ? residue : quotient, shape);
  }
  /// The mask a rounded launch must declare: for every axis whose blocks leave
  /// a remainder, `program_id[axis] * block + lane < extent`. A divisible axis
  /// contributes a constant-true term and disappears.
  MaskFacts expectedMask() const {
    const auto &shape = program.tensors.front().shape;
    const Shape blockShape{program.launch.block[0], program.launch.block[1]};
    MaskFacts expected = constantMask(true);
    for (unsigned axis = 0; axis != 2; ++axis) {
      int64_t span = 0;
      if (!multiply(program.launch.grid[axis], program.launch.block[axis],
                    span) ||
          span == shape[axis])
        continue;
      Affine lane;
      lane.coefficients[axis] = program.launch.block[axis];
      lane.coefficients[3 + axis] = 1;
      expected = andMasks(
          expected,
          boundsMask({LaneBound{normalize(lane, blockShape), shape[axis]}}));
    }
    return expected;
  }
  bool maskMatchesExpected(const MaskFacts &mask) const {
    MaskFacts expected = expectedMask();
    if (mask.kind == MaskFacts::Kind::Unknown ||
        expected.kind == MaskFacts::Kind::Unknown)
      return false;
    if (mask.kind != expected.kind)
      return false;
    if (mask.kind == MaskFacts::Kind::Constant)
      return mask.constant == expected.constant;
    if (mask.bounds.size() != expected.bounds.size())
      return false;
    return llvm::all_of(mask.bounds, [&](const LaneBound &term) {
      return containsBound(expected.bounds, term);
    });
  }
  /// Discharge every decomposed row-major index: the mask must contain the
  /// exact term `residue < divisor` the provisional division recorded.
  bool requirementsSatisfied(const Address &address,
                             const MaskFacts &mask) const {
    for (const ResidueRequirement &requirement : address.requiresMaskBound) {
      bool proven = llvm::any_of(mask.bounds, [&](const LaneBound &term) {
        return term.expression == requirement.expression &&
               term.bound == requirement.divisor;
      });
      if (!proven)
        return false;
    }
    return true;
  }
  bool rectangular(const Address &address, const Shape &shape) const {
    if (shape != Shape{program.launch.block[0], program.launch.block[1]})
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

//===----------------------------------------------------------------------===//
// Counted contractions and guarded SwiGLU recovery
//===----------------------------------------------------------------------===//
//
// PF11. A captured kernel expresses a contraction as a counted K loop whose
// body loads one A tile and one or two B tiles and accumulates one or two dot
// results. That is a *tile* program; the native contraction operators take
// whole tensors. The recovery below therefore proves that the tile program is
// exactly the whole-tensor contraction:
//
//   * the launch tiles the output domain exactly (no tails in this family),
//   * the counted loop traverses exactly the declared K extent in whole BK
//     blocks,
//   * every load reads the compact row-major tensor it names at the canonical
//     `row = pid*block + lane`, `col = k + lane` coordinates,
//   * the accumulators are explicit loop-carried values seeded either by a
//     zero tile or by a caller tensor, and
//   * the two-projection form shares its left operand and applies exactly
//     `silu(gate) * up`.
//
// Anything else is rejected by name. A pattern that is rejected is never
// approximated.

/// One scalar index expression: `sum(coefficient * symbol) + constant` over
/// program ids, the two tile lanes and the counted loop's induction variable.
struct ContractionIndex {
  std::array<int64_t, 6> coefficients{};
  int64_t constant = 0;
  bool operator==(const ContractionIndex &) const = default;
};
constexpr size_t kPid0 = 0;
constexpr size_t kPid1 = 1;
constexpr size_t kLane0 = 3;
constexpr size_t kLane1 = 4;
constexpr size_t kInduction = 5;

bool uniformIndex(const ContractionIndex &value) {
  return llvm::all_of(value.coefficients, [](int64_t n) { return n == 0; });
}
std::optional<ContractionIndex> addIndex(ContractionIndex a,
                                         const ContractionIndex &b) {
  if (!add(a.constant, b.constant, a.constant))
    return {};
  for (size_t i = 0; i != a.coefficients.size(); ++i)
    if (!add(a.coefficients[i], b.coefficients[i], a.coefficients[i]))
      return {};
  return a;
}
std::optional<ContractionIndex> scaleIndex(ContractionIndex a, int64_t factor) {
  if (!multiply(a.constant, factor, a.constant))
    return {};
  for (int64_t &coefficient : a.coefficients)
    if (!multiply(coefficient, factor, coefficient))
      return {};
  return a;
}

/// The counted loop a contraction is carried by. The canonical family has one
/// positive-step loop whose bounds are exact constants and whose iteration
/// count is integral.
struct CountedKernelLoop {
  int64_t lower = 0;
  int64_t upper = 0;
  int64_t step = 1;
};

/// What was learned about one captured value. Index facts are what prove the
/// access pattern; the decomposition records the `div`/`rem` pair an `address`
/// was built from, so a load is only accepted when it decomposes the compact
/// row-major tensor it names.
struct ContractionFacts {
  FrontendValue value;
  std::optional<ContractionIndex> index;
  std::optional<ContractionIndex> numerator;
  int64_t divisor = 0;
  bool constantMask = false;
  bool constantMaskValue = false;
  bool floatConstant = false;
  double floatValue = 0.0;
  std::string opcode;
  std::string tensor;
  std::optional<ContractionIndex> rows;
  std::optional<ContractionIndex> columns;
  std::vector<std::string> operands;
};

enum class ContractionKind { Matmul, SwiGLU };

struct ContractionRecovery {
  ContractionKind kind = ContractionKind::Matmul;
  size_t lhs = 0;
  size_t rhs0 = 0;
  size_t rhs1 = 0;
  std::optional<size_t> accumulator;
  size_t output = 0;
  int64_t M = 0, N = 0, K = 0;
  int64_t blockM = 0, blockN = 0, blockK = 0;
  SourceSpan span;
};

class ContractionProof {
public:
  explicit ContractionProof(const FrontendProgram &program)
      : program(program) {}

  llvm::Expected<ContractionRecovery> run() {
    const NumericalContract &n = program.numericalContract;
    if (n.storageDtype != "f32" || n.computeDtype != "f32" ||
        n.accumulatorDtype != "f32" || n.rounding != "rtne" ||
        n.reductionOrder != "source" || n.reassociation || n.fma ||
        n.subnormal != "preserve")
      return unsupported(
          "counted contractions require the complete strict-f32 policy");
    // Every cast is an explicit rounding site. The native contraction and its
    // fused epilogue realize the source exactly; they cannot represent a
    // round-trip the source declares, so such a pattern is declined rather
    // than silently evaluated with the extra rounding removed.
    if (!n.castSites.empty())
      return numerical("explicit intermediate casts are not preserved by the "
                       "native contraction epilogue");

    const FrontendOperation *loop = nullptr;
    const FrontendOperation *store = nullptr;
    unsigned stores = 0;
    for (const FrontendOperation &op : program.operations) {
      if (op.opcode == "store") {
        if (++stores != 1)
          return unsupported("exactly one output write is supported", op.span);
        store = &op;
      } else if (op.opcode == "counted_loop") {
        if (loop)
          return unsupported("exactly one counted loop is supported", op.span);
        loop = &op;
      } else if (op.opcode == "dot") {
        return unsupported("a dot must be carried by the counted loop",
                           op.span);
      }
    }
    if (!loop || !store)
      return unsupported("a counted loop and one complete store are required");

    for (size_t i = 0; i != program.tensors.size(); ++i) {
      const TensorSpec &tensor = program.tensors[i];
      ContractionFacts fact;
      fact.value = {tensor.name, "ptr<" + tensor.dtype + ">", {}};
      fact.opcode = "tensor";
      fact.tensor = tensor.name;
      facts[tensor.name] = std::move(fact);
      if (tensor.direction == "input")
        inputs.push_back(i);
      else
        output = i;
    }
    if (inputs.empty() || program.tensors[output].direction != "output")
      return unsupported("one write output is required");

    // The epilogue after the loop consumes the loop's results, so their typed
    // values are in scope before the top-level chain is evaluated. The loop
    // itself is recognized below; nothing here assumes its access pattern.
    for (const FrontendOperation &op : program.operations) {
      if (op.opcode != "counted_loop")
        continue;
      for (const FrontendValue &result : op.results) {
        ContractionFacts fact;
        fact.value = result;
        fact.opcode = "loop_result";
        facts[result.id] = std::move(fact);
      }
    }

    // Evaluate the top-level address/accumulator chain in program order. The
    // loop body is evaluated separately below, with its own block arguments.
    for (const FrontendOperation &op : program.operations) {
      if (op.opcode == "counted_loop" || op.opcode == "store" ||
          op.opcode == "return")
        continue;
      if (llvm::Error error =
              evaluate(op, facts, CountedKernelLoop{}, producers))
        return llvm::Expected<ContractionRecovery>(std::move(error));
    }

    if (llvm::Error error = recognizeStore(*store))
      return llvm::Expected<ContractionRecovery>(std::move(error));
    if (llvm::Error error = recognizeLoop(*loop))
      return llvm::Expected<ContractionRecovery>(std::move(error));
    if (llvm::Error error = validateTensorRoles())
      return llvm::Expected<ContractionRecovery>(std::move(error));
    if (llvm::Error error = checkUses())
      return llvm::Expected<ContractionRecovery>(std::move(error));

    recovery.span = store->span;
    return recovery;
  }

private:
  const FrontendProgram &program;
  std::map<std::string, ContractionFacts> facts;
  std::map<std::string, const FrontendOperation *> producers;
  std::vector<size_t> inputs;
  size_t output = 0;
  ContractionRecovery recovery;

  llvm::Error unsupported(llvm::Twine message,
                          std::optional<SourceSpan> span = {}) {
    return frontendError("unsupported_frontend_program", message, span,
                         "import");
  }
  llvm::Error numerical(llvm::Twine message,
                        std::optional<SourceSpan> span = {}) {
    return frontendError("unsupported_numerical_contract", message, span,
                         "import");
  }
  llvm::Error coverage(llvm::Twine message,
                       std::optional<SourceSpan> span = {}) {
    return frontendError("invalid_launch_coverage", message, span, "import");
  }

  /// Interval of `expression` over every symbol's legal range. The counted
  /// loop's induction variable takes `lower + t * step`, so its largest value
  /// is `upper - step` for a whole number of iterations.
  std::optional<std::pair<int64_t, int64_t>>
  interval(const ContractionIndex &expression, const Shape &shape,
           const CountedKernelLoop &loop) const {
    int64_t low = expression.constant, high = expression.constant;
    auto extend = [&](int64_t coefficient, int64_t minimum, int64_t maximum) {
      int64_t a = 0, b = 0;
      if (!multiply(coefficient, minimum, a) ||
          !multiply(coefficient, maximum, b))
        return false;
      if (a > b)
        std::swap(a, b);
      return add(low, a, low) && add(high, b, high);
    };
    for (size_t axis = 0; axis != 3; ++axis)
      if (!extend(expression.coefficients[axis], 0,
                  program.launch.grid[axis] - 1))
        return {};
    for (size_t lane = 0; lane != 2; ++lane) {
      const int64_t extent = lane < shape.size() ? shape[lane] : 1;
      if (!extend(expression.coefficients[kLane0 + lane], 0, extent - 1))
        return {};
    }
    if (loop.step <= 0 || loop.upper < loop.lower ||
        (loop.upper - loop.lower) % loop.step != 0)
      return {};
    const int64_t iterations = (loop.upper - loop.lower) / loop.step;
    const int64_t last = iterations > 0 ? loop.upper - loop.step : loop.lower;
    if (!extend(expression.coefficients[kInduction], loop.lower, last))
      return {};
    return std::pair<int64_t, int64_t>{low, high};
  }

  /// `index` with the coefficients that cannot vary zeroed. A one-block axis
  /// makes its program id constant; a singleton lane makes its lane index
  /// constant; a single-iteration loop makes its induction variable constant.
  ContractionIndex normalize(ContractionIndex index, const Shape &shape,
                             const CountedKernelLoop &loop) const {
    for (size_t axis = 0; axis != 3; ++axis)
      if (program.launch.grid[axis] == 1)
        index.coefficients[axis] = 0;
    for (size_t lane = 0; lane != 2; ++lane)
      if (lane >= shape.size() || shape[lane] == 1)
        index.coefficients[kLane0 + lane] = 0;
    if (loop.upper <= loop.lower + loop.step)
      index.coefficients[kInduction] = 0;
    return index;
  }

  /// Aligns a lower-rank index value with `to` by its trailing axes, the same
  /// convention the capture's broadcast lowering uses.
  ContractionIndex promote(ContractionIndex expression, const Shape &from,
                           const Shape &to) const {
    ContractionIndex result = expression;
    result.coefficients[kLane0] = result.coefficients[kLane1] = 0;
    const size_t shift = to.size() - from.size();
    for (size_t i = 0; i < from.size(); ++i)
      result.coefficients[kLane0 + shift + i] =
          expression.coefficients[kLane0 + i];
    return result;
  }

  llvm::Error evaluate(const FrontendOperation &op,
                       std::map<std::string, ContractionFacts> &map,
                       const CountedKernelLoop &loop,
                       std::map<std::string, const FrontendOperation *> &ops) {
    std::vector<const ContractionFacts *> operands;
    for (const std::string &name : op.operands) {
      auto found = map.find(name);
      if (found == map.end())
        return unsupported("undefined or escaping value '" + name + "'",
                           op.span);
      operands.push_back(&found->second);
    }
    ContractionFacts result;
    result.opcode = op.opcode;
    result.operands = op.operands;
    if (!op.results.empty())
      result.value = op.results.front();
    const std::string &code = op.opcode;

    if (code == "constant") {
      if (result.value.dtype == "index") {
        ContractionIndex index;
        index.constant = op.attributes.getInteger("value").value_or(0);
        result.index = index;
      } else if (result.value.dtype == "i1") {
        result.constantMask = true;
        result.constantMaskValue =
            op.attributes.getBoolean("value").value_or(false);
      } else if (result.value.dtype == "f32") {
        result.floatConstant = true;
        result.floatValue = op.attributes.getNumber("value").value_or(0.0);
      }
    } else if (code == "program_id") {
      ContractionIndex index;
      const int64_t axis = op.attributes.getInteger("axis").value_or(0);
      if (axis < 0 || axis > 2)
        return unsupported("program axis must be 0..2", op.span);
      index.coefficients[size_t(axis)] = 1;
      result.index = index;
    } else if (code == "range") {
      ContractionIndex index;
      index.coefficients[kLane0] = 1;
      index.constant = op.attributes.getInteger("start").value_or(0);
      result.index = index;
    } else if (code == "expand_dims" || code == "broadcast") {
      if (operands[0]->index) {
        ContractionIndex index = *operands[0]->index;
        index.coefficients[kLane0] = index.coefficients[kLane1] = 0;
        for (size_t axis = 0; axis != operands[0]->value.shape.size(); ++axis) {
          int64_t mapped =
              code == "expand_dims"
                  ? int64_t(axis) +
                        (int64_t(axis) >=
                         op.attributes.getInteger("axis").value_or(0))
                  : (*op.attributes.getArray("dimensions"))[axis]
                        .getAsInteger()
                        .value_or(0);
          if (mapped < 0 || mapped > 1)
            return unsupported("index view leaves the rank-two lane space",
                               op.span);
          index.coefficients[kLane0 + mapped] =
              operands[0]->index->coefficients[kLane0 + axis];
        }
        result.index = index;
      } else if (operands[0]->constantMask) {
        // A view of a whole-tile mask is still that mask.
        result.constantMask = true;
        result.constantMaskValue = operands[0]->constantMaskValue;
      } else {
        return unsupported("only index and constant-mask views are supported",
                           op.span);
      }
    } else if (code == "add" || code == "sub" || code == "mul" ||
               code == "div" || code == "rem") {
      if (result.value.dtype != "index") {
        // Data arithmetic is recorded; the epilogue checks decide whether the
        // expression is the one the fused family realizes.
        if (!op.results.empty()) {
          ops[result.value.id] = &op;
          map[result.value.id] = std::move(result);
        }
        return llvm::Error::success();
      }
      if (!operands[0]->index || !operands[1]->index)
        return unsupported("non-affine index operand '" + code + "'", op.span);
      const Shape &shape = result.value.shape;
      ContractionIndex a =
          promote(*operands[0]->index, operands[0]->value.shape, shape);
      ContractionIndex b =
          promote(*operands[1]->index, operands[1]->value.shape, shape);
      if (code == "add") {
        auto sum = addIndex(a, b);
        if (!sum)
          return unsupported("index arithmetic overflows", op.span);
        result.index = *sum;
      } else if (code == "sub") {
        auto negated = scaleIndex(b, -1);
        auto difference = negated ? addIndex(a, *negated) : std::nullopt;
        if (!difference)
          return unsupported("index arithmetic overflows", op.span);
        result.index = *difference;
      } else if (code == "mul") {
        if (uniformIndex(a))
          result.index = scaleIndex(b, a.constant);
        else if (uniformIndex(b))
          result.index = scaleIndex(a, b.constant);
        if (!result.index)
          return unsupported("index multiplication needs one constant factor",
                             op.span);
      } else {
        if (!uniformIndex(b) || b.constant <= 0)
          return unsupported("index " + code + " needs a positive constant",
                             op.span);
        ContractionIndex quotient, remainder;
        quotient.constant = a.constant / b.constant;
        remainder.constant = a.constant % b.constant;
        for (size_t i = 0; i != a.coefficients.size(); ++i) {
          quotient.coefficients[i] = a.coefficients[i] / b.constant;
          remainder.coefficients[i] = a.coefficients[i] % b.constant;
        }
        auto bounds = interval(remainder, shape, loop);
        if (!bounds || bounds->first < 0 || bounds->second >= b.constant)
          return unsupported(
              "the index decomposition is not exact on every lane", op.span);
        result.numerator = a;
        result.divisor = b.constant;
        result.index = code == "div" ? quotient : remainder;
      }
    } else if (code == "address") {
      if (operands[0]->tensor.empty() || !operands[1]->index ||
          !operands[2]->index)
        return unsupported("address is not affine", op.span);
      // `address` reads a named base tensor plus a row and a column index. The
      // capture always derives both from one `div`/`rem` pair; keeping that
      // requirement means a load cannot silently name coordinates that the
      // declared tensor's storage does not have.
      result.tensor = operands[0]->tensor;
      const Shape &shape = result.value.shape;
      result.rows =
          promote(*operands[1]->index, operands[1]->value.shape, shape);
      result.columns =
          promote(*operands[2]->index, operands[2]->value.shape, shape);
      if (!operands[1]->numerator || !operands[2]->numerator ||
          !(*operands[1]->numerator == *operands[2]->numerator) ||
          operands[1]->divisor != operands[2]->divisor)
        return unsupported("address rows and columns do not decompose one "
                           "row-major index",
                           op.span);
      result.numerator = operands[1]->numerator;
      result.divisor = operands[1]->divisor;
    } else if (code == "load") {
      if (!operands[0]->rows || operands[0]->tensor.empty())
        return unsupported("load has no affine address", op.span);
      if (!operands[1]->constantMask || !operands[1]->constantMaskValue)
        return unsupported("contraction loads must read every lane", op.span);
      result.tensor = operands[0]->tensor;
      result.rows = operands[0]->rows;
      result.columns = operands[0]->columns;
      result.divisor = operands[0]->divisor;
    } else if (code == "dot" || code == "silu" || code == "neg" ||
               code == "select" || code == "compare" || code == "cast" ||
               code == "ceildiv") {
      // Recorded for the pattern checks; their value facts carry no index.
    } else {
      return unsupported(
          "opcode is outside the counted-contraction family: " + code, op.span);
    }

    if (!op.results.empty()) {
      const std::string &id = result.value.id;
      ops[id] = &op;
      map[id] = std::move(result);
    }
    return llvm::Error::success();
  }

  llvm::Error recognizeStore(const FrontendOperation &op) {
    auto pointerIt = facts.find(op.operands[0]);
    auto valueIt = facts.find(op.operands[1]);
    auto maskIt = facts.find(op.operands[2]);
    if (pointerIt == facts.end() || valueIt == facts.end() ||
        maskIt == facts.end())
      return unsupported("the store operands are not defined", op.span);
    const ContractionFacts &pointer = pointerIt->second;
    if (pointer.tensor != program.tensors[output].name)
      return unsupported("the only write must target the output tensor",
                         op.span);
    if (!pointer.rows || !pointer.columns)
      return unsupported("the store has no affine address", op.span);
    if (!maskIt->second.constantMask || !maskIt->second.constantMaskValue)
      return coverage("the contraction store must write every lane", op.span);
    const Shape &tile = pointer.value.shape;
    if (tile.size() != 2)
      return unsupported("the output tile must be rank two", op.span);
    const int64_t blockM = tile[0], blockN = tile[1];
    const Shape &shape = program.tensors[output].shape;
    if (pointer.divisor != shape[1])
      return unsupported("the store address does not decompose the output's "
                         "row-major storage",
                         op.span);
    if (!matches(*pointer.rows, rowsForm(blockM), tile, CountedKernelLoop{}))
      return unsupported("the store rows are not the output tile's rows",
                         op.span);
    if (!matches(*pointer.columns, columnsForm(blockN), tile,
                 CountedKernelLoop{}))
      return unsupported("the store columns are not the output tile's columns",
                         op.span);
    if (program.launch.grid[2] != 1 || program.launch.block[2] != 1)
      return coverage("grid z and block z must both be one", op.span);
    if (program.launch.block[0] != blockM || program.launch.block[1] != blockN)
      return coverage("the launch block must be the output tile", op.span);
    if (shape[0] % blockM != 0 || shape[1] % blockN != 0 ||
        program.launch.grid[0] != shape[0] / blockM ||
        program.launch.grid[1] != shape[1] / blockN)
      return coverage("the launch grid must tile the output domain exactly",
                      op.span);
    recovery.output = output;
    recovery.M = shape[0];
    recovery.N = shape[1];
    recovery.blockM = blockM;
    recovery.blockN = blockN;
    recovery.span = op.span;
    storeValue = valueIt->first;
    return llvm::Error::success();
  }

  llvm::Error recognizeLoop(const FrontendOperation &op) {
    if (op.regions.size() != 1)
      return unsupported("the counted loop needs one typed region", op.span);
    if (op.operands.size() < 3)
      return unsupported("the counted loop needs bounds, a step and carried "
                         "state",
                         op.span);
    auto uniform = [&](const std::string &id) -> std::optional<int64_t> {
      auto found = facts.find(id);
      if (found == facts.end() || !found->second.index ||
          !uniformIndex(*found->second.index))
        return {};
      return found->second.index->constant;
    };
    std::optional<int64_t> lower = uniform(op.operands[0]);
    std::optional<int64_t> upper = uniform(op.operands[1]);
    std::optional<int64_t> step = op.attributes.getInteger("step");
    if (!lower || !upper || !step || *step <= 0)
      return unsupported("the counted loop bounds must be constant", op.span);
    CountedKernelLoop loop{*lower, *upper, *step};
    if (loop.lower != 0)
      return unsupported("the counted K loop must start at zero", op.span);

    const size_t arms = op.operands.size() - 2;
    if (op.results.size() != arms || arms < 1 || arms > 2)
      return unsupported("a contraction carries one or two dot accumulators",
                         op.span);

    // The block is the accumulator tile; the K block is the loop step.
    const int64_t blockM = recovery.blockM, blockN = recovery.blockN;
    if (op.results[0].shape != Shape{blockM, blockN} ||
        op.results[0].dtype != "f32")
      return unsupported("the accumulator must be an f32 output tile", op.span);

    // Seed values: a zero tile, or (matmul only) a caller tensor read at the
    // output tile's coordinates.
    std::optional<size_t> accumulator;
    for (size_t arm = 0; arm != arms; ++arm) {
      auto found = facts.find(op.operands[2 + arm]);
      if (found == facts.end())
        return unsupported("the accumulator seed is not defined", op.span);
      const ContractionFacts &seed = found->second;
      const bool zero = seed.floatConstant && seed.value.dtype == "f32" &&
                        seed.value.shape == Shape{blockM, blockN} &&
                        seed.floatValue == 0.0;
      if (zero)
        continue;
      if (seed.opcode != "load" || !seed.rows || !seed.columns ||
          seed.divisor == 0)
        return unsupported("an accumulator must be zero or a caller tensor "
                           "tile",
                           op.span);
      if (arms != 1)
        return numerical("fused SwiGLU requires zero accumulators; a caller "
                         "accumulator is not part of its numerical contract");
      if (accumulator)
        return unsupported("only one caller accumulator is supported", op.span);
      const TensorSpec *tensor = tensorNamed(seed.tensor);
      if (!tensor || tensor->direction != "input" ||
          tensor->shape != Shape{recovery.M, recovery.N})
        return unsupported("a caller accumulator must be an [M, N] input",
                           op.span);
      if (!matches(*seed.rows, rowsForm(blockM), Shape{blockM, blockN}, loop) ||
          !matches(*seed.columns, columnsForm(blockN), Shape{blockM, blockN},
                   loop) ||
          seed.divisor != recovery.N)
        return unsupported("the caller accumulator tile is not the output "
                           "tile",
                           op.span);
      accumulator = tensorIndex(seed.tensor);
    }
    recovery.accumulator = accumulator;

    // Evaluate the body with its own arguments in scope.
    std::map<std::string, ContractionFacts> body = facts;
    const FrontendRegion &region = op.regions.front();
    if (region.arguments.size() != arms + 1)
      return unsupported("the loop body must take the induction variable and "
                         "one argument per accumulator",
                         op.span);
    if (region.arguments[0].dtype != "index")
      return unsupported("the loop induction variable must be an index",
                         op.span);
    ContractionFacts induction;
    ContractionIndex inductionIndex;
    inductionIndex.coefficients[kInduction] = 1;
    induction.index = inductionIndex;
    induction.value = region.arguments[0];
    induction.opcode = "induction";
    body[region.arguments[0].id] = induction;
    for (size_t arm = 0; arm != arms; ++arm) {
      ContractionFacts argument;
      argument.value = region.arguments[arm + 1];
      argument.opcode = "block_argument";
      body[region.arguments[arm + 1].id] = argument;
    }

    std::map<std::string, const FrontendOperation *> bodyProducers;
    std::vector<const FrontendOperation *> dots;
    for (const FrontendOperation &nested : region.operations) {
      if (nested.opcode == "yield") {
        if (nested.operands.size() != arms)
          return unsupported("the loop yields one value per accumulator",
                             nested.span);
        continue;
      }
      if (llvm::Error error = evaluate(nested, body, loop, bodyProducers))
        return error;
      if (nested.opcode == "dot")
        dots.push_back(&nested);
      if (nested.opcode == "store")
        return unsupported("a contraction body may not write memory",
                           nested.span);
    }
    if (dots.size() != arms)
      return unsupported("each accumulator must be updated by exactly one dot",
                         op.span);

    // The yield hands each dot result back.
    const FrontendOperation *yield = nullptr;
    for (const FrontendOperation &nested : region.operations)
      if (nested.opcode == "yield")
        yield = &nested;
    if (!yield)
      return unsupported("the counted loop has no yield", op.span);
    for (size_t arm = 0; arm != arms; ++arm) {
      if (dots[arm]->results.size() != 1 ||
          dots[arm]->results.front().id != yield->operands[arm])
        return unsupported("the loop yields something other than its dot "
                           "results",
                           op.span);
    }

    // Each dot accumulates into its own block argument.
    for (size_t arm = 0; arm != arms; ++arm) {
      const FrontendOperation &dot = *dots[arm];
      if (dot.operands.size() != 3)
        return unsupported("a dot takes two inputs and an accumulator",
                           dot.span);
      if (dot.operands[2] != region.arguments[arm + 1].id)
        return unsupported("a dot must accumulate into its own carried value",
                           dot.span);
    }

    // Shapes and access forms of the two arms.
    auto loadOf = [&](const FrontendOperation &dot,
                      size_t index) -> const ContractionFacts * {
      auto found = body.find(dot.operands[index]);
      if (found == body.end() || found->second.opcode != "load")
        return nullptr;
      return &found->second;
    };
    const ContractionFacts *lhs = loadOf(*dots[0], 0);
    if (!lhs)
      return unsupported("the left dot operand must be a tile load",
                         dots[0]->span);
    const TensorSpec *lhsTensor = tensorNamed(lhs->tensor);
    if (!lhsTensor || lhsTensor->direction != "input")
      return unsupported("the left operand must read a declared input",
                         dots[0]->span);
    const int64_t K = lhsTensor->shape[1];
    if (loop.upper != K)
      return unsupported("the counted loop must traverse exactly the declared "
                         "K extent",
                         dots[0]->span);
    if (K % loop.step != 0)
      return unsupported("the K step must divide the reduction extent",
                         dots[0]->span);
    recovery.K = K;
    recovery.blockK = loop.step;
    if (lhs->value.shape != Shape{blockM, loop.step} || lhs->divisor != K ||
        !matches(*lhs->rows, rowsForm(blockM), Shape{blockM, loop.step},
                 loop) ||
        !matches(*lhs->columns, contractionColumnsForm(),
                 Shape{blockM, loop.step}, loop) ||
        lhsTensor->shape != Shape{recovery.M, K})
      return unsupported("the left operand is not the canonical [M, K] tile",
                         dots[0]->span);
    recovery.lhs = tensorIndex(lhs->tensor);

    const ContractionFacts *rhs0 = loadOf(*dots[0], 1);
    if (!rhs0)
      return unsupported("the right dot operand must be a tile load",
                         dots[0]->span);
    const TensorSpec *rhs0Tensor = tensorNamed(rhs0->tensor);
    if (!rhs0Tensor || rhs0Tensor->direction != "input" ||
        rhs0Tensor->shape != Shape{K, recovery.N} ||
        rhs0->value.shape != Shape{loop.step, blockN} ||
        rhs0->divisor != recovery.N ||
        !matches(*rhs0->rows, contractionRowsForm(), Shape{loop.step, blockN},
                 loop) ||
        !matches(*rhs0->columns, columnsForm(blockN), Shape{loop.step, blockN},
                 loop))
      return unsupported("the right operand is not the canonical [K, N] tile",
                         dots[0]->span);
    recovery.rhs0 = tensorIndex(rhs0->tensor);

    if (arms == 1) {
      recovery.kind = ContractionKind::Matmul;
      if (storeValue != op.results[0].id)
        return unsupported("the stored value must be the accumulator", op.span);
      return llvm::Error::success();
    }

    // Fused SwiGLU: both projections read the *same* left-hand tile, and the
    // epilogue is exactly silu(gate) * up.
    recovery.kind = ContractionKind::SwiGLU;
    const ContractionFacts *lhsUp = loadOf(*dots[1], 0);
    if (!lhsUp || lhsUp->tensor != lhs->tensor ||
        dots[1]->operands[0] != dots[0]->operands[0])
      return unsupported("fused SwiGLU requires both projections to read one "
                         "shared input tile",
                         dots[1]->span);
    const ContractionFacts *rhs1 = loadOf(*dots[1], 1);
    if (!rhs1)
      return unsupported("the up projection must read a tile", dots[1]->span);
    const TensorSpec *rhs1Tensor = tensorNamed(rhs1->tensor);
    if (!rhs1Tensor || rhs1Tensor->direction != "input" ||
        rhs1Tensor->shape != Shape{K, recovery.N} ||
        rhs1->value.shape != Shape{loop.step, blockN} ||
        rhs1->divisor != recovery.N ||
        !matches(*rhs1->rows, contractionRowsForm(), Shape{loop.step, blockN},
                 loop) ||
        !matches(*rhs1->columns, columnsForm(blockN), Shape{loop.step, blockN},
                 loop))
      return unsupported("the up projection is not the canonical [K, N] tile",
                         dots[1]->span);
    if (rhs1->tensor == rhs0->tensor)
      return unsupported("fused SwiGLU requires two distinct projection "
                         "weights",
                         dots[1]->span);
    recovery.rhs1 = tensorIndex(rhs1->tensor);

    auto valueIt = producers.find(storeValue);
    if (valueIt == producers.end() || valueIt->second->opcode != "mul" ||
        valueIt->second->operands.size() != 2)
      return unsupported("the stored value must be the gated product", op.span);
    const FrontendOperation &product = *valueIt->second;
    auto siluIt = producers.find(product.operands[0]);
    if (siluIt == producers.end() || siluIt->second->opcode != "silu" ||
        siluIt->second->operands.size() != 1 ||
        siluIt->second->operands[0] != op.results[0].id ||
        product.operands[1] != op.results[1].id)
      return unsupported("the fused epilogue must be silu(gate) * up", op.span);
    return llvm::Error::success();
  }

  /// The declared tensors must be exactly the roles the pattern reads.
  llvm::Error validateTensorRoles() {
    std::vector<size_t> used{recovery.lhs, recovery.rhs0};
    if (recovery.kind == ContractionKind::SwiGLU)
      used.push_back(recovery.rhs1);
    if (recovery.accumulator)
      used.push_back(*recovery.accumulator);
    llvm::sort(used);
    std::vector<size_t> declared = inputs;
    llvm::sort(declared);
    if (used != declared)
      return unsupported("every declared input must be exactly one operand of "
                         "the recovered contraction");
    std::vector<size_t> roles{recovery.lhs, recovery.rhs0};
    if (recovery.kind == ContractionKind::SwiGLU)
      roles.push_back(recovery.rhs1);
    if (recovery.accumulator)
      roles.push_back(*recovery.accumulator);
    for (size_t index : roles)
      if (program.tensors[index].dtype != "f32")
        return unsupported("counted contractions are uniform-f32");
    if (program.tensors[output].dtype != "f32")
      return unsupported("counted contractions are uniform-f32");
    return llvm::Error::success();
  }

  /// Every value the pattern consumes is consumed exactly as the pattern says.
  /// An extra reader is an extra expression the recovery did not model.
  llvm::Error checkUses() {
    std::map<std::string, unsigned> uses;
    auto count = [&](const FrontendOperation &op) {
      for (const std::string &name : op.operands)
        ++uses[name];
    };
    std::function<void(const std::vector<FrontendOperation> &)> walk =
        [&](const std::vector<FrontendOperation> &operations) {
          for (const FrontendOperation &op : operations) {
            count(op);
            for (const FrontendRegion &region : op.regions)
              walk(region.operations);
          }
        };
    for (const FrontendOperation &op : program.operations)
      if (op.opcode == "counted_loop") {
        for (const FrontendRegion &region : op.regions)
          walk(region.operations);
      } else {
        count(op);
      }
    for (const FrontendOperation &op : program.operations) {
      if (op.opcode != "counted_loop")
        continue;
      // Each accumulator result is consumed exactly once by the epilogue:
      // `silu(gate)` reads the gate, and the gated product reads both results.
      for (size_t arm = 0; arm != op.results.size(); ++arm) {
        auto found = uses.find(op.results[arm].id);
        if (found == uses.end() || found->second != 1u)
          return unsupported("a loop result has an extra reader", op.span);
      }
      for (const FrontendRegion &region : op.regions)
        for (const FrontendOperation &nested : region.operations) {
          if (nested.opcode != "dot")
            continue;
          for (size_t index = 0; index != 2; ++index) {
            auto found = uses.find(nested.operands[index]);
            if (found == uses.end())
              continue;
            const bool shared =
                recovery.kind == ContractionKind::SwiGLU && index == 0;
            if (found->second != (shared ? 2u : 1u))
              return unsupported("a dot input has an extra reader",
                                 nested.span);
          }
          for (const FrontendValue &result : nested.results) {
            auto found = uses.find(result.id);
            if (found == uses.end() || found->second != 1u)
              return unsupported("a dot result has an extra reader",
                                 nested.span);
          }
        }
    }
    return llvm::Error::success();
  }

  const TensorSpec *tensorNamed(const std::string &name) const {
    for (const TensorSpec &tensor : program.tensors)
      if (tensor.name == name)
        return &tensor;
    return nullptr;
  }
  size_t tensorIndex(const std::string &name) const {
    for (size_t i = 0; i != program.tensors.size(); ++i)
      if (program.tensors[i].name == name)
        return i;
    return program.tensors.size();
  }
  bool matches(const ContractionIndex &actual, const ContractionIndex &expected,
               const Shape &shape, const CountedKernelLoop &loop) const {
    return normalize(actual, shape, loop) == normalize(expected, shape, loop);
  }
  static ContractionIndex rowsForm(int64_t blockM) {
    ContractionIndex index;
    index.coefficients[kPid0] = blockM;
    index.coefficients[kLane0] = 1;
    return index;
  }
  static ContractionIndex columnsForm(int64_t blockN) {
    ContractionIndex index;
    index.coefficients[kPid1] = blockN;
    index.coefficients[kLane1] = 1;
    return index;
  }
  static ContractionIndex contractionRowsForm() {
    ContractionIndex index;
    index.coefficients[kInduction] = 1;
    index.coefficients[kLane0] = 1;
    return index;
  }
  static ContractionIndex contractionColumnsForm() {
    ContractionIndex index;
    index.coefficients[kInduction] = 1;
    index.coefficients[kLane1] = 1;
    return index;
  }

  std::string storeValue;
};

llvm::Expected<mlir::llk::ScheduleEntry>
anchorScheduleFor(const mlir::llk::machine::MachineModel &machine, int64_t M,
                  int64_t N, int64_t K, unsigned arms);

/// Emits the recovered contraction as the source-level operator the mapping
/// pipeline realizes, derives its contract from the contraction family, and
/// exports the anchor search space through the same driver the concrete export
/// uses.
llvm::Expected<ImportedProgram>
importContraction(const FrontendRequest &request, mlir::MLIRContext &context,
                  const mlir::llk::machine::MachineModel *machine) {
  namespace native = mlir::llk;
  auto proof = ContractionProof(request.program()).run();
  if (!proof)
    return proof.takeError();
  const ContractionRecovery &recovery = *proof;
  if (recovery.span.line > UINT32_MAX || recovery.span.column > UINT32_MAX)
    return frontendError("invalid_frontend_schema",
                         "source location exceeds MLIR's coordinate range",
                         recovery.span, "import");
  mlir::llk::ScheduleEntry anchor = mlir::llk::selectBoundedBaselineSchedule();
  anchor.tail_policy = "pad";
  if (machine) {
    const unsigned arms = recovery.kind == ContractionKind::SwiGLU ? 2u : 1u;
    auto derived =
        anchorScheduleFor(*machine, recovery.M, recovery.N, recovery.K, arms);
    if (!derived)
      return frontendError("unsupported_frontend_program",
                           llvm::toString(derived.takeError()), recovery.span,
                           "import");
    anchor = *derived;
  }
  using namespace mlir;
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<tensor::TensorDialect>();
  context.getOrLoadDialect<linalg::LinalgDialect>();
  context.getOrLoadDialect<native::LLKDialect>();
  context.getOrLoadDialect<micro::MicroDialect>();
  auto loc = FileLineColLoc::get(&context, recovery.span.file,
                                 unsigned(recovery.span.line),
                                 unsigned(recovery.span.column));
  ImportedProgram imported;
  imported.module = ModuleOp::create(loc);
  OpBuilder builder(&context);
  builder.setInsertionPointToStart(imported.module->getBody());
  const auto &program = request.program();
  Type f32 = builder.getF32Type();
  SmallVector<int64_t, 2> shape{recovery.M, recovery.N};
  auto resultType = RankedTensorType::get(shape, f32);
  SmallVector<Type, 4> inputTypes;
  for (const TensorSpec &tensor : program.tensors)
    if (tensor.direction == "input")
      inputTypes.push_back(RankedTensorType::get(tensor.shape, f32));
  auto function =
      func::FuncOp::create(builder, loc, program.entry,
                           builder.getFunctionType(inputTypes, resultType));
  Block *body = function.addEntryBlock();
  builder.setInsertionPointToStart(body);
  std::map<size_t, Value> arguments;
  size_t next = 0;
  for (size_t i = 0; i != program.tensors.size(); ++i)
    if (program.tensors[i].direction == "input")
      arguments[i] = body->getArgument(next++);

  // A zero accumulator is the one seed the caller cannot observe: the compiler
  // owns the init port and supplies zeros. A caller accumulator is a real
  // input and reaches the op unchanged.
  Value init;
  if (recovery.accumulator) {
    init = arguments.at(*recovery.accumulator);
  } else {
    Value empty = tensor::EmptyOp::create(builder, loc, shape, f32);
    Value zero = arith::ConstantOp::create(builder, loc, f32,
                                           builder.getF32FloatAttr(0.0F));
    init = linalg::FillOp::create(builder, loc, ValueRange{zero},
                                  ValueRange{empty})
               .getResult(0);
  }
  native::MathMode mode = native::MathMode::strict;
  if (recovery.kind == ContractionKind::SwiGLU &&
      program.numericalContract.activationApproximation == "native")
    mode = native::MathMode::bounded_fast;
  auto accumulatorType = TypeAttr::get(f32);
  Operation *root = nullptr;
  if (recovery.kind == ContractionKind::Matmul) {
    auto matmul = native::MatmulOp::create(
        builder, loc, resultType, arguments.at(recovery.lhs),
        arguments.at(recovery.rhs0), init, accumulatorType,
        native::MathModeAttr::get(&context, mode));
    root = matmul.getOperation();
    func::ReturnOp::create(builder, loc, matmul.getResult());
  } else {
    auto fused = native::FusedSwiGLUOp::create(
        builder, loc, resultType, arguments.at(recovery.lhs),
        arguments.at(recovery.rhs0), arguments.at(recovery.rhs1), init,
        accumulatorType,
        native::ActivationAttr::get(&context, native::Activation::silu),
        native::MathModeAttr::get(&context, mode));
    root = fused.getOperation();
    func::ReturnOp::create(builder, loc, fused.getResult());
  }
  // The source's numerical policy is part of its contract identity: an exact
  // activation and a relaxed one are different programs, never the same
  // program with a flag.
  root->setAttr("llk.numerical_contract",
                builder.getStringAttr(native::stringifyMathMode(mode)));
  (*imported.module)
      ->setAttr("llk.frontend.request",
                builder.getStringAttr(request.canonicalJSON()));
  if (failed(verify(*imported.module)))
    return frontendError("invalid_frontend_schema",
                         "structured import failed MLIR verification",
                         recovery.span, "import");

  if (llvm::Error error =
          native::exportBaselineSearchSpaces(*imported.module, anchor))
    return frontendError("unsupported_frontend_program",
                         llvm::toString(std::move(error)), recovery.span,
                         "import");
  auto space = native::perf::loadSearchSpace(*imported.module);
  if (!space)
    return frontendError("unsupported_frontend_program",
                         llvm::toString(space.takeError()), recovery.span,
                         "import");
  imported.space = std::move(*space);
  // The source family's boundary realization is the anchored `pad` schedule;
  // every source-owned choice is pinned to the anchor, so each candidate
  // realizes the same schedule the contract describes.
  auto pinInteger = [&](llvm::StringRef name, int64_t value) -> bool {
    for (native::perf::SearchParam &parameter : imported.space.params) {
      if (parameter.name != name.str())
        continue;
      auto found = llvm::find_if(parameter.choices, [&](const auto &choice) {
        return choice.isInteger() && choice.integer() == value;
      });
      if (found == parameter.choices.end())
        return false;
      parameter.choices = {*found};
    }
    return true;
  };
  auto pinSymbol = [&](llvm::StringRef name, llvm::StringRef value) -> bool {
    for (native::perf::SearchParam &parameter : imported.space.params) {
      if (parameter.name != name.str())
        continue;
      auto found = llvm::find_if(parameter.choices, [&](const auto &choice) {
        return choice.isSymbolic() && choice.symbol() == value;
      });
      if (found == parameter.choices.end())
        return false;
      parameter.choices = {*found};
    }
    return true;
  };
  // Layout, memory path and owner hierarchy are placement policy, which the
  // mapping stage solves. Binding them here would pin one placement the target
  // never proved for every node of the kernel, so they are not offered as
  // source choices at all.
  llvm::erase_if(imported.space.params, [](const native::perf::SearchParam &p) {
    return p.kind == "layout" || p.kind == "memory_path" ||
           p.kind == "owner_mapping";
  });
  // A constraint naming a parameter the space no longer declares is illegal,
  // so the placement-only legality records go with the parameters.
  llvm::StringSet<> declared;
  for (const native::perf::SearchParam &parameter : imported.space.params)
    declared.insert(parameter.name);
  llvm::erase_if(imported.space.constraints,
                 [&](const native::perf::SearchConstraint &constraint) {
                   return llvm::any_of(constraint.params,
                                       [&](const std::string &name) {
                                         return !declared.contains(name);
                                       });
                 });
  if (!pinInteger("BM", anchor.BM) || !pinInteger("BN", anchor.BN) ||
      !pinInteger("BK", anchor.BK) ||
      !pinInteger("pipeline_stages", anchor.pipeline_stages) ||
      !pinSymbol("fragment_shape", anchor.fragment_shape) ||
      !pinSymbol("tail_policy", anchor.tail_policy))
    return frontendError("unsupported_frontend_program",
                         "the exported search space does not offer the "
                         "anchored schedule",
                         recovery.span, "import");

  auto registry = native::workload::makeDefaultSourceAdapters();
  const auto *adapter =
      registry.findByKey(native::workload::contractionSourceAdapterKey());
  if (!adapter)
    return frontendError("unsupported_frontend_program",
                         "contraction source adapter is not registered",
                         recovery.span, "import");
  auto described = adapter->describe(root);
  if (!described)
    return frontendError("unsupported_frontend_program",
                         llvm::toString(described.takeError()), recovery.span,
                         "import");
  imported.contract = std::move(*described);
  imported.selection = {program.entry, 0};
  imported.adapterKey = native::workload::contractionSourceAdapterKey().str();
  // The matmul init port is compiler-owned when the source seeds its
  // accumulator with zeros; a caller accumulator or the fused op has no
  // compiler-owned input at all.
  if (!recovery.accumulator && recovery.kind == ContractionKind::Matmul)
    imported.internalDestinationPort = 2;
  for (size_t i = 0; i != program.tensors.size(); ++i) {
    if (program.tensors[i].direction != "input")
      continue;
    unsigned nativePort = 0;
    if (i == recovery.lhs)
      nativePort = 0;
    else if (i == recovery.rhs0)
      nativePort = 1;
    else if (recovery.kind == ContractionKind::SwiGLU && i == recovery.rhs1)
      nativePort = 2;
    else if (recovery.accumulator && i == *recovery.accumulator)
      nativePort = 2;
    else
      return frontendError("unsupported_frontend_program",
                           "a declared input has no native port", recovery.span,
                           "import");
    imported.publicToNativePorts.push_back({unsigned(i), nativePort});
  }
  return imported;
}
llvm::Expected<mlir::llk::ScheduleEntry>
anchorScheduleFor(const mlir::llk::machine::MachineModel &machine, int64_t M,
                  int64_t N, int64_t K, unsigned arms) {
  auto volume = [](llvm::ArrayRef<int64_t> shape) {
    int64_t product = 1;
    for (int64_t extent : shape)
      product *= extent;
    return product;
  };
  const std::vector<int64_t> *fragment = nullptr;
  for (const mlir::llk::machine::ComputeNode &compute : machine.computes) {
    if (compute.kind != "matrix_engine")
      continue;
    for (const std::vector<int64_t> &shape : compute.shapes) {
      if (shape.size() != 3)
        continue;
      if (!fragment || volume(shape) > volume(*fragment))
        fragment = &shape;
    }
  }
  if (!fragment)
    return llvm::make_error<llvm::StringError>(
        "the machine declares no rank-three matrix instruction fragment",
        llvm::inconvertibleErrorCode());

  // The static boundary policy materializes a padded copy of every operand per
  // spatial tile, so the source is tiled to keep that footprint inside the
  // machine's staging capacity. Divisors keep the padding a no-op.
  uint64_t stagingBytes = 0, accumulatorCapacity = 0;
  for (const mlir::llk::machine::MemoryNode &memory : machine.memories) {
    if (memory.kind == "sram")
      stagingBytes = std::max<uint64_t>(stagingBytes, memory.capacityBytes);
    else if (memory.kind == "acc")
      accumulatorCapacity =
          std::max<uint64_t>(accumulatorCapacity, memory.capacityBytes);
  }
  auto divisorsDescending = [](int64_t extent) {
    std::vector<int64_t> values;
    for (int64_t divisor = extent; divisor >= 1; --divisor)
      if (extent % divisor == 0)
        values.push_back(divisor);
    return values;
  };
  constexpr int64_t elementBytes = 4; // counted contractions are uniform f32.
  auto fits = [&](int64_t BM, int64_t BN) {
    // Every accumulator arm is resident for the whole K loop.
    if (accumulatorCapacity &&
        static_cast<uint64_t>(arms) * BM * BN * elementBytes >
            accumulatorCapacity)
      return false;
    if (stagingBytes == 0)
      return true;
    const int64_t occurrences = (M / BM) * (N / BN);
    const uint64_t operandBytes = static_cast<uint64_t>(M) * K +
                                  static_cast<uint64_t>(K) * N +
                                  static_cast<uint64_t>(M) * N;
    const uint64_t accumulatorBytes = static_cast<uint64_t>(M) * N;
    return static_cast<uint64_t>(occurrences) *
               (operandBytes + accumulatorBytes) * elementBytes <=
           stagingBytes;
  };
  for (int64_t BM : divisorsDescending(M))
    for (int64_t BN : divisorsDescending(N)) {
      if (!fits(BM, BN))
        continue;
      // K only changes the reduction loop's trip count, not the per-tile
      // materialization, so the full K extent is always preferred.
      int64_t BK = K;
      mlir::llk::ScheduleEntry anchor =
          mlir::llk::selectBoundedBaselineSchedule();
      anchor.BM = BM;
      anchor.BN = BN;
      anchor.BK = BK;
      anchor.fragment_shape =
          (llvm::Twine((*fragment)[0]) + "x" + llvm::Twine((*fragment)[1]) +
           "x" + llvm::Twine((*fragment)[2]))
              .str();
      anchor.mma_shape = anchor.fragment_shape;
      anchor.tail_policy = "pad";
      return anchor;
    }
  return llvm::make_error<llvm::StringError>(
      "the contraction does not fit the target's accumulator or staging "
      "capacity",
      llvm::inconvertibleErrorCode());
}
} // namespace

namespace {
llvm::Expected<ImportedProgram>
importRequest(const FrontendRequest &request, mlir::MLIRContext &context,
              const mlir::llk::machine::MachineModel *machine) {
  // A counted loop or a dot is never an elementwise program: those opcodes are
  // the contraction family's alone, and the recovery below is the only proof
  // that can accept them.
  const bool contraction = llvm::any_of(
      request.program().operations, [](const FrontendOperation &op) {
        return op.opcode == "dot" || op.opcode == "counted_loop";
      });
  if (contraction)
    return importContraction(request, context, machine);
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
  // The tail proof is an explicit source fact: the rounded launch's mask is
  // exactly the in-bounds predicate and every active lane is written once. The
  // adapter may realize a non-dividing tile only when it can see that fact, and
  // it records the block the proof was made for so a retuned candidate cannot
  // inherit it.
  if (proof->tail) {
    generic->setAttr("llk.tail_policy", builder.getStringAttr("masked-edge"));
    generic->setAttr(
        "llk.tail_block",
        DenseI64ArrayAttr::get(&context, {proof->block0, proof->block1}));
  }
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
  // The elementwise kernel's destination is declared but read by nobody, so
  // the compiler owns that native port and fills it.
  imported.internalDestinationPort = 2;
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
  if (proof->tail) {
    mlir::llk::perf::SearchParam parameter;
    parameter.name = "frontend_tail_policy";
    parameter.kind = "tail_policy";
    parameter.role = "tail_policy";
    parameter.choices.emplace_back(std::string("masked-edge"));
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
} // namespace

llvm::Expected<ImportedProgram> importProgram(const FrontendRequest &request,
                                              mlir::MLIRContext &context) {
  return importRequest(request, context, nullptr);
}

llvm::Expected<ImportedProgram>
importProgram(const FrontendRequest &request, mlir::MLIRContext &context,
              const mlir::llk::machine::MachineModel &machine) {
  return importRequest(request, context, &machine);
}
} // namespace llk::frontend
