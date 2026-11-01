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
#include "llvm/ADT/SmallVector.h"
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
} // namespace llk::frontend
