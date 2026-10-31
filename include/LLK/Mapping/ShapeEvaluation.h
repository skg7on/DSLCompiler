//===- ShapeEvaluation.h - Evaluate checked shape index arithmetic --------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Dynamic MicroIR carries runtime index arithmetic: tile offsets, `ceildiv`
// trip counts and boundary extents such as `min(B, D - offset)`. Analysing one
// concrete shape sample means evaluating that arithmetic, not just substituting
// M/N/K: inside a loop the boundary still depends on the iteration, so the walk
// must bind each induction variable before a handler asks for its value.
//
// `ShapeEvaluationContext` is that explicit environment. It maps index SSA
// values -- the bound induction variables, plus any sample-materialized
// dimension value -- to concrete nonnegative integers. `evaluateShapeIndex`
// supports only the checked arithmetic and comparisons shape lowering emits;
// anything else is an explicit error, never an assumed value.
//
// This lives in the generic mapping layer because the performance DAG and plan
// cost aggregation must consume the same evaluated facts. It holds no target
// policy and performs no dialect lowering.
//
//===----------------------------------------------------------------------===//

#ifndef LLK_MAPPING_SHAPEEVALUATION_H
#define LLK_MAPPING_SHAPEEVALUATION_H

#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Error.h"

#include <cstdint>
#include <optional>

namespace mlir::llk::mapping {

/// The concrete values one sampled iteration binds to index SSA values.
class ShapeEvaluationContext {
public:
  void bind(mlir::Value value, int64_t resolved) {
    if (value)
      values[value] = resolved;
  }

  std::optional<int64_t> lookup(mlir::Value value) const {
    auto found = values.find(value);
    if (found == values.end())
      return std::nullopt;
    return found->second;
  }

  bool empty() const { return values.empty(); }
  void clear() { values.clear(); }

  llvm::DenseMap<mlir::Value, int64_t> &all() { return values; }
  const llvm::DenseMap<mlir::Value, int64_t> &all() const { return values; }

private:
  llvm::DenseMap<mlir::Value, int64_t> values;
};

/// Evaluates `value` as a nonnegative index in `context`.
///
/// Bound values and integer constants are resolved directly. Checked index
/// arithmetic -- add/sub/mul/divui/remui/ceildiv, min/max, comparisons,
/// select, index casts and partial integer extends/truncates -- is folded
/// recursively. Every division checks its divisor, every operation checks
/// overflow, and a negative result is rejected: those are exactly the
/// conditions the emitted guards exist to rule out, so reaching one means the
/// sample does not satisfy the contract.
///
/// Any other defining operation is an error naming it. A caller must treat that
/// as an unsupported *analysis* fact, never as a zero or a default.
llvm::Expected<int64_t>
evaluateShapeIndex(mlir::Value value, const ShapeEvaluationContext &context);

} // namespace mlir::llk::mapping

#endif // LLK_MAPPING_SHAPEEVALUATION_H
