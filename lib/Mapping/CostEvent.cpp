//===- CostEvent.cpp - Shared cost-event vocabulary -----------------------===//

#include "LLK/Mapping/CostEvent.h"

#include <array>

namespace mlir::llk::mapping {

namespace {
struct KindInfo {
  CostEventKind kind;
  llvm::StringLiteral name;
};

/// Declaration order is the canonical order.
constexpr std::array<KindInfo, 5> kKinds{{
    {CostEventKind::Compute, "compute"},
    {CostEventKind::TransferHop, "transfer_hop"},
    {CostEventKind::Transform, "transform"},
    {CostEventKind::Synchronization, "synchronization"},
    {CostEventKind::Capacity, "capacity"},
}};
} // namespace

llvm::StringRef stringifyCostEventKind(CostEventKind kind) {
  for (const KindInfo &info : kKinds)
    if (info.kind == kind)
      return info.name;
  return "";
}

std::optional<CostEventKind> symbolizeCostEventKind(llvm::StringRef text) {
  for (const KindInfo &info : kKinds)
    if (info.name == text)
      return info.kind;
  return std::nullopt;
}

} // namespace mlir::llk::mapping
