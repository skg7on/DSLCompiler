//===- Importer.h - Proven frontend import into structured source IR ------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLK_FRONTEND_IMPORTER_H
#define LLK_FRONTEND_IMPORTER_H
#include "LLK/Conversion/Workload/SourceAdapter.h"
#include "LLK/Frontend/Program.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
namespace llk::frontend {
struct PublicPortMap {
  unsigned publicPort = 0;
  unsigned nativePort = 0;
};
struct ImportedProgram {
  mlir::OwningOpRef<mlir::ModuleOp> module;
  mlir::llk::workload::SourceSelection selection;
  mlir::llk::workload::WorkloadContract contract;
  mlir::llk::perf::SearchSpace space;
  std::string adapterKey;
  std::vector<PublicPortMap> publicToNativePorts;
  unsigned internalDestinationPort = 2;
};
/// Proves the bounded access/effect/coverage contract before creating IR.
/// The caller's context must outlive the returned owning module.
llvm::Expected<ImportedProgram> importProgram(const FrontendRequest &request,
                                              mlir::MLIRContext &context);
} // namespace llk::frontend
#endif
