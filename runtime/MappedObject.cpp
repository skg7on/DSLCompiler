//===- MappedObject.cpp - Target-neutral AOT object emission -------------===//
//
// Part of the LLK Compiler Project, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "LLK/Runtime/MappedObject.h"

#include "LLK/Runtime/JitCache.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"

#include <memory>
#include <string>
#include <utility>

namespace llk {
namespace {

std::string hexHash(uint64_t value) {
  return llvm::formatv("{0:x}", value).str();
}

} // namespace

llvm::Expected<std::string> emitMappedObject(PreparedMappedKernel &prepared,
                                             const MappedObjectOptions &options,
                                             llvm::raw_ostream &objectStream) {
  if (options.triple.empty())
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "AOT emission requires --target-triple");
  mlir::ModuleOp module = prepared.module.get();
  auto function = module.lookupSymbol<mlir::func::FuncOp>(prepared.entrySymbol);
  if (!function)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "AOT emission cannot find kernel '" +
                                       prepared.entrySymbol + "'");
  function->setAttr("llvm.emit_c_interface",
                    mlir::UnitAttr::get(module.getContext()));

  mlir::registerBuiltinDialectTranslation(*module.getContext());
  mlir::registerLLVMDialectTranslation(*module.getContext());
  mlir::PassManager pm(module.getContext());
  addKernelToLLVMPasses(pm);
  if (mlir::failed(pm.run(module)))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "lowering kernel '" + prepared.entrySymbol +
                                       "' to the LLVM dialect failed");

  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule)
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "translating kernel '" + prepared.entrySymbol + "' to LLVM IR failed");
  std::string wrapper = "_mlir_ciface_" + prepared.entrySymbol;
  if (!llvmModule->getFunction(wrapper))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "no C-interface wrapper '" + wrapper + "'");

  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();
  // LLVM 21 moved these APIs from a triple string to a parsed `Triple`. Guard
  // the older form, which is the only one an earlier LLVM accepts.
  llvm::Triple triple(options.triple);
  std::string error;
#if LLVM_VERSION_MAJOR >= 21
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, error);
#else
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(options.triple, error);
#endif
  if (!target)
    return llvm::createStringError(llvm::inconvertibleErrorCode(), error);
  std::string features;
  for (const std::string &feature : options.features) {
    if (!features.empty())
      features += ',';
    features += feature;
  }
  llvm::TargetOptions targetOptions;
#if LLVM_VERSION_MAJOR >= 21
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      triple, options.cpu, features, targetOptions, std::nullopt));
#else
  std::unique_ptr<llvm::TargetMachine> machine(target->createTargetMachine(
      options.triple, options.cpu, features, targetOptions, std::nullopt));
#endif
  if (!machine)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "cannot create a target machine for " +
                                       options.triple);
#if LLVM_VERSION_MAJOR >= 21
  llvmModule->setTargetTriple(triple);
#else
  llvmModule->setTargetTriple(options.triple);
#endif
  llvmModule->setDataLayout(machine->createDataLayout());

  // `addPassesToEmitFile` writes through a `raw_pwrite_stream`, while the
  // caller only promises a `raw_ostream`. Buffer the object so any stream the
  // caller has works without a downcast of unproven validity.
  llvm::SmallVector<char, 0> objectBytes;
  llvm::raw_svector_ostream objectBuffer(objectBytes);
  llvm::legacy::PassManager codegen;
  if (machine->addPassesToEmitFile(codegen, objectBuffer, nullptr,
                                   llvm::CodeGenFileType::ObjectFile))
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "target cannot emit an object file");
  codegen.run(*llvmModule);
  objectStream.write(objectBytes.data(), objectBytes.size());
  objectStream.flush();
  return wrapper;
}

llvm::Error writeMappedObjectIdentity(const MappedObjectIdentity &identity,
                                      llvm::raw_ostream &jsonStream) {
  llvm::json::Object record;
  record["schemaVersion"] = identity.schemaVersion;
  record["compilerVersion"] = identity.compilerVersion;
  record["llvmVersion"] = identity.llvmVersion;
  record["targetTriple"] = identity.targetTriple;
  record["cpu"] = identity.cpu;
  llvm::json::Array features;
  for (const std::string &feature : identity.targetFeatures)
    features.push_back(feature);
  record["targetFeatures"] = std::move(features);
  record["entrySymbol"] = identity.entrySymbol;
  record["wrapperSymbol"] = identity.wrapperSymbol;
  record["abiVersion"] = identity.abiVersion;
  record["abiHash"] = hexHash(identity.abiHash);
  record["sourceGraphHash"] = hexHash(identity.sourceGraphHash);
  record["planId"] = hexHash(identity.planId);
  record["machineHash"] = hexHash(identity.machineHash);
  record["targetHash"] = hexHash(identity.targetHash);
  record["selectedGroupsVerified"] = identity.selectedGroupsVerified;
  record["backendGroupsRealized"] = identity.backendGroupsRealized;
  record["referenceGroupsLowered"] = identity.referenceGroupsLowered;
  record["executionIdentity"] = identity.executionIdentity;
  jsonStream << llvm::json::Value(std::move(record)) << "\n";
  jsonStream.flush();
  return llvm::Error::success();
}

} // namespace llk
