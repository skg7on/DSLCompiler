#include "LLK/Mapping/CodegenRequirements.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/Error.h"

#include "gtest/gtest.h"

#include <string>

using mlir::llk::mapping::TargetCodegenRequirements;

namespace {
TargetCodegenRequirements avx2Requirements() {
  return {"x86_64", "haswell", {"avx2"}};
}
} // namespace

TEST(MappedCodegenRequirements, RejectsWrongArchitecture) {
  llvm::StringMap<bool> features;
  features["avx2"] = true;
  llvm::Error error =
      checkHostRequirements(avx2Requirements(), "aarch64", features);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("x86_64"), std::string::npos);
}

TEST(MappedCodegenRequirements, RejectsMissingRequiredAVX2) {
  llvm::StringMap<bool> features;
  features["avx2"] = false;
  llvm::Error error =
      checkHostRequirements(avx2Requirements(), "x86_64", features);
  ASSERT_TRUE(static_cast<bool>(error));
  EXPECT_NE(llvm::toString(std::move(error)).find("avx2"), std::string::npos);
}

TEST(MappedCodegenRequirements, AcceptsRequiredAVX2) {
  llvm::StringMap<bool> features;
  features["avx2"] = true;
  EXPECT_FALSE(static_cast<bool>(
      checkHostRequirements(avx2Requirements(), "x86_64", features)));
}

TEST(MappedCodegenRequirements, ChecksOptionalFmaOnlyWhenRequired) {
  llvm::StringMap<bool> features;
  features["avx2"] = true;
  TargetCodegenRequirements avx2AndFma = {"x86_64", "haswell", {"avx2", "fma"}};
  llvm::Error rejected = checkHostRequirements(avx2AndFma, "x86_64", features);
  ASSERT_TRUE(static_cast<bool>(rejected));
  llvm::consumeError(std::move(rejected));

  features["fma"] = true;
  EXPECT_FALSE(
      static_cast<bool>(checkHostRequirements(avx2AndFma, "x86_64", features)));
}
