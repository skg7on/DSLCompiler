#include "LLK/Conversion/LLKKernel/LLKKernelLowering.h"
#include "gtest/gtest.h"
using namespace mlir;
namespace {
class Dummy : public llk::LLKKernelLowering {
  std::string name;

public:
  explicit Dummy(std::string name) : name(std::move(name)) {}
  StringRef getOpName() const override { return name; }
  LogicalResult inferContract(Operation *,
                              llk::KernelContract &) const override {
    return failure();
  }
};
TEST(LLKKernelRegistryTest, LookupOrderCapabilitiesAndStableOwnership) {
  llk::LLKKernelRegistry registry;
  EXPECT_EQ(registry.lookup("missing"), nullptr);
  registry.add<Dummy>("first");
  auto *first = registry.lookup("first");
  ASSERT_NE(first, nullptr);
  EXPECT_FALSE(first->hasLinalgLowering());
  EXPECT_FALSE(first->hasMicroLowering());
  for (int i = 0; i < 100; ++i)
    registry.add<Dummy>(std::to_string(i));
  EXPECT_EQ(first, registry.lookup("first"));
  EXPECT_EQ(first, registry.entries().front());
  EXPECT_EQ(registry.entries()[1]->getOpName(), "0");
  EXPECT_EQ(registry.entries().size(), 101u);
  registry.add<Dummy>("first");
  EXPECT_EQ(registry.entries().size(), 101u);
  EXPECT_EQ(registry.lookup("first"), first);
}
} // namespace
