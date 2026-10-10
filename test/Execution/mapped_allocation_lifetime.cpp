#include "LLK/Conversion/MappedKernelAbi.h"
#include "LLK/Runtime/MappedExecutable.h"
#include "mapped_jit_test_factory.h"

#include "gtest/gtest.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/MemoryBuffer.h"

#include <array>
#include <cstdlib>
#include <map>
#include <mutex>
#include <set>
#include <vector>

namespace {
class AllocationTracker;
thread_local AllocationTracker *activeTracker = nullptr;

class AllocationTracker {
public:
  void *allocate(size_t size) {
    void *pointer = std::malloc(size);
    record(pointer, size, false);
    return pointer;
  }

  void *alignedAllocate(size_t alignment, size_t size) {
    void *pointer = std::aligned_alloc(alignment, size);
    record(pointer, size, true);
    return pointer;
  }

  void release(void *pointer) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      auto found = live_.find(pointer);
      if (found == live_.end()) {
        ++doubleFreeCount_;
        return;
      }
      liveBytes_ -= found->second;
      live_.erase(found);
      freed_.insert(pointer);
      ++releaseCount_;
    }
    std::free(pointer);
  }

  std::set<void *> outstandingPointers() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::set<void *> result;
    for (const auto &[pointer, size] : live_) {
      (void)size;
      result.insert(pointer);
    }
    return result;
  }

  bool freedAny(llvm::ArrayRef<void *> pointers) const {
    std::lock_guard<std::mutex> lock(mutex_);
    for (void *pointer : pointers)
      if (freed_.contains(pointer))
        return true;
    return false;
  }

  uint64_t doubleFreeCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return doubleFreeCount_;
  }
  uint64_t allocateCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return allocateCount_;
  }
  uint64_t releaseCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return releaseCount_;
  }
  uint64_t mallocCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return mallocCount_;
  }
  uint64_t alignedAllocateCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return alignedAllocateCount_;
  }
  size_t maximumLiveBytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return maximumLiveBytes_;
  }

private:
  void record(void *pointer, size_t size, bool aligned) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!pointer) {
      ++doubleFreeCount_;
      return;
    }
    live_[pointer] = size;
    liveBytes_ += size;
    maximumLiveBytes_ = std::max(maximumLiveBytes_, liveBytes_);
    ++allocateCount_;
    if (aligned)
      ++alignedAllocateCount_;
    else
      ++mallocCount_;
  }

  mutable std::mutex mutex_;
  std::map<void *, size_t> live_;
  std::set<void *> freed_;
  uint64_t allocateCount_ = 0;
  uint64_t mallocCount_ = 0;
  uint64_t alignedAllocateCount_ = 0;
  uint64_t releaseCount_ = 0;
  uint64_t doubleFreeCount_ = 0;
  size_t liveBytes_ = 0;
  size_t maximumLiveBytes_ = 0;
};

void *allocateHook(size_t size) { return activeTracker->allocate(size); }
void *alignedAllocateHook(size_t alignment, size_t size) {
  return activeTracker->alignedAllocate(alignment, size);
}
void releaseHook(void *pointer) { activeTracker->release(pointer); }

class ActiveTracker {
public:
  explicit ActiveTracker(AllocationTracker &tracker) {
    EXPECT_EQ(activeTracker, nullptr);
    activeTracker = &tracker;
  }
  ~ActiveTracker() { activeTracker = nullptr; }
};

llk::InvocationBuffer2D buffer(std::vector<float> &storage) {
  return {{storage.data(), storage.data(), 0, 2, 2, 2, 1},
          llk::InvocationElementType::F32,
          storage.size() * sizeof(float)};
}

mlir::OwningOpRef<mlir::ModuleOp> loadFixture(mlir::MLIRContext &context,
                                              llvm::StringRef name) {
  auto source = llvm::MemoryBuffer::getFile(std::string(LLK_SOURCE_DIR) +
                                            "/test/Execution/Inputs/issue129/" +
                                            name.str() + ".mlir");
  if (!source)
    return {};
  return mlir::parseSourceString<mlir::ModuleOp>((*source)->getBuffer(),
                                                 &context);
}

TEST(MappedAllocationLifetime, ReleasesScratchOnEveryCallAndBorrowsBuffers) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto source = loadFixture(context, "scratch_gemm");
  ASSERT_TRUE(source);

  llvm::Expected<llk::PreparedMappedKernel> prepared =
      llk::prepareMappedKernelForInvocation(*source, "gemm");
  ASSERT_TRUE(static_cast<bool>(prepared))
      << llvm::toString(prepared.takeError());
  source = nullptr;

  AllocationTracker tracker;
  ActiveTracker active(tracker);
  llk::testing::TestAllocatorHooks hooks{allocateHook, alignedAllocateHook,
                                         releaseHook};
  llvm::Expected<std::unique_ptr<llk::MappedExecutable>> executable =
      llk::testing::createMappedExecutableForTest(std::move(*prepared), {},
                                                  hooks);
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());

  std::vector<float> lhs{1, 2, 3, 4};
  std::vector<float> rhs{5, 6, 7, 8};
  std::vector<float> output(4, -1);
  llk::InvocationBuffer2D lhsBuffer = buffer(lhs);
  llk::InvocationBuffer2D rhsBuffer = buffer(rhs);
  llk::InvocationBuffer2D outputBuffer = buffer(output);
  llvm::SmallVector<void *> callerPointers{lhs.data(), rhs.data(),
                                           output.data()};
  const std::set<void *> empty;
  for (unsigned iteration = 0; iteration < 100; ++iteration) {
    lhs[0] = static_cast<float>(iteration + 1);
    output.assign(4, -1);
    std::set<void *> before = tracker.outstandingPointers();
    ASSERT_TRUE(before.empty());
    if (llvm::Error error =
            (*executable)->invoke({lhsBuffer, rhsBuffer}, {outputBuffer}))
      FAIL() << llvm::toString(std::move(error));
    EXPECT_EQ(tracker.outstandingPointers(), before);
    EXPECT_FALSE(tracker.freedAny(callerPointers));
    EXPECT_FLOAT_EQ(output[0], lhs[0] * 5 + 2 * 7);
    EXPECT_FLOAT_EQ(output[1], lhs[0] * 6 + 2 * 8);
    EXPECT_FLOAT_EQ(output[2], 3 * 5 + 4 * 7);
    EXPECT_FLOAT_EQ(output[3], 3 * 6 + 4 * 8);
  }
  EXPECT_EQ(tracker.doubleFreeCount(), 0u);
  EXPECT_GT(tracker.allocateCount(), 100u);
  EXPECT_GT(tracker.mallocCount(), 0u);
  EXPECT_EQ(tracker.allocateCount(), tracker.releaseCount());
  EXPECT_GT(tracker.maximumLiveBytes(), 0u);

  lhsBuffer.elementType = llk::InvocationElementType::BF16;
  uint64_t allocationsBefore = tracker.allocateCount();
  if (llvm::Error error =
          (*executable)->invoke({lhsBuffer, rhsBuffer}, {outputBuffer}))
    llvm::consumeError(std::move(error));
  else
    FAIL() << "wrong dtype was accepted";
  EXPECT_EQ(tracker.allocateCount(), allocationsBefore);
  EXPECT_FALSE(tracker.freedAny(callerPointers));
  executable->reset();
  EXPECT_FALSE(tracker.freedAny(callerPointers));
}

TEST(MappedAllocationLifetime, CopiesBorrowedAndSharedResultsToCallerBuffers) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  AllocationTracker tracker;
  ActiveTracker active(tracker);
  llk::testing::TestAllocatorHooks hooks{allocateHook, alignedAllocateHook,
                                         releaseHook};

  {
    auto source = loadFixture(context, "borrowed_return");
    ASSERT_TRUE(source);
    auto prepared = llk::prepareMappedKernelForInvocation(*source, "borrow");
    ASSERT_TRUE(static_cast<bool>(prepared))
        << llvm::toString(prepared.takeError());
    source = nullptr;
    auto executable = llk::testing::createMappedExecutableForTest(
        std::move(*prepared), {}, hooks);
    ASSERT_TRUE(static_cast<bool>(executable))
        << llvm::toString(executable.takeError());

    std::vector<float> input{9, 8, 7, 6};
    std::vector<float> output(4, -1);
    llk::InvocationBuffer2D inputBuffer = buffer(input);
    llk::InvocationBuffer2D outputBuffer = buffer(output);
    llvm::SmallVector<void *> callerPointers{input.data(), output.data()};
    uint64_t allocationsBefore = tracker.allocateCount();
    if (llvm::Error error =
            (*executable)->invoke({inputBuffer}, {outputBuffer}))
      FAIL() << llvm::toString(std::move(error));
    EXPECT_EQ(output, input);
    EXPECT_EQ(tracker.allocateCount(), allocationsBefore);
    EXPECT_FALSE(tracker.freedAny(callerPointers));
  }

  {
    auto source = loadFixture(context, "shared_results");
    ASSERT_TRUE(source);
    auto prepared = llk::prepareMappedKernelForInvocation(*source, "shared");
    ASSERT_TRUE(static_cast<bool>(prepared))
        << llvm::toString(prepared.takeError());
    source = nullptr;
    auto executable = llk::testing::createMappedExecutableForTest(
        std::move(*prepared), {}, hooks);
    ASSERT_TRUE(static_cast<bool>(executable))
        << llvm::toString(executable.takeError());

    std::vector<float> first(4, -1), second(4, -1);
    llk::InvocationBuffer2D firstBuffer = buffer(first);
    llk::InvocationBuffer2D secondBuffer = buffer(second);
    llvm::SmallVector<void *> callerPointers{first.data(), second.data()};
    std::set<void *> before = tracker.outstandingPointers();
    if (llvm::Error error =
            (*executable)->invoke({}, {firstBuffer, secondBuffer}))
      FAIL() << llvm::toString(std::move(error));
    EXPECT_EQ(first, (std::vector<float>{1, 2, 3, 4}));
    EXPECT_EQ(second, first);
    EXPECT_EQ(tracker.outstandingPointers(), before);
    EXPECT_FALSE(tracker.freedAny(callerPointers));
  }
  EXPECT_EQ(tracker.doubleFreeCount(), 0u);
}

TEST(MappedAllocationLifetime,
     ReleasesNestedLoopScratchAfterTransformedCopies) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();
  auto source = loadFixture(context, "nested_alloc");
  ASSERT_TRUE(source);
  auto prepared = llk::prepareMappedKernelForInvocation(*source, "nested");
  ASSERT_TRUE(static_cast<bool>(prepared))
      << llvm::toString(prepared.takeError());
  source = nullptr;

  AllocationTracker tracker;
  ActiveTracker active(tracker);
  llk::testing::TestAllocatorHooks hooks{allocateHook, alignedAllocateHook,
                                         releaseHook};
  auto executable = llk::testing::createMappedExecutableForTest(
      std::move(*prepared), {}, hooks);
  ASSERT_TRUE(static_cast<bool>(executable))
      << llvm::toString(executable.takeError());
  std::vector<float> input{1, 2, 3, 4};
  std::vector<float> output(4, -1);
  llk::InvocationBuffer2D inputBuffer = buffer(input);
  llk::InvocationBuffer2D outputBuffer = buffer(output);
  llvm::SmallVector<void *> callerPointers{input.data(), output.data()};
  std::set<void *> before = tracker.outstandingPointers();
  if (llvm::Error error = (*executable)->invoke({inputBuffer}, {outputBuffer}))
    FAIL() << llvm::toString(std::move(error));
  EXPECT_EQ(output, input);
  EXPECT_EQ(tracker.outstandingPointers(), before);
  EXPECT_FALSE(tracker.freedAny(callerPointers));
  EXPECT_EQ(tracker.doubleFreeCount(), 0u);
}
} // namespace
