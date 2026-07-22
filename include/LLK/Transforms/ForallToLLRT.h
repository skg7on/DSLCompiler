#ifndef LLK_TRANSFORMS_FORALLTOLLRT_H
#define LLK_TRANSFORMS_FORALLTOLLRT_H

#include <memory>

namespace mlir {
class Pass;
}

namespace mlir {
namespace llk {

std::unique_ptr<mlir::Pass> createForallToLLRTPass();

} // namespace llk
} // namespace mlir

#endif
