#ifndef LLK_TRANSFORMS_LINEARIZEFORALL_H
#define LLK_TRANSFORMS_LINEARIZEFORALL_H

#include <memory>

namespace mlir {
class Pass;
}

namespace mlir {
namespace llk {

std::unique_ptr<mlir::Pass> createLinearizeForallPass();

} // namespace llk
} // namespace mlir

#endif
