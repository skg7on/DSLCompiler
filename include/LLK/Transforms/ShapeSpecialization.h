#ifndef LLK_TRANSFORMS_SHAPESPECIALIZATION_H
#define LLK_TRANSFORMS_SHAPESPECIALIZATION_H

#include <memory>

namespace mlir {
class Pass;
}

namespace mlir {
namespace llk {

std::unique_ptr<mlir::Pass> createShapeSpecializationPass();

} // namespace llk
} // namespace mlir

#endif
