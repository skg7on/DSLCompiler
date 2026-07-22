#ifndef LLK_TRANSFORMS_SERIALPARALLELDISPATCH_H
#define LLK_TRANSFORMS_SERIALPARALLELDISPATCH_H

#include <memory>

namespace mlir {
class Pass;
}

namespace mlir {
namespace llk {

std::unique_ptr<mlir::Pass> createSerialParallelDispatchPass();

} // namespace llk
} // namespace mlir

#endif
