#ifndef LLK_TRANSFORMS_SCHEDULESELECTION_H
#define LLK_TRANSFORMS_SCHEDULESELECTION_H

#include <memory>

namespace mlir {
class Pass;
}

namespace mlir {
namespace llk {

std::unique_ptr<mlir::Pass> createScheduleSelectionPass();

} // namespace llk
} // namespace mlir

#endif
