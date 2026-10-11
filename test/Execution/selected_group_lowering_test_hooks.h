#ifndef LLK_TEST_EXECUTION_SELECTEDGROUPLOWERINGTESTHOOKS_H
#define LLK_TEST_EXECUTION_SELECTEDGROUPLOWERINGTESTHOOKS_H

#include "LLK/Conversion/MappedCompilation.h"
#include "LLK/Mapping/MappingTarget.h"

namespace llk::testing {
llvm::Error
lowerSelectedOperationsForTest(mlir::ModuleOp module,
                               const mlir::llk::mapping::MappingTarget &target,
                               const mlir::llk::mapping::CoveringPlan &plan,
                               MappedCompilation &compilation);
} // namespace llk::testing

#endif // LLK_TEST_EXECUTION_SELECTEDGROUPLOWERINGTESTHOOKS_H
