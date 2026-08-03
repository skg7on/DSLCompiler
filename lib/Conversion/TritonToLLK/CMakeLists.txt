add_mlir_library(TritonToLLKPasses
  GridToForall.cpp
  TritonToStructured.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRTransformUtils
)
