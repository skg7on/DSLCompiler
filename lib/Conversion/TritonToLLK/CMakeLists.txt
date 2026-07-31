add_mlir_library(TritonToLLKPasses
  GridToForall.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRTransformUtils
)
