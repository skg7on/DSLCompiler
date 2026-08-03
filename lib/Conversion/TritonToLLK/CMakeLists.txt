add_mlir_library(TritonToLLKPasses
  GridToForall.cpp
  TritonToStructured.cpp
  BlockPointerToVector.cpp
  SharedMemToScratch.cpp
  AtomicToLLRT.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRVectorDialect
  MLIRMemRefDialect
  MLIRLLVMDialect
  MLIRTransformUtils
)
