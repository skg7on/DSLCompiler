add_mlir_library(TritonToLLKPasses
  GridToForall.cpp
  TritonToStructured.cpp
  BlockPointerToVector.cpp
  TritonCPUVerifier.cpp
  SharedMemToScratch.cpp
  AtomicToLLRT.cpp

)

mlir_target_link_libraries(TritonToLLKPasses PUBLIC
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
