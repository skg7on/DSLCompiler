add_llk_library(TritonToLLKPasses
  GridToForall.cpp
  TritonToStructured.cpp
  BlockPointerToVector.cpp
  TritonCPUVerifier.cpp
  SharedMemToScratch.cpp
  AtomicToLLRT.cpp
  LINK_LIBRARIES LLK::MLIR
)
