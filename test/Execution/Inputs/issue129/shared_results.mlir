module {
  func.func @shared() -> (memref<2x2xf32>, memref<2x2xf32>) {
    %value = memref.alloc() : memref<2x2xf32>
    return %value, %value : memref<2x2xf32>, memref<2x2xf32>
  }
}
