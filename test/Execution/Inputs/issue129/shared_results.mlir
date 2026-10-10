module {
  func.func @shared() -> (memref<2x2xf32>, memref<2x2xf32>) {
    %value = memref.alloc() : memref<2x2xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %v1 = arith.constant 1.0 : f32
    %v2 = arith.constant 2.0 : f32
    %v3 = arith.constant 3.0 : f32
    %v4 = arith.constant 4.0 : f32
    memref.store %v1, %value[%c0, %c0] : memref<2x2xf32>
    memref.store %v2, %value[%c0, %c1] : memref<2x2xf32>
    memref.store %v3, %value[%c1, %c0] : memref<2x2xf32>
    memref.store %v4, %value[%c1, %c1] : memref<2x2xf32>
    return %value, %value : memref<2x2xf32>, memref<2x2xf32>
  }
}
