module {
  func.func @nested(%arg: memref<2x2xf32>) -> memref<2x2xf32> {
    %output = memref.alloc() : memref<2x2xf32>
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c2 step %c1 {
      %scratch = memref.alloc() : memref<2x2xf32>
      memref.copy %arg, %scratch : memref<2x2xf32> to memref<2x2xf32>
      memref.copy %scratch, %output : memref<2x2xf32> to memref<2x2xf32>
    }
    return %output : memref<2x2xf32>
  }
}
