module {
  func.func @gemm(%lhs: memref<2x2xf32>, %rhs: memref<2x2xf32>) -> memref<2x2xf32> {
    %output = memref.alloc() : memref<2x2xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    scf.for %i = %c0 to %c2 step %c1 {
      %scratch = memref.alloc() : memref<2xf32>
      scf.for %j = %c0 to %c2 step %c1 {
        %zero = arith.constant 0.0 : f32
        %sum = scf.for %k = %c0 to %c2 step %c1 iter_args(%acc = %zero) -> (f32) {
          %a = memref.load %lhs[%i, %k] : memref<2x2xf32>
          %b = memref.load %rhs[%k, %j] : memref<2x2xf32>
          %product = arith.mulf %a, %b : f32
          %next = arith.addf %acc, %product : f32
          scf.yield %next : f32
        }
        memref.store %sum, %scratch[%j] : memref<2xf32>
      }
      scf.for %j = %c0 to %c2 step %c1 {
        %value = memref.load %scratch[%j] : memref<2xf32>
        memref.store %value, %output[%i, %j] : memref<2x2xf32>
      }
    }
    return %output : memref<2x2xf32>
  }
}
