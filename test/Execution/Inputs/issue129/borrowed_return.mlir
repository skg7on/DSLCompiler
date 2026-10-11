module {
  func.func @borrow(%arg: memref<2x2xf32>) -> memref<2x2xf32> {
    return %arg : memref<2x2xf32>
  }
}
