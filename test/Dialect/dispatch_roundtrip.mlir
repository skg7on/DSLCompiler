// RUN: llk-opt %s | FileCheck %s

func.func @dispatch_test(%x: tensor<?x4096xbf16>, %init: tensor<?x4096xbf16>) -> tensor<?x4096xbf16> {
  %0 = "llk.dispatch"(%x, %init) ({
  ^bb0(%arg0: tensor<?x4096xbf16>, %arg1: tensor<?x4096xbf16>):
    "llk.return"(%arg0) : (tensor<?x4096xbf16>) -> ()
  }) {M = 0 : i64, N = 4096 : i64, K = 4096 : i64} : (tensor<?x4096xbf16>, tensor<?x4096xbf16>) -> tensor<?x4096xbf16>
  return %0 : tensor<?x4096xbf16>
}
// CHECK: llk.dispatch
// CHECK: M = 0
