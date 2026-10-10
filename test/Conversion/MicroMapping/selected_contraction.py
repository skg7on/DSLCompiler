#!/usr/bin/env python3
"""Selected AVX2 lowering must realize the staged bf16 GEMM as Vector IR."""

import os
import subprocess
import sys


def run(args, *, input_text=None):
    result = subprocess.run(args, input=input_text, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError("command failed: %s\n%s\n%s" %
                           (" ".join(args), result.stdout, result.stderr))
    return result.stdout


def main():
    llk_opt, llk_compile, source_dir, work_dir = sys.argv[1:]
    os.makedirs(work_dir, exist_ok=True)
    source = os.path.join(source_dir,
                          "test/Conversion/MicroMapping/matmul_e2e.mlir")
    exported = run([llk_opt, "--llk-to-micro=schedule-db=missing.json", source])
    selected_ir = run([
        llk_compile, "--mapping-target=x86-avx2",
        "--mapping-root=" + source_dir, "--mapping-mode=exact",
        "--mapping-backend=selected-target", "--mapping-stop=lowered",
        "--emit=mlir", "-",
    ], input_text=exported)

    assert "backend=selected-target" in selected_ir
    assert "backend-groups-realized=0" not in selected_ir
    assert "reference-lowered-ops=0" in selected_ir, (
        "selected staged bf16 GEMM fell back to reference lowering\n" +
        selected_ir)
    assert "micro.structural_mappings" in selected_ir and (
        "llk.avx2.selected_handoff" in selected_ir), (
            "selected movement provenance did not survive host lowering\n" +
            selected_ir)
    assert selected_ir.count('op = "micro.tile_async_copy"') >= 2 and (
        'micro.selected_connections' in selected_ir and
        'route = ["dram.0"]' in selected_ir), (
            "staged GEMM lost its ordered source-copy route provenance\n" +
            selected_ir)
    assert "vector.contract" in selected_ir or "vector.reduction" in selected_ir, (
        "the selected matrix operation did not reach Vector contraction IR\n" +
        selected_ir)
    assert "vector<8xf32>" in selected_ir, (
        "the selected AVX2 width or f32 accumulator was lost\n" + selected_ir)
    assert "arith.extf" in selected_ir, (
        "bf16 operands were not explicitly widened to f32\n" + selected_ir)
    assert "arith.truncf" in selected_ir, (
        "the f32 accumulator was not narrowed to the declared bf16 output\n" +
        selected_ir)

    transform_fixture = os.path.join(
        source_dir, "test/Execution/Inputs/issue129/required_transform.mlir")
    transformed_ir = run([llk_opt, "--micro-to-linalg", transform_fixture])
    assert "linalg.generic" in transformed_ir, (
        "the required transform was lowered as an identity copy\n" +
        transformed_ir)
    assert "affine_map<(d0, d1) -> (d1, d0)>" in transformed_ir, (
        "the nonidentity physical map was discarded\n" + transformed_ir)
    assert "micro.transform" not in transformed_ir

    selected_transform_ir = run([
        llk_compile, "--mapping-target=x86-avx2",
        "--mapping-root=" + source_dir, "--mapping-mode=exact",
        "--mapping-backend=selected-target", "--mapping-stop=lowered",
        "--emit=mlir", transform_fixture,
    ])
    assert "llk.backend_connections_realized = 1" in selected_transform_ir, (
        "the selected connection transform was not realized by the AVX2 "
        "backend\n" + selected_transform_ir)
    assert "vector.transfer_write" in selected_transform_ir and (
        "permutation_map = #map" in selected_transform_ir and
        "#map = affine_map<(d0, d1) -> (d1, d0)>" in
        selected_transform_ir), (
            "the selected transposed layout did not reach vector transfers\n" +
            selected_transform_ir)
    assert "micro.transform" not in selected_transform_ir

    reduction_fixture = os.path.join(
        source_dir, "test/Execution/Inputs/issue129/reduce_sum.mlir")
    reduction_ir = run([
        llk_compile, "--mapping-target=x86-avx2",
        "--mapping-root=" + source_dir, "--mapping-mode=exact",
        "--mapping-backend=selected-target", "--mapping-stop=lowered",
        "--emit=mlir", reduction_fixture,
    ])
    assert "vector.multi_reduction" in reduction_ir or (
        "vector.reduction" in reduction_ir), (
            "the selected sum did not reach Vector reduction IR\n" +
            reduction_ir)
    assert "vector<8xf32>" in reduction_ir
    assert "target-lowered-ops=1" in reduction_ir and (
        "reference-lowered-ops=0" in reduction_ir), (
            "selected sum reduction fell back to reference lowering\n" +
            reduction_ir)

    bf16_fixture = os.path.join(
        source_dir, "test/Execution/Inputs/issue129/matmul_bf16.mlir")
    bf16_ir = run([
        llk_compile, "--mapping-target=x86-avx2",
        "--mapping-root=" + source_dir, "--mapping-mode=exact",
        "--mapping-backend=selected-target", "--mapping-stop=lowered",
        "--emit=mlir", bf16_fixture,
    ])
    assert "vector.contract" in bf16_ir and "vector<8x8xf32>" in bf16_ir and (
        "arith.extf" in bf16_ir), (
            "selected bf16 MMA did not widen exactly into f32 Vector IR\n" +
            bf16_ir)
    assert "target-lowered-ops=1" in bf16_ir and (
        "reference-lowered-ops=0" in bf16_ir), (
            "selected bf16 MMA fell back to reference lowering\n" + bf16_ir)

    f32_fixture = os.path.join(
        source_dir, "test/Execution/Inputs/issue129/matmul_f32.mlir")
    f32_ir = run([
        llk_compile, "--mapping-target=x86-avx2",
        "--mapping-root=" + source_dir, "--mapping-mode=exact",
        "--mapping-backend=selected-target", "--mapping-stop=lowered",
        "--emit=mlir", f32_fixture,
    ])
    assert "vector.contract" in f32_ir and "vector<8x8xf32>" in f32_ir, (
        "selected f32 MMA did not reach the declared AVX2 contraction width\n" +
        f32_ir)
    assert "target-lowered-ops=1" in f32_ir and (
        "reference-lowered-ops=0" in f32_ir), (
            "selected f32 MMA fell back to reference lowering\n" + f32_ir)
    assert "iter_args(%arg4 = %arg2)" in f32_ir and (
        "vector.transfer_read %subview_1" in f32_ir), (
            "the f32 contraction did not accumulate into the initialized "
            "output tile\n" + f32_ir)


if __name__ == "__main__":
    main()
