#!/usr/bin/env python3
"""Require target operation coverage for CLI search and frozen replay."""

import argparse
import pathlib
import subprocess
import tempfile
import unittest


def run(command):
    return subprocess.run([str(arg) for arg in command], capture_output=True,
                          text=True)


class FrameworkCoverageCli(unittest.TestCase):
    def test_non_f32_operations_are_rejected_before_any_compile_stop(self):
        with tempfile.TemporaryDirectory(prefix="llk-coverage-") as temp:
            work = pathlib.Path(temp)
            for fixture in ("reduce.mlir", "add_mul.mlir"):
                source = work / fixture
                source.write_text((SOURCE / "test/Execution/Inputs/framework" /
                                   fixture).read_text().replace("f32", "bf16"))
                report = work / (fixture + ".json")
                # A declaration-only mapping target can produce a frozen
                # selection without the package emitter's operation checks.
                # Replay must revalidate those checks with the real package.
                mapped = run([
                    OPT,
                    "--micro-map=target=generic-ai-accel "
                    f"machine={SOURCE}/machines/generic-ai-accel-v2.yaml "
                    f"layouts={SOURCE}/mapping/generic-ai-accel/layouts.llkmap "
                    f"rules={SOURCE}/mapping/generic-ai-accel/rules.llkmap "
                    "emitters=accel_vector_add,accel_vector_mul,accel_mxu,"
                    "accel_copy,accel_reduce_sum mode=exact top-k=1 "
                    f"report={report} report-only=1", source,
                ])
                self.assertEqual(mapped.returncode, 0, mapped.stderr)
                for replay in (False, True):
                    for stop in ("mapped-micro", "lowered"):
                        with self.subTest(fixture=fixture, replay=replay,
                                          stop=stop):
                            command = [
                                COMPILE, "--mapping-target=generic-ai-accel",
                                f"--mapping-root={SOURCE}",
                                "--mapping-mode=exact",
                                "--mapping-backend=reference",
                                f"--mapping-stop={stop}", "--emit=mlir",
                            ]
                            if replay:
                                command.append(f"--plan-report={report}")
                            result = run([*command, source])
                            self.assertNotEqual(result.returncode, 0,
                                                result.stdout[:500])
                            self.assertIn("target_coverage:", result.stderr)
                            self.assertIn("f32", result.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("llk_compile", type=pathlib.Path)
    parser.add_argument("llk_opt", type=pathlib.Path)
    parser.add_argument("source_dir", type=pathlib.Path)
    args, remaining = parser.parse_known_args()
    COMPILE = args.llk_compile.resolve()
    OPT = args.llk_opt.resolve()
    SOURCE = args.source_dir.resolve()
    unittest.main(argv=[__file__, *remaining])
