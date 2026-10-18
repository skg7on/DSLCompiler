#!/usr/bin/env python3
"""Exercise the generic-boundary guard with real source and CMake files."""

import contextlib
import io
import pathlib
import tempfile
import unittest

import framework_boundary_scan as scanner


class BoundaryScanTest(unittest.TestCase):
    def scan(self, source, link="LLKMachine LLVM"):
        with tempfile.TemporaryDirectory(prefix="framework-boundary-test-") as tmp:
            root = pathlib.Path(tmp)
            for directory in scanner.GENERIC_DIRECTORIES:
                (root / directory).mkdir(parents=True)
            (root / "lib/Mapping/policy.cpp").write_text(source)
            for path in scanner.GENERIC_CMAKELISTS:
                (root / path).write_text(f"target_link_libraries(Core {link})\n")
            with contextlib.redirect_stdout(io.StringIO()):
                return scanner.main(tmp)

    def test_comments_and_neutral_policy_pass(self):
        self.assertEqual(self.scan(
            '// "avx2_vector_add" llk.fused_swiglu\n'
            '/* target.name() == "x86-avx2" */\n'
            'return target.rules().lookup(key);\n'), 0)

    def test_actual_opaque_target_and_operator_identifiers_fail(self):
        for token in ("avx2_vector_add", "avx2_mma", "llk.fused_swiglu",
                      "x86-avx2", "generic-ai-accel", "accel_vector_mul",
                      "linalg.matmul", "llk.attention"):
            with self.subTest(token=token):
                self.assertEqual(self.scan(
                    f'if (key == "{token}") return specialPolicy();\n'), 1)

    def test_reverse_library_dependencies_fail(self):
        for library in ("LLKPerf", "LLKRuntime", "LLKMappedCompilation"):
            with self.subTest(library=library):
                self.assertEqual(self.scan("", library), 1)


if __name__ == "__main__":
    unittest.main()
