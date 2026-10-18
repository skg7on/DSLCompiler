#!/usr/bin/env python3
"""Check that the real evidence producer claims only the stages it runs."""

import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


class SelectedEvidenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="framework-evidence-")
        producer = pathlib.Path(__file__).with_name("framework_selected_evidence.py")
        subprocess.run([sys.executable, str(producer), *TOOL_ARGUMENTS,
                        cls.directory.name], check=True, capture_output=True,
                       text=True)
        manifest = pathlib.Path(cls.directory.name) / \
            "framework_selected_evidence.manifest.json"
        cls.rows = {row["symbol"]: row
                    for row in json.loads(manifest.read_text())["rows"]}

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_selected_only_compilation_does_not_claim_reference_lowering(self):
        for symbol, row in self.rows.items():
            with self.subTest(symbol=symbol):
                self.assertEqual(row["stages"]["reference_lowering"],
                                 "not_checked")

    def test_lowering_does_not_claim_numerical_verification(self):
        for symbol, row in self.rows.items():
            with self.subTest(symbol=symbol):
                self.assertEqual(row.get("numeric", {}).get("status"),
                                 "not_checked")
                self.assertNotIn("verified_by", row.get("numeric", {}))

    def test_rank_one_composition_has_eight_outputs_and_abi_refusal(self):
        row = self.rows["add_mul_reduce"]
        self.assertEqual(row.get("output_shape"), [8])
        self.assertEqual(row["numeric"]["elements"], 8)
        self.assertEqual(row["numeric"]["formula"],
                         "sum_j((left[i,j] + right[i,j]) * left[i,j])")
        self.assertEqual(row["stages"]["invocation"]["status"], "unsupported")
        self.assertIn("rank-two", row["stages"]["invocation"]["reason"])


if __name__ == "__main__":
    TOOL_ARGUMENTS = sys.argv[1:4]
    del sys.argv[1:4]
    unittest.main()
