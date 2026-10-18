#!/usr/bin/env python3
"""Regression cases for the selected-lowering evidence acceptance gate."""

import contextlib
import copy
import io
import json
import pathlib
import tempfile
import unittest

import framework_manifest_validator as validator


def manifest_fixture():
    rows = []
    for symbol, filename, count, shape in (
            ("reduce_sum", "reduce.mlir", 1, [8]),
            ("add_mul", "add_mul.mlir", 2, [8, 8]),
            ("add_mul_reduce", "add_mul_reduce.mlir", 3, [8])):
        identity = ("compiler=0.1.0+test|backend=selected-target|target=x86-avx2|"
                    "machine-content=123|architecture=x86_64|cpu=generic|"
                    "features=avx2|math-mode=default")
        replay = f"graph=0x{count}1|binding=0x0|plan=0x{count}2"
        report = (f"schema_version: 1\nmachine: x86-avx2\nkernel: {symbol}\n"
                  "level: 1\ncomplete: true\nincomplete_reasons: []\n"
                  "totals:\n  flops: 0\n  bytes:\n    sram: 256\n"
                  "operations:\n  vector: 1\nbounds:\n  compute_cycles: 1\n"
                  "  memory_cycles: 0\npredicted_cycles: 8\n"
                  "tiles:\n  live_bytes:\n    sram: 256\n  layouts: {}\n"
                  "  owners: {}\ncapacity_violations: []\n")
        invocation = "not_checked" if len(shape) == 2 else {
            "status": "unsupported",
            "reason": "runtime_capability: mapped ABI v1 requires rank-two ports",
        }
        rows.append({
            "source": "test/Execution/Inputs/framework/" + filename,
            "symbol": symbol,
            "target": {"name": "x86-avx2",
                       "machine": "machines/x86-avx2-v2.yaml"},
            "compiler": identity,
            "mode": {"backend": "selected-target", "stop": "lowered"},
            "plan_id": f"0x{count}2", "replay_identity": replay,
            "counters": {"target_lowered_ops": count,
                         "selected_groups_verified": count,
                         "backend_groups_realized": count,
                         "reference_lowered_ops": 0,
                         "reference_groups_lowered": 0},
            "stages": {"mapping": "supported", "physical_plan": "supported",
                       "selected_lowering": "supported",
                       "reference_lowering": "not_checked",
                       "invocation": invocation},
            "analysis": {"format": "micro-perf-yaml", "yaml": report,
                         "compiler": identity, "replay_identity": replay,
                         "plan_id": f"0x{count}2"},
            "output_shape": shape,
            "numeric": {"status": "not_checked", "elements": 64 if
                        len(shape) == 2 else 8},
        })
    return {"schema_version": 1, "kind": "framework-selected-evidence",
            "host": "test-host", "rows": rows}


class ManifestValidatorTest(unittest.TestCase):
    def verdict(self, manifest):
        with tempfile.TemporaryDirectory(prefix="framework-manifest-test-") as tmp:
            path = pathlib.Path(tmp) / "evidence.manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                try:
                    code = validator.main(tmp)
                except (AttributeError, TypeError, ValueError) as error:
                    self.fail(f"validator crashed on malformed input: {error}")
            return code, output.getvalue()

    def rejects(self, manifest):
        code, output = self.verdict(manifest)
        self.assertEqual(code, 1, output)

    def test_complete_unique_cases_pass(self):
        code, output = self.verdict(manifest_fixture())
        self.assertEqual(code, 0, output)

    def test_missing_required_case_fails(self):
        manifest = manifest_fixture()
        manifest["rows"].pop()
        self.rejects(manifest)

    def test_duplicate_case_cannot_replace_missing_case(self):
        manifest = manifest_fixture()
        manifest["rows"][2] = copy.deepcopy(manifest["rows"][0])
        self.rejects(manifest)

    def test_wrong_source_symbol_pair_fails(self):
        manifest = manifest_fixture()
        manifest["rows"][0]["source"] = manifest["rows"][1]["source"]
        self.rejects(manifest)

    def test_selected_groups_require_verification_and_realization(self):
        for counter in ("selected_groups_verified", "backend_groups_realized"):
            for value in (0, -1, True, "1", None):
                with self.subTest(counter=counter, value=value):
                    manifest = manifest_fixture()
                    manifest["rows"][0]["counters"][counter] = value
                    self.rejects(manifest)

    def test_every_verified_group_must_be_realized(self):
        manifest = manifest_fixture()
        manifest["rows"][2]["counters"]["backend_groups_realized"] = 2
        self.rejects(manifest)

    def test_reference_fallback_fails(self):
        for counter in ("reference_lowered_ops", "reference_groups_lowered"):
            with self.subTest(counter=counter):
                manifest = manifest_fixture()
                manifest["rows"][0]["counters"][counter] = 1
                self.rejects(manifest)

    def test_false_completeness_cannot_be_hidden_by_comment(self):
        manifest = manifest_fixture()
        row = manifest["rows"][0]
        row["analysis"]["yaml"] = row["analysis"]["yaml"].replace(
            "complete: true", "complete: false\n# complete: true")
        self.rejects(manifest)

    def test_report_requires_actual_typed_traffic_and_storage_maps(self):
        for fragment in ("  live_bytes:\n    sram: -1",
                         "  live_bytes:\n    sram: true", "  live_bytes: 4",
                         "  live_bytes: []"):
            with self.subTest(fragment=fragment):
                manifest = manifest_fixture()
                report = manifest["rows"][0]["analysis"]["yaml"]
                manifest["rows"][0]["analysis"]["yaml"] = report.replace(
                    "  live_bytes:\n    sram: 256", fragment)
                self.rejects(manifest)

    def test_empty_counter_maps_are_valid(self):
        manifest = manifest_fixture()
        for row in manifest["rows"]:
            row["analysis"]["yaml"] = row["analysis"]["yaml"].replace(
                "  bytes:\n    sram: 256", "  bytes: {}").replace(
                "  live_bytes:\n    sram: 256", "  live_bytes: {}")
        self.assertEqual(self.verdict(manifest)[0], 0)

    def test_duplicate_complete_key_fails(self):
        manifest = manifest_fixture()
        manifest["rows"][0]["analysis"]["yaml"] += "complete: true\n"
        self.rejects(manifest)

    def test_nonempty_incomplete_or_capacity_reasons_fail(self):
        for key in ("incomplete_reasons", "capacity_violations"):
            with self.subTest(key=key):
                manifest = manifest_fixture()
                row = manifest["rows"][0]
                row["analysis"]["yaml"] = row["analysis"]["yaml"].replace(
                    key + ": []", key + ':\n  - "unresolved resource"')
                self.rejects(manifest)

    def test_report_kernel_and_machine_match_row(self):
        for old, new in (("kernel: reduce_sum", "kernel: other"),
                         ("machine: x86-avx2", "machine: other")):
            with self.subTest(new=new):
                manifest = manifest_fixture()
                row = manifest["rows"][0]
                row["analysis"]["yaml"] = row["analysis"]["yaml"].replace(old, new)
                self.rejects(manifest)

    def test_wrong_mode_or_compiler_identity_fails(self):
        for section, key, value in (
                ("mode", "backend", "reference"),
                ("mode", "stop", "mapped-micro"),
                ("target", "name", "generic-ai-accel")):
            with self.subTest(section=section, key=key):
                manifest = manifest_fixture()
                manifest["rows"][0][section][key] = value
                self.rejects(manifest)
        manifest = manifest_fixture()
        manifest["rows"][0]["compiler"] = manifest["rows"][0]["compiler"].replace(
            "backend=selected-target", "backend=reference")
        self.rejects(manifest)

    def test_row_plan_matches_replay_and_analyzed_artifact(self):
        for section, key in ((None, "plan_id"), ("analysis", "plan_id"),
                             ("analysis", "replay_identity"),
                             ("analysis", "compiler")):
            with self.subTest(section=section, key=key):
                manifest = manifest_fixture()
                row = manifest["rows"][0]
                (row[section] if section else row)[key] = "unrelated"
                self.rejects(manifest)

    def test_unsupported_stage_requires_reason(self):
        manifest = manifest_fixture()
        manifest["rows"][0]["stages"]["invocation"] = "unsupported"
        self.rejects(manifest)

    def test_selected_only_evidence_cannot_credit_reference_or_numeric_runs(self):
        manifest = manifest_fixture()
        manifest["rows"][0]["stages"]["reference_lowering"] = "supported"
        self.rejects(manifest)
        manifest = manifest_fixture()
        manifest["rows"][2]["numeric"]["verified_by"] = "unrelated-test"
        self.rejects(manifest)

    def test_rank_one_invocation_and_output_semantics_are_explicit(self):
        for key, value in (("output_shape", [8, 8]), ("numeric", {
                "status": "not_checked", "elements": 64}),
                ("stages", {"mapping": "supported", "physical_plan": "supported",
                            "selected_lowering": "supported",
                            "reference_lowering": "not_checked",
                            "invocation": "not_checked"})):
            with self.subTest(key=key):
                manifest = manifest_fixture()
                manifest["rows"][2][key] = value
                self.rejects(manifest)

    def test_output_shape_is_an_integer_extent(self):
        manifest = manifest_fixture()
        manifest["rows"][0]["output_shape"] = [8.0]
        self.rejects(manifest)

    def test_malformed_shapes_fail_without_traceback(self):
        for manifest in ([], {"schema_version": 1, "rows": {}},
                         {**manifest_fixture(), "rows": [None]},
                         {**manifest_fixture(), "rows": "bad"}):
            with self.subTest(manifest=manifest):
                self.rejects(manifest)
        for field in ("target", "mode", "counters", "stages", "analysis"):
            with self.subTest(field=field):
                manifest = manifest_fixture()
                manifest["rows"][0][field] = "bad"
                self.rejects(manifest)


if __name__ == "__main__":
    unittest.main()
