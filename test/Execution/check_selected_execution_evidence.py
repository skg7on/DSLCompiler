#!/usr/bin/env python3
"""Fail closed unless selected AVX2 LLVM and plan evidence is complete."""

from __future__ import annotations

import argparse
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def validate_evidence(directory: Path) -> list[str]:
    errors: list[str] = []
    manifests = sorted(directory.glob("*.manifest.json"))
    if not manifests:
        return [f"no selected execution manifests found in {directory}"]

    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{manifest_path.name}: cannot read manifest: {error}")
            continue
        if not isinstance(manifest, dict):
            errors.append(f"{manifest_path.name}: manifest root must be an object")
            continue

        prefix = manifest_path.name
        required_fields = (
            "entry_symbol",
            "backend",
            "target",
            "plan_id",
            "machine_hash",
            "execution_identity",
            "architecture",
            "cpu",
            "required_features",
            "selected_groups_verified",
            "backend_groups_realized",
            "reference_groups_lowered",
            "abi_hash",
            "llvm_dialect_file",
            "llvm_ir_file",
        )
        for field in required_fields:
            if field not in manifest:
                errors.append(f"{prefix}: missing field {field}")
        if any(field not in manifest for field in required_fields):
            continue

        identity = manifest["execution_identity"]
        count_fields = (
            "selected_groups_verified",
            "backend_groups_realized",
            "reference_groups_lowered",
        )
        malformed_counts = False
        for field in count_fields:
            if type(manifest[field]) is not int:
                errors.append(f"{prefix}: {field} must be an integer")
                malformed_counts = True
        if malformed_counts:
            continue
        if manifest.get("schema_version") != 1:
            errors.append(f"{prefix}: unsupported schema version")
        if manifest["backend"] != "selected-target":
            errors.append(f"{prefix}: backend is not selected-target")
        if manifest["architecture"] not in ("x86_64", "amd64"):
            errors.append(f"{prefix}: architecture is not x86_64")
        if (
            not isinstance(manifest["required_features"], list)
            or any(not isinstance(feature, str) for feature in manifest["required_features"])
            or "avx2" not in manifest["required_features"]
        ):
            errors.append(f"{prefix}: required feature avx2 is absent")
        if not isinstance(identity, str) or "backend=selected-target" not in identity:
            errors.append(f"{prefix}: execution identity is not selected-target")
        if isinstance(identity, str) and "features=avx2" not in identity:
            errors.append(f"{prefix}: execution identity omits avx2")
        if isinstance(identity, str) and "|abi=" not in identity:
            errors.append(f"{prefix}: execution identity omits the ABI hash")
        if isinstance(identity, str) and "|lowered-ir-sha256=" not in identity:
            errors.append(f"{prefix}: execution identity omits lowered IR hash")
        if not manifest["plan_id"] or not manifest["machine_hash"]:
            errors.append(f"{prefix}: plan or machine identity is empty")
        if not manifest["abi_hash"]:
            errors.append(f"{prefix}: ABI hash is empty")
        if manifest["selected_groups_verified"] <= 0:
            errors.append(f"{prefix}: no selected groups were verified")
        if manifest["backend_groups_realized"] <= 0:
            errors.append(f"{prefix}: no selected groups were realized")
        if manifest["reference_groups_lowered"] != 0:
            errors.append(f"{prefix}: selected compilation fell back to reference groups")

        artifact_contents: dict[str, str] = {}
        for field in ("llvm_dialect_file", "llvm_ir_file"):
            name = manifest[field]
            if not isinstance(name, str) or Path(name).name != name:
                errors.append(f"{prefix}: {field} must be a local filename")
                continue
            artifact_path = directory / name
            try:
                contents = artifact_path.read_text(encoding="utf-8")
            except OSError as error:
                errors.append(f"{prefix}: cannot read {name}: {error}")
                continue
            if not contents.strip():
                errors.append(f"{prefix}: {name} is empty")
                continue
            artifact_contents[field] = contents
        dialect = artifact_contents.get("llvm_dialect_file", "")
        llvm_ir = artifact_contents.get("llvm_ir_file", "")
        if "llvm.func" not in dialect:
            errors.append(f"{prefix}: LLVM dialect module has no llvm.func")
        if "define " not in llvm_ir:
            errors.append(f"{prefix}: translated LLVM IR has no function definition")

    return errors


class SelectedExecutionEvidenceValidatorTest(unittest.TestCase):
    def write_fixture(self, directory: Path, **overrides: object) -> None:
        (directory / "kernel.llvm-dialect.mlir").write_text(
            "module { llvm.func @kernel() }\n", encoding="utf-8"
        )
        (directory / "kernel.ll").write_text(
            "define void @kernel() { ret void }\n", encoding="utf-8"
        )
        manifest: dict[str, object] = {
            "schema_version": 1,
            "entry_symbol": "kernel",
            "backend": "selected-target",
            "target": "x86-avx2",
            "plan_id": "129",
            "machine_hash": "42",
            "execution_identity": (
                "compiler=x|backend=selected-target|target=x86-avx2|features=avx2"
                "|abi=42|lowered-ir-sha256=ABCD"
            ),
            "architecture": "x86_64",
            "cpu": "haswell",
            "required_features": ["avx2"],
            "selected_groups_verified": 2,
            "backend_groups_realized": 2,
            "reference_groups_lowered": 0,
            "abi_hash": "42",
            "llvm_dialect_file": "kernel.llvm-dialect.mlir",
            "llvm_ir_file": "kernel.ll",
        }
        manifest.update(overrides)
        (directory / "kernel.manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    def test_complete_selected_avx2_evidence_passes(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_fixture(directory)
            self.assertEqual(validate_evidence(directory), [])

    def test_empty_directory_fails(self) -> None:
        with TemporaryDirectory() as temporary:
            self.assertIn("no selected execution manifests", " ".join(
                validate_evidence(Path(temporary))
            ))

    def test_reference_fallback_fails(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_fixture(directory, backend="reference", reference_groups_lowered=1)
            errors = validate_evidence(directory)
            self.assertTrue(any("not selected-target" in error for error in errors))
            self.assertTrue(any("fell back to reference" in error for error in errors))

    def test_missing_llvm_artifact_fails(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_fixture(directory)
            (directory / "kernel.ll").unlink()
            self.assertTrue(
                any("cannot read kernel.ll" in error for error in validate_evidence(directory))
            )

    def test_malformed_group_count_fails_cleanly(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.write_fixture(directory, backend_groups_realized="many")
            errors = validate_evidence(directory)
            self.assertTrue(any("backend_groups_realized" in error for error in errors))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(
            SelectedExecutionEvidenceValidatorTest
        )
        return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1
    if args.directory is None:
        parser.error("directory is required unless --self-test is used")
    errors = validate_evidence(args.directory)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"validated {len(list(args.directory.glob('*.manifest.json')))} selected evidence manifests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
