"""Qualification of the pinned upstream Triton toolchain."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import llk.triton as triton_export
from llk._errors import UnsupportedFeatureError

LOCK_PATH = Path(triton_export.__file__).parent / "toolchain-lock.json"


def _lock():
    return json.loads(LOCK_PATH.read_text(encoding="utf-8"))


def test_lock_records_the_pinned_upstream_release():
    lock = triton_export.toolchain_lock()
    assert lock["schema_version"] == 1
    assert lock["protocol_version"] == triton_export.PROTOCOL_VERSION
    assert lock["triton_version"] == "3.6.0"
    assert lock["template_schema"] == triton_export.SCHEMA_VERSION
    assert lock["target"] == "cuda:80"
    assert len(lock["source_revision"]) == 40
    assert len(lock["llvm_revision"]) == 40
    assert lock["qualified_kernels"] == ("add", "multiply")


def test_real_toolchain_matches_the_lock(triton_path):
    identity = triton_export.qualify_toolchain(triton_python=str(triton_path))
    lock = triton_export.toolchain_lock()
    assert identity.triton_version == lock["triton_version"]
    assert identity.llvm_revision == lock["llvm_revision"]
    assert identity.source_revision == lock["source_revision"]
    assert identity.protocol_version == lock["protocol_version"]
    assert identity.target == lock["target"]


def test_lock_can_be_regenerated_from_the_actual_binaries(triton_path, tmp_path):
    regenerated_path = tmp_path / "regenerated-lock.json"
    regenerated = triton_export.generate_toolchain_lock(
        regenerated_path, triton_python=str(triton_path)
    )
    checked = triton_export.toolchain_lock()
    for field in (
        "triton_version",
        "source_revision",
        "llvm_revision",
        "protocol_version",
        "target",
        "template_schema",
    ):
        assert regenerated[field] == checked[field]
    assert regenerated["qualified_kernels"] == list(checked["qualified_kernels"])
    identity = triton_export.qualify_toolchain(
        triton_python=str(triton_path), lock_path=regenerated_path
    )
    assert identity.triton_version == "3.6.0"


def test_missing_toolchain_fails_qualification(tmp_path):
    absent = tmp_path / "no such triton"
    with pytest.raises(UnsupportedFeatureError) as caught:
        triton_export.qualify_toolchain(triton_python=str(absent))
    assert caught.value.code == "upstream_toolchain_unqualified"


@pytest.mark.parametrize(
    "field,value",
    [
        ("triton_version", "3.5.0"),
        ("llvm_revision", "0" * 40),
        ("source_revision", "0" * 40),
        ("protocol_version", 99),
    ],
)
def test_mismatched_lock_fails_qualification(triton_path, tmp_path, field, value):
    lock = _lock()
    lock[field] = value
    modified = tmp_path / "modified-lock.json"
    modified.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(UnsupportedFeatureError) as caught:
        triton_export.qualify_toolchain(
            triton_python=str(triton_path), lock_path=modified
        )
    assert caught.value.code == "upstream_toolchain_unqualified"


@pytest.mark.parametrize(
    "program",
    [
        "import torch\nimport triton\nimport llk.torch\n",
        "import llk.torch\nimport triton\nimport torch\n",
    ],
)
def test_torch_triton_llk_import_orders_coexist(triton_path, program):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(Path(triton_export.__file__).resolve().parents[2]), str(triton_path)]
    )
    script = program + "print('order-ok', triton.__version__, llk.torch.__name__)\n"
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "order-ok 3.6.0 llk.torch" in result.stdout
