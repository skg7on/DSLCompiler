"""PF20: installed machine/rule/layout resources stay complete and untampered."""


from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


import llk
from llk import UnsupportedFeatureError
from llk._resources import (
    CONFIG_ROOT,
    manifest,
    packaged_config_root,
    validate_configuration,
)
from llk.frontend import CompileOptions

import pytest

pytestmark = pytest.mark.frontend_unit


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(llk.__file__).resolve().parent


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_packaged_copies_match_the_repository_resources():
    data = manifest()
    assert data["schema_version"] == 1
    assert data["root"] == "config"
    files = data["files"]
    assert "machines/x86-avx2-v2.yaml" in files
    assert "mapping/x86-avx2/layouts.llkmap" in files
    assert "mapping/x86-avx2/rules.llkmap" in files
    for relative, digest in files.items():
        repository = ROOT / relative
        packaged = PACKAGE / "config" / relative
        assert repository.is_file(), relative
        assert packaged.is_file(), relative
        assert _sha256(repository) == digest, relative
        assert _sha256(packaged) == digest, relative
        assert packaged.read_bytes() == repository.read_bytes(), relative


def test_manifest_covers_every_shipped_machine_and_mapping_file():
    data = manifest()
    repository_files = set()
    for pattern in ("machines/*.yaml", "mapping/*/*.llkmap"):
        for path in ROOT.glob(pattern):
            repository_files.add(path.relative_to(ROOT).as_posix())
    assert set(data["files"]) == repository_files


def test_default_compile_options_load_the_packaged_snapshot():
    options = CompileOptions()
    assert Path(options.config_root).resolve() == Path(CONFIG_ROOT).resolve()
    assert "machines/x86-avx2-v2.yaml" in options.config_files
    assert "mapping/x86-avx2/rules.llkmap" in options.config_files
    assert options.config_files["mapping/x86-avx2/rules.llkmap"].strip()


def test_packaged_root_is_inside_the_installed_package():
    root = Path(packaged_config_root())
    assert root.is_dir()
    assert root.is_relative_to(PACKAGE)


def test_missing_packaged_resource_fails_explicitly(tmp_path):
    root = tmp_path / "config"
    shutil.copytree(packaged_config_root(), root)
    (root / "machines" / "x86-avx2-v2.yaml").unlink()
    with pytest.raises(UnsupportedFeatureError) as caught:
        validate_configuration(root)
    assert caught.value.code == "native_configuration_missing"
    assert "machines/x86-avx2-v2.yaml" in caught.value.message


def test_tampered_packaged_resource_fails_explicitly(tmp_path):
    root = tmp_path / "config"
    shutil.copytree(packaged_config_root(), root)
    target = root / "mapping" / "x86-avx2" / "rules.llkmap"
    target.write_text(target.read_text(encoding="utf-8") + "\n# drift\n",
                      encoding="utf-8")
    with pytest.raises(UnsupportedFeatureError) as caught:
        validate_configuration(root)
    assert caught.value.code == "native_configuration_corrupt"
    assert "mapping/x86-avx2/rules.llkmap" in caught.value.message


def test_manifest_is_resolvable_through_importlib_resources():
    resource = Path(str(packaged_config_root()))
    data = json.loads((PACKAGE / "resources.json").read_text(encoding="utf-8"))
    assert data["files"]
    assert resource.is_dir()


def test_explicit_config_snapshot_is_not_replaced_by_package_defaults():
    options = CompileOptions(config_files={"custom/target.yaml": "target: test\n"})
    assert options.config_files == {"custom/target.yaml": "target: test\n"}
