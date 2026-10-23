"""Build and install the actual wheel outside the source checkout."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv

import pytest


PYTHON_ROOT = Path(__file__).resolve().parents[2] / "python"


@pytest.fixture(scope="module")
def wheel(tmp_path_factory):
    assert (PYTHON_ROOT / "pyproject.toml").is_file(), "PF01 packaging metadata missing"
    root = tmp_path_factory.mktemp("llk-wheel")
    source = root / "source"
    shutil.copytree(PYTHON_ROOT, source,
                    ignore=shutil.ignore_patterns("__pycache__", "build", "*.egg-info"))
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation",
         "--outdir", str(root / "dist"), str(source)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheels = list((root / "dist").glob("*.whl"))
    assert len(wheels) == 1
    return wheels[0]


@pytest.fixture(scope="module")
def installed_python(wheel, tmp_path_factory):
    root = tmp_path_factory.mktemp("llk-installed")
    # Symlinks retain the shared-library lookup path of standalone CPython.
    venv.EnvBuilder(with_pip=True, symlinks=True).create(root / "venv")
    python = root / "venv" / "bin" / "python"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PIP_NO_CACHE_DIR="1")
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    result = subprocess.run(
        [str(python), "-I", "-m", "pip", "install", "--no-index", "--no-deps",
         "--disable-pip-version-check", str(wheel)],
        cwd=root, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return python, root, env


def test_clean_wheel_import_without_torch(installed_python):
    python, root, env = installed_python
    result = subprocess.run(
        [str(python), "-I", "-c", """
import importlib.util, json, sys
assert importlib.util.find_spec('torch') is None
import llk
import llk.frontend
import llk.torch
assert importlib.util.find_spec('jsonschema') is None
assert llk.frontend.CompileOptions(config_files={}).backend == 'reference'
assert llk.frontend.TensorSpec('a', (2, 3), 'f32', (3, 1)).shape == (2, 3)
assert llk.torch.register() is None
assert 'torch' not in sys.modules
assert callable(llk.jit)
assert llk.torch.get_backend() is llk.torch.get_backend()
try:
    llk.torch.get_backend()(None, [])
except llk.UnsupportedFeatureError as error:
    assert error.code == 'native_backend_unavailable'
else:
    raise AssertionError('Native unavailability was not reported')
print(json.dumps({'file':llk.__file__, 'capabilities':llk.capabilities()}))
"""],
        cwd=root, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert str(root / "venv") in report["file"]
    assert str(PYTHON_ROOT) not in report["file"]
    assert report["capabilities"]["native_backend"] == {
        "available": False, "reason": "native_backend_unavailable",
    }


def test_installed_compatibility_resource_and_optional_dependencies(installed_python):
    python, root, env = installed_python
    result = subprocess.run(
        [str(python), "-I", "-c", """
import importlib.metadata as metadata, importlib.resources as resources, json
import llk
manifest=json.loads(resources.files('llk').joinpath('compatibility.json').read_text())
assert manifest['schema_version']==1
assert manifest['validation_baseline']['python']==['3.11','3.12']
assert manifest['validation_baseline']['torch']=='2.9.0'
assert llk.capabilities()['compatibility']==manifest
first=llk.capabilities()
first['native_backend']['available']=True
assert llk.capabilities()['native_backend']['available'] is False
dist=metadata.distribution('llk-compiler')
assert any('torch==2.9.0' in r.replace(' ','') and 'extra' in r for r in dist.requires)
assert not any(r.startswith('torch') and 'extra' not in r for r in dist.requires)
"""],
        cwd=root, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("name", [
    "LLKError", "FrontendSyntaxError", "UnsupportedFeatureError",
    "TensorContractError", "CompilationError", "ExecutionError",
])
def test_structured_error_hierarchy(name):
    import llk

    cls = getattr(llk, name)
    error = cls("specific diagnostic", code="example_failure")
    assert isinstance(error, llk.LLKError)
    assert error.code == "example_failure"
    assert error.message == "specific diagnostic"
    assert "specific diagnostic" in str(error)
