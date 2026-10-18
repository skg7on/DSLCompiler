"""PF01 regression tests against the installed, real Torch registry."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap


PYTHON_ROOT = Path(__file__).resolve().parents[2] / "python"


def run_torch_case(code, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(PYTHON_ROOT),
               PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run(
        [sys.executable, "-u", "-c", textwrap.dedent(code)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_registration_uses_the_public_singleton(tmp_path):
    run_torch_case("""
        import torch
        import torch._dynamo
        import llk.torch

        registered = torch._dynamo.lookup_backend("llk")
        assert callable(registered)
        assert registered is llk.torch.get_backend()
        assert llk.torch.register() is None
        assert torch._dynamo.lookup_backend("llk") is registered
    """, tmp_path)


def test_import_and_repeated_registration_never_reset_dynamo(tmp_path):
    run_torch_case("""
        import importlib
        from unittest.mock import patch
        import torch
        import torch._dynamo

        with patch.object(torch._dynamo, "reset", wraps=torch._dynamo.reset) as reset:
            import llk.torch
            llk.torch.register()
            llk.torch.register()
            importlib.reload(llk.torch)
            assert reset.call_count == 0
    """, tmp_path)


def test_backend_and_public_module_reload_preserve_registered_identity(tmp_path):
    run_torch_case("""
        import importlib
        import torch
        import torch._dynamo
        import llk.torch
        import llk.torch._dynamo_backend as implementation

        original = llk.torch.get_backend()
        importlib.reload(implementation)
        implementation.register()
        importlib.reload(llk.torch)
        assert implementation.get_backend() is original
        assert llk.torch.get_backend() is original
        assert torch._dynamo.lookup_backend("llk") is original
    """, tmp_path)


def test_registration_does_not_replace_another_owner(tmp_path):
    run_torch_case("""
        import torch
        import torch._dynamo
        from llk import UnsupportedFeatureError

        def foreign_backend(gm, inputs):
            raise AssertionError("Foreign backend must not execute")

        torch._dynamo.register_backend(compiler_fn=foreign_backend, name="llk")
        try:
            import llk.torch
        except UnsupportedFeatureError as error:
            assert error.code == "backend_registration_conflict"
        else:
            raise AssertionError("Registration conflict was silently ignored")
        assert torch._dynamo.lookup_backend("llk") is foreign_backend
    """, tmp_path)


def test_backend_execution_reports_unavailable_without_calling_graph(tmp_path):
    run_torch_case("""
        from unittest.mock import patch
        import torch
        import llk
        import llk.torch

        graph = torch.fx.symbolic_trace(lambda x: x + 1)
        with patch.object(graph, "forward", wraps=graph.forward) as forward:
            try:
                llk.torch.get_backend()(graph, [torch.ones(2)])
            except llk.UnsupportedFeatureError as error:
                assert error.code == "native_backend_unavailable"
            else:
                raise AssertionError("Backend silently returned eager execution")
            assert forward.call_count == 0
        assert llk.capabilities()["native_backend"]["available"] is False
    """, tmp_path)


def test_real_torch_compile_preserves_the_unavailable_error(tmp_path):
    run_torch_case("""
        import torch
        import llk
        import llk.torch
        from torch._dynamo.exc import BackendCompilerFailed

        compiled = torch.compile(lambda x: x + 1, backend="llk", fullgraph=True)
        try:
            compiled(torch.ones(2))
        except BackendCompilerFailed as error:
            assert isinstance(error.inner_exception, llk.UnsupportedFeatureError)
            assert error.inner_exception.code == "native_backend_unavailable"
        else:
            raise AssertionError("Unimplemented native backend appeared to execute")
    """, tmp_path)


def test_broken_torch_dependency_is_not_reported_as_torch_absence(tmp_path):
    run_torch_case("""
        import importlib.abc
        import sys

        class BrokenTorch(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "torch":
                    raise ModuleNotFoundError("broken Torch dependency",
                                              name="missing_torch_dependency")

        sys.meta_path.insert(0, BrokenTorch())
        try:
            import llk.torch
        except ModuleNotFoundError as error:
            assert error.name == "missing_torch_dependency"
        else:
            raise AssertionError("Broken installation was hidden as optional absence")
    """, tmp_path)
