import importlib.util
import sys
import types
from pathlib import Path


def test_app_module_exposes_spaces_entrypoint_without_auto_start(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    app_path = repo_root / "app.py"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("startup should not run during module import")

    fake_uvicorn = types.SimpleNamespace(run=fail_if_called)
    fake_spaces = types.SimpleNamespace(GPU=lambda fn: fn)

    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setitem(sys.modules, "spaces", fake_spaces)

    spec = importlib.util.spec_from_file_location("hf_space_app", app_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert callable(getattr(module, "app", None))
