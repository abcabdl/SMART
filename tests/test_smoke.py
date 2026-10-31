import importlib
import sys
from pathlib import Path


def test_import_demo_runner() -> None:
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    sys.path.insert(0, str(src_path))
    importlib.import_module("execution.demo_runner")
