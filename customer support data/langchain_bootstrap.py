import importlib.util
import sys
import sysconfig
from pathlib import Path


def ensure_stdlib_logging() -> None:
    """
    Force-load Python stdlib logging so local `logging/` directory
    does not shadow it during third-party imports.
    """
    existing = sys.modules.get("logging")
    if existing is not None and hasattr(existing, "getLogger"):
        return

    stdlib_path = Path(sysconfig.get_paths()["stdlib"]) / "logging" / "__init__.py"
    spec = importlib.util.spec_from_file_location("logging", stdlib_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load stdlib logging module.")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["logging"] = module
