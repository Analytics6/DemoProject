import importlib.util
import os
import sys
import sysconfig


def _load_stdlib_logging():
    stdlib_path = sysconfig.get_paths().get("stdlib", "")
    module_path = os.path.join(stdlib_path, "logging", "__init__.py")
    if not os.path.exists(module_path):
        return None

    spec = importlib.util.spec_from_file_location("logging", module_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


stdlib_logging = _load_stdlib_logging()
if stdlib_logging is not None:
    globals().update(stdlib_logging.__dict__)
    __all__ = getattr(stdlib_logging, "__all__", [])
else:
    raise RuntimeError("Unable to load Python standard library logging module")
