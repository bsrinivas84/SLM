from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, relative_path: str) -> ModuleType:
    spec = spec_from_file_location(name, REPO_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load {relative_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
