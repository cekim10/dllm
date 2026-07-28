"""
Core dLLM modules with lazy submodule loading.

Run from repo root:
  python -c "from dllm.core import samplers; print(samplers)"
"""

from importlib import import_module

__all__ = ["eval", "samplers", "schedulers", "trainers"]


def __getattr__(name: str):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
