from .harmony import Harmony, run_harmony

__version__ = '2.0.0'


def __getattr__(name):
    if name == "compute_lisi":
        from .lisi import compute_lisi
        return compute_lisi
    raise AttributeError(f"module 'harmonypy' has no attribute {name}")
