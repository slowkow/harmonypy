from .harmony import Harmony, run_harmony
from .lisi import compute_lisi

# JAX-accelerated version (optional)
try:
    from .harmony_jax import HarmonyJAX, run_harmony_jax, JAX_AVAILABLE
except ImportError:
    JAX_AVAILABLE = False
    HarmonyJAX = None
    run_harmony_jax = None

__version__ = '0.0.10'
