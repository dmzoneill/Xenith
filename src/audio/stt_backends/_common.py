"""Helpers shared by the STT backends."""

import importlib.util
import traceback

import numpy as np

# Errors the optional PyTorch / Whisper / OpenVINO / Optimum / Transformers
# stacks raise at runtime (device, driver, model-file and tensor problems).
# Anything else is a programming error and should surface.
RUNTIME_ERRORS = (
    RuntimeError,
    ValueError,
    TypeError,
    OSError,
    AttributeError,
    KeyError,
)


def module_available(name: str) -> bool:
    """True when an optional module can be found, without importing it."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Audio as float32 scaled to a peak amplitude of 1 (silence unchanged)."""
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio


def load_failed(tag: str, error: Exception) -> bool:
    """Report a model-load failure with its traceback; always False."""
    print(f"[{tag}] Failed to load model: {error}")
    traceback.print_exc()
    return False
