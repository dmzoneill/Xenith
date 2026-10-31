"""Optional audio dependencies (numpy / sounddevice) shared by the voice input modules"""

from typing import Any

np: Any = None
sd: Any = None

try:
    import numpy as _numpy
    import sounddevice as _sounddevice

    np = _numpy
    sd = _sounddevice
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Warning: sounddevice or numpy not available. Voice input will be disabled.")


def _audio_errors() -> tuple[type[BaseException], ...]:
    """Errors raised by sounddevice streams and device queries"""
    base: tuple[type[BaseException], ...] = (OSError, ValueError, RuntimeError)
    return (sd.PortAudioError, *base) if sd is not None else base


AUDIO_ERRORS = _audio_errors()

__all__ = ["AUDIO_AVAILABLE", "AUDIO_ERRORS", "np", "sd"]
