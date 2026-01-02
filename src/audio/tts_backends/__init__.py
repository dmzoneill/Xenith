"""TTS backend implementations

This module provides a pluggable TTS (Text-to-Speech) backend system.
Currently supported backends:

- MeloTTS: High-quality OpenVINO-based TTS with NPU acceleration for BERT preprocessing

Usage:
    from src.audio.tts_backends import get_tts_backend, print_tts_status

    # Auto-select best available backend
    tts = get_tts_backend()
    if tts and tts.load():
        result = tts.synthesize("Hello world")
        print(f"Audio saved to: {result.audio_path}")

    # Or specify backend and device
    tts = get_tts_backend(backend="melotts", device="npu")
"""

from .base import TTSBackend, TTSResult
from .factory import (
    get_tts_backend,
    list_available_backends,
    print_tts_status,
    register_backend,
)

__all__ = [
    "TTSBackend",
    "TTSResult",
    "get_tts_backend",
    "list_available_backends",
    "print_tts_status",
    "register_backend",
]
