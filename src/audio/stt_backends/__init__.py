"""Speech-to-Text (STT) backends for Xenith

Provides a unified interface for different STT engines:
- OpenAI Whisper (CUDA/CPU) - High accuracy, GPU accelerated
- OpenVINO Whisper (NPU/GPU/CPU) - Ultra-low power on Intel hardware

Usage:
    from audio.stt_backends import get_stt_backend, list_available_backends

    # Get the configured backend
    backend = get_stt_backend("openvino", device="NPU")

    # Transcribe audio
    result = backend.transcribe(audio_array)
"""

from .base import STTBackend, STTResult
from .factory import get_stt_backend, list_available_backends, print_backend_status

__all__ = [
    "STTBackend",
    "STTResult",
    "get_stt_backend",
    "list_available_backends",
    "print_backend_status",
]
