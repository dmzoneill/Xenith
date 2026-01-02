"""LLM backend implementations

This module provides a pluggable LLM (Language Model) backend system.
Currently supported backends:

- OpenVINO: Run LLMs on Intel NPU/GPU/CPU with INT4 quantization

Usage:
    from src.audio.llm_backends import get_llm_backend, print_llm_status

    # Auto-select best available backend
    llm = get_llm_backend(model="qwen2.5-1.5b")
    if llm and llm.load():
        result = llm.generate("What is the weather like?")
        print(result.text)

    # Or specify backend and device
    llm = get_llm_backend(backend="openvino", model="phi-3-mini", device="npu")

Available models (NPU-compatible):
    - qwen3-0.6b: Smallest, basic commands (~1-2W)
    - tinyllama-1.1b: Small chat model (~2-3W)
    - qwen2.5-1.5b: Recommended for voice assistant (~3-5W)
    - deepseek-r1-1.5b: Reasoning-focused (~3-5W)
    - phi-3-mini: High quality (~5-10W)
    - phi-3.5-mini: Latest Phi model (~5-10W)

GPU-only models:
    - mistral-7b: High quality
    - deepseek-r1-7b: Reasoning model
"""

from .base import LLMBackend, LLMResult, LLMConfig, Message
from .factory import (
    get_llm_backend,
    list_available_backends,
    list_available_models,
    print_llm_status,
    register_backend,
)

__all__ = [
    "LLMBackend",
    "LLMResult",
    "LLMConfig",
    "Message",
    "get_llm_backend",
    "list_available_backends",
    "list_available_models",
    "print_llm_status",
    "register_backend",
]

