"""Factory for creating LLM backends"""

from typing import Optional, List, Type, Dict

from .base import LLMBackend, LLMConfig


# Registry of available backends
_BACKENDS: Dict[str, Type[LLMBackend]] = {}


def register_backend(name: str, backend_class: Type[LLMBackend]) -> None:
    """Register an LLM backend"""
    _BACKENDS[name.lower()] = backend_class


def _load_backends() -> None:
    """Dynamically load and register available backends"""
    global _BACKENDS

    if _BACKENDS:
        return  # Already loaded

    # Try to import OpenVINO backend
    try:
        from .openvino_backend import OpenVINOLLMBackend

        if OpenVINOLLMBackend.is_available():
            register_backend("openvino", OpenVINOLLMBackend)
            print("[LLM] OpenVINO backend available")
        else:
            print("[LLM] OpenVINO backend not available (openvino_genai not installed)")
    except ImportError as e:
        print(f"[LLM] OpenVINO backend import failed: {e}")

    # Future: Add more backends (OpenAI, Anthropic, etc.)


def get_llm_backend(
    backend: str = "auto",
    model: str = "qwen2.5-1.5b",
    device: str = "auto",
    **kwargs,
) -> Optional[LLMBackend]:
    """Get an LLM backend instance

    Args:
        backend: Backend name ("auto", "openvino") or specific backend
        model: Model identifier (e.g., "qwen2.5-1.5b", "phi-3-mini")
        device: Device preference ("auto", "cpu", "gpu", "npu")
        **kwargs: Backend-specific options

    Returns:
        LLMBackend instance or None if no backend available
    """
    _load_backends()

    if not _BACKENDS:
        print("[LLM] No LLM backends available")
        return None

    # Auto-select backend
    if backend == "auto":
        # Priority: OpenVINO (for NPU support)
        for name in ["openvino"]:
            if name in _BACKENDS:
                backend = name
                break
        else:
            # Use first available
            backend = list(_BACKENDS.keys())[0]

    backend = backend.lower()
    if backend not in _BACKENDS:
        print(f"[LLM] Unknown backend: {backend}")
        print(f"[LLM] Available backends: {list(_BACKENDS.keys())}")
        return None

    backend_class = _BACKENDS[backend]

    # Create instance
    try:
        instance = backend_class(model=model, device=device, **kwargs)
        return instance
    except Exception as e:
        print(f"[LLM] Failed to create {backend} backend: {e}")
        return None


def list_available_backends() -> List[str]:
    """List available LLM backends"""
    _load_backends()
    return list(_BACKENDS.keys())


def list_available_models(backend: str = "openvino") -> Dict[str, dict]:
    """List available models for a backend"""
    _load_backends()

    if backend not in _BACKENDS:
        return {}

    return _BACKENDS[backend].MODELS


def print_llm_status() -> None:
    """Print LLM backend status"""
    _load_backends()

    print("\n=== LLM Backend Status ===")
    if not _BACKENDS:
        print("No LLM backends available!")
        return

    for name, backend_class in _BACKENDS.items():
        print(f"\n{name}:")
        print(f"  Available: {backend_class.is_available()}")
        device_info = backend_class.get_device_info()
        print(f"  Devices: {device_info.get('devices', [])}")
        print(f"  Default device: {device_info.get('default', 'cpu')}")
        print(f"  Models:")
        for model_id, info in backend_class.MODELS.items():
            npu = "✓" if info.get("npu_compatible") else "✗"
            print(
                f"    - {model_id} ({info['size']}) NPU:{npu} - {info['description']}"
            )

