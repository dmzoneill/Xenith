"""Factory for creating TTS backends"""

from typing import Optional, List, Type, Dict

from .base import TTSBackend


# Registry of available backends
_BACKENDS: Dict[str, Type[TTSBackend]] = {}


def register_backend(name: str, backend_class: Type[TTSBackend]) -> None:
    """Register a TTS backend"""
    _BACKENDS[name.lower()] = backend_class


def _load_backends() -> None:
    """Dynamically load and register available backends"""
    global _BACKENDS

    if _BACKENDS:
        return  # Already loaded

    # Try to import Piper backend (fastest, keeps model in memory)
    try:
        from .piper_backend import PiperTTSBackend

        if PiperTTSBackend.is_available():
            register_backend("piper", PiperTTSBackend)
            print("[TTS] Piper backend available (fast, ~50ms/sentence)")
        else:
            print("[TTS] Piper backend not available")
    except ImportError as e:
        print(f"[TTS] Piper backend import failed: {e}")

    # Try to import MeloTTS backend
    try:
        from .melotts_backend import MeloTTSBackend

        if MeloTTSBackend.is_available():
            register_backend("melotts", MeloTTSBackend)
            print("[TTS] MeloTTS backend available (high quality)")
        else:
            print("[TTS] MeloTTS backend not available (binary not found)")
    except ImportError as e:
        print(f"[TTS] MeloTTS backend import failed: {e}")


def get_tts_backend(
    backend: str = "auto",
    device: str = "auto",
    voice: str = "EN-Default",
    **kwargs,
) -> Optional[TTSBackend]:
    """Get a TTS backend instance

    Args:
        backend: Backend name ("auto", "melotts") or specific backend
        device: Device preference ("auto", "cpu", "gpu", "npu")
        voice: Voice/speaker to use
        **kwargs: Backend-specific options

    Returns:
        TTSBackend instance or None if no backend available
    """
    _load_backends()

    if not _BACKENDS:
        print("[TTS] No TTS backends available")
        return None

    # Auto-select backend
    if backend == "auto":
        # Priority: Piper (fastest), then MeloTTS (highest quality)
        for name in ["piper", "melotts"]:
            if name in _BACKENDS:
                backend = name
                break
        else:
            # Use first available
            backend = list(_BACKENDS.keys())[0]

    backend = backend.lower()
    if backend not in _BACKENDS:
        print(f"[TTS] Unknown backend: {backend}")
        print(f"[TTS] Available backends: {list(_BACKENDS.keys())}")
        return None

    backend_class = _BACKENDS[backend]

    # Create instance
    try:
        instance = backend_class(voice=voice, device=device, **kwargs)
        return instance
    except Exception as e:
        print(f"[TTS] Failed to create {backend} backend: {e}")
        return None


def list_available_backends() -> List[str]:
    """List available TTS backends"""
    _load_backends()
    return list(_BACKENDS.keys())


def print_tts_status() -> None:
    """Print TTS backend status"""
    _load_backends()

    print("\n=== TTS Backend Status ===")
    if not _BACKENDS:
        print("No TTS backends available!")
        return

    for name, backend_class in _BACKENDS.items():
        print(f"\n{name}:")
        print(f"  Available: {backend_class.is_available()}")
        device_info = backend_class.get_device_info()
        print(f"  Devices: {device_info.get('devices', [])}")
        if "notes" in device_info:
            for dev, note in device_info["notes"].items():
                print(f"    {dev}: {note}")
