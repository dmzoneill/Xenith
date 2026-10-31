"""Factory for creating TTS backends"""

import importlib

from .base import TTSBackend

# Registry of available backends
_BACKENDS: dict[str, type[TTSBackend]] = {}


def register_backend(name: str, backend_class: type[TTSBackend]) -> None:
    """Register a TTS backend"""
    _BACKENDS[name.lower()] = backend_class


def _load_backends() -> None:
    """Dynamically load and register available backends"""

    if _BACKENDS:
        return  # Already loaded

    # Try to import Piper backend (fastest, keeps model in memory)
    try:
        # Optional backend: imported lazily so a missing dependency only disables it.
        piper_cls = importlib.import_module(".piper_backend", __package__).PiperTTSBackend

        if piper_cls.is_available():
            register_backend("piper", piper_cls)
            print("[TTS] Piper backend available (fast, ~50ms/sentence)")
        else:
            print("[TTS] Piper backend not available")
    except ImportError as e:
        print(f"[TTS] Piper backend import failed: {e}")

    # Try to import MeloTTS backend
    try:
        melo_cls = importlib.import_module(".melotts_backend", __package__).MeloTTSBackend

        if melo_cls.is_available():
            register_backend("melotts", melo_cls)
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
) -> TTSBackend | None:
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
        # Priority: Piper (fastest), then MeloTTS (highest quality), then first available
        preferred = [name for name in ("piper", "melotts") if name in _BACKENDS]
        backend = preferred[0] if preferred else next(iter(_BACKENDS))

    backend_class = _BACKENDS.get(backend.lower())
    if backend_class is None:
        print(f"[TTS] Unknown backend: {backend} (available: {', '.join(_BACKENDS)})")
        return None

    try:
        return backend_class(voice=voice, device=device, **kwargs)
    except (KeyError, OSError, RuntimeError, ValueError, TypeError) as e:
        print(f"[TTS] Failed to create {backend} backend: {e}")
        return None


def list_available_backends() -> list[str]:
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
        info = backend_class.get_device_info()
        lines = [
            f"\n{name}:",
            f"  Available: {backend_class.is_available()}",
            f"  Devices: {info.get('devices', [])}",
        ]
        lines.extend(f"    {dev}: {note}" for dev, note in info.get("notes", {}).items())
        print("\n".join(lines))
