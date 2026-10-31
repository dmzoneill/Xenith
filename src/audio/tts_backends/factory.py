"""Factory for creating TTS backends"""

# Same registry shape as llm_backends.factory, for a different backend base class.
# pylint: disable=duplicate-code

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
        # pylint: disable-next=import-outside-toplevel
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
        # Optional backend: imported lazily so a missing dependency only disables it.
        # pylint: disable-next=import-outside-toplevel
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
        # Priority: Piper (fastest), then MeloTTS (highest quality)
        for name in ["piper", "melotts"]:
            if name in _BACKENDS:
                backend = name
                break
        else:
            # Use first available
            backend = next(iter(_BACKENDS))

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
    # Backend constructors may raise anything (missing models, device errors); report and fall back.
    # pylint: disable-next=broad-exception-caught
    except Exception as e:  # noqa: BLE001
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
        print(f"\n{name}:")
        print(f"  Available: {backend_class.is_available()}")
        device_info = backend_class.get_device_info()
        print(f"  Devices: {device_info.get('devices', [])}")
        if "notes" in device_info:
            for dev, note in device_info["notes"].items():
                print(f"    {dev}: {note}")
