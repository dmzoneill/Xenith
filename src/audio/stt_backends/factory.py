"""Factory for creating STT backends"""

from typing import Dict, List, Optional, Type

from .base import STTBackend


# Registry of available backends
_BACKENDS: Dict[str, Type[STTBackend]] = {}


def _register_backends():
    """Register all available backends"""
    global _BACKENDS

    # Try to import and register each backend

    # OpenAI Whisper (PyTorch/CUDA)
    try:
        from .whisper_backend import WhisperBackend

        if WhisperBackend.is_available():
            _BACKENDS["whisper"] = WhisperBackend
            _BACKENDS["openai-whisper"] = WhisperBackend
            _BACKENDS["cuda"] = WhisperBackend  # Alias for CUDA users
    except ImportError:
        pass

    # OpenVINO (NPU/GPU/CPU)
    try:
        from .openvino_backend import OpenVINOBackend, OpenVINOOptimumBackend

        if OpenVINOBackend.is_available():
            _BACKENDS["openvino"] = OpenVINOBackend
            _BACKENDS["openvino-genai"] = OpenVINOBackend
            _BACKENDS["npu"] = OpenVINOBackend  # Alias for NPU users
            _BACKENDS["intel"] = OpenVINOBackend  # Alias

        if OpenVINOOptimumBackend.is_available():
            _BACKENDS["openvino-optimum"] = OpenVINOOptimumBackend
            _BACKENDS["optimum"] = OpenVINOOptimumBackend
    except ImportError:
        pass


# Initialize backends on module load
_register_backends()


def list_available_backends() -> List[str]:
    """List all available STT backends

    Returns:
        List of backend names that can be used with get_stt_backend()
    """
    # Re-check in case new packages were installed
    _register_backends()

    # Return unique backend names (not aliases)
    unique = set()
    for name, backend_cls in _BACKENDS.items():
        if name in ("whisper", "openvino", "openvino-optimum"):
            unique.add(name)

    return sorted(unique)


def get_backend_info() -> Dict[str, dict]:
    """Get detailed information about all available backends

    Returns:
        Dict mapping backend names to their device info
    """
    _register_backends()

    info = {}
    seen_classes = set()

    for name, backend_cls in _BACKENDS.items():
        if backend_cls in seen_classes:
            continue
        seen_classes.add(backend_cls)

        info[name] = {
            "class": backend_cls.__name__,
            "available": backend_cls.is_available(),
            "devices": backend_cls.get_device_info(),
        }

    return info


def get_stt_backend(
    backend: str = "auto",
    model_name: str = "base",
    device: str = "auto",
) -> Optional[STTBackend]:
    """Get an STT backend instance

    Args:
        backend: Backend name ("whisper", "openvino", "auto")
        model_name: Model size ("tiny", "base", "small", "medium", "large")
        device: Device for inference ("auto", "cpu", "cuda", "npu", "gpu")

    Returns:
        Configured STTBackend instance, or None if no backend available

    Examples:
        # Use OpenVINO on Intel NPU (ultra-low power)
        backend = get_stt_backend("openvino", device="NPU")

        # Use Whisper on NVIDIA GPU (high performance)
        backend = get_stt_backend("whisper", device="cuda")

        # Auto-select best available
        backend = get_stt_backend("auto")
    """
    _register_backends()

    backend_lower = backend.lower()

    if backend_lower == "auto":
        # Priority: NPU (lowest power) > Intel GPU > NVIDIA GPU > CPU
        # This prioritizes power efficiency for always-on voice detection

        # 1. Check for Intel NPU first (ultra-low power ~1-3W)
        if "openvino" in _BACKENDS:
            ov_info = _BACKENDS["openvino"].get_device_info()
            if ov_info.get("npu_available"):
                print("[STT] Auto-selected: OpenVINO on NPU (ultra-low power ~1-3W)")
                return _BACKENDS["openvino"](model_name, "NPU")

        # 2. Check for Intel Arc/Iris GPU (power efficient ~5-15W)
        if "openvino" in _BACKENDS:
            ov_info = _BACKENDS["openvino"].get_device_info()
            devices = ov_info.get("devices", [])
            # Look for Intel GPU (GPU.0 is typically the Intel iGPU)
            if "GPU.0" in devices:
                print("[STT] Auto-selected: OpenVINO on Intel GPU (GPU.0, ~5-15W)")
                return _BACKENDS["openvino"](model_name, "GPU.0")
            elif ov_info.get("gpu_available"):
                print("[STT] Auto-selected: OpenVINO on GPU")
                return _BACKENDS["openvino"](model_name, "GPU")

        # 3. Check for NVIDIA GPU via Whisper/CUDA (high power ~50-100W)
        if "whisper" in _BACKENDS:
            whisper_info = _BACKENDS["whisper"].get_device_info()
            if whisper_info.get("cuda_available"):
                print("[STT] Auto-selected: Whisper on NVIDIA CUDA (high performance)")
                return _BACKENDS["whisper"](model_name, "cuda")

        # 4. Fall back to CPU
        if "openvino" in _BACKENDS:
            print("[STT] Auto-selected: OpenVINO on CPU")
            return _BACKENDS["openvino"](model_name, "CPU")

        if "whisper" in _BACKENDS:
            print("[STT] Auto-selected: Whisper on CPU")
            return _BACKENDS["whisper"](model_name, "cpu")

        print("[STT] No STT backend available!")
        print("[STT] Install one of:")
        print("[STT]   pip install openai-whisper torch  # For Whisper/CUDA")
        print("[STT]   pip install openvino openvino-genai  # For OpenVINO/NPU")
        return None

    # Specific backend requested
    if backend_lower in _BACKENDS:
        backend_cls = _BACKENDS[backend_lower]

        # Handle device auto-selection for specific backends
        if device.lower() == "auto":
            device_info = backend_cls.get_device_info()
            device = device_info.get("default", "cpu")

        return backend_cls(model_name, device)

    # Backend not found
    available = list_available_backends()
    print(f"[STT] Backend '{backend}' not found.")
    if available:
        print(f"[STT] Available backends: {', '.join(available)}")
    else:
        print("[STT] No backends available. Install one of:")
        print("[STT]   pip install openai-whisper torch")
        print("[STT]   pip install openvino openvino-genai")

    return None


def print_backend_status():
    """Print status of all STT backends for debugging"""
    print("\n" + "=" * 60)
    print("STT Backend Status")
    print("=" * 60)

    info = get_backend_info()

    if not info:
        print("\nNo STT backends available!")
        print("\nInstall options:")
        print("  Whisper (CUDA/CPU): pip install openai-whisper torch")
        print("  OpenVINO (NPU/GPU): pip install openvino openvino-genai")
        return

    for name, backend_info in info.items():
        print(f"\n{name} ({backend_info['class']})")
        print("-" * 40)

        devices = backend_info.get("devices", {})

        if devices.get("npu_available"):
            print(f"  ✓ NPU: {devices.get('npu_name', 'Available')}")
        if devices.get("cuda_available"):
            print(f"  ✓ CUDA: {devices.get('cuda_device_name', 'Available')}")
        if devices.get("gpu_available"):
            print(f"  ✓ GPU: {devices.get('gpu_name', 'Available')}")

        available_devices = devices.get("devices", [])
        if available_devices:
            print(f"  All devices: {', '.join(available_devices)}")

        print(f"  Default device: {devices.get('default', 'cpu')}")

    print("\n" + "=" * 60)
