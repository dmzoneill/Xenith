# Speech-to-Text (STT) Backends

This module provides a unified interface for different speech-to-text engines in Xenith.

## Available Backends

### 1. OpenVINO Backend (`openvino`)

Uses Intel OpenVINO for optimized inference on Intel hardware.

**Supported Devices:**
- `NPU` - Intel Neural Processing Unit (ultra-low power, ~1-3W)
- `GPU.0` - Intel integrated GPU
- `GPU.1` - Secondary GPU (NVIDIA if present)
- `CPU` - Intel CPU with AVX optimizations

**Installation:**
```bash
pip install openvino openvino-genai
```

**For NPU support**, see [Intel NPU Setup Guide](INTEL_NPU_SETUP.md).

### 2. Whisper Backend (`whisper`)

Uses OpenAI's original Whisper implementation with PyTorch.

**Supported Devices:**
- `cuda` - NVIDIA GPU with CUDA
- `cpu` - CPU fallback

**Installation:**
```bash
pip install openai-whisper torch
```

## Usage

### Basic Usage

```python
from audio.stt_backends import get_stt_backend

# Auto-select best backend
backend = get_stt_backend("auto")

# Or specify backend and device
backend = get_stt_backend("openvino", model_name="base", device="NPU")

# Load model
backend.load_model()

# Transcribe audio (numpy array, float32, 16kHz mono)
result = backend.transcribe(audio_array, language="en")
print(result.text)

# Cleanup
backend.unload_model()
```

### Check Available Backends

```python
from audio.stt_backends import list_available_backends, print_backend_status

# List backend names
print(list_available_backends())  # ['openvino', 'whisper']

# Print detailed status
print_backend_status()
```

## API Reference

### `STTResult`

Dataclass returned by `transcribe()`:

```python
@dataclass
class STTResult:
    text: str                          # Transcribed text
    language: Optional[str] = None     # Detected language
    confidence: Optional[float] = None # Confidence score
    segments: Optional[List[dict]] = None  # Word-level timestamps
    duration: Optional[float] = None   # Audio duration in seconds
```

### `STTBackend` (Abstract Base Class)

All backends implement this interface:

```python
class STTBackend(ABC):
    def __init__(self, model_name: str, device: str): ...
    def load_model(self) -> bool: ...
    def transcribe(self, audio: np.ndarray, language: str = "en") -> STTResult: ...
    def unload_model(self) -> None: ...
    
    @classmethod
    def is_available(cls) -> bool: ...
    
    @classmethod
    def get_device_info(cls) -> dict: ...
```

## Configuration

In `config/config.yaml`:

```yaml
audio:
  stt:
    backend: "auto"     # or "openvino", "whisper"
    device: "auto"      # or "NPU", "GPU.0", "cuda", "cpu"
    model: "base"       # tiny, base, small, medium, large
```

### Auto-Selection Priority

When both `backend` and `device` are set to `"auto"`:

1. **Intel NPU** (OpenVINO) - ~1-3W, ultra-low power
2. **Intel GPU** (OpenVINO, GPU.0) - ~5-15W, efficient
3. **NVIDIA GPU** (Whisper/CUDA) - ~50-100W, high performance
4. **CPU** (OpenVINO) - fallback

## Adding New Backends

1. Create a new file in `stt_backends/` (e.g., `my_backend.py`)
2. Inherit from `STTBackend` and implement all abstract methods
3. Register in `factory.py`:

```python
# In _register_backends():
try:
    from .my_backend import MyBackend
    if MyBackend.is_available():
        _BACKENDS["my-backend"] = MyBackend
except ImportError:
    pass
```

## Testing

```bash
# Test all backends
python test_stt_backends.py

# Test specific backend
python test_stt_backends.py --backend openvino --device NPU

# Test with audio file
python test_stt_backends.py --test-audio sample.wav
```



