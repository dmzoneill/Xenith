# TTS Backends Reference

This document describes the Text-to-Speech (TTS) backend system in Xenith.

## Overview

Xenith uses a pluggable TTS backend system that supports multiple speech synthesis engines:

| Backend | Speed | Quality | Best For |
|---------|-------|---------|----------|
| **Piper** ⭐ | ~100ms | High | Fast streaming response |
| MeloTTS | ~300ms | Highest | Best quality |
| espeak-ng | ~20ms | Low | Ultra-low latency fallback |

## Piper TTS (Recommended)

**Piper** is the default TTS backend, providing fast neural synthesis with high-quality voices.

### Features

- **Fast synthesis**: ~100ms per sentence
- **In-memory output**: No file I/O (direct numpy array)
- **Persistent loading**: Model stays in memory for instant synthesis
- **High-quality voices**: Neural network voices from VITS

### Available Voices

| Voice ID | Piper Voice | Quality | Speed |
|----------|-------------|---------|-------|
| `EN-Default` | en_US-ryan-high | High | ~100ms |
| `EN-Fast` | en_US-lessac-medium | Medium | ~80ms |
| `EN-US` | en_US-ryan-high | High | ~100ms |
| `EN-BR` | en_US-lessac-medium | Medium | ~80ms |
| `EN-AU` | en_GB-alan-medium | Medium | ~80ms |

### Usage

```python
from src.audio.tts_backends import get_tts_backend

# Get Piper backend (auto-selected as default)
tts = get_tts_backend(voice="EN-Default")

if tts.load():
    # Synthesize to memory (fast, no file I/O)
    result = tts.synthesize("Hello, world!")
    
    # audio_data is numpy float32 array
    print(f"Audio shape: {result.audio_data.shape}")
    print(f"Sample rate: {result.sample_rate}")
    
    # Optionally save to file
    result = tts.synthesize("Hello!", output_path="/tmp/hello.wav")
```

### In-Memory Audio (Key Feature)

Piper returns audio directly as a numpy array, enabling **zero-latency** handoff to the audio player:

```python
# In-memory (fast)
result = tts.synthesize("Hello")  # No output_path
audio_array = result.audio_data   # numpy float32
sample_rate = result.sample_rate  # 22050 Hz

# Direct playback
import sounddevice as sd
sd.play(audio_array, sample_rate)
```

## MeloTTS (High Quality)

A high-quality neural TTS with Intel NPU acceleration for BERT preprocessing.

### Features

- **Highest quality**: Best voice naturalness
- **NPU acceleration**: BERT runs on Intel NPU for low power
- **Multiple voices**: US, British, Indian, Australian
- **INT8 quantization**: Fast inference with DeepFilterNet denoising

### Device Support

| Component | CPU | GPU | NPU |
|-----------|-----|-----|-----|
| BERT (preprocessing) | ✅ | ✅ | ✅ |
| TTS (synthesis) | ✅ | ✅ | ❌ |
| DeepFilterNet | ✅ | ✅ | ⚠️ |

### When to Use

- When voice quality is more important than speed
- For generating audio files (not real-time)
- When power isn't a concern

## Configuration

TTS is configured in `config/config.yaml`:

```yaml
audio:
  tts:
    # Backend: "auto", "piper", "melotts"
    #   - auto: Piper (fast) preferred, falls back to MeloTTS
    backend: "auto"
    
    # Device for TTS model: "auto", "cpu", "gpu"
    device: "auto"
    
    # Voice: "EN-Default", "EN-Fast", "EN-US", "EN-BR", "EN-AU"
    voice: "EN-Default"
    
    # Language: "EN" (English)
    language: "EN"
    
    # Speed multiplier (0.5 = half speed, 2.0 = double speed)
    speed: 1.0
    
    # MeloTTS-specific options
    use_npu_bert: true    # Use NPU for BERT preprocessing
    use_quantized: true   # Use INT8 quantized model
    
    enabled: true
```

## Python API

### Basic Usage

```python
from src.audio.tts_backends import get_tts_backend, list_available_backends

# List available backends
backends = list_available_backends()
print(f"Available: {backends}")

# Get default backend (Piper)
tts = get_tts_backend()

if tts and tts.load():
    # Synthesize to memory
    result = tts.synthesize("Hello, world!")
    
    # result.audio_data is numpy float32 array
    # result.sample_rate is typically 22050 Hz
    # result.duration is audio length in seconds
```

### Specifying Backend

```python
# Use Piper explicitly
tts = get_tts_backend(backend="piper", voice="EN-Default")

# Use MeloTTS for highest quality
tts = get_tts_backend(backend="melotts", voice="EN-US")
```

### Streaming Pipeline Integration

The streaming pipeline uses Piper for fast sentence-by-sentence synthesis:

```python
from src.audio.streaming_pipeline import StreamingPipeline, StreamingConfig

config = StreamingConfig(
    tts_voice="EN-Default",  # Uses Piper
)

pipeline = StreamingPipeline(config)
pipeline.load()

# TTS happens in parallel with LLM generation
result = pipeline.process_streaming("Hello, how are you?")
```

## TTSBackend Base Class

All TTS backends implement this interface:

```python
class TTSBackend(ABC):
    def __init__(self, voice: str, device: str, language: str): ...
    
    @property
    def name(self) -> str: ...
    
    @property
    def is_loaded(self) -> bool: ...
    
    @abstractmethod
    def load(self) -> bool: ...
    
    @abstractmethod
    def synthesize(
        self,
        text: str,
        output_path: Optional[Path] = None,
        speed: float = 1.0,
        **kwargs
    ) -> TTSResult: ...
    
    @abstractmethod
    def get_available_voices(self) -> List[str]: ...
    
    def unload(self) -> None: ...
    
    @classmethod
    def is_available(cls) -> bool: ...
```

## TTSResult Dataclass

```python
@dataclass
class TTSResult:
    audio_path: Optional[Path] = None      # Path if saved to file
    audio_data: Optional[np.ndarray] = None  # Numpy audio (float32)
    sample_rate: int = 22050               # Sample rate in Hz
    duration: Optional[float] = None       # Audio duration in seconds
    voice: Optional[str] = None            # Voice used
    error: Optional[str] = None            # Error message if failed
```

## Performance Comparison

| Backend | Load Time | Synthesis (short) | Synthesis (long) |
|---------|-----------|-------------------|------------------|
| Piper | ~200ms | ~80-100ms | ~200-300ms |
| MeloTTS | ~500ms | ~250-350ms | ~500-800ms |
| espeak-ng | ~10ms | ~20-30ms | ~50-100ms |

## Installation

### Piper

Piper models are downloaded automatically on first use:

```bash
# Test Piper
python test_tts_backends.py
```

Models are cached in `~/.cache/piper/`.

### MeloTTS

MeloTTS requires building from source:

```bash
cd vendor/MeloTTS.cpp
cmake -DUSE_BERT_NPU=ON -S . -B build
cmake --build build --config Release
```

See [MeloTTS Setup](#melotts-installation) for details.

## Troubleshooting

### Piper Voice Not Found

If voice download fails, manually download from HuggingFace:

```bash
mkdir -p ~/.cache/piper
cd ~/.cache/piper

# Download Ryan high-quality voice
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/high/en_US-ryan-high.onnx.json
```

### Audio Quality Issues

1. Use high-quality voice: `voice: "EN-Default"` (ryan-high)
2. For MeloTTS, ensure DeepFilterNet is enabled
3. Check sample rate matches playback device

### Slow Synthesis

1. Ensure model is pre-loaded (call `tts.load()` at startup)
2. Use Piper instead of MeloTTS for streaming
3. Check CPU isn't throttled

## See Also

- [Voice Pipeline Architecture](VOICE_PIPELINE.md) - Complete pipeline overview
- [STT Backends Reference](STT_BACKENDS.md) - Speech-to-text documentation
- [LLM Backends Reference](LLM_BACKENDS.md) - Language model documentation
- [Scripts Reference](SCRIPTS.md) - Test scripts and commands
